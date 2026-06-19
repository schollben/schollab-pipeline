import os
import os.path
import json

CAIMAN_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
CAIMAN_DEFAULTS = {
	'n_processes': 4,
	'threads': {
		'OMP_NUM_THREADS': 1,
		'MKL_NUM_THREADS': 1,
		'OPENBLAS_NUM_THREADS': 1,
		'NUMEXPR_NUM_THREADS': 1,
	}
}


def _load_caiman_config():
	"""Load CaImAn runtime settings before importing NumPy/CaImAn thread libraries."""
	cfg = dict(CAIMAN_DEFAULTS)
	cfg['threads'] = dict(CAIMAN_DEFAULTS['threads'])
	if os.path.exists(CAIMAN_CONFIG_PATH):
		with open(CAIMAN_CONFIG_PATH, encoding='utf-8') as f:
			file_cfg = json.load(f)
		cfg.update({k: v for k, v in file_cfg.items() if k != 'threads'})
		cfg['threads'].update(file_cfg.get('threads', {}))
	return cfg


CAIMAN_CONFIG = _load_caiman_config()

for _thread_env, _thread_default in CAIMAN_CONFIG.get('threads', {}).items():
	# Respect admin/user overrides while defaulting to one thread per CaImAn worker.
	os.environ.setdefault(_thread_env, str(_thread_default))

import cv2
import math
import h5py
import glob
import pathlib
import subprocess
import numpy as np
from datetime import datetime
import sys

try:
	cv2.setNumThreads(0)
except:
	pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
import tifffile

# Import from renamed modules in the same caiman/ directory
from registration_gui import get_registration_options
from pipeline_job import apply_skip_caiman
from tif_to_h5 import tif_stacks_to_h5
from tiff_compat import tiff_writer_append

def _caiman_n_processes():
	raw = os.environ.get('CAIMAN_N_PROCESSES', CAIMAN_CONFIG.get('n_processes', 4))
	try:
		n_processes = int(raw)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"CAIMAN_N_PROCESSES must be an integer, got {raw!r}") from exc
	if n_processes < 1:
		raise ValueError(f"CAIMAN_N_PROCESSES must be >= 1, got {n_processes}")
	return n_processes


def _caiman_thread_setenv_args():
	"""Pass thread caps into systemd so the worker sees them before importing NumPy."""
	args = []
	for key in sorted(CAIMAN_CONFIG.get('threads', {})):
		if key in os.environ:
			args.append(f"--setenv={key}={os.environ[key]}")
	return args


def _fast_path_setenv_args():
	"""Keep FAST path overrides visible after systemd detaches the worker."""
	args = []
	for key in ('FAST_DIR', 'FAST_SCRATCH_DIR'):
		if key in os.environ:
			args.append(f"--setenv={key}={os.environ[key]}")
	return args

def _schollab_conda_root():
	"""
	Directory that contains bin/conda and envs/{caiman,FAST}.
	Set SCHOLLAB_CONDA_ROOT when conda lives under ~/miniconda3 (or elsewhere).
	"""
	root = os.environ.get('SCHOLLAB_CONDA_ROOT')
	if root:
		return os.path.expanduser(root)
	return os.path.join(os.path.expanduser('~'), 'miniforge3')


# Paths on the server — workers/pipeline_worker.py runs under the caiman env and
# calls the FAST step via subprocess using FAST_PYTHON
REPO_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_SCRIPT = os.path.join(REPO_DIR, 'workers', 'pipeline_worker.py')
CAIMAN_PYTHON = os.path.join(_schollab_conda_root(), 'envs', 'caiman', 'bin', 'python')
UNIT_NAME     = 'schollab-PreProcess2PImages'
JOB_PATH      = '/tmp/pipeline_job.json'
# Written at launch so PreProcess2PImages.sh --stop/--attach target this run only.
ACTIVE_UNIT_PATH = '/tmp/pipeline_active_unit.txt'


def _run_motion_correction(parent_dir, fnames, opts, mc_dict):
	caiman_processes = _caiman_n_processes()
	print(f"  CaImAn n_processes: {caiman_processes}")
	c, dview, n_processes = cm.cluster.setup_cluster(
		backend='local', n_processes=caiman_processes, single_thread=False)
	try:
		print(f"  CaImAn motion correction starting: {fnames[0]}")
		mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
		mc.motion_correct(save_movie=True)
		print(f"  CaImAn motion correction finished: {parent_dir}")
		return _save_motion_outputs(parent_dir, mc, mc_dict)
	finally:
		# Always tear down workers; leaked clusters can keep CPUs busy after failures.
		try:
			cm.stop_server(dview=dview)
			print("  CaImAn cluster stopped.")
		except Exception as exc:
			print(f"  WARNING: failed to stop CaImAn cluster cleanly: {exc}")


def _save_motion_outputs(parent_dir, mc, mc_dict):
	if mc_dict['pw_rigid']:
		numframes = len(mc.x_shifts_els)
		np.savetxt(os.path.join(parent_dir, 'nonrigid_x_shifts.csv'), mc.x_shifts_els, delimiter=',')
		np.savetxt(os.path.join(parent_dir, 'nonrigid_y_shifts.csv'), mc.y_shifts_els, delimiter=',')
	else:
		numframes = len(mc.shifts_rig)
		np.savetxt(os.path.join(parent_dir, 'rigid_shifts.csv'), mc.shifts_rig, delimiter=',')

	fnames_new = mc.mmap_file
	if not fnames_new:
		raise RuntimeError(
			"Motion correction returned no memmap paths (mc.mmap_file empty). "
			"Check CaImAn version, input shape, and disk space."
		)
	print(f"  CaImAn memmap output: {fnames_new[0]}")
	return numframes, fnames_new


def _write_registered_h5(parent_dir, source_h5, mmap_h5, numframes, save_sample, sample_name):
	# Rename unregistered.h5 -> registered.h5 before writing corrected frames.
	registered_h5 = os.path.join(parent_dir, 'registered.h5')
	os.replace(source_h5, registered_h5)
	datafile = h5py.File(registered_h5, 'w')
	frames_written = 0
	mov = None

	try:
		datafile.create_dataset("mov", (numframes, 512, 512))
		print(f"  Rewriting corrected H5: {registered_h5} ({numframes} frames)")
		# Load the CaImAn mmap once; chunk slices avoid repeatedly reopening the same file.
		mov = cm.load(mmap_h5)
		for i in range(0, math.floor(numframes / 1000)):
			temp_data = np.array(mov[frames_written:frames_written + 1000, :, :])
			actual_frames = temp_data.shape[0]
			datafile["mov"][frames_written:frames_written + actual_frames, :, :] = temp_data
			frames_written += actual_frames

		if numframes > frames_written:
			temp_data = np.array(mov[frames_written:mov.shape[0], :, :])
			datafile["mov"][frames_written:mov.shape[0], :, :] = temp_data
			frames_written = mov.shape[0]
			del temp_data

		print(f"  Rewrote corrected H5 frames: {frames_written}")
		_write_sample_tiff(parent_dir, sample_name, datafile, numframes, save_sample)
	finally:
		if mov is not None:
			del mov
		datafile.close()
		print(f"  Closed H5 file: {registered_h5}")


def _write_sample_tiff(parent_dir, sample_name, datafile, numframes, save_sample):
	if not save_sample:
		print("  Sample TIFF disabled.")
		return

	sample_frames = min(4000, numframes)
	sample_path = os.path.join(parent_dir, sample_name)
	print(f"  Writing sample TIFF: {sample_path} ({sample_frames} frames)")
	with tifffile.TiffWriter(sample_path, bigtiff=False, imagej=False) as tif:
		for i in range(0, sample_frames):
			curfr = datafile["mov"][i,:,:].astype(np.int16)
			tiff_writer_append(tif, curfr, contiguous=False)
	print(f"  Wrote sample TIFF: {sample_path}")


def _cleanup_memmaps(fnames_new, keep_memmap):
	if keep_memmap or not fnames_new:
		return

	print(f"  Removing CaImAn memmap file(s): {len(fnames_new)}")
	for fname in fnames_new:
		if os.path.exists(fname):
			try:
				os.remove(fname)
				print(f"  Removed memmap: {fname}")
			except OSError as exc:
				print(f"  WARNING: failed to remove memmap {fname}: {exc}")


def register_one_session(parent_dir, mc_dict, keep_memmap, save_sample, sample_name):
	# Prefer unregistered.h5 — glob *registered.h5 sorts registered.h5 before unregistered.h5
	# and would run motion correction on the wrong file when both exist.
	unreg = os.path.join(parent_dir, 'unregistered.h5')
	if os.path.isfile(unreg):
		fnames = [unreg]
	else:
		fnames = sorted(glob.glob(os.path.join(parent_dir, "*registered.h5")))
	if not fnames:
		raise FileNotFoundError(
			f"No *registered.h5 input in {parent_dir} (expected unregistered.h5 for a fresh run)."
		)
	mc_dict['fnames'] = fnames
	mc_dict['upsample_factor_grid'] = 8
	opts = params.CNMFParams(params_dict=mc_dict)
	fnames_new = []

	try:
		numframes, fnames_new = _run_motion_correction(parent_dir, fnames, opts, mc_dict)
		_write_registered_h5(parent_dir, fnames[0], fnames_new[0], numframes, save_sample, sample_name)
	finally:
		# Memmaps are large temporary files; clean them even when H5/sample writing fails.
		_cleanup_memmaps(fnames_new, keep_memmap)


def register_bulk(sessions_to_run, process_selections):
	"""
	Run CaImAn motion correction on prepared .h5 stacks.

	Parameters:
		sessions_to_run (list): Directory paths to find data in.
		process_selections (np.array): 4×N bool array — rows are:
			[TIFs→H5, first rigid, additional rigid, NoRMCorre]
	"""
	fr           = 30
	decay_time   = 1
	dxy          = (1.0, 1.0)
	max_shift_um = (32, 32)
	patch_motion_um = (64., 64.)
	max_shifts   = [int(a/b) for a, b in zip(max_shift_um, dxy)]
	strides      = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
	overlaps     = (32, 32)
	max_deviation_rigid = 3

	mc_dict = {
		'fr': fr, 'decay_time': decay_time, 'dxy': dxy,
		'pw_rigid': False, 'max_shifts': max_shifts,
		'strides': strides, 'overlaps': overlaps,
		'max_deviation_rigid': max_deviation_rigid,
		'border_nan': 'copy', 'nonneg_movie': False,
		'use_cuda': False, 'niter_rig': 5
	}

	for i in range(0, len(sessions_to_run)):
		print(sessions_to_run[i])
		n_procs = 0

		if process_selections[0, i]:
			# Verify source TIFs exist BEFORE deleting anything.
			# Deleting registered.h5 first and then failing on missing TIFs
			# would permanently destroy the only copy of the registered data.
			# Exclude CaImAn sample exports — same idea as tif_to_h5._acquisition_tif_paths.
			source_tifs = []
			for f in glob.glob(os.path.join(sessions_to_run[i], "*.tif")):
				if 'References' in f:
					continue
				b = os.path.basename(f)
				if b.endswith('_rigid.tif') or b.endswith('_nonrigid.tif'):
					continue
				source_tifs.append(f)
			if not source_tifs:
				print(f"WARNING: No source TIFs found in {sessions_to_run[i]}")
				print(f"  Skipping TIFs→H5 step — registered.h5 left untouched.")
			else:
				for stale_h5 in ['unregistered.h5', 'registered.h5']:
					stale_path = os.path.join(sessions_to_run[i], stale_h5)
					if os.path.exists(stale_path):
						os.remove(stale_path)
						print(f"Deleted stale file: {stale_path}")
				h5_name = os.path.join(sessions_to_run[i], 'unregistered.h5')
				tif_stacks_to_h5(sessions_to_run[i], h5_name, frame_offset=False)

		if process_selections[1, i]:
			n_procs += 1
			# Each pass must set its mode explicitly; NoRMCorre mutates this flag below.
			mc_dict['pw_rigid'] = False
			register_one_session(sessions_to_run[i], mc_dict, keep_memmap=False,
				save_sample=True, sample_name=f"{n_procs:02}_rigid.tif")

		if process_selections[2, i]:
			# Prevent a prior non-rigid pass/folder from making this rigid pass expensive.
			mc_dict['pw_rigid'] = False
			register_one_session(sessions_to_run[i], mc_dict, keep_memmap=False,
				save_sample=True, sample_name=f"{n_procs:02}_rigid.tif")
			n_procs += 1

		if process_selections[3, i]:
			# NoRMCorre is the only pass that should use piecewise-rigid motion correction.
			mc_dict['pw_rigid'] = True
			register_one_session(sessions_to_run[i], mc_dict, keep_memmap=False,
				save_sample=True, sample_name=f"{n_procs:02}_nonrigid.tif")
			n_procs += 1

		# TIFs→H5 leaves unregistered.h5; only motion steps produce registered.h5.
		# If the user checked TIF conversion but none of the motion columns, FAST will
		# have nothing to read — warn here so journalctl shows the real cause.
		if process_selections[0, i] and not any(process_selections[1:4, i]):
			unreg = os.path.join(sessions_to_run[i], 'unregistered.h5')
			reg   = os.path.join(sessions_to_run[i], 'registered.h5')
			if os.path.isfile(unreg) and not os.path.isfile(reg):
				print(
					"WARNING: TIFs→H5 wrote unregistered.h5 but no motion step was selected.\n"
					"  Enable at least one of: First Rigid, Addl. Rigid, or NoRMCorre — "
					"otherwise registered.h5 is never created and FAST will skip this folder."
				)


if __name__ == '__main__':
	# Get folder selections from the GUI — returns (paths, 4×N bool array, skip_caiman)
	workdirs, proc_opts, skip_caiman = get_registration_options()
	if workdirs is None:
		print("No folders selected. Exiting.")
		sys.exit(0)

	proc_opts = apply_skip_caiman(proc_opts, skip_caiman)

	# Write job file consumed by workers/pipeline_worker.py.
	# Keeping this as a plain JSON file means any alternative launcher
	# (CLI, web UI, etc.) can produce the same file to trigger the pipeline.
	run_id = datetime.now().strftime('%Y%m%d%H%M%S')
	unit_name = f'{UNIT_NAME}-{run_id}'
	job = {
		"sessions": workdirs.tolist(),
		"process_selections": proc_opts.tolist(),
		"skip_caiman": skip_caiman,
		"run_id": run_id,
		"unit_name": unit_name,
	}
	pathlib.Path(JOB_PATH).write_text(json.dumps(job, indent=2))
	pathlib.Path(ACTIVE_UNIT_PATH).write_text(unit_name)
	print(f"Job written to {JOB_PATH}")
	print(f"Sessions queued: {len(workdirs)}")
	print(f"skip_caiman: {skip_caiman}")
	print(f"Run unit: {unit_name}")

	# Clear any stale failed unit from a previous run (legacy fixed name).
	subprocess.run(
		["systemctl", "--user", "reset-failed", UNIT_NAME],
		check=False
	)

	# Launch workers/pipeline_worker.py as a transient systemd user service.
	# Unique unit name per run so journal "Consumed CPU time" is scoped to this job.
	systemd_cmd = [
		"systemd-run", "--user",
		f"--unit={unit_name}",
		"--collect",
		"--description=Schollab caiman+FAST pipeline",
		f"--setenv=HOME={os.path.expanduser('~')}",
		# Worker subprocess needs same prefix as GUI so FAST_PYTHON resolves correctly.
		f"--setenv=SCHOLLAB_CONDA_ROOT={_schollab_conda_root()}",
	]
	systemd_cmd.extend(_caiman_thread_setenv_args())
	systemd_cmd.extend(_fast_path_setenv_args())
	if 'CAIMAN_N_PROCESSES' in os.environ:
		# Keep one-off CLI overrides visible after systemd detaches the worker.
		systemd_cmd.append(f"--setenv=CAIMAN_N_PROCESSES={os.environ['CAIMAN_N_PROCESSES']}")
	systemd_cmd.extend([
		CAIMAN_PYTHON, WORKER_SCRIPT, JOB_PATH
	])
	subprocess.run(systemd_cmd, check=True)

	print(f"Pipeline launched as systemd service '{unit_name}'.")
	print(f"Monitor: journalctl --user -f -u {unit_name}")
