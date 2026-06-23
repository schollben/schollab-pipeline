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
import time
import numpy as np
from datetime import datetime
import sys

try:
	cv2.setNumThreads(0)
except:
	pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline
import tifffile

# Import from renamed modules in the same caiman/ directory
from registration_gui import get_registration_options
from pipeline_job import apply_skip_caiman
from pipeline_run_log import CAIMAN_STEP_LABELS
from tif_to_h5 import tif_stacks_to_h5
from tiff_compat import tiff_writer_append

global mc


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


def _h5_artifact_line(folder, filename):
	"""Format H5 artifact with size for run log."""
	path = os.path.join(folder, filename)
	if os.path.isfile(path):
		size_gb = os.path.getsize(path) / 1e9
		return f'{filename} ({size_gb:.2f} GB)'
	return None


def _sample_tif_artifacts(folder):
	"""List rigid/nonrigid sample TIFFs produced by motion correction."""
	arts = []
	for pattern in ('*_rigid.tif', '*_nonrigid.tif'):
		for path in sorted(glob.glob(os.path.join(folder, pattern))):
			arts.append(os.path.basename(path))
	return arts


def _selected_caiman_labels(row):
	"""GUI column labels that were checked for this folder."""
	return [CAIMAN_STEP_LABELS[i] for i, on in enumerate(row) if on]


def _finalize_caiman_summary(folder, steps, row):
	"""Derive succeeded / failed / incomplete from step outcomes and artifacts."""
	selected = _selected_caiman_labels(row)
	failed = next((s for s in steps if s.get('status') == 'failed'), None)
	if failed:
		return {
			'steps': steps,
			'selected': selected,
			'result': 'failed',
			'error': failed.get('error', 'step failed'),
		}
	reg = os.path.join(folder, 'registered.h5')
	unreg = os.path.join(folder, 'unregistered.h5')
	if os.path.isfile(reg):
		return {'steps': steps, 'selected': selected, 'result': 'succeeded', 'error': None}
	if row[0] and not any(row[1:4]) and os.path.isfile(unreg):
		return {
			'steps': steps,
			'selected': selected,
			'result': 'incomplete',
			'error': 'no motion step — registered.h5 not created',
		}
	if not any(row):
		return {'steps': steps, 'selected': selected, 'result': 'skipped', 'error': None}
	return {
		'steps': steps,
		'selected': selected,
		'result': 'incomplete',
		'error': 'registered.h5 missing after CaImAn',
	}


def _register_one_folder(folder, row, mc_dict):
	"""
	Run CaImAn steps for one folder; return summary dict for pipeline run log.

	Does not raise on step failure — caller checks summary['result'].
	"""
	steps = []
	n_procs = 0
	print(folder)

	if row[0]:
		t0 = time.perf_counter()
		source_tifs = []
		for f in glob.glob(os.path.join(folder, "*.tif")):
			if 'References' in f:
				continue
			b = os.path.basename(f)
			if b.endswith('_rigid.tif') or b.endswith('_nonrigid.tif'):
				continue
			source_tifs.append(f)
		if not source_tifs:
			print(f"WARNING: No source TIFs found in {folder}")
			print(f"  Skipping TIFs→H5 step — registered.h5 left untouched.")
			steps.append({
				'name': CAIMAN_STEP_LABELS[0],
				'status': 'skipped',
				'duration_s': round(time.perf_counter() - t0, 1),
				'detail': 'no source TIFs',
			})
		else:
			try:
				for stale_h5 in ['unregistered.h5', 'registered.h5']:
					stale_path = os.path.join(folder, stale_h5)
					if os.path.exists(stale_path):
						os.remove(stale_path)
						print(f"Deleted stale file: {stale_path}")
				h5_name = os.path.join(folder, 'unregistered.h5')
				tif_stacks_to_h5(folder, h5_name, frame_offset=False)
				art = _h5_artifact_line(folder, 'unregistered.h5')
				steps.append({
					'name': CAIMAN_STEP_LABELS[0],
					'status': 'ok',
					'duration_s': round(time.perf_counter() - t0, 1),
					'artifacts_line': art or 'unregistered.h5',
				})
			except Exception as exc:
				steps.append({
					'name': CAIMAN_STEP_LABELS[0],
					'status': 'failed',
					'duration_s': round(time.perf_counter() - t0, 1),
					'error': str(exc),
				})
				return _finalize_caiman_summary(folder, steps, row)

	for step_idx, sample_suffix in ((1, 'rigid'), (2, 'rigid'), (3, 'nonrigid')):
		if not row[step_idx]:
			continue
		label = CAIMAN_STEP_LABELS[step_idx]
		t0 = time.perf_counter()
		try:
			if step_idx == 3:
				mc_dict['pw_rigid'] = True
			else:
				mc_dict['pw_rigid'] = False
			n_procs += 1
			register_one_session(
				folder, mc_dict, keep_memmap=False,
				save_sample=True, sample_name=f"{n_procs:02}_{sample_suffix}.tif",
			)
			arts = [_h5_artifact_line(folder, 'registered.h5')] + _sample_tif_artifacts(folder)
			steps.append({
				'name': label,
				'status': 'ok',
				'duration_s': round(time.perf_counter() - t0, 1),
				'artifacts_line': ', '.join(a for a in arts if a),
			})
		except Exception as exc:
			steps.append({
				'name': label,
				'status': 'failed',
				'duration_s': round(time.perf_counter() - t0, 1),
				'error': str(exc),
				'detail': '(no registered.h5)',
			})
			return _finalize_caiman_summary(folder, steps, row)

	if row[0] and not any(row[1:4]):
		unreg = os.path.join(folder, 'unregistered.h5')
		reg = os.path.join(folder, 'registered.h5')
		if os.path.isfile(unreg) and not os.path.isfile(reg):
			print(
				"WARNING: TIFs→H5 wrote unregistered.h5 but no motion step was selected.\n"
				"  Enable at least one of: First Rigid, Addl. Rigid, or NoRMCorre — "
				"otherwise registered.h5 is never created and FAST will skip this folder."
			)
	return _finalize_caiman_summary(folder, steps, row)


def register_bulk(sessions_to_run, process_selections):
	"""
	Run CaImAn motion correction on prepared .h5 stacks.

	Parameters:
		sessions_to_run (list): Directory paths to find data in.
		process_selections (np.array): 4×N bool array — rows are:
			[TIFs→H5, first rigid, additional rigid, NoRMCorre]

	Returns:
		list[dict]: per-folder summary for pipeline run log (one entry per session).
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

	summaries = []
	for i in range(0, len(sessions_to_run)):
		row = [bool(process_selections[j, i]) for j in range(4)]
		summaries.append(_register_one_folder(sessions_to_run[i], row, mc_dict))
	return summaries


if __name__ == '__main__':
	# Get folder selections from the GUI — returns (paths, 4×N bool array, skip_caiman, run_mode, scheduled_at)
	workdirs, proc_opts, skip_caiman, run_mode, scheduled_at = get_registration_options()
	if workdirs is None:
		print("No folders selected. Exiting.")
		sys.exit(0)

	proc_opts = apply_skip_caiman(proc_opts, skip_caiman)

	from pipeline_launcher import (
		build_immediate_job,
		build_scheduled_job,
		launch_job_now,
		persist_batch_job,
		schedule_job,
		write_job,
		JOB_PATH,
	)

	if run_mode == 'schedule':
		if not scheduled_at:
			print("ERROR: Schedule mode requires a valid scheduled_at.")
			sys.exit(1)
		job = build_scheduled_job(workdirs, proc_opts, skip_caiman, scheduled_at)
		persisted = persist_batch_job(job)
		write_job(job, JOB_PATH)
		write_job(job, persisted)
		print(f"Job written to {JOB_PATH}")
		print(f"Persistent copy: {persisted}")
		print(f"Sessions queued: {len(workdirs)}")
		print(f"skip_caiman: {skip_caiman}")
		print(f"batch_id: {job['batch_id']}")
		print(f"Summary log: {job['run_log_path']}")
		print(f"Verbose log: {job['verbose_log_path']}")
		schedule_job(persisted, scheduled_at)
	else:
		# Immediate run — legacy job JSON shape (no batch_id / batch_log_path required).
		job = build_immediate_job(workdirs, proc_opts, skip_caiman)
		write_job(job, JOB_PATH)
		print(f"Job written to {JOB_PATH}")
		print(f"Sessions queued: {len(workdirs)}")
		print(f"skip_caiman: {skip_caiman}")
		print(f"Run unit: {job['unit_name']}")
		launch_job_now(JOB_PATH)
