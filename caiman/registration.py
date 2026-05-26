import os
import os.path
import cv2
import math
import h5py
import glob
import json
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
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline
import tifffile

# Import from renamed modules in the same caiman/ directory
from registration_gui import get_registration_options
from tif_to_h5 import tif_stacks_to_h5
from tiff_compat import tiff_writer_append

global mc

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

	c, dview, n_processes = cm.cluster.setup_cluster(
		backend='local', n_processes=None, single_thread=False)
	mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))

	mc.motion_correct(save_movie=True)

	if mc_dict['pw_rigid']:
		numframes = len(mc.x_shifts_els)
		np.savetxt(os.path.join(parent_dir, 'nonrigid_x_shifts.csv'), mc.x_shifts_els, delimiter=',')
		np.savetxt(os.path.join(parent_dir, 'nonrigid_y_shifts.csv'), mc.y_shifts_els, delimiter=',')
	else:
		numframes = len(mc.shifts_rig)
		np.savetxt(os.path.join(parent_dir, 'rigid_shifts.csv'), mc.shifts_rig, delimiter=',')

	fnames_new = mc.mmap_file
	if not fnames_new:
		cm.stop_server(dview=dview)
		raise RuntimeError(
			"Motion correction returned no memmap paths (mc.mmap_file empty). "
			"Check CaImAn version, input shape, and disk space."
		)

	# Rename unregistered.h5 → registered.h5 and write corrected frames into it
	os.replace(fnames[0], os.path.join(parent_dir, 'registered.h5'))
	datafile = h5py.File(os.path.join(parent_dir, 'registered.h5'), 'w')
	datafile.create_dataset("mov", (numframes, 512, 512))

	frames_written = 0

	for i in range(0, math.floor(numframes / 1000)):
		mov = cm.load(fnames_new[0])
		temp_data = np.array(mov[frames_written:frames_written + 1000, :, :])
		actual_frames = temp_data.shape[0]
		datafile["mov"][frames_written:frames_written + actual_frames, :, :] = temp_data
		frames_written += actual_frames
		del mov

	if numframes > frames_written:
		mov = cm.load(fnames_new[0])
		temp_data = np.array(mov[frames_written:mov.shape[0], :, :])
		datafile["mov"][frames_written:mov.shape[0], :, :] = temp_data
		del mov
		del temp_data

	cm.stop_server(dview=dview)

	if not keep_memmap:
		for i in range(0, len(fnames_new)):
			os.remove(fnames_new[i])

	if save_sample:
		with tifffile.TiffWriter(os.path.join(parent_dir, sample_name), bigtiff=False, imagej=False) as tif:
			for i in range(0, min(4000, numframes)):
				curfr = datafile["mov"][i,:,:].astype(np.int16)
				tiff_writer_append(tif, curfr, contiguous=False)

	datafile.close()


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
	# Get folder selections from the GUI — returns (paths array, 4×N bool array)
	workdirs, proc_opts = get_registration_options()
	if workdirs is None:
		print("No folders selected. Exiting.")
		sys.exit(0)

	# Write job file consumed by workers/pipeline_worker.py.
	# Keeping this as a plain JSON file means any alternative launcher
	# (CLI, web UI, etc.) can produce the same file to trigger the pipeline.
	job = {
		"sessions": workdirs.tolist(),
		"process_selections": proc_opts.tolist()
	}
	pathlib.Path(JOB_PATH).write_text(json.dumps(job, indent=2))
	print(f"Job written to {JOB_PATH}")
	print(f"Sessions queued: {len(workdirs)}")

	# Clear any stale failed unit from a previous run
	subprocess.run(
		["systemctl", "--user", "reset-failed", UNIT_NAME],
		check=False
	)

	# Launch workers/pipeline_worker.py as a persistent systemd user service.
	# Runs under the caiman python env; worker calls FAST via subprocess.
	# loginctl enable-linger must already be set (done by PreProcess2PImages.sh).
	subprocess.run([
		"systemd-run", "--user",
		f"--unit={UNIT_NAME}",
		"--description=Schollab caiman+FAST pipeline",
		f"--setenv=HOME={os.path.expanduser('~')}",
		# Worker subprocess needs same prefix as GUI so FAST_PYTHON resolves correctly.
		f"--setenv=SCHOLLAB_CONDA_ROOT={_schollab_conda_root()}",
		CAIMAN_PYTHON, WORKER_SCRIPT, JOB_PATH
	], check=True)

	print(f"Pipeline launched as systemd service '{UNIT_NAME}'.")
	print(f"Monitor: journalctl --user -f -u {UNIT_NAME}")
