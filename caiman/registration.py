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
import h5py
import glob
import pathlib
import shutil
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
from pipeline_job import apply_skip_caiman, want_unregistered_h5
from pipeline_run_log import CAIMAN_STEP_LABELS
from tif_to_h5 import (
	count_tiff_frames,
	list_acquisition_tiffs,
	motion_correction_inputs,
	require_matching_frame_count,
	stage_plain_tiffs,
	tif_stacks_to_h5,
)
from tiff_compat import tiff_writer_append
from std_projection import (
	STD_TIFF_NAME,
	STD_WINDOW,
	std_of_window,
	write_std_projection_tiff,
)

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

	# One mmap per input stack — do not keep only [0] or registered.h5 is truncated.
	fnames_new = _as_path_list(mc.mmap_file)
	if not fnames_new:
		raise RuntimeError(
			"Motion correction returned no memmap paths (mc.mmap_file empty). "
			"Check CaImAn version, input shape, and disk space."
		)
	print(
		f"  CaImAn memmap output ({len(fnames_new)} file(s)): "
		f"{os.path.basename(fnames_new[0])}"
	)
	return numframes, fnames_new


def _try_write_std_projection(parent_dir, datafile, numframes):
	"""
	Write std_projection.tif after registered.h5 / sample TIFF already exist.

	Why best-effort: STD is additive. A write failure must not skip 01_rigid.tif
	or mark motion correction failed.
	"""
	try:
		n_full = numframes // STD_WINDOW
		remainder = numframes % STD_WINDOW
		std_frames = []
		n_zero = 0
		for i in range(n_full):
			start = i * STD_WINDOW
			chunk = np.array(datafile["mov"][start:start + STD_WINDOW, :, :])
			proj = std_of_window(chunk)
			# Zero windows mean the H5 was padded; do not emit black STD pages.
			if not np.any(proj):
				n_zero += 1
				continue
			std_frames.append(proj)
		if n_zero:
			print(
				f"  STD projection: skipped {n_zero} all-zero "
				f"{STD_WINDOW}-frame window(s)"
			)
		# Remainder is already in registered.h5 from the original rewrite loop.
		if remainder:
			print(
				f"  STD projection: {n_full} complete {STD_WINDOW}-frame windows; "
				f"{remainder} leftover frames remain in registered.h5 "
				"(not omitted from motion correction)."
			)
		write_std_projection_tiff(parent_dir, std_frames)
	except Exception as exc:
		print(
			f"  WARNING: STD projection TIFF failed; "
			f"registered.h5 and sample TIFF kept: {exc}"
		)


def _as_path_list(paths):
	if not paths:
		return []
	if isinstance(paths, (str, bytes)):
		return [paths]
	return list(paths)


def _create_mov_dataset(datafile, expected_frames):
	if expected_frames is not None:
		datafile.create_dataset("mov", (expected_frames, 512, 512))
		return
	# Unknown length (H5-only input): grow as each CaImAn mmap is copied.
	datafile.create_dataset("mov", (0, 512, 512), maxshape=(None, 512, 512))


def _copy_one_mmap(datafile, mmap_path, dest_start, expected_frames):
	# Load one stack mmap at a time — all 22k frames at once would be tens of GB.
	mov = cm.load(mmap_path)
	try:
		n = int(mov.shape[0])
		if expected_frames is None:
			datafile["mov"].resize(dest_start + n, axis=0)
		print(f"  Copying mmap ({n} frames): {os.path.basename(mmap_path)}")
		written = 0
		while written < n:
			step = min(1000, n - written)
			temp = np.array(mov[written:written + step, :, :])
			lo = dest_start + written
			datafile["mov"][lo:lo + temp.shape[0], :, :] = temp
			written += temp.shape[0]
			del temp
		return n
	finally:
		del mov


def _write_registered_h5(parent_dir, mmap_paths, numframes, save_sample, sample_name,
		expected_frames=None):
	# Write via a temp name then replace. Never os.replace the MC input —
	# that is usually an acquisition TIFF, not unregistered.h5.
	mmap_paths = _as_path_list(mmap_paths)
	if not mmap_paths:
		raise RuntimeError("Motion correction returned no memmap paths.")
	registered_h5 = os.path.join(parent_dir, 'registered.h5')
	tmp_h5 = registered_h5 + '.tmp'
	if os.path.isfile(tmp_h5):
		os.remove(tmp_h5)
	datafile = None

	try:
		datafile = h5py.File(tmp_h5, 'w')
		_create_mov_dataset(datafile, expected_frames)
		print(
			f"  Rewriting corrected H5: {registered_h5} "
			f"from {len(mmap_paths)} memmap(s)"
		)
		offset = 0
		for path in mmap_paths:
			offset += _copy_one_mmap(datafile, path, offset, expected_frames)
		# Size from concatenated mmap pixels, not shift count or mmap[0] only.
		require_matching_frame_count(offset, expected_frames)
		if offset != numframes:
			print(
				f"  WARNING: CaImAn shifts={numframes} frames, "
				f"memmaps={offset}; writing registered.h5 from memmaps."
			)
		print(f"  Rewrote corrected H5 frames: {offset}")
		_write_sample_tiff(parent_dir, sample_name, datafile, offset, save_sample)
		# Additive: STD TIFF only after existing MC artifacts are written.
		_try_write_std_projection(parent_dir, datafile, offset)
	except Exception:
		if datafile is not None:
			datafile.close()
			datafile = None
		if os.path.isfile(tmp_h5):
			os.remove(tmp_h5)
		raise
	finally:
		if datafile is not None:
			datafile.close()
			print(f"  Closed H5 file: {registered_h5}")

	os.replace(tmp_h5, registered_h5)


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
	fnames_new = _as_path_list(fnames_new)
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


def _all_tiff_inputs(paths):
	return bool(paths) and all(
		p.lower().endswith(('.tif', '.tiff')) for p in paths
	)


def _mc_stage_dir(parent_dir):
	base = os.path.join(os.path.expanduser('~'), 'caiman_data', 'temp')
	name = os.path.basename(parent_dir.rstrip('/')) or 'session'
	return os.path.join(base, f'mc_input_{name}')


def _cleanup_stage_dir(stage_dir):
	if not stage_dir or not os.path.isdir(stage_dir):
		return
	try:
		shutil.rmtree(stage_dir)
		print(f"  Removed staged MC TIFFs: {stage_dir}")
	except OSError as exc:
		print(f"  WARNING: failed to remove staged MC TIFFs {stage_dir}: {exc}")


def register_one_session(parent_dir, mc_dict, keep_memmap, save_sample, sample_name):
	# Acquisition TIFFs first. Do not require or consume unregistered.h5.
	fnames = motion_correction_inputs(parent_dir)
	print(f"  CaImAn input ({len(fnames)} path(s)): {os.path.basename(fnames[0])}")
	expected_frames = None
	stage_dir = None
	# Workers load TIFFs themselves; strip OME linking so every stack is used.
	if _all_tiff_inputs(fnames):
		expected_frames = count_tiff_frames(fnames)
		stage_dir = _mc_stage_dir(parent_dir)
		print(
			f"  Staging {len(fnames)} stacks ({expected_frames} frames) "
			"as non-OME TIFFs for CaImAn..."
		)
		fnames = stage_plain_tiffs(fnames, stage_dir)
	mc_dict['fnames'] = fnames
	mc_dict['upsample_factor_grid'] = 8
	opts = params.CNMFParams(params_dict=mc_dict)
	fnames_new = []

	try:
		numframes, fnames_new = _run_motion_correction(parent_dir, fnames, opts, mc_dict)
		_write_registered_h5(
			parent_dir, fnames_new, numframes, save_sample, sample_name,
			expected_frames=expected_frames,
		)
	finally:
		# Memmaps are large temporary files; clean them even when H5/sample writing fails.
		_cleanup_memmaps(fnames_new, keep_memmap)
		_cleanup_stage_dir(stage_dir)


def _h5_artifact_line(folder, filename):
	"""Format H5 artifact with size for run log."""
	path = os.path.join(folder, filename)
	if os.path.isfile(path):
		size_gb = os.path.getsize(path) / 1e9
		return f'{filename} ({size_gb:.2f} GB)'
	return None


def _sample_tif_artifacts(folder):
	"""List rigid/nonrigid sample TIFFs and STD projection from motion correction."""
	arts = []
	for pattern in ('*_rigid.tif', '*_nonrigid.tif'):
		for path in sorted(glob.glob(os.path.join(folder, pattern))):
			arts.append(os.path.basename(path))
	std_path = os.path.join(folder, STD_TIFF_NAME)
	if os.path.isfile(std_path):
		arts.append(STD_TIFF_NAME)
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


def _delete_stale_h5(folder):
	for name in ('unregistered.h5', 'registered.h5'):
		path = os.path.join(folder, name)
		if os.path.exists(path):
			os.remove(path)
			print(f"Deleted stale file: {path}")


def _try_write_unregistered_h5(folder, steps):
	"""Opt-in TIFs→H5 with no motion step. Returns False if the write failed."""
	t0 = time.perf_counter()
	if not list_acquisition_tiffs(folder):
		print(f"WARNING: No source TIFs found in {folder}")
		print(f"  Skipping TIFs→H5 step — registered.h5 left untouched.")
		steps.append({
			'name': CAIMAN_STEP_LABELS[0],
			'status': 'skipped',
			'duration_s': round(time.perf_counter() - t0, 1),
			'detail': 'no source TIFs',
		})
		return True
	try:
		_delete_stale_h5(folder)
		h5_name = os.path.join(folder, 'unregistered.h5')
		tif_stacks_to_h5(folder, h5_name, frame_offset=False)
		art = _h5_artifact_line(folder, 'unregistered.h5')
		steps.append({
			'name': CAIMAN_STEP_LABELS[0],
			'status': 'ok',
			'duration_s': round(time.perf_counter() - t0, 1),
			'artifacts_line': art or 'unregistered.h5',
		})
		return True
	except Exception as exc:
		steps.append({
			'name': CAIMAN_STEP_LABELS[0],
			'status': 'failed',
			'duration_s': round(time.perf_counter() - t0, 1),
			'error': str(exc),
		})
		return False


def _run_motion_steps(folder, row, mc_dict, steps):
	"""Run selected rigid/NoRMCorre steps. Returns False on the first failure."""
	n_procs = 0
	for step_idx, sample_suffix in ((1, 'rigid'), (2, 'rigid'), (3, 'nonrigid')):
		if not row[step_idx]:
			continue
		label = CAIMAN_STEP_LABELS[step_idx]
		t0 = time.perf_counter()
		try:
			mc_dict['pw_rigid'] = step_idx == 3
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
			return False
	return True


def _register_one_folder(folder, row, mc_dict):
	"""
	Run CaImAn steps for one folder; return summary dict for pipeline run log.

	Does not raise on step failure — caller checks summary['result'].
	"""
	steps = []
	print(folder)

	# MC never writes unregistered.h5, even if TIFs→H5 is also checked.
	if row[0] and any(row[1:4]):
		print(
			"  TIFs→H5 ignored — motion correction reads acquisition TIFFs; "
			"unregistered.h5 is not written."
		)
		steps.append({
			'name': CAIMAN_STEP_LABELS[0],
			'status': 'skipped',
			'duration_s': 0.0,
			'detail': 'MC reads TIFFs; unregistered.h5 not written',
		})
	elif want_unregistered_h5(row):
		if not _try_write_unregistered_h5(folder, steps):
			return _finalize_caiman_summary(folder, steps, row)

	if not _run_motion_steps(folder, row, mc_dict, steps):
		return _finalize_caiman_summary(folder, steps, row)

	if want_unregistered_h5(row):
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
	Run CaImAn motion correction on acquisition TIFFs (or existing H5 fallback).

	Parameters:
		sessions_to_run (list): Directory paths to find data in.
		process_selections (np.array): 4×N bool array — rows are:
			[TIFs→H5, first rigid, additional rigid, NoRMCorre]
		TIFs→H5 writes unregistered.h5 only when no motion step is selected.

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
		print(f"Log dir: {job['run_log_dir']}")
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
