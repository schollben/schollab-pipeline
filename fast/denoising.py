# Copyright (C) [2025] [Yiqun Wang]
# SPDX-License-Identifier: GPL-3.0-or-later
# updated by Scholl Lab, 2025-07-13
# FAST Pipeline Main Module

# This module orchestrates the complete FAST (Fluorescence Analysis and Source Tracking)
# pipeline for processing two-photon microscopy data. It handles the conversion of HDF5
# registered data to training formats, model training, inference on test data, and
# conversion of results back to HDF5 format.

# PREREQUISITES:
#     - Motion correction must be completed FIRST
#     - Input data must contain a 'registered.h5' file (output from motion correction)
#     - CUDA-compatible GPU is required
#     - All dependencies from requirements.txt installed
#     - config.json present in the same directory as this script
#     - train.py must use args.results_dir (not train_folder parent) for checkpoint dir
#     - This script should be run with the FAST environment activated
#     - Intermediate files go to {session}/scratch/ (removed after each run)
#     - To watch GPU: 'watch -n 2 nvidia-smi'

# WORKFLOW:
#     1. Convert registered.h5 to TIFF stacks → session/scratch/registered/
#     2. Train deep learning model on selected frames → checkpoint on session root
#     3. Run inference on all registered TIFFs → session/scratch/result/
#     4. Convert denoised result TIFFs → inference.h5 on session root
#     5. Copy example TIFF, delete session/scratch/, write completion sentinel

# INPUT:
#     - config.json: data folders, hyperparameters, paths

# OUTPUT (per folder):
#     - checkpoint/: Trained model weights and configuration (permanent drive)
#     - inference.h5: Denoised output in HDF5 format (permanent drive)
#     - *.tif: Example result TIFF stack (permanent drive)
#     - _fast_complete: Sentinel file — only present after full successful run

# NOTES:
#     - CUDA is mandatory; CPU-only execution is not supported
#     - Set CUDA_LAUNCH_BLOCKING=1 before running to surface CUDA errors as tracebacks
#     - Intermediate TIFF directories live on tmpfs and are deleted post-processing
#     - exFAT filesystem errors=remount-ro: use tmpfs for intermediates to avoid trigger

import argparse
import gc
import glob
import h5py
import json
import logging
import numpy as np
import os
import shutil
import signal
import subprocess
import threading
import time
import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import psutil
import tifffile
import torch

from train import goTraining
from test import goTesting
from utils.config import json2args
from utils.h5_utils import h5_to_tiff, sort_tif_stack_paths


FAST_CONFIG_PATH = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	'config.json'
)
LEGACY_FAST_CONFIG_PATH = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	'pipeline_config.json'
)


def default_fast_config_path() -> str:
	"""Prefer fast/config.json; tolerate legacy fast/pipeline_config.json during migration."""
	if os.path.exists(FAST_CONFIG_PATH):
		return FAST_CONFIG_PATH
	if os.path.exists(LEGACY_FAST_CONFIG_PATH):
		print(
			f"WARNING: Using legacy FAST config path: {LEGACY_FAST_CONFIG_PATH}\n"
			"  Rename it to fast/config.json when convenient."
		)
		return LEGACY_FAST_CONFIG_PATH
	return FAST_CONFIG_PATH


# =============================================================================
# Configuration
# =============================================================================

def load_pipeline_config(path: str) -> dict:
	"""
	Load pipeline orchestration config from JSON.

	Separates folder list and runtime parameters from code so denoising.py
	never needs to be edited for routine runs — only fast/config.json does.
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"FAST config not found: {path}")
	with open(path) as f:
		cfg = json.load(f)
	required = [
		'fast_dir', 'skip_training', 'train_frames', 'tiff_chunk_size',
		'h5_write_batch_frames', 'minibatch_size', 'batch_size', 'num_workers', 'epochs', 'data_folders'
	]
	missing = [k for k in required if k not in cfg]
	if missing:
		raise KeyError(f"Missing keys in FAST config: {missing}")
	return cfg


def resolve_config_path(value: Optional[str], env_name: str, default: str) -> str:
	"""Resolve machine-local paths from env/config without hardcoding one user home."""
	raw = os.environ.get(env_name) or value or default
	return os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))


@dataclass
class PipelineConfig:
	"""
	Runtime configuration loaded from fast/config.json.

	Using a dataclass (rather than a raw dict) gives attribute access,
	type clarity, and a single place to add defaults or validation.
	"""
	fast_dir:         str
	skip_training:    bool
	train_frames:     int
	tiff_chunk_size:  int
	h5_write_batch_frames: int
	minibatch_size:   int
	batch_size:       int
	num_workers:      int
	epochs:           int
	base_config_path: str   # derived: fast_dir/userparams.json

	@staticmethod
	def from_dict(cfg: dict) -> 'PipelineConfig':
		fast_dir = resolve_config_path(
			cfg.get('fast_dir'), 'FAST_DIR', os.path.join('~', 'Documents', 'FAST')
		)
		return PipelineConfig(
			fast_dir         = fast_dir,
			skip_training    = cfg['skip_training'],
			train_frames     = cfg['train_frames'],
			tiff_chunk_size  = cfg['tiff_chunk_size'],
			h5_write_batch_frames = cfg['h5_write_batch_frames'],
			minibatch_size   = cfg['minibatch_size'],
			batch_size       = cfg['batch_size'],
			num_workers      = cfg['num_workers'],
			epochs           = cfg['epochs'],
			base_config_path = os.path.join(fast_dir, 'userparams.json'),
		)


# =============================================================================
# Path management
# =============================================================================

from folder_paths import FolderPaths, remove_checkpoint_for_rerun, remove_scratch_for_rerun

def setup_logging(log_path: str, batch_log_path: Optional[str] = None, file_mode: str = 'w') -> logging.Logger:
	"""
	Configure file + console logger.

	FileHandler keeps the file descriptor open for the process lifetime —
	faster than open()/close() on every call, important given the
	MemoryMonitor writes a DEBUG line every 30 seconds.

	Log levels:
	  DEBUG   — memory stats, directory operations (file only)
	  INFO    — step start/end, folder progress (file + console)
	  WARNING — non-fatal issues (file + console)
	  ERROR   — exceptions with full traceback (file + console)
	"""
	logger = logging.getLogger('FAST')
	logger.setLevel(logging.DEBUG)
	fmt = logging.Formatter(
		'[%(asctime)s] [%(levelname)-8s] %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	fh = logging.FileHandler(log_path, mode=file_mode)
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(fmt)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(ch)
	if batch_log_path:
		bh = logging.FileHandler(batch_log_path, mode='a')
		bh.setLevel(logging.INFO)
		bh.setFormatter(fmt)
		logger.addHandler(bh)
	return logger


def log_startup_info(logger: logging.Logger, log_path: str, cfg: PipelineConfig):
	"""
	Log pipeline configuration and hardware at startup.

	Called once at the beginning of main() so every log file is fully
	self-contained and diagnosable without cross-referencing other files.
	"""
	logger.info(f"FAST Pipeline  |  log: {log_path}")
	logger.info(
		f"Config: EPOCHS={cfg.epochs} TRAIN_FRAMES={cfg.train_frames} "
		f"TIFF_CHUNK_SIZE={cfg.tiff_chunk_size} H5_WRITE_BATCH={cfg.h5_write_batch_frames} "
		f"MINIBATCH={cfg.minibatch_size} WORKERS={cfg.num_workers} "
		f"SKIP_TRAINING={cfg.skip_training} SCRATCH=<each data_folder>/scratch"
	)
	logger.info(
		f"GPU: {torch.cuda.get_device_name(0)}  |  "
		f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB VRAM"
	)


class MemoryMonitor:
	"""
	Background daemon thread that periodically logs system and GPU resource usage.

	Runs independently of the main pipeline thread so memory stats are captured
	even if the main thread hangs or crashes silently inside a C extension
	(e.g. CUDA, h5py, tifffile). Includes GPU temperature and power draw
	to help diagnose thermal-related crashes.
	"""

	def __init__(self, logger: logging.Logger, interval: int = 30):
		self.logger   = logger
		self.interval = interval
		self._step    = 'init'
		self._stop    = threading.Event()
		# daemon=True ensures the thread does not block process exit
		self._thread  = threading.Thread(target=self._run, daemon=True)

	def set_step(self, step: str):
		"""Called at the start of each pipeline step so logs are labelled correctly."""
		self._step = step

	def start(self):
		self._thread.start()

	def stop(self):
		self._stop.set()

	def _run(self):
		while not self._stop.wait(self.interval):
			ram = psutil.virtual_memory()
			parts = [
				f"step={self._step}",
				f"RAM {ram.used/1e9:.1f}/{ram.total/1e9:.1f}GB ({ram.percent:.0f}%)",
			]
			if torch.cuda.is_available():
				alloc  = torch.cuda.memory_allocated() / 1e9   # live tensors
				reserv = torch.cuda.memory_reserved() / 1e9    # PyTorch allocator pool
				total  = torch.cuda.get_device_properties(0).total_memory / 1e9
				parts.append(
					f"GPU alloc={alloc:.1f} reserved={reserv:.1f} total={total:.1f}GB"
				)
				# Temperature and power via nvidia-smi — negligible at 30s interval
				try:
					r = subprocess.run(
						['nvidia-smi',
						 '--query-gpu=temperature.gpu,power.draw',
						 '--format=csv,noheader,nounits'],
						capture_output=True, text=True, timeout=5
					)
					temp, power = r.stdout.strip().split(', ')
					parts.append(f"temp={temp}C power={power}W")
				except Exception:
					pass  # non-critical — omit if nvidia-smi unavailable
			self.logger.debug(' | '.join(parts))


class StepRecorder:
	"""Collect per-step outcomes for pipeline run log summary JSON."""

	def __init__(self):
		self.steps = []

	def record(self, name, status, duration_s, detail=None, error=None):
		entry = {
			'name': name,
			'status': status,
			'duration_s': round(duration_s, 1),
		}
		if detail:
			entry['detail'] = detail
		if error:
			entry['error'] = error
		self.steps.append(entry)

	def skip_pair(self, reason):
		"""Record steps 1-2 skipped together (resume / skip_training)."""
		for name in ('step1_tiff_export', 'step2_training'):
			self.record(name, 'skipped', 0, detail=reason)


@contextmanager
def log_step(logger: logging.Logger, monitor: MemoryMonitor, name: str, recorder=None):
	"""
	Context manager that wraps a pipeline step with:
	  - Step label in MemoryMonitor so memory logs are tagged correctly
	  - INFO log on entry and exit with elapsed wall-clock time
	  - ERROR + full traceback via logger.exception() on failure, then re-raises
	  - Optional StepRecorder for combined pipeline run log
	"""
	monitor.set_step(name)
	logger.info(f"{'─'*50}")
	logger.info(f"START  {name}")
	t0 = time.time()
	detail_holder = {'detail': None}

	class StepDetail:
		def set(self, value):
			detail_holder['detail'] = value

	try:
		yield StepDetail()
		elapsed = time.time() - t0
		logger.info(f"DONE   {name}  ({elapsed:.1f}s)")
		if recorder is not None:
			recorder.record(name, 'ok', elapsed, detail=detail_holder['detail'])
	except Exception as exc:
		elapsed = time.time() - t0
		logger.error(f"FAILED {name}  ({elapsed:.1f}s)")
		logger.exception("Traceback:")
		if recorder is not None:
			recorder.record(name, 'failed', elapsed, error=str(exc))
		raise


# =============================================================================
# Helpers
# =============================================================================

def _reset_dir(path: str, logger: logging.Logger):
	"""
	Delete directory if it exists, then recreate it empty.

	Used in Step 1 (always re-export registered/ from h5 as source of truth)
	and Step 3 (clear result/ to prevent stale partial TIFFs from a previous
	crash causing the stack-size assertion error in tif_stacks_to_h5).
	"""
	if os.path.exists(path):
		shutil.rmtree(path)
		logger.debug(f"Cleared: {path}")
	os.makedirs(path)
	logger.debug(f"Created: {path}")


def _find_latest_checkpoint_config(checkpoint_root: str) -> Optional[str]:
	"""
	Return path to config.json in the most recent checkpoint subdir, or None.

	Checkpoint subdirs are named YYYYMMDDHHMM by goTraining so reverse
	alphabetical = reverse chronological. Searches newest-first and skips
	any incomplete subdirs (those without config.json) to avoid the
	FileNotFoundError that plagued earlier runs.
	"""
	if not os.path.isdir(checkpoint_root):
		return None
	for subdir in sorted(os.listdir(checkpoint_root), reverse=True):
		cfg = os.path.join(checkpoint_root, subdir, 'config.json')
		if os.path.isfile(cfg):
			return cfg
	return None


def setup_cuda():
	"""Assert CUDA is available and configure cuDNN for optimal training performance."""
	assert torch.cuda.is_available(), "Currently, we only support CUDA version"
	torch.backends.cudnn.enabled   = True
	torch.backends.cudnn.benchmark = True


# =============================================================================
# Pipeline steps
# =============================================================================

def step1_export_tiffs(
	paths: FolderPaths, cfg: PipelineConfig, logger: logging.Logger, monitor: MemoryMonitor,
	recorder=None,
):
	"""
	Convert registered.h5 to TIFF chunks and copy first chunk to training/.

	Always clears registered/ and training/ first — registered.h5 is the
	source of truth. Both dirs live under session/scratch/ for FAST intermediates.
	Only the first chunk is copied to training/ since self-supervised
	training needs only a representative sample of the recording.
	"""
	with log_step(logger, monitor, 'step1_tiff_export', recorder) as step_detail:
		_reset_dir(paths.registered, logger)
		_reset_dir(paths.training, logger)
		logger.info(f"  TIFF chunk size: {cfg.tiff_chunk_size} frames")
		h5_to_tiff(paths.h5, output_dir=paths.registered, chunk_size=cfg.tiff_chunk_size)

		tif_files = sort_tif_stack_paths(glob.glob(os.path.join(paths.registered, '*.tif')))
		if not tif_files:
			raise FileNotFoundError(f"No TIFFs created in {paths.registered}")

		first_tif = tif_files[0]
		shutil.copy2(first_tif, os.path.join(paths.training, os.path.basename(first_tif)))
		logger.info(f"  Copied {os.path.basename(first_tif)} → training/")
		logger.info(f"  Total TIFF chunks available for inference: {len(tif_files)}")
		step_detail.set(f'registered/ ({len(tif_files)} chunks)')


def step2_train(
	paths: FolderPaths, cfg: PipelineConfig,
	logger: logging.Logger, monitor: MemoryMonitor,
	recorder=None,
):
	"""
	Train the Unet_Lite denoiser on the first TIFF chunk.

	Uses self-supervised spatiotemporal loss — no clean ground truth needed.
	Hyperparameters come from PipelineConfig (loaded from fast/config.json).
	The checkpoint is written to paths.checkpoint on the permanent drive —
	NOT to tmpfs — so it survives across runs.

	NOTE: train.py must derive checkpoint_dir from args.results_dir (not
	args.train_folder parent) for the checkpoint to land on the correct drive.
	"""
	with log_step(logger, monitor, 'step2_training', recorder) as step_detail:
		with open(cfg.base_config_path, 'r') as f:
			params = json.load(f)

		# Override userparams.json defaults with fast/config.json values
		params.update({
			'train_frames':   cfg.train_frames,
			'miniBatch_size': cfg.minibatch_size,
			'batch_size':     cfg.batch_size,
			'num_workers':    cfg.num_workers,
			'save_freq':      cfg.epochs,  # write checkpoint only at final epoch
			'epochs':         cfg.epochs,
			'results_dir':    paths.root,  # checkpoint goes to permanent drive
			'mode':           'train',
		})

		# Per-folder working copy of config — deleted in Step 5
		run_config_path = os.path.join(paths.root, '_run_config.json')
		with open(run_config_path, 'w') as f:
			json.dump(params, f, indent=4)

		args = json2args(run_config_path)
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
		args.train_folder = paths.training
		logger.info(f"  Training data: {args.train_folder}")
		goTraining(args)
		step_detail.set('checkpoint/')


def step3_inference(
	paths: FolderPaths, checkpoint_config: str,
	logger: logging.Logger, monitor: MemoryMonitor,
	recorder=None,
):
	"""
	Run denoising inference on all registered TIFF chunks using the trained model.

	result/ is always cleared first to prevent stale TIFFs from a previous
	crash causing a stack-size assertion failure in tif_stacks_to_h5 (Step 4).

	goTesting uses args.test_path for input and args.results_dir for output.
	It silently returns (no exception) on missing files, so we validate both
	before and after to surface failures as proper exceptions.
	"""
	with log_step(logger, monitor, 'step3_inference', recorder) as step_detail:
		# Guard: registered/ must have TIFFs — goTesting silently returns otherwise
		reg_tifs = sort_tif_stack_paths(glob.glob(os.path.join(paths.registered, '*.tif')))
		if not reg_tifs:
			raise FileNotFoundError(
				f"registered/ has no TIFFs at {paths.registered}. "
				f"Delete the checkpoint and re-run to trigger a clean start."
			)
		logger.info(f"  Input TIFFs: {len(reg_tifs)} in {paths.registered}")

		# Always clear result/ — stale TIFFs from a crash cause Step 4 assertion errors
		_reset_dir(paths.result, logger)

		logger.info(f"  Checkpoint config: {checkpoint_config}")
		with open(checkpoint_config, 'r') as f:
			test_params = json.load(f)

		# Guard: checkpoint weights file must exist — goTesting silently returns otherwise
		ckpt_weights = test_params.get('checkpoint_path', '')
		if not os.path.exists(ckpt_weights):
			raise FileNotFoundError(
				f"Checkpoint weights not found: {ckpt_weights}\n"
				f"  (from config: {checkpoint_config})"
			)

		# goTesting writes result TIFFs under session/scratch/ (not the session root).
		test_params['results_dir'] = paths.scratch
		# Write a temporary copy so the permanent checkpoint config is not mutated
		tmp_config = os.path.join(paths.root, '_inference_config.json')
		with open(tmp_config, 'w') as f:
			json.dump(test_params, f, indent=4)

		args = json2args(tmp_config)
		# goTesting reads TIFFs from args.test_path
		args.test_path = paths.registered
		logger.info(f"  Test data: {args.test_path}")
		logger.info(f"  Output dir: {paths.result}")

		goTesting(args)

		# Validate goTesting actually produced output — it silently returns on error
		result_tifs = sort_tif_stack_paths(glob.glob(os.path.join(paths.result, '*.tif')))
		if not result_tifs:
			raise RuntimeError(
				f"goTesting returned without writing any TIFFs to {paths.result}.\n"
				f"  test_path: {args.test_path}\n"
				f"  checkpoint: {ckpt_weights}\n"
				f"  Check GPU memory and checkpoint validity."
			)
		logger.info(f"  Result TIFFs written: {len(result_tifs)}")
		step_detail.set(f'result/ ({len(result_tifs)} TIFFs)')

		# Clean up temporary inference config
		if os.path.exists(tmp_config):
			os.remove(tmp_config)


def _stream_result_tifs_to_h5(result_dir: str, h5_savename: str, write_batch_frames: int):
	"""
	Merge result TIFF stacks to H5 using small frame batches to limit peak RAM.
	"""
	tif_fnames = sort_tif_stack_paths(glob.glob(os.path.join(result_dir, "*.tif")))
	if not tif_fnames:
		raise FileNotFoundError(f"No result TIFFs found in {result_dir}")
	if write_batch_frames < 1:
		raise ValueError("h5_write_batch_frames must be >= 1")

	with tifffile.TiffFile(tif_fnames[0]) as first_stack_handle:
		stack_depth = len(first_stack_handle.pages)
		first_shape = first_stack_handle.pages[0].shape

	if len(first_shape) < 2:
		raise ValueError(f"Unexpected TIFF page shape in {tif_fnames[0]}: {first_shape}")
	stack_width, stack_height = first_shape[-2], first_shape[-1]

	if stack_depth > 1:
		for tif_path in tif_fnames[1:-1]:
			with tifffile.TiffFile(tif_path) as tif_stack_handle:
				this_stack_depth = len(tif_stack_handle.pages)
			if this_stack_depth != stack_depth:
				raise AssertionError(
					f"Stack sizes inconsistent: expected {stack_depth} frames "
					f"but got {this_stack_depth} for file {tif_path}"
				)

	with tifffile.TiffFile(tif_fnames[-1]) as last_stack_handle:
		last_stack_length = len(last_stack_handle.pages)

	out_data_frames = (stack_depth * (len(tif_fnames) - 1)) + last_stack_length
	write_end_ind = 0

	with h5py.File(h5_savename, 'w') as f_out:
		f_out.create_dataset('mov', (out_data_frames, stack_width, stack_height))
		for tif_path in tif_fnames:
			with tifffile.TiffFile(tif_path) as tif_stack_handle:
				pages = tif_stack_handle.pages
				page_count = len(pages)
				for batch_start in range(0, page_count, write_batch_frames):
					batch_end = min(batch_start + write_batch_frames, page_count)
					batch = np.stack([pages[j].asarray() for j in range(batch_start, batch_end)], axis=0)
					write_start_ind = write_end_ind
					write_end_ind = write_start_ind + batch.shape[0]
					f_out['mov'][write_start_ind:write_end_ind, :, :] = batch
					del batch


def step4_export_h5(
	paths: FolderPaths, cfg: PipelineConfig, logger: logging.Logger, monitor: MemoryMonitor,
	recorder=None,
):
	"""
	Merge all inference TIFF chunks from result/ (tmpfs) into inference.h5 (permanent drive).

	Output uses HDF5 key 'mov' — the convention expected by CaImAn CNMF
	for the next pipeline stage (source extraction / ROI detection).
	Output file size is logged to make truncated writes easy to detect.
	"""
	with log_step(logger, monitor, 'step4_h5_export', recorder) as step_detail:
		logger.info(f"  H5 write batch size: {cfg.h5_write_batch_frames} frames")
		_stream_result_tifs_to_h5(paths.result, paths.inference_h5, cfg.h5_write_batch_frames)
		size_gb = os.path.getsize(paths.inference_h5) / 1e9
		logger.info(f"  Saved: {paths.inference_h5}  ({size_gb:.2f} GB)")
		step_detail.set(f'inference.h5 ({size_gb:.2f} GB)')


def _remove_folder_scratch(paths: FolderPaths, logger: logging.Logger):
	"""
	Delete {session}/scratch/ after a FAST run.

	Why: scratch holds only FAST intermediates; permanent outputs live on root.
	Called from process_folder finally so failures still clean up.
	"""
	if remove_scratch_for_rerun(paths.scratch):
		logger.info(f"  Removed scratch: {paths.scratch}")


def step5_cleanup(
	paths: FolderPaths, logger: logging.Logger, monitor: MemoryMonitor,
	recorder=None,
):
	"""
	Post-processing cleanup:
	  1. Copy one example result TIFF to the root folder for quick inspection
	  2. Remove temporary _run_config.json
	  3. Write the _fast_complete sentinel file

	The sentinel is written LAST — its presence is the only reliable signal
	that the full pipeline completed. Scratch removal happens in process_folder
	finally (success or failure).
	"""
	with log_step(logger, monitor, 'step5_cleanup', recorder) as step_detail:
		result_tifs = sort_tif_stack_paths(glob.glob(os.path.join(paths.result, '*.tif')))
		if result_tifs:
			dest = os.path.join(paths.root, os.path.basename(result_tifs[0]))
			shutil.copy2(result_tifs[0], dest)
			logger.info(f"  Example TIFF: {os.path.basename(dest)}")
		else:
			logger.warning("  No result TIFFs found to copy as example")

		run_config = os.path.join(paths.root, '_run_config.json')
		if os.path.exists(run_config):
			os.remove(run_config)

		# Sentinel written last — presence = pipeline fully completed
		with open(paths.sentinel, 'w') as f:
			f.write(datetime.datetime.now().isoformat())
		logger.info(f"  Written: {paths.sentinel}")
		step_detail.set('_fast_complete')


def _write_fast_summary(path, summary):
	"""Write FAST folder summary JSON for pipeline_worker."""
	if not path:
		return
	os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(summary, f, indent=2)


def _inference_h5_detail(paths):
	if os.path.isfile(paths.inference_h5):
		size_gb = os.path.getsize(paths.inference_h5) / 1e9
		return f'inference.h5 {size_gb:.2f} GB'
	return None


def process_folder(
	dataFolder: str, cfg: PipelineConfig,
	logger: logging.Logger, monitor: MemoryMonitor,
	recorder=None, summary_out=None,
):
	"""
	Orchestrate the full FAST pipeline for a single data folder.

	Auto-skip / resume logic (checked in order):
	  1. _fast_complete exists → fully done, skip entirely
	  2. Otherwise run FAST: delete scratch/ and checkpoint/ (unless skip_training
	     for checkpoint) so each re-run starts clean; then Steps 1–5 or skip
	     1–2 when skip_training reuses an existing checkpoint

	Scratch is removed at run start and again in finally after each attempt.
	"""
	paths = FolderPaths.from_root(dataFolder)
	recorder = recorder or StepRecorder()

	if not os.path.exists(paths.h5):
		logger.warning(
			f"SKIPPING — registered.h5 not found: {dataFolder}\n"
			f"  Run CaImAn motion correction on this folder first."
		)
		summary = {
			'result': 'not_run',
			'reason': 'registered.h5 not found',
			'steps': [],
		}
		_write_fast_summary(summary_out, summary)
		return summary

	# Sentinel check — only written after Step 5 fully completes
	if os.path.exists(paths.sentinel):
		logger.info(f"SKIPPING (complete): {dataFolder}")
		detail = _inference_h5_detail(paths)
		summary = {
			'result': 'skipped',
			'reason': '_fast_complete present',
			'detail': detail,
			'steps': [],
		}
		_write_fast_summary(summary_out, summary)
		return summary

	logger.info(f"\n{'='*60}")
	logger.info(f"Processing: {dataFolder}")
	logger.info(f"{'='*60}")

	if remove_checkpoint_for_rerun(paths.checkpoint, cfg.skip_training):
		logger.info(f"  Removed existing checkpoint/ for fresh training: {paths.checkpoint}")
	if remove_scratch_for_rerun(paths.scratch):
		logger.info(f"  Removed existing scratch/ for clean re-run: {paths.scratch}")

	ran_fast = False
	try:
		ran_fast = True
		checkpoint_config = _find_latest_checkpoint_config(paths.checkpoint)
		skip_training     = cfg.skip_training or (checkpoint_config is not None)

		if not skip_training:
			step1_export_tiffs(paths, cfg, logger, monitor, recorder)
			step2_train(paths, cfg, logger, monitor, recorder)
			checkpoint_config = _find_latest_checkpoint_config(paths.checkpoint)
		else:
			reason = 'skip_training flag' if cfg.skip_training else f'checkpoint: {checkpoint_config}'
			logger.info(f"[Steps 1-2] SKIPPED ({reason})")
			recorder.skip_pair(reason)

			reg_tifs = glob.glob(os.path.join(paths.registered, '*.tif'))
			if not reg_tifs:
				if os.path.isdir(paths.registered):
					logger.warning("  registered/ exists but is EMPTY — re-running Step 1")
				else:
					logger.info("  registered/ missing — re-running Step 1")
				step1_export_tiffs(paths, cfg, logger, monitor, recorder)

		if checkpoint_config is None:
			raise FileNotFoundError(
				f"No valid checkpoint config.json found in {paths.checkpoint}"
			)

		step3_inference(paths, checkpoint_config, logger, monitor, recorder)
		step4_export_h5(paths, cfg, logger, monitor, recorder)
		step5_cleanup(paths, logger, monitor, recorder)

		logger.info(f"Done: {dataFolder}")
		logger.info(f"  checkpoint/  — model weights + config")
		logger.info(f"  inference.h5 — denoised output")
		logger.info(f"  *.tif        — example result stack")
		summary = {'result': 'succeeded', 'steps': recorder.steps}
		_write_fast_summary(summary_out, summary)
		return summary
	except Exception as exc:
		summary = {
			'result': 'failed',
			'steps': recorder.steps,
			'error': str(exc),
		}
		if not os.path.isfile(paths.inference_h5):
			summary['artifacts_note'] = 'checkpoint/ (partial), no inference.h5, no _fast_complete'
		_write_fast_summary(summary_out, summary)
		raise
	finally:
		if ran_fast:
			_remove_folder_scratch(paths, logger)


# =============================================================================
# Entry point
# =============================================================================

def main():

	# # Force synchronous CUDA execution so errors produce tracebacks instead of silent death
    # os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

	# Accept optional --config argument so different config files can be used
	# without modifying this script: python denoising.py --config my_config.json
	parser = argparse.ArgumentParser(description='FAST denoising pipeline')
	parser.add_argument(
		'--config',
		default=default_fast_config_path(),
		help='Path to FAST config JSON (default: fast/config.json next to denoising.py)'
	)
	parser.add_argument(
		'--batch-log',
		default=os.environ.get('SCHOLLAB_BATCH_LOG'),
		help='Append INFO+ logs to consolidated verbose pipeline log'
	)
	parser.add_argument(
		'--log-file',
		default=os.environ.get('SCHOLLAB_FAST_LOG'),
		help='FAST step-detail log path (repo log/ when run via pipeline_worker)'
	)
	parser.add_argument(
		'--summary-out',
		default=None,
		help='Write per-folder FAST summary JSON for combined pipeline run log'
	)
	cli = parser.parse_args()

	raw_cfg = load_pipeline_config(cli.config)
	cfg     = PipelineConfig.from_dict(raw_cfg)
	folders = raw_cfg['data_folders']

	setup_cuda()

	run_ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	if cli.log_file:
		# Worker passes repo log/{ts}.fast.log — append across folders in one run.
		log_path = cli.log_file
		logs_dir = os.path.dirname(log_path) or '.'
		log_mode = 'a'
	else:
		logs_dir = os.path.join(cfg.fast_dir, 'logs')
		os.makedirs(logs_dir, exist_ok=True)
		log_path = os.path.join(logs_dir, f'_pipeline_log_{run_ts}.txt')
		log_mode = 'w'
	os.makedirs(logs_dir, exist_ok=True)
	batch_log_path = cli.batch_log or None

	logger  = setup_logging(log_path, batch_log_path=batch_log_path, file_mode=log_mode)
	monitor = MemoryMonitor(logger, interval=30)
	monitor.start()

	log_startup_info(logger, log_path, cfg)

	marker_path = os.path.join(logs_dir, '_pipeline_status.json')
	status      = {'status': 'running', 'started': run_ts}

	def _write_marker(s: str, extra: dict = None):
		status['status'] = s
		status['updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		if extra:
			status.update(extra)
		with open(marker_path, 'w') as f:
			json.dump(status, f, indent=2)

	_write_marker('running')

	def _on_signal(signum, _):
		# Catches SIGTERM and SIGHUP for a clean shutdown.
		# SIGKILL cannot be caught — status file will remain 'running' in that case.
		logger.warning(f"Caught signal {signum} — pipeline interrupted")
		_write_marker('interrupted', {'signal': signum})
		monitor.stop()
		raise SystemExit(1)

	for sig in (signal.SIGTERM, signal.SIGHUP):
		signal.signal(sig, _on_signal)

	total = len(folders)
	logger.info(f"Folders to process: {total}")

	try:
		for i, folder in enumerate(folders, 1):
			logger.info(f"\n{'#'*60}")
			logger.info(f"Folder {i}/{total}: {folder}")
			logger.info(f"{'#'*60}")
			status['current_folder'] = folder
			_write_marker('running')

			try:
				process_folder(
					folder, cfg, logger, monitor,
					summary_out=cli.summary_out,
				)
				_write_marker('running', {'last_completed_folder': folder})
			except Exception:
				logger.exception(
					f"Pipeline failed on folder: {status.get('current_folder', 'unknown')}"
				)
				_write_marker('error')
				raise

			# Explicit memory cleanup between folders.
			# PyTorch's CUDA allocator retains reserved blocks across calls (for reuse)
			# but this accumulates across many training+inference cycles and can
			# trigger OOM on a later folder even if individual usage looks fine.
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()            # release reserved-but-free CUDA blocks
				torch.cuda.reset_peak_memory_stats() # reset peak tracker for next folder
			logger.debug("GPU and Python memory cleared between folders")

			_write_marker('running', {'last_completed_folder': folder})

		logger.info(f"\n{'='*60}")
		logger.info(f"All {total} folder(s) complete!")
		logger.info(f"{'='*60}")
		_write_marker('complete')

	except Exception:
		# logger.exception() includes full traceback automatically (exc_info=True)
		logger.exception(
			f"Pipeline failed on folder: {status.get('current_folder', 'unknown')}"
		)
		_write_marker('error')
		raise

	finally:
		# Always stop the monitor — even on clean exit, signal, or exception
		monitor.stop()


if __name__ == '__main__':
	main()