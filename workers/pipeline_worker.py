#!/usr/bin/env python3
"""
pipeline_worker.py
------------------
Headless pipeline worker — reads a job JSON file and processes each folder
sequentially: caiman motion correction → FAST denoising.

Launched as a systemd user service by registration.py (via systemd-run),
so it runs outside the login session cgroup and survives display/GDM crashes.

Usage (normally invoked by registration.py, not run directly). Debug from repo root:
    python workers/pipeline_worker.py /tmp/pipeline_job.json
    python workers/pipeline_worker.py /tmp/pipeline_job.json --skip-caiman
"""

import argparse
import os
import sys
import json
import resource
import subprocess
import time
from datetime import datetime

# Repo root is parent of workers/ — worker moved from repo root so we need two dirnames.
REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAIMAN_DIR  = os.path.join(REPO_DIR, 'caiman')
FAST_DIR    = os.path.join(REPO_DIR, 'fast')
sys.path.insert(0, CAIMAN_DIR)

from registration import register_bulk
from pipeline_job import apply_skip_caiman, resolve_skip_caiman, write_run_timing
import numpy as np  # registration applies CaImAn thread caps before NumPy is loaded.


def _schollab_conda_root():
	"""
	Directory that contains bin/conda and envs/{caiman,FAST}.
	Must stay aligned with caiman/registration.py (same env var / default).
	"""
	root = os.environ.get('SCHOLLAB_CONDA_ROOT')
	if root:
		return os.path.expanduser(root)
	return os.path.join(os.path.expanduser('~'), 'miniforge3')


# Direct python path for FAST env — no conda activation needed.
FAST_PYTHON     = os.path.join(_schollab_conda_root(), 'envs', 'FAST', 'bin', 'python')
FAST_SCRIPT     = os.path.join(FAST_DIR, 'denoising.py')
FAST_BASE_CFG   = os.path.join(FAST_DIR, 'config.json')
FAST_BASE_CFG_LEGACY = os.path.join(FAST_DIR, 'pipeline_config.json')


class _BatchLogTee:
	"""Mirror stdout/stderr to consolidated batch log while keeping journal output."""

	def __init__(self, path, stream):
		os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
		self.file = open(path, 'a', encoding='utf-8')
		self.stream = stream

	def write(self, data):
		self.stream.write(data)
		self.file.write(data)
		self.file.flush()

	def flush(self):
		self.stream.flush()
		self.file.flush()

	def fileno(self):
		return self.stream.fileno()


def _attach_batch_log(batch_log_path):
	"""Tee worker prints to batch log; return originals for restore."""
	if not batch_log_path:
		return None, None, None
	orig_out, orig_err = sys.stdout, sys.stderr
	header = f"\n{'=' * 60}\nworker start {datetime.now().isoformat(timespec='seconds')}\n{'=' * 60}\n"
	with open(batch_log_path, 'a', encoding='utf-8') as f:
		f.write(header)
	sys.stdout = _BatchLogTee(batch_log_path, orig_out)
	sys.stderr = _BatchLogTee(batch_log_path, orig_err)
	return orig_out, orig_err, batch_log_path


def _detach_batch_log(orig_out, orig_err, batch_log_path, footer):
	if orig_out is None:
		return
	sys.stdout = orig_out
	sys.stderr = orig_err
	if batch_log_path:
		with open(batch_log_path, 'a', encoding='utf-8') as f:
			f.write(footer)


def _rusage_cpu_seconds(ru):
	"""User + system CPU seconds from a resource usage struct."""
	return ru.ru_utime + ru.ru_stime


def _cpu_usage_since(rusage_self_t0, rusage_child_t0):
	"""Worker + wait()'d child CPU seconds since run start (under-counts CaImAn pool)."""
	rs = resource.getrusage(resource.RUSAGE_SELF)
	rc = resource.getrusage(resource.RUSAGE_CHILDREN)
	cpu_self = _rusage_cpu_seconds(rs) - _rusage_cpu_seconds(rusage_self_t0)
	cpu_child = _rusage_cpu_seconds(rc) - _rusage_cpu_seconds(rusage_child_t0)
	return cpu_self, cpu_child


def _print_run_summary(run_t0, rusage_self_t0, rusage_child_t0, n_sessions, run_id, wall_start_epoch):
	"""
	Print wall and CPU time for this worker invocation only.

	Why: systemd's trailing "Consumed CPU time" sums all cores in the cgroup and
	often exceeds wall time (e.g. 5 h CPU for a 1.5 h run with n_processes=16).
	This summary labels wall vs CPU explicitly; timing file survives OOM kills.
	"""
	wall_s = time.perf_counter() - run_t0
	cpu_self, cpu_child = _cpu_usage_since(rusage_self_t0, rusage_child_t0)
	cpu_total = cpu_self + cpu_child
	write_run_timing(
		run_id,
		wall_start_epoch=wall_start_epoch,
		run_t0=run_t0,
		state="complete",
		n_sessions=n_sessions,
		cpu_self_s=round(cpu_self, 1),
		cpu_child_s=round(cpu_child, 1),
		exit_code=0,
	)
	print("")
	print("=" * 60)
	print(f"pipeline_worker: run summary ({n_sessions} folder(s))")
	print(f"  wall time: {wall_s / 3600:.2f} h ({wall_s:.1f} s)")
	print(f"  CPU time:  {cpu_total / 3600:.2f} h ({cpu_total:.1f} s) — worker {cpu_self:.1f} s + subprocess {cpu_child:.1f} s")
	print("  note: systemd 'Consumed CPU time' includes all CaImAn pool cores; it is not wall clock.")
	print("=" * 60)


def load_fast_base_config():
	"""Load the base FAST config once — per-folder runs override only data_folders."""
	cfg_path = FAST_BASE_CFG
	if not os.path.exists(cfg_path) and os.path.exists(FAST_BASE_CFG_LEGACY):
		cfg_path = FAST_BASE_CFG_LEGACY
		print(f"WARNING: Using legacy FAST config path: {cfg_path}")
		print("  Rename it to fast/config.json when convenient.")
	if not os.path.exists(cfg_path):
		raise FileNotFoundError(
			f"FAST base config not found: {FAST_BASE_CFG}\n"
			f"  Check that fast/config.json exists in the repo."
		)
	with open(cfg_path) as f:
		return json.load(f)


def run_fast_on_folder(folder, base_cfg, folder_idx):
	"""
	Write a single-folder FAST config and run denoising.py as a blocking subprocess.
	Blocking is intentional — ensures caiman and FAST complete for folder N
	before moving to folder N+1.
	"""
	# Override only data_folders; all other settings (epochs, scratch_dir, etc.)
	# come from the base config on the server
	single_cfg = {**base_cfg, "data_folders": [folder]}
	cfg_path   = f"/tmp/fast_cfg_{folder_idx}.json"
	fast_env   = os.environ.copy()
	fast_threads = base_cfg.get("threads", {})

	for key, value in fast_threads.items():
		# FAST runs in a separate Python env; keep its numerical thread caps independent.
		fast_env[key] = str(value)

	batch_log = os.environ.get('SCHOLLAB_BATCH_LOG')
	cmd = [FAST_PYTHON, FAST_SCRIPT, '--config', cfg_path]
	if batch_log:
		cmd.extend(['--batch-log', batch_log])

	with open(cfg_path, "w") as f:
		json.dump(single_cfg, f, indent=2)

	print(f"  [FAST] Running denoising on: {folder}")
	if fast_threads:
		thread_summary = ", ".join(f"{key}={fast_env[key]}" for key in sorted(fast_threads))
		print(f"  [FAST] Thread env: {thread_summary}")
	if batch_log:
		print(f"  [FAST] Batch log append: {batch_log}")
	result = subprocess.run(
		cmd,
		# Inherit stdout/stderr so output appears in journalctl
		env=fast_env,
		check=False
	)
	if result.returncode != 0:
		print(f"  [FAST] WARNING: denoising.py exited with code {result.returncode} for {folder}")
	else:
		print(f"  [FAST] Done: {folder}")


def _parse_args():
	parser = argparse.ArgumentParser(description='Schollab caiman+FAST pipeline worker')
	parser.add_argument('job_json_path', help='Path to pipeline job JSON (e.g. /tmp/pipeline_job.json)')
	parser.add_argument(
		'--skip-caiman',
		action='store_true',
		help='Skip CaImAn registration; run FAST only (requires registered.h5 per folder)'
	)
	return parser.parse_args()


def main():
	args = _parse_args()
	job_path = args.job_json_path
	if not os.path.exists(job_path):
		print(f"ERROR: Job file not found: {job_path}")
		sys.exit(1)

	with open(job_path) as f:
		job = json.load(f)

	sessions = job["sessions"]
	proc_selections = np.array(job["process_selections"])   # shape: 4×N
	skip_caiman = resolve_skip_caiman(job.get("skip_caiman", False), args.skip_caiman)
	proc_selections = apply_skip_caiman(proc_selections, skip_caiman)
	n_sessions = len(sessions)
	run_id = job.get("run_id", "unknown")
	batch_log_path_val = job.get("batch_log_path") or os.environ.get("SCHOLLAB_BATCH_LOG")
	orig_out, orig_err, _ = _attach_batch_log(batch_log_path_val)

	run_t0 = time.perf_counter()
	wall_start_epoch = time.time()
	rusage_self_t0 = resource.getrusage(resource.RUSAGE_SELF)
	rusage_child_t0 = resource.getrusage(resource.RUSAGE_CHILDREN)

	wall_start_local = datetime.fromtimestamp(wall_start_epoch).strftime('%Y-%m-%d %H:%M:%S')
	print(f"pipeline_worker: {n_sessions} folder(s) to process (run_id={run_id})")
	if job.get("batch_id"):
		print(f"pipeline_worker: batch_id={job['batch_id']}")
	print(f"pipeline_worker: wall clock start {wall_start_local}")
	print(
		"pipeline_worker: systemd 'Consumed CPU time' at exit is total core-seconds "
		"(often > wall time when CaImAn n_processes>1); use --status for wall duration."
	)
	write_run_timing(
		run_id,
		wall_start_epoch=wall_start_epoch,
		run_t0=run_t0,
		state="starting",
		n_sessions=n_sessions,
	)
	if batch_log_path_val:
		print(f"pipeline_worker: batch log {batch_log_path_val}")
	print(f"skip_caiman: {skip_caiman}")
	if skip_caiman:
		print("  CaImAn steps skipped — FAST only (requires registered.h5)")
	for s in sessions:
		print(f"  {s}")
	print()

	base_cfg = load_fast_base_config()

	try:
		for i, folder in enumerate(sessions):
			print(f"\n{'='*60}")
			print(f"Folder {i+1}/{n_sessions}: {folder}")
			print(f"{'='*60}")

			if not skip_caiman:
				# Step 1: caiman motion correction for this folder only.
				print(f"  [caiman] Starting motion correction...")
				try:
					register_bulk([folder], proc_selections[:, i:i+1])
					print(f"  [caiman] Done: {folder}")
					cpu_self, cpu_child = _cpu_usage_since(rusage_self_t0, rusage_child_t0)
					write_run_timing(
						run_id,
						wall_start_epoch=wall_start_epoch,
						run_t0=run_t0,
						state="caiman_done",
						n_sessions=n_sessions,
						folder_idx=i + 1,
						folder=folder,
						cpu_self_s=round(cpu_self, 1),
						cpu_child_s=round(cpu_child, 1),
					)
				except Exception as e:
					print(f"  [caiman] ERROR on {folder}: {e}")
					print(f"  Skipping FAST for this folder and continuing.")
					continue
			else:
				print("  [caiman] Skipped (skip_caiman=true)")

			# FAST consumes registered.h5 from CaImAn. TIFs→H5 alone only makes unregistered.h5.
			registered_h5 = os.path.join(folder, 'registered.h5')
			if not os.path.isfile(registered_h5):
				print(
					"  [FAST] Skipping — registered.h5 not found.\n"
					"  Run CaImAn first, or uncheck Skip CaImAn (FAST only) in the GUI."
				)
				continue

			# Step 2: FAST denoising — reads registered.h5 written by caiman
			run_fast_on_folder(folder, base_cfg, i)
			cpu_self, cpu_child = _cpu_usage_since(rusage_self_t0, rusage_child_t0)
			write_run_timing(
				run_id,
				wall_start_epoch=wall_start_epoch,
				run_t0=run_t0,
				state="fast_done",
				n_sessions=n_sessions,
				folder_idx=i + 1,
				folder=folder,
				cpu_self_s=round(cpu_self, 1),
				cpu_child_s=round(cpu_child, 1),
			)

		print(f"\n{'='*60}")
		print("pipeline_worker: all folders complete.")
	finally:
		_print_run_summary(run_t0, rusage_self_t0, rusage_child_t0, n_sessions, run_id, wall_start_epoch)
		footer = f"\n{'=' * 60}\nworker end {datetime.now().isoformat(timespec='seconds')}\n{'=' * 60}\n"
		_detach_batch_log(orig_out, orig_err, batch_log_path_val, footer)


if __name__ == '__main__':
	main()
