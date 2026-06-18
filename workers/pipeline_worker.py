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

# Repo root is parent of workers/ — worker moved from repo root so we need two dirnames.
REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAIMAN_DIR  = os.path.join(REPO_DIR, 'caiman')
FAST_DIR    = os.path.join(REPO_DIR, 'fast')
sys.path.insert(0, CAIMAN_DIR)

from registration import register_bulk
from pipeline_job import apply_skip_caiman, resolve_skip_caiman
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


def _rusage_cpu_seconds(ru):
	"""User + system CPU seconds from a resource usage struct."""
	return ru.ru_utime + ru.ru_stime


def _print_run_summary(run_t0, rusage_self_t0, rusage_child_t0, n_sessions):
	"""
	Print wall and CPU time for this worker invocation only.

	Why: systemd's trailing "Consumed CPU time" reuses a fixed unit name and often
	under-counts FAST subprocess work — this summary is scoped to the current run.
	"""
	wall_s = time.perf_counter() - run_t0
	rs = resource.getrusage(resource.RUSAGE_SELF)
	rc = resource.getrusage(resource.RUSAGE_CHILDREN)
	cpu_self = _rusage_cpu_seconds(rs) - _rusage_cpu_seconds(rusage_self_t0)
	cpu_child = _rusage_cpu_seconds(rc) - _rusage_cpu_seconds(rusage_child_t0)
	cpu_total = cpu_self + cpu_child
	print("")
	print("=" * 60)
	print(f"pipeline_worker: run summary ({n_sessions} folder(s))")
	print(f"  wall time: {wall_s / 3600:.2f} h ({wall_s:.1f} s)")
	print(f"  CPU time:  {cpu_total:.1f} s (worker {cpu_self:.1f} s + FAST {cpu_child:.1f} s)")
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

	with open(cfg_path, "w") as f:
		json.dump(single_cfg, f, indent=2)

	print(f"  [FAST] Running denoising on: {folder}")
	if fast_threads:
		thread_summary = ", ".join(f"{key}={fast_env[key]}" for key in sorted(fast_threads))
		print(f"  [FAST] Thread env: {thread_summary}")
	result = subprocess.run(
		[FAST_PYTHON, FAST_SCRIPT, "--config", cfg_path],
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

	run_t0 = time.perf_counter()
	rusage_self_t0 = resource.getrusage(resource.RUSAGE_SELF)
	rusage_child_t0 = resource.getrusage(resource.RUSAGE_CHILDREN)

	print(f"pipeline_worker: {n_sessions} folder(s) to process (run_id={run_id})")
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

		print(f"\n{'='*60}")
		print("pipeline_worker: all folders complete.")
	finally:
		_print_run_summary(run_t0, rusage_self_t0, rusage_child_t0, n_sessions)


if __name__ == '__main__':
	main()
