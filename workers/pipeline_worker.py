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
"""

import os
import sys
import json
import subprocess
import numpy as np

# Repo root is parent of workers/ — worker moved from repo root so we need two dirnames.
REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAIMAN_DIR  = os.path.join(REPO_DIR, 'caiman')
FAST_DIR    = os.path.join(REPO_DIR, 'fast')
sys.path.insert(0, CAIMAN_DIR)

from registration import register_bulk

# Direct python path for FAST env — no conda activation needed.
# Must match the miniforge3 env on the server.
FAST_PYTHON     = os.path.join(os.path.expanduser('~'), 'miniforge3', 'envs', 'FAST', 'bin', 'python')
FAST_SCRIPT     = os.path.join(FAST_DIR, 'denoising.py')
FAST_BASE_CFG   = os.path.join(FAST_DIR, 'pipeline_config.json')


def load_fast_base_config():
	"""Load the base FAST config once — per-folder runs override only data_folders."""
	if not os.path.exists(FAST_BASE_CFG):
		raise FileNotFoundError(
			f"FAST base config not found: {FAST_BASE_CFG}\n"
			f"  Check that fast/pipeline_config.json exists in the repo."
		)
	with open(FAST_BASE_CFG) as f:
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

	with open(cfg_path, "w") as f:
		json.dump(single_cfg, f, indent=2)

	print(f"  [FAST] Running denoising on: {folder}")
	result = subprocess.run(
		[FAST_PYTHON, FAST_SCRIPT, "--config", cfg_path],
		# Inherit stdout/stderr so output appears in journalctl
		check=False
	)
	if result.returncode != 0:
		print(f"  [FAST] WARNING: denoising.py exited with code {result.returncode} for {folder}")
	else:
		print(f"  [FAST] Done: {folder}")


def main():
	if len(sys.argv) < 2:
		print("Usage: python workers/pipeline_worker.py <job_json_path>")
		sys.exit(1)

	job_path = sys.argv[1]
	if not os.path.exists(job_path):
		print(f"ERROR: Job file not found: {job_path}")
		sys.exit(1)

	with open(job_path) as f:
		job = json.load(f)

	sessions         = job["sessions"]
	proc_selections  = np.array(job["process_selections"])   # shape: 4×N

	print(f"pipeline_worker: {len(sessions)} folder(s) to process")
	for s in sessions:
		print(f"  {s}")
	print()

	base_cfg = load_fast_base_config()

	for i, folder in enumerate(sessions):
		print(f"\n{'='*60}")
		print(f"Folder {i+1}/{len(sessions)}: {folder}")
		print(f"{'='*60}")

		# Step 1: caiman motion correction for this folder only.
		# Passes a 4×1 slice so register_bulk processes exactly one session.
		print(f"  [caiman] Starting motion correction...")
		try:
			register_bulk([folder], proc_selections[:, i:i+1])
			print(f"  [caiman] Done: {folder}")
		except Exception as e:
			print(f"  [caiman] ERROR on {folder}: {e}")
			print(f"  Skipping FAST for this folder and continuing.")
			continue

		# FAST consumes registered.h5 from CaImAn. TIFs→H5 alone only makes unregistered.h5.
		registered_h5 = os.path.join(folder, 'registered.h5')
		if not os.path.isfile(registered_h5):
			print(
				"  [FAST] Skipping — registered.h5 not found.\n"
				"  If you used TIFs→.H5, also check at least one motion step "
				"(First Rigid, Addl. Rigid, or NoRMCorre) so CaImAn writes registered.h5."
			)
			continue

		# Step 2: FAST denoising — reads registered.h5 written by caiman
		run_fast_on_folder(folder, base_cfg, i)

	print(f"\n{'='*60}")
	print("pipeline_worker: all folders complete.")
	print(f"{'='*60}")


if __name__ == '__main__':
	main()
