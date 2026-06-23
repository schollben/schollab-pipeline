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
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime

# Repo root is parent of workers/ — worker moved from repo root so we need two dirnames.
REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAIMAN_DIR  = os.path.join(REPO_DIR, 'caiman')
FAST_DIR    = os.path.join(REPO_DIR, 'fast')
sys.path.insert(0, CAIMAN_DIR)

from registration import register_bulk
from pipeline_job import apply_skip_caiman, attach_log_paths, resolve_skip_caiman, write_run_timing
from pipeline_run_log import (
	append_folder_block,
	classify_folder_outcome,
	write_run_footer,
	write_run_header,
)
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
	"""Mirror stdout/stderr to consolidated verbose log while keeping journal output."""

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


def _attach_verbose_log(verbose_log_path):
	"""Tee worker prints to verbose log; return originals for restore."""
	if not verbose_log_path:
		return None, None, None
	orig_out, orig_err = sys.stdout, sys.stderr
	header = f"\n{'=' * 60}\nworker start {datetime.now().isoformat(timespec='seconds')}\n{'=' * 60}\n"
	with open(verbose_log_path, 'a', encoding='utf-8') as f:
		f.write(header)
	sys.stdout = _BatchLogTee(verbose_log_path, orig_out)
	sys.stderr = _BatchLogTee(verbose_log_path, orig_err)
	return orig_out, orig_err, verbose_log_path


def _detach_verbose_log(orig_out, orig_err, verbose_log_path, footer):
	if orig_out is None:
		return
	sys.stdout = orig_out
	sys.stderr = orig_err
	if verbose_log_path:
		with open(verbose_log_path, 'a', encoding='utf-8') as f:
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
	Print wall and CPU time at end of run (not at startup — keeps journal skimmable).

	Why: systemd's trailing "Consumed CPU time" sums all cores in the cgroup and
	often exceeds wall time (e.g. 5 h CPU for a 1.5 h run with n_processes=16).
	"""
	wall_s = time.perf_counter() - run_t0
	cpu_self, cpu_child = _cpu_usage_since(rusage_self_t0, rusage_child_t0)
	cpu_total = cpu_self + cpu_child
	wall_start_local = datetime.fromtimestamp(wall_start_epoch).strftime('%Y-%m-%d %H:%M:%S')
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
	print(f"pipeline_worker: run complete ({n_sessions} folder(s), run_id={run_id})")
	print(f"  started:    {wall_start_local}")
	print(f"  wall time:  {wall_s / 3600:.2f} h ({wall_s:.1f} s)")
	print(
		f"  CPU time:   {cpu_total / 3600:.2f} h ({cpu_total:.1f} s)  "
		f"(worker {cpu_self:.1f} s + subprocess {cpu_child:.1f} s)"
	)
	print(f"  timing:     bash PreProcess2PImages.sh --status  (or /tmp/pipeline_run_{run_id}.timing.json)")
	print(
		"  note:       if systemd prints 'Consumed … CPU time' after this, that value is "
		"total core-seconds (often > wall time when CaImAn n_processes>1) — use wall time above."
	)
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


def _read_fast_summary(summary_path):
	"""Load FAST per-folder summary JSON written by denoising.py."""
	if not summary_path or not os.path.isfile(summary_path):
		return None
	with open(summary_path, encoding='utf-8') as f:
		return json.load(f)


def run_fast_on_folder(folder, base_cfg, folder_idx, run_id, log_paths):
	"""
	Run denoising.py for one folder; return FAST summary dict for run log.
	"""
	single_cfg = {**base_cfg, "data_folders": [folder]}
	cfg_path   = f"/tmp/fast_cfg_{folder_idx}.json"
	summary_path = f"/tmp/fast_summary_{run_id}_{folder_idx}.json"
	fast_env   = os.environ.copy()
	fast_threads = base_cfg.get("threads", {})

	for key, value in fast_threads.items():
		fast_env[key] = str(value)

	verbose_log = log_paths.get('verbose_log_path') or os.environ.get('SCHOLLAB_BATCH_LOG')
	fast_log = log_paths.get('fast_log_path') or os.environ.get('SCHOLLAB_FAST_LOG')
	cmd = [FAST_PYTHON, FAST_SCRIPT, '--config', cfg_path, '--summary-out', summary_path]
	if verbose_log:
		cmd.extend(['--batch-log', verbose_log])
		fast_env['SCHOLLAB_BATCH_LOG'] = verbose_log
	if fast_log:
		cmd.extend(['--log-file', fast_log])
		fast_env['SCHOLLAB_FAST_LOG'] = fast_log

	with open(cfg_path, "w") as f:
		json.dump(single_cfg, f, indent=2)

	print(f"  [FAST] Running denoising on: {folder}")
	if fast_threads:
		thread_summary = ", ".join(f"{key}={fast_env[key]}" for key in sorted(fast_threads))
		print(f"  [FAST] Thread env: {thread_summary}")
	if fast_log:
		print(f"  [FAST] Detail log: {fast_log}")

	result = subprocess.run(cmd, env=fast_env, check=False)
	summary = _read_fast_summary(summary_path)
	if summary is None:
		summary = {
			'result': 'failed' if result.returncode != 0 else 'succeeded',
			'steps': [],
			'error': f'denoising.py exited with code {result.returncode}',
		}
	elif result.returncode != 0 and summary.get('result') == 'succeeded':
		summary['result'] = 'failed'
		summary['error'] = f'denoising.py exited with code {result.returncode}'

	if result.returncode != 0:
		print(f"  [FAST] WARNING: denoising.py exited with code {result.returncode} for {folder}")
	else:
		print(f"  [FAST] Done: {folder}")
	return summary


def _parse_args():
	parser = argparse.ArgumentParser(description='Schollab caiman+FAST pipeline worker')
	parser.add_argument('job_json_path', help='Path to pipeline job JSON (e.g. /tmp/pipeline_job.json)')
	parser.add_argument(
		'--skip-caiman',
		action='store_true',
		help='Skip CaImAn registration; run FAST only (requires registered.h5 per folder)'
	)
	return parser.parse_args()


def _ensure_log_paths(job):
	"""Fill repo log/ paths on job if missing (legacy JSON or hand-written jobs)."""
	if job.get('run_log_path') and job.get('verbose_log_path') and job.get('fast_log_path'):
		return job
	return attach_log_paths(job, REPO_DIR)


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
	job = _ensure_log_paths(job)
	run_log_path = job.get("run_log_path")
	verbose_log_path = job.get("verbose_log_path") or job.get("batch_log_path")
	log_paths = {
		'run_log_path': run_log_path,
		'verbose_log_path': verbose_log_path,
		'fast_log_path': job.get('fast_log_path'),
	}

	orig_out, orig_err, _ = _attach_verbose_log(verbose_log_path)

	run_t0 = time.perf_counter()
	wall_start_epoch = time.time()
	rusage_self_t0 = resource.getrusage(resource.RUSAGE_SELF)
	rusage_child_t0 = resource.getrusage(resource.RUSAGE_CHILDREN)
	folder_outcomes = []

	if run_log_path:
		write_run_header(
			run_log_path,
			run_id=run_id,
			unit_name=job.get('unit_name', '?'),
			sessions=sessions,
			skip_caiman=skip_caiman,
			batch_id=job.get('batch_id'),
		)

	print(f"pipeline_worker: {n_sessions} folder(s) to process (run_id={run_id})")
	if job.get("batch_id"):
		print(f"pipeline_worker: batch_id={job['batch_id']}")
	write_run_timing(
		run_id,
		wall_start_epoch=wall_start_epoch,
		run_t0=run_t0,
		state="starting",
		n_sessions=n_sessions,
	)
	if run_log_path:
		print(f"pipeline_worker: summary log {run_log_path}")
	if verbose_log_path:
		print(f"pipeline_worker: verbose log {verbose_log_path}")
	print(f"skip_caiman: {skip_caiman}")
	if skip_caiman:
		print("  CaImAn steps skipped — FAST only (requires registered.h5)")
	for s in sessions:
		print(f"  {s}")
	print()

	base_cfg = load_fast_base_config()

	try:
		for i, folder in enumerate(sessions):
			folder_t0 = time.perf_counter()
			caiman_summary = None
			fast_summary = None

			print(f"\n{'='*60}")
			print(f"Folder {i+1}/{n_sessions}: {folder}")
			print(f"{'='*60}")

			try:
				if not skip_caiman:
					print(f"  [caiman] Starting motion correction...")
					caiman_summaries = register_bulk([folder], proc_selections[:, i:i + 1])
					caiman_summary = caiman_summaries[0]
					if caiman_summary.get('result') == 'failed':
						print(f"  [caiman] ERROR on {folder}: {caiman_summary.get('error')}")
						print(f"  Skipping FAST for this folder and continuing.")
					else:
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
				else:
					print("  [caiman] Skipped (skip_caiman=true)")
					caiman_summary = {'result': 'skipped', 'steps': [], 'selected': []}

				registered_h5 = os.path.join(folder, 'registered.h5')
				if caiman_summary and caiman_summary.get('result') == 'failed':
					fast_summary = {
						'result': 'not_run',
						'reason': 'CaImAn failed',
						'steps': [],
					}
				elif not os.path.isfile(registered_h5):
					print(
						"  [FAST] Skipping — registered.h5 not found.\n"
						"  Run CaImAn first, or uncheck Skip CaImAn (FAST only) in the GUI."
					)
					fast_summary = {
						'result': 'not_run',
						'reason': 'registered.h5 missing',
						'steps': [],
					}
				else:
					fast_summary = run_fast_on_folder(
						folder, base_cfg, i, run_id, log_paths,
					)
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
			finally:
				if run_log_path:
					wall_s = time.perf_counter() - folder_t0
					append_folder_block(
						run_log_path,
						folder_idx=i + 1,
						n_folders=n_sessions,
						folder=folder,
						wall_s=wall_s,
						caiman_summary=caiman_summary,
						fast_summary=fast_summary,
						skip_caiman=skip_caiman,
					)
					folder_outcomes.append(
						classify_folder_outcome(caiman_summary, fast_summary, skip_caiman)
					)

		print(f"\n{'='*60}")
		print("pipeline_worker: all folders complete.")
	finally:
		wall_s = time.perf_counter() - run_t0
		if run_log_path:
			write_run_footer(
				run_log_path,
				run_log_path=run_log_path,
				wall_s=wall_s,
				folder_outcomes=folder_outcomes,
			)
		_print_run_summary(run_t0, rusage_self_t0, rusage_child_t0, n_sessions, run_id, wall_start_epoch)
		footer = f"\n{'=' * 60}\nworker end {datetime.now().isoformat(timespec='seconds')}\n{'=' * 60}\n"
		_detach_verbose_log(orig_out, orig_err, verbose_log_path, footer)


if __name__ == '__main__':
	main()
