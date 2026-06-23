#!/usr/bin/env python3
"""
pipeline_dispatcher.py
----------------------
Entry point for scheduled batch timers. Checks the global batch lock,
defers if another batch is running, then runs pipeline_worker sequentially.
"""
import json
import os
import subprocess
import sys
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAIMAN_DIR = os.path.join(REPO_DIR, 'caiman')
sys.path.insert(0, CAIMAN_DIR)

from pipeline_job import (  # noqa: E402
	is_batch_running,
	release_batch_lock,
	try_acquire_batch_lock,
	validate_job,
	ACTIVE_UNIT_PATH,
)
from pipeline_launcher import (  # noqa: E402
	CAIMAN_PYTHON,
	DEFER_MINUTES,
	WORKER_SCRIPT,
	defer_scheduled_job,
)


def _log_batch_header(job_path, batch_log_path):
	with open(batch_log_path, 'a', encoding='utf-8') as f:
		f.write(f"\n{'=' * 60}\n")
		f.write(f"Batch dispatcher start {datetime.now().isoformat(timespec='seconds')}\n")
		f.write(f"Job: {job_path}\n")
		with open(job_path, encoding='utf-8') as jf:
			job = json.load(jf)
		f.write(f"batch_id: {job.get('batch_id', '?')}\n")
		f.write(f"scheduled_at: {job.get('scheduled_at', '(immediate)')}\n")
		f.write(f"folders ({len(job.get('sessions', []))}):\n")
		for folder in job.get('sessions', []):
			f.write(f"  {folder}\n")
		f.write(f"{'=' * 60}\n")


def _log_batch_footer(batch_log_path, exit_code):
	with open(batch_log_path, 'a', encoding='utf-8') as f:
		f.write(f"\n{'=' * 60}\n")
		f.write(f"Batch dispatcher end {datetime.now().isoformat(timespec='seconds')}\n")
		f.write(f"exit_code: {exit_code}\n")
		f.write(f"{'=' * 60}\n")


def main():
	if len(sys.argv) < 2:
		print("Usage: pipeline_dispatcher.py /path/to/batch_job.json")
		sys.exit(1)

	job_path = sys.argv[1]
	if not os.path.isfile(job_path):
		print(f"ERROR: Job file not found: {job_path}")
		sys.exit(1)

	with open(job_path, encoding='utf-8') as f:
		job = json.load(f)
	validate_job(job)

	if is_batch_running():
		defer_scheduled_job(job_path, defer_minutes=DEFER_MINUTES)
		sys.exit(0)

	batch_id = job.get('batch_id', 'unknown')
	unit_name = job.get('unit_name', f'scheduled-{batch_id}')
	verbose_log = (
		job.get('verbose_log_path') or job.get('batch_log_path')
		or os.environ.get('SCHOLLAB_BATCH_LOG')
	)

	if not try_acquire_batch_lock(batch_id, unit_name, os.getpid()):
		defer_scheduled_job(job_path, defer_minutes=DEFER_MINUTES)
		sys.exit(0)

	exit_code = 1
	try:
		with open(ACTIVE_UNIT_PATH, 'w', encoding='utf-8') as f:
			f.write(unit_name)

		env = os.environ.copy()
		if verbose_log:
			os.makedirs(os.path.dirname(verbose_log), exist_ok=True)
			_log_batch_header(job_path, verbose_log)
			env['SCHOLLAB_BATCH_LOG'] = verbose_log
		if job.get('fast_log_path'):
			env['SCHOLLAB_FAST_LOG'] = job['fast_log_path']

		print(f"pipeline_dispatcher: starting batch {batch_id}")
		print(f"  job: {job_path}")
		if job.get('run_log_path'):
			print(f"  summary log: {job['run_log_path']}")
		if verbose_log:
			print(f"  verbose log: {verbose_log}")

		result = subprocess.run(
			[CAIMAN_PYTHON, WORKER_SCRIPT, job_path],
			env=env,
			check=False,
		)
		exit_code = result.returncode
		if verbose_log:
			_log_batch_footer(verbose_log, exit_code)
	finally:
		release_batch_lock()

	sys.exit(exit_code)


if __name__ == '__main__':
	main()
