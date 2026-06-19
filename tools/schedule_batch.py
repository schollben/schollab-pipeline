#!/usr/bin/env python3
"""
schedule_batch.py — CLI for scheduled pipeline batches.

Examples:
  python tools/schedule_batch.py --job /tmp/pipeline_job.json --at "2026-06-19T02:00:00"
  python tools/schedule_batch.py --list
  python tools/schedule_batch.py --cancel 20260619-020000-a1b2
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'caiman'))

from pipeline_launcher import (  # noqa: E402
	JOB_PATH,
	cancel_scheduled_batch,
	launch_job_now,
	print_scheduled_batches,
	schedule_job,
	write_job,
)
from pipeline_job import validate_job  # noqa: E402


def _parse_args():
	parser = argparse.ArgumentParser(description='Schedule schollab pipeline batches')
	parser.add_argument('--job', help='Path to batch job JSON')
	parser.add_argument('--at', help='ISO local datetime to start (e.g. 2026-06-19T02:00:00)')
	parser.add_argument('--now', action='store_true', help='Run job immediately instead of scheduling')
	parser.add_argument('--list', action='store_true', help='List scheduled batches')
	parser.add_argument('--cancel', metavar='BATCH_ID', help='Cancel a scheduled batch by batch_id')
	return parser.parse_args()


def main():
	args = _parse_args()

	if args.list:
		print_scheduled_batches()
		return

	if args.cancel:
		cancel_scheduled_batch(args.cancel)
		return

	job_path = args.job or JOB_PATH
	if not os.path.isfile(job_path):
		print(f"ERROR: Job file not found: {job_path}")
		sys.exit(1)

	with open(job_path, encoding='utf-8') as f:
		job = json.load(f)
	validate_job(job)
	write_job(job, job_path)

	if args.now or not args.at:
		if args.at:
			print("WARNING: --at ignored with --now")
		launch_job_now(job_path)
		return

	if not args.at:
		print("ERROR: Provide --at for scheduling, or --now to run immediately.")
		sys.exit(1)

	schedule_job(job_path, args.at)


if __name__ == '__main__':
	main()
