"""
Pipeline job helpers — shared by GUI, registration.py, and pipeline_worker.

Why separate module: skip_caiman logic is testable without wx or systemd.
"""
import json
import os
import secrets
import time
from datetime import datetime, timezone

BATCH_LOCK_PATH = '/tmp/pipeline_batch.lock'
JOB_PATH = '/tmp/pipeline_job.json'
ACTIVE_UNIT_PATH = '/tmp/pipeline_active_unit.txt'
UNIT_NAME = 'schollab-PreProcess2PImages'
FAST_CONFIG_REL = os.path.join('fast', 'config.json')
LEGACY_FAST_CONFIG_REL = os.path.join('fast', 'pipeline_config.json')


def run_timing_path(run_id):
	"""Per-run timing file — survives SIGKILL so --status can show wall duration."""
	return f"/tmp/pipeline_run_{run_id}.timing.json"


def write_run_timing(
	run_id,
	*,
	wall_start_epoch,
	run_t0,
	state,
	n_sessions,
	folder_idx=None,
	folder=None,
	cpu_self_s=None,
	cpu_child_s=None,
	exit_code=None,
):
	"""
	Persist wall-clock progress for this run.

	Why: systemd's trailing "Consumed CPU time" is cumulative core-seconds across
	all worker processes — it often exceeds wall time when n_processes>1. OOM kills
	skip Python finally blocks, so we rewrite this file at each major step.
	"""
	wall_elapsed_s = time.perf_counter() - run_t0
	payload = {
		"run_id": run_id,
		"wall_start_iso": datetime.fromtimestamp(wall_start_epoch, tz=timezone.utc).astimezone().isoformat(timespec="seconds"),
		"wall_elapsed_s": round(wall_elapsed_s, 1),
		"state": state,
		"n_sessions": n_sessions,
		"folder_idx": folder_idx,
		"folder": folder,
		"cpu_self_s": cpu_self_s,
		"cpu_child_s": cpu_child_s,
		"exit_code": exit_code,
		"updated_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
	}
	path = run_timing_path(run_id)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)
	return path


def make_batch_id(scheduled_at=None):
	"""Stable batch ID for timer units, logs, and scheduled_jobs filename."""
	stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
	if scheduled_at:
		try:
			dt = datetime.fromisoformat(scheduled_at)
			stamp = dt.strftime('%Y%m%d-%H%M%S')
		except ValueError:
			pass
	return f"{stamp}-{secrets.token_hex(2)}"


def unit_name_for_run(run_id):
	return f'{UNIT_NAME}-{run_id}'


def _repo_root():
	return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_fast_dir():
	"""Resolve FAST install dir from env or fast/config.json."""
	if os.environ.get('FAST_DIR'):
		return os.path.abspath(os.path.expanduser(os.path.expandvars(os.environ['FAST_DIR'])))
	repo = _repo_root()
	for rel in (FAST_CONFIG_REL, LEGACY_FAST_CONFIG_REL):
		cfg_path = os.path.join(repo, rel)
		if os.path.exists(cfg_path):
			with open(cfg_path, encoding='utf-8') as f:
				raw = json.load(f).get('fast_dir', '~/Documents/FAST')
			return os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))
	return os.path.abspath(os.path.expanduser('~/Documents/FAST'))


def scheduled_jobs_dir(fast_dir=None):
	path = os.path.join(fast_dir or resolve_fast_dir(), 'scheduled_jobs')
	os.makedirs(path, exist_ok=True)
	return path


def batch_log_path(fast_dir, batch_id):
	logs = os.path.join(fast_dir or resolve_fast_dir(), 'logs')
	os.makedirs(logs, exist_ok=True)
	return os.path.join(logs, f'batch_{batch_id}.log')


def validate_job(job):
	"""Raise ValueError if job JSON is missing keys or has inconsistent shape."""
	required = ('sessions', 'process_selections')
	for key in required:
		if key not in job:
			raise ValueError(f"Job missing required key: {key}")
	sessions = job['sessions']
	selections = job['process_selections']
	if not sessions:
		raise ValueError('Job must include at least one session folder')
	if len(selections) != 4:
		raise ValueError('process_selections must have 4 rows (TIFs→H5, rigid×2, nonrigid)')
	n = len(sessions)
	for row_idx, row in enumerate(selections):
		if len(row) != n:
			raise ValueError(
				f'process_selections row {row_idx} length {len(row)} != sessions {n}'
			)


def persist_job(job, dest_path):
	validate_job(job)
	os.makedirs(os.path.dirname(dest_path), exist_ok=True)
	with open(dest_path, 'w', encoding='utf-8') as f:
		json.dump(job, f, indent=2)
	return dest_path


def read_batch_lock():
	if not os.path.isfile(BATCH_LOCK_PATH):
		return None
	try:
		with open(BATCH_LOCK_PATH, encoding='utf-8') as f:
			return json.load(f)
	except (json.JSONDecodeError, OSError):
		return None


def _pid_alive(pid):
	try:
		os.kill(int(pid), 0)
		return True
	except (OSError, ValueError, TypeError):
		return False


def is_batch_running():
	"""True if lock file points to an active worker unit or live PID."""
	lock = read_batch_lock()
	if not lock:
		return False
	unit = lock.get('unit_name')
	if unit:
		import subprocess
		result = subprocess.run(
			['systemctl', '--user', 'is-active', unit],
			capture_output=True, text=True, check=False,
		)
		if result.stdout.strip() == 'active':
			return True
	pid = lock.get('pid')
	if pid and _pid_alive(pid):
		return True
	return False


def write_batch_lock(batch_id, unit_name, pid):
	payload = {
		'batch_id': batch_id,
		'unit_name': unit_name,
		'pid': pid,
		'started_at': datetime.now().astimezone().isoformat(timespec='seconds'),
	}
	with open(BATCH_LOCK_PATH, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2)


def release_batch_lock():
	if os.path.isfile(BATCH_LOCK_PATH):
		os.remove(BATCH_LOCK_PATH)


def try_acquire_batch_lock(batch_id, unit_name, pid):
	if is_batch_running():
		return False
	write_batch_lock(batch_id, unit_name, pid)
	return True


def apply_skip_caiman(process_selections, skip_caiman):
	"""
	When skip_caiman is set, force all CaImAn step flags off regardless of GUI state.

	Returns a new array/list with the same shape as process_selections.
	"""
	if not skip_caiman:
		return process_selections
	try:
		import numpy as np
		return np.zeros_like(process_selections, dtype=bool)
	except ImportError:
		# Fallback if numpy unavailable in a minimal test env.
		return [[False] * len(row) for row in process_selections]


def folders_missing_registered_h5(session_paths):
	"""Return session folders that lack registered.h5 (FAST input)."""
	missing = []
	for folder in session_paths:
		path = os.path.join(folder.rstrip('/'), 'registered.h5')
		if not os.path.isfile(path):
			missing.append(folder)
	return missing


def resolve_skip_caiman(job_skip_caiman, cli_skip_caiman=False):
	"""CLI --skip-caiman overrides job JSON when explicitly passed."""
	return bool(cli_skip_caiman or job_skip_caiman)
