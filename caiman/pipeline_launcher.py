"""
Launch and schedule pipeline batch jobs via systemd user units.

Extracted from registration.py so GUI, CLI, and timer dispatcher share one path.
"""
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from pipeline_job import (
	attach_log_paths,
	make_batch_id,
	persist_job,
	resolve_fast_dir,
	scheduled_jobs_dir,
	unit_name_for_run,
	validate_job,
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_SCRIPT = os.path.join(REPO_DIR, 'workers', 'pipeline_worker.py')
DISPATCHER_SCRIPT = os.path.join(REPO_DIR, 'workers', 'pipeline_dispatcher.py')
UNIT_NAME = 'schollab-PreProcess2PImages'
BATCH_TIMER_PREFIX = 'schollab-batch'
JOB_PATH = '/tmp/pipeline_job.json'
ACTIVE_UNIT_PATH = '/tmp/pipeline_active_unit.txt'
SYSTEMD_USER_DIR = os.path.join(os.path.expanduser('~'), '.config', 'systemd', 'user')
DEFER_MINUTES = 5


def _schollab_conda_root():
	root = os.environ.get('SCHOLLAB_CONDA_ROOT')
	if root:
		return os.path.expanduser(root)
	return os.path.join(os.path.expanduser('~'), 'miniforge3')


CAIMAN_PYTHON = os.path.join(_schollab_conda_root(), 'envs', 'caiman', 'bin', 'python')


def _load_caiman_config():
	cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
	if os.path.exists(cfg_path):
		with open(cfg_path, encoding='utf-8') as f:
			return json.load(f)
	return {'threads': {}}


CAIMAN_CONFIG = _load_caiman_config()


def _caiman_thread_setenv_args():
	args = []
	for key in sorted(CAIMAN_CONFIG.get('threads', {})):
		if key in os.environ:
			args.append(f"--setenv={key}={os.environ[key]}")
	return args


def _fast_path_setenv_args():
	args = []
	for key in ('FAST_DIR',):
		if key in os.environ:
			args.append(f"--setenv={key}={os.environ[key]}")
	return args


def _systemd_run_base_env():
	cmd = [
		'systemd-run', '--user',
		'--collect',
		f"--setenv=HOME={os.path.expanduser('~')}",
		f"--setenv=SCHOLLAB_CONDA_ROOT={_schollab_conda_root()}",
	]
	cmd.extend(_caiman_thread_setenv_args())
	cmd.extend(_fast_path_setenv_args())
	if 'CAIMAN_N_PROCESSES' in os.environ:
		cmd.append(f"--setenv=CAIMAN_N_PROCESSES={os.environ['CAIMAN_N_PROCESSES']}")
	return cmd


def build_immediate_job(workdirs, proc_opts, skip_caiman):
	"""
	Build a job for immediate GUI/CLI launch — same JSON shape as before batch scheduling.
	"""
	run_id = datetime.now().strftime('%Y%m%d%H%M%S')
	job = {
		'sessions': workdirs.tolist() if hasattr(workdirs, 'tolist') else list(workdirs),
		'process_selections': proc_opts.tolist() if hasattr(proc_opts, 'tolist') else proc_opts,
		'skip_caiman': bool(skip_caiman),
		'run_id': run_id,
		'unit_name': unit_name_for_run(run_id),
	}
	attach_log_paths(job, REPO_DIR)
	validate_job(job)
	return job


def build_scheduled_job(workdirs, proc_opts, skip_caiman, scheduled_at, fast_dir=None):
	"""Build a scheduled batch job with batch_id, log path, and scheduled_at."""
	run_id = datetime.now().strftime('%Y%m%d%H%M%S')
	batch_id = make_batch_id(scheduled_at)
	job = {
		'batch_id': batch_id,
		'run_id': run_id,
		'unit_name': unit_name_for_run(run_id),
		'sessions': workdirs.tolist() if hasattr(workdirs, 'tolist') else list(workdirs),
		'process_selections': proc_opts.tolist() if hasattr(proc_opts, 'tolist') else proc_opts,
		'skip_caiman': bool(skip_caiman),
		'scheduled_at': scheduled_at,
	}
	attach_log_paths(job, REPO_DIR)
	validate_job(job)
	return job


# Backward alias for tests/tools that expect build_job for scheduled batches.
def build_job(workdirs, proc_opts, skip_caiman, scheduled_at=None, fast_dir=None):
	if scheduled_at:
		return build_scheduled_job(workdirs, proc_opts, skip_caiman, scheduled_at, fast_dir=fast_dir)
	return build_immediate_job(workdirs, proc_opts, skip_caiman)


def write_job(job, path=None):
	"""Write job JSON to path (default /tmp/pipeline_job.json)."""
	out = path or JOB_PATH
	validate_job(job)
	Path(out).write_text(json.dumps(job, indent=2), encoding='utf-8')
	return out


def persist_batch_job(job, fast_dir=None):
	"""Copy job to persistent scheduled_jobs dir (survives reboot)."""
	fast_dir = fast_dir or resolve_fast_dir()
	dest = os.path.join(scheduled_jobs_dir(fast_dir), f"{job['batch_id']}.json")
	persist_job(job, dest)
	return dest


def _sanitize_unit_suffix(batch_id):
	return re.sub(r'[^a-zA-Z0-9-]', '-', batch_id)


def _batch_unit_names(batch_id):
	suffix = _sanitize_unit_suffix(batch_id)
	service = f'{BATCH_TIMER_PREFIX}-{suffix}.service'
	timer = f'{BATCH_TIMER_PREFIX}-{suffix}.timer'
	return service, timer


def _parse_on_calendar(scheduled_at):
	dt = datetime.fromisoformat(scheduled_at)
	return dt.strftime('%Y-%m-%d %H:%M:%S')


def _write_batch_systemd_units(batch_id, job_path, on_calendar):
	os.makedirs(SYSTEMD_USER_DIR, exist_ok=True)
	service_name, timer_name = _batch_unit_names(batch_id)
	service_path = os.path.join(SYSTEMD_USER_DIR, service_name)
	timer_path = os.path.join(SYSTEMD_USER_DIR, timer_name)

	service_text = f"""[Unit]
Description=Schollab scheduled batch {batch_id}

[Service]
Type=oneshot
Environment=SCHOLLAB_BATCH_JOB_PATH={job_path}
Environment=SCHOLLAB_CONDA_ROOT={_schollab_conda_root()}
ExecStart={CAIMAN_PYTHON} {DISPATCHER_SCRIPT} {job_path}
"""
	timer_text = f"""[Unit]
Description=Timer for Schollab batch {batch_id}

[Timer]
OnCalendar={on_calendar}
Persistent=true

[Install]
WantedBy=timers.target
"""
	Path(service_path).write_text(service_text, encoding='utf-8')
	Path(timer_path).write_text(timer_text, encoding='utf-8')
	return service_name, timer_name


def _systemctl_user(*args, check=True):
	return subprocess.run(['systemctl', '--user', *args], check=check)


def _worker_log_env_args(job):
	"""Pass repo log paths into transient systemd worker."""
	args = []
	verbose = job.get('verbose_log_path') or job.get('batch_log_path')
	fast_log = job.get('fast_log_path')
	if verbose:
		args.append(f"--setenv=SCHOLLAB_BATCH_LOG={verbose}")
	if fast_log:
		args.append(f"--setenv=SCHOLLAB_FAST_LOG={fast_log}")
	return args


def launch_job_now(job_path):
	"""Launch pipeline_worker immediately as a transient systemd service."""
	with open(job_path, encoding='utf-8') as f:
		job = json.load(f)
	validate_job(job)

	run_id = job.get('run_id') or datetime.now().strftime('%Y%m%d%H%M%S')
	unit_name = job.get('unit_name') or unit_name_for_run(run_id)
	job['run_id'] = run_id
	job['unit_name'] = unit_name
	attach_log_paths(job, REPO_DIR)
	write_job(job, job_path)
	Path(ACTIVE_UNIT_PATH).write_text(unit_name, encoding='utf-8')

	# Each run uses a unique unit name — no need to reset-failed on the legacy
	# fixed name schollab-PreProcess2PImages.service (often not loaded → noisy error).

	cmd = _systemd_run_base_env()
	cmd.extend([
		f'--unit={unit_name}',
		'--description=Schollab caiman+FAST pipeline',
	])
	cmd.extend(_worker_log_env_args(job))
	cmd.extend([CAIMAN_PYTHON, WORKER_SCRIPT, job_path])
	subprocess.run(cmd, check=True)
	print(f"Pipeline launched as systemd service '{unit_name}'.")
	print(f"Monitor: journalctl --user -f -u {unit_name}")
	if job.get('run_log_dir'):
		print(f"Log dir: {job['run_log_dir']}")
	if job.get('run_log_path'):
		print(f"Summary log: {job['run_log_path']}")
	if job.get('verbose_log_path'):
		print(f"Verbose log: {job['verbose_log_path']}")
	return unit_name


def schedule_job(job_path, scheduled_at):
	"""Install a one-shot systemd user timer for a persisted batch job."""
	with open(job_path, encoding='utf-8') as f:
		job = json.load(f)
	job['scheduled_at'] = scheduled_at
	if not job.get('batch_id'):
		job['batch_id'] = make_batch_id(scheduled_at)
	attach_log_paths(job, REPO_DIR)
	validate_job(job)

	persisted = persist_batch_job(job)
	write_job(job, persisted)
	write_job(job, JOB_PATH)

	on_calendar = _parse_on_calendar(scheduled_at)
	_, timer_name = _write_batch_systemd_units(job['batch_id'], persisted, on_calendar)
	_systemctl_user('daemon-reload')
	_systemctl_user('enable', timer_name)
	_systemctl_user('start', timer_name)

	print(f"Batch scheduled: {job['batch_id']}")
	print(f"  Start time:   {scheduled_at}")
	print(f"  Job file:     {persisted}")
	print(f"  Timer unit:   {timer_name}")
	print(f"  Log dir:      {job['run_log_dir']}")
	print(f"  Summary log:  {job['run_log_path']}")
	print(f"  Verbose log:  {job['verbose_log_path']}")
	print(f"  List timers:  systemctl --user list-timers '{BATCH_TIMER_PREFIX}-*'")
	return job['batch_id'], timer_name


def defer_scheduled_job(job_path, defer_minutes=DEFER_MINUTES):
	"""Re-arm timer when another batch is still running."""
	new_time = datetime.now() + timedelta(minutes=defer_minutes)
	new_at = new_time.replace(microsecond=0).isoformat(timespec='seconds')
	print(
		f"Batch deferred {defer_minutes} min — another batch is running. "
		f"Next try: {new_at}"
	)
	return schedule_job(job_path, new_at)


def cancel_scheduled_batch(batch_id):
	"""Stop and remove timer/service units for a scheduled batch."""
	service_name, timer_name = _batch_unit_names(batch_id)
	_systemctl_user('stop', timer_name, check=False)
	_systemctl_user('disable', timer_name, check=False)
	for name in (timer_name, service_name):
		path = os.path.join(SYSTEMD_USER_DIR, name)
		if os.path.exists(path):
			os.remove(path)
	_systemctl_user('daemon-reload', check=False)
	print(f"Cancelled scheduled batch: {batch_id}")


def list_scheduled_batches(fast_dir=None):
	fast_dir = fast_dir or resolve_fast_dir()
	jobs_dir = scheduled_jobs_dir(fast_dir)
	results = []
	if os.path.isdir(jobs_dir):
		for name in sorted(os.listdir(jobs_dir)):
			if not name.endswith('.json'):
				continue
			path = os.path.join(jobs_dir, name)
			try:
				with open(path, encoding='utf-8') as f:
					job = json.load(f)
				job['_job_path'] = path
				results.append(job)
			except (json.JSONDecodeError, OSError):
				continue
	return results


def print_scheduled_batches():
	jobs = list_scheduled_batches()
	if not jobs:
		print("No scheduled batch jobs found.")
		return
	print("── Scheduled batches ──────────────────────────────")
	for job in jobs:
		print(f"  batch_id:     {job.get('batch_id', '?')}")
		print(f"  scheduled_at: {job.get('scheduled_at', '(immediate)')}")
		print(f"  folders:      {len(job.get('sessions', []))}")
		print(f"  job file:     {job.get('_job_path', '')}")
		print(f"  log dir:      {job.get('run_log_dir', '')}")
		print(f"  summary log:  {job.get('run_log_path', '')}")
		print(f"  verbose log:  {job.get('verbose_log_path', job.get('batch_log_path', ''))}")
		_, timer_name = _batch_unit_names(job.get('batch_id', ''))
		active = subprocess.run(
			['systemctl', '--user', 'is-active', timer_name],
			capture_output=True, text=True, check=False,
		)
		print(f"  timer:        {timer_name} ({active.stdout.strip() or 'unknown'})")
		print("")
