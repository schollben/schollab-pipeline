"""
Pipeline job helpers — shared by GUI, registration.py, and pipeline_worker.

Why separate module: skip_caiman logic is testable without wx or systemd.
"""
import json
import os
import time
from datetime import datetime, timezone


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
