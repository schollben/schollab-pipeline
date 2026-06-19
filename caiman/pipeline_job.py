"""
Pipeline job helpers — shared by GUI, registration.py, and pipeline_worker.

Why separate module: skip_caiman logic is testable without wx or systemd.
"""
import os

import numpy as np


def apply_skip_caiman(process_selections, skip_caiman):
	"""
	When skip_caiman is set, force all CaImAn step flags off regardless of GUI state.

	Returns a new array/list with the same shape as process_selections.
	"""
	if not skip_caiman:
		return process_selections
	return np.zeros_like(process_selections, dtype=bool)


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
