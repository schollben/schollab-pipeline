"""
Unit tests for pipeline job helpers (skip_caiman flag).

Run from repo root: python3 -m unittest test.test_pipeline_job
"""
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'caiman'))

from pipeline_job import (  # noqa: E402
	apply_skip_caiman,
	folders_missing_registered_h5,
	resolve_skip_caiman,
	run_timing_path,
	write_run_timing,
)


class TestApplySkipCaiman(unittest.TestCase):
	def test_leaves_selections_when_false(self):
		import numpy as np
		orig = np.array([[True, False], [True, True], [False, False], [False, True]])
		out = apply_skip_caiman(orig, False)
		self.assertTrue(np.array_equal(orig, out))

	def test_zeros_all_when_true(self):
		import numpy as np
		orig = np.array([[True, False], [True, True], [False, False], [False, True]])
		out = apply_skip_caiman(orig, True)
		self.assertFalse(out.any())


class TestResolveSkipCaiman(unittest.TestCase):
	def test_cli_overrides_false_job(self):
		self.assertTrue(resolve_skip_caiman(False, cli_skip_caiman=True))

	def test_job_true_without_cli(self):
		self.assertTrue(resolve_skip_caiman(True, cli_skip_caiman=False))


class TestFoldersMissingRegisteredH5(unittest.TestCase):
	def test_detects_missing(self):
		with tempfile.TemporaryDirectory() as tmp:
			has_h5 = os.path.join(tmp, 'session_a')
			no_h5 = os.path.join(tmp, 'session_b')
			os.makedirs(has_h5)
			os.makedirs(no_h5)
			open(os.path.join(has_h5, 'registered.h5'), 'w').close()
			missing = folders_missing_registered_h5([has_h5, no_h5])
			self.assertEqual(missing, [no_h5])


class TestRunTiming(unittest.TestCase):
	def test_timing_path_includes_run_id(self):
		self.assertEqual(run_timing_path('20260618180354'), '/tmp/pipeline_run_20260618180354.timing.json')

	def test_write_run_timing_creates_json(self):
		with tempfile.TemporaryDirectory() as tmp:
			run_id = 'testrun'
			path = os.path.join(tmp, f'pipeline_run_{run_id}.timing.json')
			orig = run_timing_path
			try:
				import pipeline_job as pj
				pj.run_timing_path = lambda rid: path
				write_run_timing(
					run_id,
					wall_start_epoch=1_700_000_000.0,
					run_t0=0.0,
					state='starting',
					n_sessions=1,
				)
				self.assertTrue(os.path.isfile(path))
				with open(path) as f:
					data = __import__('json').load(f)
				self.assertEqual(data['run_id'], run_id)
				self.assertEqual(data['state'], 'starting')
			finally:
				pj.run_timing_path = orig


if __name__ == '__main__':
	unittest.main()
