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
	batch_log_path,
	folders_missing_registered_h5,
	make_batch_id,
	resolve_skip_caiman,
	run_timing_path,
	validate_job,
	want_unregistered_h5,
	write_run_timing,
)
from pipeline_launcher import (  # noqa: E402
	build_immediate_job,
	build_job,
	build_scheduled_job,
	glibc_tunables_for_caiman,
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


class TestWantUnregisteredH5(unittest.TestCase):
	def test_default_mc_does_not_write(self):
		# GUI default: TIFs→H5 off, First Rigid on.
		self.assertFalse(want_unregistered_h5([False, True, False, False]))

	def test_both_checked_does_not_write(self):
		self.assertFalse(want_unregistered_h5([True, True, False, False]))

	def test_requested_without_motion(self):
		self.assertTrue(want_unregistered_h5([True, False, False, False]))


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


class TestBatchJobValidation(unittest.TestCase):
	def test_validate_job_accepts_well_formed(self):
		job = {
			'sessions': ['/data/a', '/data/b'],
			'process_selections': [
				[True, True],
				[True, False],
				[False, False],
				[False, False],
			],
		}
		validate_job(job)

	def test_validate_job_rejects_row_mismatch(self):
		job = {
			'sessions': ['/data/a', '/data/b'],
			'process_selections': [[True], [True], [False], [False]],
		}
		with self.assertRaises(ValueError):
			validate_job(job)

	def test_make_batch_id_uses_scheduled_stamp(self):
		bid = make_batch_id('2026-06-19T02:00:00')
		self.assertTrue(bid.startswith('20260619-020000-'))

	def test_build_job_includes_log_paths(self):
		import numpy as np
		job = build_scheduled_job(
			np.array(['/data/a']),
			np.array([[True], [True], [False], [False]]),
			False,
			'2026-06-19T02:00:00',
		)
		self.assertIn('batch_id', job)
		self.assertIn('run_log_dir', job)
		self.assertIn('run_log_path', job)
		self.assertIn('verbose_log_path', job)
		self.assertIn('fast_log_path', job)
		self.assertIn('batch_log_path', job)
		self.assertIn('scheduled_at', job)
		self.assertTrue(job['run_log_path'].endswith('/summary.log'))
		self.assertTrue(job['verbose_log_path'].endswith('/verbose.log'))

	def test_build_immediate_job_has_log_paths(self):
		"""Immediate jobs include repo log paths."""
		import numpy as np
		job = build_immediate_job(
			np.array(['/data/a', '/data/b']),
			np.array([[True, False], [True, True], [False, False], [False, False]]),
			False,
		)
		validate_job(job)
		self.assertIn('run_id', job)
		self.assertIn('unit_name', job)
		self.assertIn('run_log_path', job)
		self.assertNotIn('batch_id', job)
		self.assertNotIn('scheduled_at', job)

	def test_batch_log_path_creates_logs_dir(self):
		with tempfile.TemporaryDirectory() as tmp:
			path = batch_log_path(tmp, 'test-batch')
			self.assertTrue(path.endswith('batch_test-batch.log'))
			self.assertTrue(os.path.isdir(os.path.join(tmp, 'logs')))


class TestGlibcTunablesForCaiman(unittest.TestCase):
	def test_default_forces_execstack(self):
		self.assertEqual(glibc_tunables_for_caiman(''), 'glibc.rtld.execstack=2')

	def test_appends_when_other_tunables_present(self):
		self.assertEqual(
			glibc_tunables_for_caiman('glibc.malloc.check=3'),
			'glibc.malloc.check=3:glibc.rtld.execstack=2',
		)

	def test_leaves_existing_execstack_alone(self):
		self.assertEqual(
			glibc_tunables_for_caiman('glibc.rtld.execstack=0'),
			'glibc.rtld.execstack=0',
		)


if __name__ == '__main__':
	unittest.main()
