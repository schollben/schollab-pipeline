"""
Unit tests for combined pipeline run log helpers.

Run from repo root: python3 -m unittest test.test_pipeline_run_log
"""
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'caiman'))

from pipeline_run_log import (  # noqa: E402
	append_folder_block,
	format_run_id_ts,
	log_paths_for_run,
	overall_outcome_line,
)
from pipeline_job import attach_log_paths  # noqa: E402
from pipeline_launcher import build_immediate_job, build_scheduled_job  # noqa: E402


class TestLogPaths(unittest.TestCase):
	def test_format_run_id_ts(self):
		self.assertEqual(format_run_id_ts('20260528143012'), '20260528-143012')

	def test_log_paths_for_run(self):
		with tempfile.TemporaryDirectory() as tmp:
			paths = log_paths_for_run(tmp, '20260528143012')
			self.assertTrue(paths['run_log_path'].endswith('log/20260528-143012.log'))
			self.assertTrue(paths['verbose_log_path'].endswith('.verbose.log'))
			self.assertTrue(paths['fast_log_path'].endswith('.fast.log'))
			self.assertTrue(os.path.isdir(os.path.join(tmp, 'log')))

	def test_attach_log_paths_sets_batch_alias(self):
		job = {'run_id': '20260528143012', 'sessions': ['/a'], 'process_selections': [[True]] * 4}
		with tempfile.TemporaryDirectory() as tmp:
			attach_log_paths(job, tmp)
			self.assertEqual(job['batch_log_path'], job['verbose_log_path'])


class TestOutcomeLines(unittest.TestCase):
	def test_full_success(self):
		line = overall_outcome_line(
			{'result': 'succeeded'},
			{'result': 'succeeded'},
		)
		self.assertIn('fully complete', line)

	def test_caiman_failed(self):
		line = overall_outcome_line(
			{'result': 'failed'},
			{'result': 'not_run'},
		)
		self.assertIn('CaImAn failed', line)
		self.assertIn('FAST was not run', line)

	def test_fast_failed(self):
		line = overall_outcome_line(
			{'result': 'succeeded'},
			{'result': 'failed'},
		)
		self.assertIn('FAST failed', line)


class TestAppendFolderBlock(unittest.TestCase):
	def test_writes_block(self):
		with tempfile.TemporaryDirectory() as tmp:
			path = os.path.join(tmp, 'log', '20260528-143012.log')
			append_folder_block(
				path,
				folder_idx=1,
				n_folders=2,
				folder='/data/session',
				wall_s=120.5,
				caiman_summary={
					'selected': ['TIFs→H5', 'First Rigid'],
					'result': 'succeeded',
					'steps': [
						{'name': 'TIFs→H5', 'status': 'ok', 'duration_s': 10.0, 'artifacts_line': 'unregistered.h5'},
					],
				},
				fast_summary={'result': 'succeeded', 'steps': []},
			)
			with open(path, encoding='utf-8') as f:
				text = f.read()
			self.assertIn('Folder 1/2', text)
			self.assertIn('OVERALL:', text)
			self.assertIn('/data/session', text)


class TestJobBuilders(unittest.TestCase):
	def test_immediate_job_has_log_paths(self):
		import numpy as np
		job = build_immediate_job(
			np.array(['/data/a']),
			np.array([[True], [True], [False], [False]]),
			False,
		)
		self.assertIn('run_log_path', job)
		self.assertIn('verbose_log_path', job)
		self.assertIn('fast_log_path', job)

	def test_scheduled_job_has_repo_log_paths(self):
		import numpy as np
		with tempfile.TemporaryDirectory() as tmp:
			repo = os.path.join(tmp, 'repo')
			os.makedirs(repo)
			# build_scheduled_job uses REPO_DIR from launcher — patch via attach on result
			job = build_scheduled_job(
				np.array(['/data/a']),
				np.array([[True], [True], [False], [False]]),
				False,
				'2026-06-19T02:00:00',
			)
			attach_log_paths(job, repo)
			self.assertTrue(job['run_log_path'].startswith(os.path.join(repo, 'log')))
			self.assertNotIn('Documents', job['run_log_path'])


if __name__ == '__main__':
	unittest.main()
