"""
Unit tests for FAST FolderPaths (per-session scratch/).

Run from repo root: python3 -m unittest test.test_folder_paths
"""
import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fast'))

from folder_paths import FolderPaths  # noqa: E402


class TestFolderPaths(unittest.TestCase):
	def test_scratch_colocated_with_session(self):
		root = '/mnt/bigdata/BRUKER/TSeries-001'
		paths = FolderPaths.from_root(root)
		self.assertEqual(paths.root, root)
		self.assertEqual(paths.scratch, os.path.join(root, 'scratch'))
		self.assertEqual(paths.registered, os.path.join(root, 'scratch', 'registered'))
		self.assertEqual(paths.checkpoint, os.path.join(root, 'checkpoint'))


if __name__ == '__main__':
	unittest.main()
