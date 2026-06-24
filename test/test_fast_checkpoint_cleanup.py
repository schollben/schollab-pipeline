"""
Unit tests for scratch/checkpoint cleanup on FAST re-run.
"""
import os
import shutil
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fast'))

from folder_paths import (  # noqa: E402
	FolderPaths,
	remove_checkpoint_for_rerun,
	remove_scratch_for_rerun,
)


class TestRemoveCheckpointForRerun(unittest.TestCase):
	def test_removes_checkpoint_when_not_skip_training(self):
		root = tempfile.mkdtemp()
		try:
			paths = FolderPaths.from_root(root)
			os.makedirs(paths.checkpoint)
			open(os.path.join(paths.checkpoint, 'stub'), 'w').close()
			removed = remove_checkpoint_for_rerun(paths.checkpoint, skip_training=False)
			self.assertTrue(removed)
			self.assertFalse(os.path.isdir(paths.checkpoint))
		finally:
			shutil.rmtree(root, ignore_errors=True)

	def test_keeps_checkpoint_when_skip_training(self):
		root = tempfile.mkdtemp()
		try:
			paths = FolderPaths.from_root(root)
			os.makedirs(paths.checkpoint)
			removed = remove_checkpoint_for_rerun(paths.checkpoint, skip_training=True)
			self.assertFalse(removed)
			self.assertTrue(os.path.isdir(paths.checkpoint))
		finally:
			shutil.rmtree(root, ignore_errors=True)

	def test_no_op_when_checkpoint_missing(self):
		root = tempfile.mkdtemp()
		try:
			paths = FolderPaths.from_root(root)
			removed = remove_checkpoint_for_rerun(paths.checkpoint, skip_training=False)
			self.assertFalse(removed)
		finally:
			shutil.rmtree(root, ignore_errors=True)


class TestRemoveScratchForRerun(unittest.TestCase):
	def test_removes_scratch_when_present(self):
		root = tempfile.mkdtemp()
		try:
			paths = FolderPaths.from_root(root)
			os.makedirs(paths.registered)
			open(os.path.join(paths.registered, 'stub.tif'), 'w').close()
			removed = remove_scratch_for_rerun(paths.scratch)
			self.assertTrue(removed)
			self.assertFalse(os.path.isdir(paths.scratch))
		finally:
			shutil.rmtree(root, ignore_errors=True)

	def test_no_op_when_scratch_missing(self):
		root = tempfile.mkdtemp()
		try:
			paths = FolderPaths.from_root(root)
			removed = remove_scratch_for_rerun(paths.scratch)
			self.assertFalse(removed)
		finally:
			shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
	unittest.main()
