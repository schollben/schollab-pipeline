"""
Unit tests for motion-corrected STD projection (1000-frame windows).

Run from repo root: python3 -m unittest test.test_std_projection
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'caiman'))

from std_projection import (  # noqa: E402
	STD_TIFF_NAME,
	STD_WINDOW,
	is_pipeline_artifact_tif,
	std_of_window,
	std_projection_stack,
	write_std_projection_tiff,
)


class TestArtifactTifFilter(unittest.TestCase):
	def test_excludes_std_and_sample_previews(self):
		self.assertTrue(is_pipeline_artifact_tif('std_projection.tif'))
		self.assertTrue(is_pipeline_artifact_tif('/data/session/std_projection.tif'))
		self.assertTrue(is_pipeline_artifact_tif('01_rigid.tif'))
		self.assertTrue(is_pipeline_artifact_tif('02_nonrigid.tif'))

	def test_keeps_acquisition_names(self):
		self.assertFalse(is_pipeline_artifact_tif('TSeries_001_Ch2.tif'))
		self.assertFalse(is_pipeline_artifact_tif('file_00001_ch2.tif'))
		self.assertFalse(is_pipeline_artifact_tif(STD_TIFF_NAME.replace('.tif', '_src.tif')))


class TestStdProjectionStack(unittest.TestCase):
	def test_window_constant_is_1000(self):
		self.assertEqual(STD_WINDOW, 1000)
		self.assertEqual(STD_TIFF_NAME, 'std_projection.tif')

	def test_complete_windows_only_drop_remainder(self):
		# 2500 frames → two STD pages; last 500 frames are not a full window.
		movie = np.zeros((2500, 4, 4), dtype=np.float32)
		movie[:1000] = 1.0
		movie[1000:2000] = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1)
		stack = std_projection_stack(movie, window=1000)
		self.assertEqual(stack.shape, (2, 4, 4))
		np.testing.assert_allclose(stack[0], 0.0, atol=1e-6)
		expected = np.std(np.arange(1000, dtype=np.float64))
		np.testing.assert_allclose(stack[1], expected, atol=1e-5)

	def test_exactly_one_window(self):
		movie = np.ones((1000, 3, 3), dtype=np.float32)
		stack = std_projection_stack(movie, window=1000)
		self.assertEqual(stack.shape, (1, 3, 3))
		np.testing.assert_allclose(stack[0], 0.0, atol=1e-6)

	def test_fewer_than_window_yields_empty(self):
		movie = np.ones((999, 2, 2), dtype=np.float32)
		stack = std_projection_stack(movie, window=1000)
		self.assertEqual(stack.shape, (0, 2, 2))

	def test_std_of_window_matches_numpy(self):
		chunk = np.array([[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]])
		got = std_of_window(chunk)
		want = np.std(chunk.astype(np.float64), axis=0).astype(np.float32)
		np.testing.assert_allclose(got, want)

	def test_write_skipped_when_no_frames(self):
		# Do not create an empty TIFF when motion-corrected length < 1000.
		self.assertIsNone(write_std_projection_tiff('/tmp', []))
		self.assertIsNone(write_std_projection_tiff('/tmp', np.empty((0, 2, 2))))

	def test_write_emits_plain_stack_without_ome(self):
		# One file, one page per window, no OME-XML (Fiji would open that twice).
		frames = np.stack([
			np.ones((2, 2), dtype=np.float32),
			np.zeros((2, 2), dtype=np.float32),
		])
		fake_tifffile = mock.MagicMock()
		with mock.patch.dict(sys.modules, {'tifffile': fake_tifffile}):
			with tempfile.TemporaryDirectory() as tmp:
				path = write_std_projection_tiff(tmp, frames)
				self.assertEqual(path, os.path.join(tmp, STD_TIFF_NAME))
		fake_tifffile.imwrite.assert_called_once()
		args, kwargs = fake_tifffile.imwrite.call_args
		self.assertEqual(args[0], path)
		np.testing.assert_array_equal(args[1], frames)
		self.assertEqual(kwargs.get('ome'), False)
		self.assertIsNone(kwargs.get('metadata'))


if __name__ == '__main__':
	unittest.main()
