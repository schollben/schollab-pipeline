"""
Tests for caiman/regen_projections.py (STD/AVG from an existing registered.h5).

Run from repo root: python3 -m unittest test.test_regen_projections
"""
import os
import sys
import tempfile
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'caiman'))

# Real h5py/tifffile are needed to round-trip files; skip on dev machines
# without them instead of breaking the whole suite.
try:
	import h5py  # noqa: E402
	import tifffile  # noqa: E402
except ImportError as exc:
	raise unittest.SkipTest(f"h5py/tifffile not installed: {exc}")

from regen_projections import main, regen_folder  # noqa: E402
from std_projection import (  # noqa: E402
	AVG_TIFF_NAME,
	STD_TIFF_NAME,
	STD_WINDOW,
	projection_frames,
)


def _write_h5(folder, mov):
	with h5py.File(os.path.join(folder, 'registered.h5'), 'w') as f:
		f.create_dataset('mov', data=mov)


class TestProjectionFrames(unittest.TestCase):

	def test_skips_zero_window_and_leftover(self):
		rng = np.random.default_rng(0)
		mov = rng.normal(size=(3 * STD_WINDOW + 7, 4, 4)).astype(np.float32)
		mov[STD_WINDOW:2 * STD_WINDOW] = 0
		std, avg = projection_frames(mov, mov.shape[0])
		self.assertEqual(len(std), 2)
		self.assertEqual(len(avg), 2)
		np.testing.assert_allclose(avg[1], mov[2 * STD_WINDOW:3 * STD_WINDOW].mean(0), rtol=1e-5)


class TestRegenFolder(unittest.TestCase):

	def setUp(self):
		self._tmp = tempfile.TemporaryDirectory()
		self.dir = self._tmp.name

	def tearDown(self):
		self._tmp.cleanup()

	def test_missing_h5_fails(self):
		self.assertFalse(regen_folder(self.dir))
		self.assertEqual(main([self.dir]), 1)

	def test_writes_both_tiffs_matching_h5(self):
		rng = np.random.default_rng(1)
		mov = rng.normal(10, 2, size=(2 * STD_WINDOW + 5, 6, 6)).astype(np.float32)
		_write_h5(self.dir, mov)
		self.assertTrue(regen_folder(self.dir))
		std = tifffile.imread(os.path.join(self.dir, STD_TIFF_NAME))
		avg = tifffile.imread(os.path.join(self.dir, AVG_TIFF_NAME))
		self.assertEqual(std.shape, (2, 6, 6))
		np.testing.assert_allclose(std[0], mov[:STD_WINDOW].std(0), rtol=1e-4)
		np.testing.assert_allclose(avg[1], mov[STD_WINDOW:2 * STD_WINDOW].mean(0), rtol=1e-5)

	def test_existing_tiffs_kept_unless_force(self):
		_write_h5(self.dir, np.ones((STD_WINDOW, 2, 2), dtype=np.float32))
		for name in (STD_TIFF_NAME, AVG_TIFF_NAME):
			open(os.path.join(self.dir, name), 'w').close()
		self.assertTrue(regen_folder(self.dir))
		self.assertEqual(os.path.getsize(os.path.join(self.dir, STD_TIFF_NAME)), 0)
		self.assertTrue(regen_folder(self.dir, force=True))
		self.assertGreater(os.path.getsize(os.path.join(self.dir, STD_TIFF_NAME)), 0)

	def test_short_movie_reports_failure(self):
		_write_h5(self.dir, np.ones((STD_WINDOW - 1, 2, 2), dtype=np.float32))
		self.assertFalse(regen_folder(self.dir))


if __name__ == '__main__':
	unittest.main()
