"""
Motion correction reads acquisition TIFFs; unregistered.h5 is opt-in only.

Run from repo root: python3 -m unittest test.test_skip_unregistered
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'caiman'))

# tif_to_h5 imports these at module load; this suite only tests path listing.
for _mod in ('tifffile', 'h5py', 'tqdm'):
	sys.modules.setdefault(_mod, mock.MagicMock())

from tif_to_h5 import (  # noqa: E402
	list_acquisition_tiffs,
	motion_correction_inputs,
	require_matching_frame_count,
	count_tiff_frames,
	stage_plain_tiffs,
)


def _touch(folder, name):
	path = os.path.join(folder, name)
	open(path, 'w').close()
	return path


class TestListAcquisitionTiffs(unittest.TestCase):
	def test_keeps_ch2_drops_ch1_and_artifacts(self):
		with tempfile.TemporaryDirectory() as tmp:
			ch2 = _touch(tmp, 'TSeries_001_Ch2_stack.tif')
			_touch(tmp, 'TSeries_001_Ch1_stack.tif')
			_touch(tmp, '01_rigid.tif')
			_touch(tmp, 'std_projection.tif')
			got = list_acquisition_tiffs(tmp)
			self.assertEqual(got, [ch2])

	def test_empty_folder_returns_empty(self):
		with tempfile.TemporaryDirectory() as tmp:
			self.assertEqual(list_acquisition_tiffs(tmp), [])

	def test_scanimage_without_channel_token_keeps_all(self):
		with tempfile.TemporaryDirectory() as tmp:
			a = _touch(tmp, 'file_00001_00001.tif')
			b = _touch(tmp, 'file_00001_00002.tif')
			got = list_acquisition_tiffs(tmp)
			self.assertEqual(got, [a, b])


class TestMotionCorrectionInputs(unittest.TestCase):
	def test_prefers_tiffs_over_existing_unregistered(self):
		with tempfile.TemporaryDirectory() as tmp:
			tif = _touch(tmp, 'TSeries_001_Ch2_stack.tif')
			_touch(tmp, 'unregistered.h5')
			self.assertEqual(motion_correction_inputs(tmp), [tif])

	def test_falls_back_to_unregistered_when_no_tiffs(self):
		with tempfile.TemporaryDirectory() as tmp:
			unreg = _touch(tmp, 'unregistered.h5')
			self.assertEqual(motion_correction_inputs(tmp), [unreg])

	def test_falls_back_to_registered_when_only_that_exists(self):
		with tempfile.TemporaryDirectory() as tmp:
			reg = _touch(tmp, 'registered.h5')
			self.assertEqual(motion_correction_inputs(tmp), [reg])

	def test_raises_when_folder_has_nothing(self):
		with tempfile.TemporaryDirectory() as tmp:
			with self.assertRaises(FileNotFoundError):
				motion_correction_inputs(tmp)


class TestRequireMatchingFrameCount(unittest.TestCase):
	def test_allows_unknown_expected(self):
		require_matching_frame_count(5000, None)

	def test_allows_equal_counts(self):
		require_matching_frame_count(21973, 21973)

	def test_rejects_short_mmap(self):
		with self.assertRaises(RuntimeError) as ctx:
			require_matching_frame_count(5000, 21973)
		self.assertIn('5000', str(ctx.exception))
		self.assertIn('21973', str(ctx.exception))


class TestCountTiffFrames(unittest.TestCase):
	def test_sums_pages_without_ome_join(self):
		import tif_to_h5 as mod

		class FakeTif:
			def __init__(self, n_pages):
				self.pages = [None] * n_pages

			def __enter__(self):
				return self

			def __exit__(self, *args):
				return False

		with mock.patch.object(
			mod, '_tiff_file_no_ome',
			side_effect=[FakeTif(1000), FakeTif(973)],
		):
			self.assertEqual(count_tiff_frames(['a.ome.tif', 'b.ome.tif']), 1973)


class TestStagePlainTiffs(unittest.TestCase):
	def test_writes_numbered_non_ome_names(self):
		import tif_to_h5 as mod
		with tempfile.TemporaryDirectory() as tmp:
			src = os.path.join(tmp, 'TSeries_Ch2_000001.ome.tif')
			open(src, 'w').close()
			dest_dir = os.path.join(tmp, 'stage')
			fake = np.zeros((2, 4, 4), dtype=np.uint16)
			with mock.patch.object(mod, '_imread_no_ome', return_value=fake), \
					mock.patch.object(mod.tifffile, 'imwrite') as imwrite:
				out = stage_plain_tiffs([src], dest_dir)
			self.assertEqual([os.path.basename(p) for p in out], ['mc_0000.tif'])
			imwrite.assert_called_once()
			self.assertEqual(imwrite.call_args[0][0], out[0])


class TestRegisterFolderSkipsUnregistered(unittest.TestCase):
	"""_register_one_folder must not call tif_stacks_to_h5 when MC is selected."""

	@classmethod
	def setUpClass(cls):
		mods = {
			'cv2': mock.MagicMock(),
			'h5py': mock.MagicMock(),
			'tifffile': mock.MagicMock(),
			'caiman': mock.MagicMock(),
			'caiman.motion_correction': mock.MagicMock(),
			'caiman.source_extraction': mock.MagicMock(),
			'caiman.source_extraction.cnmf': mock.MagicMock(),
			'caiman.source_extraction.cnmf.cnmf': mock.MagicMock(),
			'caiman.source_extraction.cnmf.params': mock.MagicMock(),
			'caiman.utils': mock.MagicMock(),
			'caiman.utils.utils': mock.MagicMock(),
			'caiman.summary_images': mock.MagicMock(),
			'registration_gui': mock.MagicMock(),
		}
		cls._patcher = mock.patch.dict(sys.modules, mods)
		cls._patcher.start()
		sys.modules.pop('registration', None)
		import registration as reg_mod
		cls.reg = reg_mod

	@classmethod
	def tearDownClass(cls):
		cls._patcher.stop()
		sys.modules.pop('registration', None)

	def test_mc_only_does_not_write_unregistered(self):
		reg = self.reg
		with mock.patch.object(reg, 'register_one_session') as mc, \
				mock.patch.object(reg, 'tif_stacks_to_h5') as to_h5, \
				tempfile.TemporaryDirectory() as tmp:
			_touch(tmp, 'registered.h5')
			summary = reg._register_one_folder(
				tmp, [False, True, False, False], {},
			)
			to_h5.assert_not_called()
			mc.assert_called_once()
			self.assertEqual(summary['result'], 'succeeded')

	def test_both_checked_still_skips_unregistered_write(self):
		reg = self.reg
		with mock.patch.object(reg, 'register_one_session') as mc, \
				mock.patch.object(reg, 'tif_stacks_to_h5') as to_h5, \
				tempfile.TemporaryDirectory() as tmp:
			_touch(tmp, 'registered.h5')
			summary = reg._register_one_folder(
				tmp, [True, True, False, False], {},
			)
			to_h5.assert_not_called()
			mc.assert_called_once()
			skipped = [s for s in summary['steps'] if s['name'] == 'TIFs→H5']
			self.assertEqual(skipped[0]['status'], 'skipped')

	def test_all_tiff_inputs(self):
		reg = self.reg
		self.assertTrue(reg._all_tiff_inputs(['a.tif', 'b.ome.tif']))
		self.assertFalse(reg._all_tiff_inputs(['a.h5']))
		self.assertFalse(reg._all_tiff_inputs([]))

	def test_as_path_list_wraps_string(self):
		reg = self.reg
		self.assertEqual(reg._as_path_list('/tmp/a.mmap'), ['/tmp/a.mmap'])
		self.assertEqual(reg._as_path_list(['/tmp/a.mmap', '/tmp/b.mmap']),
			['/tmp/a.mmap', '/tmp/b.mmap'])
		self.assertEqual(reg._as_path_list([]), [])

	def test_register_one_session_stages_tiffs_and_checks_count(self):
		# Why: CaImAn workers must see non-OME copies, and registered.h5
		# must be sized from the acquisition frame count, not mmap[0] only.
		reg = self.reg
		mmaps = ['/tmp/mmap0', '/tmp/mmap1']
		with tempfile.TemporaryDirectory() as tmp:
			src = _touch(tmp, 'TSeries_Ch2_000001.ome.tif')
			staged = [os.path.join(tmp, 'mc_0000.tif')]
			with mock.patch.object(reg, 'motion_correction_inputs', return_value=[src]), \
					mock.patch.object(reg, 'count_tiff_frames', return_value=21973), \
					mock.patch.object(reg, 'stage_plain_tiffs', return_value=staged) as stage, \
					mock.patch.object(
						reg, '_run_motion_correction',
						return_value=(21973, mmaps),
					), \
					mock.patch.object(reg, '_write_registered_h5') as write_h5, \
					mock.patch.object(reg, '_cleanup_memmaps'), \
					mock.patch.object(reg, '_cleanup_stage_dir') as cleanup:
				reg.register_one_session(tmp, {}, False, False, '01_rigid.tif')
			stage.assert_called_once()
			self.assertEqual(write_h5.call_args[0][1], mmaps)
			self.assertEqual(write_h5.call_args.kwargs.get('expected_frames'), 21973)
			cleanup.assert_called_once()

	def test_tifs_to_h5_only_writes_unregistered(self):
		reg = self.reg
		with mock.patch.object(reg, 'register_one_session') as mc, \
				mock.patch.object(reg, 'tif_stacks_to_h5') as to_h5, \
				mock.patch.object(reg, 'list_acquisition_tiffs', return_value=['a.tif']), \
				tempfile.TemporaryDirectory() as tmp:
			unreg = os.path.join(tmp, 'unregistered.h5')

			def write_unreg(folder, h5_name, **kwargs):
				open(h5_name, 'w').close()

			to_h5.side_effect = write_unreg
			summary = reg._register_one_folder(
				tmp, [True, False, False, False], {},
			)
			to_h5.assert_called_once()
			mc.assert_not_called()
			self.assertTrue(os.path.isfile(unreg))
			self.assertEqual(summary['result'], 'incomplete')


if __name__ == '__main__':
	unittest.main()
