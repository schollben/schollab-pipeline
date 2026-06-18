"""
Unit tests for TIFF chunk stack ordering (100k+ frame datasets).

Run from repo root: python3 -m unittest test.test_tif_stack_sort
"""
import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'caiman'))

from tiff_compat import (  # noqa: E402
	chunk_index_from_basename,
	chunk_index_pad_width,
	format_chunk_tif_name,
	sort_tif_stack_paths,
)


class TestChunkPadWidth(unittest.TestCase):
	def test_scales_with_chunk_count(self):
		self.assertEqual(chunk_index_pad_width(20), 2)
		self.assertEqual(chunk_index_pad_width(99), 2)
		self.assertEqual(chunk_index_pad_width(100), 3)
		self.assertEqual(chunk_index_pad_width(1000), 4)


class TestFormatChunkTifName(unittest.TestCase):
	def test_100_chunks_use_three_digits(self):
		self.assertEqual(format_chunk_tif_name('registered', 100, 100), 'registered_100.tif')
		self.assertEqual(format_chunk_tif_name('registered', 11, 100), 'registered_011.tif')


class TestChunkIndexFromBasename(unittest.TestCase):
	def test_parses_export_and_inference_names(self):
		self.assertEqual(chunk_index_from_basename('registered_100.tif'), 100)
		self.assertEqual(chunk_index_from_basename('202506181430_registered_011.tif'), 11)

	def test_ignores_scanimage_channel_suffix(self):
		self.assertIsNone(chunk_index_from_basename('file_00001_ch2.tif'))


class TestSortTifStackPaths(unittest.TestCase):
	def test_lex_broken_order_fixed_for_100_stacks(self):
		paths = [f'/tmp/registered_{i:02d}.tif' for i in (9, 10, 100, 11, 12)]
		ordered = sort_tif_stack_paths(paths)
		indices = [chunk_index_from_basename(os.path.basename(p)) for p in ordered]
		self.assertEqual(indices, [9, 10, 11, 12, 100])

	def test_padded_export_names_sort_correctly(self):
		paths = [
			'/scratch/registered_100.tif',
			'/scratch/registered_010.tif',
			'/scratch/registered_011.tif',
		]
		ordered = sort_tif_stack_paths(paths)
		indices = [chunk_index_from_basename(os.path.basename(p)) for p in ordered]
		self.assertEqual(indices, [10, 11, 100])


if __name__ == '__main__':
	unittest.main()
