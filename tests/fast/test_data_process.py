"""
Tests for fast/datasets/data_process.py — pure tensor operations.

Skipped entirely when torch is not genuinely installed (only the mock
from conftest.py is present), since we need real tensor arithmetic.
"""
import sys
import pytest
from unittest.mock import MagicMock

# Skip the whole module if torch is mocked or absent
_torch_mod = sys.modules.get('torch')
if _torch_mod is None or isinstance(_torch_mod, MagicMock):
    pytest.skip('torch not installed', allow_module_level=True)

import torch
from data_process import space_to_depth, generate_subimages, sampler


class TestSpaceToDepth:
    def test_output_shape(self):
        x = torch.zeros(1, 1, 8, 8)
        out = space_to_depth(x, block_size=2)
        # channels *= block_size^2; h,w /= block_size
        assert out.shape == (1, 4, 4, 4)

    def test_block_size_1_preserves_shape(self):
        x = torch.zeros(2, 3, 16, 16)
        out = space_to_depth(x, block_size=1)
        assert out.shape == (2, 3, 16, 16)

    def test_batch_dimension_preserved(self):
        x = torch.zeros(5, 1, 4, 4)
        out = space_to_depth(x, block_size=2)
        assert out.shape[0] == 5


class TestSampler:
    """Neighbor2Neighbor sampler — masks should be deterministic for a fixed seed."""

    def _make_img(self, n=1, c=1, h=8, w=8):
        return torch.zeros(n, c, h, w)

    def test_returns_two_mask_lists(self):
        img = self._make_img()
        m1, m2 = sampler(img, operation_seed_counter=0)
        assert isinstance(m1, list) and isinstance(m2, list)

    def test_mask_count_equals_channels(self):
        img = self._make_img(c=3)
        m1, m2 = sampler(img, operation_seed_counter=0)
        assert len(m1) == 3
        assert len(m2) == 3

    def test_same_seed_gives_same_masks(self):
        img = self._make_img()
        m1a, m2a = sampler(img, operation_seed_counter=7)
        m1b, m2b = sampler(img, operation_seed_counter=7)
        assert torch.equal(m1a[0], m1b[0])
        assert torch.equal(m2a[0], m2b[0])

    def test_different_seeds_give_different_masks(self):
        img = self._make_img()
        m1a, _ = sampler(img, operation_seed_counter=0)
        m1b, _ = sampler(img, operation_seed_counter=99)
        # With overwhelming probability two random masks differ
        assert not torch.equal(m1a[0], m1b[0])

    def test_masks_are_boolean(self):
        img = self._make_img()
        m1, m2 = sampler(img, operation_seed_counter=0)
        assert m1[0].dtype == torch.bool
        assert m2[0].dtype == torch.bool

    def test_masks_are_disjoint(self):
        """mask1 and mask2 should not both be True at the same index."""
        img = self._make_img()
        m1, m2 = sampler(img, operation_seed_counter=0)
        overlap = m1[0] & m2[0]
        assert not overlap.any(), 'Neighbor2Neighbor masks must not overlap'


class TestGenerateSubimages:
    def test_output_shape_is_half(self):
        n, c, h, w = 1, 1, 8, 8
        img = torch.ones(n, c, h, w)
        masks, _ = sampler(img, operation_seed_counter=0)
        out = generate_subimages(img, masks)
        assert out.shape == (n, c, h // 2, w // 2)

    def test_batch_size_preserved(self):
        img = torch.ones(3, 1, 8, 8)
        masks, _ = sampler(img, operation_seed_counter=0)
        out = generate_subimages(img, masks)
        assert out.shape[0] == 3

    def test_dtype_preserved(self):
        img = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
        masks, _ = sampler(img, operation_seed_counter=0)
        out = generate_subimages(img, masks)
        assert out.dtype == torch.float32
