"""
Tests for caiman/tiff_compat.py — TiffWriter API compatibility shim.

The shim bridges old tifffile (save) and modern tifffile (write).
All tests use mock writers; no file I/O required.
"""
from unittest.mock import MagicMock, call
import pytest
import numpy as np

from tiff_compat import tiff_writer_append


@pytest.fixture
def frame():
    return np.zeros((64, 64), dtype=np.uint16)


class TestModernApi:
    """Writer has a .write() method (current tifffile)."""

    def test_write_called(self, frame):
        writer = MagicMock(spec=['write'])
        tiff_writer_append(writer, frame)
        writer.write.assert_called_once_with(frame)

    def test_write_contiguous_false_no_kwarg(self, frame):
        writer = MagicMock(spec=['write'])
        tiff_writer_append(writer, frame, contiguous=False)
        writer.write.assert_called_once_with(frame)

    def test_write_contiguous_true_passes_kwarg(self, frame):
        writer = MagicMock(spec=['write'])
        tiff_writer_append(writer, frame, contiguous=True)
        writer.write.assert_called_once_with(frame, contiguous=True)

    def test_save_not_called(self, frame):
        writer = MagicMock(spec=['write'])
        tiff_writer_append(writer, frame)
        assert not hasattr(writer, 'save') or not writer.save.called


class TestLegacyApi:
    """Writer has no .write() method — falls back to .save() (old tifffile)."""

    def test_save_called(self, frame):
        writer = MagicMock(spec=['save'])
        tiff_writer_append(writer, frame)
        writer.save.assert_called_once_with(frame)

    def test_save_contiguous_false_no_kwarg(self, frame):
        writer = MagicMock(spec=['save'])
        tiff_writer_append(writer, frame, contiguous=False)
        writer.save.assert_called_once_with(frame)

    def test_save_contiguous_true_passes_kwarg(self, frame):
        writer = MagicMock(spec=['save'])
        tiff_writer_append(writer, frame, contiguous=True)
        writer.save.assert_called_once_with(frame, contiguous=True)

    def test_save_contiguous_unsupported_falls_back(self, frame):
        """If save() rejects contiguous= keyword, retry without it."""
        writer = MagicMock(spec=['save'])
        writer.save.side_effect = [TypeError('unexpected keyword'), None]
        tiff_writer_append(writer, frame, contiguous=True)
        assert writer.save.call_count == 2
        writer.save.assert_called_with(frame)
