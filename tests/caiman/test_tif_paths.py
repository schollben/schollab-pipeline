"""
Tests for caiman/tif_to_h5.py path-selection helpers.

These functions contain the only real logic in the TIF→H5 conversion
path that can be exercised without touching the filesystem.
"""
import re
import pytest

from tif_to_h5 import _scanimage_tiffs_one_cycle, _acquisition_tif_paths

# ---------------------------------------------------------------------------
# _scanimage_tiffs_one_cycle
# ---------------------------------------------------------------------------

CYCLE_RE_FILE   = re.compile(r'^file_(\d+)_')
CYCLE_RE_TSERIES = re.compile(r'^TSeries_(\d+)_')


class TestScanImageTiffsOneCycle:
    def test_no_regex_match_returns_sorted_paths(self):
        paths = ['/d/gamma.tif', '/d/alpha.tif', '/d/beta.tif']
        result = _scanimage_tiffs_one_cycle(paths, CYCLE_RE_FILE)
        assert result == sorted(paths)

    def test_single_cycle_returns_sorted_members(self):
        paths = [
            '/d/file_00001_Ch2_000003.tif',
            '/d/file_00001_Ch2_000001.tif',
            '/d/file_00001_Ch2_000002.tif',
        ]
        result = _scanimage_tiffs_one_cycle(paths, CYCLE_RE_FILE)
        assert result == sorted(paths)

    def test_picks_cycle_with_most_files(self):
        # cycle 00001 has 3 files, cycle 00002 has 2 — should pick 00001
        paths = [
            '/d/file_00001_Ch2_000001.tif',
            '/d/file_00001_Ch2_000002.tif',
            '/d/file_00001_Ch2_000003.tif',
            '/d/file_00002_Ch2_000001.tif',
            '/d/file_00002_Ch2_000002.tif',
        ]
        result = _scanimage_tiffs_one_cycle(paths, CYCLE_RE_FILE)
        assert all('file_00001' in p for p in result)
        assert len(result) == 3

    def test_tie_broken_by_largest_key(self):
        # Both cycles have 2 files — lexically largest key (00003) wins
        paths = [
            '/d/file_00001_Ch2_000001.tif',
            '/d/file_00001_Ch2_000002.tif',
            '/d/file_00003_Ch2_000001.tif',
            '/d/file_00003_Ch2_000002.tif',
        ]
        result = _scanimage_tiffs_one_cycle(paths, CYCLE_RE_FILE)
        assert all('file_00003' in p for p in result)

    def test_result_is_sorted(self):
        paths = [
            '/d/file_00001_Ch2_000003.tif',
            '/d/file_00001_Ch2_000001.tif',
        ]
        result = _scanimage_tiffs_one_cycle(paths, CYCLE_RE_FILE)
        assert result == sorted(result)

    def test_empty_paths_returns_empty(self):
        assert _scanimage_tiffs_one_cycle([], CYCLE_RE_FILE) == []

    def test_tseries_regex(self):
        paths = [
            '/d/TSeries_001_frame001.tif',
            '/d/TSeries_001_frame002.tif',
            '/d/TSeries_002_frame001.tif',
        ]
        result = _scanimage_tiffs_one_cycle(paths, CYCLE_RE_TSERIES)
        assert all('TSeries_001' in p for p in result)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _acquisition_tif_paths
# ---------------------------------------------------------------------------

class TestAcquisitionTifPaths:
    def test_excludes_rigid_sample(self):
        paths = ['/d/01_rigid.tif', '/d/TSeries_001_frame001.tif']
        result = _acquisition_tif_paths(paths)
        assert '/d/01_rigid.tif' not in result

    def test_excludes_nonrigid_sample(self):
        paths = ['/d/01_nonrigid.tif', '/d/TSeries_001_frame001.tif']
        result = _acquisition_tif_paths(paths)
        assert '/d/01_nonrigid.tif' not in result

    def test_raises_when_all_filtered(self):
        paths = ['/d/01_rigid.tif', '/d/02_nonrigid.tif']
        with pytest.raises(AssertionError):
            _acquisition_tif_paths(paths)

    def test_prefers_tseries_over_file_prefix(self):
        paths = [
            '/d/file_00001_Ch2_000001.tif',
            '/d/TSeries_001_frame001.tif',
            '/d/TSeries_001_frame002.tif',
        ]
        result = _acquisition_tif_paths(paths)
        assert all('TSeries' in p for p in result)
        assert not any('file_' in p for p in result)

    def test_falls_back_to_file_prefix(self):
        paths = [
            '/d/file_00001_Ch2_000001.tif',
            '/d/file_00001_Ch2_000002.tif',
        ]
        result = _acquisition_tif_paths(paths)
        assert all('file_00001' in p for p in result)

    def test_returns_other_tifs_unchanged(self):
        # Paths with neither TSeries_ nor file_ prefix, no _rigid/_nonrigid
        paths = ['/d/recording_001.tif', '/d/recording_002.tif']
        result = _acquisition_tif_paths(paths)
        assert result == paths

    def test_mixed_rigid_and_acquisition(self):
        paths = [
            '/d/01_rigid.tif',
            '/d/02_nonrigid.tif',
            '/d/file_00001_Ch2_000001.tif',
            '/d/file_00001_Ch2_000002.tif',
        ]
        result = _acquisition_tif_paths(paths)
        assert len(result) == 2
        assert all('file_00001' in p for p in result)

    def test_single_acquisition_tif(self):
        paths = ['/d/01_rigid.tif', '/d/recording.tif']
        result = _acquisition_tif_paths(paths)
        assert result == ['/d/recording.tif']
