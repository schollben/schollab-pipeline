"""
Tests for pure config/path logic in fast/denoising.py.

Covers load_pipeline_config, resolve_config_path, PipelineConfig.from_dict,
FolderPaths.from_root, and _find_latest_checkpoint_config.
No GPU, no file I/O beyond temp files.
"""
import json
import os
import pytest
from unittest.mock import patch

import denoising
from denoising import (
    load_pipeline_config,
    resolve_config_path,
    PipelineConfig,
    FolderPaths,
    _find_latest_checkpoint_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CFG = {
    'fast_dir':             '/data/fast',
    'scratch_dir':          '/tmp/scratch',
    'skip_training':        False,
    'train_frames':         100,
    'tiff_chunk_size':      500,
    'h5_write_batch_frames': 200,
    'minibatch_size':       4,
    'batch_size':           8,
    'num_workers':          2,
    'epochs':               30,
    'data_folders':         ['/data/session1'],
}


# ---------------------------------------------------------------------------
# load_pipeline_config
# ---------------------------------------------------------------------------

class TestLoadPipelineConfig:
    def test_valid_config_returned(self, tmp_path):
        cfg_file = tmp_path / 'config.json'
        cfg_file.write_text(json.dumps(VALID_CFG))
        result = load_pipeline_config(str(cfg_file))
        assert result['epochs'] == 30
        assert result['data_folders'] == ['/data/session1']

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config(str(tmp_path / 'nonexistent.json'))

    def test_missing_required_key_raises(self, tmp_path):
        bad_cfg = {k: v for k, v in VALID_CFG.items() if k != 'epochs'}
        cfg_file = tmp_path / 'config.json'
        cfg_file.write_text(json.dumps(bad_cfg))
        with pytest.raises(KeyError, match='epochs'):
            load_pipeline_config(str(cfg_file))

    def test_multiple_missing_keys_listed(self, tmp_path):
        bad_cfg = {'fast_dir': '/d', 'scratch_dir': '/s'}
        cfg_file = tmp_path / 'config.json'
        cfg_file.write_text(json.dumps(bad_cfg))
        with pytest.raises(KeyError):
            load_pipeline_config(str(cfg_file))

    def test_extra_keys_allowed(self, tmp_path):
        extra = {**VALID_CFG, 'my_extra_param': 42}
        cfg_file = tmp_path / 'config.json'
        cfg_file.write_text(json.dumps(extra))
        result = load_pipeline_config(str(cfg_file))
        assert result['my_extra_param'] == 42


# ---------------------------------------------------------------------------
# resolve_config_path
# ---------------------------------------------------------------------------

class TestResolveConfigPath:
    def test_env_var_takes_precedence(self):
        with patch.dict(os.environ, {'MY_DIR': '/from/env'}):
            result = resolve_config_path('/from/value', 'MY_DIR', '/from/default')
        assert result == os.path.abspath('/from/env')

    def test_value_used_when_no_env(self):
        env = {k: v for k, v in os.environ.items() if k != 'MY_DIR'}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_config_path('/from/value', 'MY_DIR', '/from/default')
        assert result == os.path.abspath('/from/value')

    def test_default_used_when_neither(self):
        env = {k: v for k, v in os.environ.items() if k != 'MY_DIR'}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_config_path(None, 'MY_DIR', '/from/default')
        assert result == os.path.abspath('/from/default')

    def test_tilde_expanded(self):
        env = {k: v for k, v in os.environ.items() if k != 'MY_DIR'}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_config_path('~/mydir', 'MY_DIR', '/default')
        assert '~' not in result
        assert result.startswith('/')

    def test_result_is_absolute(self):
        env = {k: v for k, v in os.environ.items() if k != 'MY_DIR'}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_config_path('relative/path', 'MY_DIR', '/default')
        assert os.path.isabs(result)


# ---------------------------------------------------------------------------
# PipelineConfig.from_dict
# ---------------------------------------------------------------------------

class TestPipelineConfigFromDict:
    def test_all_fields_set(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ('FAST_DIR', 'FAST_SCRATCH_DIR')}
        with patch.dict(os.environ, env, clear=True):
            cfg = PipelineConfig.from_dict(VALID_CFG)
        assert cfg.skip_training is False
        assert cfg.train_frames == 100
        assert cfg.epochs == 30
        assert cfg.batch_size == 8
        assert cfg.num_workers == 2

    def test_base_config_path_derived(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ('FAST_DIR', 'FAST_SCRATCH_DIR')}
        with patch.dict(os.environ, env, clear=True):
            cfg = PipelineConfig.from_dict(VALID_CFG)
        assert cfg.base_config_path == os.path.join(cfg.fast_dir, 'userparams.json')

    def test_env_var_overrides_fast_dir(self):
        with patch.dict(os.environ, {'FAST_DIR': '/override/fast'}):
            cfg = PipelineConfig.from_dict(VALID_CFG)
        assert '/override/fast' in cfg.fast_dir

    def test_env_var_overrides_scratch_dir(self):
        with patch.dict(os.environ, {'FAST_SCRATCH_DIR': '/override/scratch'}):
            cfg = PipelineConfig.from_dict(VALID_CFG)
        assert '/override/scratch' in cfg.scratch_dir


# ---------------------------------------------------------------------------
# FolderPaths.from_root
# ---------------------------------------------------------------------------

class TestFolderPathsFromRoot:
    def _make(self, root='/data/TSeries-20250101', scratch='/tmp/scratch'):
        return FolderPaths.from_root(root, scratch)

    def test_root_stored(self):
        fp = self._make()
        assert fp.root == '/data/TSeries-20250101'

    def test_scratch_uses_folder_id(self):
        fp = self._make(root='/data/MySeries', scratch='/tmp/sc')
        assert fp.scratch == '/tmp/sc/MySeries'

    def test_h5_in_root(self):
        fp = self._make(root='/data/MySeries')
        assert fp.h5 == '/data/MySeries/registered.h5'

    def test_registered_in_scratch(self):
        fp = self._make(root='/data/MySeries', scratch='/tmp/sc')
        assert fp.registered == '/tmp/sc/MySeries/registered'

    def test_training_in_scratch(self):
        fp = self._make(root='/data/MySeries', scratch='/tmp/sc')
        assert fp.training == '/tmp/sc/MySeries/training'

    def test_result_in_scratch(self):
        fp = self._make(root='/data/MySeries', scratch='/tmp/sc')
        assert fp.result == '/tmp/sc/MySeries/result'

    def test_checkpoint_in_root(self):
        fp = self._make(root='/data/MySeries')
        assert fp.checkpoint == '/data/MySeries/checkpoint'

    def test_inference_h5_in_root(self):
        fp = self._make(root='/data/MySeries')
        assert fp.inference_h5 == '/data/MySeries/inference.h5'

    def test_sentinel_in_root(self):
        fp = self._make(root='/data/MySeries')
        assert fp.sentinel == '/data/MySeries/_fast_complete'

    def test_trailing_slash_stripped_from_folder_id(self):
        fp = FolderPaths.from_root('/data/MySeries/', '/tmp/sc')
        assert fp.scratch == '/tmp/sc/MySeries'


# ---------------------------------------------------------------------------
# _find_latest_checkpoint_config
# ---------------------------------------------------------------------------

class TestFindLatestCheckpointConfig:
    def test_missing_directory_returns_none(self, tmp_path):
        result = _find_latest_checkpoint_config(str(tmp_path / 'nonexistent'))
        assert result is None

    def test_empty_directory_returns_none(self, tmp_path):
        result = _find_latest_checkpoint_config(str(tmp_path))
        assert result is None

    def test_finds_config_in_subdir(self, tmp_path):
        run = tmp_path / '202501010900'
        run.mkdir()
        cfg = run / 'config.json'
        cfg.write_text('{}')
        result = _find_latest_checkpoint_config(str(tmp_path))
        assert result == str(cfg)

    def test_picks_newest_subdir(self, tmp_path):
        for ts in ['202501010800', '202501011200', '202501010600']:
            d = tmp_path / ts
            d.mkdir()
            (d / 'config.json').write_text('{}')
        result = _find_latest_checkpoint_config(str(tmp_path))
        assert '202501011200' in result

    def test_skips_incomplete_subdirs(self, tmp_path):
        # newest dir has no config.json — should fall back to older one
        (tmp_path / '202501011200').mkdir()          # incomplete: no config.json
        old = tmp_path / '202501010800'
        old.mkdir()
        (old / 'config.json').write_text('{}')
        result = _find_latest_checkpoint_config(str(tmp_path))
        assert '202501010800' in result

    def test_all_incomplete_returns_none(self, tmp_path):
        (tmp_path / '202501011200').mkdir()
        (tmp_path / '202501010800').mkdir()
        result = _find_latest_checkpoint_config(str(tmp_path))
        assert result is None
