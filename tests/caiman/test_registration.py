"""
Tests for pure helper functions in caiman/registration.py.

These functions handle env-var parsing, systemd argument construction,
and conda path resolution — all pure logic with no CaImAn dependency.
"""
import os
import pytest
from unittest.mock import patch

import registration


class TestCaimanNProcesses:
    def test_default_from_config(self):
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(registration, 'CAIMAN_CONFIG', {'n_processes': 4}):
            os.environ.pop('CAIMAN_N_PROCESSES', None)
            assert registration._caiman_n_processes() == 4

    def test_reads_from_env_var(self):
        with patch.dict(os.environ, {'CAIMAN_N_PROCESSES': '8'}):
            assert registration._caiman_n_processes() == 8

    def test_env_var_takes_precedence_over_config(self):
        with patch.dict(os.environ, {'CAIMAN_N_PROCESSES': '2'}), \
             patch.object(registration, 'CAIMAN_CONFIG', {'n_processes': 16}):
            assert registration._caiman_n_processes() == 2

    def test_non_integer_env_raises(self):
        with patch.dict(os.environ, {'CAIMAN_N_PROCESSES': 'four'}):
            with pytest.raises(ValueError, match='must be an integer'):
                registration._caiman_n_processes()

    def test_zero_raises(self):
        with patch.dict(os.environ, {'CAIMAN_N_PROCESSES': '0'}):
            with pytest.raises(ValueError, match='>= 1'):
                registration._caiman_n_processes()

    def test_negative_raises(self):
        with patch.dict(os.environ, {'CAIMAN_N_PROCESSES': '-3'}):
            with pytest.raises(ValueError, match='>= 1'):
                registration._caiman_n_processes()

    def test_float_string_raises(self):
        with patch.dict(os.environ, {'CAIMAN_N_PROCESSES': '2.5'}):
            with pytest.raises(ValueError, match='must be an integer'):
                registration._caiman_n_processes()


class TestCaimanThreadSetenvArgs:
    def test_returns_setenv_for_present_thread_vars(self):
        cfg = {'threads': {'OMP_NUM_THREADS': 1, 'MKL_NUM_THREADS': 1}}
        with patch.object(registration, 'CAIMAN_CONFIG', cfg), \
             patch.dict(os.environ, {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '4'}):
            args = registration._caiman_thread_setenv_args()
        assert '--setenv=OMP_NUM_THREADS=2' in args
        assert '--setenv=MKL_NUM_THREADS=4' in args

    def test_absent_thread_vars_excluded(self):
        cfg = {'threads': {'OMP_NUM_THREADS': 1}}
        env = {k: v for k, v in os.environ.items() if k != 'OMP_NUM_THREADS'}
        with patch.object(registration, 'CAIMAN_CONFIG', cfg), \
             patch.dict(os.environ, env, clear=True):
            args = registration._caiman_thread_setenv_args()
        assert args == []

    def test_empty_threads_config(self):
        with patch.object(registration, 'CAIMAN_CONFIG', {'threads': {}}):
            assert registration._caiman_thread_setenv_args() == []

    def test_output_is_sorted(self):
        cfg = {'threads': {'ZZZ': 1, 'AAA': 1}}
        with patch.object(registration, 'CAIMAN_CONFIG', cfg), \
             patch.dict(os.environ, {'ZZZ': '1', 'AAA': '1'}):
            args = registration._caiman_thread_setenv_args()
        keys = [a.split('=')[1] for a in args]
        assert keys == sorted(keys)


class TestFastPathSetenvArgs:
    def test_includes_fast_dir_when_set(self):
        with patch.dict(os.environ, {'FAST_DIR': '/data/fast'}):
            args = registration._fast_path_setenv_args()
        assert '--setenv=FAST_DIR=/data/fast' in args

    def test_includes_scratch_dir_when_set(self):
        with patch.dict(os.environ, {'FAST_SCRATCH_DIR': '/tmp/scratch'}):
            args = registration._fast_path_setenv_args()
        assert '--setenv=FAST_SCRATCH_DIR=/tmp/scratch' in args

    def test_excludes_absent_vars(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ('FAST_DIR', 'FAST_SCRATCH_DIR')}
        with patch.dict(os.environ, env, clear=True):
            args = registration._fast_path_setenv_args()
        assert args == []

    def test_both_vars_present(self):
        with patch.dict(os.environ, {'FAST_DIR': '/d', 'FAST_SCRATCH_DIR': '/s'}):
            args = registration._fast_path_setenv_args()
        assert len(args) == 2


class TestSchollabCondaRoot:
    def test_default_is_miniforge3_in_home(self):
        env = {k: v for k, v in os.environ.items() if k != 'SCHOLLAB_CONDA_ROOT'}
        with patch.dict(os.environ, env, clear=True):
            result = registration._schollab_conda_root()
        assert result == os.path.join(os.path.expanduser('~'), 'miniforge3')

    def test_env_var_override(self):
        with patch.dict(os.environ, {'SCHOLLAB_CONDA_ROOT': '~/myconda'}):
            result = registration._schollab_conda_root()
        assert result == os.path.expanduser('~/myconda')

    def test_env_var_tilde_expanded(self):
        with patch.dict(os.environ, {'SCHOLLAB_CONDA_ROOT': '~/envs'}):
            result = registration._schollab_conda_root()
        assert '~' not in result
