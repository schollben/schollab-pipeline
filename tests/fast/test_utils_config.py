"""
Tests for fast/utils/config.py — JSON ↔ argparse Namespace helpers.
No external dependencies beyond stdlib.
"""
import json
import argparse
import pytest

from utils.config import args2json, json2args


class TestJson2Args:
    def test_returns_namespace_with_correct_attrs(self, tmp_path):
        cfg = {'lr': 0.001, 'epochs': 50, 'model': 'unet'}
        p = tmp_path / 'cfg.json'
        p.write_text(json.dumps(cfg))
        ns = json2args(str(p))
        assert ns.lr == 0.001
        assert ns.epochs == 50
        assert ns.model == 'unet'

    def test_dot_access_works(self, tmp_path):
        p = tmp_path / 'cfg.json'
        p.write_text(json.dumps({'batch_size': 8}))
        ns = json2args(str(p))
        assert ns.batch_size == 8

    def test_boolean_values_preserved(self, tmp_path):
        p = tmp_path / 'cfg.json'
        p.write_text(json.dumps({'skip_training': True}))
        ns = json2args(str(p))
        assert ns.skip_training is True

    def test_nested_structures_preserved(self, tmp_path):
        p = tmp_path / 'cfg.json'
        p.write_text(json.dumps({'folders': ['/a', '/b']}))
        ns = json2args(str(p))
        assert ns.folders == ['/a', '/b']

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            json2args(str(tmp_path / 'nonexistent.json'))

    def test_empty_config_gives_empty_namespace(self, tmp_path):
        p = tmp_path / 'cfg.json'
        p.write_text('{}')
        ns = json2args(str(p))
        assert ns.__dict__ == {}


class TestArgs2Json:
    def test_writes_file(self, tmp_path):
        ns = argparse.Namespace(lr=0.01, epochs=10)
        out = tmp_path / 'out.json'
        args2json(ns, str(out))
        assert out.exists()

    def test_roundtrip(self, tmp_path):
        ns = argparse.Namespace(lr=0.01, epochs=10, name='run1')
        out = tmp_path / 'out.json'
        args2json(ns, str(out))
        with open(out) as f:
            data = json.load(f)
        assert data == {'lr': 0.01, 'epochs': 10, 'name': 'run1'}

    def test_output_is_valid_json(self, tmp_path):
        ns = argparse.Namespace(x=1)
        out = tmp_path / 'out.json'
        args2json(ns, str(out))
        json.loads(out.read_text())  # should not raise
