"""
Tests for fast/utils/general.py — learning rate schedule math.
"""
import math
import pytest
from unittest.mock import MagicMock

from utils.general import adjust_learning_rate


def _mock_optimizer(initial_lr: float):
    opt = MagicMock()
    opt.param_groups = [{'lr': initial_lr}]
    return opt


class TestAdjustLearningRate:
    def test_epoch_zero_gives_init_lr(self):
        opt = _mock_optimizer(0.01)
        adjust_learning_rate(opt, epoch=0, max_epoch=100, init_lr=0.01)
        assert opt.param_groups[0]['lr'] == pytest.approx(0.01, rel=1e-6)

    def test_lr_decreases_over_epochs(self):
        init_lr, max_epoch = 0.1, 100
        opt1 = _mock_optimizer(init_lr)
        opt2 = _mock_optimizer(init_lr)
        adjust_learning_rate(opt1, epoch=10,  max_epoch=max_epoch, init_lr=init_lr)
        adjust_learning_rate(opt2, epoch=50,  max_epoch=max_epoch, init_lr=init_lr)
        assert opt1.param_groups[0]['lr'] > opt2.param_groups[0]['lr']

    def test_polynomial_formula(self):
        init_lr, epoch, max_epoch, power = 0.1, 20, 100, 0.9
        opt = _mock_optimizer(init_lr)
        adjust_learning_rate(opt, epoch=epoch, max_epoch=max_epoch,
                             init_lr=init_lr, power=power)
        expected = round(init_lr * ((1 - epoch / max_epoch) ** power), 8)
        assert opt.param_groups[0]['lr'] == pytest.approx(expected, rel=1e-7)

    def test_power_one_gives_linear_decay(self):
        init_lr, epoch, max_epoch = 1.0, 25, 100
        opt = _mock_optimizer(init_lr)
        adjust_learning_rate(opt, epoch=epoch, max_epoch=max_epoch,
                             init_lr=init_lr, power=1.0)
        expected = round(init_lr * (1 - epoch / max_epoch), 8)
        assert opt.param_groups[0]['lr'] == pytest.approx(expected, rel=1e-7)

    def test_all_param_groups_updated(self):
        opt = MagicMock()
        opt.param_groups = [{'lr': 0.1}, {'lr': 0.1}]
        adjust_learning_rate(opt, epoch=10, max_epoch=100, init_lr=0.1)
        for group in opt.param_groups:
            assert group['lr'] < 0.1
