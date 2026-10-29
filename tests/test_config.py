"""Tests for ``HeraclesConfig`` parsing and validation."""

import pytest

from heracles.config import AUTO_RATE_START, HeraclesConfig


def test_auto_rates_respect_learn_flags():
    cfg = HeraclesConfig()
    assert cfg.heritable_eps == AUTO_RATE_START and cfg.learn_heritable_eps
    assert cfg.stochastic_delta == AUTO_RATE_START and cfg.learn_stochastic_delta
    fixed = HeraclesConfig(learn_heritable_eps=False, learn_stochastic_delta=False)
    assert fixed.heritable_eps == AUTO_RATE_START and not fixed.learn_heritable_eps
    assert fixed.stochastic_delta == AUTO_RATE_START and not fixed.learn_stochastic_delta


def test_zero_rate_cannot_be_learned():
    with pytest.raises(ValueError):
        HeraclesConfig(heritable_eps=0.0)
    cfg = HeraclesConfig(heritable_eps=0.0, learn_heritable_eps=False)
    assert cfg.heritable_eps == 0.0


def test_config_round_trip_through_dict():
    cfg = HeraclesConfig(dimension=16, learn_stochastic_delta=False)
    back, ignored = HeraclesConfig.from_params(cfg.to_dict())
    assert back == cfg and ignored == []
