"""Shared test helpers."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _fresh_rapidnj_lookup():
    """The binary lookup is cached per process; tests that monkeypatch it need
    a clean cache before and after."""
    from heracles.trees.rapidnj import _resolve_binary_cached
    _resolve_binary_cached.cache_clear()
    yield
    _resolve_binary_cached.cache_clear()


def _toy_states(n=12, S=6, K=4, seed=0, missing_frac=0.0):
    """Small character matrix with the all-unedited root as row 0."""
    rng = np.random.default_rng(seed)
    leaves = rng.integers(0, K + 1, size=(n, S)).astype(np.int64)
    if missing_frac:
        leaves[rng.random(leaves.shape) < missing_frac] = -1
    return np.vstack([np.zeros((1, S), dtype=np.int64), leaves])


def _hamming(states):
    """Unweighted per-site mismatch distance, used only to seed an initialisation."""
    d = (states[:, None, :] != states[None, :, :]).mean(-1).astype(np.float64)
    np.fill_diagonal(d, 0.0)
    return d
