"""Tests for heracles.embedding.initialization: landmark MDS honours the requested dimension."""

import numpy as np
import pytest
import torch

from heracles.embedding.initialization import (
    init_points, lorentz_manifold, choose_landmarks, classical_mds, default_n_landmarks,
    landmark_mds_from_rows,
)


def _rank40_dists(n, seed=0):
    """(n, n) Euclidean distances between n Gaussian points in 40 dimensions.

    A tree metric is not needed for the shape and rank behaviour under test;
    this matrix is well spread and has exact rank 40.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 40))
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    return d


def test_default_n_landmarks_exceeds_dim():
    assert default_n_landmarks(10) == 500
    assert default_n_landmarks(500) == 1001
    assert default_n_landmarks(1937) == 3875
    for d in (1, 250, 500, 791, 1582, 1937):
        assert default_n_landmarks(d) > d


def test_choose_landmarks_includes_founder_and_is_seeded():
    a = choose_landmarks(1000, 50, seed=3)
    b = choose_landmarks(1000, 50, seed=3)
    assert a.shape == (50,) and 0 in a and len(set(a.tolist())) == 50
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, choose_landmarks(1000, 50, seed=4))


def _landmark_mds(d, dim, n_landmarks=None, seed=0):
    n = d.shape[0]
    k = default_n_landmarks(dim) if n_landmarks is None else n_landmarks
    if n <= k:
        return classical_mds(d, dim)
    lm = choose_landmarks(n, k, seed=seed)
    return landmark_mds_from_rows(d[lm, :], lm, dim)


def test_landmark_mds_shape_above_500_dims():
    """The landmark count always exceeds the dimension, so it never caps it."""
    d = _rank40_dists(1200, seed=1)
    coords = _landmark_mds(d, dim=800)
    assert coords.shape == (1200, 800)
    assert np.isfinite(coords).all()


def test_landmark_mds_matches_classical_on_leading_dims():
    d = _rank40_dists(900, seed=2)
    dim = 45
    c_land = _landmark_mds(d, dim, n_landmarks=400)
    c_full = classical_mds(d, dim)
    # the exact data lives in 40 dims, so at dim=45 both are (near) exact and
    # the inter-point distances must agree; at dim < rank the two eigenbases
    # truncate differently and only agree loosely
    rng = np.random.default_rng(0)
    i = rng.integers(0, 900, 2000); j = rng.integers(0, 900, 2000)
    dl = np.linalg.norm(c_land[i] - c_land[j], axis=1)
    df = np.linalg.norm(c_full[i] - c_full[j], axis=1)
    assert np.corrcoef(dl, df)[0, 1] > 0.99


def test_landmark_mds_zeroes_noise_dims_instead_of_amplifying():
    # exact rank 40 data: dims beyond 40 are numerical noise and must be zero
    d = _rank40_dists(800, seed=5)
    coords = _landmark_mds(d, dim=120, n_landmarks=300)
    assert coords.shape == (800, 120)
    tail = np.abs(coords[:, 60:]).max()
    head = np.abs(coords[:, :20]).max()
    assert tail < 1e-6 * head


def test_landmark_mds_from_rows_rejects_too_few_landmarks():
    d = _rank40_dists(100)
    lm = np.arange(10)
    with pytest.raises(ValueError, match="more landmarks than dimensions"):
        landmark_mds_from_rows(d[lm], lm, dim=10)


def test_classical_mds_pads_when_dim_exceeds_n():
    d = _rank40_dists(8)
    coords = classical_mds(d, 12)
    assert coords.shape == (8, 12)
    assert np.abs(coords[:, 8:]).max() == 0.0


def test_init_points_low_memory_runs_at_requested_dimension():
    n, dim = 700, 600
    d = _rank40_dists(n, seed=7)
    manifold, point_dim = lorentz_manifold(dim)
    lm = choose_landmarks(n, default_n_landmarks(dim), seed=0)
    pts, meta = init_points(manifold, dim, seed_dists=d, init_method="mds", landmarks=lm)
    assert pts.shape == (n, point_dim) == (n, dim + 1)
    assert meta["init_dimension"] == dim
    assert meta["n_landmarks"] == min(default_n_landmarks(dim), n)
    assert meta["init_method_used"] == "landmark_mds"
    assert torch.isfinite(pts).all()


def test_init_points_guard_raises_on_wrong_width(monkeypatch):
    import heracles.embedding.initialization as ini
    monkeypatch.setattr(ini, "classical_mds", lambda d, dim: np.zeros((d.shape[0], 3)))
    manifold, _ = lorentz_manifold(5)
    with pytest.raises(RuntimeError, match="wrong dimension"):
        init_points(manifold, 5, seed_dists=_rank40_dists(20), init_method="mds")
