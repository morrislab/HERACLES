"""Integration tests for Heracles(edit_model=SequentialModel(...))."""

import tempfile

import numpy as np
import pandas as pd
import pytest


def _toy_cm(n_cells=30, n_cassettes=4, cassette_size=5, n_alleles=5, seed=42,
            missing_frac=0.05):
    """Generate a valid sequential character matrix.

    Within each cassette, edits are contiguous from the left (ordered editing):
    e.g. [3, 2, 0, 0, 0] is valid, [0, 3, 0, 2, 0] is not.
    """
    rng = np.random.default_rng(seed)
    S = n_cassettes * cassette_size
    cm = np.zeros((n_cells, S), dtype=np.int64)
    for cell in range(n_cells):
        for c in range(n_cassettes):
            n_edits = rng.integers(0, cassette_size + 1)
            for k in range(n_edits):
                cm[cell, c * cassette_size + k] = rng.integers(1, n_alleles + 1)
    if missing_frac > 0:
        mask = rng.random((n_cells, n_cassettes)) < missing_frac
        for cell in range(n_cells):
            for c in range(n_cassettes):
                if mask[cell, c]:
                    cm[cell, c * cassette_size:(c + 1) * cassette_size] = -1
    return pd.DataFrame(cm, index=["c%d" % i for i in range(n_cells)])


# the model object

def test_sequential_model_options_and_round_trip():
    from heracles.models import SequentialModel, get_model
    m = SequentialModel(cassette_size=5, initiation_ratio=2.0, learn_initiation_ratio=True)
    d = m.to_dict()
    assert d == {"name": "sequential", "cassette_size": 5, "initiation_ratio": 2.0,
                 "learn_initiation_ratio": True}
    assert get_model(d) == m


def test_sequential_model_requires_cassette_size():
    from heracles.models import SequentialModel
    with pytest.raises(ValueError, match="needs cassette_size"):
        SequentialModel()
    with pytest.raises(ValueError, match="cassette_size must be >= 1"):
        SequentialModel(cassette_size=0)


def test_sequential_model_rejects_low_memory(tmp_path):
    from heracles import Heracles
    from heracles.models import SequentialModel
    cm = _toy_cm(n_cells=12, cassette_size=3, n_cassettes=2)
    m = Heracles(edit_model=SequentialModel(cassette_size=3), low_memory=True, outdir=tmp_path / "a")
    with pytest.raises(ValueError, match="low_memory is not available"):
        m.fit(cm)


def test_sequential_model_checks_matrix_width(tmp_path):
    from heracles import Heracles
    from heracles.models import SequentialModel
    cm = _toy_cm(n_cells=12, cassette_size=3, n_cassettes=2)     # 6 columns
    m = Heracles(edit_model=SequentialModel(cassette_size=4), outdir=tmp_path / "w")
    with pytest.raises(ValueError, match="whole cassettes"):
        m.fit(cm)


# end-to-end fit

def _seq(**kw):
    from heracles.models import SequentialModel
    return SequentialModel(cassette_size=5, **kw)


def test_sequential_fit_end_to_end():
    from heracles import Heracles
    cm = _toy_cm(n_cells=30, cassette_size=5, n_cassettes=4)
    with tempfile.TemporaryDirectory() as d:
        m = Heracles(edit_model=_seq(), max_iterations=30, convergence_thresh=None,
                     init_method="mds", seed=0, outdir=d)
        m.fit(cm)
    assert m.leaf_dists_ is not None
    assert m.leaf_dists_.shape == (30, 30)
    assert "nj" in m.trees_
    assert m.n_iter_ == 30
    assert "SequentialModel" in repr(m)
    with pytest.raises(ValueError, match="polish is not available"):
        m.polish(cm)


def test_sequential_fit_with_learned_params():
    from heracles import Heracles
    cm = _toy_cm(n_cells=30, cassette_size=5, n_cassettes=4)
    with tempfile.TemporaryDirectory() as d:
        m = Heracles(edit_model=_seq(initiation_ratio=2.0, learn_initiation_ratio=True),
                     learn_mutation_rate=True, max_iterations=30, convergence_thresh=None,
                     init_method="mds", seed=0, outdir=d)
        m.fit(cm)
    assert m.mutation_rate_ is not None
    # the learned value, not the starting 2.0
    assert m.initiation_ratio_ == pytest.approx(m.history_["initiation_ratio"][-1])
    assert m.initiation_ratio_ != 2.0


# save / load

def test_sequential_save_load_roundtrip():
    import os
    from heracles import Heracles
    cm = _toy_cm(n_cells=30, cassette_size=5, n_cassettes=4)
    with tempfile.TemporaryDirectory() as d:
        m = Heracles(edit_model=_seq(), max_iterations=20, convergence_thresh=None,
                     init_method="mds", seed=0, outdir=d)
        m.fit(cm)
        m.save()
        m2 = Heracles.load(os.path.join(d, "model.npz"))
    assert m2.edit_model == _seq()
    np.testing.assert_array_almost_equal(m.leaf_dists_, m2.leaf_dists_)
    from heracles.trees.utils import as_newick
    assert as_newick(m2.trees_["nj"]) == as_newick(m.trees_["nj"])


def test_sequential_get_params():
    from heracles import Heracles
    with tempfile.TemporaryDirectory() as d:
        m = Heracles(edit_model=_seq(), outdir=d)
    p = m.get_params()
    assert p["edit_model"]["name"] == "sequential"
    assert p["edit_model"]["cassette_size"] == 5


# K=1 equivalence with Cas9

def test_k1_sequential_matches_cas9_loss():
    """At cassette_size=1, sequential reduces to independent sites (= Cas9)."""
    import torch
    from heracles.models.sequential import sequential_quartet_loss_fn
    from heracles.embedding.objective import quartet_loss_fn
    from heracles.embedding.priors import build_allele_priors
    from heracles.embedding.initialization import seed_distances, lorentz_manifold

    rng = np.random.default_rng(99)
    n, S, A = 20, 10, 4
    states = rng.integers(0, A + 1, (n, S))
    states[states == A] = 0
    states = np.vstack([np.zeros((1, S), dtype=np.int64), states])
    prior = build_allele_priors(states[1:], prior="per_site_uniform")
    ls = {str(i): states[i] for i in range(n + 1)}
    labels = [str(i) for i in range(n + 1)]
    est = seed_distances(np.array([ls[l] for l in labels]))

    torch.manual_seed(7)
    cas9_fn = quartet_loss_fn(states, prior, n + 1, est_dists=est,
                               band_kmin=2, band_kmax=8)
    torch.manual_seed(7)
    seq_fn = sequential_quartet_loss_fn(states, prior, n + 1,
                                        cassette_size=1, est_dists=est,
                                        band_kmin=2, band_kmax=8)

    manifold, pdim = lorentz_manifold(8)
    tangent = torch.randn(n + 1, 8, dtype=torch.float64) * 0.1
    zeros = torch.zeros(n + 1, 1, dtype=torch.float64)
    points = manifold.expmap0(torch.cat([zeros, tangent], dim=1))
    rate = torch.tensor(1.0, dtype=torch.float64)

    torch.manual_seed(42)
    gen1 = torch.Generator().manual_seed(42)
    loss_cas9, _ = cas9_fn(points, manifold, torch.tensor(1.0), rate,
                           torch.tensor(0.0), torch.tensor(0.0), gen1)

    torch.manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    loss_seq, _ = seq_fn(points, manifold, torch.tensor(1.0), rate,
                         torch.tensor(0.0), torch.tensor(0.0), gen2)

    assert abs(loss_cas9.item() - loss_seq.item()) < 0.1, (
        "K=1 sequential loss %.4f vs Cas9 loss %.4f differ by more than 0.1"
        % (loss_seq.item(), loss_cas9.item()))


# CLI flags

def test_cli_builds_sequential_model():
    import argparse
    from heracles.models import SequentialModel, add_model_arguments, model_from_args
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    args = parser.parse_args(["--edit-model", "sequential", "--cassette-size", "5",
                              "--initiation-ratio", "2.0", "--learn-initiation-ratio"])
    assert model_from_args(args) == SequentialModel(cassette_size=5, initiation_ratio=2.0,
                                                    learn_initiation_ratio=True)
