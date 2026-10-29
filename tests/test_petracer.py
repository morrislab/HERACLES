"""Tests for the PETracer model (cassette-level missing data, heracles.models.petracer).

What is checked:
  1. the pattern trick: the kernels at mutation_rate = 0 on the missing
     pattern equal the closed-form missing-data factor (eps = 0, so it is
     delta^#missing (1-delta)^#observed per leaf set);
  2. the cassette correction on the full quartet loss: with no missing data
     it differs from the per-site loss by a geometry-free constant, and with
     cassette-wise missing data it equals the closed-form cassette likelihood;
  3. an end-to-end fit with per-cassette dropout rates, save/load round trip;
  4. configuration validation and the partial-cassette warning;
  5. cassette-aware Felsenstein pruning in ``heracles.polish.ancestral``;
  6. per-cell dropout tables in the embedding kernels and through ``fit()``.
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import _toy_states, _hamming

from heracles.config import HeraclesConfig
from heracles.embedding.ctmc import leaf_marginal_logprob
from heracles.models import Cas9Model, PETracerModel
from heracles.models.missing import (CassetteCorrection, cassette_missingness_report,
                                     site_groups_from_size)
from heracles.fit import Heracles
from heracles.embedding.priors import build_allele_priors
from heracles.embedding.objective import quartet_loss_fn


# helpers

def _cassette_states(n=14, n_cassettes=4, size=3, K=3, seed=1, miss_frac=0.3):
    """Toy founder-augmented matrix whose missingness is cassette-wise."""
    rng = np.random.default_rng(seed)
    S = n_cassettes * size
    leaves = rng.integers(0, K + 1, size=(n, S)).astype(np.int64)
    drop = rng.random((n, n_cassettes)) < miss_frac
    leaves[np.repeat(drop, size, axis=1)] = -1
    return np.vstack([np.zeros((1, S), dtype=np.int64), leaves])


def _loss_at(states, cassette, delta, eps=0.0, seed=0):
    """Quartet loss at a fixed embedding and a fixed quartet batch."""
    delta_t = delta if torch.is_tensor(delta) else torch.tensor(float(delta), dtype=torch.float64)
    from heracles.embedding.initialization import lorentz_manifold, init_points
    m, S = states.shape
    prior = build_allele_priors(states[1:], prior="global_uniform")
    loss_fn = quartet_loss_fn(states, prior, m, est_dists=_hamming(states),
                               batch_size=40, cassette=cassette)
    manifold, _ = lorentz_manifold(4)
    torch.manual_seed(seed)
    pts, _ = init_points(manifold, 4, seed_dists=_hamming(states), init_method="mds")
    batch = loss_fn.make_eval_batch(torch.Generator().manual_seed(7))
    rng = torch.Generator().manual_seed(seed + 1)
    radius = torch.tensor(1.0, dtype=torch.float64)
    mu = torch.tensor(1.0, dtype=torch.float64)
    loss, _ = loss_fn(pts, manifold, radius, mu, torch.tensor(float(eps), dtype=torch.float64),
                      delta_t, rng, eval_batch=batch)
    return float(loss), loss_fn, batch


# 1. the pattern trick on the kernels

def test_leaf_marginal_at_zero_rate_is_missing_factor():
    states = _cassette_states()
    cas = CassetteCorrection(site_groups_from_size(states.shape[1], 3), states)
    delta = 0.2
    depths = torch.linspace(0.1, 2.0, states.shape[0], dtype=torch.float64)
    leaves = torch.arange(1, states.shape[0])
    prior = torch.as_tensor(build_allele_priors(states[1:], prior="global_uniform"))
    got = leaf_marginal_logprob(cas.pattern_states, depths, torch.tensor(0.0, dtype=torch.float64),
                                prior, leaves, heritable_eps=0.0, stochastic_delta=delta)
    miss = torch.as_tensor(states[1:] < 0)
    want = torch.where(miss, np.log(delta), np.log(1 - delta))
    assert torch.allclose(got, want.to(torch.float64))


# 2. the correction on the full loss

def test_cassette_loss_without_missing_differs_by_constant():
    states = _toy_states(n=12, S=6, K=3, seed=3, missing_frac=0.0)
    groups = site_groups_from_size(6, 3)
    delta = 0.1
    site_loss, _, _ = _loss_at(states, None, delta)
    cas_loss, loss_fn, _ = _loss_at(states, CassetteCorrection(groups, states), delta)
    # every (quartet, site) log-term gains back (1 - 1/3) * 4 log(1 - delta) and
    # every root term (1 - 1/3) * log(1 - delta) (both negative numbers that the
    # per-site model over-charged); the loss is minus the weighted mean, so
    # loss_cassette - loss_site = + (1 - 1/3) * mean pattern log-probability
    w = loss_fn.root_weight
    expect = (1 - w) * (2 / 3) * 4 * np.log(1 - delta) + w * (2 / 3) * np.log(1 - delta)
    assert cas_loss - site_loss == pytest.approx(expect, rel=1e-10)


def test_cassette_loss_matches_closed_form_with_missing():
    """At eps = 0 the pattern factor is geometry-free, so the cassette loss is
    the per-site loss with the surplus (1 - 1/|c|) * [m log d + (1-m) log(1-d)]
    per leaf added back, averaged the same way."""
    states = _cassette_states(seed=5)
    groups = site_groups_from_size(states.shape[1], 3)
    delta = 0.25
    site_loss, loss_fn, batch = _loss_at(states, None, delta)
    cas_loss, _, _ = _loss_at(states, CassetteCorrection(groups, states), delta)
    miss = states < 0
    per_leaf = np.where(miss, np.log(delta), np.log(1 - delta))      # (m, S)
    q_pat = per_leaf[batch.numpy()].sum(axis=1)                       # (B, S)
    root_pat = per_leaf[1:]                                           # all leaves (root_frac=1)
    w = loss_fn.root_weight
    expect = (1 - w) * (2 / 3) * q_pat.mean() + w * (2 / 3) * root_pat.mean()
    assert cas_loss - site_loss == pytest.approx(expect, rel=1e-10)


def test_cassette_correction_with_heritable_eps_is_finite_and_differs():
    states = _cassette_states(seed=6)
    groups = site_groups_from_size(states.shape[1], 3)
    a, _, _ = _loss_at(states, None, 0.2, eps=0.3)
    b, _, _ = _loss_at(states, CassetteCorrection(groups, states), 0.2, eps=0.3)
    assert np.isfinite(a) and np.isfinite(b) and a != b


# 3. end-to-end fit, per-cassette rates, round trip

def test_cassette_fit_per_cassette_rates_roundtrip(tmp_path):
    states = _cassette_states(n=12, n_cassettes=3, size=2, seed=9)
    cm = pd.DataFrame(states[1:], index=["cell_%d" % i for i in range(12)])
    m = Heracles(dimension=4, max_iterations=30, convergence_thresh=None,
                 heritable_eps=0.0, learn_heritable_eps=False,
                 stochastic_delta=0.05, learn_stochastic_delta=True,
                 edit_model=PETracerModel(cassette_size=2, separate_dropout_rates=True),
                 seed=0, allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    m.fit(cm)
    assert len(m.stochastic_delta_cassettes_) == 3
    assert len(m.stochastic_delta_sites_) == 6
    # the per-site vector is the per-cassette one repeated along each cassette
    assert m.stochastic_delta_sites_ == pytest.approx(
        np.repeat(m.stochastic_delta_cassettes_, 2).tolist())
    assert all(0.0 < v < 1.0 for v in m.stochastic_delta_cassettes_)
    params = m.get_params()
    assert params["edit_model"] == {"name": "petracer", "cassette_size": 2,
                                    "separate_dropout_rates": True}
    assert params["fitted"]["stochastic_delta_cassettes_"] == pytest.approx(m.stochastic_delta_cassettes_)
    assert os.path.exists(os.path.join(tmp_path, "model.npz"))
    loaded = Heracles.load(os.path.join(tmp_path, "model.npz"))
    assert loaded.edit_model == PETracerModel(cassette_size=2, separate_dropout_rates=True)
    assert loaded.stochastic_delta_cassettes_ == pytest.approx(m.stochastic_delta_cassettes_)


def test_cassette_fit_shared_delta_records_meta(tmp_path):
    states = _cassette_states(n=12, n_cassettes=3, size=2, seed=10)
    cm = pd.DataFrame(states[1:], index=["cell_%d" % i for i in range(12)])
    m = Heracles(dimension=4, max_iterations=20, convergence_thresh=None,
                 edit_model=PETracerModel(cassette_size=2),
                 seed=0, allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    m.fit(cm)
    assert m.stochastic_delta_cassettes_ is None
    assert 0.0 < m.stochastic_delta_ < 1.0
    assert m.leaf_dists_.shape == (12, 12)


# 4. validation and reporting

def test_site_groups_from_size():
    assert site_groups_from_size(6, 3).tolist() == [0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match="whole cassettes"):
        site_groups_from_size(7, 3)


def test_petracer_model_options(tmp_path):
    assert PETracerModel().cassette_size == 3                   # the PETracer barcode
    assert PETracerModel().process == Cas9Model().process      # same edit process
    with pytest.raises(ValueError, match="cassette_size must be >= 1"):
        PETracerModel(cassette_size=0)
    with pytest.raises(ValueError, match="requires learn_stochastic_delta"):
        Heracles(edit_model=PETracerModel(separate_dropout_rates=True),
                 learn_stochastic_delta=False, stochastic_delta=0.1, outdir=tmp_path / "x")
    states = _cassette_states(n=8, n_cassettes=2, size=3, seed=2)
    cm = pd.DataFrame(states[1:], index=["c%d" % i for i in range(8)])
    with pytest.raises(ValueError, match="whole cassettes"):
        Heracles(edit_model=PETracerModel(cassette_size=4), outdir=tmp_path / "y").fit(cm)


def test_missingness_report_and_partial_warning(caplog):
    states = _cassette_states(n=10, n_cassettes=2, size=3, seed=2)
    groups = site_groups_from_size(6, 3)
    rep = cassette_missingness_report(states[1:] < 0, groups)
    assert rep["n_partial"] == 0 and rep["n_cell_cassettes"] == 20
    states[1, 0] = -1 if states[1, 0] >= 0 else 0        # break one cassette
    cas = CassetteCorrection(groups, states)
    assert cas.report["n_partial"] == 1
    with caplog.at_level(logging.WARNING, logger="heracles.models.cas9"):
        cas.log_report()
    assert "PARTIALLY missing" in caplog.text


# 5. cassette-aware Felsenstein pruning (heracles.polish.ancestral, site_groups=)

def _tree(newick):
    from heracles.trees.utils import parse_newick
    return parse_newick(newick, rooting="force-rooted")


def _prune(newick, values, eps, delta, site_groups=None, mu=0.9, prior="global_uniform",
           posteriors=True):
    from heracles.polish.ancestral import infer_ancestral_states
    labels = ["A", "B", "C", "D"]
    cm = pd.DataFrame(np.asarray(values, dtype=np.int64), index=labels,
                      columns=["r%d" % j for j in range(np.asarray(values).shape[1])])
    return infer_ancestral_states(_tree(newick), cm, mutation_rate=mu, allele_prior=prior,
                                  heritable_eps=eps, stochastic_delta=delta,
                                  site_groups=site_groups, posteriors=posteriors)


_NWK = "((A:0.3,B:0.5):0.4,(C:0.6,D:0.2):0.3);"
# 2 cassettes x 3 sites; A lost cassette 0, D lost cassette 1
_VALS = [[-1, -1, -1, 1, 0, 2],
         [1, 0, 2, 0, 1, 1],
         [0, 2, 2, 1, 0, 1],
         [1, 1, 0, -1, -1, -1]]
_GROUPS = [0, 0, 0, 1, 1, 1]


def test_pruning_cassette_matches_closed_form_at_eps0():
    delta = 0.2
    per_site = _prune(_NWK, _VALS, 0.0, delta)
    cas = _prune(_NWK, _VALS, 0.0, delta, site_groups=_GROUPS)
    miss = np.asarray(_VALS) < 0
    pat = np.where(miss, np.log(delta), np.log(1 - delta)).sum(axis=0)     # (S,)
    want = per_site.loglik_per_site - (2 / 3) * pat
    assert cas.loglik_per_site == pytest.approx(want, rel=1e-12)
    assert cas.loglik == pytest.approx(want.sum(), rel=1e-12)


def test_pruning_cassette_equals_explicit_pattern_pass_with_eps():
    """With eps > 0 the pattern factor depends on the tree; it must equal an
    explicit pruning of the 0/-1 matrix at mutation_rate = 0."""
    eps, delta = 0.3, 0.15
    per_site = _prune(_NWK, _VALS, eps, delta)
    pattern = np.where(np.asarray(_VALS) < 0, -1, 0)
    pat = _prune(_NWK, pattern, eps, delta, mu=0.0)
    cas = _prune(_NWK, _VALS, eps, delta, site_groups=_GROUPS)
    want = per_site.loglik_per_site - (2 / 3) * pat.loglik_per_site
    assert cas.loglik_per_site == pytest.approx(want, rel=1e-12)
    # reported arrays are folded back to the S data sites
    assert cas.posteriors.shape[1] == 6 and cas.state_values.shape[0] == 6
    assert np.allclose(cas.posteriors, per_site.posteriors)


def test_pruning_cassette_no_missing_is_constant_shift():
    vals = [[1, 0, 2, 1, 0, 2], [1, 0, 2, 0, 1, 1], [0, 2, 2, 1, 0, 1], [1, 1, 0, 2, 1, 0]]
    delta = 0.1
    a = _prune(_NWK, vals, 0.0, delta, posteriors=False).loglik
    b = _prune(_NWK, vals, 0.0, delta, site_groups=_GROUPS, posteriors=False).loglik
    assert b - a == pytest.approx(-(2 / 3) * 6 * 4 * np.log(1 - delta), rel=1e-12)


def test_pruning_per_site_delta_vector():
    scalar = _prune(_NWK, _VALS, 0.1, 0.2, posteriors=False).loglik
    vec = _prune(_NWK, _VALS, 0.1, np.full(6, 0.2), posteriors=False).loglik
    assert vec == pytest.approx(scalar, rel=1e-12)
    other = _prune(_NWK, _VALS, 0.1, np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.05]),
                   posteriors=False).loglik
    assert other != pytest.approx(scalar)


def test_branch_length_fit_and_refit_run_with_cassette_model():
    from heracles.polish.ancestral import optimize_branch_lengths
    from heracles.polish.nni import guided_refit
    labels = ["A", "B", "C", "D"]
    cm = pd.DataFrame(np.asarray(_VALS, dtype=np.int64), index=labels)
    t = _tree(_NWK)
    rates = dict(mutation_rate=0.9, heritable_eps=0.2, stochastic_delta=0.2,
                 site_groups=_GROUPS)
    ll0 = _prune(_NWK, _VALS, 0.2, 0.2, site_groups=_GROUPS, posteriors=False).loglik
    ll, _, _ = optimize_branch_lengths(t, cm, allele_prior="global_uniform", **rates)
    assert ll >= ll0 - 1e-9 and np.isfinite(ll)
    # the NNI machinery accepts the same rates dict (site_groups passes through)
    full = np.ones((5, 5)) - np.eye(5)
    res = guided_refit(_tree(_NWK), cm, full, labels, rates=rates,
                       allele_prior="global_uniform", support_threshold=1.01,
                       max_rounds=1, branch_lengths="ml", min_gain=0.0,
                       seed_sweeps=1, local_sweeps=1, log=None)
    assert np.isfinite(res.loglik_end) and res.loglik_end >= res.loglik_start - 1e-9


def test_pruning_per_cell_delta_matrix():
    n, S = 4, 6
    scalar = _prune(_NWK, _VALS, 0.1, 0.2, posteriors=False).loglik
    same = _prune(_NWK, _VALS, 0.1, np.full((n, S), 0.2), posteriors=False).loglik
    assert same == pytest.approx(scalar, rel=1e-12)
    d = np.full((n, S), 0.2); d[0, :] = 0.6        # leaf A (row 0) has its own rate
    diff = _prune(_NWK, _VALS, 0.1, d, posteriors=False).loglik
    assert diff != pytest.approx(scalar)
    # also valid with the cassette model (stacked pattern sites get the same per-cell rates)
    cas = _prune(_NWK, _VALS, 0.1, d, site_groups=_GROUPS, posteriors=False).loglik
    assert np.isfinite(cas)
    with pytest.raises(ValueError, match="per-cell stochastic_delta"):
        _prune(_NWK, _VALS, 0.1, np.full((3, S), 0.2), posteriors=False)


# 6. per-cell dropout in the embedding kernels and through fit()

def test_kernels_per_cell_delta_matrix_matches_scalar_and_closed_form():
    states = _cassette_states(seed=11)
    m, S = states.shape
    depths = torch.linspace(0.1, 2.0, m, dtype=torch.float64)
    leaves = torch.arange(1, m)
    prior = torch.as_tensor(build_allele_priors(states[1:], prior="global_uniform"))
    scalar = leaf_marginal_logprob(torch.as_tensor(states), depths, torch.tensor(1.0, dtype=torch.float64), prior, leaves,
                                   heritable_eps=0.0, stochastic_delta=torch.tensor(0.2, dtype=torch.float64))
    table = torch.full((m, S), 0.2, dtype=torch.float64)
    same = leaf_marginal_logprob(torch.as_tensor(states), depths, torch.tensor(1.0, dtype=torch.float64), prior, leaves,
                                 heritable_eps=0.0, stochastic_delta=table)
    assert torch.allclose(scalar, same)
    # per-cell rates: at eps = 0 each leaf's missing entries contribute log delta_i, observed log(1 - delta_i)
    g = torch.Generator().manual_seed(3)
    table = torch.rand((m, S), generator=g, dtype=torch.float64) * 0.6 + 0.05
    got = leaf_marginal_logprob(torch.as_tensor(states), depths, torch.tensor(1.0, dtype=torch.float64), prior, leaves,
                                heritable_eps=0.0, stochastic_delta=table)
    scalar0 = leaf_marginal_logprob(torch.as_tensor(states), depths, torch.tensor(1.0, dtype=torch.float64), prior, leaves,
                                    heritable_eps=0.0, stochastic_delta=torch.tensor(0.0, dtype=torch.float64))
    miss = torch.as_tensor(states[1:] < 0)
    # observed entries: log(1 - delta) added to the complete-data term; missing entries: log delta (complete-data term is floored)
    want_obs = scalar0 + torch.log1p(-table[1:])
    assert torch.allclose(got[~miss], want_obs[~miss])
    assert torch.allclose(got[miss], torch.log(table[1:])[miss])


def test_quartet_loss_per_cell_delta_differs_from_scalar():
    states = _cassette_states(seed=12)
    m, S = states.shape
    cas = CassetteCorrection(site_groups_from_size(S, 3), states)
    a, _, _ = _loss_at(states, cas, 0.2, eps=0.1)
    g = torch.Generator().manual_seed(5)
    table = torch.rand((m, S), generator=g, dtype=torch.float64) * 0.5 + 0.05
    table[0] = 0.0
    b, _, _ = _loss_at(states, cas, table, eps=0.1)
    c, _, _ = _loss_at(states, cas, torch.full((m, S), 0.2, dtype=torch.float64), eps=0.1)
    assert c == pytest.approx(a, rel=1e-12) and b != pytest.approx(a) and np.isfinite(b)


def test_fit_with_cell_dropout_table(tmp_path):
    states = _cassette_states(n=12, n_cassettes=3, size=2, seed=13)
    cm = pd.DataFrame(states[1:], index=["cell_%d" % i for i in range(12)], columns=["s%d" % j for j in range(6)])
    rng = np.random.default_rng(0)
    table = pd.DataFrame(rng.uniform(0.02, 0.6, size=cm.shape), index=cm.index[::-1], columns=cm.columns)  # rows shuffled on purpose
    m = Heracles(dimension=4, max_iterations=20, convergence_thresh=None,
                 edit_model=PETracerModel(cassette_size=2), heritable_eps=0.05, learn_heritable_eps=True,
                 seed=0, allele_prior="global_uniform", init_method="mds", outdir=tmp_path)
    m.fit(cm, cell_dropout=table)
    assert m.stochastic_delta_sites_ is None
    assert m.stochastic_delta_ == pytest.approx(float(table.loc[cm.index].to_numpy().mean()))
    assert m.leaf_dists_.shape == (12, 12) and np.isfinite(m.leaf_dists_).all()
    with pytest.raises(ValueError, match="lacks"):
        Heracles(dimension=4, max_iterations=5, outdir=tmp_path / "b", allele_prior="global_uniform",
                 init_method="mds").fit(cm, cell_dropout=table.iloc[:-1])
