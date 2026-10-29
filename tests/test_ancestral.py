"""Tests for Felsenstein ancestral-state inference (``heracles.polish.ancestral``).

Run with:

    .venv/bin/python -m pytest tests/test_ancestral.py -q

Ordered by what would silently corrupt a result if wrong:

  1. the pruning likelihood on a single root-leaf path IS ``leaf_marginal_logprob``;
  2. on a cherry it IS the pairwise joint ``pair_edit_joint_missing`` for every
     observation-pattern class;
  3. it is a probability distribution (sums to 1 over all observation combos)
     and posteriors normalise at every node and site;
  4. at eps = delta = 0 it reduces to the complete-data likelihood (checked
     against an independent brute-force sum over ancestral states) and DEL
     posteriors vanish;
  5. the calling rule: raw argmax is never negative, thresholded calls go -1
     exactly for DEL-dominant or low-confidence posteriors;
  6. global and broadcast per-site priors agree; the npz round-trips;
  7. per-edge branch-length curves match a full recomputation, and the
     optimiser climbs monotonically and recovers simulated lengths.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from heracles.polish.ancestral import (
    AncestralResult,
    DEL_STATE,
    OTHER_STATE,
    PAD_STATE,
    call_states,
    infer_ancestral_states,
    load_posteriors_npz,
    save_posteriors_npz,
)
from heracles.embedding.ctmc import leaf_marginal_logprob, pair_edit_joint_missing

torch.set_default_dtype(torch.float64)

MU = 0.9
PRIOR = np.array([1.0, 0.55, 0.45])       # alleles {1, 2}; q sums to 1


def _tree(newick):
    from heracles.trees.utils import parse_newick
    return parse_newick(newick, rooting="force-rooted")


def _cm(values, labels, n_sites=None):
    arr = np.atleast_2d(np.asarray(values, dtype=np.int64))
    cols = ["r%d" % j for j in range(arr.shape[1])]
    return pd.DataFrame(arr, index=labels, columns=cols)


def _prune_loglik(newick, values, labels, eps, delta, prior=PRIOR, mu=MU):
    res = infer_ancestral_states(_tree(newick), _cm(values, labels),
                                 mutation_rate=mu, allele_prior=prior,
                                 heritable_eps=eps, stochastic_delta=delta)
    return res


# 1. root-leaf path == leaf_marginal_logprob
@pytest.mark.parametrize("eps", [0.0, 0.15])
@pytest.mark.parametrize("delta", [0.0, 0.3])
def test_path_matches_leaf_marginal(eps, delta):
    t = 0.7
    obs = [0, 2, -1]                                        # one site per case
    res = _prune_loglik("(A:%g);" % t, [obs], ["A"], eps, delta)

    states = torch.tensor([[0] * 3, obs], dtype=torch.long)  # root row 0
    ref = leaf_marginal_logprob(states, torch.tensor([0.0, t]),
                                torch.tensor(MU), torch.as_tensor(PRIOR),
                                torch.tensor([1]), heritable_eps=eps,
                                stochastic_delta=delta)
    # the reference floors impossible observations at log(1e-12); ancestral.py
    # floors at 1e-300, so clamp to the coarser floor before comparing
    got = np.maximum(res.loglik_per_site, np.log(1e-12))
    np.testing.assert_allclose(got, ref[0].numpy(), rtol=1e-10, atol=1e-12)


# 2. cherry == pair_edit_joint_missing, all observation-pattern classes
@pytest.mark.parametrize("eps", [0.0, 0.15])
@pytest.mark.parametrize("delta", [0.0, 0.3])
def test_cherry_matches_pairwise_joint(eps, delta):
    a, b, c = 0.4, 0.3, 0.5                                  # stem, leaf i, leaf j

    def one(v):
        return torch.tensor([[float(v)]])

    for s_i, s_j in itertools.product([0, 1, 2, -1], repeat=2):
        res = _prune_loglik("((I:%g,J:%g):%g);" % (b, c, a),
                            [[s_i], [s_j]], ["I", "J"], eps, delta)
        si, sj = torch.tensor([[s_i]]), torch.tensor([[s_j]])
        mi, mj = si < 0, sj < 0
        f = lambda m: m.to(torch.float64)
        ref = pair_edit_joint_missing(
            one(np.exp(-MU * a)), one(np.exp(-MU * b)), one(np.exp(-MU * c)),
            one(PRIOR[max(s_i, 0)]), one(PRIOR[max(s_j, 0)]),
            one(np.exp(-eps * a)),
            one(np.exp(-eps * b) * (1 - delta)), one(np.exp(-eps * c) * (1 - delta)),
            f((si == 0) & (sj == 0)), f((si == 0) & (sj > 0)),
            f((si > 0) & (sj == 0)), f((si > 0) & (si == sj)),
            f((si > 0) & (sj > 0) & (si != sj)),
            f((si == 0) & mj), f((si > 0) & mj),
            f(mi & (sj == 0)), f(mi & (sj > 0)), f(mi & mj))
        np.testing.assert_allclose(np.exp(res.loglik), float(ref),
                                   rtol=1e-10, atol=1e-12,
                                   err_msg="pattern (%d, %d)" % (s_i, s_j))


# 3. distribution: sums to 1 over observations; posteriors normalise
@pytest.mark.parametrize("eps,delta", [(0.0, 0.0), (0.1, 0.2)])
def test_likelihood_sums_to_one(eps, delta):
    newick = "((I:0.3,J:0.5):0.4,K:0.8);"
    total = 0.0
    for combo in itertools.product([0, 1, 2, -1], repeat=3):
        res = _prune_loglik(newick, [[v] for v in combo], ["I", "J", "K"],
                            eps, delta)
        total += float(np.exp(res.loglik))
    np.testing.assert_allclose(total, 1.0, rtol=1e-10)


def test_posteriors_normalise():
    rng = np.random.default_rng(0)
    values = rng.choice([0, 1, 2, -1], size=(4, 6), p=[0.4, 0.25, 0.25, 0.1])
    res = infer_ancestral_states(
        _tree("(((A:0.2,B:0.3):0.2,C:0.6):0.1,D:0.9);"),
        _cm(values, list("ABCD")),
        mutation_rate=MU, allele_prior=PRIOR,
        heritable_eps=0.1, stochastic_delta=0.2)
    sums = res.posteriors.sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-9)
    # padding carries no mass
    pad = res.state_values[None, :, :] == PAD_STATE
    assert float(np.abs(res.posteriors * pad).max()) == 0.0


# 4. complete-data reduction, vs independent brute force
def _brute_force_complete(obs_i, obs_j, obs_k, a, b, c, d, q, mu=MU):
    """Sum over ancestral states u (below root, ((I:b,J:c)u:a,K:d)root)."""
    alleles = [0, 1, 2]

    def trans(x, y, t):
        if x == 0:
            return np.exp(-mu * t) if y == 0 else (1 - np.exp(-mu * t)) * q[y]
        return 1.0 if y == x else 0.0

    return sum(trans(0, u, a) * trans(u, obs_i, b) * trans(u, obs_j, c)
               for u in alleles) * trans(0, obs_k, d)


def test_complete_data_reduction():
    a, b, c, d = 0.4, 0.3, 0.5, 0.8
    newick = "((I:%g,J:%g):%g,K:%g);" % (b, c, a, d)
    for combo in itertools.product([0, 1, 2], repeat=3):
        res = _prune_loglik(newick, [[v] for v in combo], ["I", "J", "K"],
                            0.0, 0.0)
        ref = _brute_force_complete(*combo, a, b, c, d, PRIOR)
        np.testing.assert_allclose(np.exp(res.loglik), ref, rtol=1e-10,
                                   err_msg="combo %s" % (combo,))
        assert float(res.del_posterior.max()) == 0.0


# 5. calling rule
def test_argmax_calls_never_negative():
    rng = np.random.default_rng(1)
    values = rng.choice([0, 1, 2, -1], size=(6, 8), p=[0.2, 0.2, 0.2, 0.4])
    res = infer_ancestral_states(
        _tree("(((A:0.2,B:0.3):0.2,(C:0.4,D:0.1):0.3):0.1,(E:0.5,F:0.2):0.4);"),
        _cm(values, list("ABCDEF")),
        mutation_rate=MU, allele_prior=PRIOR,
        heritable_eps=0.4, stochastic_delta=0.3)   # heavy missing pressure
    assert res.argmax_calls.min() >= 0


def _synthetic_result(posteriors, state_values):
    posteriors = np.asarray(posteriors, dtype=np.float64)
    state_values = np.asarray(state_values, dtype=np.int64)
    S = state_values.shape[0]
    n_states = (state_values != PAD_STATE).sum(axis=1).astype(np.int64)
    s_idx = np.arange(S)
    del_p = posteriors[:, s_idx, n_states - 1]
    other_p = posteriors[:, s_idx, n_states - 2]
    called = posteriors.copy()
    called[:, s_idx, n_states - 1] = 0.0
    called[:, s_idx, n_states - 2] = 0.0
    argmax = state_values[s_idx[None, :], called.argmax(axis=2)]
    return AncestralResult(
        node_labels=["nd0"], is_leaf=np.array([False]), posteriors=posteriors,
        state_values=state_values, n_states=n_states, argmax_calls=argmax,
        del_posterior=del_p, other_posterior=other_p,
        loglik_per_site=np.zeros(S), loglik=0.0,
        site_names=["r%d" % j for j in range(S)], tree_newick="",
        mutation_rate=MU, heritable_eps=0.0, stochastic_delta=0.0,
        root_edge_length=0.0)


def test_threshold_calls():
    state_values = [[0, 5, OTHER_STATE, DEL_STATE]] * 3
    res = _synthetic_result([[
        [0.10, 0.10, 0.00, 0.80],     # DEL-dominant -> -1
        [0.45, 0.35, 0.20, 0.00],     # conf 0.45/0.80 = 0.5625
        [0.05, 0.90, 0.00, 0.05],     # confident allele 5
    ]], state_values)

    calls, info = call_states(res, min_call_posterior=0.5)
    assert calls.tolist() == [[-1, 0, 5]]
    assert info["del_masked"].tolist() == [[True, False, False]]
    assert info["lowconf_masked"].tolist() == [[False, False, False]]

    calls, info = call_states(res, min_call_posterior=0.6)
    assert calls.tolist() == [[-1, -1, 5]]
    assert info["lowconf_masked"].tolist() == [[False, True, False]]
    assert set(np.unique(calls)) <= {-1, 0, 5}


# 6. prior forms agree; npz round-trip
def test_global_and_per_site_priors_agree():
    rng = np.random.default_rng(2)
    values = rng.choice([0, 1, 2, -1], size=(4, 5), p=[0.4, 0.25, 0.25, 0.1])
    per_site = np.tile(PRIOR[None, :], (values.shape[1], 1))
    kw = dict(mutation_rate=MU, heritable_eps=0.1, stochastic_delta=0.2)
    tree = "(((A:0.2,B:0.3):0.2,C:0.6):0.1,D:0.9);"
    r1 = infer_ancestral_states(_tree(tree), _cm(values, list("ABCD")),
                                allele_prior=PRIOR, **kw)
    r2 = infer_ancestral_states(_tree(tree), _cm(values, list("ABCD")),
                                allele_prior=per_site, **kw)
    np.testing.assert_allclose(r1.loglik_per_site, r2.loglik_per_site, rtol=1e-12)
    np.testing.assert_allclose(r1.posteriors, r2.posteriors, rtol=1e-12)


def test_npz_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    values = rng.choice([0, 1, 2, -1], size=(4, 5), p=[0.4, 0.25, 0.25, 0.1])
    res = infer_ancestral_states(
        _tree("(((A:0.2,B:0.3):0.2,C:0.6):0.1,D:0.9);"),
        _cm(values, list("ABCD")),
        mutation_rate=MU, allele_prior=PRIOR,
        heritable_eps=0.1, stochastic_delta=0.2, root_edge_length=0.05)
    path = tmp_path / "anc.npz"
    save_posteriors_npz(path, res)
    back = load_posteriors_npz(path)
    assert back.node_labels == res.node_labels
    np.testing.assert_array_equal(back.state_values, res.state_values)
    np.testing.assert_array_equal(back.n_states, res.n_states)
    np.testing.assert_allclose(back.posteriors, res.posteriors)
    np.testing.assert_array_equal(back.argmax_calls, res.argmax_calls)
    assert back.mutation_rate == res.mutation_rate
    assert back.root_edge_length == res.root_edge_length
    assert back.tree_newick == res.tree_newick


# 7. branch-length optimisation: per-edge curve == full recomputation,
#    monotone sweeps, and recovery of simulated lengths
from heracles.polish.ancestral import (  # noqa: E402
    _edge_loglik, index_tree, prepare_model, pruning_passes,
    optimize_branch_lengths)


def test_edge_loglik_matches_full_recomputation():
    newick = "((I:0.3,J:0.5):0.4,K:0.8);"
    values = [[1, 0, 2, -1], [1, 2, 0, 0], [0, 2, 1, 1]]
    labels = ["I", "J", "K"]
    eps, delta, root_edge = 0.15, 0.3, 0.2
    tree = _tree(newick)
    cm = _cm(values, labels)
    model = prepare_model(cm, mutation_rate=MU, allele_prior=PRIOR,
                           heritable_eps=eps, stochastic_delta=delta, missing_state=-1)
    nodes, _, row_of = index_tree(tree, model)
    pr = pruning_passes(tree, model, row_of, nodes, root_edge_length=root_edge)
    edge = tree.find_node_with_taxon_label("J")
    parent = edge.parent_node
    above = pr.G[pr.idx_of[id(parent)]].copy()
    for other in parent.child_nodes():
        if other is not edge:
            above *= pr.M[id(other)]
    F_child = pr.F[pr.idx_of[id(edge)]]
    curve = [_edge_loglik(t, above, F_child, model) for t in (0.1, 0.5, 1.3)]
    full = []
    for t in (0.1, 0.5, 1.3):
        edge.edge_length = t
        full.append(infer_ancestral_states(_tree(tree.as_string(schema="newick")),
                                           cm, mutation_rate=MU, allele_prior=PRIOR,
                                           heritable_eps=eps, stochastic_delta=delta,
                                           root_edge_length=root_edge).loglik)
    diffs = np.array(curve) - np.array(full)
    assert np.allclose(diffs, diffs[0], atol=1e-9)     # equal up to a constant


def test_optimize_branch_lengths_monotone_and_recovers():
    rng = np.random.default_rng(3)
    newick = ("(((a:0.3,b:0.3):0.4,(c:0.2,d:0.2):0.5):0.6,"
              "((e:0.4,f:0.4):0.3,(g:0.1,h:0.1):0.6):0.7);")
    labels = list("abcdefgh")
    truth = _tree(newick)
    mu, eps, delta = 1.0, 0.05, 0.1
    n_alleles, n_sites = 4, 3000
    prior = np.ones(n_alleles + 1) / n_alleles
    prior[0] = 1.0
    state = {}
    for node in truth.preorder_node_iter():
        t = 0.0 if node.parent_node is None else node.edge_length
        parent = np.zeros(n_sites, dtype=np.int64) if node.parent_node is None \
            else state[node.parent_node].copy()
        s = parent.copy()
        deleted = rng.random(n_sites) < 1.0 - np.exp(-eps * t)
        mutate = (parent == 0) & (rng.random(n_sites) < 1.0 - np.exp(-mu * t))
        s[mutate] = rng.integers(1, n_alleles + 1, size=mutate.sum())
        s[deleted | (parent == -1)] = -1
        state[node] = s
    rows = []
    for l in labels:
        s = state[truth.find_node_with_taxon_label(l)].copy()
        s[rng.random(n_sites) < delta] = -1
        rows.append(s)
    cm = _cm(np.array(rows), labels)
    true_len = {frozenset(x.taxon.label for x in n.leaf_iter()): n.edge_length
                for n in truth.preorder_node_iter() if n.parent_node is not None}

    tree = _tree(newick)
    for n in tree.preorder_node_iter():
        if n.parent_node is not None:
            n.edge_length = 1.0                        # bad start
    ll0 = infer_ancestral_states(_tree(tree.as_string(schema="newick")), cm,
                                 mutation_rate=mu, allele_prior=prior,
                                 heritable_eps=eps, stochastic_delta=delta).loglik
    lls = [ll0]
    for _ in range(4):
        ll, root_edge, done = optimize_branch_lengths(
            tree, cm, mutation_rate=mu, allele_prior=prior, heritable_eps=eps,
            stochastic_delta=delta, root_edge_length=0.0, n_sweeps=1,
            optimize_root_edge=False)
        lls.append(ll)
    assert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:]))
    assert lls[-1] > ll0 + 10
    for n in tree.preorder_node_iter():
        if n.parent_node is None:
            continue
        key = frozenset(x.taxon.label for x in n.leaf_iter())
        assert abs(n.edge_length - true_len[key]) < 0.15, (key, n.edge_length, true_len[key])


def test_npz_round_trip_without_posteriors(tmp_path):
    rng = np.random.default_rng(4)
    values = rng.choice([0, 1, 2, -1], size=(4, 5), p=[0.4, 0.25, 0.25, 0.1])
    res = infer_ancestral_states(
        _tree("(((A:0.2,B:0.3):0.2,C:0.6):0.1,D:0.9);"),
        _cm(values, list("ABCD")),
        mutation_rate=MU, allele_prior=PRIOR,
        heritable_eps=0.1, stochastic_delta=0.2, root_edge_length=0.05,
        posteriors=False)
    path = tmp_path / "anc_nopost.npz"
    save_posteriors_npz(path, res)
    back = load_posteriors_npz(path)
    assert back.posteriors is None and back.del_posterior is None
    assert back.loglik == res.loglik
    assert back.node_labels == res.node_labels
