"""Tests for the quartet composite likelihood (``heracles.embedding.objective``).

Run with:

    .venv/bin/python -m pytest tests/test_quartet_embedding.py -q

Tests:
  1. normalisation: balanced and caterpillar kernels sum to 1 over all patterns
  2. shape detection: balanced vs caterpillar correctly identified
  3. branch lengths on additive trees: recovered exactly
  4. cherry and sub-triple kernels: pair_edit_joint is a valid probability,
     triple_site_logprob sums to 1
  5. gradient flow: the quartet objective has finite, non-zero gradients
  6. end to end: embed() reduces the loss; Heracles().fit(cm) produces trees
  7. quartet samplers: the neighbourhood sampler is the default and its
     settings are recorded; a custom sampler can replace it
"""

import itertools

import numpy as np
import pytest
import torch

from heracles.fit import build_allele_priors
from heracles.embedding.initialization import lorentz_manifold, init_points
from heracles.embedding.ctmc import (
    gromov_quartet_penalty_from_distances,
    pair_edit_joint,
    triple_site_logprob,
)
from heracles.embedding.objective import (quartet_loss_fn, quartet_branch_lengths,
                                          quartet_site_logprob, _quartet_site_logprob_balanced,
                                          _quartet_site_logprob_caterpillar)
from heracles.embedding.sampler import _stratified_quartets

from conftest import _toy_states, _hamming

torch.set_default_dtype(torch.float64)


# reference objective assembled from the public kernels
def _quartet_joint_logprob(states, points, manifold, mutation_rate, allele_prior, quartets,
                          radius=1.0, root=0, heritable_eps=0.0, stochastic_delta=0.0):
    """Mean negative log P over sampled quartets and characters."""
    st = torch.as_tensor(states, dtype=torch.long)
    allele_prior_t = torch.as_tensor(allele_prior, dtype=points.dtype)
    mutation_rate_t = torch.as_tensor(mutation_rate, dtype=points.dtype) if not torch.is_tensor(mutation_rate) else mutation_rate
    quartets = torch.as_tensor(quartets, dtype=torch.long)

    depths = radius * manifold.dist(points[root].unsqueeze(0), points)
    p0 = points[quartets[:, 0]]
    p1 = points[quartets[:, 1]]
    p2 = points[quartets[:, 2]]
    p3 = points[quartets[:, 3]]
    d01 = radius * manifold.dist(p0, p1)
    d02 = radius * manifold.dist(p0, p2)
    d03 = radius * manifold.dist(p0, p3)
    d12 = radius * manifold.dist(p1, p2)
    d13 = radius * manifold.dist(p1, p3)
    d23 = radius * manifold.dist(p2, p3)

    result = quartet_branch_lengths(depths, (d01, d02, d03, d12, d13, d23),
                                    quartets)
    (is_balanced,
     idx_bal, a_bal, b1_bal, c1_bal, d1_bal, b2_bal, c2_bal, d2_bal,
     outgroup_idx, sub_triple_idx, a_cat, c_out_cat,
     root_u_sub, u_v_sub, v_cherry1_sub, v_cherry2_sub, u_outgroup_sub) = result

    logP = quartet_site_logprob(
        st, is_balanced,
        idx_bal, a_bal, b1_bal, c1_bal, d1_bal, b2_bal, c2_bal, d2_bal,
        outgroup_idx, sub_triple_idx, a_cat, c_out_cat,
        root_u_sub, u_v_sub, v_cherry1_sub, v_cherry2_sub, u_outgroup_sub,
        mutation_rate_t, allele_prior_t, heritable_eps=heritable_eps, stochastic_delta=stochastic_delta)
    return -logP.mean()


def _balanced_tree_metric():
    """5-taxon tree metric (root + 4 leaves) with balanced topology ((1,2),(3,4)).

    Tree: o --0.3-- w --0.2-- u --0.1-- leaf1
                      |         `--0.15-- leaf2
                      `--0.25-- v --0.12-- leaf3
                                  `--0.08-- leaf4
    """
    a = 0.3    # o -> w
    b1 = 0.2   # w -> u
    c1 = 0.1   # u -> leaf1
    d1 = 0.15  # u -> leaf2
    b2 = 0.25  # w -> v
    c2 = 0.12  # v -> leaf3
    d2 = 0.08  # v -> leaf4

    depths = torch.tensor([0.0, a + b1 + c1, a + b1 + d1, a + b2 + c2, a + b2 + d2])
    dists = {}
    dists[(1, 2)] = c1 + d1
    dists[(3, 4)] = c2 + d2
    dists[(1, 3)] = c1 + b1 + b2 + c2
    dists[(1, 4)] = c1 + b1 + b2 + d2
    dists[(2, 3)] = d1 + b1 + b2 + c2
    dists[(2, 4)] = d1 + b1 + b2 + d2
    return depths, dists, (a, b1, c1, d1, b2, c2, d2)


def _caterpillar_tree_metric():
    """5-taxon tree metric (root + 4 leaves) with caterpillar topology (1,(2,(3,4))).

    Tree: o --0.2-- w1 --0.1-- leaf1
                      `--0.15-- w2 --0.12-- leaf2
                                   `--0.18-- w3 --0.08-- leaf3
                                                `--0.06-- leaf4
    """
    a = 0.2      # o -> w1
    c_out = 0.1  # w1 -> leaf1 (outgroup)
    b_top = 0.15 # w1 -> w2
    c_mid = 0.12 # w2 -> leaf2
    b_bot = 0.18 # w2 -> w3
    c3 = 0.08    # w3 -> leaf3
    c4 = 0.06    # w3 -> leaf4

    depths = torch.tensor([
        0.0,
        a + c_out,                      # leaf1
        a + b_top + c_mid,              # leaf2
        a + b_top + b_bot + c3,         # leaf3
        a + b_top + b_bot + c4,         # leaf4
    ])
    dists = {}
    dists[(1, 2)] = c_out + b_top + c_mid
    dists[(1, 3)] = c_out + b_top + b_bot + c3
    dists[(1, 4)] = c_out + b_top + b_bot + c4
    dists[(2, 3)] = c_mid + b_bot + c3
    dists[(2, 4)] = c_mid + b_bot + c4
    dists[(3, 4)] = c3 + c4
    return depths, dists, (a, c_out, b_top, c_mid, b_bot, c3, c4)


def _make_dists_6(dists_dict, quartet):
    """Extract the 6 pairwise distances for a quartet from a distance dict."""
    i, j, k, l = quartet
    pairs = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
    return tuple(
        torch.tensor([dists_dict[tuple(sorted(p))]]) for p in pairs
    )


# 1. Normalisation: balanced
class TestNormalisationBalanced:
    def test_complete_data_sums_to_one(self):
        """Balanced quartet logprob sums to 1 over all (K+1)^4 patterns."""
        K = 3
        a, b1, c1, d1, b2, c2, d2 = 0.3, 0.2, 0.1, 0.15, 0.25, 0.12, 0.08
        lam = torch.tensor(1.5)
        ql = torch.ones(K + 1) / K
        ql[0] = 1.0

        patterns = list(itertools.product(range(K + 1), repeat=4))
        N = len(patterns)
        # one quartet, one site per pattern: root row (all 0) + 4 leaf rows
        st = torch.zeros(5, N, dtype=torch.long)
        for pi, pat in enumerate(patterns):
            for li in range(4):
                st[li + 1, pi] = pat[li]
        idx = torch.tensor([[1, 2, 3, 4]])
        logP = _quartet_site_logprob_balanced(
            st, idx,
            torch.tensor([a]), torch.tensor([b1]),
            torch.tensor([c1]), torch.tensor([d1]),
            torch.tensor([b2]), torch.tensor([c2]),
            torch.tensor([d2]),
            lam, ql)
        total = logP.exp().sum(dim=1)  # sum over all sites (= all patterns)
        assert total.shape == (1,)
        np.testing.assert_allclose(total.item(), 1.0, atol=1e-10)

    def test_missing_data_sums_to_one(self):
        """Balanced quartet with missing data sums to 1 over observable patterns."""
        K = 3
        a, b1, c1, d1, b2, c2, d2 = 0.3, 0.2, 0.1, 0.15, 0.25, 0.12, 0.08
        lam = torch.tensor(1.5)
        ql = torch.ones(K + 1) / K
        ql[0] = 1.0
        eps = torch.tensor(0.1)
        delta = torch.tensor(0.05)

        # All observable patterns: each leaf is 0..K or -1 (missing)
        vals = list(range(K + 1)) + [-1]
        patterns = list(itertools.product(vals, repeat=4))
        N = len(patterns)
        st = torch.zeros(5, N, dtype=torch.long)
        for pi, pat in enumerate(patterns):
            for li in range(4):
                st[li + 1, pi] = pat[li]

        idx = torch.tensor([[1, 2, 3, 4]])
        logP = _quartet_site_logprob_balanced(
            st, idx,
            torch.tensor([a]), torch.tensor([b1]),
            torch.tensor([c1]), torch.tensor([d1]),
            torch.tensor([b2]), torch.tensor([c2]),
            torch.tensor([d2]),
            lam, ql, heritable_eps=eps, stochastic_delta=delta)
        total = logP.exp().sum(dim=1)
        np.testing.assert_allclose(total.item(), 1.0, atol=1e-10)


# 1. Normalisation: caterpillar
class TestNormalisationCaterpillar:
    def test_complete_data_sums_to_one(self):
        K = 3
        lam = torch.tensor(1.5)
        ql = torch.ones(K + 1) / K
        ql[0] = 1.0

        # Caterpillar: outgroup=leaf1, sub-triple=(leaf2,leaf3,leaf4)
        a_cat = torch.tensor([0.2])
        c_out = torch.tensor([0.1])
        # Sub-triple branch lengths (cherry=(3,4), outgroup=2)
        a_sub = torch.tensor([0.15])  # w1->w2
        b_sub = torch.tensor([0.18])  # w2->w3
        c_sub = torch.tensor([0.08])  # w3->leaf3
        d_sub = torch.tensor([0.06])  # w3->leaf4
        e_sub = torch.tensor([0.12])  # w2->leaf2

        patterns = list(itertools.product(range(K + 1), repeat=4))
        N = len(patterns)
        st = torch.zeros(5, N, dtype=torch.long)
        for pi, pat in enumerate(patterns):
            st[1, pi] = pat[0]  # outgroup leaf
            st[2, pi] = pat[1]  # sub-triple leaf (outgroup of sub-triple)
            st[3, pi] = pat[2]  # sub-triple cherry leaf 1
            st[4, pi] = pat[3]  # sub-triple cherry leaf 2

        outgroup_idx = torch.tensor([[1]])  # leaf1
        sub_triple_idx = torch.tensor([[3, 4, 2]])  # cherry=(3,4), outgroup=2

        logP = _quartet_site_logprob_caterpillar(
            st, outgroup_idx, sub_triple_idx,
            a_cat, c_out, a_sub, b_sub, c_sub, d_sub, e_sub,
            lam, ql)
        total = logP.exp().sum(dim=1)
        np.testing.assert_allclose(total.item(), 1.0, atol=1e-10)

    def test_missing_data_sums_to_one(self):
        K = 3
        lam = torch.tensor(1.5)
        ql = torch.ones(K + 1) / K
        ql[0] = 1.0
        eps = torch.tensor(0.1)
        delta = torch.tensor(0.05)

        a_cat = torch.tensor([0.2])
        c_out = torch.tensor([0.1])
        a_sub = torch.tensor([0.15])
        b_sub = torch.tensor([0.18])
        c_sub = torch.tensor([0.08])
        d_sub = torch.tensor([0.06])
        e_sub = torch.tensor([0.12])

        vals = list(range(K + 1)) + [-1]
        patterns = list(itertools.product(vals, repeat=4))
        N = len(patterns)
        st = torch.zeros(5, N, dtype=torch.long)
        for pi, pat in enumerate(patterns):
            st[1, pi] = pat[0]
            st[2, pi] = pat[1]
            st[3, pi] = pat[2]
            st[4, pi] = pat[3]

        outgroup_idx = torch.tensor([[1]])
        sub_triple_idx = torch.tensor([[3, 4, 2]])

        logP = _quartet_site_logprob_caterpillar(
            st, outgroup_idx, sub_triple_idx,
            a_cat, c_out, a_sub, b_sub, c_sub, d_sub, e_sub,
            lam, ql, heritable_eps=eps, stochastic_delta=delta)
        total = logP.exp().sum(dim=1)
        np.testing.assert_allclose(total.item(), 1.0, atol=1e-10)


# 2. Shape detection
class TestShapeDetection:
    def test_balanced_detected(self):
        depths, dists, _ = _balanced_tree_metric()
        quartets = torch.tensor([[1, 2, 3, 4]])
        dists_6 = _make_dists_6(dists, [1, 2, 3, 4])
        result = quartet_branch_lengths(depths, dists_6, quartets)
        is_balanced = result[0]
        assert bool(is_balanced[0]), "balanced tree should be detected as balanced"

    def test_caterpillar_detected(self):
        depths, dists, _ = _caterpillar_tree_metric()
        quartets = torch.tensor([[1, 2, 3, 4]])
        dists_6 = _make_dists_6(dists, [1, 2, 3, 4])
        result = quartet_branch_lengths(depths, dists_6, quartets)
        is_balanced = result[0]
        assert not bool(is_balanced[0]), "caterpillar tree should be detected as caterpillar"


# 3. Branch lengths on additive trees
class TestBranchLengths:
    def test_balanced_branch_lengths(self):
        depths, dists, true_bl = _balanced_tree_metric()
        a, b1, c1, d1, b2, c2, d2 = true_bl
        quartets = torch.tensor([[1, 2, 3, 4]])
        dists_6 = _make_dists_6(dists, [1, 2, 3, 4])
        result = quartet_branch_lengths(depths, dists_6, quartets)
        (is_balanced, idx_bal,
         a_r, b1_r, c1_r, d1_r, b2_r, c2_r, d2_r,
         *_rest) = result
        assert bool(is_balanced[0])
        # The recovered idx should pair (1,2) and (3,4)
        left = set(idx_bal[0, :2].tolist())
        right = set(idx_bal[0, 2:].tolist())
        assert left == {1, 2} or left == {3, 4}
        if left == {1, 2}:
            np.testing.assert_allclose(a_r.item(), a, atol=1e-10)
            np.testing.assert_allclose(b1_r.item(), b1, atol=1e-10)
            np.testing.assert_allclose(b2_r.item(), b2, atol=1e-10)
        else:
            np.testing.assert_allclose(a_r.item(), a, atol=1e-10)
            np.testing.assert_allclose(b1_r.item(), b2, atol=1e-10)
            np.testing.assert_allclose(b2_r.item(), b1, atol=1e-10)

    def test_caterpillar_branch_lengths(self):
        depths, dists, true_bl = _caterpillar_tree_metric()
        a, c_out, b_top, c_mid, b_bot, c3, c4 = true_bl
        quartets = torch.tensor([[1, 2, 3, 4]])
        dists_6 = _make_dists_6(dists, [1, 2, 3, 4])
        result = quartet_branch_lengths(depths, dists_6, quartets)
        (is_balanced, _idx_bal, *_bal_bl,
         outgroup_idx, sub_triple_idx, a_cat, c_out_cat,
         a_sub, b_sub, c_sub, d_sub, e_sub) = result
        assert not bool(is_balanced[0])
        # Outgroup should be leaf1
        assert outgroup_idx[0, 0].item() == 1
        np.testing.assert_allclose(a_cat.item(), a, atol=1e-10)
        np.testing.assert_allclose(c_out_cat.item(), c_out, atol=1e-10)
        # Sub-triple: cherry should be (3,4), outgroup should be 2
        cherry_set = set(sub_triple_idx[0, :2].tolist())
        assert cherry_set == {3, 4}
        assert sub_triple_idx[0, 2].item() == 2
        np.testing.assert_allclose(a_sub.item(), b_top, atol=1e-10)
        np.testing.assert_allclose(b_sub.item(), b_bot, atol=1e-10)
        np.testing.assert_allclose(e_sub.item(), c_mid, atol=1e-10)


# 4. Cherry and sub-triple kernels
def test_pair_edit_joint_is_a_probability_for_every_cherry_pattern():
    """pair_edit_joint returns one non-negative probability for each cherry pattern."""
    K = 3
    lam = torch.tensor(1.5)
    ql = torch.ones(K + 1) / K
    ql[0] = 1.0

    b1, c1, d1 = 0.2, 0.1, 0.15
    eB = torch.exp(-lam * b1).unsqueeze(0)
    eC = torch.exp(-lam * c1).unsqueeze(0)
    eD = torch.exp(-lam * d1).unsqueeze(0)

    for si, sj in itertools.product(range(K + 1), repeat=2):
        st = torch.zeros(3, 1, dtype=torch.long)
        st[1, 0] = si
        st[2, 0] = sj
        SI, SJ = st[1:2], st[2:3]
        qi = ql[SI.clamp_min(0)]
        qj = ql[SJ.clamp_min(0)]
        ff = lambda m: m.to(torch.float64)
        both0 = ff((SI == 0) & (SJ == 0))
        i0 = ff((SI == 0) & (SJ > 0))
        j0 = ff((SI > 0) & (SJ == 0))
        same = ff((SI > 0) & (SI == SJ))
        diff = ff((SI > 0) & (SJ > 0) & (SI != SJ))
        P_pair = pair_edit_joint(eB, eC, eD, qi, qj, both0, i0, j0, same, diff)
        assert P_pair.shape == (1, 1)
        assert P_pair.item() >= 0


def test_caterpillar_sub_triple_kernel_sums_to_one():
    """triple_site_logprob on the caterpillar's sub-triple sums to 1 over all patterns."""
    K = 3
    lam = torch.tensor(1.5)
    ql = torch.ones(K + 1) / K
    ql[0] = 1.0
    a_sub, b_sub = 0.15, 0.18
    c_sub, d_sub, e_sub = 0.08, 0.06, 0.12

    # Build states for just the sub-triple (3 leaves)
    patterns_3 = list(itertools.product(range(K + 1), repeat=3))
    N = len(patterns_3)
    st = torch.zeros(5, N, dtype=torch.long)
    for pi, pat in enumerate(patterns_3):
        st[2, pi] = pat[0]  # sub-triple outgroup (leaf2)
        st[3, pi] = pat[1]  # cherry leaf1 (leaf3)
        st[4, pi] = pat[2]  # cherry leaf2 (leaf4)

    sub_idx = torch.tensor([[3, 4, 2]])  # cherry=(3,4), outgroup=2
    logP = triple_site_logprob(
        st, sub_idx,
        torch.tensor([a_sub]), torch.tensor([b_sub]),
        torch.tensor([c_sub]), torch.tensor([d_sub]),
        torch.tensor([e_sub]),
        lam, ql)
    total = logP.exp().sum(dim=1)
    np.testing.assert_allclose(total.item(), 1.0, atol=1e-10)


# 5. Gradient flow
def test_gradient_flows_through_quartet_objective():
    states = _toy_states(n=8, S=4, K=3)
    ql = build_allele_priors(states, prior="global_uniform")
    dim = 3
    manifold = lorentz_manifold(dim)
    est = _hamming(states)
    torch.manual_seed(42)
    pts, _ = init_points(manifold, dim, seed_dists=est, init_method="mds")
    pts = pts.clone().requires_grad_(True)
    lam = torch.tensor(1.5)
    quartets = torch.tensor([[1, 2, 3, 4], [2, 3, 5, 7], [1, 4, 6, 8]])
    loss = _quartet_joint_logprob(states, pts, manifold, lam, ql, quartets)
    loss.backward()
    assert pts.grad is not None
    assert not torch.isnan(pts.grad).any()
    assert pts.grad.abs().sum() > 0


# 6. End to end
def test_embed_runs_and_reduces_loss():
    import pandas as pd
    from heracles.fit import embed
    states = _toy_states(n=10, S=8, K=4)
    cm = pd.DataFrame(states[1:], index=["c%d" % i for i in range(10)])
    emb = embed(cm, dimension=3, max_iterations=50, seed=0, init_method="mds",
                heritable_eps=1e-5, stochastic_delta=0.1,
                allele_prior="global_uniform")
    leaf_dists, meta = emb["leaf_dists"], emb["meta"]
    assert leaf_dists.shape == (10, 10)
    assert meta["objective"] == "quartet"
    hist = meta["history"]["loss"]
    assert hist[-1] < hist[0], "loss should decrease"


def test_quartet_loss_exposes_gromov_penalty_from_its_likelihood_batch():
    """The quartet treelike penalty reuses, rather than resamples, its batch."""
    states = _toy_states(n=8, S=5, K=3)
    ql = build_allele_priors(states, prior="global_uniform")
    manifold = lorentz_manifold(3)
    points, _ = init_points(manifold, 3, seed_dists=_hamming(states), init_method="mds")
    loss_fn = quartet_loss_fn(states, ql, states.shape[0], batch_size=12,
                               sampler=_stratified_quartets)

    # Recreate the batch the loss function will draw from this seed.
    seed = 17
    quartets = _stratified_quartets(8, 12,
                                    generator=torch.Generator().manual_seed(seed))
    p0, p1, p2, p3 = (points[quartets[:, col]] for col in range(4))
    expected = gromov_quartet_penalty_from_distances(
        manifold.dist(p0, p1), manifold.dist(p0, p2), manifold.dist(p0, p3),
        manifold.dist(p1, p2), manifold.dist(p1, p3), manifold.dist(p2, p3))

    loss, _ = loss_fn(points, manifold, torch.tensor(1.0), torch.tensor(1.0),
                      torch.tensor(0.0), torch.tensor(0.0),
                      torch.Generator().manual_seed(seed))
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss_fn.batch_gromov_penalty, expected)


def test_heracles_quartet_integration(tmp_path):
    import pandas as pd
    from heracles import Heracles
    rng = np.random.default_rng(42)
    n, S, K = 12, 10, 4
    cm = pd.DataFrame(
        rng.integers(0, K + 1, size=(n, S)),
        index=[f"cell_{i}" for i in range(n)],
        columns=[f"site_{j}" for j in range(S)])
    cm.iloc[rng.random(cm.shape) < 0.1] = -1

    model = Heracles(
        readouts=["nj", "upgma"],
        dimension=3, max_iterations=50, seed=0,
        heritable_eps=1e-5, stochastic_delta=0.1,
        init_method="mds", allele_prior="global_uniform",
        outdir=tmp_path)
    model.fit(cm)
    assert model.trees_ is not None
    assert "nj_binary" in model.trees_
    assert "upgma_binary" in model.trees_


# 7. Quartet samplers

def test_quartet_sampler_is_neighbourhood_by_default_and_overridable():
    """The neighbourhood sampler is the default; a custom sampler is recorded by name."""
    from heracles.embedding.objective import quartet_loss_fn
    # A real matrix, not an all-zero one: the neighbourhood sampler needs a tau with
    # some structure, and an all-unedited matrix has none by construction.
    states = _toy_states(n=12, S=8, K=4, seed=0)
    est = _hamming(states)
    prior = np.ones(5) / 5.0
    m = states.shape[0]
    fn = quartet_loss_fn(states, prior, m, est_dists=est)
    assert fn.quartet_sampler == "neighbourhood" and fn.quartet_sampler_config is not None
    assert fn.meta["quartet_sampler"] == "neighbourhood"
    from heracles.config import HeraclesConfig
    fn_u = quartet_loss_fn(states, prior, m, config=HeraclesConfig(sampler_strategy="uniform"))
    assert fn_u.quartet_sampler == "uniform" and fn_u.quartet_sampler_config == {"strategy": "uniform"}
    fn2 = quartet_loss_fn(states, prior, m, sampler=_stratified_quartets)
    assert fn2.quartet_sampler == "_stratified_quartets"
    assert fn2.quartet_sampler_config is None


def _tau_setup(n=30, S=20, K=5, seed=0):
    """(order, tau, states, est) for a small non-degenerate character matrix."""
    from heracles.embedding.sampler import build_tau_neighbours
    states = _toy_states(n=n, S=S, K=K, seed=seed)
    est = _hamming(states)
    order, tau = build_tau_neighbours(est)
    return order, tau, states, est


def test_build_tau_neighbours_matches_the_upgma_construction():
    """tau is the same founder-referenced depth matrix the UPGMA read-out builds."""
    order, tau, _, est = _tau_setup(n=12, S=10)
    d0, D = est[0, 1:], est[1:, 1:]
    expected = np.clip(0.5 * (d0[:, None] + d0[None, :] - D), 0.0, None)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(tau, expected)
    assert tau.shape == (12, 12) and order.shape == (12, 11)
    np.testing.assert_allclose(tau, tau.T)


def test_build_tau_neighbours_excludes_each_leaf_from_its_own_ranking():
    """A leaf must never be offered as its own close relative."""
    order, tau, _, _ = _tau_setup(n=20)
    assert bool((order != np.arange(20)[:, None]).all())
    # Ranking is by descending tau.
    ranked = np.take_along_axis(tau, order, axis=1)
    assert bool((np.diff(ranked, axis=1) <= 1e-12).all())


def test_neighbourhood_sampler_requires_a_founder_augmented_est_dists():
    from heracles.embedding.objective import quartet_loss_fn
    from heracles.fit import build_allele_priors
    _, _, states, est = _tau_setup(n=12, S=10)
    ql = build_allele_priors(states[1:], prior="global_uniform")
    m = states.shape[0]
    with pytest.raises(ValueError, match="est_dists"):
        quartet_loss_fn(states, ql, m)
    with pytest.raises(ValueError, match="founder-augmented"):
        quartet_loss_fn(states, ql, m, est_dists=est[1:, 1:])


@pytest.mark.parametrize("strategy", ["neighbourhood", "uniform"])
def test_embed_records_the_sampler_config(strategy):
    import pandas as pd
    from heracles.fit import embed
    _, _, states, est = _tau_setup(n=16, S=12, K=4)
    cm = pd.DataFrame(states[1:], index=["c%d" % i for i in range(16)])
    meta = embed(cm, dimension=3, max_iterations=60, seed=0, init_method="mds",
                 heritable_eps=1e-5, stochastic_delta=0.1, sampler_strategy=strategy,
                 allele_prior="global_uniform")["meta"]
    hist = meta["history"]["loss"]
    assert hist[-1] < hist[0], "loss should decrease"
    assert meta["quartet_sampler"] == strategy
    if strategy == "neighbourhood":
        assert meta["quartet_sampler_config"] == {"strategy": "neighbourhood", "max_k": 1024.0,
                                                  "table_k": 15, "tau_source": "est_dists"}
        stats = meta["quartet_sampler_stats"]
        assert 0.0 < stats["accept_rate"] <= 1.0
        assert stats["fallback_frac"] == 0.0
    else:
        assert meta["quartet_sampler_config"] == {"strategy": "uniform"}
