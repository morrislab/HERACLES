"""Tests for the shared CTMC kernels (``heracles.embedding.ctmc``).

Run with:

    .venv/bin/python -m pytest tests/test_ctmc.py -q

Checked, in order of what would silently corrupt a result if wrong:

  1. the triple joint and the root-leaf marginal are probability distributions
     (they sum to 1 over all patterns), with and without missing data;
  2. per-site dropout rates act site by site, deletions on a shared branch
     correlate missingness, and the kernels reduce to the right special cases;
  3. the branch lengths read off an exactly additive configuration are the true
     ones, with the right cherry, and stay non-negative on arbitrary geometry;
  4. ``sample_distinct`` draws distinct indices and rejects impossible requests.
"""

import itertools

import numpy as np
import pytest
import torch
import geoopt

from heracles.embedding.ctmc import (
    leaf_marginal_logprob,
    triple_branch_lengths,
    triple_site_logprob,
    sample_distinct,
)


torch.set_default_dtype(torch.float64)


# normalisation and special cases
@pytest.mark.parametrize("branches", [
    (0.3, 0.2, 0.5, 0.4, 0.9),        # generic
    (1e-8, 1e-8, 1e-8, 1e-8, 1e-8),   # degenerate star (everything ~unedited)
    (3.0, 2.0, 2.5, 2.5, 4.0),        # near-saturated
    (0.0, 0.7, 0.1, 0.2, 0.3),        # zero root branch (u always unedited)
])
def test_triple_normalisation(branches):
    """Sum over every (s_i, s_j, s_k) pattern is 1, so the joint is a real likelihood.

    Enumerating all (K+1)^3 patterns as a synthetic 'character matrix' (one site per
    pattern) lets the vectorised code score them all at once, exactly as it would real
    data.
    """
    K = 3
    q = np.array([1.0] + [1.0 / K] * K)                  # q[0] is the unedited placeholder
    patterns = list(itertools.product(range(K + 1), repeat=3))
    # taxa: 0 = root (unused here), 1..3 = the three leaves; one site per pattern.
    states = np.zeros((4, len(patterns)), dtype=np.int64)
    for s, pat in enumerate(patterns):
        states[1:, s] = pat
    st = torch.as_tensor(states, dtype=torch.long)

    a, b, c, d, e = (torch.tensor([float(x)]) for x in branches)
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logP = triple_site_logprob(st, idx, a, b, c, d, e, torch.tensor(1.3),
                               torch.as_tensor(q))
    assert float(torch.exp(logP).sum()) == pytest.approx(1.0, abs=1e-10)


def test_leaf_marginal_normalisation():
    """The root-leaf marginal term is also a distribution over the K+1 states."""
    K = 4
    q = np.array([1.0] + [1.0 / K] * K)
    states = np.zeros((2, K + 1), dtype=np.int64)
    states[1] = np.arange(K + 1)
    logP = leaf_marginal_logprob(torch.as_tensor(states, dtype=torch.long),
                                 torch.tensor([0.0, 0.8]), torch.tensor(1.1),
                                 torch.as_tensor(q), torch.tensor([1]))
    assert float(torch.exp(logP).sum()) == pytest.approx(1.0, abs=1e-12)


def _enumerate_patterns(K, with_missing):
    """(states matrix, pattern list) with one site per possible (s_p, s_q, s_r) pattern."""
    vals = ([-1] if with_missing else []) + list(range(K + 1))
    patterns = list(itertools.product(vals, repeat=3))
    states = np.zeros((4, len(patterns)), dtype=np.int64)
    for s, pat in enumerate(patterns):
        states[1:, s] = pat
    return torch.as_tensor(states, dtype=torch.long), patterns


@pytest.mark.parametrize("eps,delta", [(0.3, 0.0), (0.0, 0.2), (0.25, 0.15), (1.5, 0.4)])
@pytest.mark.parametrize("branches", [(0.3, 0.2, 0.5, 0.4, 0.9), (2.0, 1.5, 1.0, 2.5, 0.7)])
def test_triple_missing_normalisation(eps, delta, branches):
    """With deletion and/or dropout on, the observed-pattern probabilities still sum to 1.

    The sum runs over every pattern in ({missing} + {unedited} + K alleles)^3, which is the
    full observable space, so this catches a double-counted or dropped configuration in
    either the u-unedited or the u-edited branch.
    """
    K = 3
    q = np.array([1.0] + [1.0 / K] * K)
    st, _ = _enumerate_patterns(K, with_missing=True)
    a, b, c, d, e = (torch.tensor([float(x)]) for x in branches)
    logP = triple_site_logprob(st, torch.tensor([[1, 2, 3]]), a, b, c, d, e,
                               torch.tensor(1.3), torch.as_tensor(q),
                               heritable_eps=torch.tensor(eps), stochastic_delta=torch.tensor(delta))
    assert float(torch.exp(logP).sum()) == pytest.approx(1.0, abs=1e-10)


def test_per_site_delta_matches_scalar_column_by_column():
    """An (S,) dropout vector must act on each site exactly like a scalar run with
    that site's rate: column s of the vector run equals column s of the scalar run
    at delta = delta[s]. Checks the triple kernel and the root-leaf term."""
    K = 3
    q = np.array([1.0] + [1.0 / K] * K)
    st, _ = _enumerate_patterns(K, with_missing=True)
    S = st.shape[1]
    a, b, c, d, e = (torch.tensor([x]) for x in (0.3, 0.2, 0.5, 0.4, 0.9))
    rates = (0.05, 0.35)
    delta_vec = torch.tensor([rates[s % 2] for s in range(S)], dtype=torch.float64)
    kw = dict(heritable_eps=torch.tensor(0.2))
    vec = triple_site_logprob(st, torch.tensor([[1, 2, 3]]), a, b, c, d, e,
                              torch.tensor(1.3), torch.as_tensor(q),
                              stochastic_delta=delta_vec, **kw)
    assert vec.shape == (1, S)
    depths = torch.tensor([0.0, 0.8, 0.6, 1.1])
    vec_leaf = leaf_marginal_logprob(st, depths, torch.tensor(1.1), torch.as_tensor(q),
                                     torch.tensor([1, 2, 3]), stochastic_delta=delta_vec, **kw)
    for r in rates:
        scal = triple_site_logprob(st, torch.tensor([[1, 2, 3]]), a, b, c, d, e,
                                   torch.tensor(1.3), torch.as_tensor(q),
                                   stochastic_delta=torch.tensor(r), **kw)
        scal_leaf = leaf_marginal_logprob(st, depths, torch.tensor(1.1), torch.as_tensor(q),
                                          torch.tensor([1, 2, 3]), stochastic_delta=torch.tensor(r), **kw)
        cols = [s for s in range(S) if rates[s % 2] == r]
        assert torch.allclose(vec[:, cols], scal[:, cols], atol=1e-12)
        assert torch.allclose(vec_leaf[:, cols], scal_leaf[:, cols], atol=1e-12)
    # the two rates really differ somewhere (otherwise the check is vacuous)
    assert not torch.allclose(vec[:, 0::2][:, :1], vec[:, 1::2][:, :1])


def test_missingness_is_correlated_through_the_shared_branch():
    """A deletion on the cherry stem b makes the two cherry leaves go missing TOGETHER.

    This is the signal a per-leaf dropout model cannot express: with eps > 0 and delta = 0,
    P(both cherry leaves missing) must exceed P(i missing) * P(j missing), because a single
    deletion event on b blanks both. With eps = 0 and dropout only, the two are independent
    and the same comparison is an equality.
    """
    K, lam = 2, 1.0
    q = np.array([1.0] + [1.0 / K] * K)
    st, patterns = _enumerate_patterns(K, with_missing=True)
    a, b, c, d, e = (torch.tensor([x]) for x in (0.4, 0.9, 0.3, 0.3, 0.5))

    def marginals(eps, delta):
        P = torch.exp(triple_site_logprob(st, torch.tensor([[1, 2, 3]]), a, b, c, d, e,
                                          torch.tensor(lam), torch.as_tensor(q),
                                          heritable_eps=torch.tensor(eps), stochastic_delta=torch.tensor(delta)))[0]
        pi = sum(float(P[s]) for s, p in enumerate(patterns) if p[0] < 0)
        pj = sum(float(P[s]) for s, p in enumerate(patterns) if p[1] < 0)
        both = sum(float(P[s]) for s, p in enumerate(patterns) if p[0] < 0 and p[1] < 0)
        return pi, pj, both

    pi, pj, both = marginals(eps=0.5, delta=0.0)          # heritable deletion only
    assert both > pi * pj + 1e-6, "shared-branch deletion did not correlate missingness"

    pi, pj, both = marginals(eps=0.0, delta=0.3)          # dropout only
    assert both == pytest.approx(pi * pj, abs=1e-10), "dropout should be independent"


def test_leaf_marginal_missing_normalisation():
    """The root-leaf term sums to 1 over {missing} + {unedited} + K alleles."""
    K = 4
    q = np.array([1.0] + [1.0 / K] * K)
    states = np.zeros((2, K + 2), dtype=np.int64)
    states[1] = np.array([-1] + list(range(K + 1)))
    logP = leaf_marginal_logprob(torch.as_tensor(states), torch.tensor([0.0, 0.8]),
                                 torch.tensor(1.1), torch.as_tensor(q), torch.tensor([1]),
                                 heritable_eps=torch.tensor(0.2), stochastic_delta=torch.tensor(0.15))
    assert float(torch.exp(logP).sum()) == pytest.approx(1.0, abs=1e-12)


def test_triple_reduces_to_outgroup_marginal_when_cherry_is_uninformative():
    """With b = c = d = 0 the cherry collapses onto v = u, so leaves i and j copy u.

    Then P(0, 0, s_k) must equal A * L_out(s_k): the only way both cherry leaves are
    unedited is u unedited, after which k evolves alone down e.
    """
    q = np.array([1.0, 0.5, 0.5])
    states = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 2]], dtype=np.int64)
    lam, a_v, e_v = 1.7, 0.6, 0.9
    zero = torch.zeros(1)
    logP = triple_site_logprob(
        torch.as_tensor(states), torch.tensor([[1, 2, 3]]),
        torch.tensor([a_v]), zero, zero, zero, torch.tensor([e_v]),
        torch.tensor(lam), torch.as_tensor(q))
    A, E = np.exp(-lam * a_v), np.exp(-lam * e_v)
    expected = np.array([A * E, A * (1 - E) * 0.5, A * (1 - E) * 0.5])
    np.testing.assert_allclose(torch.exp(logP)[0].numpy(), expected, rtol=1e-12)


# branch lengths
def test_branch_lengths_exact_on_additive_tree():
    """On an exact 4-taxon tree metric the five branch lengths are recovered exactly.

    Tree: o -> u (0.5), u -> v (0.3), v -> i (0.2), v -> j (0.4), u -> k (1.1).
    The cherry must come out as (i, j) whatever order the triple is given in, and the
    branch lengths must match regardless of that ordering.
    """
    A, Bl, C, D, E = 0.5, 0.3, 0.2, 0.4, 1.1
    d_oi, d_oj, d_ok = A + Bl + C, A + Bl + D, A + E
    d_ij, d_ik, d_jk = C + D, Bl + C + E, Bl + D + E
    depths = torch.tensor([0.0, d_oi, d_oj, d_ok])

    for order in itertools.permutations([1, 2, 3]):
        triples = torch.tensor([list(order)], dtype=torch.long)
        pair = {(1, 2): d_ij, (2, 1): d_ij, (1, 3): d_ik, (3, 1): d_ik,
                (2, 3): d_jk, (3, 2): d_jk}
        d01 = torch.tensor([pair[(order[0], order[1])]])
        d02 = torch.tensor([pair[(order[0], order[2])]])
        d12 = torch.tensor([pair[(order[1], order[2])]])
        idx, a, b, c, d, e = triple_branch_lengths(depths, d01, d02, d12, triples)

        assert set(idx[0, :2].tolist()) == {1, 2}, "cherry must be the (i, j) pair"
        assert int(idx[0, 2]) == 3, "outgroup must be k"
        assert float(a) == pytest.approx(A, abs=1e-12)
        assert float(b) == pytest.approx(Bl, abs=1e-12)
        assert float(e) == pytest.approx(E, abs=1e-12)
        # c and d follow the cherry ordering the code chose.
        got = {int(idx[0, 0]): float(c), int(idx[0, 1]): float(d)}
        assert got[1] == pytest.approx(C, abs=1e-12)
        assert got[2] == pytest.approx(D, abs=1e-12)


@pytest.mark.parametrize("branches", [(0.3, 0.2, 0.5, 0.4, 0.9), (1.1, 0.0, 0.6, 0.6, 1.4)])
def test_triple_missing_reduces_to_complete_data(branches):
    """At eps = delta = 0 the missing-aware code returns the complete-data joint exactly.

    Checked pattern by pattern on complete data, and separately that every pattern
    containing a -1 then has probability 0, which is why the fit refuses missing
    data at those rates instead of silently returning -inf.
    """
    K = 3
    q = np.array([1.0] + [1.0 / K] * K)
    a, b, c, d, e = (torch.tensor([float(x)]) for x in branches)
    args = (torch.tensor([[1, 2, 3]]), a, b, c, d, e, torch.tensor(1.3), torch.as_tensor(q))

    st_full, _ = _enumerate_patterns(K, with_missing=False)
    off = triple_site_logprob(st_full, *args)
    on = triple_site_logprob(st_full, *args, heritable_eps=torch.tensor(0.0), stochastic_delta=torch.tensor(0.0))
    np.testing.assert_allclose(off.numpy(), on.numpy(), rtol=0, atol=0)

    st_miss, patterns = _enumerate_patterns(K, with_missing=True)
    P = torch.exp(triple_site_logprob(st_miss, *args, heritable_eps=torch.tensor(0.0),
                                      stochastic_delta=torch.tensor(0.0)))[0]
    has_missing = np.array([any(v < 0 for v in pat) for pat in patterns])
    assert float(P[has_missing].max()) < 1e-11


def test_branch_lengths_are_non_negative_on_random_geometry():
    """Random (non-tree-like) hyperbolic points still give five non-negative branches."""
    manifold = geoopt.Lorentz(k=1.0)
    torch.manual_seed(3)
    pts = manifold.expmap0(torch.cat(
        [torch.zeros(30, 1), torch.randn(30, 6) * 0.7], dim=1))
    depths = manifold.dist(pts[0].unsqueeze(0), pts)
    triples = sample_distinct(29, 3, 200, torch.Generator().manual_seed(0), None)
    p0, p1, p2 = pts[triples[:, 0]], pts[triples[:, 1]], pts[triples[:, 2]]
    _, a, b, c, d, e = triple_branch_lengths(
        depths, manifold.dist(p0, p1), manifold.dist(p0, p2), manifold.dist(p1, p2),
        triples)
    for name, t in zip("abcde", (a, b, c, d, e)):
        assert bool((t >= 0).all()), "branch %s went negative" % name
        assert bool(torch.isfinite(t).all())


def test_sample_distinct_exact_when_k_equals_n():
    from heracles.embedding.ctmc import sample_distinct
    g = torch.Generator().manual_seed(0)
    t = sample_distinct(n=4, k=4, count=200, device=None, generator=g)
    assert t.shape == (200, 4)
    for row in t.tolist():
        assert sorted(row) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        sample_distinct(n=3, k=4, count=1, device=None, generator=g)
