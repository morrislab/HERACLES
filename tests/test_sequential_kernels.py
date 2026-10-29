"""Tests for the sequential-edit (cassette) likelihood.

Five claims, in dependency order:

1. The chain transition probabilities are right: rows normalise, they match
   ``expm(Q t)``, two short branches compose into one long one, and gradients survive
   ``t == 0`` exactly (branch lengths are ``clamp_min(0)``'d upstream, and
   ``gammainc(1, 0)`` has a NaN gradient if the input is not floored).
2. At K=1 every sequential function reproduces its Cas9 counterpart, including
   the caterpillar quartet (the one the Cas9 code reaches through
   ``triple_site_logprob`` and which the sequential path deliberately does not).
3. At K>1 the joint normalises: summed over every reachable cassette
   configuration, the pair, triple, quartet and leaf marginal each give 1.
   Also with missing data on, and reducing to the complete-data case at
   ``eps = delta = 0``.
4. ``shared_prefix_length`` skips missing cells rather than scoring them as
   disagreements, and ``cassette_seed_distances`` reproduces the Cas9
   weighted-Hamming scheme at K=1.
5. Gradients flow, and the whole thing fits end to end.
"""

import itertools

import numpy as np
import pytest
import torch

from heracles.models import sequential as sc
from heracles.embedding.initialization import seed_distances
from heracles.embedding.objective import (_quartet_site_logprob_balanced,
                                          _quartet_site_logprob_caterpillar)
from heracles.embedding.ctmc import (_make_learnable_prior,
                           leaf_marginal_logprob,
                           triple_site_logprob)

torch.set_default_dtype(torch.float64)

EPS, DELTA = 0.07, 0.11


# helpers

def _prior(n_alleles):
    """Global uniform prior over ``n_alleles`` alleles, with the 1.0 placeholder."""
    p = np.zeros(n_alleles + 1)
    p[0] = 1.0
    p[1:] = 1.0 / n_alleles
    return p


def _configs(K, n_alleles):
    """Every reachable cassette configuration: sum_e A^e rows of width K."""
    rows = []
    for end in range(K + 1):
        for combo in itertools.product(range(1, n_alleles + 1), repeat=end):
            rows.append(list(combo) + [0] * (K - end))
    return np.array(rows, dtype=np.int64)


def _cass(states, K, n_alleles):
    cass, _ = sc.cassette_summaries(torch.as_tensor(states), _prior(n_alleles), K)
    return cass


def _toy_cassettes(n, C, K, n_alleles, seed=0, missing_frac=0.0):
    """Ordered-cassette character matrix with a founder row of zeros."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n + 1, C * K), dtype=np.int64)
    for i in range(1, n + 1):
        for c in range(C):
            end = rng.integers(0, K + 1)
            block = np.zeros(K, dtype=np.int64)
            block[:end] = rng.integers(1, n_alleles + 1, size=end)
            if missing_frac and rng.random() < missing_frac:
                block[:] = -1
            out[i, c * K:(c + 1) * K] = block
    return out


def _branches(B, values):
    return [torch.full((B,), float(v)) for v in values]


def _pair(cass, idx_i, idx_j, branches, rate, eps, delta):
    """Cherry factor for explicit index pairs, one entry per ancestor count."""
    leaf_i, leaf_j = sc._select_leaves(cass, torch.stack([idx_i, idx_j], dim=1))
    return sc._pair_factor(cass.cassette_size, leaf_i, leaf_j, *branches,
                           torch.tensor(rate), eps, delta, "equal")


def test_learnable_prior_reaches_sequential_likelihood():
    """The sequential loss must backpropagate into the shared allele prior."""
    states = np.array([
        [0, 0, 0],
        [1, 2, 0],
        [1, 3, 0],
    ], dtype=np.int64)
    initial = torch.as_tensor(_prior(3))
    get_prior, raw = _make_learnable_prior(initial)
    fixed, _ = sc.cassette_summaries(states, initial, cassette_size=3)
    current = sc.with_allele_prior(fixed, get_prior())
    score = sc.sequential_leaf_marginal_logprob(
        current, torch.tensor([0.0, 0.7, 0.9]), torch.tensor(1.0),
        torch.tensor([1, 2])).sum()
    score.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert float(raw.grad[1:].abs().sum()) > 0


def test_prior_refresh_keeps_static_cassette_data():
    """Refreshing a prior changes only its running log sums."""
    states = np.array([[0, 0], [1, 2]], dtype=np.int64)
    cassettes, _ = sc.cassette_summaries(states, _prior(2), cassette_size=2)
    updated = sc.with_allele_prior(
        cassettes, torch.tensor([1.0, 0.8, 0.2]))
    assert updated.alleles.data_ptr() == cassettes.alleles.data_ptr()
    assert updated.edit_count.data_ptr() == cassettes.edit_count.data_ptr()
    assert torch.allclose(updated.log_prior_upto[1, 0],
                          torch.log(torch.tensor([1.0, 0.8, 0.16])))


# 1. transitions

@pytest.mark.parametrize("K", [1, 2, 5])
@pytest.mark.parametrize("rate", [0.05, 1.0, 10.0])
def test_transition_rows_sum_to_one(K, rate):
    t = torch.tensor([0.0, 1e-9, 0.3, 2.0, 50.0])
    logT = sc.log_transition(torch.tensor(rate), t, K)
    total = logT.exp().sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-12)


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_transition_matches_matrix_exponential(K):
    """The analytic Poisson/gammainc form against expm(Q t), the oracle."""
    t = torch.tensor([1e-6, 0.25, 1.0, 4.0])
    rate = torch.tensor(0.7)
    analytic = sc.log_transition(rate, t, K, rate_model="equal").exp()
    oracle = sc.log_transition(rate, t, K, rate_model="matrix_exp").exp()
    assert torch.allclose(analytic, oracle, atol=1e-10)


def test_transition_chapman_kolmogorov():
    K, rate = 4, 0.9
    t1, t2 = torch.tensor([0.4]), torch.tensor([1.1])
    P1 = sc.log_transition(torch.tensor(rate), t1, K).exp()[0, 0]
    P2 = sc.log_transition(torch.tensor(rate), t2, K).exp()[0, 0]
    P12 = sc.log_transition(torch.tensor(rate), t1 + t2, K).exp()[0, 0]
    assert torch.allclose(P1 @ P2, P12, atol=1e-12)


def test_transition_absorbing_state_is_terminal():
    K = 3
    logT = sc.log_transition(torch.tensor(1.0), torch.tensor([2.0]), K)
    P = logT.exp()[0, 0]
    assert torch.allclose(P[K, K], torch.tensor(1.0), atol=1e-14)
    assert torch.allclose(P[K, :K], torch.zeros(K), atol=1e-14)
    assert torch.allclose(torch.tril(P, diagonal=-1), torch.zeros(K + 1, K + 1),
                          atol=1e-14)


@pytest.mark.parametrize("K", [1, 2, 5])
def test_transition_gradients_finite_at_zero_branch_length(K):
    """t == 0 exactly is common; gammainc(1, 0) has a NaN gradient unfloored."""
    t = torch.zeros(3, requires_grad=True)
    rate = torch.tensor(0.8, requires_grad=True)
    logT = sc.log_transition(rate, t, K)
    # Every finite entry contributes, so any NaN anywhere reaches the grads.
    logT.clamp_min(sc.MIN_LOG_PROBABILITY).sum().backward()
    assert torch.isfinite(t.grad).all()
    assert torch.isfinite(rate.grad).all()


def test_transition_k1_matches_cas9_closed_form():
    t = torch.tensor([0.0, 0.3, 2.0])
    mu = torch.tensor(1.3)
    P = sc.log_transition(mu, t, 1).exp()[:, 0]
    assert torch.allclose(P[:, 0, 0], torch.exp(-mu * t), atol=1e-14)
    assert torch.allclose(P[:, 0, 1], 1 - torch.exp(-mu * t), atol=1e-14)
    assert torch.allclose(P[:, 1, 1], torch.ones(3), atol=1e-14)


# 2. K=1 equivalence with the Cas9 code

def _k1_setup(eps, n_alleles=3, n_sites=6, seed=1):
    """A K=1 character matrix exercising every observation pattern.

    Missing observations are only included when the deletion/dropout rates are
    non-zero. At ``eps = delta = 0`` a ``-1`` is an impossible outcome, so the
    two implementations would be compared on patterns of probability zero,
    where they differ only in how they floor it (the Cas9 path clamps at
    ``LOG_FLOOR = 1e-12``, i.e. -27.63; this one reports the true value near
    -700). ``optimize`` refuses that parameter combination for the same reason.
    """
    rng = np.random.default_rng(seed)
    low = -1 if eps > 0 else 0
    states = rng.integers(low, n_alleles + 1, size=(6, n_sites)).astype(np.int64)
    states[0] = 0
    return states, _prior(n_alleles)


@pytest.mark.parametrize("eps,delta", [(0.0, 0.0), (EPS, DELTA)])
def test_k1_leaf_marginal_matches_cas9(eps, delta):
    states, prior = _k1_setup(eps)
    cass = _cass(states, 1, 3)
    depths = torch.tensor([0.0, 0.4, 0.9, 1.3, 0.6, 1.1])
    leaves = torch.arange(1, 6)
    got = sc.sequential_leaf_marginal_logprob(
        cass, depths, torch.tensor(1.0), leaves,
        heritable_eps=eps, stochastic_delta=delta)
    want = leaf_marginal_logprob(
        torch.as_tensor(states), depths, torch.tensor(1.0),
        torch.as_tensor(prior), leaves, heritable_eps=eps, stochastic_delta=delta)
    assert torch.allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("eps,delta", [(0.0, 0.0), (EPS, DELTA)])
def test_k1_triple_matches_cas9(eps, delta):
    states, prior = _k1_setup(eps)
    cass = _cass(states, 1, 3)
    idx = torch.tensor([[1, 2, 3], [4, 5, 1], [2, 4, 5]])
    br = _branches(3, [0.3, 0.5, 0.7, 0.4, 0.9])
    got = sc.sequential_triple_site_logprob(
        cass, idx, *br, torch.tensor(1.0),
        heritable_eps=eps, stochastic_delta=delta)
    want = triple_site_logprob(
        torch.as_tensor(states), idx, *br, torch.tensor(1.0),
        torch.as_tensor(prior), heritable_eps=eps, stochastic_delta=delta)
    assert torch.allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("eps,delta", [(0.0, 0.0), (EPS, DELTA)])
def test_k1_balanced_quartet_matches_cas9(eps, delta):
    states, prior = _k1_setup(eps)
    cass = _cass(states, 1, 3)
    idx = torch.tensor([[1, 2, 3, 4], [5, 1, 2, 3], [4, 5, 1, 2]])
    br = _branches(3, [0.3, 0.4, 0.6, 0.5, 0.35, 0.8, 0.45])
    got = sc.sequential_quartet_site_logprob_balanced(
        cass, idx, *br, torch.tensor(1.0),
        heritable_eps=eps, stochastic_delta=delta)
    want = _quartet_site_logprob_balanced(
        torch.as_tensor(states), idx, *br, torch.tensor(1.0),
        torch.as_tensor(prior), heritable_eps=eps, stochastic_delta=delta)
    assert torch.allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("eps,delta", [(0.0, 0.0), (EPS, DELTA)])
def test_k1_caterpillar_quartet_matches_cas9(eps, delta):
    """The branch the Cas9 code reaches via triple_site_logprob.

    This is the load-bearing test: the sequential caterpillar is a separate
    two-level push in log space precisely so it does not inherit the Cas9
    path's probability-space clamp, and only this comparison pins it.
    """
    states, prior = _k1_setup(eps)
    cass = _cass(states, 1, 3)
    outgroup_idx = torch.tensor([[1], [5], [3]])
    sub_triple_idx = torch.tensor([[2, 3, 4], [1, 2, 3], [4, 5, 1]])
    br = _branches(3, [0.3, 0.7, 0.25, 0.4, 0.55, 0.6, 0.5])
    got = sc.sequential_quartet_site_logprob_caterpillar(
        cass, outgroup_idx, sub_triple_idx, *br, torch.tensor(1.0),
        heritable_eps=eps, stochastic_delta=delta)
    want = _quartet_site_logprob_caterpillar(
        torch.as_tensor(states), outgroup_idx, sub_triple_idx, *br,
        torch.tensor(1.0), torch.as_tensor(prior),
        heritable_eps=eps, stochastic_delta=delta)
    assert torch.allclose(got, want, atol=1e-10)


# 3. normalisation at K > 1

def test_pair_joint_normalises():
    K, A = 3, 2
    cfg = _configs(K, A)
    n = len(cfg)
    cass = _cass(cfg, K, A)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    idx_i = torch.as_tensor(ii.ravel())
    idx_j = torch.as_tensor(jj.ravel())
    B = len(idx_i)
    stem, t_i, t_j = _branches(B, [0.4, 0.7, 1.1])
    logP = _pair(cass, idx_i, idx_j, [stem, t_i, t_j], 0.8, 0.0, 0.0)
    total = logP[..., 0].exp().sum()
    assert abs(float(total) - 1.0) < 1e-10


def test_leaf_marginal_normalises():
    K, A = 3, 2
    cfg = _configs(K, A)
    cass = _cass(cfg, K, A)
    depths = torch.full((len(cfg),), 0.9)
    leaves = torch.arange(len(cfg))
    logP = sc.sequential_leaf_marginal_logprob(cass, depths, torch.tensor(0.8), leaves)
    assert abs(float(logP.exp().sum()) - 1.0) < 1e-10


def _all_quartet_indices(n):
    grids = np.meshgrid(*([np.arange(n)] * 4), indexing="ij")
    return torch.as_tensor(np.stack([g.ravel() for g in grids], axis=1))


def test_balanced_quartet_normalises():
    K, A = 3, 2
    cfg = _configs(K, A)
    cass = _cass(cfg, K, A)
    idx = _all_quartet_indices(len(cfg))
    br = _branches(len(idx), [0.3, 0.4, 0.6, 0.5, 0.35, 0.8, 0.45])
    logP = sc.sequential_quartet_site_logprob_balanced(
        cass, idx, *br, torch.tensor(0.8))
    assert abs(float(logP.exp().sum()) - 1.0) < 1e-9


def test_caterpillar_quartet_normalises():
    K, A = 3, 2
    cfg = _configs(K, A)
    cass = _cass(cfg, K, A)
    idx = _all_quartet_indices(len(cfg))
    br = _branches(len(idx), [0.3, 0.7, 0.25, 0.4, 0.55, 0.6, 0.5])
    logP = sc.sequential_quartet_site_logprob_caterpillar(
        cass, idx[:, :1], idx[:, 1:], *br, torch.tensor(0.8))
    assert abs(float(logP.exp().sum()) - 1.0) < 1e-9


def test_missing_normalisation_and_reduction():
    """With a 'cassette missing' outcome added, the pair still normalises.

    And at eps = delta = 0 the missing-aware path must collapse back onto the
    complete-data one.
    """
    K, A = 2, 2
    cfg = _configs(K, A)
    cfg_m = np.vstack([cfg, -np.ones((1, K), dtype=np.int64)])
    n = len(cfg_m)
    cass = _cass(cfg_m, K, A)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    idx_i, idx_j = torch.as_tensor(ii.ravel()), torch.as_tensor(jj.ravel())
    B = len(idx_i)
    br = _branches(B, [0.4, 0.7, 1.1])

    with_missing = _pair(cass, idx_i, idx_j, br, 0.8, EPS, DELTA)
    assert abs(float(with_missing[..., 0].exp().sum()) - 1.0) < 1e-10

    # At zero rates the missing configuration is impossible and the rest must
    # match the complete-data computation exactly.
    at_zero = _pair(cass, idx_i, idx_j, br, 0.8, 0.0, 0.0)
    observed = (idx_i < len(cfg)) & (idx_j < len(cfg))
    assert abs(float(at_zero[observed, :, 0].exp().sum()) - 1.0) < 1e-10


# 4. prefix and seed distances

def test_shared_prefix_skips_missing_cells():
    # Indexing is (batch, cassette, position, cell). Cell 0 is [1, 2, 0]
    # (two filled), cell 1 is [1, 2, 3] (three filled).
    K = 3
    alleles = torch.tensor([[[[1, 1], [2, 2], [0, 3]]]])          # (1,1,K,2)

    # Both observed: the prefix stops where cell 0 runs out of edits.
    L, _ = sc.shared_prefix_length(alleles, torch.tensor([[[False, False]]]))
    assert int(L) == 2

    # Cell 1 missing: only cell 0 constrains, so the answer is still its end.
    L, _ = sc.shared_prefix_length(alleles, torch.tensor([[[False, True]]]))
    assert int(L) == 2

    # Cell 0 missing: it must not be scored as a disagreement at position 2;
    # an unobserved cell rules nothing out, so cell 1's full prefix stands.
    L, _ = sc.shared_prefix_length(alleles, torch.tensor([[[True, False]]]))
    assert int(L) == 3

    # Everything missing constrains nothing at all.
    L, _ = sc.shared_prefix_length(alleles, torch.tensor([[[True, True]]]))
    assert int(L) == K


def test_shared_prefix_stops_at_first_disagreement():
    alleles = torch.tensor([[[[1, 1], [2, 5], [3, 3]]]])
    missing = torch.zeros(1, 1, 2, dtype=torch.bool)
    L, _ = sc.shared_prefix_length(alleles, missing)
    assert int(L) == 1


def test_seed_distances_match_weighted_hamming_at_k1():
    states = _toy_cassettes(8, C=6, K=1, n_alleles=4, seed=3, missing_frac=0.2)
    labels = ["__root__"] + [str(i) for i in range(1, states.shape[0])]
    leaf_states = {lab: states[i] for i, lab in enumerate(labels)}
    want = seed_distances(np.array([leaf_states[l] for l in labels]))
    got = sc.cassette_seed_distances(states, 1)
    assert np.allclose(got, want, atol=1e-12)


def test_validate_rejects_non_prefix_cassette():
    # Cassette [0, 1] fills position 2 while position 1 is still empty, which
    # ordered editing forbids. [1, 0] in the same row is a legal prefix.
    bad = np.array([[0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="filled position above an empty one"):
        sc.validate_cassette_matrix(bad, 2)


def test_validate_accepts_legal_prefixes():
    ok = np.array([[0, 0, 0, 0], [1, 0, 2, 3]], dtype=np.int64)
    out, report = sc.validate_cassette_matrix(ok, 2)
    assert report["n_cassettes"] == 2
    assert report["n_partial_cassettes"] == 0
    assert np.array_equal(out.numpy(), ok)


def test_validate_rejects_indivisible_width():
    with pytest.raises(ValueError, match="not divisible"):
        sc.validate_cassette_matrix(np.zeros((2, 5), dtype=np.int64), 2)


def test_partial_cassette_promoted_and_reported():
    states = np.array([[0, 0], [1, -1]], dtype=np.int64)
    out, report = sc.validate_cassette_matrix(states, 2)
    assert report["n_partial_cassettes"] == 1
    assert (out[1] == -1).all()
    with pytest.raises(ValueError, match="partially observed"):
        sc.validate_cassette_matrix(states, 2, on_partial="error")


# 5. gradients and end-to-end

def test_gradients_flow_and_survive_zero_branches():
    K, A = 2, 3
    states = _toy_cassettes(6, C=3, K=K, n_alleles=A, seed=5, missing_frac=0.2)
    cass = _cass(states, K, A)
    idx = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    br = [torch.zeros(2, requires_grad=True) for _ in range(7)]
    rate = torch.tensor(0.9, requires_grad=True)
    out = sc.sequential_quartet_site_logprob_balanced(
        cass, idx, *br, rate, heritable_eps=EPS, stochastic_delta=DELTA)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(b.grad).all() for b in br)
    assert torch.isfinite(rate.grad).all()


def test_gradcheck_balanced_quartet():
    K, A = 2, 2
    states = _toy_cassettes(5, C=2, K=K, n_alleles=A, seed=7)
    cass = _cass(states, K, A)
    idx = torch.tensor([[1, 2, 3, 4]])
    base = [0.3, 0.4, 0.6, 0.5, 0.35, 0.8, 0.45]

    def f(*branches):
        return sc.sequential_quartet_site_logprob_balanced(
            cass, idx, *branches, torch.tensor(0.8))

    inputs = tuple(torch.full((1,), v, requires_grad=True) for v in base)
    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-6)


def test_no_neg_sentinel_leaks_into_the_loss():
    K, A = 3, 4
    states = _toy_cassettes(12, C=4, K=K, n_alleles=A, seed=11, missing_frac=0.3)
    cass = _cass(states, K, A)
    idx = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    br = _branches(3, [0.3, 0.4, 0.6, 0.5, 0.35, 0.8, 0.45])
    out = sc.sequential_quartet_site_logprob_balanced(
        cass, idx, *br, torch.tensor(0.8),
        heritable_eps=EPS, stochastic_delta=DELTA)
    assert bool((out > -1e6).all())


def _fit_sequential(states, K, *, initiation_ratio=1.0, **options):
    """``heracles.embed`` with the sequential model on a founder-augmented matrix."""
    import pandas as pd
    from heracles.fit import embed
    from heracles.models import SequentialModel
    cm = pd.DataFrame(states[1:], index=["c%d" % i for i in range(states.shape[0] - 1)])
    base = dict(min_iterations=0, convergence_thresh=None, learn_heritable_eps=False,
                learn_stochastic_delta=False, learn_allele_prior=False, ultrametric_penalty=0,
                init_method="mds", allele_prior="global_uniform")
    base.update(options)
    emb = embed(cm, edit_model=SequentialModel(cassette_size=K, initiation_ratio=initiation_ratio),
                **base)
    return emb["leaf_dists"], emb["meta"]


def test_sequential_fit_runs_and_reduces_loss(tmp_path):
    K, A, C = 3, 5, 4
    states = _toy_cassettes(14, C=C, K=K, n_alleles=A, seed=13, missing_frac=0.1)
    leaf_dists, meta = _fit_sequential(states, K, dimension=4, max_iterations=40, seed=0,
                                       heritable_eps=1e-3, stochastic_delta=0.05)
    assert leaf_dists.shape == (14, 14)
    assert np.isfinite(leaf_dists).all()
    assert meta["cassette_size"] == K
    assert meta["cassette_report"]["n_cassettes"] == C
    history = meta["history"]["loss"]
    assert history[-1] < history[0]


def test_per_step_rates_differ_from_a_single_rate():
    """A faster first edit must actually change the transition matrix.

    Data simulated with a first-edit rate of 1.0 and later rates of 0.5 is
    misspecified under the single-rate form.
    """
    K = 5
    branch = torch.tensor([0.8])
    single = sc.log_transition(torch.tensor(0.5), branch, K,
                               rate_model="matrix_exp").exp()
    per_step = torch.full((K,), 0.5)
    per_step[0] = 1.0
    faster_start = sc.log_transition(per_step, branch, K,
                                     rate_model="matrix_exp").exp()
    assert torch.allclose(faster_start.sum(-1), torch.ones(1, 1, K + 1), atol=1e-12)
    # Starting faster leaves fewer cassettes still unedited.
    assert float(faster_start[0, 0, 0, 0]) < float(single[0, 0, 0, 0])
    assert not torch.allclose(faster_start, single, atol=1e-6)


def test_fit_with_faster_initiation_runs():
    K, A, C = 5, 30, 6
    states = _toy_cassettes(16, C=C, K=K, n_alleles=A, seed=17, missing_frac=0.1)
    leaf_dists, meta = _fit_sequential(states, K, initiation_ratio=2.0, dimension=4,
                                       max_iterations=30, seed=0,
                                       heritable_eps=1e-3, stochastic_delta=0.05)
    assert meta["rate_model"] == "two_rate"
    assert meta["initiation_ratio"] == 2.0
    assert np.isfinite(leaf_dists).all()
    history = meta["history"]["loss"]
    assert history[-1] < history[0]


@pytest.mark.parametrize("K", [2, 3, 5])
@pytest.mark.parametrize("alpha", [1.5, 2.0, 3.0])
def test_two_rate_matches_matrix_exp(K, alpha):
    """Closed-form two_rate must agree with the matrix_exp reference."""
    t = torch.tensor([0.01, 0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    rate = torch.ones(K, dtype=torch.float64)
    rate[0] = alpha
    ref = sc.log_transition(rate, t, K, rate_model="matrix_exp")
    got = sc.log_transition(rate, t, K, rate_model="two_rate")
    s = torch.arange(K + 1)
    forward = s.view(-1, 1) <= s.view(1, -1)
    torch.testing.assert_close(
        got[:, :, forward], ref[:, :, forward], atol=1e-4, rtol=1e-4)


# banded sampler is shared with the Cas9 quartet objective

def test_build_banded_sampler_plain_draw_matches_direct_sampler():
    """No ramp, no local outgroups: identical quartets to sample_quartets_banded."""
    from heracles.embedding.sampler import (build_banded_sampler, build_tau_neighbours,
                                            sample_quartets_banded)
    K, A, C = 3, 5, 6
    states = _toy_cassettes(30, C=C, K=K, n_alleles=A, seed=3)
    est = sc.cassette_seed_distances(states, K)
    sampler, cfg, stats = build_banded_sampler(
        est, states.shape[0], band_kmin=2, band_kmax=16, band_kmax_end=None,
        band_local_frac=0.0, band_local_mult_max=None)
    order, tau = build_tau_neighbours(est)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    q_shared = sampler(30, 40, generator=g1)
    q_direct, _ = sample_quartets_banded(
        30, 40, generator=g2, order=torch.as_tensor(order),
        tau=torch.as_tensor(tau, dtype=torch.float64), kmin=2, kmax=16)
    assert torch.equal(q_shared, q_direct)
    assert cfg == {"kmin": 2, "kmax": 16, "kmax_end": None, "local_frac": 0.0,
                   "local_mult_max": None, "tau_source": "est_dists"}
    assert "kmax_effective" not in stats


def test_build_banded_sampler_ramps_kmax_over_ramp_steps():
    from heracles.embedding.sampler import build_banded_sampler
    K, A, C = 3, 5, 6
    states = _toy_cassettes(40, C=C, K=K, n_alleles=A, seed=4)
    est = sc.cassette_seed_distances(states, K)
    sampler, cfg, stats = build_banded_sampler(
        est, states.shape[0], band_kmax=16, band_kmax_end=32, ramp_steps=4)
    assert cfg["kmax_end"] == 32
    seen = []
    g = torch.Generator().manual_seed(0)
    for _ in range(6):
        sampler(40, 20, generator=g)
        seen.append(stats["kmax_effective"])
    assert seen == [16, 20, 24, 28, 32, 32]


def test_sequential_embed_records_local_outgroup_sampler_config():
    """The banded-sampler options reach the sequential likelihood and its metadata."""
    K, A, C = 3, 5, 4
    states = _toy_cassettes(24, C=C, K=K, n_alleles=A, seed=5)
    _, meta = _fit_sequential(states, K, dimension=4, max_iterations=20, seed=0,
                              band_kmin=2, band_kmax=4, band_kmax_end=8,
                              band_local_frac=1.0, band_local_mult_max=4)
    cfg = meta["quartet_sampler_config"]
    assert cfg["kmax_end"] == 8 and cfg["local_frac"] == 1.0
    assert cfg["local_mult_max"] == 4.0
    stats = meta["quartet_sampler_stats"]
    assert stats["local_frac_accepted"] > 0.0
    assert "kmax_effective" in stats


def test_cas9_and_sequential_loss_factories_share_sampler_config():
    from heracles.embedding.objective import quartet_loss_fn
    K, A, C = 3, 5, 4
    states = _toy_cassettes(20, C=C, K=K, n_alleles=A, seed=6)
    est = sc.cassette_seed_distances(states, K)
    m = states.shape[0]
    kw = dict(est_dists=est, band_kmin=2,
              band_kmax=8, band_kmax_end=32, band_local_frac=0.5,
              band_local_mult_max=4)
    cas9 = quartet_loss_fn(states, _prior(A), m, **kw)
    seq = sc.sequential_quartet_loss_fn(states, _prior(A), m, cassette_size=K, quartet_sampler="banded", **kw)
    assert cas9.quartet_sampler_config == seq.quartet_sampler_config

