"""CTMC kernels for the irreversible-edit model with missing data.

Per-site probability kernels for the Cas9 edit CTMC with heritable deletion
(``heritable_eps``) and stochastic dropout (``stochastic_delta``), evaluated on
branch lengths read off the embedding: pair joints, the rooted-triple kernel
(:func:`triple_site_logprob`), the root-to-leaf marginal
(:func:`leaf_marginal_logprob`), the Gromov four-point penalty, a
distinct-index sampler and the learnable allele-prior decorator
(:func:`with_learnable_prior`).

Conventions: taxon 0 is the founder (all-zero root), leaves are 1..n;
``states`` is an ``(m, S)`` long tensor with ``-1`` for missing; allele
priors are ``(K,)`` global or ``(S, K)`` per-site with slot 0 = unedited.
"""

import functools

import torch

from heracles.constants import LOG_FLOOR, MASK_LOGIT


__all__ = [
    'observation_weight',
    'leaf_dropout',
    'pair_edit_joint',
    'triple_branch_lengths',
    'pair_edit_joint_missing',
    'triple_site_logprob',
    'leaf_marginal_logprob',
    'sample_distinct',
    'gromov_quartet_penalty_from_distances',
    'gromov_quartet_penalty',
    'with_learnable_prior',
]


def observation_weight(no_del, obs_a, obs_b, missing_a, missing_b):
    """Probability of two sibling leaves' observed/missing pattern once their allele is fixed.

    Used when the edit happened above the leaves' common ancestor, so only
    whether each leaf is observed matters. A deletion on the shared branch
    blanks both leaves. Returns probabilities broadcast from the inputs.

    Args:
        no_del: survival of the shared branch against heritable deletion.
        obs_a, obs_b: probability that each leaf is observed.
        missing_a, missing_b: boolean masks, True where the leaf is missing.
    """
    ma = _to_float_mask(missing_a)
    mb = _to_float_mask(missing_b)
    return (no_del * (obs_a + ma * (1 - 2 * obs_a))
                   * (obs_b + mb * (1 - 2 * obs_b))
            + (1 - no_del) * ma * mb)


def _to_float_mask(mask):
    """Cast a boolean mask to float64."""
    return mask.to(torch.float64)


def leaf_dropout(stochastic_delta, rows):
    """Return ``1 - stochastic_delta`` for the leaves in ``rows`` (a ``(B,)`` long tensor).

    ``stochastic_delta`` is a scalar, an ``(S,)`` per-site vector, or an ``(m, S)``
    per-cell, per-site table with the founder at row 0. Only the table is
    indexed by ``rows``; the other forms broadcast over ``(B, S)``.
    """
    if torch.is_tensor(stochastic_delta) and stochastic_delta.dim() == 2:
        return 1.0 - stochastic_delta[rows]
    return 1.0 - stochastic_delta


_SITE_IDX_CACHE = {}


def _index_prior(allele_prior, states):
    """Look up the allele prior of each observed state; accepts ``(K,)`` or ``(S, K)`` priors."""
    clamped = states.clamp_min(0)
    if allele_prior.dim() == 1:
        if not states.is_cuda:
            return allele_prior[clamped]
        # GPU: index a per-site view of the global prior: same values, but the
        # backward scatter-add spreads over S*K slots instead of K, which is
        # much faster for a learned prior (otherwise every gradient add
        # contends for the same K addresses); on CPU the direct lookup is faster
        allele_prior = allele_prior.unsqueeze(0).expand(states.shape[-1], -1)
    key = (allele_prior.shape[0], states.device)
    site_idx = _SITE_IDX_CACHE.get(key)
    if site_idx is None:
        site_idx = torch.arange(allele_prior.shape[0], device=states.device)
        _SITE_IDX_CACHE[key] = site_idx
    return allele_prior[site_idx, clamped]


# Cherry index -> permutation of a triple's three leaf slots so that the first two slots
# are the cherry and the third is the outgroup. Row r corresponds to the r-th entry of
# the stacked tau vector (tau_ij, tau_ik, tau_jk) being the largest.
_CHERRY_PERM = torch.tensor([[0, 1, 2],      # cherry (i, j), outgroup k
                             [0, 2, 1],      # cherry (i, k), outgroup j
                             [1, 2, 0]],     # cherry (j, k), outgroup i
                            dtype=torch.long)
_CHERRY_PERM_CACHE = {}


def _cherry_perm(device):
    """Return _CHERRY_PERM on the given device, cached after first call."""
    if device not in _CHERRY_PERM_CACHE:
        _CHERRY_PERM_CACHE[device] = _CHERRY_PERM.to(device)
    return _CHERRY_PERM_CACHE[device]


def pair_edit_joint(no_edit_shared, no_edit_i, no_edit_j, prior_i, prior_j,
                    both_unedited, first_unedited, second_unedited, same_edit, diff_edit):
    """Joint probability of two sibling leaves' states on complete data.

    Conditions on the top of the shared branch being unedited.

    Args:
        no_edit_shared, no_edit_i, no_edit_j: ``exp(-mu t)`` for the shared branch
            and for each leaf's own branch.
        prior_i, prior_j: allele prior of each leaf's observed state.
        both_unedited, first_unedited, second_unedited, same_edit, diff_edit: float
            masks of the five state patterns (``first_unedited``: leaf i unedited,
            leaf j edited).

    Returns:
        Tensor ``P(s_i, s_j)``, broadcast from the inputs.
    """
    return (no_edit_shared * no_edit_i * no_edit_j) * both_unedited \
        + (no_edit_shared * no_edit_i * (1 - no_edit_j) * prior_j) * first_unedited \
        + (no_edit_shared * (1 - no_edit_i) * prior_i * no_edit_j) * second_unedited \
        + (no_edit_shared * (1 - no_edit_i) * (1 - no_edit_j) * prior_i * prior_i + (1 - no_edit_shared) * prior_i) * same_edit \
        + (no_edit_shared * (1 - no_edit_i) * (1 - no_edit_j) * prior_i * prior_j) * diff_edit


def triple_branch_lengths(depths, d01, d02, d12, triples, taxa_depths=None):
    """Read the five branch lengths of each rooted triple off the embedding.

    The pair with the deepest common ancestor (largest Gromov product with
    respect to the root) is the cherry and the third leaf is the outgroup. All
    lengths are clamped at 0. Gradients flow through the distances; the cherry
    choice is a discrete selection redone on every call.

    Args:
        depths: (m,) root-to-taxon distances; entry 0 is the root (0).
        d01, d02, d12: (B,) leaf-leaf distances for slot pairs (0,1), (0,2), (1,2)
            of each triple.
        triples: (B, 3) long tensor of taxon indices.
        taxa_depths: optional pre-gathered ``depths[triples]``, shape (B, 3).

    Returns:
        tuple: ``(idx, root_u, u_v, v_cherry1, v_cherry2, u_outgroup)``. ``idx`` is
        the (B, 3) taxon tensor reordered as (cherry leaf 1, cherry leaf 2,
        outgroup). The rest are (B,) branch lengths: root to the split point ``u``,
        ``u`` to the cherry ancestor ``v``, ``v`` to each cherry leaf, and ``u`` to
        the outgroup.
    """
    if taxa_depths is None:
        taxa_depths = depths[triples]                                 # (B, 3)
    d_i, d_j, d_k = taxa_depths[:, 0], taxa_depths[:, 1], taxa_depths[:, 2]
    tau_ij = (0.5 * (d_i + d_j - d01)).clamp_min(0.0)
    tau_ik = (0.5 * (d_i + d_k - d02)).clamp_min(0.0)
    tau_jk = (0.5 * (d_j + d_k - d12)).clamp_min(0.0)
    taus = torch.stack([tau_ij, tau_ik, tau_jk], dim=1)               # (B, 3)

    cherry = taus.argmax(dim=1)                                       # (B,)
    cherry_depth = taus.gather(1, cherry.unsqueeze(1)).squeeze(1)     # deepest MRCA
    perm = _cherry_perm(triples.device)[cherry]                       # (B, 3)
    idx = triples.gather(1, perm)                                     # (B, 3) reordered
    depths_perm = taxa_depths.gather(1, perm)                         # (B, 3), dense

    root_u = (0.5 * (taus.sum(dim=1) - cherry_depth)).clamp_min(0.0)
    u_v = (cherry_depth - root_u).clamp_min(0.0)
    v_cherry1 = (depths_perm[:, 0] - cherry_depth).clamp_min(0.0)
    v_cherry2 = (depths_perm[:, 1] - cherry_depth).clamp_min(0.0)
    u_outgroup = (depths_perm[:, 2] - root_u).clamp_min(0.0)
    return idx, root_u, u_v, v_cherry1, v_cherry2, u_outgroup


def pair_edit_joint_missing(no_edit_shared, no_edit_i, no_edit_j, prior_i, prior_j,
                            no_del_shared, observed_i, observed_j,
                            both_unedited, first_unedited_second_edited,
                            first_edited_second_unedited, same_edit, diff_edit,
                            i_unedited_j_missing, i_edited_j_missing,
                            j_unedited_i_missing, j_edited_i_missing, both_missing):
    """Joint probability of two sibling leaves' observed pattern with missing data.

    Extends :func:`pair_edit_joint` with heritable deletion and stochastic
    dropout: either leaf can be missing alone, and both can be missing (also
    through a deletion on the shared branch).

    Args:
        no_edit_shared, no_edit_i, no_edit_j, prior_i, prior_j: as in
            :func:`pair_edit_joint`.
        no_del_shared: survival of the shared branch against heritable deletion.
        observed_i, observed_j: probability that each leaf is observed.
        both_unedited, first_unedited_second_edited, first_edited_second_unedited,
            same_edit, diff_edit: the five both-observed masks of :func:`pair_edit_joint`.
        i_unedited_j_missing, i_edited_j_missing, j_unedited_i_missing,
            j_edited_i_missing, both_missing: float masks of the missing patterns.

    Returns:
        Tensor of probabilities, broadcast from the inputs.
    """
    P_joint = pair_edit_joint(no_edit_shared, no_edit_i, no_edit_j, prior_i, prior_j,
                              both_unedited, first_unedited_second_edited,
                              first_edited_second_unedited, same_edit, diff_edit)
    P_edit_i = (no_edit_shared * no_edit_i) * i_unedited_j_missing + ((1 - no_edit_shared * no_edit_i) * prior_i) * i_edited_j_missing      # leaf-i marginal
    P_edit_j = (no_edit_shared * no_edit_j) * j_unedited_i_missing + ((1 - no_edit_shared * no_edit_j) * prior_j) * j_edited_i_missing      # leaf-j marginal
    return no_del_shared * observed_i * observed_j * P_joint \
        + no_del_shared * observed_i * (1 - observed_j) * P_edit_i \
        + no_del_shared * (1 - observed_i) * observed_j * P_edit_j \
        + ((1 - no_del_shared) + no_del_shared * (1 - observed_i) * (1 - observed_j)) * both_missing


def _triple_site_logprob_parts(states, idx, root_u, u_v, v_cherry1, v_cherry2, u_outgroup, mutation_rate, allele_prior, heritable_eps=0.0, stochastic_delta=0.0):
    """Compute :func:`triple_site_logprob` and also return its intermediates.

    Returns ``(logP, parts)``; ``parts`` holds the state rows, missing masks,
    priors and observation factors so the caterpillar quartet kernel can reuse
    them.
    """
    states_cherry_1, states_cherry_2, states_outgroup = states[idx[:, 0]], states[idx[:, 1]], states[idx[:, 2]]   # (B, S)
    no_edit_root_to_internal = torch.exp(-mutation_rate * root_u).unsqueeze(1)           # (B, 1)
    no_edit_to_cherry_ancestor = torch.exp(-mutation_rate * u_v).unsqueeze(1)
    no_edit_to_cherry_1 = torch.exp(-mutation_rate * v_cherry1).unsqueeze(1)
    no_edit_to_cherry_2 = torch.exp(-mutation_rate * v_cherry2).unsqueeze(1)
    no_edit_to_outgroup = torch.exp(-mutation_rate * u_outgroup).unsqueeze(1)
    prior_cherry_1 = _index_prior(allele_prior, states_cherry_1)
    prior_cherry_2 = _index_prior(allele_prior, states_cherry_2)
    prior_outgroup = _index_prior(allele_prior, states_outgroup)

    missing_cherry_1, missing_cherry_2, missing_outgroup = states_cherry_1 < 0, states_cherry_2 < 0, states_outgroup < 0   # per-leaf missing masks
    # both cherry leaves observed: >0 / ==0 already exclude the -1 code, so these five
    # masks partition exactly the both-observed sites.
    both_unedited = _to_float_mask((states_cherry_1 == 0) & (states_cherry_2 == 0))
    first_unedited_second_edited = _to_float_mask((states_cherry_1 == 0) & (states_cherry_2 > 0))
    first_edited_second_unedited = _to_float_mask((states_cherry_1 > 0) & (states_cherry_2 == 0))
    same_edit = _to_float_mask((states_cherry_1 > 0) & (states_cherry_1 == states_cherry_2))
    diff_edit = _to_float_mask((states_cherry_1 > 0) & (states_cherry_2 > 0) & (states_cherry_1 != states_cherry_2))

    # deletion survival on the two internal branches, and per-leaf observation weights.
    no_del_root_to_internal = torch.exp(-heritable_eps * root_u).unsqueeze(1)
    no_del_to_cherry_ancestor = torch.exp(-heritable_eps * u_v).unsqueeze(1)
    observed_cherry_1 = torch.exp(-heritable_eps * v_cherry1).unsqueeze(1) * leaf_dropout(stochastic_delta, idx[:, 0])
    observed_cherry_2 = torch.exp(-heritable_eps * v_cherry2).unsqueeze(1) * leaf_dropout(stochastic_delta, idx[:, 1])
    observed_outgroup = torch.exp(-heritable_eps * u_outgroup).unsqueeze(1) * leaf_dropout(stochastic_delta, idx[:, 2])

    # u unedited: cherry factor times outgroup factor, both missing-aware.
    M = pair_edit_joint_missing(
        no_edit_to_cherry_ancestor, no_edit_to_cherry_1, no_edit_to_cherry_2,
        prior_cherry_1, prior_cherry_2,
        no_del_to_cherry_ancestor, observed_cherry_1, observed_cherry_2,
        both_unedited, first_unedited_second_edited,
        first_edited_second_unedited, same_edit, diff_edit,
        _to_float_mask((states_cherry_1 == 0) & missing_cherry_2),
        _to_float_mask((states_cherry_1 > 0) & missing_cherry_2),
        _to_float_mask(missing_cherry_1 & (states_cherry_2 == 0)),
        _to_float_mask(missing_cherry_1 & (states_cherry_2 > 0)),
        _to_float_mask(missing_cherry_1 & missing_cherry_2))
    L_out = observed_outgroup * (no_edit_to_outgroup * _to_float_mask(states_outgroup == 0)
            + (1 - no_edit_to_outgroup) * prior_outgroup * _to_float_mask(states_outgroup > 0)) \
            + (1 - observed_outgroup) * _to_float_mask(missing_outgroup)

    # u edited to alpha: every observed leaf must carry that same allele.
    # alpha is pinned by the first observed leaf; with all three missing the sum over
    # alpha collapses to 1 (q sums to 1), which is what the 1.0 fallback encodes.
    q_alpha = torch.where((states_cherry_1 > 0), prior_cherry_1,
                          torch.where((states_cherry_2 > 0), prior_cherry_2,
                          torch.where((states_outgroup > 0), prior_outgroup, 1.0)))
    # no observed leaf may be unedited, and all observed leaves must agree
    edited_ok = _to_float_mask(((states_cherry_1 > 0) | missing_cherry_1) & ((states_cherry_2 > 0) | missing_cherry_2) & ((states_outgroup > 0) | missing_outgroup))
    agree = _to_float_mask((missing_cherry_1 | missing_cherry_2 | (states_cherry_1 == states_cherry_2))
                         & (missing_cherry_1 | missing_outgroup | (states_cherry_1 == states_outgroup))
                         & (missing_cherry_2 | missing_outgroup | (states_cherry_2 == states_outgroup)))
    W_cherry = observation_weight(no_del_to_cherry_ancestor, observed_cherry_1, observed_cherry_2, missing_cherry_1, missing_cherry_2)
    W_out = observed_outgroup * _to_float_mask(~missing_outgroup) + (1 - observed_outgroup) * _to_float_mask(missing_outgroup)

    all_missing_f = _to_float_mask(missing_cherry_1 & missing_cherry_2 & missing_outgroup)
    P = no_del_root_to_internal * (no_edit_root_to_internal * M * L_out
              + (1 - no_edit_root_to_internal) * q_alpha * W_cherry * W_out * edited_ok * agree) \
        + (1 - no_del_root_to_internal) * all_missing_f
    parts = {
        "states": (states_cherry_1, states_cherry_2, states_outgroup),
        "missing": (missing_cherry_1, missing_cherry_2, missing_outgroup),
        "priors": (prior_cherry_1, prior_cherry_2, prior_outgroup),
        "observed": (observed_cherry_1, observed_cherry_2, observed_outgroup),
        "no_del_root_to_internal": no_del_root_to_internal,
        "no_del_to_cherry_ancestor": no_del_to_cherry_ancestor,
        "W_cherry": W_cherry,
        "W_out": W_out,
        "all_missing_f": all_missing_f,
    }
    return torch.log(P.clamp_min(LOG_FLOOR)), parts


def triple_site_logprob(states, idx, root_u, u_v, v_cherry1, v_cherry2,
                        u_outgroup, mutation_rate, allele_prior,
                        heritable_eps=0.0, stochastic_delta=0.0):
    """Per-triple, per-site log probability of the observed states under the edit CTMC.

    With ``heritable_eps = stochastic_delta = 0`` this is the complete-data
    likelihood of the rooted triple. Otherwise it is the probability of the
    observed pattern: a deletion on ``root_u`` blanks all three leaves, one on
    ``u_v`` blanks both cherry leaves, and dropout blanks single leaves.

    Args:
        states: (m, S) long character matrix with the root as row 0; 0 unedited,
            >0 allele, -1 missing.
        idx, root_u, u_v, v_cherry1, v_cherry2, u_outgroup: outputs of
            :func:`triple_branch_lengths`.
        allele_prior: (K,) or (S, K) allele prior, slot 0 = 1 placeholder.
        heritable_eps: heritable deletion rate, scalar (0 disables the channel).
        stochastic_delta: dropout probability, scalar, ``(S,)`` or ``(m, S)``
            (see :func:`leaf_dropout`); 0 disables it.

    Returns:
        torch.Tensor: (B, S) log probabilities.
    """
    logP, _ = _triple_site_logprob_parts(
        states, idx, root_u, u_v, v_cherry1, v_cherry2, u_outgroup,
        mutation_rate, allele_prior,
        heritable_eps=heritable_eps, stochastic_delta=stochastic_delta)
    return logP


def leaf_marginal_logprob(states, depths, mutation_rate, allele_prior, leaves, heritable_eps=0.0, stochastic_delta=0.0):
    """Per-leaf, per-site log probability of the observed state along the root-to-leaf path.

    Sampled triples and quartets never contain the founder, so this term is
    what fixes each leaf's absolute depth ``d(o, x)``. With missing data the
    leaf is observed with probability
    ``exp(-heritable_eps * d(o, x)) * (1 - stochastic_delta)``.

    Args:
        states: (m, S) long character matrix, root as row 0.
        depths: (m,) root-to-taxon distances.
        allele_prior: (K,) or (S, K) allele prior.
        leaves: (L,) long taxon indices (>= 1).

    Returns:
        torch.Tensor: (L, S) log probabilities.
    """
    leaf_states = states[leaves]                               # (L, S)
    root_to_leaf_depth = depths[leaves]
    no_edit_leaf = torch.exp(-mutation_rate * root_to_leaf_depth).unsqueeze(1)       # (L, 1)
    observed_leaf = torch.exp(-heritable_eps * root_to_leaf_depth).unsqueeze(1) * leaf_dropout(stochastic_delta, leaves)
    P = observed_leaf * (no_edit_leaf * _to_float_mask(leaf_states == 0)
        + (1 - no_edit_leaf) * _index_prior(allele_prior, leaf_states) * _to_float_mask(leaf_states > 0)) \
        + (1 - observed_leaf) * _to_float_mask(leaf_states < 0)
    return torch.log(P.clamp_min(LOG_FLOOR))


def sample_distinct(n, k, count, generator, device):
    """Sample ``count`` rows of ``k`` distinct indices from 1..n; a ``(count, k)`` long tensor.

    Draws on the CPU (*generator*, or the global RNG if None) so a given seed
    gives the same indices on any device, then moves the result to ``device``
    (None keeps it on the CPU). Raises ``ValueError`` if ``k > n``.
    """
    kw = {"generator": generator} if generator is not None else {}
    t = torch.randint(1, n + 1, (count, k), **kw)
    pairs = torch.combinations(torch.arange(k), r=2)
    if k > n:
        raise ValueError("cannot draw %d distinct indices from 1..%d" % (k, n))

    def _bad_rows():
        bad = torch.zeros(count, dtype=torch.bool)
        for a, b in pairs:
            bad |= t[:, a] == t[:, b]
        return bad

    for _ in range(_DISTINCT_RETRIES):
        bad = _bad_rows()
        if not bool(bad.any()):
            break
        t[bad] = torch.randint(1, n + 1, (int(bad.sum()), k), **kw)
    else:
        # rejection is hopeless when k is close to n: draw the remaining rows
        # exactly as the first k entries of a random permutation
        for row in torch.nonzero(_bad_rows()).flatten().tolist():
            t[row] = torch.randperm(n, **kw)[:k] + 1
    return t.to(device) if device is not None else t


_DISTINCT_RETRIES = 64


def gromov_quartet_penalty_from_distances(d01, d02, d03, d12, d13, d23):
    """Mean Gromov four-point delta from batched quartet distances ``d01 ... d23``, each (B,).

    Returns a scalar tensor: the batch mean of half the gap between the two largest
    of ``d01 + d23``, ``d02 + d13`` and ``d03 + d12`` (0 for a tree metric).
    """
    s1 = d01 + d23
    s2 = d02 + d13
    s3 = d03 + d12
    sums = torch.stack([s1, s2, s3], dim=1)
    sorted_sums, _ = sums.sort(dim=1)
    return ((sorted_sums[:, 2] - sorted_sums[:, 1]) / 2.0).mean()


def gromov_quartet_penalty(points, manifold, radius, n_quartets, rng):
    """Mean Gromov four-point delta over ``n_quartets`` random leaf quartets, as a scalar tensor.

    The delta is 0 when the distances fit a tree and grows as the embedding
    departs from a tree metric. The founder (row 0) is never sampled. The
    result is differentiable with respect to ``points``.

    Args:
        points: (m, d+1) embedding coordinates, founder at row 0.
        radius: scale multiplying the manifold distances.
    """
    n = points.shape[0] - 1
    _dev = points.device
    idx = sample_distinct(n, 4, n_quartets, rng, _dev)
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    return gromov_quartet_penalty_from_distances(
        radius * manifold.dist(points[a], points[b]),
        radius * manifold.dist(points[a], points[c]),
        radius * manifold.dist(points[a], points[d]),
        radius * manifold.dist(points[b], points[c]),
        radius * manifold.dist(points[b], points[d]),
        radius * manifold.dist(points[c], points[d]))


def _make_learnable_prior(allele_prior_init):
    """Wrap a fixed ``(K,)`` or ``(S, K)`` prior as a learnable softmax.

    Returns ``(get_allele_prior, param)``. The learned prior starts uniform over
    the alleles with a nonzero initial prior; the others stay at 0 and slot 0
    stays 1.
    """
    if allele_prior_init.dim() == 1:
        valid = (allele_prior_init > 0).clone()
        valid[0] = False
        raw = torch.nn.Parameter(torch.zeros_like(allele_prior_init))
        neg_inf = torch.tensor(MASK_LOGIT, dtype=raw.dtype, device=raw.device)

        def get_allele_prior():
            masked = torch.where(valid, raw, neg_inf)
            probs = torch.softmax(masked, dim=0)
            out = torch.zeros_like(probs)
            out[0] = 1.0
            out = out + probs * valid.float()
            return out

        return get_allele_prior, raw

    S, K_max = allele_prior_init.shape
    valid = (allele_prior_init > 0).clone()
    valid[:, 0] = False
    raw = torch.nn.Parameter(torch.zeros_like(allele_prior_init))
    neg_inf = torch.tensor(MASK_LOGIT, dtype=raw.dtype, device=raw.device)

    def get_allele_prior():
        masked = torch.where(valid, raw, neg_inf)
        probs = torch.softmax(masked, dim=1)
        out = torch.zeros_like(probs)
        out[:, 0] = 1.0
        out = out + probs * valid.float()
        return out

    return get_allele_prior, raw


def _resolve_allele_prior(allele_prior, learn_allele_prior, device=None):
    """Return ``(get_allele_prior, param)`` for a fixed or learnable prior; ``param`` is None when fixed."""
    allele_prior_init = torch.as_tensor(allele_prior, dtype=torch.float64)
    if device is not None:
        allele_prior_init = allele_prior_init.to(torch.device(device))
    if learn_allele_prior:
        return _make_learnable_prior(allele_prior_init)

    def get_allele_prior():
        return allele_prior_init

    return get_allele_prior, None


def with_learnable_prior(core_factory):
    """Decorator that adds the fixed-or-learnable allele prior to a loss factory.

    The decorated core is called as
    ``core_factory(states, get_allele_prior, m, **kwargs)``, where
    ``get_allele_prior()`` returns the current prior tensor. The public factory
    has the signature ``(states, allele_prior, m, *, learn_allele_prior=False,
    device=None, ...)`` and sets ``loss_fn.allele_prior_param`` (the trainable
    parameter, or None) so :func:`heracles.embedding.optimize.optimize` trains it.
    """
    @functools.wraps(core_factory)
    def factory(states, allele_prior, m, **kwargs):
        learn_allele_prior = kwargs.pop("learn_allele_prior", False)
        get_allele_prior, allele_param = _resolve_allele_prior(
            allele_prior, learn_allele_prior, kwargs.get("device"))
        loss_fn = core_factory(states, get_allele_prior, m, **kwargs)
        loss_fn.allele_prior_param = allele_param
        return loss_fn

    return factory


