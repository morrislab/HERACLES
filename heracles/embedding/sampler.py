"""The quartet samplers: neighbourhood (the default) and uniform.

The neighbourhood sampler draws every quartet from one cell's neighbourhood:
the cells are ranked by estimated common-ancestor depth, a neighbourhood size
is drawn log-uniformly, and the three partners come from inside it
(:func:`sample_quartets_neighbourhood`). The uniform sampler draws four
distinct cells at random (:func:`sample_quartets_uniform`).

Samplers are looked up by name in :data:`SAMPLERS`, the value of the
``sampler_strategy`` option.  To add one, register a builder::

    from heracles.embedding.sampler import register_sampler

    @register_sampler("mine")
    def build_mine(est_dists, n_points, *, sampler_max_k, knn_columns,
                   character_matrix, missing_state, device):
        def sampler(n_leaves, n_quartets, generator=None, device=None):
            ...                      # (n_quartets, 4) distinct taxa in 1..n_leaves
            return quartets
        return sampler, {"strategy": "mine"}, {}

and fit with ``Heracles(sampler_strategy="mine")``.  The builder gets the
``(n_points, n_points)`` founder-augmented seed distances (``None`` in
low-memory mode, where ``knn_columns`` and ``character_matrix`` are set
instead: build a neighbour table that wide from the states) and returns the sampler, a dict recording its settings and a dict it
may update with per-batch statistics.  A one-off sampler can also be passed
directly as ``quartet_loss_fn(..., sampler=callable)``.
"""

import logging
import math
import time
import warnings

import numpy as np
import torch

from heracles.constants import HAMMING_ROW_CHUNK, QUARTETS_PER_CELL, SAMPLER_MAX_K
from heracles.embedding.ctmc import sample_distinct
from heracles.embedding.initialization import seed_distances

logger = logging.getLogger("heracles")


__all__ = [
    "SAMPLERS",
    "register_sampler",
    "build_tau_neighbours",
    "build_tau_neighbours_from_cm",
    "build_tau_topk",
    "sample_quartets_neighbourhood",
    "sample_quartets_uniform",
    "sampler_reach",
    "n_distinct_quartets",
    "default_batch_size",
    "build_quartet_sampler",
]

#: quartet sampler builders by ``sampler_strategy`` name (see the module docstring)
SAMPLERS = {}


def register_sampler(name):
    """Decorator: make a sampler builder available as ``sampler_strategy=name``."""
    def deco(builder):
        SAMPLERS[name] = builder
        return builder
    return deco


def _stratified_quartets(n_leaves, n_quartets, generator=None, device=None):
    """Leaf-balanced quartet batch: a random permutation of the leaves cut into groups of 4.

    Used as the fixed evaluation batch for the convergence check, not for
    training.
    """
    if n_leaves < 4:
        raise ValueError(
            "need at least 4 leaves to sample quartets, got %d" % n_leaves)
    kw = {"generator": generator} if generator is not None else {}
    per_round = n_leaves // 4
    rounds = -(-n_quartets // per_round)
    chunks = []
    for _ in range(rounds):
        perm = torch.randperm(n_leaves, device=device, **kw)[:per_round * 4] + 1
        chunks.append(perm.view(per_round, 4))
    return torch.cat(chunks)[:n_quartets]


def sample_quartets_uniform(n_leaves, n_quartets, generator=None, device=None):
    """Sample ``n_quartets`` quartets of distinct leaves uniformly from taxa ``1..n_leaves``.

    Every unordered quartet is equally likely, so the minibatch gradient is an
    unbiased estimate of the full quartet composite likelihood. The founder
    (index 0) never appears; it enters only through the depths.

    Returns:
        tuple: ``(quartets, stats)``: a (B, 4) long tensor of distinct 1-based
        taxon indices, and an empty stats dict.
    """
    if n_leaves < 4:
        raise ValueError(
            "need at least 4 leaves to sample quartets, got %d" % n_leaves)
    return sample_distinct(n_leaves, 4, n_quartets, generator, device), {}


def _tau_matrix(dists):
    """``(n, n)`` common-ancestor depths from a founder-augmented ``(n+1, n+1)`` distance matrix.

    ``tau_ij = (d(0,i) + d(0,j) - d(i,j)) / 2``, clamped at 0, zero diagonal.
    """
    d = np.asarray(dists, dtype=np.float64)
    d0 = d[0, 1:]
    D = d[1:, 1:]
    tau = 0.5 * (d0[:, None] + d0[None, :] - D)
    np.clip(tau, 0.0, None, out=tau)
    np.fill_diagonal(tau, 0.0)
    return tau


def build_tau_neighbours(dists):
    """Common-ancestor depths and a neighbour ranking from a founder-augmented distance matrix.

    Args:
        dists: (n+1, n+1) distances, founder at index 0.

    Returns:
        tuple: ``(order, tau)``. ``tau`` is the (n, n) leaf-only matrix
        ``tau_ij = (d(0,i) + d(0,j) - d(i,j)) / 2`` (depth of the common ancestor
        of i and j, clamped at 0, zero diagonal). ``order`` is (n, n-1) int64: for
        each leaf, the other leaves by descending ``tau`` (column 0 = closest),
        ties by ascending index. Both use 0-based leaf indices (leaf i is taxon
        i + 1).
    """
    tau = _tau_matrix(dists)
    n = tau.shape[0]
    # Rank by descending tau, then drop each leaf's own column wherever it landed.
    order_full = np.argsort(-tau, axis=1, kind="stable")
    keep = order_full != np.arange(n)[:, None]
    order = order_full[keep].reshape(n, n - 1)
    return order, tau


def _edited_fraction(cm, missing_state):
    """Founder-to-leaf seed distance: each leaf's share of edited sites among observed ones."""
    valid = cm != missing_state
    return (valid & (cm != 0)).sum(axis=1).astype(np.float64) / np.maximum(valid.sum(axis=1), 1)


def build_tau_neighbours_from_cm(cm, k, missing_state=-1, device=None, row_chunk=1024,
                                 max_onehot_cols=65536):
    """Top-``k`` neighbour ranking computed from the character matrix, never building n x n.

    Gives the stable top-``k`` of :func:`build_tau_neighbours` on the
    weighted-Hamming seed distances
    (:func:`heracles.embedding.initialization.seed_distances`), computed in row blocks
    with matrix products so it runs fast on a GPU. Falls back to a numpy
    version when the allele one-hot would exceed ``max_onehot_cols`` columns.

    Args:
        cm: (n, S) leaf-only character matrix (no founder row).
        k: neighbours to keep per leaf (capped at n - 1).
        missing_state: value marking missing entries.
        device: torch device for the computation (default CPU).
        row_chunk: rows per block.
        max_onehot_cols: one-hot width above which the numpy fallback is used.

    Returns:
        tuple: ``(order, tau_knn, d_root)``: the ``(n, k)`` 0-based neighbour
        indices by descending tau (ties by index), the matching tau values, and
        each leaf's fraction of edited sites among its observed sites.
    """
    cm = np.asarray(cm)
    n, S = cm.shape
    k = min(int(k), n - 1)
    K = int(cm.max()) if n else 0
    if S * max(K, 1) > max_onehot_cols:
        return _build_tau_neighbours_from_cm_numpy(cm, k, missing_state)

    dev = torch.device(device) if device is not None else torch.device("cpu")
    st = torch.as_tensor(cm, dtype=torch.long, device=dev)
    valid = st != int(missing_state)
    edited = valid & (st > 0)
    unedited = valid & (st == 0)
    V = valid.to(torch.float32)
    E = edited.to(torch.float32)
    Z = unedited.to(torch.float32)
    # one-hot of the edited allele per site: column s*K + (state-1)
    OH = torch.zeros((n, S * K), dtype=torch.float32, device=dev)
    rows_e, cols_e = edited.nonzero(as_tuple=True)
    OH[rows_e, cols_e * K + (st[rows_e, cols_e] - 1)] = 1.0

    d_root = _edited_fraction(cm, missing_state)
    d_root_t = torch.as_tensor(d_root, dtype=torch.float64, device=dev)

    order = np.empty((n, k), dtype=np.int64)
    tau_knn = np.empty((n, k), dtype=np.float64)
    ar_n = torch.arange(n, device=dev)
    with torch.no_grad():
        for i0 in range(0, n, row_chunk):
            i1 = min(i0 + row_chunk, n)
            r = i1 - i0
            num_present = V[i0:i1] @ V.T                       # (r, n) exact counts
            count = (Z[i0:i1] @ E.T + E[i0:i1] @ Z.T
                     + 2.0 * (E[i0:i1] @ E.T - OH[i0:i1] @ OH.T))
            D = torch.zeros((r, n), dtype=torch.float64, device=dev)
            ok = num_present > 0
            D[ok] = count.double()[ok] / num_present.double()[ok]
            tau = 0.5 * (d_root_t[i0:i1, None] + d_root_t[None, :] - D)
            tau.clamp_(min=0.0)
            tau[torch.arange(r, device=dev), ar_n[i0:i1]] = -float("inf")   # never itself
            sel_idx, sel_tau = _stable_topk(tau, k)
            order[i0:i1] = sel_idx.cpu().numpy()
            tau_knn[i0:i1] = sel_tau.cpu().numpy()
    return order, tau_knn, d_root


def _stable_topk(tau, k):
    """Per row, the ``k`` largest entries ordered by (-tau, index), as a stable full sort would give.

    Everything above the k-th value, then the lowest-index ties at that value.
    Returns ``(indices, values)``, both (rows, k).
    """
    r = tau.shape[0]
    kth = torch.topk(tau, k, dim=1).values[:, -1]
    above = tau > kth[:, None]
    ties = tau == kth[:, None]
    n_above = above.sum(1)
    take = above | (ties & (ties.cumsum(1) <= (k - n_above)[:, None]))
    sel_idx = take.nonzero(as_tuple=True)[1].view(r, k)     # index-ascending per row
    sel_tau = tau.gather(1, sel_idx)
    srt = torch.sort(-sel_tau, dim=1, stable=True).indices
    return sel_idx.gather(1, srt), sel_tau.gather(1, srt)


def build_tau_topk(dists, k, row_chunk=1024):
    """Top-``k`` neighbour ranking and the tau matrix from a founder-augmented distance matrix.

    The ranking is the first ``k`` columns of :func:`build_tau_neighbours`
    (same tau, same tie order), found in row blocks instead of a full (n, n) sort.

    Args:
        dists: (n+1, n+1) distances, founder at index 0.
        k: neighbours to keep per leaf (capped at n - 1).
        row_chunk: rows per block.

    Returns:
        tuple: ``(order, tau_knn, tau)``: the (n, k) 0-based neighbour indices by
        descending tau, the matching tau values, and the (n, n) tau matrix of
        :func:`build_tau_neighbours`.
    """
    tau = _tau_matrix(dists)
    n = tau.shape[0]
    k = min(int(k), n - 1)
    tau_t = torch.from_numpy(tau)
    order = np.empty((n, k), dtype=np.int64)
    tau_knn = np.empty((n, k), dtype=np.float64)
    for i0 in range(0, n, row_chunk):
        i1 = min(i0 + row_chunk, n)
        rows = tau_t[i0:i1].clone()
        rows[torch.arange(i1 - i0), torch.arange(i0, i1)] = -float("inf")   # never itself
        sel_idx, sel_tau = _stable_topk(rows, k)
        order[i0:i1] = sel_idx.numpy()
        tau_knn[i0:i1] = sel_tau.numpy()
    return order, tau_knn, tau


def nearest_neighbours(states, est_dists, k, *, missing_state=-1, device=None):
    """Each cell's ``k`` nearest cells, nearest first, in the ranking the neighbourhood sampler uses.

    Args:
        states: (n+1, S) character matrix, founder at row 0.
        est_dists: (n+1, n+1) founder-augmented seed distances, or ``None`` to
            rank from the character matrix without building n x n (low-memory mode).
        k: neighbours per cell (capped at n - 1).

    Returns:
        np.ndarray: (n, k) 0-based cell indices (founder excluded).
    """
    if est_dists is None:
        return build_tau_neighbours_from_cm(np.asarray(states)[1:], k, missing_state=missing_state,
                                            device=device)[0]
    return build_tau_topk(est_dists, k)[0]


def _build_tau_neighbours_from_cm_numpy(cm, k, missing_state=-1):
    """Numpy version of :func:`build_tau_neighbours_from_cm` (same outputs), one row block at a time."""
    cm = np.asarray(cm)
    n, S = cm.shape
    k = min(k, n - 1)

    d_root = _edited_fraction(cm, missing_state)

    order = np.empty((n, k), dtype=np.int64)
    tau_knn = np.empty((n, k), dtype=np.float64)

    for i0 in range(0, n, HAMMING_ROW_CHUNK):
        i1 = min(i0 + HAMMING_ROW_CHUNK, n)
        D_chunk = seed_distances(cm, np.arange(i0, i1), missing_state=missing_state)
        tau_chunk = 0.5 * (d_root[i0:i1, None] + d_root[None, :] - D_chunk)
        np.clip(tau_chunk, 0.0, None, out=tau_chunk)
        rows = torch.from_numpy(tau_chunk)
        rows[torch.arange(i1 - i0), torch.arange(i0, i1)] = -float("inf")   # never itself
        sel_idx, sel_tau = _stable_topk(rows, k)
        order[i0:i1], tau_knn[i0:i1] = sel_idx.numpy(), sel_tau.numpy()

    return order, tau_knn, d_root


def _make_tau_fn(cm, d_root, missing_state=-1, device=None):
    """Return ``tau_fn(i, j)`` that computes tau on the fly from the character matrix.

    ``i`` and ``j`` are (M,) long tensors of 0-based leaf indices; the result is
    (M,) float64 on ``device``.
    """
    dev = torch.device(device) if device is not None else None
    cm_t = torch.as_tensor(np.asarray(cm), dtype=torch.long)
    d_root_t = torch.as_tensor(np.asarray(d_root), dtype=torch.float64)
    if dev is not None:
        cm_t, d_root_t = cm_t.to(dev), d_root_t.to(dev)
    miss = int(missing_state)

    def tau_fn(i, j):
        if dev is not None:
            i, j = i.to(dev, non_blocking=True), j.to(dev, non_blocking=True)
        si, sj = cm_t[i], cm_t[j]                 # (M, S)
        valid = (si != miss) & (sj != miss)
        num_present = valid.sum(1).clamp(min=1).double()
        differ = valid & (si != sj)
        one_unedited = differ & ((si == 0) | (sj == 0))
        both_edited = differ & ~one_unedited
        d_ij = (one_unedited.sum(1).double()
                + 2.0 * both_edited.sum(1).double()) / num_present
        return (0.5 * (d_root_t[i] + d_root_t[j] - d_ij)).clamp_(min=0.0)

    return tau_fn


def _log_uniform_rank(shape, n_avail, kw):
    """1-based ranks ``floor(exp(U log(n_avail + 1)))``, each in ``[1, n_avail]``."""
    log_top = math.log(n_avail + 1.0)
    return torch.exp(torch.rand(*shape, **kw).double() * log_top).long().clamp_(1, n_avail)


def _resolve_max_k(max_k, n_leaves):
    """Neighbourhood cap as a leaf count: None stays None, a value <= 1 is a fraction of the leaves."""
    if max_k is None:
        return None
    v = float(max_k)
    if v <= 0:
        raise ValueError("max_k must be positive, got %r" % (max_k,))
    return max(3, int(round(v * n_leaves)) if v <= 1.0 else int(v))


def _neighbourhood_draw(M, n_avail, kw, k_max=None):
    """Three distinct sorted neighbour columns below a size ``k`` drawn log-uniformly over ``[3, k_max]``.

    ``k_max`` defaults to ``n_avail`` (every column of the ranking).
    """
    top = n_avail if k_max is None else min(max(int(k_max), 3), n_avail)
    k = _log_uniform_rank((M,), top, kw)
    k = k.clamp_(min=min(3, n_avail))
    # three distinct columns below k: draw from shrinking ranges and step over taken ones
    x1 = (torch.rand(M, **kw).double() * k).long()
    x2 = (torch.rand(M, **kw).double() * (k - 1)).long()
    x2 = x2 + (x2 >= x1).long()
    lo, hi = torch.minimum(x1, x2), torch.maximum(x1, x2)
    x3 = (torch.rand(M, **kw).double() * (k - 2)).long()
    x3 = x3 + (x3 >= lo).long()
    x3 = x3 + (x3 >= hi).long()
    col, _ = torch.stack([x1, x2, x3], dim=1).sort(dim=1)
    return col


def sample_quartets_neighbourhood(n_leaves, n_quartets, generator=None, device=None,
                                  order=None, tau_knn=None, tau_fn=None, max_k=None,
                                  oversample=3, max_rounds=8):
    """Quartets from one leaf's neighbourhood: ``b, c, d`` drawn together from its ``k`` nearest.

    Each proposal draws a leaf ``a`` and a neighbourhood size ``k``
    log-uniform over ``[3, K]``, then three distinct leaves uniformly from
    ``a``'s ``k`` nearest neighbours; the nearest of the three becomes ``b``.
    All four leaves therefore sit in one neighbourhood whose size is spread
    evenly over every doubling, so edges above small clades and deep edges get
    quartets alike. ``K`` is ``max_k`` (None: every other leaf; a value <= 1:
    that fraction of the leaves), capped at the columns of ``order``. A
    proposal is accepted when the four leaves are distinct and
    ``max(tau_ac, tau_ad, tau_bc, tau_bd) < tau_ab``; after ``max_rounds`` the
    remainder is filled with uniform quartets.

    Args:
        n_leaves: leaves are taxa ``1..n_leaves``.
        n_quartets: batch size B.
        generator: torch CPU generator, or None.
        device: device of the returned tensor.
        order: (n, K) neighbour ranking by descending tau (0-based leaf indices).
        tau_knn: (n, K) tau of each leaf with the neighbours in ``order``; the
            pairs with ``a`` are read from it.
        tau_fn: callable ``(i, j) -> tau`` on 0-based leaf indices, for the
            pairs ``(b, c)`` and ``(b, d)``.
        max_k: cap on the neighbourhood size (see above).
        oversample: proposals drawn per still-needed quartet each round.
        max_rounds: rounds before the remainder is filled with uniform quartets.

    Returns:
        tuple: ``(quartets, stats)``: a (B, 4) long tensor of distinct 1-based
        taxon indices, and ``accept_rate`` / ``fallback_frac``.
    """
    if n_leaves < 4:
        raise ValueError("need at least 4 leaves to sample quartets, got %d" % n_leaves)
    if order is None or tau_knn is None or tau_fn is None:
        raise ValueError("sample_quartets_neighbourhood needs `order`, `tau_knn` and `tau_fn`")
    order_t = torch.as_tensor(order, dtype=torch.long)
    tau_knn_t = torch.as_tensor(tau_knn, dtype=torch.float64)
    n_avail = order_t.shape[1]
    k_max = _resolve_max_k(max_k, n_leaves)
    kw = {"generator": generator} if generator is not None else {}

    kept, got, drawn, n_ok = [], 0, 0, 0
    for _ in range(max_rounds):
        need = n_quartets - got
        if need <= 0:
            break
        M = max(int(need * oversample), 1)
        a = torch.randint(0, n_leaves, (M,), **kw)
        drawn += M
        col = _neighbourhood_draw(M, n_avail, kw, k_max)
        rows = a[:, None].expand(M, 3)
        nb = order_t[rows, col]
        t_a = tau_knn_t[rows, col]                  # tau of a with b, c, d
        b, c, d = nb.unbind(1)
        q = torch.stack([a, b, c, d], dim=1)
        srt, _ = q.sort(dim=1)
        distinct = (srt[:, 1:] > srt[:, :-1]).all(dim=1)
        t_b = tau_fn(torch.cat([b, b]), torch.cat([c, d])).view(2, M).to(t_a.device)
        cross = torch.maximum(torch.maximum(t_a[:, 1], t_a[:, 2]),
                              torch.maximum(t_b[0], t_b[1]))
        ok = distinct & (cross < t_a[:, 0])
        n_ok += int(ok.sum())
        take = q[ok][:need]
        if take.numel():
            kept.append(take)
            got += take.shape[0]

    n_fallback = n_quartets - got
    if n_fallback > 0:
        kept.append(sample_distinct(n_leaves, 4, n_fallback, generator, None) - 1)
    out = torch.cat(kept)[:n_quartets] + 1
    if device is not None:
        out = out.to(device)
    return out, {"accept_rate": (n_ok / drawn) if drawn else 0.0,
                 "fallback_frac": n_fallback / max(n_quartets, 1)}


def sampler_reach(sampler_strategy, sampler_max_k, n_leaves):
    """Neighbours per leaf the quartet sampler reads, or None when it needs no ranking.

    For the neighbourhood sampler this is the cap ``sampler_max_k`` (a count,
    a fraction <= 1 of ``n_leaves``, or every other leaf when None). A k-NN
    table (``knn_columns``) with this many columns yields exactly the same
    proposals as the full ranking, so low-memory mode uses it as the default
    ``knn_columns``. The uniform sampler reads no ranking.
    """
    check_strategy(sampler_strategy)
    if sampler_strategy == "uniform":
        return None
    k = _resolve_max_k(sampler_max_k, n_leaves)
    return n_leaves - 1 if k is None else min(k, n_leaves - 1)


def check_strategy(sampler_strategy):
    """Raises ``ValueError`` unless *sampler_strategy* names a registered sampler."""
    if sampler_strategy not in SAMPLERS:
        raise ValueError("sampler_strategy must be one of %s, got %r"
                         % (", ".join(SAMPLERS), sampler_strategy))


def n_distinct_quartets(n):
    """Number of distinct leaf quartets, ``C(n, 4)``."""
    return n * (n - 1) * (n - 2) * (n - 3) // 24


def default_batch_size(n):
    """Default quartets per step: ``min(QUARTETS_PER_CELL * n, C(n, 4))``."""
    return min(QUARTETS_PER_CELL * n, n_distinct_quartets(n))


def _check_est_dists(est_dists, n_points):
    """``est_dists`` as an array, checked to be the founder-augmented (n_points, n_points) matrix."""
    if est_dists is None:
        raise ValueError(
            "the quartet sampler needs est_dists (the (n_points, n_points) "
            "founder-augmented distance matrix); got None")
    est_arr = np.asarray(est_dists)
    if est_arr.shape != (n_points, n_points):
        raise ValueError(
            "est_dists must be the (n_points, n_points) founder-augmented "
            "matrix, got %r for n_points=%d; passing the leaf-only "
            "(n, n) block would shift every index by one"
            % (est_arr.shape, n_points))
    return est_arr


def _uniform_sampler(degenerate=False):
    """The uniform sampler; ``degenerate`` marks the neighbourhood sampler's fallback."""
    sampler_stats = {}

    def sampler(n_leaves, n_quartets, generator=None, device=None):
        q, s = sample_quartets_uniform(n_leaves, n_quartets, generator=generator, device=device)
        sampler_stats.update(s)
        return q

    sampler_config = {"strategy": "uniform"}
    if degenerate:
        sampler_config["degenerate"] = True
    return sampler, sampler_config, sampler_stats


@register_sampler("uniform")
def build_uniform_sampler(est_dists, n_points, **_unused):
    """``sampler_strategy="uniform"``: :func:`sample_quartets_uniform`, no distances needed."""
    return _uniform_sampler()


@register_sampler("neighbourhood")
def build_neighbourhood_sampler(est_dists, n_points, *, sampler_max_k=SAMPLER_MAX_K,
                                knn_columns=None, character_matrix=None, missing_state=-1,
                                device=None):
    """``sampler_strategy="neighbourhood"``: :func:`sample_quartets_neighbourhood`.

    Keeps each leaf's top-K neighbours with their tau (K = the neighbourhood
    cap, or ``knn_columns`` columns in low-memory mode). Tau for the pairs
    without the first leaf comes from the full tau matrix, or in low-memory mode
    is computed on the fly from the character matrix.
    """
    n = n_points - 1
    k_max = _resolve_max_k(sampler_max_k, n)
    t0 = time.perf_counter()
    if knn_columns is not None:
        if character_matrix is None:
            raise ValueError("knn_columns needs character_matrix (the k-NN tables "
                             "are built from the characters)")
        cm_leaves = np.asarray(character_matrix)[1:]  # drop founder row
        k_table = min(int(knn_columns), n - 1)
        order, tau_knn, d_root = build_tau_neighbours_from_cm(
            cm_leaves, k_table, missing_state=missing_state, device=device)
        tau_fn = _make_tau_fn(cm_leaves, d_root, missing_state=missing_state, device=device)
        tau_source = "knn_%d" % k_table
    else:
        k_table = n - 1 if k_max is None else min(k_max, n - 1)
        order, tau_knn, tau = build_tau_topk(_check_est_dists(est_dists, n_points), k_table)
        tau_t = torch.as_tensor(tau, dtype=torch.float64)

        def tau_fn(i, j):
            return tau_t[i, j]
        tau_source = "est_dists"
    if not bool((tau_knn > 0).any()):
        # Degenerate input: every pair meets at the founder, so no pair is "closer"
        # than any other and the acceptance test can never hold. Happens for an
        # all-zero or fully-missing character matrix. Draw uniformly instead of
        # spinning through max_rounds and silently returning a uniform batch anyway.
        warnings.warn(
            "neighbourhood quartet sampler: tau is entirely zero (no pair is closer "
            "than any other), falling back to uniform sampling", stacklevel=3)
        return _uniform_sampler(degenerate=True)
    order = torch.as_tensor(order, dtype=torch.long)
    tau_knn = torch.as_tensor(tau_knn, dtype=torch.float64)
    logger.info("neighbourhood sampler: top-%d table for n=%d (%.1f MB), max_k=%s, tau %s; %.1fs on %s",
                k_table, n, (order.nelement() * 8 + tau_knn.nelement() * 8) / 1024 / 1024,
                k_max, "from the full matrix" if tau_source == "est_dists" else "on the fly",
                time.perf_counter() - t0, device or "cpu")
    sampler_stats = {}

    def sampler(n_leaves, n_quartets, generator=None, device=None):
        q, s = sample_quartets_neighbourhood(n_leaves, n_quartets, generator=generator,
                                             device=device, order=order, tau_knn=tau_knn,
                                             tau_fn=tau_fn, max_k=k_max)
        sampler_stats.update(s)
        return q

    sampler_config = {"strategy": "neighbourhood", "max_k": sampler_max_k, "table_k": k_table,
                      "tau_source": tau_source}
    return sampler, sampler_config, sampler_stats


def build_quartet_sampler(est_dists, n_points, *, sampler_strategy="neighbourhood",
                          sampler_max_k=SAMPLER_MAX_K, knn_columns=None, character_matrix=None,
                          missing_state=-1, device=None):
    """Build the quartet sampler named by *sampler_strategy* (see :data:`SAMPLERS`).

    Args:
        est_dists: (n_points, n_points) founder-augmented seed distances. May be
            None when ``knn_columns`` and ``character_matrix`` are both given,
            or for the uniform sampler.
        n_points: number of points including the founder.
        sampler_strategy: a key of :data:`SAMPLERS`: "neighbourhood" (default;
            :func:`sample_quartets_neighbourhood`) or "uniform"
            (:func:`sample_quartets_uniform`, which needs no distances).
        sampler_max_k: "neighbourhood" only: cap on the neighbourhood size
            (None = every other leaf; a value <= 1 is a fraction of the leaves).
            Each leaf keeps only that many neighbours.
        knn_columns: low-memory fits: keep only this many neighbours per leaf
            (O(n k) memory), ranked from ``character_matrix``, and compute tau on
            the fly; None works from ``est_dists``.
        character_matrix: (n_points, S) states with the founder at row 0; used
            for the k-NN table and the on-the-fly tau when ``knn_columns`` is set.
        missing_state: value marking missing entries.
        device: device for the k-NN table and the on-the-fly tau.

    Returns:
        tuple: ``(sampler, sampler_config, sampler_stats)``. ``sampler`` is called
        as ``sampler(n_leaves, n_quartets, generator=None, device=None)``, returns
        a (B, 4) tensor and updates the ``sampler_stats`` dict in place on every
        call. ``sampler_config`` records the settings used.

    Raises:
        ValueError: for an unknown strategy, or if ``est_dists`` is needed but
            missing or its shape is not (n_points, n_points) (a leaf-only matrix
            would shift every index by one).
    """
    check_strategy(sampler_strategy)
    return SAMPLERS[sampler_strategy](
        est_dists, n_points, sampler_max_k=sampler_max_k, knn_columns=knn_columns,
        character_matrix=character_matrix, missing_state=missing_state, device=device)
