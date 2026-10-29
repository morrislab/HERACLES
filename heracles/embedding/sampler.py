"""The banded local-outgroup quartet sampler.

Quartets are drawn so that each cell is paired with relatives ranked by
estimated common-ancestor depth, from close neighbours out to a local
outgroup. :func:`build_banded_sampler` builds the sampler from a distance
matrix or, without one, from k-NN tables computed directly from the
character matrix; :func:`sample_quartets_banded` draws a batch.
"""

import logging
import math
import time as _time
import warnings

import numpy as np
import torch

from heracles.constants import (HAMMING_ROW_CHUNK, LOCAL_RING_MULT, QUARTETS_PER_CELL,
                                SAMPLER_RAMP_STEPS)
from heracles.embedding.ctmc import (
    sample_distinct,
)

logger = logging.getLogger("heracles")


__all__ = [
    'build_tau_neighbours',
    'build_tau_neighbours_from_cm',
    'sample_quartets_banded',
    'banded_sampler_reach',
    'n_distinct_quartets',
    'default_batch_size',
    'build_banded_sampler',
]


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
    d = np.asarray(dists, dtype=np.float64)
    d0 = d[0, 1:]
    D = d[1:, 1:]
    tau = 0.5 * (d0[:, None] + d0[None, :] - D)
    np.clip(tau, 0.0, None, out=tau)
    np.fill_diagonal(tau, 0.0)

    n = tau.shape[0]
    # Rank by descending tau, then drop each leaf's own column wherever it landed.
    order_full = np.argsort(-tau, axis=1, kind="stable")
    keep = order_full != np.arange(n)[:, None]
    order = order_full[keep].reshape(n, n - 1)
    return order, tau


def _topk_stable_rows(tau_chunk, k, self_idx):
    """Per row, the ``k`` largest-tau leaves (excluding the leaf itself), ordered as a stable full sort would."""
    nrows, n = tau_chunk.shape
    order = np.empty((nrows, k), dtype=np.int64)
    vals = np.empty((nrows, k), dtype=np.float64)
    for r in range(nrows):
        neg = -tau_chunk[r]
        neg[self_idx[r]] = np.inf                     # never its own neighbour
        if k >= n - 1:
            idx = np.argsort(neg, kind="stable")[:k]
        else:
            part = np.argpartition(neg, k - 1)[:k]
            v = neg[part].max()                       # k-th smallest value
            below = np.flatnonzero(neg < v)
            ties = np.flatnonzero(neg == v)
            idx = np.concatenate([below, ties[:k - len(below)]])
            idx = idx[np.lexsort((idx, neg[idx]))]  # by (-tau, index)
        order[r] = idx
        vals[r] = -neg[idx]
    return order, vals


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
            # exact stable top-k: everything above the k-th value, then the
            # lowest-index ties at that value, ordered by (-tau, index)
            kth = torch.topk(tau, k, dim=1).values[:, -1]
            above = tau > kth[:, None]
            ties = tau == kth[:, None]
            n_above = above.sum(1)
            take = above | (ties & (ties.cumsum(1) <= (k - n_above)[:, None]))
            sel_idx = take.nonzero(as_tuple=True)[1].view(r, k)     # index-ascending per row
            sel_tau = tau.gather(1, sel_idx)
            srt = torch.sort(-sel_tau, dim=1, stable=True).indices
            order[i0:i1] = sel_idx.gather(1, srt).cpu().numpy()
            tau_knn[i0:i1] = sel_tau.gather(1, srt).cpu().numpy()
    return order, tau_knn, d_root


def _build_tau_neighbours_from_cm_numpy(cm, k, missing_state=-1):
    """Numpy version of :func:`build_tau_neighbours_from_cm` (same outputs), one row block at a time."""
    cm = np.asarray(cm)
    n, S = cm.shape
    k = min(k, n - 1)

    d_root = _edited_fraction(cm, missing_state)

    order = np.empty((n, k), dtype=np.int64)
    tau_knn = np.empty((n, k), dtype=np.float64)

    from heracles.embedding.initialization import seed_distances
    for i0 in range(0, n, HAMMING_ROW_CHUNK):
        i1 = min(i0 + HAMMING_ROW_CHUNK, n)
        D_chunk = seed_distances(cm, np.arange(i0, i1), missing_state=missing_state)
        tau_chunk = 0.5 * (d_root[i0:i1, None] + d_root[None, :] - D_chunk)
        np.clip(tau_chunk, 0.0, None, out=tau_chunk)
        order[i0:i1], tau_knn[i0:i1] = _topk_stable_rows(
            tau_chunk, k, np.arange(i0, i1))

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


def sample_quartets_banded(n_leaves, n_quartets, generator=None, device=None,
                           order=None, tau=None, kmin=2, kmax=16,
                           oversample=3, max_rounds=8, tau_has_signal=None,
                           local_frac=0.0, local_mult_max=None,
                           tau_fn=None):
    """Sample quartets whose first pair are close relatives, so edges above small clades get signal.

    Each proposal draws a rank bound ``k`` log-uniformly in ``[kmin, kmax]``, a
    leaf ``a``, a neighbour ``b`` of ``a`` with tau rank below ``k``, and ``c, d``
    uniformly. For a ``local_frac`` share of proposals ``c`` is instead drawn
    just beyond ``b`` in ``a``'s ranking (a local outgroup). A proposal is
    accepted when the four leaves are distinct and
    ``max(tau_ac, tau_ad, tau_bc, tau_bd) < tau_ab``.

    Args:
        n_leaves: leaves are taxa ``1..n_leaves``.
        n_quartets: batch size B.
        generator: torch CPU generator, or None.
        device: device of the returned tensor.
        order: neighbour ranking from :func:`build_tau_neighbours` or a k-NN variant.
        tau: (n, n) tau matrix; give this or ``tau_fn``.
        kmin, kmax: neighbour-rank band, clamped to the columns of ``order``.
        oversample: proposals drawn per still-needed quartet each round.
        max_rounds: rounds before the remainder is filled with uniform quartets.
        tau_has_signal: precomputed ``(tau > 0).any()``; computed when None.
        local_frac: share of proposals with a local-outgroup ``c``, in [0, 1].
        local_mult_max: if set (>= 1), ``c``'s rank is drawn from ``[r+1, (r+1) m)``,
            with ``r`` the rank of ``b`` and ``m`` log-uniform in
            ``[1, local_mult_max]``; if None, from the fixed ring ``[k, 4k)``.
        tau_fn: callable ``(i, j) -> tau`` on 0-based leaf indices, used when
            ``tau`` is None.

    Returns:
        tuple: ``(quartets, stats)``. ``quartets`` is a (B, 4) long tensor of
        distinct 1-based taxon indices. ``stats`` holds ``accept_rate``,
        ``fallback_frac`` (share filled uniformly), ``local_frac_accepted`` and
        ``degenerate``. When tau is zero everywhere the batch is uniform, a
        warning is issued, ``degenerate`` is True and ``local_frac_accepted`` is
        absent.

    Raises:
        ValueError: if ``n_leaves < 4``, ``order`` is missing, neither ``tau`` nor
            ``tau_fn`` is given, or ``local_frac`` or ``local_mult_max`` is out of
            range.
    """
    if n_leaves < 4:
        raise ValueError(
            "need at least 4 leaves to sample quartets, got %d" % n_leaves)
    if order is None:
        raise ValueError(
            "sample_quartets_banded needs `order` from build_tau_neighbours")
    if tau is None and tau_fn is None:
        raise ValueError(
            "sample_quartets_banded needs `tau` (n, n) or `tau_fn` (callable)")

    if tau_has_signal is None:   # not precomputed by the caller: check per call
        if tau is not None:
            tau_has_signal = (bool((tau > 0).any()) if torch.is_tensor(tau)
                              else bool(np.any(np.asarray(tau) > 0.0)))
        else:
            tau_has_signal = True  # tau_fn mode; assume signal exists
    if not tau_has_signal:
        # Degenerate input: every pair meets at the founder, so no pair is "closer" than
        # any other and the acceptance test max(cross) < tau_ab can never hold. Happens
        # for an all-zero or fully-missing character matrix. Fall back to uniform rather
        # than spinning through max_rounds and silently returning a uniform batch anyway.
        warnings.warn(
            "banded quartet sampler: tau is entirely zero (no pair is closer than any "
            "other), falling back to uniform sampling", stacklevel=2)
        out = sample_distinct(n_leaves, 4, n_quartets, generator, device)
        return out, {"accept_rate": 0.0, "fallback_frac": 1.0, "degenerate": True}

    n_avail = order.shape[1]
    lo = max(1, min(int(kmin), n_avail))
    hi = max(lo, min(int(kmax), n_avail))
    log_lo, log_hi = math.log(lo), math.log(hi)
    local_frac = float(local_frac)
    if not 0.0 <= local_frac <= 1.0:
        raise ValueError("local_frac must be in [0, 1], got %r" % (local_frac,))
    if local_mult_max is not None and float(local_mult_max) < 1.0:
        raise ValueError("local_mult_max must be >= 1, got %r" % (local_mult_max,))

    order_t = torch.as_tensor(order, dtype=torch.long)
    if tau is not None:
        tau_t = torch.as_tensor(tau, dtype=torch.float64)
    else:
        tau_t = None
    _tau = tau_fn if tau_t is None else lambda i, j: tau_t[i, j]
    kw = {"generator": generator} if generator is not None else {}

    kept = []
    got = 0
    drawn = 0
    n_ok = 0
    n_ok_local = 0
    for _ in range(max_rounds):
        need = n_quartets - got
        if need <= 0:
            break
        M = max(int(need * oversample), 1)
        drawn += M

        u = torch.rand(M, **kw)
        k = torch.exp(u * (log_hi - log_lo) + log_lo).long().clamp_(1, n_avail)
        a = torch.randint(0, n_leaves, (M,), **kw)
        r = (torch.rand(M, **kw) * k).long().clamp_(0, n_avail - 1)
        b = order_t[a, r]
        c = torch.randint(0, n_leaves, (M,), **kw)
        d = torch.randint(0, n_leaves, (M,), **kw)
        is_local = torch.zeros(M, dtype=torch.bool)
        if local_frac > 0.0:
            # c from a's rank band [k, LOCAL_RING_MULT k): the nearest leaves outside
            # the band b came from, i.e. the sister-group candidates. Falls back to
            # uniform c when the band would run past the last neighbour.
            is_local = torch.rand(M, **kw) < local_frac
            if local_mult_max is None:
                r_lo = k
                r_hi = torch.minimum(int(LOCAL_RING_MULT) * k, torch.full_like(k, n_avail))
            else:
                # Anchor just past b's own rank r and reach a log-uniform multiple
                # of it, so c concentrates on the next-most-similar cells.
                log_m = torch.rand(M, **kw) * math.log(float(local_mult_max))
                r_lo = r + 1
                r_hi = torch.ceil(r_lo.double() * torch.exp(log_m)).long()
                r_hi = torch.maximum(r_hi, r_lo + 1)
                r_hi = torch.minimum(r_hi, torch.full_like(k, n_avail))
            width = (r_hi - r_lo).clamp_min(0)
            r_c = r_lo + (torch.rand(M, **kw) * width).long()
            r_c = r_c.clamp_(0, n_avail - 1)
            c_local = order_t[a, r_c]
            is_local &= width > 0
            c = torch.where(is_local, c_local, c)

        q = torch.stack([a, b, c, d], dim=1)
        srt, _ = q.sort(dim=1)
        distinct = (srt[:, 1:] > srt[:, :-1]).all(dim=1)

        if tau_t is None:
            # one gather for all five pairs (fewer launches when tau_fn is on a GPU)
            t_all = _tau(torch.cat([a, a, a, b, b]), torch.cat([b, c, d, c, d])).view(5, M)
            t_ab = t_all[0]
            cross = torch.maximum(torch.maximum(t_all[1], t_all[2]),
                                  torch.maximum(t_all[3], t_all[4]))
        else:
            t_ab = _tau(a, b)
            cross = torch.maximum(torch.maximum(_tau(a, c), _tau(a, d)),
                                  torch.maximum(_tau(b, c), _tau(b, d)))
        ok = distinct & (cross < t_ab).to(distinct.device)
        # Count acceptances before truncating to `need`: `got/drawn` would be capped
        # at 1/oversample and report the fill rate rather than the acceptance rate.
        n_ok += int(ok.sum())
        n_ok_local += int((ok & is_local).sum())

        take = q[ok][:need]
        if take.numel():
            kept.append(take)
            got += take.shape[0]

    n_fallback = n_quartets - got
    if n_fallback > 0:
        # Never hang: top up with uniform quartets and report how many.
        kept.append(sample_distinct(n_leaves, 4, n_fallback, generator, None) - 1)

    out = (torch.cat(kept)[:n_quartets] + 1)
    if device is not None:
        out = out.to(device)
    stats = {"accept_rate": (n_ok / drawn) if drawn else 0.0,
             "fallback_frac": n_fallback / max(n_quartets, 1),
             "local_frac_accepted": (n_ok_local / n_ok) if n_ok else 0.0,
             "degenerate": False}
    return out, stats


def banded_sampler_reach(band_kmax, band_kmax_end=None, band_local_frac=0.0,
                         band_local_mult_max=None):
    """Largest neighbour rank :func:`sample_quartets_banded` can read with these settings.

    A k-NN table (``sampler_knn``) with at least this many columns yields
    exactly the same proposals as the full ranking; low-memory mode uses it as
    the default ``sampler_knn``.
    """
    kmax = int(band_kmax)
    if band_kmax_end is not None:
        kmax = max(kmax, int(band_kmax_end))
    if float(band_local_frac) <= 0.0:
        return kmax
    mult = LOCAL_RING_MULT if band_local_mult_max is None else float(band_local_mult_max)
    return int(math.ceil(kmax * mult))


def n_distinct_quartets(n):
    """Number of distinct leaf quartets, ``C(n, 4)``."""
    return n * (n - 1) * (n - 2) * (n - 3) // 24


def default_batch_size(n):
    """Default quartets per step: ``min(QUARTETS_PER_CELL * n, C(n, 4))``."""
    return min(QUARTETS_PER_CELL * n, n_distinct_quartets(n))


def _band_defaults(**given):
    """Fill ``None`` band settings from the ``HeraclesConfig`` defaults."""
    from heracles.config import HeraclesConfig
    defaults = HeraclesConfig()
    return {k: (getattr(defaults, k) if v is None else v) for k, v in given.items()}


def build_banded_sampler(est_dists, n_points, *, band_kmin=None, band_kmax=None,
                         band_kmax_end="default", band_local_frac=None,
                         band_local_mult_max="default", ramp_steps=SAMPLER_RAMP_STEPS,
                         sampler_knn=None, character_matrix=None,
                         missing_state=-1,
                         device=None):
    """Build the banded quartet sampler used by the quartet objective.

    Band settings left unset take the :class:`heracles.config.HeraclesConfig`
    defaults. ``band_kmax_end`` and ``band_local_mult_max`` accept None as a
    real value, so their "unset" marker is the string ``"default"``.

    Args:
        est_dists: (n_points, n_points) founder-augmented seed distances. May be
            None when ``sampler_knn`` and ``character_matrix`` are both given.
        n_points: number of points including the founder.
        band_kmin, band_kmax, band_local_frac, band_local_mult_max: see
            :func:`sample_quartets_banded`.
        band_kmax_end: if set, ``kmax`` ramps linearly from ``band_kmax`` to this
            value over the first ``ramp_steps`` sampler calls (one call per step).
        sampler_knn: if an int k, keep only the top-k neighbours per leaf (O(n k)
            memory) and compute tau on the fly; None keeps the full (n, n) tau.
        character_matrix: (n_points, S) states with the founder at row 0; used
            for the on-the-fly tau when ``sampler_knn`` is set.
        missing_state: value marking missing entries.
        device: device for the k-NN table and the on-the-fly tau.

    Returns:
        tuple: ``(sampler, sampler_config, sampler_stats)``. ``sampler`` is called
        as ``sampler(n_leaves, n_quartets, generator=None, device=None)``, returns
        a (B, 4) tensor and updates the ``sampler_stats`` dict in place on every
        call. ``sampler_config`` records the band settings used.

    Raises:
        ValueError: if ``est_dists`` is needed but missing, or its shape is not
            (n_points, n_points) (a leaf-only matrix would shift every index by one).
    """
    d = _band_defaults(band_kmin=band_kmin, band_kmax=band_kmax, band_local_frac=band_local_frac)
    band_kmin, band_kmax, band_local_frac = d["band_kmin"], d["band_kmax"], d["band_local_frac"]
    if band_kmax_end == "default":
        band_kmax_end = _band_defaults(band_kmax_end=None)["band_kmax_end"]
    if band_local_mult_max == "default":
        band_local_mult_max = _band_defaults(band_local_mult_max=None)["band_local_mult_max"]
    _band_tau_fn = None
    if sampler_knn is not None:
        k = int(sampler_knn)
        if character_matrix is None:
            raise ValueError("sampler_knn needs character_matrix (the k-NN tables "
                             "are built from the characters)")
        cm_leaves = np.asarray(character_matrix)[1:]  # drop founder row
        _t_knn = _time.perf_counter()
        _band_order, _tau_knn, _d_root = build_tau_neighbours_from_cm(
            cm_leaves, k, missing_state=missing_state, device=device)
        logger.info("k-NN table from the character matrix: %.1fs on %s",
                    _time.perf_counter() - _t_knn, device or "cpu")
        _band_tau_fn = _make_tau_fn(
            cm_leaves, _d_root, missing_state=missing_state, device=device)
        _band_order = torch.as_tensor(_band_order, dtype=torch.long)
        _band_tau = None
        _band_has_signal = bool((_tau_knn > 0).any())
        logger.info("k-NN sampler: k=%d, n=%d (%.1f MB order + on-the-fly tau)",
                     k, n_points - 1,
                     _band_order.nelement() * 8 / 1024 / 1024)
    else:
        if est_dists is None:
            raise ValueError(
                "the banded sampler needs est_dists (the (n_points, n_points) "
                "founder-augmented distance matrix); got None")
        est_arr = np.asarray(est_dists)
        if est_arr.shape != (n_points, n_points):
            raise ValueError(
                "est_dists must be the (n_points, n_points) founder-augmented "
                "matrix, got %r for n_points=%d; passing the leaf-only "
                "(n, n) block would shift every index by one"
                % (est_arr.shape, n_points))
        _band_order, _band_tau = build_tau_neighbours(est_arr)
        _band_order = torch.as_tensor(_band_order, dtype=torch.long)
        _band_tau = torch.as_tensor(_band_tau, dtype=torch.float64)
        _band_has_signal = bool((_band_tau > 0).any())
    _sampler_step = [0]
    sampler_stats = {}

    if band_kmax_end is not None and band_kmax_end != band_kmax:
        _kmax_start = float(band_kmax)
        _kmax_end = float(band_kmax_end)

        def sampler(n_leaves, n_quartets, generator=None, device=None):
            t = _sampler_step[0]
            _sampler_step[0] = t + 1
            kmax_t = int(round(_kmax_start + (_kmax_end - _kmax_start)
                               * min(t / float(ramp_steps), 1.0)))
            kmax_t = max(int(band_kmin), kmax_t)
            q, s = sample_quartets_banded(
                n_leaves, n_quartets, generator=generator, device=device,
                order=_band_order, tau=_band_tau, kmin=band_kmin, kmax=kmax_t,
                tau_has_signal=_band_has_signal, local_frac=band_local_frac,
                local_mult_max=band_local_mult_max, tau_fn=_band_tau_fn)
            s["kmax_effective"] = kmax_t
            sampler_stats.update(s)
            return q
    else:
        def sampler(n_leaves, n_quartets, generator=None, device=None):
            q, s = sample_quartets_banded(
                n_leaves, n_quartets, generator=generator, device=device,
                order=_band_order, tau=_band_tau, kmin=band_kmin, kmax=band_kmax,
                tau_has_signal=_band_has_signal, local_frac=band_local_frac,
                local_mult_max=band_local_mult_max, tau_fn=_band_tau_fn)
            sampler_stats.update(s)
            return q

    _tau_source = "knn_%d" % sampler_knn if sampler_knn else "est_dists"
    sampler_config = {"kmin": int(band_kmin), "kmax": int(band_kmax),
                      "kmax_end": int(band_kmax_end) if band_kmax_end is not None else None,
                      "local_frac": float(band_local_frac),
                      "local_mult_max": (float(band_local_mult_max)
                                         if band_local_mult_max is not None else None),
                      "tau_source": _tau_source}
    return sampler, sampler_config, sampler_stats
