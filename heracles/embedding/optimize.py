"""The optimisation loop and distance read-out for the hyperbolic embedding.

:func:`optimize` runs Riemannian Adam on the Lorentz model, with a two-stage
adaptive tree-likeness penalty and an optional adaptive ultrametric penalty.
:func:`pairwise_dists` and :func:`iter_dist_rows` compute the embedding's
distance matrix, whole or in row blocks.
"""

import logging
import time as _time

import geoopt
import numpy as np
import torch

from heracles.embedding.ctmc import gromov_quartet_penalty
from heracles.embedding.initialization import lorentz_manifold
from heracles.embedding.priors import missing_log_prior, resolve_prior
from heracles.constants import (DIST_ROW_CHUNK, MASK_LOGIT, SEED_OFFSET_EVAL,
                                SEED_OFFSET_PENALTY, SEED_OFFSET_SAMPLER)
from heracles.runinfo import resolve_device

logger = logging.getLogger("heracles")


__all__ = [
    'ULTRAMETRIC_LR_DUAL',
    'ULTRAMETRIC_MAX_WEIGHT',
    'ULTRAMETRIC_MIN_WEIGHT',
    'ULTRAMETRIC_SHRINK_FACTOR',
    'TREELIKE_N_QUARTETS',
    'TREELIKE_LR_DUAL',
    'TREELIKE_MAX_WEIGHT',
    'TREELIKE_MIN_WEIGHT',
    'iter_dist_rows',
    'pairwise_dists',
    'optimize',
]


# Dual-ascent controller settings for the two penalties (fixed; the
# user-facing knobs are in HeraclesConfig).
ULTRAMETRIC_LR_DUAL = 1.0
ULTRAMETRIC_MAX_WEIGHT = 10.0
ULTRAMETRIC_MIN_WEIGHT = 0.05
#: 'auto' ultrametric target = this fraction of the initial depth CV
ULTRAMETRIC_SHRINK_FACTOR = 0.75
TREELIKE_N_QUARTETS = 100      # independent quartets for objectives without their own batch
TREELIKE_LR_DUAL = 1.0
TREELIKE_MAX_WEIGHT = 100.0
TREELIKE_MIN_WEIGHT = 0.05

_INIT_FLOOR = 1e-6
_CONVERGED_ZERO = 1e-30


class _DualAscent:
    """Subgradient (dual-ascent) update of a penalty weight toward a target:
    ``w <- clip(w + lr * (measured - target), min_weight, max_weight)``."""

    def __init__(self, *, lr, target, min_weight, max_weight, weight=0.0):
        self.lr, self.target = lr, target
        self.min_weight, self.max_weight = min_weight, max_weight
        self.weight = weight

    def step(self, measured):
        self.weight = max(self.min_weight, min(
            self.weight + self.lr * (measured - self.target), self.max_weight))
        return self.weight


def _lorentz_boost_to_origin(points, anchor):
    """Apply the Lorentz boost (an isometry) that maps ``anchor`` to (1, 0, ..., 0).

    Puts the founder at the origin so it can be frozen without changing the loss.
    """
    a0 = anchor[0]
    a_s = anchor[1:]
    x0 = points[:, 0]
    x_s = points[:, 1:]
    a_dot_x = (a_s.unsqueeze(0) * x_s).sum(dim=1)
    new_t = a0 * x0 - a_dot_x
    new_s = (-a_s.unsqueeze(0) * x0.unsqueeze(1)
             + x_s
             + a_s.unsqueeze(0) * a_dot_x.unsqueeze(1) / (1 + a0))
    return torch.cat([new_t.unsqueeze(1), new_s], dim=1)


def _relative_change(current, previous):
    """Relative change between two loss values, safe at zero."""
    if abs(previous) < _CONVERGED_ZERO:
        return 0.0 if abs(current) < _CONVERGED_ZERO else float("inf")
    return abs(current - previous) / abs(previous)


def iter_dist_rows(coords, *, radius=1.0, chunk=DIST_ROW_CHUNK, device=None, col_chunk=16384):
    """Yield the Lorentz distance matrix of ``coords`` in row blocks of *chunk* rows.

    *coords* is ``(m, d+1)`` (numpy or tensor); *radius* scales the distances;
    *col_chunk* columns per inner product bound peak memory.  Yields
    ``(i0, i1, block)``: rows ``i0:i1`` as an ``(i1 - i0, m)`` float64 numpy array
    with an exactly zero diagonal, matching ``manifold.dist`` to float64 round-off.
    """
    pts = (coords.detach().to(torch.float64) if torch.is_tensor(coords)
           else torch.as_tensor(np.asarray(coords), dtype=torch.float64))
    if device is not None:
        pts = pts.to(resolve_device(device))
    m = pts.shape[0]
    j_pts = pts.clone()
    j_pts[:, 0] = -j_pts[:, 0]                     # J y: Minkowski sign on the time-like axis
    with torch.no_grad():
        for i0 in range(0, m, chunk):
            i1 = min(i0 + chunk, m)
            x = pts[i0:i1]
            out = torch.empty((i1 - i0, m), dtype=torch.float64, device=pts.device)
            for j0 in range(0, m, col_chunk):
                j1 = min(j0 + col_chunk, m)
                inner = x @ j_pts[j0:j1].T                   # <x, y>_L, <= -1 on the manifold
                out[:, j0:j1] = radius * torch.acosh(torch.clamp(-inner, min=1.0))
            idx = torch.arange(i0, i1, device=pts.device)
            out[idx - i0, idx] = 0.0
            yield i0, i1, out.cpu().numpy()


def pairwise_dists(coords, *, radius=1.0, dtype=np.float64, chunk=DIST_ROW_CHUNK, device=None):
    """Return the full ``(m, m)`` Lorentz distance matrix of ``coords`` (zero diagonal).

    Built from :func:`iter_dist_rows`, so in-memory and streamed read-outs see the
    same numbers. ``coords`` may be a numpy array or a torch tensor on any device.
    """
    m = coords.shape[0]
    Dg = np.empty((m, m), dtype=dtype)
    for i0, i1, block in iter_dist_rows(coords, radius=radius, chunk=chunk, device=device):
        Dg[i0:i1] = block
    return Dg


def optimize(states, dimension, *, loss_fn,
              has_missing=False, steps=1500, lr=0.02, seed=0,
              mutation_rate_init=1.0, learn_mutation_rate=False,
              init_points=None, init_meta=None, init_time=0.0,
              snapshot_epochs=None,
              heritable_eps_init=0.0, stochastic_delta_init=0.0,
              learn_heritable_eps=False, learn_stochastic_delta=False,
              stochastic_delta_per_site=False,
              site_groups=None,
              heritable_eps_prior_variance="auto",
              stochastic_delta_prior_variance="auto",
              convergence_thresh=None,
              patience=3, check_every=100,
              lr_schedule=None,
              ultrametric_penalty=0,
              ultrametric_target_cv=0.25,
              ultrametric_lr_dual=1.0,
              ultrametric_max_weight=10.0,
              treelike_penalty_n_quartets=100,
              treelike_penalty_target_delta=0.01,
              treelike_penalty_lr_dual=1.0,
              treelike_penalty_max_weight=100.0,
              treelike_penalty_min_weight=0.05,
              min_iterations=0,
              device=None,
              dist_dtype=np.float64,
              materialize_dists=True,
              grad_accum=1):
    """Fit the embedding with Riemannian Adam on the Lorentz model and return its distances.

    Called by :func:`heracles.fit.embed`, which maps the fields of
    :class:`heracles.config.HeraclesConfig` onto these keyword arguments (see that
    class for meanings and defaults). The founder (row 0) is fixed at the origin;
    the leaf coordinates are learned, and optionally the mutation rate,
    ``heritable_eps`` and ``stochastic_delta``. Stage 1 fits the likelihood alone
    until it converges or uses half of ``steps``; stage 2 adds the tree-likeness
    penalty, whose weight adapts toward ``treelike_penalty_target_delta``.

    Args:
        loss_fn: callable ``(points, manifold, radius, mutation_rate,
            heritable_eps, stochastic_delta, rng) -> (loss, numel)``, e.g. from
            :func:`heracles.embedding.objective.quartet_loss_fn`.
        init_points: (m, d+1) starting coordinates (required).
        stochastic_delta_init: a scalar, or a fixed (m, S) per-cell table.
        site_groups: optional (S,) cassette index per site; with
            ``stochastic_delta_per_site`` one dropout rate is learned per cassette.
        materialize_dists: if False, ``leaf_dists`` is None and ``meta["coords"]`` holds the result.

    Returns ``(leaf_dists, meta)``: the (n, n) leaf distances (or None) and a dict
    with the fitted rates, ``coords``, ``history``, ``n_iter``, ``converged`` and
    timings. Raises ``ValueError`` if ``init_points`` is missing, the delta table
    has the wrong shape or is set to be learned, or the data have missing entries
    while both missing-data rates are zero and fixed.
    """
    torch.manual_seed(seed)
    dev = resolve_device(device)
    manifold, _ = lorentz_manifold(dimension)

    _delta_init_arr = np.asarray(stochastic_delta_init, dtype=np.float64)
    if _delta_init_arr.ndim == 2:
        # a fixed per-cell, per-site dropout table (founder row 0 included)
        if learn_stochastic_delta:
            raise ValueError("a per-cell stochastic_delta table cannot be learned; "
                             "pass learn_stochastic_delta=False")
        if _delta_init_arr.shape != tuple(states.shape):
            raise ValueError("per-cell stochastic_delta must be (m, S) = %s, got %s"
                             % (tuple(states.shape), _delta_init_arr.shape))
    elif _delta_init_arr.ndim != 0:
        raise ValueError("stochastic_delta_init must be a scalar or an (m, S) table")
    if has_missing:
        eps_ok = heritable_eps_init > 0 or learn_heritable_eps
        delta_ok = bool((_delta_init_arr > 0).any()) or learn_stochastic_delta
        if not (eps_ok or delta_ok):
            raise ValueError(
                "states contain missing entries (-1) but heritable_eps and "
                "stochastic_delta are both zero with learning off: the "
                "missing-data likelihood is -inf. Set a positive init value "
                "or enable learning.")

    if init_points is None:
        raise ValueError("optimize needs the starting coordinates (heracles.embedding.initialization.init_points)")
    _init_meta = dict(init_meta or {})
    all_init = torch.as_tensor(init_points, dtype=torch.float64).to(dev)

    # Fix the founder (row 0) at the north pole to break the translational
    # gauge freedom that otherwise lets coordinates drift to infinity without
    # changing the loss.
    with torch.no_grad():
        all_init = _lorentz_boost_to_origin(all_init, all_init[0])
        all_init = manifold.projx(all_init)
    founder = all_init[0].detach()
    leaf_points = geoopt.ManifoldParameter(all_init[1:].clone(), manifold=manifold)

    def _all_points():
        return torch.cat([founder.unsqueeze(0), leaf_points], dim=0)

    # Global scale of the embedding.  Fixed at 1: the scale and the mutation
    # rate are one gauge, so only one of them can be a parameter.
    radius = torch.tensor(1.0, dtype=torch.float64, device=dev)

    log_mutation_rate = torch.log(torch.tensor(float(mutation_rate_init), device=dev))
    if learn_mutation_rate:
        log_mutation_rate = torch.nn.Parameter(log_mutation_rate.clone())
    params = ([leaf_points]
              + ([log_mutation_rate] if learn_mutation_rate else []))

    if learn_heritable_eps:
        log_eps = torch.nn.Parameter(
            torch.log(torch.tensor(max(float(heritable_eps_init), _INIT_FLOOR),
                                   device=dev)))
        params.append(log_eps)
    else:
        heritable_eps_const = torch.tensor(float(heritable_eps_init),
                                 dtype=torch.float64, device=dev)

    _delta_groups = None
    if learn_stochastic_delta:
        _d0 = min(max(float(stochastic_delta_init), _INIT_FLOOR), 1 - _INIT_FLOOR)
        if stochastic_delta_per_site:
            # one logit per site (or per cassette, expanded to sites each step),
            # all started at the shared init value; the likelihood kernels
            # broadcast an (S,) delta over the (B, S) site axis
            if site_groups is not None:
                _delta_groups = torch.as_tensor(np.asarray(site_groups), dtype=torch.long,
                                                device=dev)
                _n_rates = int(_delta_groups.max()) + 1
            else:
                _n_rates = states.shape[1]
            _d0_t = torch.full((_n_rates,), _d0, dtype=torch.float64, device=dev)
        else:
            _d0_t = torch.tensor(_d0, device=dev)
        logit_delta = torch.nn.Parameter(torch.logit(_d0_t))
        params.append(logit_delta)
    else:
        stochastic_delta_const = torch.as_tensor(_delta_init_arr, dtype=torch.float64,
                                                 device=dev)

    def _current_delta():
        """(delta for the kernels, delta the prior sees): per-cassette rates
        are expanded to sites for the first and kept per cassette for the
        second, so the Beta prior counts each learned rate once."""
        if not learn_stochastic_delta:
            return stochastic_delta_const, None
        d = torch.sigmoid(logit_delta)
        if _delta_groups is not None:
            return d[_delta_groups], d
        return d, d

    allele_prior_param = getattr(loss_fn, "allele_prior_param", None)
    if allele_prior_param is not None:
        params.append(allele_prior_param)

    initiation_ratio_param = getattr(loss_fn, "initiation_ratio_param", None)
    if initiation_ratio_param is not None:
        params.append(initiation_ratio_param)

    eps_gamma = (resolve_prior(heritable_eps_init,
                                heritable_eps_prior_variance, "gamma")
                 if learn_heritable_eps else None)
    delta_beta = (resolve_prior(stochastic_delta_init,
                                 stochastic_delta_prior_variance, "beta")
                  if learn_stochastic_delta else None)

    opt = geoopt.optim.RiemannianAdam(params, lr=lr)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps // 2, eta_min=lr * 0.01)
    else:
        scheduler = None

    # Coordinate snapshots: {epoch: (m, d) array}, taken at snapshot_epochs.
    # Becomes coords_history_ on the estimator and snapshot_<epoch> in the npz.
    snapshot_at = set(int(e) for e in (snapshot_epochs or []))
    snapshots = {}

    def _snapshot(epoch):
        with torch.no_grad():
            snapshots[epoch] = _all_points().detach().cpu().numpy().copy()

    if 0 in snapshot_at:
        _snapshot(0)

    rng = torch.Generator()
    rng.manual_seed(seed + SEED_OFFSET_SAMPLER)

    penalty_rng = torch.Generator()
    penalty_rng.manual_seed(seed + SEED_OFFSET_PENALTY)

    # Fixed evaluation batch for a stable convergence check
    _make_eval_batch = getattr(loss_fn, "make_eval_batch", None)
    _has_eval_set = convergence_thresh is not None and _make_eval_batch is not None
    if _has_eval_set:
        _eval_rng = torch.Generator()
        _eval_rng.manual_seed(seed + SEED_OFFSET_EVAL)
        _eval_batch = _make_eval_batch(_eval_rng).to(dev)
        _eval_loss_history = []

    last = None
    loss_history = []
    eps_history = [] if learn_heritable_eps else None
    delta_history = [] if learn_stochastic_delta else None
    init_ratio_history = [] if initiation_ratio_param is not None else None
    _stage = 1
    _stage1_iters = None
    _stage1_converged = False
    _stage2_start_step = None
    _stage_budget = steps // 2

    if treelike_penalty_min_weight is None:
        treelike_penalty_min_weight = TREELIKE_MIN_WEIGHT
    use_treelike = False
    treelike_ctrl = _DualAscent(lr=treelike_penalty_lr_dual,
                            target=treelike_penalty_target_delta,
                            min_weight=treelike_penalty_min_weight,
                            max_weight=treelike_penalty_max_weight)
    treelike_weight = 0.0
    treelike_weight_history = []
    gromov_delta_history = []

    ultrametric_adaptive = (ultrametric_penalty == "adaptive")
    ultrametric_auto_target = False
    if ultrametric_adaptive and ultrametric_target_cv == "auto":
        ultrametric_auto_target = True
        with torch.no_grad():
            _init_pts = _all_points()
            _init_depths = radius * manifold.dist(
                _init_pts[0].unsqueeze(0), _init_pts[1:])
            _init_cv = float(torch.std(_init_depths) / torch.mean(_init_depths))
            ultrametric_target_cv = _init_cv * ULTRAMETRIC_SHRINK_FACTOR
    if ultrametric_adaptive:
        ultrametric_weight = 0.0
    elif isinstance(ultrametric_penalty, (int, float)) and ultrametric_penalty > 0:
        ultrametric_weight = float(ultrametric_penalty)
    else:
        ultrametric_weight = 0.0
    use_ultrametric = ultrametric_adaptive or ultrametric_weight > 0
    ultrametric_ctrl = _DualAscent(lr=ultrametric_lr_dual, target=ultrametric_target_cv,
                         min_weight=ULTRAMETRIC_MIN_WEIGHT,
                         max_weight=ultrametric_max_weight, weight=ultrametric_weight)
    ultrametric_weight_history = [] if use_ultrametric else None
    _depth_cv_history = [] if use_ultrametric else None

    converged = False
    streak = 0
    actual_steps = steps
    _t_init_end = _time.perf_counter()

    def _transition_to_stage2(step_num, did_converge):
        nonlocal _stage, _stage2_start_step, use_treelike
        nonlocal treelike_weight, streak, converged
        nonlocal _stage1_iters, _stage1_converged, scheduler
        _stage1_iters = step_num + 1
        _stage1_converged = did_converge
        logger.info("stage 1 -> 2 at step %d (stage 1 converged=%s)",
                    step_num + 1, did_converge)
        _stage = 2
        _stage2_start_step = step_num + 1
        use_treelike = True
        treelike_ctrl.weight = 0.0
        treelike_weight = 0.0
        streak = 0
        converged = False
        if _has_eval_set:
            _eval_loss_history.clear()
        if lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=_stage_budget, eta_min=lr * 0.01)

    for step in range(steps):
        opt.zero_grad()

        # grad_accum > 1: the step's batch is k micro-batches drawn in turn
        # from the same generator; each contributes loss_k / k, so the
        # accumulated gradient is that of the mean over the whole batch.  The parameter
        # transforms are rebuilt per micro-batch because backward() frees
        # their graph.
        k_acc = int(grad_accum)
        loss_acc = 0.0
        gromov_acc = 0.0
        for _micro in range(k_acc):
            mutation_rate = torch.exp(log_mutation_rate)
            heritable_eps = torch.exp(log_eps) if learn_heritable_eps else heritable_eps_const
            stochastic_delta, delta_for_prior = _current_delta()
            points = _all_points()
            loss, numel = loss_fn(points, manifold, radius, mutation_rate, heritable_eps, stochastic_delta, rng)

            if learn_heritable_eps or learn_stochastic_delta:
                # one prior per step, spread over the k micro-batches
                loss = loss - missing_log_prior(
                    heritable_eps if learn_heritable_eps else None,
                    delta_for_prior,
                    eps_gamma, delta_beta) / (numel * k_acc)

            if use_treelike:
                # The quartet objective exposes the four-point defect of its own
                # likelihood batch; other objectives get an independent uniform
                # quartet batch.
                gromov_penalty = getattr(loss_fn, "batch_gromov_penalty", None)
                if gromov_penalty is None:
                    gromov_penalty = gromov_quartet_penalty(
                        points, manifold, radius, treelike_penalty_n_quartets,
                        penalty_rng)
                loss = loss + treelike_weight * gromov_penalty
                gromov_acc = gromov_acc + gromov_penalty.detach() / k_acc

            if use_ultrametric:
                # Use the root-to-leaf depths the objective already computed,
                # if it exposes them.
                depths = getattr(loss_fn, "batch_depths", None)
                if depths is None:
                    depths = radius * manifold.dist(points[0].unsqueeze(0), points[1:])
                _depth_var = torch.var(depths)
                loss = loss + ultrametric_weight * _depth_var

            if k_acc == 1:
                loss.backward()
            else:
                (loss / k_acc).backward()
            loss_acc = loss_acc + loss.detach() / k_acc
        loss = loss_acc
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if use_treelike:
            gromov_penalty_val = float(gromov_acc)
            gromov_delta_history.append(gromov_penalty_val)
            treelike_weight = treelike_ctrl.step(gromov_penalty_val)
            treelike_weight_history.append(treelike_weight)
        else:
            gromov_delta_history.append(0.0)
            treelike_weight_history.append(0.0)

        if use_ultrametric:
            with torch.no_grad():
                # std = sqrt of the variance already computed for the loss
                _cv = float(torch.sqrt(_depth_var) / depths.mean())
            _depth_cv_history.append(_cv)
            if ultrametric_adaptive:
                ultrametric_weight = ultrametric_ctrl.step(_cv)
            ultrametric_weight_history.append(ultrametric_weight)

        last = loss.item()
        loss_history.append(last)
        if eps_history is not None:
            eps_history.append(heritable_eps.item())
        if delta_history is not None:
            delta_history.append(stochastic_delta.mean().item())   # per-site: site mean
        if init_ratio_history is not None:
            init_ratio_history.append(float(torch.exp(initiation_ratio_param).item()))

        if (step + 1) in snapshot_at:
            _snapshot(step + 1)

        _eff_min = min_iterations
        if _stage == 2 and _stage2_start_step is not None:
            _eff_min = _stage2_start_step + max(
                min_iterations, check_every * (patience + 1))
        if ((step + 1) % check_every == 0 and step >= check_every
                and step + 1 >= _eff_min):
            _msg = "step %d/%d: loss=%.6g rel=%.3g streak=%d" % (
                step + 1, steps, last,
                _relative_change(last, loss_history[step - check_every]),
                streak)
            if learn_heritable_eps:
                _msg += " eps=%.4g" % heritable_eps.item()
            if learn_stochastic_delta:
                if stochastic_delta_per_site:
                    _msg += " delta=%.4g[%.4g..%.4g]" % (
                        stochastic_delta.mean().item(), stochastic_delta.min().item(),
                        stochastic_delta.max().item())
                else:
                    _msg += " delta=%.4g" % stochastic_delta.item()
            if initiation_ratio_param is not None:
                _msg += " init_ratio=%.4g" % torch.exp(initiation_ratio_param).item()
            if use_treelike:
                _msg += " treelike_w=%.3g" % treelike_weight
            if use_ultrametric:
                _msg += " ultrametric_w=%.3g" % ultrametric_weight
            _msg += " stage=%d" % _stage
            logger.info(_msg)
            if convergence_thresh is not None:
                if _has_eval_set:
                    with torch.no_grad():
                        eval_pts = _all_points()
                        eval_radius = radius
                        eval_mu = torch.exp(log_mutation_rate)
                        eval_heritable_eps = torch.exp(log_eps) if learn_heritable_eps else heritable_eps_const
                        eval_stochastic_delta, _ = _current_delta()
                        eval_loss, _ = loss_fn(eval_pts, manifold, eval_radius, eval_mu, eval_heritable_eps, eval_stochastic_delta, rng,
                                               eval_batch=_eval_batch)
                        _eval_val = eval_loss.item()
                    _eval_loss_history.append(_eval_val)
                    rel = (_relative_change(_eval_val, _eval_loss_history[-2])
                           if len(_eval_loss_history) >= 2 else float("inf"))
                else:
                    rel = _relative_change(last, loss_history[step - check_every])
                if rel < convergence_thresh:
                    streak += 1
                    if streak >= patience:
                        if _stage == 1:
                            _transition_to_stage2(step, True)
                            continue
                        actual_steps = step + 1
                        converged = True
                        logger.info(
                            "converged at step %d (rel=%.3g < %g, patience=%d)",
                            step + 1, rel, convergence_thresh, patience)
                        break
                else:
                    streak = 0

        if _stage == 1 and step + 1 >= _stage_budget:
            _transition_to_stage2(step, False)
        elif (_stage == 2 and _stage2_start_step is not None
                and step + 1 - _stage2_start_step >= _stage_budget):
            actual_steps = step + 1
            break

    if not converged:
        logger.info("stopped at step %d without convergence", actual_steps)
    _t_optim_end = _time.perf_counter()

    with torch.no_grad():
        radius_val = 1.0
        mutation_rate_val = float(torch.exp(log_mutation_rate))
        heritable_eps_val = (float(torch.exp(log_eps))
                   if learn_heritable_eps else float(heritable_eps_init))
        stochastic_delta_sites = None
        stochastic_delta_cassettes = None
        if learn_stochastic_delta and stochastic_delta_per_site:
            _d_sites, _d_rates = _current_delta()
            stochastic_delta_sites = _d_sites.detach().cpu().numpy().tolist()
            if _delta_groups is not None:
                stochastic_delta_cassettes = _d_rates.detach().cpu().numpy().tolist()
            stochastic_delta_val = float(np.mean(stochastic_delta_sites))
        elif learn_stochastic_delta:
            stochastic_delta_val = float(torch.sigmoid(logit_delta))
        elif _delta_init_arr.ndim == 2:
            stochastic_delta_val = float(_delta_init_arr[1:].mean())    # founder row excluded
        else:
            stochastic_delta_val = float(_delta_init_arr)
        final_points = _all_points()
        if materialize_dists:
            Dg = pairwise_dists(final_points, radius=radius_val, dtype=dist_dtype)
            leaf_dists = Dg[1:, 1:]
        else:
            # low-memory mode: the (m, m) matrix is never built here; the
            # estimator derives it lazily from ``coords`` only when asked
            Dg = leaf_dists = None
        coords = final_points.detach().cpu().numpy().copy()

    # One container for every per-step scalar series, aligned by step index.
    history = {"loss": loss_history}
    if eps_history is not None:
        history["heritable_eps"] = eps_history
    if delta_history is not None:
        history["stochastic_delta"] = delta_history
    if _has_eval_set:
        history["eval_loss"] = _eval_loss_history
    if init_ratio_history is not None:
        history["initiation_ratio"] = init_ratio_history
    if ultrametric_weight_history is not None:
        history["ultrametric_weight"] = ultrametric_weight_history
        history["depth_cv"] = _depth_cv_history
    history["treelike_weight"] = treelike_weight_history
    history["gromov_delta"] = gromov_delta_history

    meta = {
        "mutation_rate": mutation_rate_val,
        "heritable_eps": heritable_eps_val, "stochastic_delta": stochastic_delta_val,
        "stochastic_delta_sites": stochastic_delta_sites,
        "stochastic_delta_cassettes": stochastic_delta_cassettes,
        "initiation_ratio_learned": (float(torch.exp(initiation_ratio_param).item())
                                     if initiation_ratio_param is not None else None),
        "init_method": _init_meta.get("init_method_used"),
        "loss": last, "history": history,
        "snapshots": snapshots, "full_dists": Dg, "coords": coords,
        "n_iter": actual_steps if converged else len(loss_history),
        "converged": converged,
        "init_time": init_time,
        "optimizer_time": _t_optim_end - _t_init_end,
        "init_meta": _init_meta,
    }
    if ultrametric_weight_history is not None:
        meta["final_ultrametric_weight"] = ultrametric_weight
        meta["ultrametric_target_cv"] = ultrametric_target_cv
    if ultrametric_auto_target:
        meta["ultrametric_init_cv"] = _init_cv
    meta["final_penalty_weight"] = treelike_weight
    meta["two_stage"] = True
    meta["stage1_iters"] = _stage1_iters
    meta["stage1_converged"] = _stage1_converged
    meta["stage2_iters"] = (actual_steps - _stage2_start_step
                            if _stage2_start_step is not None else 0)
    meta["stage_transition_step"] = _stage2_start_step
    allele_prior_param = getattr(loss_fn, "allele_prior_param", None)
    if allele_prior_param is not None:
        with torch.no_grad():
            valid = (allele_prior_param.data > MASK_LOGIT / 10)
            masked = torch.where(valid, allele_prior_param.data,
                                 torch.tensor(MASK_LOGIT, dtype=allele_prior_param.dtype,
                                              device=allele_prior_param.device))
            sm_dim = -1 if allele_prior_param.dim() == 1 else 1
            probs = torch.softmax(masked, dim=sm_dim)
            out = torch.zeros_like(probs)
            if out.dim() == 1:
                out[0] = 1.0
            else:
                out[:, 0] = 1.0
            out = out + probs * valid.float()
            meta["learned_allele_prior"] = out.cpu().numpy()
    return leaf_dists, meta
