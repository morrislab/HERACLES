"""The optimisation loop and distance read-out for the hyperbolic embedding.

:func:`optimize` runs Riemannian Adam on the Lorentz model, with a two-stage
adaptive tree-likeness penalty and an optional adaptive ultrametric penalty.
:func:`pairwise_dists` and :func:`iter_dist_rows` compute the embedding's
distance matrix, whole or in row blocks.
"""

import logging
import time

import geoopt
import numpy as np
import torch

from heracles.embedding.ctmc import gromov_quartet_penalty
from heracles.embedding.initialization import lorentz_manifold
from heracles.embedding.priors import missing_log_prior, resolve_prior
from heracles.constants import (CONVERGENCE_CHECK_EVERY, CONVERGENCE_PATIENCE, DIST_ROW_CHUNK,
                                SEED_OFFSET_EVAL, SEED_OFFSET_PENALTY, SEED_OFFSET_SAMPLER)
from heracles.runinfo import resolve_device

logger = logging.getLogger("heracles")


__all__ = [
    "ULTRAMETRIC_LR_DUAL",
    "ULTRAMETRIC_MAX_WEIGHT",
    "ULTRAMETRIC_MIN_WEIGHT",
    "ULTRAMETRIC_SHRINK_FACTOR",
    "TREELIKE_N_QUARTETS",
    "TREELIKE_LR_DUAL",
    "TREELIKE_MAX_WEIGHT",
    "TREELIKE_MIN_WEIGHT",
    "iter_dist_rows",
    "pairwise_dists",
    "optimize",
]


# Dual-ascent controller settings for the two penalties (fixed).
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


class _Rates:
    """The learnable scalars besides the coordinates: mutation rate, ``heritable_eps`` and
    ``stochastic_delta``, with their priors.

    ``current()`` returns the values the kernels take; a per-cassette delta is
    expanded to sites for them and kept per cassette for the Beta prior, so the
    prior counts each learned rate once.
    """

    def __init__(self, states, dev, *, mutation_rate_init, learn_mutation_rate,
                 heritable_eps_init, learn_heritable_eps, stochastic_delta_init,
                 learn_stochastic_delta, stochastic_delta_per_site, site_groups,
                 heritable_eps_prior_variance, stochastic_delta_prior_variance):
        self.learn_mutation_rate = learn_mutation_rate
        self.learn_heritable_eps = learn_heritable_eps
        self.learn_stochastic_delta = learn_stochastic_delta
        self.per_site = stochastic_delta_per_site
        self.heritable_eps_init = heritable_eps_init
        self.params = []

        delta_init = np.asarray(stochastic_delta_init, dtype=np.float64)
        if delta_init.ndim == 2:
            # a fixed per-cell, per-site dropout table (founder row 0 included)
            if learn_stochastic_delta:
                raise ValueError("a per-cell stochastic_delta table cannot be learned; "
                                 "pass learn_stochastic_delta=False")
            if delta_init.shape != tuple(states.shape):
                raise ValueError("per-cell stochastic_delta must be (m, S) = %s, got %s"
                                 % (tuple(states.shape), delta_init.shape))
        elif delta_init.ndim != 0:
            raise ValueError("stochastic_delta_init must be a scalar or an (m, S) table")
        self.delta_init = delta_init
        if bool((np.asarray(states) < 0).any()):
            eps_ok = heritable_eps_init > 0 or learn_heritable_eps
            delta_ok = bool((delta_init > 0).any()) or learn_stochastic_delta
            if not (eps_ok or delta_ok):
                raise ValueError(
                    "states contain missing entries (-1) but heritable_eps and "
                    "stochastic_delta are both zero with learning off: the "
                    "missing-data likelihood is -inf. Set a positive init value "
                    "or enable learning.")

        self.log_mutation_rate = torch.log(torch.tensor(float(mutation_rate_init), device=dev))
        if learn_mutation_rate:
            self.log_mutation_rate = torch.nn.Parameter(self.log_mutation_rate.clone())
            self.params.append(self.log_mutation_rate)

        if learn_heritable_eps:
            self.log_eps = torch.nn.Parameter(
                torch.log(torch.tensor(max(float(heritable_eps_init), _INIT_FLOOR), device=dev)))
            self.params.append(self.log_eps)
        else:
            self.eps_const = torch.tensor(float(heritable_eps_init), dtype=torch.float64, device=dev)

        self.delta_groups = None
        if learn_stochastic_delta:
            d0 = min(max(float(stochastic_delta_init), _INIT_FLOOR), 1 - _INIT_FLOOR)
            if stochastic_delta_per_site:
                # one logit per site (or per cassette, expanded to sites each step),
                # all started at the shared init value
                if site_groups is not None:
                    self.delta_groups = torch.as_tensor(np.asarray(site_groups), dtype=torch.long,
                                                        device=dev)
                    n_rates = int(self.delta_groups.max()) + 1
                else:
                    n_rates = states.shape[1]
                d0_t = torch.full((n_rates,), d0, dtype=torch.float64, device=dev)
            else:
                d0_t = torch.tensor(d0, device=dev)
            self.logit_delta = torch.nn.Parameter(torch.logit(d0_t))
            self.params.append(self.logit_delta)
        else:
            self.delta_const = torch.as_tensor(delta_init, dtype=torch.float64, device=dev)

        self.eps_gamma = (resolve_prior(heritable_eps_init, heritable_eps_prior_variance, "gamma")
                          if learn_heritable_eps else None)
        self.delta_beta = (resolve_prior(stochastic_delta_init, stochastic_delta_prior_variance,
                                         "beta")
                           if learn_stochastic_delta else None)

    @property
    def has_prior(self):
        return self.learn_heritable_eps or self.learn_stochastic_delta

    def current(self):
        """``(mutation_rate, heritable_eps, stochastic_delta, delta_for_prior)`` as tensors."""
        mutation_rate = torch.exp(self.log_mutation_rate)
        heritable_eps = torch.exp(self.log_eps) if self.learn_heritable_eps else self.eps_const
        if not self.learn_stochastic_delta:
            return mutation_rate, heritable_eps, self.delta_const, None
        d = torch.sigmoid(self.logit_delta)
        if self.delta_groups is not None:
            return mutation_rate, heritable_eps, d[self.delta_groups], d
        return mutation_rate, heritable_eps, d, d

    def log_prior(self, heritable_eps, delta_for_prior):
        return missing_log_prior(heritable_eps if self.learn_heritable_eps else None,
                                 delta_for_prior, self.eps_gamma, self.delta_beta)

    def describe(self, heritable_eps, stochastic_delta):
        """Log fragment with the learned rates."""
        msg = ""
        if self.learn_heritable_eps:
            msg += " eps=%.4g" % heritable_eps.item()
        if self.learn_stochastic_delta:
            if self.per_site:
                msg += " delta=%.4g[%.4g..%.4g]" % (
                    stochastic_delta.mean().item(), stochastic_delta.min().item(),
                    stochastic_delta.max().item())
            else:
                msg += " delta=%.4g" % stochastic_delta.item()
        return msg

    def final(self):
        """Fitted values for the fit record (floats and lists)."""
        out = {"mutation_rate": float(torch.exp(self.log_mutation_rate)),
               "heritable_eps": (float(torch.exp(self.log_eps)) if self.learn_heritable_eps
                                 else float(self.heritable_eps_init)),
               "stochastic_delta_sites": None, "stochastic_delta_cassettes": None}
        if self.learn_stochastic_delta and self.per_site:
            _, _, d_sites, d_rates = self.current()
            out["stochastic_delta_sites"] = d_sites.detach().cpu().numpy().tolist()
            if self.delta_groups is not None:
                out["stochastic_delta_cassettes"] = d_rates.detach().cpu().numpy().tolist()
            out["stochastic_delta"] = float(np.mean(out["stochastic_delta_sites"]))
        elif self.learn_stochastic_delta:
            out["stochastic_delta"] = float(torch.sigmoid(self.logit_delta))
        elif self.delta_init.ndim == 2:
            out["stochastic_delta"] = float(self.delta_init[1:].mean())    # founder row excluded
        else:
            out["stochastic_delta"] = float(self.delta_init)
        return out


class _Penalties:
    """The two adaptive penalties and the two-stage schedule.

    Stage 1 fits the likelihood (plus the ultrametric penalty when on); stage 2
    adds the four-point tree-likeness penalty, whose weight the dual-ascent
    controller moves toward ``target_delta``.  Per-step histories are kept here.
    """

    def __init__(self, *, treelike_penalty, target_delta, ultrametric_penalty,
                 ultrametric_target_cv, init_depths, steps):
        self.stage = 1
        self.stage1_iters = None
        self.stage1_converged = False
        self.stage2_start = None
        self.budget = steps // 2

        self.use_treelike = False
        self.treelike_ctrl = _DualAscent(
            lr=TREELIKE_LR_DUAL, target=target_delta,
            min_weight=TREELIKE_MIN_WEIGHT if treelike_penalty else 0.0,
            max_weight=TREELIKE_MAX_WEIGHT if treelike_penalty else 0.0)
        self.treelike_weight = 0.0
        self.treelike_weight_history = []
        self.gromov_delta_history = []

        self.ultrametric_adaptive = (ultrametric_penalty == "adaptive")
        self.ultrametric_init_cv = None
        if self.ultrametric_adaptive and ultrametric_target_cv == "auto":
            with torch.no_grad():
                self.ultrametric_init_cv = float(torch.std(init_depths) / torch.mean(init_depths))
            ultrametric_target_cv = self.ultrametric_init_cv * ULTRAMETRIC_SHRINK_FACTOR
        self.ultrametric_target_cv = ultrametric_target_cv
        if self.ultrametric_adaptive:
            weight = 0.0
        elif isinstance(ultrametric_penalty, (int, float)) and ultrametric_penalty > 0:
            weight = float(ultrametric_penalty)
        else:
            weight = 0.0
        self.use_ultrametric = self.ultrametric_adaptive or weight > 0
        self.ultrametric_weight = weight
        self.ultrametric_ctrl = _DualAscent(lr=ULTRAMETRIC_LR_DUAL, target=ultrametric_target_cv,
                                            min_weight=ULTRAMETRIC_MIN_WEIGHT,
                                            max_weight=ULTRAMETRIC_MAX_WEIGHT, weight=weight)
        self.ultrametric_weight_history = [] if self.use_ultrametric else None
        self.depth_cv_history = [] if self.use_ultrametric else None

    def transition_to_stage2(self, step_num, did_converge):
        self.stage1_iters = step_num + 1
        self.stage1_converged = did_converge
        logger.info("stage 1 -> 2 at step %d (stage 1 converged=%s)", step_num + 1, did_converge)
        self.stage = 2
        self.stage2_start = step_num + 1
        self.use_treelike = True
        self.treelike_ctrl.weight = 0.0
        self.treelike_weight = 0.0

    def begin_step(self):
        self._gromov_acc = 0.0

    def add(self, loss, points, loss_fn, manifold, radius, penalty_rng, k_acc):
        """Loss with the active penalties of one micro-batch added."""
        if self.use_treelike:
            # The quartet objective exposes the four-point defect of its own
            # likelihood batch; other objectives get an independent uniform batch.
            gromov_penalty = getattr(loss_fn, "batch_gromov_penalty", None)
            if gromov_penalty is None:
                gromov_penalty = gromov_quartet_penalty(points, manifold, radius,
                                                        TREELIKE_N_QUARTETS, penalty_rng)
            loss = loss + self.treelike_weight * gromov_penalty
            self._gromov_acc = self._gromov_acc + gromov_penalty.detach() / k_acc
        if self.use_ultrametric:
            # the root-to-leaf depths the objective already computed, if it exposes them
            depths = getattr(loss_fn, "batch_depths", None)
            if depths is None:
                depths = radius * manifold.dist(points[0].unsqueeze(0), points[1:])
            self._depths, self._depth_var = depths, torch.var(depths)
            loss = loss + self.ultrametric_weight * self._depth_var
        return loss

    def end_step(self):
        """Dual-ascent updates after the optimiser step; records the histories."""
        if self.use_treelike:
            gromov_val = float(self._gromov_acc)
            self.gromov_delta_history.append(gromov_val)
            self.treelike_weight = self.treelike_ctrl.step(gromov_val)
            self.treelike_weight_history.append(self.treelike_weight)
        else:
            self.gromov_delta_history.append(0.0)
            self.treelike_weight_history.append(0.0)
        if self.use_ultrametric:
            with torch.no_grad():
                cv = float(torch.sqrt(self._depth_var) / self._depths.mean())
            self.depth_cv_history.append(cv)
            if self.ultrametric_adaptive:
                self.ultrametric_weight = self.ultrametric_ctrl.step(cv)
            self.ultrametric_weight_history.append(self.ultrametric_weight)

    def describe(self):
        msg = ""
        if self.use_treelike:
            msg += " treelike_w=%.3g" % self.treelike_weight
        if self.use_ultrametric:
            msg += " ultrametric_w=%.3g" % self.ultrametric_weight
        return msg + " stage=%d" % self.stage

    def history(self):
        out = {}
        if self.ultrametric_weight_history is not None:
            out["ultrametric_weight"] = self.ultrametric_weight_history
            out["depth_cv"] = self.depth_cv_history
        out["treelike_weight"] = self.treelike_weight_history
        out["gromov_delta"] = self.gromov_delta_history
        return out

    def meta(self, actual_steps):
        out = {}
        if self.ultrametric_weight_history is not None:
            out["final_ultrametric_weight"] = self.ultrametric_weight
            out["ultrametric_target_cv"] = self.ultrametric_target_cv
        if self.ultrametric_init_cv is not None:
            out["ultrametric_init_cv"] = self.ultrametric_init_cv
        out["final_penalty_weight"] = self.treelike_weight
        out["stage1_iters"] = self.stage1_iters
        out["stage1_converged"] = self.stage1_converged
        out["stage2_iters"] = (actual_steps - self.stage2_start
                               if self.stage2_start is not None else 0)
        out["stage_transition_step"] = self.stage2_start
        return out


def _cosine_schedule(opt, lr, t_max):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=lr * 0.01)


def optimize(states, dimension, *, loss_fn, init_points,
             steps=1500, lr=0.02, seed=0, lr_schedule=None, min_iterations=0,
             convergence_thresh=None, snapshot_epochs=None,
             mutation_rate_init=1.0, learn_mutation_rate=False,
             heritable_eps_init=0.0, learn_heritable_eps=False,
             stochastic_delta_init=0.0, learn_stochastic_delta=False,
             stochastic_delta_per_site=False, site_groups=None,
             heritable_eps_prior_variance="auto", stochastic_delta_prior_variance="auto",
             treelike_penalty=True, treelike_penalty_target_delta=0.01,
             ultrametric_penalty=0, ultrametric_target_cv=0.25,
             grad_accum=1, device=None, dist_dtype=np.float64, materialize_dists=True):
    """Fit the embedding with Riemannian Adam on the Lorentz model and return its distances.

    Called by :func:`heracles.fit.embed`, which maps the fields of
    :class:`heracles.config.HeraclesConfig` onto these keyword arguments (see that
    class for meanings and defaults). The founder (row 0) is fixed at the origin;
    the leaf coordinates are learned, and optionally the mutation rate,
    ``heritable_eps`` and ``stochastic_delta``. Stage 1 fits the likelihood alone
    until it converges or uses half of ``steps``; stage 2 adds the tree-likeness
    penalty, whose weight adapts toward ``treelike_penalty_target_delta``.  The
    controller settings are the module constants.

    Args:
        loss_fn: callable ``(points, manifold, radius, mutation_rate,
            heritable_eps, stochastic_delta, rng) -> (loss, numel)``, e.g. from
            :func:`heracles.embedding.objective.quartet_loss_fn`.
        init_points: (m, d+1) starting coordinates.
        stochastic_delta_init: a scalar, or a fixed (m, S) per-cell table.
        site_groups: optional (S,) cassette index per site; with
            ``stochastic_delta_per_site`` one dropout rate is learned per cassette.
        treelike_penalty: False holds the stage-2 penalty weight at 0.
        grad_accum: micro-batches per step (the loss of each is divided by
            ``grad_accum``, so the gradient is that of the mean over the batch).
        materialize_dists: if False, ``leaf_dists`` is None and ``meta["coords"]`` holds the result.

    Returns ``(leaf_dists, meta)``: the (n, n) leaf distances (or None) and a dict
    with the fitted rates, ``coords``, ``history``, ``n_iter``, ``converged`` and
    the optimiser time. Raises ``ValueError`` if the delta table has the wrong
    shape or is set to be learned, or the data have missing entries while both
    missing-data rates are zero and fixed.
    """
    torch.manual_seed(seed)
    dev = resolve_device(device)
    manifold = lorentz_manifold(dimension)

    rates = _Rates(states, dev, mutation_rate_init=mutation_rate_init,
                   learn_mutation_rate=learn_mutation_rate,
                   heritable_eps_init=heritable_eps_init, learn_heritable_eps=learn_heritable_eps,
                   stochastic_delta_init=stochastic_delta_init,
                   learn_stochastic_delta=learn_stochastic_delta,
                   stochastic_delta_per_site=stochastic_delta_per_site, site_groups=site_groups,
                   heritable_eps_prior_variance=heritable_eps_prior_variance,
                   stochastic_delta_prior_variance=stochastic_delta_prior_variance)

    # Fix the founder (row 0) at the north pole to break the translational
    # gauge freedom that otherwise lets coordinates drift to infinity without
    # changing the loss.
    all_init = torch.as_tensor(init_points, dtype=torch.float64).to(dev)
    with torch.no_grad():
        all_init = _lorentz_boost_to_origin(all_init, all_init[0])
        all_init = manifold.projx(all_init)
    founder = all_init[0].detach()
    leaf_points = geoopt.ManifoldParameter(all_init[1:].clone(), manifold=manifold)

    def all_points():
        return torch.cat([founder.unsqueeze(0), leaf_points], dim=0)

    # Global scale of the embedding, fixed at 1: the scale and the mutation
    # rate are one gauge, so only one of them can be a parameter.
    radius = torch.tensor(1.0, dtype=torch.float64, device=dev)

    params = [leaf_points] + rates.params
    allele_prior_param = getattr(loss_fn, "allele_prior_param", None)
    if allele_prior_param is not None:
        params.append(allele_prior_param)
    initiation_ratio_param = getattr(loss_fn, "initiation_ratio_param", None)
    if initiation_ratio_param is not None:
        params.append(initiation_ratio_param)

    opt = geoopt.optim.RiemannianAdam(params, lr=lr)
    scheduler = _cosine_schedule(opt, lr, steps // 2) if lr_schedule == "cosine" else None

    snapshot_at = set(int(e) for e in (snapshot_epochs or []))
    snapshots = {}

    def snapshot(epoch):
        with torch.no_grad():
            snapshots[epoch] = all_points().detach().cpu().numpy().copy()

    if 0 in snapshot_at:
        snapshot(0)

    rng = torch.Generator()
    rng.manual_seed(seed + SEED_OFFSET_SAMPLER)
    penalty_rng = torch.Generator()
    penalty_rng.manual_seed(seed + SEED_OFFSET_PENALTY)

    # Fixed evaluation batch for a stable convergence check
    make_eval_batch = getattr(loss_fn, "make_eval_batch", None)
    has_eval_set = convergence_thresh is not None and make_eval_batch is not None
    eval_loss_history = []
    if has_eval_set:
        eval_rng = torch.Generator()
        eval_rng.manual_seed(seed + SEED_OFFSET_EVAL)
        eval_batch = make_eval_batch(eval_rng).to(dev)

    loss_history = []
    eps_history = [] if learn_heritable_eps else None
    delta_history = [] if learn_stochastic_delta else None
    init_ratio_history = [] if initiation_ratio_param is not None else None

    with torch.no_grad():
        init_pts = all_points()
        init_depths = radius * manifold.dist(init_pts[0].unsqueeze(0), init_pts[1:])
    penalties = _Penalties(treelike_penalty=treelike_penalty,
                           target_delta=treelike_penalty_target_delta,
                           ultrametric_penalty=ultrametric_penalty,
                           ultrametric_target_cv=ultrametric_target_cv,
                           init_depths=init_depths, steps=steps)

    converged = False
    streak = 0
    actual_steps = steps
    k_acc = int(grad_accum)
    t_init_end = time.perf_counter()

    def to_stage2(step, did_converge):
        nonlocal streak, converged, scheduler
        penalties.transition_to_stage2(step, did_converge)
        streak = 0
        converged = False
        eval_loss_history.clear()
        if lr_schedule == "cosine":
            scheduler = _cosine_schedule(opt, lr, penalties.budget)

    for step in range(steps):
        opt.zero_grad()
        penalties.begin_step()
        loss_acc = 0.0
        # the k micro-batches share the generator; the parameter transforms are
        # rebuilt per micro-batch because backward() frees their graph
        for _ in range(k_acc):
            mutation_rate, heritable_eps, stochastic_delta, delta_for_prior = rates.current()
            points = all_points()
            loss, numel = loss_fn(points, manifold, radius, mutation_rate, heritable_eps,
                                  stochastic_delta, rng)
            if rates.has_prior:
                # one prior per step, spread over the k micro-batches
                loss = loss - rates.log_prior(heritable_eps, delta_for_prior) / (numel * k_acc)
            loss = penalties.add(loss, points, loss_fn, manifold, radius, penalty_rng, k_acc)
            if k_acc == 1:
                loss.backward()
            else:
                (loss / k_acc).backward()
            loss_acc = loss_acc + loss.detach() / k_acc
        loss = loss_acc
        opt.step()
        if scheduler is not None:
            scheduler.step()
        penalties.end_step()

        last = loss.item()
        loss_history.append(last)
        if eps_history is not None:
            eps_history.append(heritable_eps.item())
        if delta_history is not None:
            delta_history.append(stochastic_delta.mean().item())   # per-site: site mean
        if init_ratio_history is not None:
            init_ratio_history.append(float(torch.exp(initiation_ratio_param).item()))
        if (step + 1) in snapshot_at:
            snapshot(step + 1)

        eff_min = min_iterations
        if penalties.stage == 2:
            eff_min = penalties.stage2_start + max(
                min_iterations, CONVERGENCE_CHECK_EVERY * (CONVERGENCE_PATIENCE + 1))
        if ((step + 1) % CONVERGENCE_CHECK_EVERY == 0 and step >= CONVERGENCE_CHECK_EVERY
                and step + 1 >= eff_min):
            msg = "step %d/%d: loss=%.6g rel=%.3g streak=%d" % (
                step + 1, steps, last,
                _relative_change(last, loss_history[step - CONVERGENCE_CHECK_EVERY]), streak)
            msg += rates.describe(heritable_eps, stochastic_delta)
            if initiation_ratio_param is not None:
                msg += " init_ratio=%.4g" % torch.exp(initiation_ratio_param).item()
            logger.info(msg + penalties.describe())
            if convergence_thresh is not None:
                if has_eval_set:
                    with torch.no_grad():
                        eval_mu, eval_eps, eval_delta, _ = rates.current()
                        eval_loss, _ = loss_fn(all_points(), manifold, radius, eval_mu, eval_eps,
                                               eval_delta, rng, eval_batch=eval_batch)
                    eval_loss_history.append(eval_loss.item())
                    rel = (_relative_change(eval_loss_history[-1], eval_loss_history[-2])
                           if len(eval_loss_history) >= 2 else float("inf"))
                else:
                    rel = _relative_change(last, loss_history[step - CONVERGENCE_CHECK_EVERY])
                if rel < convergence_thresh:
                    streak += 1
                    if streak >= CONVERGENCE_PATIENCE:
                        if penalties.stage == 1:
                            to_stage2(step, True)
                            continue
                        actual_steps = step + 1
                        converged = True
                        logger.info("converged at step %d (rel=%.3g < %g, patience=%d)",
                                    step + 1, rel, convergence_thresh, CONVERGENCE_PATIENCE)
                        break
                else:
                    streak = 0

        if penalties.stage == 1 and step + 1 >= penalties.budget:
            to_stage2(step, False)
        elif penalties.stage == 2 and step + 1 - penalties.stage2_start >= penalties.budget:
            actual_steps = step + 1
            break

    if not converged:
        logger.info("stopped at step %d without convergence", actual_steps)
    optimizer_time = time.perf_counter() - t_init_end

    with torch.no_grad():
        fitted = rates.final()
        final_points = all_points()
        if materialize_dists:
            Dg = pairwise_dists(final_points, radius=1.0, dtype=dist_dtype)
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
    if has_eval_set:
        history["eval_loss"] = eval_loss_history
    if init_ratio_history is not None:
        history["initiation_ratio"] = init_ratio_history
    history.update(penalties.history())

    meta = dict(fitted)
    meta.update({
        "initiation_ratio_learned": (float(torch.exp(initiation_ratio_param).item())
                                     if initiation_ratio_param is not None else None),
        "loss": loss_history[-1] if loss_history else None, "history": history,
        "snapshots": snapshots, "full_dists": Dg, "coords": coords,
        "n_iter": actual_steps if converged else len(loss_history),
        "converged": converged,
        "optimizer_time": optimizer_time,
    })
    meta.update(penalties.meta(actual_steps))
    if allele_prior_param is not None:
        # the prior the fit used: the same masked softmax as every training step
        with torch.no_grad():
            meta["learned_allele_prior"] = loss_fn.get_allele_prior().cpu().numpy()
    return leaf_dists, meta
