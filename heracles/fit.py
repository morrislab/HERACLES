"""The ``Heracles`` estimator and the lower-level :func:`embed` function.

    from heracles import Heracles
    model = Heracles(outdir="runs/example")     # published defaults
    model.fit(char_matrix)                       # DataFrame or CSV path
    model.trees_["nj"]                           # tree rooted at the founder

Options are fields of :class:`heracles.config.HeraclesConfig`.
"""

import json
import logging
import math
import os
import time as _time
import warnings

import numpy as np
import pandas as pd
import torch
import geoopt

from heracles.config import HeraclesConfig
from heracles.constants import (CONVERGENCE_CHECK_EVERY, CONVERGENCE_PATIENCE, DIM_PER_SQRT_N,
                                PHYLIP_BYTES_PER_ENTRY)
from heracles.models import get_model, model_from_legacy_params, unregistered_parts
from heracles.embedding.initialization import (
    choose_landmarks, default_n_landmarks, init_points, is_newick_string, lorentz_manifold,
)
from heracles.embedding.optimize import (
    optimize, pairwise_dists, ULTRAMETRIC_LR_DUAL, ULTRAMETRIC_MAX_WEIGHT, TREELIKE_N_QUARTETS,
    TREELIKE_LR_DUAL, TREELIKE_MAX_WEIGHT, TREELIKE_MIN_WEIGHT,
)
from heracles.embedding.priors import build_allele_priors, missing_log_prior, resolve_prior  # noqa: F401 (re-exported)
from heracles.embedding.sampler import (banded_sampler_reach, default_batch_size,
                                        n_distinct_quartets)
from heracles.trees.readouts import (  # noqa: F401 (re-exported)
    READOUTS, heracles_nj, heracles_upgma,
)
from heracles.runinfo import git_commit, resolve_device, setup_run_logging, teardown_run_logging
from heracles.trees.utils import FOUNDER_LABEL, as_newick, write_newick  # noqa: F401 (re-exported)


__all__ = [
    'load_character_matrix',
    'embedding_dimension',
    'embed',
    'memory_plan',
    'Heracles',
]


_NOT_FIT_MSG = "This Heracles instance is not fitted yet. Call .fit() first."

logger = logging.getLogger(__name__)


def load_character_matrix(char_matrix, missing_state=-1):
    """Reads and checks a character matrix; returns ``(states, labels)``.

    *char_matrix* is a cells x sites DataFrame indexed by cell name, or a CSV
    path (``str`` or path-like) with cell names in the first column.  Values are
    allele ids: 0 = unedited, positive = an edit, *missing_state* = missing.
    Returns an ``(n, S)`` int64 matrix, with *missing_state* recoded to -1, and
    the n cell names as strings.  Raises ``ValueError`` on non-numeric columns,
    NaN, non-integer values, other negative codes or repeated cell names.
    """
    if isinstance(char_matrix, (str, os.PathLike)):
        cm = pd.read_csv(char_matrix, index_col=0)
    elif isinstance(char_matrix, pd.DataFrame):
        cm = char_matrix
    else:
        raise TypeError("char_matrix must be a DataFrame or a CSV path, got %s"
                        % type(char_matrix).__name__)
    labels = [str(x) for x in cm.index]
    text_cols = [c for c in cm.columns if not pd.api.types.is_numeric_dtype(cm[c])]
    if text_cols:
        raise ValueError("column %r is not numeric; cell names belong in the index "
                         "(pd.read_csv(path, index_col=0))" % (text_cols[0],))
    values = cm.to_numpy(dtype=np.float64)
    n_nan = int(np.isnan(values).sum())
    if n_nan:
        raise ValueError("%d entries of the character matrix are NaN; mark missing data "
                         "with missing_state (default -1)" % n_nan)
    if not np.array_equal(values, np.round(values)):
        raise ValueError("the character matrix holds non-integer values; allele ids "
                         "must be integers")
    states = values.astype(np.int64)
    bad = np.unique(states[(states < 0) & (states != missing_state) & (states != -1)])
    if bad.size:
        raise ValueError("negative value(s) %s are not the missing code (missing_state=%d); "
                         "set missing_state to the code your data uses"
                         % (bad[:5].tolist(), missing_state))
    if len(set(labels)) != len(labels):
        dup = pd.Index(labels)[pd.Index(labels).duplicated()][0]
        raise ValueError("cell name %r appears more than once" % (dup,))
    if missing_state != -1:
        states[states == missing_state] = -1
    return states, labels


def _check_readouts(names):
    """Raises ``ValueError`` for names not in :data:`READOUTS`."""
    bad = [r for r in names if r not in READOUTS]
    if bad:
        hint = ("; 'rapidnj' is now 'nj', which uses RapidNJ whenever it is available"
                if "rapidnj" in bad else "")
        raise ValueError(f"unknown readout(s): {bad} (valid: {list(READOUTS)}){hint}")


def _loss_fn_meta(loss_fn):
    """Returns a copy of the loss function's ``meta`` dict for the fit record."""
    out = {}
    for k, v in (getattr(loss_fn, "meta", None) or {}).items():
        if k == "quartet_sampler_stats":
            v = dict(v) if v else None
        elif isinstance(v, dict):
            v = dict(v)
        out[k] = v
    return out


def _diagnostics_from_flat_keys(meta):
    """Builds ``diagnostics_`` for a model.npz that stores them as flat ``fitted`` keys."""
    fitted = meta.get("fitted") or {}
    diag = {k: fitted.get(k + "_") for k in _DIAGNOSTIC_KEYS if k + "_" in fitted}
    init_meta = dict(diag.get("init_meta") or {})
    diag.setdefault("init_tree_newick", init_meta.pop("init_tree_newick", None))
    diag["init_meta"] = init_meta
    diag.setdefault("init_method_used", init_meta.get("init_method_used"))
    diag["telemetry"] = meta.get("telemetry")
    return diag


def embedding_dimension(n_cells):
    """Returns the default embedding dimension, ``ceil(5 * sqrt(n_cells))``, at least 2."""
    return max(2, math.ceil(DIM_PER_SQRT_N * math.sqrt(n_cells)))


def embed(char_matrix, *, config=None, edit_model=None, init_dists=None,
          snapshot_epochs=None, cell_dropout=None, **options):
    """Fits the embedding once and returns what the tree read-outs need.

    :meth:`Heracles.fit` calls this; use it directly for lower-level control.

    Args:
        edit_model: An :class:`~heracles.models.EditModel` or its name (default ``Cas9Model()``).
        init_dists: ``(n+1, n+1)`` distances (founder first) that replace the
            seed distances for the initialisation.
        cell_dropout: Fixed stochastic dropout probability per cell and site
            (cells x sites DataFrame or ``(n, S)`` array in matrix row order);
            turns off ``learn_stochastic_delta``.
        **options: Override individual ``config`` fields.

    Returns:
        A dict with ``leaf_dists`` ``(n, n)``, ``full_dists`` ``(n+1, n+1)``
        (founder at index 0; both None when ``low_memory`` is set), ``labels``,
        ``coords`` (``(n+1, dim+1)`` Lorentz-model coordinates), ``meta``
        (fitted parameters, loss history, timings) and entries for the read-outs.

    Raises:
        ValueError: ``cell_dropout`` does not match the matrix or [0, 1], or
            ``low_memory`` is set for an edit model that does not support it.
    """
    cfg = config if config is not None else HeraclesConfig()
    if options:
        cfg = cfg.replace(**options)
    device = resolve_device(cfg.device)
    model = get_model(edit_model)
    mats, labels = load_character_matrix(char_matrix, missing_state=cfg.missing_state)
    model.validate(mats.shape[1], cfg)
    if cfg.low_memory and not model.supports_low_memory:
        raise ValueError("low_memory is not available for %s" % type(model).__name__)
    logger.info("edit model: %s", model.describe())
    n_cells = mats.shape[0]
    dimension = cfg.dimension
    if dimension is None:
        dimension = embedding_dimension(n_cells)
    logger.info(
        "embedding %d cells x %d sites: dimension=%s init_method=%s "
        "max_iterations=%d seed=%d",
        n_cells, mats.shape[1], dimension, cfg.init_method,
        cfg.max_iterations, cfg.seed)
    states_aug = np.vstack([np.zeros((1, mats.shape[1]), dtype=np.int64), mats])
    labels_aug = [FOUNDER_LABEL] + labels
    delta_init, learn_delta = cfg.stochastic_delta, cfg.learn_stochastic_delta
    if cell_dropout is not None:
        if hasattr(cell_dropout, "loc"):
            cd = cell_dropout.copy(); cd.index = cd.index.astype(str)
            missing_rows = [lab for lab in labels if lab not in cd.index]
            if missing_rows:
                raise ValueError("cell_dropout lacks %d cells, e.g. %s" % (len(missing_rows), missing_rows[:3]))
            cd = cd.loc[labels].to_numpy(dtype=np.float64)
        else:
            cd = np.asarray(cell_dropout, dtype=np.float64)
        if cd.shape != mats.shape:
            raise ValueError("cell_dropout must be (n_cells, S) = %s, got %s" % (mats.shape, cd.shape))
        if (cd < 0).any() or (cd > 1).any():
            raise ValueError("cell_dropout entries must be probabilities in [0, 1]")
        delta_init = np.vstack([np.zeros((1, mats.shape[1])), cd])     # founder never drops out
        learn_delta = False
        logger.info("cell_dropout table: %d x %d, mean %.4f, range %.3g..%.3g (fixed; "
                    "stochastic_delta is not learned)", *cd.shape, cd.mean(), cd.min(), cd.max())

    prior_array = build_allele_priors(states_aug[1:], prior=cfg.allele_prior)
    dist_dtype = np.float32 if cfg.low_memory else np.float64
    # low-memory mode: no (m, m) array anywhere in the fit. Landmark rows
    # seed the init, a k-NN table drives the sampler, and the fitted distances
    # are derived from the coordinates on demand.
    low_memory = bool(cfg.low_memory)
    landmarks = None
    m_points = states_aug.shape[0]
    if low_memory:
        n_lm = min(default_n_landmarks(dimension), m_points)
        landmarks = choose_landmarks(m_points, n_lm, seed=cfg.seed)
        est_aug = None                                  # the sampler works from k-NN tables
        if cfg.sampler_knn is None:
            reach = banded_sampler_reach(cfg.band_kmax, cfg.band_kmax_end,
                                         cfg.band_local_frac, cfg.band_local_mult_max)
            cfg = cfg.replace(sampler_knn=min(reach, n_cells - 1))
        logger.info("low_memory: %d landmarks seed the init, sampler_knn=%d, "
                    "no (%d x %d) matrix is built during the fit",
                    n_lm, cfg.sampler_knn, m_points, m_points)
    else:
        est_aug = model.seed_distances(states_aug, dtype=dist_dtype)
    for line in memory_plan(n_cells, dimension, low_memory=low_memory,
                            dist_dtype=dist_dtype):
        logger.info("memory plan: %s", line)

    grad_accum = int(cfg.grad_accum)
    batch_arg, root_weight = cfg.batch_size, None
    if grad_accum > 1:
        # same objective as one full batch, split into micro-batches
        b_total = int(cfg.batch_size) if cfg.batch_size is not None else default_batch_size(n_cells)
        b_total = max(1, min(b_total, n_distinct_quartets(n_cells)))
        batch_arg = -(-b_total // grad_accum)
        root_weight = n_cells / (b_total + n_cells)
        logger.info("grad_accum=%d: %d quartets per step as %d micro-batches of %d",
                    grad_accum, b_total, grad_accum, batch_arg)
    loss_fn = model.loss_function(states_aug, prior_array, seed_dists=est_aug,
                                  batch_size=batch_arg, root_weight=root_weight,
                                  learn_allele_prior=cfg.learn_allele_prior,
                                  config=cfg, device=device)
    has_missing = bool((states_aug < 0).any())

    t_init = _time.perf_counter()
    manifold, _ = lorentz_manifold(dimension)
    points0, init_meta = init_points(
        manifold, dimension, states=states_aug, seed_dists=est_aug, labels=labels_aug,
        init_dists=init_dists, init_method=cfg.init_method, landmarks=landmarks)
    init_time = _time.perf_counter() - t_init
    logger.info("init done: method=%s, %.1fs (tree %.1fs, tree distances %.1fs, mds %.1fs)",
                init_meta["init_method_used"], init_time, init_meta["tree_creation_s"],
                init_meta["distance_extraction_s"], init_meta["embedding_s"])

    leaf_dists, meta = optimize(
        states_aug, dimension,
        loss_fn=loss_fn, has_missing=has_missing,
        init_points=points0, init_meta=init_meta, init_time=init_time,
        steps=cfg.max_iterations, lr=cfg.lr, seed=cfg.seed,
        learn_mutation_rate=cfg.learn_mutation_rate,
        snapshot_epochs=snapshot_epochs,
        heritable_eps_init=cfg.heritable_eps,
        stochastic_delta_init=delta_init,
        learn_heritable_eps=cfg.learn_heritable_eps,
        learn_stochastic_delta=learn_delta,
        stochastic_delta_per_site=model.separate_dropout,
        site_groups=model.missing_units(mats.shape[1]),
        heritable_eps_prior_variance=cfg.heritable_eps_prior_variance,
        stochastic_delta_prior_variance=cfg.stochastic_delta_prior_variance,
        convergence_thresh=cfg.convergence_thresh,
        patience=CONVERGENCE_PATIENCE, check_every=CONVERGENCE_CHECK_EVERY,
        lr_schedule=cfg.lr_schedule,
        ultrametric_penalty=cfg.ultrametric_penalty,
        ultrametric_target_cv=cfg.ultrametric_target_cv,
        ultrametric_lr_dual=ULTRAMETRIC_LR_DUAL,
        ultrametric_max_weight=ULTRAMETRIC_MAX_WEIGHT,
        treelike_penalty_n_quartets=TREELIKE_N_QUARTETS,
        treelike_penalty_target_delta=cfg.treelike_penalty_target_delta,
        treelike_penalty_lr_dual=TREELIKE_LR_DUAL,
        treelike_penalty_max_weight=TREELIKE_MAX_WEIGHT if cfg.treelike_penalty else 0.0,
        treelike_penalty_min_weight=TREELIKE_MIN_WEIGHT if cfg.treelike_penalty else 0.0,
        min_iterations=cfg.min_iterations,
        device=device,
        dist_dtype=dist_dtype,
        materialize_dists=not low_memory,
        grad_accum=grad_accum)
    meta.update(_loss_fn_meta(loss_fn))
    if cell_dropout is not None:
        meta["cell_dropout"] = {"shape": list(cd.shape), "mean": float(cd.mean()),
                                "min": float(cd.min()), "max": float(cd.max())}

    # Record the dimension the optimiser actually ran at (coords are Lorentz
    # points, one extra time-like coordinate), not just the one requested.
    realised = int(np.asarray(meta["coords"]).shape[1]) - 1
    if realised != int(dimension):
        raise RuntimeError("embedding ran at %d dimensions but %d were configured"
                           % (realised, dimension))
    meta["dimension"] = realised
    logger.info("embedding dimension realised: %d", realised)
    return dict(leaf_dists=leaf_dists, full_dists=meta["full_dists"],
                labels=labels, coords=meta.get("coords"), meta=meta,
                states_aug=states_aug,
                allele_prior=meta.get("learned_allele_prior", prior_array),
                device=str(device), dist_dtype=dist_dtype,
                low_memory=low_memory, sampler_knn=cfg.sampler_knn, edit_model=model)


def memory_plan(n_cells, dimension, *, low_memory=False, dist_dtype=np.float64,
                readouts=("nj",)):
    """Estimates the host memory of a fit in GB, as a list of readable lines.

    Figures are array sizes (n x n terms dominate above about 20k cells), not
    measured peaks: add about 20 percent plus the Python and torch baseline.
    """
    m = n_cells + 1
    d = dimension + 1
    GB = 1024.0 ** 3
    nn = m * m
    f = np.dtype(dist_dtype).itemsize
    lines = ["n=%d d=%d mode=%s" % (n_cells, dimension,
                                    "low_memory (no n x n array in the optimisation)"
                                    if low_memory else "standard")]
    if low_memory:
        k = min(default_n_landmarks(dimension), m)
        lines.append("fit: landmark seed block %dx%d %.1f GB, coords %dx%d %.1f GB, "
                     "k-NN tables O(n k) < 1 GB" % (k, m, k * m * 8 / GB, m, d, m * d * 8 / GB))
        lines.append("init (rapidnj_clamped): RapidNJ process peaks at ~8 n^2 bytes = %.0f GB "
                     "and reads a ~%.0f GB PHYLIP file "
                     "(HERACLES_TMPDIR); this python process holds one row block + the %dx%d "
                     "landmark block" % (nn * 8 / GB, nn * PHYLIP_BYTES_PER_ENTRY / GB, k, m))
        lines.append("read-outs (inherently n^2): upgma needs ~2x the condensed n(n-1)/2 float64 "
                     "vector (scipy linkage copies it) = %.1f GB; nj: PHYLIP text on disk "
                     "~%.0f GB (under HERACLES_TMPDIR) + RapidNJ ~8 n^2 bytes = %.0f GB "
                     "(without RapidNJ the fallback needs the full %dx%d matrix, %.1f GB)"
                     % (nn * 8 / GB, nn * PHYLIP_BYTES_PER_ENTRY / GB, nn * 8 / GB, m, m, nn * f / GB))
        lines.append("full_dists_/leaf_dists_ are computed on first access: %dx%d %s %.1f GB"
                     % (m, m, np.dtype(dist_dtype).name, nn * f / GB))
    else:
        lines.append("fit: seed matrix %.1f GB (%s) + RapidNJ init tree distances %.1f GB "
                     "float64 + sampler tau+order %.1f GB + fitted matrix %.1f GB"
                     % (nn * f / GB, np.dtype(dist_dtype).name, nn * 8 / GB,
                        nn * 16 / GB, nn * f / GB))
        lines.append("read-outs: upgma ~%.1f GB (two n x n float64 + condensed, copied by scipy); "
                     "nj: PHYLIP text on disk ~%.0f GB + RapidNJ ~8 n^2 bytes"
                     % (nn * 3 * 8 / GB, nn * PHYLIP_BYTES_PER_ENTRY / GB))
        if n_cells >= 20000:
            lines.append("hint: low_memory=True removes the n x n arrays from the optimisation")
    return lines


_FITTED_SCALARS = (
    "mutation_rate_", "heritable_eps_", "stochastic_delta_", "stochastic_delta_sites_",
    "stochastic_delta_cassettes_", "initiation_ratio_",
    "n_iter_", "converged_", "dimension_",
)

# optimiser internals kept in ``diagnostics_`` (one dict) rather than as attributes
_DIAGNOSTIC_KEYS = (
    "init_time", "optimizer_time", "readout_times", "init_method_used", "init_meta",
    "init_tree_newick", "sampler_knn", "stage1_iters", "stage1_converged",
    "stage_transition_step", "stage2_iters", "final_penalty_weight",
    "final_ultrametric_weight", "ultrametric_target_cv", "ultrametric_init_cv",
    "telemetry",
)
_CONFIG_FIELDS = frozenset(HeraclesConfig.field_names())
_FITTED_ARRAYS = ("trees_", "full_dists_", "labels_", "coords_",
                  "coords_history_", "history_", "learned_allele_prior_", "diagnostics_")


class Heracles:
    """Reconstructs a lineage tree from a CRISPR lineage-tracing character matrix.

    Places every cell and the unedited founder as points in hyperbolic space by
    fitting the edit model's likelihood, then builds trees from the distances
    between the points.  Every fit option is a field of :class:`HeraclesConfig`
    (including :class:`AdvancedConfig`), given as keyword arguments or as
    ``config=``; the defaults are the published method.  Options can be read on
    the estimator (``model.seed``) but not set there.

    Args:
        outdir: Output directory (required); ``fit()`` creates it, but its parent
            must exist.  Fits in one process share the ``"heracles"`` logger, so
            run parallel fits in separate processes.
        readouts: Trees to build, ``"nj"`` (default; RapidNJ, or a slower
            pure-Python fallback with a warning) and/or ``"upgma"``.
        edit_model: ``Cas9Model()`` (default), ``PETracerModel(cassette_size=3)``,
            ``SequentialModel(cassette_size=K)``, a registered name, or your own
            (see :mod:`heracles.models.base`).
        init_tree: Optional starting tree (Newick path or string) whose
            branch-length distances seed the embedding; ``init_method`` is ignored.
        config: A :class:`HeraclesConfig`; keyword options override its fields.

    Fitted attributes end in an underscore (``trees_``, ``leaf_dists_``,
    ``coords_``, rates such as ``mutation_rate_`` ...), are None before ``fit()``
    and survive ``save()``/``load()``; ``diagnostics_`` holds optimiser internals
    and run telemetry.
    """

    def __init__(self, *, outdir, readouts=None, edit_model=None,
                 init_tree=None, config=None, **options):
        cfg = config if config is not None else HeraclesConfig()
        if options:
            cfg = cfg.replace(**options)
        if readouts is None:
            readouts = ["nj"]
        elif isinstance(readouts, str):
            readouts = [readouts]
        _check_readouts(readouts)
        outdir = os.fspath(outdir)
        _parent = os.path.dirname(os.path.abspath(outdir))
        if not os.path.isdir(_parent):
            raise ValueError(
                f"outdir parent directory does not exist: {_parent!r} "
                f"(only the last path component may be missing; fit() creates it)")

        self.config = cfg
        self.readouts = list(readouts)
        self.edit_model = get_model(edit_model)
        self.edit_model.validate(None, cfg)
        unregistered = unregistered_parts(self.edit_model)
        if unregistered:
            warnings.warn("Heracles.load() will not be able to rebuild this edit model: "
                          "%s is not registered under its name" % ", ".join(unregistered),
                          stacklevel=2)
        self.init_tree = None if init_tree is None else os.fspath(init_tree)
        self.outdir = outdir
        self._device = str(resolve_device(cfg.device))     # resolved; config.device is as given

        for name in _FITTED_SCALARS + _FITTED_ARRAYS:
            setattr(self, name, None)

    # config fields read through the estimator (``model.seed`` == ``model.config.seed``);
    # they are set only through ``config=`` / keyword options so the two never drift
    def __getattr__(self, name):
        if name in _CONFIG_FIELDS and "config" in self.__dict__:
            return getattr(self.config, name)
        raise AttributeError("%r object has no attribute %r" % (type(self).__name__, name))

    def __setattr__(self, name, value):
        if name in _CONFIG_FIELDS:
            raise AttributeError(
                "%s is a fit option: pass it to Heracles(...) or as config=HeraclesConfig(...) "
                "instead of setting it on the estimator" % name)
        object.__setattr__(self, name, value)

    def _objective_name(self):
        return self.edit_model.name

    def _check_fitted(self):
        if self.labels_ is None:
            raise RuntimeError(_NOT_FIT_MSG)

    # -- fitted distances: eager in the standard fit, derived from coords_ on
    #    first access in low-memory mode (and for coords-only model.npz files)
    @property
    def full_dists_(self):
        """``(n+1, n+1)`` distances, founder at row 0; low-memory mode computes them on first access."""
        if self._full_dists is None and self.coords_ is not None:
            m = self.coords_.shape[0]
            dtype = np.float32 if self.config.low_memory else np.float64
            logger.info("computing the %d x %d distance matrix from the coordinates "
                        "(%.1f GB %s)", m, m, m * m * np.dtype(dtype).itemsize / 1024 ** 3,
                        np.dtype(dtype).name)
            self._full_dists = pairwise_dists(self.coords_, dtype=dtype,
                                              device=self._device)
        return self._full_dists

    @full_dists_.setter
    def full_dists_(self, value):
        self._full_dists = value

    @property
    def leaf_dists_(self):
        """``(n, n)`` leaf-to-leaf embedding distances (``full_dists_[1:, 1:]``)."""
        fd = self.full_dists_
        return None if fd is None else fd[1:, 1:]

    def fit(self, char_matrix, cell_dropout=None):
        """Fits the embedding, builds the read-out trees and saves the model.

        Writes ``model.npz``, ``tree_<readout>.nwk``, ``config.json`` and
        ``heracles.log`` into ``outdir``, and returns self.  The first read-out
        is also written with edit-free edges contracted, as
        ``tree_<readout>_collapsed.nwk`` (see :meth:`newick`).  Warns if the fit did
        not converge within ``max_iterations``.

        Args:
            char_matrix: Cells x sites DataFrame, or a CSV path.
            cell_dropout: Optional fixed dropout probability per cell and site (see :func:`embed`).
        """
        from datetime import datetime, timezone

        snapshot_epochs = None
        if self.config.snapshot_every is not None and self.config.snapshot_every > 0:
            snapshot_epochs = list(range(0, self.config.max_iterations + 1,
                                        self.config.snapshot_every))

        if (isinstance(char_matrix, pd.DataFrame)
                and isinstance(char_matrix.index, pd.RangeIndex)):
            warnings.warn("the character matrix has no cell names (index 0..n-1); if the "
                          "first column holds them, read it with pd.read_csv(path, "
                          "index_col=0)", stacklevel=2)
        start_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        commit = git_commit()

        os.makedirs(self.outdir, exist_ok=True)
        run_logger, _handlers, _prev_level = setup_run_logging(self.outdir)
        try:
            return self._fit_inner(char_matrix, snapshot_epochs, start_time,
                                   commit, cell_dropout=cell_dropout)
        finally:
            teardown_run_logging(run_logger, _handlers, _prev_level)

    def _fit_inner(self, char_matrix, snapshot_epochs, start_time, commit,
                   cell_dropout=None):
        """Body of ``fit()``, run with the log handlers attached."""
        import platform
        import resource
        import time
        from heracles import __version__

        # config.json first: the input record must survive a crashed fit;
        # save() rewrites it with the fitted values
        self._run_info = {"git_commit": commit, "package_version": __version__,
                          "timestamp": start_time}
        self._write_config_json(self.outdir)

        logger.info(
            "fit start: edit_model=%s init_method=%s "
            "max_iterations=%d device=%s version=%s git=%s outdir=%s",
            self._objective_name(), self.config.init_method,
            self.config.max_iterations, self._device, __version__, commit,
            self.outdir)

        captured_warnings = []
        t0 = time.perf_counter()

        cfg, init_dists = self.config, None
        if self.init_tree is not None:
            from heracles.embedding.initialization import load_init_tree_dists
            _, cell_labels = load_character_matrix(char_matrix, missing_state=cfg.missing_state)
            init_dists = load_init_tree_dists(self.init_tree, cell_labels)
            cfg = cfg.replace(init_method="mds")      # tree distances are embedded directly
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emb = embed(
                char_matrix, config=cfg,
                edit_model=self.edit_model,
                init_dists=init_dists,
                snapshot_epochs=snapshot_epochs,
                cell_dropout=cell_dropout)
            captured_warnings.extend(caught)

        for w in caught:
            logger.warning(str(w.message), extra={"file_only": True})
            warnings.warn(w.message, category=w.category, stacklevel=3)

        elapsed = time.perf_counter() - t0
        peak_gpu_mb = None
        if torch.cuda.is_available() and str(resolve_device(self._device)).startswith("cuda"):
            peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        mem1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes
        scale = 1 if platform.system() == "Linux" else 1024
        peak_mb = mem1 / scale / 1024

        meta = emb["meta"]
        history = meta.get("history", {"loss": []})
        loss_history = history.get("loss", [])
        n_iter = meta.get("n_iter", len(loss_history))
        converged = meta.get("converged", False)

        self.full_dists_ = emb["full_dists"]          # None in low-memory mode
        self.labels_ = emb["labels"]
        self.history_ = history
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.mutation_rate_ = meta.get("mutation_rate")
        self.heritable_eps_ = meta.get("heritable_eps")
        self.stochastic_delta_ = meta.get("stochastic_delta")     # per-site: site mean
        self.stochastic_delta_sites_ = meta.get("stochastic_delta_sites")   # (S,) list or None
        self.stochastic_delta_cassettes_ = meta.get("stochastic_delta_cassettes")   # (C,) list or None
        self.initiation_ratio_ = (meta.get("initiation_ratio_learned")
                                  or meta.get("initiation_ratio"))
        self.coords_ = meta.get("coords")
        self.dimension_ = meta.get("dimension")
        init_meta = dict(meta.get("init_meta") or {})
        diag = self.diagnostics_ = {
            "init_time": meta.get("init_time"),
            "optimizer_time": meta.get("optimizer_time"),
            "readout_times": {},
            "init_method_used": init_meta.get("init_method_used"),
            "init_tree_newick": init_meta.pop("init_tree_newick", None),
            "init_meta": init_meta,
            "sampler_knn": emb.get("sampler_knn"),     # k actually used (low_memory sets a default)
        }
        diag.update({k: meta.get(k) for k in (
            "stage1_iters", "stage1_converged", "stage_transition_step", "stage2_iters",
            "final_penalty_weight", "final_ultrametric_weight", "ultrametric_target_cv",
            "ultrametric_init_cv")})

        if snapshot_epochs and "snapshots" in meta:
            self.coords_history_ = meta["snapshots"]
        else:
            self.coords_history_ = None

        self.learned_allele_prior_ = meta.get("learned_allele_prior")

        if not converged and self.config.convergence_thresh is not None:
            _last = loss_history[-1] if loss_history else float("nan")
            msg = (f"the loss was still improving when the fit reached max_iterations="
                   f"{self.config.max_iterations} (last loss {_last:.6g}); the trees use "
                   f"the final embedding. To fit longer raise max_iterations; to stop "
                   f"sooner raise convergence_thresh.")
            captured_warnings.append(msg)
            logger.warning(msg, extra={"file_only": True})
            warnings.warn(msg, stacklevel=3)

        self.trees_ = {}
        self.save()
        logger.info("pre-readout save -> %s/model.npz",
                     os.path.basename(self.outdir))
        for r in self.readouts:
            _t_ro = time.perf_counter()
            self.trees_[r] = READOUTS[r](emb)
            diag["readout_times"][r] = time.perf_counter() - _t_ro
            logger.info("readout %s: %.2fs", r, diag["readout_times"][r])
        self._add_collapsed(self.readouts[0], char_matrix)

        n_cells = len(self.labels_)
        n_sites = int(np.asarray(emb["states_aug"]).shape[1])

        diag["telemetry"] = {
            "git_commit": commit,
            "start_time": start_time,
            "elapsed_s": elapsed,
            "init_time_s": diag["init_time"],
            "init_breakdown_s": {
                k: init_meta.get(src)
                for k, src in (("tree_creation", "tree_creation_s"),
                               ("distance_extraction", "distance_extraction_s"),
                               ("embedding", "embedding_s"))},
            "optimizer_time_s": diag["optimizer_time"],
            "readout_times_s": dict(diag["readout_times"]),
            "peak_rss_mb": peak_mb,
            "peak_gpu_mb": peak_gpu_mb,
            "n_cells": n_cells,
            "n_sites": n_sites,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_device": str(resolve_device(self._device)),
            "geoopt_version": geoopt.__version__,
            "warnings": [str(w.message) if hasattr(w, "message") else str(w)
                         for w in captured_warnings],
        }

        logger.info(
            "fit complete: %.1fs total (init %.1fs, optimizer %.1fs, "
            "readouts %.1fs), n_iter=%d, converged=%s, peak_rss=%.0f MB",
            time.perf_counter() - t0,
            diag["init_time"] or 0.0, diag["optimizer_time"] or 0.0,
            sum(diag["readout_times"].values()), n_iter, converged, peak_mb)

        self.save()
        return self

    def polish(self, char_matrix, readout="nj", *, write=True, **kwargs):
        """Improves a read-out tree with likelihood NNI moves (:mod:`heracles.polish.nni`).

        Stores the result as ``trees_[readout + "_polished"]``, and its collapsed
        form as ``trees_[readout + "_polished_collapsed"]``, and returns the
        :class:`~heracles.polish.nni.RefitResult`.  Raises ``ValueError`` if the edit
        model does not support the polish.

        Args:
            char_matrix: The matrix the model was fitted on.
            write: Re-save the model, writing ``tree_<readout>_polished.nwk`` and
                ``tree_<readout>_polished_collapsed.nwk``.
            **kwargs: Passed to :func:`heracles.polish.nni.polish_model`.
        """
        from heracles.polish.nni import polish_model
        self._check_fitted()
        if not self.edit_model.supports_polish:
            raise ValueError("the likelihood NNI polish is not available for %s"
                             % type(self.edit_model).__name__)
        res = polish_model(self, char_matrix, readout=readout,
                           log=kwargs.pop("log", logger.info), **kwargs)
        name = readout + "_polished"
        self.trees_[name] = res.tree
        self._add_collapsed(name, char_matrix)
        logger.info("polish %s: %d moves in %d rounds, log-lik %.2f -> %.2f (%.1fs)",
                    readout, len(res.moves), res.n_rounds, res.loglik_start,
                    res.loglik_end, res.wall_s)
        if write and self.outdir:
            self.save()
        return res

    def newick(self, readout="nj"):
        """Returns the Newick string of ``trees_[readout]`` (``KeyError`` if absent).

        Read-out trees are rooted at the founder.  ``"<readout>_collapsed"`` trees
        have every edge that carries no edit contracted and the founder removed;
        these are the trees to compare against a reference.
        """
        self._check_fitted()
        if readout not in (self.trees_ or {}):
            raise KeyError("no %r tree (have %s)" % (readout, sorted(self.trees_ or {})))
        nwk = as_newick(self.trees_[readout])
        return nwk if nwk.endswith(";") else nwk + ";"

    def get_params(self):
        """Returns a JSON-ready dict: package version, constructor arguments, config fields
        and, once fitted, the fitted scalars under ``"fitted"``.
        """
        from heracles import __version__
        d = {"__version__": __version__,
             "readouts": self.readouts,
             "edit_model": self.edit_model.to_dict(),
             "init_tree": (None if self.init_tree is None
                           else "<newick string>" if is_newick_string(self.init_tree)
                           else self.init_tree),
             "outdir": self.outdir}
        d.update(self.config.to_dict())
        d["device"] = self._device
        if self.labels_ is not None:
            fitted = {k: getattr(self, k) for k in _FITTED_SCALARS}
            if self.learned_allele_prior_ is not None:
                fitted["learned_allele_prior_"] = self.learned_allele_prior_.tolist()
            d["fitted"] = fitted
        return d

    def save(self, path=None):
        """Writes ``model.npz``, ``tree_<readout>.nwk`` and ``config.json``.

        *path* is the ``.npz`` file (default ``<outdir>/model.npz``); the other files go beside it.
        """
        self._check_fitted()
        if path is None:
            path = os.path.join(self.outdir, "model.npz")
        directory = os.path.dirname(os.path.abspath(path))
        self._write_trees(directory)
        self._write_config_json(directory)
        arrays = {}
        if self._full_dists is not None:          # never force the lazy build here
            arrays["full_dists"] = self._full_dists
        for name, series in (self.history_ or {}).items():
            arrays["history_%s" % name] = np.array(series, dtype=np.float64)
        if self.coords_ is not None:
            arrays["coords"] = self.coords_
        if self.learned_allele_prior_ is not None:
            arrays["learned_allele_prior"] = self.learned_allele_prior_
        if self.coords_history_:
            for epoch, snap in self.coords_history_.items():
                arrays["snapshot_%d" % epoch] = snap

        trees_nwk = {}
        if self.trees_:
            for r, t in self.trees_.items():
                trees_nwk[r] = as_newick(t)

        metadata = self.get_params()
        metadata["labels"] = self.labels_
        metadata["trees_nwk"] = trees_nwk
        metadata["diagnostics"] = self.diagnostics_

        arrays["_metadata_json"] = np.array(json.dumps(metadata, default=str))
        np.savez_compressed(path, **arrays)

    def add_readouts(self, names, *, device=None, save=True):
        """Builds more read-out trees (e.g. ``["upgma"]``) from a fitted or loaded model.

        Uses ``full_dists_`` if in memory, otherwise distances from ``coords_`` on
        *device* (default: the fit's device).  With *save*, re-saves the model,
        writing ``tree_<name>.nwk``.  Returns a dict of the new trees by name.
        """
        import time
        self._check_fitted()
        if isinstance(names, str):
            names = [names]
        _check_readouts(names)
        emb = {"full_dists": self._full_dists, "labels": self.labels_, "coords": self.coords_,
               "device": str(resolve_device(device if device is not None else self._device)),
               "dist_dtype": np.float32 if self.config.low_memory else np.float64}
        if emb["full_dists"] is None and emb["coords"] is None:
            raise RuntimeError("add_readouts needs coords_ or full_dists_")
        self.trees_ = dict(self.trees_ or {})
        self.diagnostics_ = dict(self.diagnostics_ or {})
        readout_times = self.diagnostics_.setdefault("readout_times", {})
        new = {}
        for r in names:
            t0 = time.perf_counter()
            new[r] = self.trees_[r] = READOUTS[r](emb)
            readout_times[r] = time.perf_counter() - t0
            logger.info("readout %s (added): %.2fs", r, readout_times[r])
            if r not in self.readouts:
                self.readouts.append(r)
        if save:
            self.save()
        return new

    def _add_collapsed(self, name, char_matrix):
        """Adds ``trees_[name + "_collapsed"]``: ``trees_[name]`` with edit-free edges contracted."""
        import time
        from heracles.trees.collapse import sokal_collapse
        t0 = time.perf_counter()
        states, labels = load_character_matrix(char_matrix,
                                               missing_state=self.config.missing_state)
        cm = pd.DataFrame(states, index=labels)
        self.trees_[name + "_collapsed"] = sokal_collapse(self.trees_[name], cm)
        logger.info("collapsed %s: %.2fs", name, time.perf_counter() - t0)

    def _write_trees(self, outdir):
        """``tree_<readout>.nwk`` for every tree."""
        os.makedirs(outdir, exist_ok=True)
        for rname, tree in (self.trees_ or {}).items():
            write_newick(os.path.join(outdir, "tree_%s.nwk" % rname), tree)

    def _write_config_json(self, outdir):
        """Writes ``config.json``: settings, run information and fitted values."""
        os.makedirs(outdir, exist_ok=True)
        params = self.get_params()
        fitted = params.pop("fitted", None)
        if fitted:      # the (sites x states) prior is in model.npz; keep this file readable
            fitted = {k: v for k, v in fitted.items() if k != "learned_allele_prior_"}
        record = {"fitted": fitted, "params": params}
        record.update(getattr(self, "_run_info", None) or {})
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(record, f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        """Restores a fitted model from a ``model.npz`` written by ``save()``.

        If the saved ``outdir``'s parent is missing here, ``outdir`` becomes the directory of *path*.
        """
        try:
            data = np.load(path, allow_pickle=False)
        except ValueError:
            data = None
        if not hasattr(data, "files") or "_metadata_json" not in data.files:
            raise ValueError("%s is not a model.npz written by Heracles.save()" % (path,))
        meta = json.loads(str(data["_metadata_json"]))

        cfg, ignored = HeraclesConfig.from_params(meta)
        edit_model = model_from_legacy_params(meta)
        bookkeeping = {"__version__", "readouts", "objective", "outdir", "fitted", "labels",
                       "trees_nwk", "telemetry", "device", "edit_model", "likelihood",
                       "dropout_unit", "cassette_size", "initiation_ratio",
                       "learn_initiation_ratio", "stochastic_delta_per_site",
                       "patience", "check_every", "torch_compile", "compile", "init_tree"}
        dropped = sorted(k for k in ignored if k not in bookkeeping)
        if dropped:
            logger.info("load: ignoring keys not in HeraclesConfig: %s", ", ".join(dropped))
        # An archive from another machine: fall back to the npz's directory
        # (nothing is written there unless fit() is called again).
        outdir = meta.get("outdir") or os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(os.path.dirname(os.path.abspath(outdir))):
            outdir = os.path.dirname(os.path.abspath(path))
        # a saved 'rapidnj' read-out is the 'nj' read-out (NJ built with RapidNJ)
        readouts = list(dict.fromkeys("nj" if r == "rapidnj" else r
                                      for r in (meta.get("readouts") or ["nj"])))
        obj = cls(outdir=outdir, readouts=readouts, config=cfg,
                  edit_model=edit_model)

        obj.full_dists_ = data["full_dists"] if "full_dists" in data else None
        obj.labels_ = list(meta["labels"])
        history = {key[len("history_"):]: data[key].tolist()
                   for key in data.files if key.startswith("history_")}
        obj.history_ = history or None
        fitted = meta.get("fitted", {})
        for k in _FITTED_SCALARS:
            setattr(obj, k, fitted.get(k))
        obj.diagnostics_ = meta.get("diagnostics") or _diagnostics_from_flat_keys(meta)
        if "coords" in data:
            obj.coords_ = data["coords"]
            # the coordinates fix the dimension; a recorded dimension_ that
            # disagrees (the requested rather than the realised one) is overridden
            true_dim = int(obj.coords_.shape[1]) - 1
            if obj.dimension_ is not None and int(obj.dimension_) != true_dim:
                logger.warning("load: model.npz records dimension_=%s but the "
                               "coordinates have %d dimensions; using %d",
                               obj.dimension_, true_dim, true_dim)
            obj.dimension_ = true_dim
        if "learned_allele_prior" in data:
            obj.learned_allele_prior_ = data["learned_allele_prior"]
        snapshots = {int(key.split("_", 1)[1]): data[key]
                     for key in data.files if key.startswith("snapshot_")}
        obj.coords_history_ = snapshots or None

        trees_nwk = meta.get("trees_nwk", {})
        if trees_nwk:
            from heracles.trees.tree import TaxonNamespace
            from heracles.trees.utils import parse_newick
            tns = TaxonNamespace(obj.labels_)
            obj.trees_ = {r: parse_newick(nwk, taxon_namespace=tns,
                                          rooting="force-rooted")
                          for r, nwk in trees_nwk.items()}
        return obj

    def write_report(self, path=None):
        """Writes the diagnostic plots into ``outdir`` and an HTML report; returns its path.

        *path* defaults to ``<outdir>/report.html``.
        """
        from heracles.report import generate_report, save_plots
        self._check_fitted()
        save_plots(self)
        return generate_report(self, path)

    def __repr__(self):
        fitted = " [fitted, %d iter%s]" % (
            self.n_iter_, ", converged" if self.converged_ else ""
        ) if self.n_iter_ is not None else ""
        dim_str = str(self.config.dimension) if self.config.dimension is not None else "5*sqrt(n)"
        return ("Heracles(edit_model=%s, readouts=%r, dimension=%s, "
                "max_iterations=%d, init_method=%r)%s"
                % (self.edit_model.describe(), self.readouts,
                   dim_str, self.config.max_iterations, self.config.init_method, fitted))
