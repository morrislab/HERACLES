"""Options of the method, with the published defaults.

:class:`HeraclesConfig` holds the 14 options a user normally sets;
:class:`AdvancedConfig` (``config.advanced``) holds 19 method internals.
The estimator, ``config.json``, ``save()``/``load()`` and the command-line
flags (:func:`add_config_arguments`) are all derived from these fields.
"""

import dataclasses
from dataclasses import dataclass, field, fields

import numpy as np


__all__ = [
    'AUTO_RATE_START',
    'HeraclesConfig',
    'AdvancedConfig',
    'add_config_arguments',
    'config_from_args',
]


#: starting value of a dropout rate given as ``"auto"``
AUTO_RATE_START = 1e-3


# value parsers: strings from the CLI or config.json -> Python values
def _dimension(v):
    if v is None or (isinstance(v, str) and v.lower() in ("none", "default")):
        return None
    if isinstance(v, str) and v.lower() == "auto":
        raise ValueError(
            "dimension='auto' is not supported; use None (the default, "
            "ceil(5*sqrt(n))) or an int")
    d = int(v)
    if d < 2:
        raise ValueError("dimension must be >= 2 or None, got %r" % (v,))
    return d


def _auto_or_float(v):
    if isinstance(v, str):
        return "auto" if v.lower() == "auto" else float(v)
    return float(v)


def _auto_none_or_float(v):
    if v is None:
        return None
    if isinstance(v, str):
        if v.lower() == "auto":
            return "auto"
        if v.lower() in ("none", "flat"):
            return None
        return float(v)
    return float(v)


def _adaptive_or_float(v):
    if isinstance(v, str):
        return "adaptive" if v.lower() == "adaptive" else float(v)
    return float(v)


def _none_or_float(v):
    if v is None or (isinstance(v, str) and v.lower() == "none"):
        return None
    return float(v)


def _none_or_int(v):
    if v is None or (isinstance(v, str) and v.lower() == "none"):
        return None
    return int(v)


def _none_or_str(v):
    if v is None or (isinstance(v, str) and v.lower() == "none"):
        return None
    return str(v)


def _bool(v):
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return bool(v)


def _opt(default, help, section, *, parse=None, choices=None, aliases=(), type=None):
    return field(default=default, metadata={
        "help": help, "section": section, "parse": parse,
        "choices": choices, "aliases": tuple(aliases), "type": type})


def _parse_fields(obj):
    """Parses string values of *obj*'s option fields in place."""
    for f in fields(obj):
        if f.metadata.get("nested"):
            continue
        v = getattr(obj, f.name)
        parse = f.metadata.get("parse")
        try:
            if f.metadata.get("type") is bool:
                setattr(obj, f.name, _bool(v))
            elif parse is not None:
                setattr(obj, f.name, parse(v))
            elif isinstance(v, str) and f.type in (int, float):
                setattr(obj, f.name, f.type(v))
        except (TypeError, ValueError) as e:
            detail = (": " + str(e)) if isinstance(e, ValueError) and str(e) else ""
            raise ValueError("%s: %s (got %r)%s"
                             % (f.name, f.metadata["help"], v, detail)) from e


def _json_value(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


@dataclass
class AdvancedConfig:
    """Method internals: rates and priors, quartet sampler, penalties, optimiser, large inputs.

    Most users never change them.  Every field can be passed and read flat, e.g.
    ``Heracles(band_kmax=32)`` or ``config.band_kmax``.
    """

    # -- rates and priors
    learn_mutation_rate: bool = _opt(
        False, "learn a global mutation-rate scalar", "Advanced: rates and priors", type=bool)
    heritable_eps_prior_variance: object = _opt(
        "auto", "Gamma prior variance on heritable_eps: 'auto', a float, 'none'",
        "Advanced: rates and priors", parse=_auto_none_or_float,
        aliases=("--eps-prior-variance",))
    stochastic_delta_prior_variance: object = _opt(
        "auto", "Beta prior variance on stochastic_delta: 'auto', a float, 'none'",
        "Advanced: rates and priors", parse=_auto_none_or_float,
        aliases=("--delta-prior-variance",))

    # -- quartet sampler
    batch_size: object = _opt(
        None, "quartets per step; 'none' = min(15 n, C(n, 4))", "Advanced: quartet sampler",
        parse=_none_or_int)
    band_kmin: int = _opt(2, "closest neighbour rank for the second leaf",
                          "Advanced: quartet sampler")
    band_kmax: int = _opt(16, "farthest neighbour rank for the second leaf at step 0",
                          "Advanced: quartet sampler")
    band_kmax_end: object = _opt(
        128, "band_kmax after the ramp; 'none' = no ramp", "Advanced: quartet sampler",
        parse=_none_or_int)
    band_local_frac: float = _opt(
        1.0, "share of quartets with a local-outgroup third leaf", "Advanced: quartet sampler")
    band_local_mult_max: object = _opt(
        4.0, "rank-multiplier cap for the local outgroup; 'none' = fixed ring",
        "Advanced: quartet sampler", parse=_none_or_float)

    # -- penalties
    ultrametric_penalty: object = _opt(
        "adaptive", "'adaptive', a fixed weight, or 0 = off", "Advanced: penalties",
        parse=_adaptive_or_float)
    ultrametric_target_cv: object = _opt(
        "auto", "target depth CV; 'auto' = 0.75 x the init CV", "Advanced: penalties",
        parse=_auto_or_float)
    treelike_penalty: bool = _opt(
        True, "stage-2 four-point (tree-likeness) penalty", "Advanced: penalties", type=bool)
    treelike_penalty_target_delta: float = _opt(
        0.01, "target mean four-point defect", "Advanced: penalties")

    # -- optimiser
    min_iterations: int = _opt(1500, "steps before convergence checks start",
                               "Advanced: optimiser")
    lr: float = _opt(0.02, "Riemannian Adam step size", "Advanced: optimiser")
    lr_schedule: object = _opt(None, "'cosine' or 'none'", "Advanced: optimiser",
                               parse=_none_or_str, choices=("cosine", "none"))
    snapshot_every: object = _opt(
        None, "record coordinates every k steps; 'none' = never", "Advanced: optimiser",
        parse=_none_or_int)

    # -- large inputs
    grad_accum: int = _opt(
        1, "micro-batches per step (same objective, 1/k GPU memory)", "Advanced: large inputs")
    sampler_knn: object = _opt(
        None, "k nearest neighbours kept per leaf; 'none' = full sort",
        "Advanced: large inputs", parse=_none_or_int)

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Parses string values and checks ranges (``ValueError``); called on construction."""
        _parse_fields(self)
        if self.lr_schedule not in (None, "cosine"):
            raise ValueError("lr_schedule must be 'cosine' or None, got %r" % (self.lr_schedule,))
        if int(self.grad_accum) < 1:
            raise ValueError("grad_accum must be >= 1")
        if self.band_kmin < 1 or self.band_kmax < self.band_kmin:
            raise ValueError("need 1 <= band_kmin <= band_kmax")
        if not 0.0 <= float(self.band_local_frac) <= 1.0:
            raise ValueError("band_local_frac must be in [0, 1]")
        return self

    def replace(self, **changes):
        """Returns a validated copy with some fields changed."""
        return dataclasses.replace(self, **changes)


_ADVANCED = frozenset(f.name for f in fields(AdvancedConfig))


@dataclass(init=False)
class HeraclesConfig:
    """Options of a HERACLES fit; the defaults are the published method.

    Method internals live in :class:`AdvancedConfig` (``config.advanced``) and can
    be passed and read flat too.  Strings such as ``"auto"``, ``"none"`` or numbers
    as text are parsed on construction.  Raises ``TypeError`` for an unknown option
    and ``ValueError`` for a value that cannot be parsed or is out of range.
    """

    # -- model
    dimension: object = _opt(
        None, "embedding dimension; 'none' = ceil(5*sqrt(n cells))", "Model", parse=_dimension)
    allele_prior: object = _opt(
        "per_site_uniform", "'per_site_uniform' or 'global_uniform' (Python API: also an array)",
        "Model", choices=("per_site_uniform", "global_uniform"))
    learn_allele_prior: bool = _opt(True, "learn the allele weights", "Model", type=bool)

    # -- missing data
    heritable_eps: object = _opt(
        "auto", "heritable deletion rate; 'auto' = start at 1e-3",
        "Missing data", parse=_auto_or_float, aliases=("--eps",))
    learn_heritable_eps: bool = _opt(
        True, "learn heritable_eps", "Missing data", type=bool, aliases=("--learn-eps",))
    stochastic_delta: object = _opt(
        "auto", "stochastic dropout probability; 'auto' = start at 1e-3",
        "Missing data", parse=_auto_or_float, aliases=("--delta",))
    learn_stochastic_delta: bool = _opt(
        True, "learn stochastic_delta", "Missing data", type=bool, aliases=("--learn-delta",))
    missing_state: int = _opt(-1, "integer code for a missing entry", "Missing data")

    # -- fit
    init_method: str = _opt(
        "rapidnj_clamped", "'rapidnj_clamped' (MDS fallback without the binary), 'upgma' or 'mds'",
        "Fit", choices=("rapidnj_clamped", "upgma", "mds"))
    max_iterations: int = _opt(5000, "hard cap on optimiser steps", "Fit")
    convergence_thresh: object = _opt(
        1e-4, "relative loss change that counts as converged; 'none' = never", "Fit",
        parse=_none_or_float)
    seed: int = _opt(0, "random seed", "Fit")
    device: str = _opt("cpu", "'cpu', 'cuda', or 'auto'", "Fit")
    low_memory: bool = _opt(False, "no n x n matrix during the optimisation (large inputs)", "Fit", type=bool)

    advanced: AdvancedConfig = field(default_factory=AdvancedConfig,
                                     metadata={"nested": True})

    def __init__(self, *, advanced=None, **options):
        unknown = sorted(set(options) - set(_BASIC) - _ADVANCED)
        if unknown:
            import difflib
            known = sorted(set(_BASIC) | _ADVANCED)
            hints = ["%s (did you mean %s?)" % (u, m[0]) if m else u
                     for u in unknown
                     for m in [difflib.get_close_matches(u, known, n=1)]]
            raise TypeError("unknown HeraclesConfig option(s): %s" % ", ".join(hints))
        for f in fields(HeraclesConfig):
            if not f.metadata.get("nested"):
                setattr(self, f.name, options.get(f.name, f.default))
        adv = advanced if advanced is not None else AdvancedConfig()
        adv_changes = {k: v for k, v in options.items() if k in _ADVANCED}
        self.advanced = adv.replace(**adv_changes) if adv_changes else adv
        self.validate()

    def __getattr__(self, name):
        # flat read access to the advanced options: config.band_kmax
        if name in _ADVANCED and "advanced" in self.__dict__:
            return getattr(self.__dict__["advanced"], name)
        raise AttributeError("%r object has no attribute %r" % (type(self).__name__, name))

    def validate(self):
        """Parses string values and checks ranges; called on construction.

        ``"auto"`` dropout rates are replaced by ``AUTO_RATE_START``.
        """
        _parse_fields(self)
        # "auto" dropout rates start at AUTO_RATE_START; whether they are
        # learned is governed by learn_* alone (defaults: True).
        if self.heritable_eps == "auto":
            self.heritable_eps = AUTO_RATE_START
        if self.stochastic_delta == "auto":
            self.stochastic_delta = AUTO_RATE_START
        for name, learn in (("heritable_eps", self.learn_heritable_eps),
                            ("stochastic_delta", self.learn_stochastic_delta)):
            if learn and float(getattr(self, name)) <= 0.0:
                raise ValueError(
                    "%s=%r cannot be learned from zero (the prior needs a positive "
                    "start); pass a positive value or learn_%s=False"
                    % (name, getattr(self, name), name))
        if isinstance(self.allele_prior, str):
            if self.allele_prior not in ("per_site_uniform", "global_uniform"):
                raise ValueError("allele_prior must be 'per_site_uniform', "
                                 "'global_uniform' or an array, got %r" % (self.allele_prior,))
        else:
            self.allele_prior = np.asarray(self.allele_prior, dtype=np.float64)
        if self.init_method not in ("rapidnj_clamped", "upgma", "mds"):
            raise ValueError("init_method must be 'rapidnj_clamped', 'upgma' or 'mds', got %r"
                             % (self.init_method,))
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        return self

    def replace(self, **changes):
        """Returns a validated copy with some options (basic or advanced) changed."""
        basic = {name: getattr(self, name) for name in _BASIC}
        basic.update({k: v for k, v in changes.items() if k not in _ADVANCED})
        adv = {k: v for k, v in changes.items() if k in _ADVANCED}
        return HeraclesConfig(advanced=self.advanced, **basic, **adv)

    def to_dict(self):
        """Returns every option, basic and advanced, as one flat JSON-ready dict."""
        d = {name: _json_value(getattr(self, name)) for name in _BASIC}
        d.update({name: _json_value(getattr(self.advanced, name)) for name in _ADVANCED_ORDER})
        return d

    @classmethod
    def field_names(cls):
        """Returns every option name, basic first, then advanced."""
        return tuple(_BASIC) + tuple(_ADVANCED_ORDER)

    @classmethod
    def option_fields(cls):
        """Returns the ``dataclasses.Field`` of every option, basic first."""
        return tuple(f for f in fields(cls) if not f.metadata.get("nested")) \
            + tuple(fields(AdvancedConfig))

    @classmethod
    def from_params(cls, params, base=None):
        """Builds a config from a ``get_params()`` or ``config.json`` dict.

        Returns ``(config, ignored_keys)``: keys that are not options are skipped.
        *base* fills unspecified options (default: the defaults).
        """
        known = set(cls.field_names())
        kw, ignored = {}, []
        for k, v in dict(params).items():
            if k == "dimension" and isinstance(v, str) and v.lower() == "auto":
                kw[k] = None          # read as the default dimension
                ignored.append("dimension=auto (read as the default ceil(5*sqrt(n)))")
            elif k in known:
                kw[k] = v
            elif k == "quartet_kwargs" and isinstance(v, dict):
                for qk, qv in v.items():
                    if qk in known:
                        kw[qk] = qv
                    else:
                        ignored.append("quartet_kwargs.%s" % qk)
            else:
                ignored.append(k)
        base = base if base is not None else cls()
        return base.replace(**kw), ignored


_BASIC = tuple(f.name for f in fields(HeraclesConfig) if not f.metadata.get("nested"))
_ADVANCED_ORDER = tuple(f.name for f in fields(AdvancedConfig))


# argparse integration
def _flag(name):
    return "--" + name.replace("_", "-")


def add_config_arguments(parser, *, exclude=()):
    """Adds one command-line flag per option to an argparse *parser*.

    Boolean options get ``--name/--no-name``; aliases such as ``--eps`` are
    registered too.  Unset flags keep the default.  Options named in
    *exclude* are skipped.  Returns *parser*.
    """
    import argparse
    groups = {}
    for f in HeraclesConfig.option_fields():
        if f.name in exclude:
            continue
        section = f.metadata["section"]
        g = groups.get(section)
        if g is None:
            g = groups[section] = parser.add_argument_group(section)
        flags = (_flag(f.name),) + f.metadata["aliases"]
        help_ = "%s (default: %s)" % (f.metadata["help"], f.default)
        if f.metadata.get("type") is bool:
            g.add_argument(*flags, dest=f.name, default=None,
                           action=argparse.BooleanOptionalAction, help=help_)
        else:
            choices = f.metadata.get("choices")
            g.add_argument(*flags, dest=f.name, default=None, metavar=f.name.upper(),
                           help=help_ + (" [%s]" % "|".join(choices) if choices else ""))
    return parser


def config_from_args(args, *, base=None, exclude=()):
    """Builds a config from parsed arguments of :func:`add_config_arguments`.

    Flags left unset keep the values of *base* (default: the defaults).
    """
    kw = {}
    for f in HeraclesConfig.option_fields():
        if f.name in exclude:
            continue
        v = getattr(args, f.name, None)
        if v is not None:
            kw[f.name] = v
    base = base if base is not None else HeraclesConfig()
    return base.replace(**kw)
