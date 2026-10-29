"""The edit-model framework: an edit process plus a missing-data model.

* :class:`EditProcess`: how sites get edited along the tree.  It owns the
  quartet loss that fits the embedding and, optionally, the whole-tree
  likelihood the polish uses.
* :class:`MissingData`: which sites go missing together, one at a time
  (:class:`~heracles.models.SiteDropout`) or as whole cassettes
  (:class:`~heracles.models.CassetteDropout`).
* :class:`EditModel`: the pair, which ``Heracles(edit_model=...)`` takes.

=========================  =====================  ======================
preset                     edit process           missing data
=========================  =====================  ======================
``Cas9Model()``            ``Cas9Process``        ``SiteDropout``
``PETracerModel()``        ``Cas9Process``        ``CassetteDropout(3)``
``SequentialModel(K)``     ``SequentialProcess``  ``CassetteDropout(K)``
=========================  =====================  ======================

Writing your own edit process
-----------------------------
Subclass :class:`EditProcess` as a ``dataclass`` whose fields are its options,
set ``name`` and ``supported_missing``, implement ``seed_distances`` and
``loss_function``, and register it so ``Heracles.load`` can rebuild it.  Most
processes only need a scoring function: :func:`heracles.embedding.objective.quartet_objective`
does the quartet sampling, distances, branch lengths and penalties around it.
A complete (if crude) example, scoring each quartet's cherry and each leaf's
distance from the founder by whether sites are edited::

    from heracles.embedding.initialization import seed_distances
    from heracles.embedding.objective import quartet_objective

    @register_process
    @dataclass
    class ToyProcess(EditProcess):
        name: ClassVar[str] = "toy"
        supported_missing: ClassVar[tuple] = (SiteDropout,)

        def seed_distances(self, states, *, dtype=np.float64):
            return seed_distances(states, dtype=dtype)       # weighted Hamming

        def loss_function(self, states, allele_prior, *, missing, seed_dists,
                          batch_size, root_weight, learn_allele_prior, config, device):
            def score(st, depths, branches, leaves, mutation_rate, heritable_eps,
                      stochastic_delta):
                is_bal, idx_bal, c1, d1 = branches[0], branches[1], branches[4], branches[5]
                cherry, v1, v2 = branches[10][:, :2], branches[15], branches[16]
                pair = torch.where(is_bal[:, None], idx_bal[:, :2], cherry)  # (B, 2)
                t = torch.where(is_bal, c1 + d1, v1 + v2)                    # (B,)
                a, b = st[pair[:, 0]] > 0, st[pair[:, 1]] > 0                # (B, S)
                p_same = torch.exp(-mutation_rate * t)[:, None]
                logp_q = torch.where(a == b, p_same.log(), (1 - p_same).clamp_min(1e-12).log())
                p0 = torch.exp(-mutation_rate * depths[leaves])[:, None]
                logp_root = torch.where(st[leaves] > 0, (1 - p0).clamp_min(1e-12).log(), p0.log())
                return logp_q, logp_root                                     # (B, S), (L, S)

            return quartet_objective(states, states.shape[0], score,
                                     est_dists=seed_dists, batch_size=batch_size,
                                     root_weight=root_weight, config=config, device=device)

    Heracles(outdir=..., edit_model=CustomModel(ToyProcess(), SiteDropout()))

(A real process also handles missing entries, ``st == -1``, and uses the
dropout rates; ``Cas9Process`` in :mod:`heracles.models.cas9` is the reference.)

The score function
------------------
``quartet_objective`` calls ``score(st, depths, branches, leaves, mutation_rate,
heritable_eps, stochastic_delta)`` once per step:

* ``st``: ``(m, S)`` long tensor of states, founder at row 0, ``-1`` = missing.
* ``depths``: ``(m,)`` distance of every point from the founder.
* ``branches``: the rooted shape and branch lengths of each of the ``B``
  quartets in the batch, the 18-entry tuple documented in
  :func:`heracles.embedding.objective.quartet_branch_lengths`.  Index tensors in
  it (``branches[1]``, ``[9]``, ``[10]``) are rows of ``st``.
* ``leaves``: ``(L,)`` rows of ``st`` for the founder-to-leaf terms.
* the three rates: current values (tensors); ignore the ones you do not use.

It returns ``(logp_quartets, logp_root)``: log probabilities whose first
dimension is ``B`` and ``L``, usually per site, ``(B, S)`` and ``(L, S)``.  The
loss is minus their weighted means, so any further dimension is averaged.

The loss function
-----------------
Bypassing ``quartet_objective``, ``loss_function`` may return any callable the
optimiser calls once per step::

    loss_fn(points, manifold, radius, mutation_rate, heritable_eps,
            stochastic_delta, rng, eval_batch=None) -> (loss, numel)

* ``points``: founder-augmented coordinates on the Lorentz model (row 0 = founder).
* ``rng``: ``torch.Generator`` for sampling; ``eval_batch``: a fixed batch for
  the convergence check, or ``None``.
* Returns ``loss``, a scalar tensor to minimise, and ``numel``, the number of
  likelihood terms it averages.

Optional attributes on ``loss_fn`` that the optimiser uses (``quartet_objective``
sets the first three):

* ``make_eval_batch(generator)``: builds the fixed convergence-check batch.
* ``batch_gromov_penalty``, ``batch_depths``: reused by the tree-likeness and
  ultrametric penalties instead of drawing a separate batch.
* ``allele_prior_param``: set by :func:`heracles.embedding.ctmc.with_learnable_prior`.
* ``initiation_ratio_param``: any further learnable parameter.
* ``meta``: dict merged into the fit metadata.

The tree likelihood (optional; enables the polish)
--------------------------------------------------
``tree_likelihood`` returns an object holding the topology-independent parts
of a Felsenstein pruning likelihood for one character matrix.  The polish,
:func:`heracles.polish.ancestral.optimize_branch_lengths` and ``loglik`` use only:

* ``labels``: leaf names, in the row order of ``emissions``.
* ``prior_mass``: array of shape ``(S, A)``; only its shape is read (``A`` =
  the largest number of states per site).
* ``emissions``: ``(n, S, A)``, the probability of each leaf's observed data
  given each true state.
* ``edge_P(t)``: ``(S, A, A)`` transition probabilities along a branch of
  length ``t``, row = parent state.
* ``root_prior(root_edge_length)``: ``(S, A)`` distribution of the root state.
* ``total(per_site)``: whole-tree log-likelihood from ``(S,)`` per-site values
  (usually their sum).
* ``loglik(tree, root_edge_length=0.0)``: the whole-tree score, for users;
  :meth:`heracles.polish.ancestral.TreeLikelihood.loglik` builds it from the above
  with :func:`~heracles.polish.ancestral.index_tree` and
  :func:`~heracles.polish.ancestral.pruning_passes`.

If you implement both likelihoods they must describe the same model: score a
small tree with ``tree_likelihood`` and the same quartet with the loss, and
compare (as ``tests/test_models.py`` does for ``Cas9Process``).
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


__all__ = [
    "EditProcess",
    "MissingData",
    "EditModel",
    "CustomModel",
    "MODELS",
    "PROCESSES",
    "MISSING_DATA",
    "register_model",
    "register_process",
    "register_missing",
    "get_model",
    "model_option",
    "unregistered_parts",
]


def model_option(default, help, *, parse=None, type=None):
    """Dataclass field for a model option, with its command-line *help* and *parse* function.

    ``type=bool`` makes the flag a ``--x / --no-x`` switch.
    """
    return dataclasses.field(default=default, metadata={"help": help, "parse": parse,
                                                        "type": type})


def _fields_dict(obj):
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


# Missing data
class MissingData(ABC):
    """Which sites go missing together.

    Missing entries come from heritable deletions (rate ``heritable_eps`` along
    branches) and per-cell dropout (probability ``stochastic_delta``).  A
    missing-data model says which sites one such event blanks.  With
    ``separate_rates`` one dropout probability is learned per unit instead of a
    single shared one.
    """

    name: ClassVar[str] = ""
    separate_rates: bool = False

    @abstractmethod
    def units(self, n_sites):
        """``(S,)`` unit index per site, or ``None`` when every site is its own unit."""

    def validate(self, n_sites, config):
        """Raise ``ValueError`` if this model cannot apply; ``n_sites=None`` checks options only."""
        if self.separate_rates and not config.learn_stochastic_delta:
            raise ValueError("separate_dropout_rates requires learn_stochastic_delta=True "
                             "(a fixed per-unit vector is not supported)")

    def to_dict(self):
        return {"name": self.name, **_fields_dict(self)}


# Edit process
class EditProcess(ABC):
    """How sites get edited along the tree; owns the likelihoods.

    Class attributes a subclass sets:

    * ``name``: registry key.
    * ``supported_missing``: tuple of :class:`MissingData` classes it can score.
    * ``supports_low_memory``: True if the loss works without the ``(m, m)``
      seed matrix; it then receives ``seed_dists=None``.
    """

    name: ClassVar[str] = ""
    supported_missing: ClassVar[tuple] = ()
    supports_low_memory: ClassVar[bool] = False

    @abstractmethod
    def seed_distances(self, states, *, dtype=np.float64):
        """Rough ``(m, m)`` distances between the rows of *states*.

        *states* is the founder-augmented ``(m, S)`` integer matrix (row 0 is
        the unedited founder, ``-1`` marks missing).  The distances seed the
        initialisation and aim the quartet sampler, so they only need to rank
        relatives sensibly.
        """

    @abstractmethod
    def loss_function(self, states, allele_prior, *, missing, seed_dists, batch_size,
                      root_weight, learn_allele_prior, config, device):
        """Build the per-step loss callable (contract in the module docstring).

        Args:
            allele_prior: array from :func:`heracles.embedding.priors.build_allele_priors`.
            seed_dists: ``(m, m)`` output of :meth:`seed_distances`, or ``None``
                in low-memory mode.
            batch_size: quartets per step; ``None`` uses the default.
            root_weight: weight of the root-to-leaf term; ``None`` uses the
                default (set explicitly when gradient accumulation splits a batch).
        """

    def tree_likelihood(self, char_matrix, *, missing, mutation_rate, heritable_eps,
                        stochastic_delta, allele_prior, missing_state=-1):
        """Optional: a :class:`heracles.polish.ancestral.TreeLikelihood` for *char_matrix*.

        Scores under this process, *missing* and the given rates; implementing it
        enables the likelihood NNI polish.  The base class raises ``NotImplementedError``.
        """
        raise NotImplementedError("%s has no tree likelihood, so the likelihood NNI "
                                  "polish is not available" % type(self).__name__)

    @property
    def supports_polish(self):
        return type(self).tree_likelihood is not EditProcess.tree_likelihood

    def validate(self, n_sites, config, missing):
        """Raise ``ValueError`` if this process cannot fit the matrix or score *missing* (``n_sites=None``: options only)."""
        if not isinstance(missing, self.supported_missing):
            raise ValueError("%s cannot score %s (supported: %s)"
                             % (type(self).__name__, type(missing).__name__,
                                ", ".join(c.__name__ for c in self.supported_missing)))

    def to_dict(self):
        return {"name": self.name, **_fields_dict(self)}


# Edit model = process + missing data
class EditModel(ABC):
    """An edit process plus a missing-data model; what ``Heracles(edit_model=...)`` takes.

    Presets (``Cas9Model`` and others) are dataclasses whose fields are their
    options; :class:`CustomModel` combines any process with a supported
    missing-data model.  Subclasses implement :meth:`components`.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def components(self):
        """``(EditProcess, MissingData)``."""

    @property
    def process(self):
        return self.components()[0]

    @property
    def missing(self):
        return self.components()[1]

    @property
    def supports_low_memory(self):
        return self.process.supports_low_memory

    @property
    def supports_polish(self):
        return self.process.supports_polish

    def validate(self, n_sites, config):
        process, missing = self.components()
        process.validate(n_sites, config, missing)
        missing.validate(n_sites, config)

    def seed_distances(self, states, *, dtype=np.float64):
        return self.process.seed_distances(states, dtype=dtype)

    def loss_function(self, states, allele_prior, **kwargs):
        process, missing = self.components()
        return process.loss_function(states, allele_prior, missing=missing, **kwargs)

    def tree_likelihood(self, char_matrix, **rates):
        process, missing = self.components()
        return process.tree_likelihood(char_matrix, missing=missing, **rates)

    def tree_loglik(self, tree, char_matrix, *, root_edge_length=0.0, **rates):
        """Whole-tree log-likelihood of *char_matrix* on *tree*.

        Rate keywords are those of :meth:`EditProcess.tree_likelihood`.
        """
        return self.tree_likelihood(char_matrix, **rates).loglik(
            tree, root_edge_length=root_edge_length)

    def missing_units(self, n_sites):
        return self.missing.units(n_sites)

    @property
    def separate_dropout(self):
        return bool(self.missing.separate_rates)

    def describe(self):
        """One line for logs and ``repr``."""
        opts = ", ".join("%s=%r" % kv for kv in _fields_dict(self).items())
        return "%s(%s)" % (type(self).__name__, opts)

    def to_dict(self):
        """JSON-serialisable ``{"name": ..., <option>: <value>, ...}``."""
        return {"name": self.name, **_fields_dict(self)}

    @staticmethod
    def from_dict(d):
        """Inverse of :meth:`to_dict` for a registered model."""
        d = dict(d)
        name = d.pop("name")
        if name == CustomModel.name:
            return CustomModel(_from_registry(PROCESSES, d["process"], "edit process"),
                               _from_registry(MISSING_DATA, d["missing"], "missing-data model"))
        if name not in MODELS:
            raise ValueError("unknown edit model %r (registered: %s); import the module "
                             "that defines it before loading" % (name, ", ".join(MODELS)))
        return MODELS[name](**d)


@dataclasses.dataclass
class CustomModel(EditModel):
    """Any edit process with any missing-data model it supports (default ``SiteDropout()``)."""

    name: ClassVar[str] = "custom"
    process_: EditProcess = None
    missing_: MissingData = None

    def __init__(self, process, missing=None):
        from heracles.models.missing import SiteDropout
        self.process_ = process
        self.missing_ = missing if missing is not None else SiteDropout()

    def components(self):
        return self.process_, self.missing_

    def describe(self):
        return "CustomModel(%r, %r)" % (self.process_, self.missing_)

    def to_dict(self):
        return {"name": self.name, "process": self.process_.to_dict(),
                "missing": self.missing_.to_dict()}


def _from_registry(registry, d, what):
    d = dict(d)
    name = d.pop("name")
    if name not in registry:
        raise ValueError("unknown %s %r (registered: %s); import the module that defines "
                         "it before loading" % (what, name, ", ".join(registry)))
    return registry[name](**d)


# Registries
#: preset edit models by name (``Heracles(edit_model=name)``, ``--edit-model``)
MODELS = {}
#: edit processes by name
PROCESSES = {}
#: missing-data models by name
MISSING_DATA = {}


def _qualified(cls):
    return "%s.%s" % (cls.__module__, cls.__qualname__)


def _register(registry, cls):
    if not cls.name:
        raise ValueError("%s needs a non-empty class attribute 'name'" % cls.__name__)
    if not dataclasses.is_dataclass(cls):
        raise TypeError("%s must be a dataclass (its fields are its options)" % cls.__name__)
    old = registry.get(cls.name)
    # re-running the same definition (a notebook cell, a module reload) replaces it
    if old is not None and _qualified(old) != _qualified(cls):
        raise ValueError("the name %r is already registered to %s; give %s another name"
                         % (cls.name, _qualified(old), cls.__qualname__))
    registry[cls.name] = cls
    return cls


def unregistered_parts(model):
    """Names of the parts of *model* that ``Heracles.load`` could not rebuild.

    A preset must be in :data:`MODELS`; a :class:`CustomModel`'s process and
    missing-data model must be in :data:`PROCESSES` and :data:`MISSING_DATA`.
    """
    if isinstance(model, CustomModel):
        parts = ((PROCESSES, model.process_, "@register_process"),
                 (MISSING_DATA, model.missing_, "@register_missing"))
    else:
        parts = ((MODELS, model, "@register_model"),)
    return ["%s (decorate it with %s)" % (type(obj).__name__, deco)
            for registry, obj, deco in parts if registry.get(obj.name) is not type(obj)]


def register_model(cls):
    """Class decorator: make a preset :class:`EditModel` available by name."""
    return _register(MODELS, cls)


def register_process(cls):
    """Class decorator: make an :class:`EditProcess` loadable by name."""
    return _register(PROCESSES, cls)


def register_missing(cls):
    """Class decorator: make a :class:`MissingData` loadable by name."""
    return _register(MISSING_DATA, cls)


def get_model(spec):
    """Return an :class:`EditModel` from any accepted spec.

    *spec* is an :class:`EditModel` (returned as is), a preset name, a
    :meth:`EditModel.to_dict` dict, or ``None`` for ``Cas9Model()``.  Raises
    ``ValueError`` for an unknown preset name and ``TypeError`` for other types.
    """
    if spec is None:
        spec = "cas9"
    if isinstance(spec, EditModel):
        return spec
    if isinstance(spec, str):
        if spec not in MODELS:
            raise ValueError("unknown edit model %r (registered: %s)" % (spec, ", ".join(MODELS)))
        return MODELS[spec]()
    if isinstance(spec, dict):
        return EditModel.from_dict(spec)
    raise TypeError("edit_model must be an EditModel, a name or a dict, got %r" % (spec,))
