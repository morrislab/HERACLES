"""HERACLES: lineage trees from CRISPR lineage-tracing data via hyperbolic embedding.

The entry point is the :class:`Heracles` estimator::

    from heracles import Heracles, PETracerModel

    model = Heracles(outdir="runs/example")            # published defaults
    model = Heracles(outdir="runs/example", readouts=["nj", "upgma"],
                     edit_model=PETracerModel(cassette_size=3), dimension=64)
    model.fit(char_matrix)  # writes model.npz, tree_*.nwk, config.json, heracles.log
    model.trees_            # {"nj": Tree, "upgma": Tree}
    model.leaf_dists_       # (n, n) numpy array

    from heracles import robinson_foulds, sokal_collapse
    truth = sokal_collapse(open("true_tree.nwk").read(), char_matrix)
    robinson_foulds(model.trees_["nj_collapsed"], truth, normalized=True)

Subpackages: ``heracles.models`` (edit and missing-data models),
``heracles.embedding`` (the objective, sampler and optimiser),
``heracles.trees`` (Newick, read-outs, Camin-Sokal collapse, Robinson-Foulds)
and ``heracles.polish`` (ancestral states, branch lengths, likelihood NNI).
``heracles.fit.embed`` runs the fit without the estimator.  Submodules load
on first use, so ``import heracles`` is fast.
"""

__version__ = "0.1.0"

# Silent unless the caller configures handlers; Heracles.fit() attaches file and
# stderr handlers for the duration of the fit.
import logging as _logging
_logging.getLogger("heracles").addHandler(_logging.NullHandler())

__all__ = [
    # estimator and its options
    "Heracles",
    "HeraclesConfig",
    # edit models
    "Cas9Model",
    "PETracerModel",
    "SequentialModel",
    # comparing trees
    "robinson_foulds",
    "sokal_collapse",
]

# public name -> module it lives in
_EXPORTS = {
    "Heracles": "heracles.fit",
    "HeraclesConfig": "heracles.config",
    "Cas9Model": "heracles.models",
    "PETracerModel": "heracles.models",
    "SequentialModel": "heracles.models",
    "robinson_foulds": "heracles.trees.utils",
    "sokal_collapse": "heracles.trees.collapse",
}


def __getattr__(name):
    """Imports a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    import importlib
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(__all__ + ["__version__"])
