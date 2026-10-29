"""Edit models: how a lineage recorder writes and loses information.

An edit model is an edit process (how sites get edited) plus a missing-data
model (which sites go missing together).  Presets:

=========================  ===================================================
:class:`Cas9Model`         independent sites, each edited at most once (default)
:class:`PETracerModel`     Cas9 edits; cassettes of sites lost as a unit
:class:`SequentialModel`   sites inside a cassette edited in a fixed order
=========================  ===================================================

Pick one with ``Heracles(edit_model=PETracerModel(cassette_size=3))`` or by name
(``"petracer"``).  For your own edit process, see :mod:`heracles.models.base`.
"""

import dataclasses

from heracles.models.base import (MISSING_DATA, MODELS, PROCESSES, CustomModel, EditModel,
                                  EditProcess, MissingData, get_model, model_option,
                                  register_missing, register_model, register_process,
                                  unregistered_parts)
from heracles.models.missing import CassetteDropout, SiteDropout
from heracles.models.cas9 import Cas9Model, Cas9Process
from heracles.models.petracer import PETracerModel
from heracles.models.sequential import SequentialModel, SequentialProcess


__all__ = [
    # presets
    "Cas9Model",
    "PETracerModel",
    "SequentialModel",
    # building blocks
    "EditModel",
    "EditProcess",
    "MissingData",
    "CustomModel",
    "Cas9Process",
    "SequentialProcess",
    "SiteDropout",
    "CassetteDropout",
    # registries
    "MODELS",
    "PROCESSES",
    "MISSING_DATA",
    "register_model",
    "register_process",
    "unregistered_parts",
    "register_missing",
    "get_model",
    "model_option",
    "add_model_arguments",
    "model_from_args",
    "model_from_legacy_params",
]


def _model_fields():
    """``{option name: (field, [model names])}`` over every registered model."""
    out = {}
    for name, cls in MODELS.items():
        for f in dataclasses.fields(cls):
            if "help" not in f.metadata:
                continue
            if f.name in out:
                out[f.name][1].append(name)
            else:
                out[f.name] = (f, [name])
    return out


def add_model_arguments(parser):
    """Add ``--edit-model`` and one flag per model option to *parser*; returns *parser*."""
    import argparse
    g = parser.add_argument_group("Edit model")
    g.add_argument("--edit-model", dest="edit_model", default=None, choices=sorted(MODELS),
                   help="how the recorder edits and loses sites (default: cas9)")
    for opt, (f, owners) in _model_fields().items():
        flag = "--" + opt.replace("_", "-")
        help_ = "%s [%s]" % (f.metadata.get("help", ""), ", ".join(owners))
        if f.metadata.get("type") is bool:
            g.add_argument(flag, dest="model_" + opt, default=None,
                           action=argparse.BooleanOptionalAction, help=help_)
        else:
            g.add_argument(flag, dest="model_" + opt, default=None, metavar=opt.upper(),
                           help=help_)
    return parser


def model_from_args(args):
    """Build the :class:`EditModel` from :func:`add_model_arguments` flags (``ValueError`` on another model's flag)."""
    name = getattr(args, "edit_model", None) or "cas9"
    cls = MODELS[name]
    own = {f.name: f for f in dataclasses.fields(cls)}
    kw = {}
    for opt, (f, owners) in _model_fields().items():
        v = getattr(args, "model_" + opt, None)
        if v is None:
            continue
        if opt not in own:
            raise ValueError("--%s is an option of %s, not of the %s model"
                             % (opt.replace("_", "-"), " / ".join(owners), name))
        parse = own[opt].metadata.get("parse")
        kw[opt] = parse(v) if parse is not None else v
    return cls(**kw)


def model_from_legacy_params(params):
    """Rebuild the edit model from saved parameters.

    Uses the ``edit_model`` entry when present.  Parameter files without one
    are mapped to a preset from their ``likelihood``, ``dropout_unit``,
    ``cassette_size`` and ``stochastic_delta_per_site`` keys.
    """
    params = params or {}
    if "edit_model" in params and params["edit_model"]:
        return get_model(params["edit_model"])
    separate = bool(params.get("stochastic_delta_per_site", False))
    if params.get("likelihood") == "sequential":
        return SequentialModel(cassette_size=params.get("cassette_size"),
                               initiation_ratio=params.get("initiation_ratio", 1.0),
                               learn_initiation_ratio=bool(params.get("learn_initiation_ratio",
                                                                      False)))
    if params.get("dropout_unit") == "cassette":
        return PETracerModel(cassette_size=params.get("cassette_size") or 3,
                             separate_dropout_rates=separate)
    return Cas9Model(separate_dropout_rates=separate)
