"""Command-line interface: ``heracles fit``, ``heracles polish``, ``heracles inspect``.

    heracles fit data.csv --outdir runs/exp1
    heracles fit data.csv --outdir runs/exp1 --edit-model petracer --cassette-size 3
    heracles polish data.csv --model runs/exp1/model.npz
    heracles inspect runs/exp1/model.npz

Every ``HeraclesConfig`` option is a flag (``heracles fit --help``); unset flags
keep the published defaults.  ``fit`` also writes ``report.html`` unless ``--no-report``.
"""

import argparse
import json
import os
import sys

from heracles.config import add_config_arguments, config_from_args
from heracles.models import add_model_arguments, model_from_args


def _add_fit_args(parser):
    parser.add_argument("char_matrix",
                        help="character matrix CSV (rows = cells, columns = sites)")
    parser.add_argument("--outdir", required=True,
                        help="output directory (parent must exist)")
    parser.add_argument("--readouts", nargs="*", default=["nj"],
                        help="trees to build: nj (default), upgma")
    parser.add_argument("--init-tree", "--init-tree-nwk", dest="init_tree", default=None,
                        help="Newick file of a starting tree; its branch-length distances "
                             "seed the initialisation")
    parser.add_argument("--no-report", action="store_true",
                        help="skip the HTML report")
    add_model_arguments(parser)
    add_config_arguments(parser)


def _cmd_fit(args):
    from heracles import Heracles

    model = Heracles(outdir=args.outdir, readouts=args.readouts, init_tree=args.init_tree,
                     config=config_from_args(args), edit_model=model_from_args(args))
    model.fit(args.char_matrix)
    if not args.no_report:
        model.write_report()

    print(json.dumps({
        "outdir": model.outdir,
        "converged": model.converged_,
        "n_iter": model.n_iter_,
        "n_cells": len(model.labels_),
        "readouts": list(model.trees_.keys()) if model.trees_ else [],
    }, indent=2))
    return 0


def _add_polish_args(parser):
    parser.add_argument("char_matrix",
                        help="the character matrix CSV the model was fitted on")
    parser.add_argument("--model", required=True, dest="model_path",
                        help="path to a fitted model.npz")
    parser.add_argument("--readout", default="nj",
                        help="which read-out tree to polish (default: nj)")
    parser.add_argument("--min-gain", type=float, default=None,
                        help="minimum log-likelihood gain per accepted NNI move "
                             "(default: the published 1.0)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="rounds over the candidate branches (default: 3)")
    parser.add_argument("--support-threshold", type=float, default=None,
                        help="only try branches whose quartet support is below this "
                             "(default: every branch)")
    parser.add_argument("--outdir", default=None,
                        help="where to write tree_<readout>_polished.nwk "
                             "(default: the model's directory)")


def _cmd_polish(args):
    from heracles import Heracles
    from heracles.polish.nni import POLISH_MAX_ROUNDS, POLISH_MIN_GAIN

    model = Heracles.load(args.model_path)
    if args.outdir:
        model.outdir = args.outdir
        os.makedirs(args.outdir, exist_ok=True)
    res = model.polish(
        args.char_matrix, readout=args.readout,
        min_gain=POLISH_MIN_GAIN if args.min_gain is None else args.min_gain,
        max_rounds=POLISH_MAX_ROUNDS if args.max_rounds is None else args.max_rounds,
        support_threshold=args.support_threshold, log=lambda s: print(s, file=sys.stderr))
    print(json.dumps({
        "tree": os.path.join(model.outdir, "tree_%s_polished.nwk" % args.readout),
        "moves": len(res.moves), "rounds": res.n_rounds,
        "candidates": res.n_candidates, "evaluated": res.n_evaluated,
        "loglik_start": res.loglik_start, "loglik_end": res.loglik_end,
        "wall_s": res.wall_s,
    }, indent=2))
    return 0


def _cmd_inspect(args):
    from heracles import Heracles

    model = Heracles.load(args.model_path)
    params = model.get_params()
    info = {"model_path": args.model_path,
            "n_cells": len(model.labels_) if model.labels_ else None,
            "edit_model": model.edit_model.to_dict(),
            "config": model.config.to_dict(),
            "readouts": params.get("readouts")}
    fitted = params.get("fitted") or {}
    info["fitted"] = {k: v for k, v in fitted.items()
                      if v is not None and not isinstance(v, (dict, list))}
    print(json.dumps(info, indent=2, default=str))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="heracles",
        description="HERACLES: hyperbolic embedding for CRISPR lineage-tracing trees")
    sub = parser.add_subparsers(dest="command")

    fit_p = sub.add_parser("fit", help="fit an embedding and build trees")
    _add_fit_args(fit_p)
    fit_p.set_defaults(func=_cmd_fit)

    polish_p = sub.add_parser("polish", help="likelihood NNI polish of a fitted model's tree")
    _add_polish_args(polish_p)
    polish_p.set_defaults(func=_cmd_polish)

    inspect_p = sub.add_parser("inspect", help="print a fitted model's settings and results")
    inspect_p.add_argument("model_path", help="path to model.npz")
    inspect_p.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        # input mistakes: one line, no traceback (set HERACLES_DEBUG=1 to see it)
        if os.environ.get("HERACLES_DEBUG"):
            raise
        print("heracles %s: error: %s" % (args.command, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
