"""Wall-time harness: guard refactors against slowing the fit or the polish down.

Runs fixed-length fits (no early stopping) of each shipped edit model on
seeded synthetic data, plus one polish round where the model supports it, and
records per-step optimiser time, initialisation time and polish time.
Timings are machine-specific: capture a baseline and the comparison on the
same machine, with nothing else heavy running.

Usage:
    python dev/check_speed.py run --out before.json [--reps 3]
    # ... make changes ...
    python dev/check_speed.py run --out after.json
    python dev/check_speed.py compare before.json after.json [--tol 0.10]
"""

import argparse
import json
import logging
import os
import platform
import statistics
import subprocess
import tempfile
import time
import warnings

import numpy as np
import pandas as pd


STEPS = 300


def _cas9_cm(n, S, K, missing_frac, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, K + 1, size=(n, S))
    x[rng.random(x.shape) < 0.35] = 0
    x[rng.random(x.shape) < missing_frac] = -1
    return pd.DataFrame(x, index=["c%d" % i for i in range(n)])


def _cassette_cm(n, n_cassettes, size, K, missing_frac, seed, ordered):
    """Cassette-structured matrix; missing data blanks whole cassettes.
    ``ordered`` fills each cassette left to right (sequential recorders)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, n_cassettes * size), dtype=np.int64)
    for c in range(n_cassettes):
        block = slice(c * size, (c + 1) * size)
        if ordered:
            n_edits = rng.integers(0, size + 1, size=n)
            for i in range(n):
                x[i, c * size:c * size + n_edits[i]] = rng.integers(1, K + 1, size=n_edits[i])
        else:
            vals = rng.integers(0, K + 1, size=(n, size))
            vals[rng.random(vals.shape) < 0.35] = 0
            x[:, block] = vals
        x[rng.random(n) < missing_frac, block] = -1
    return pd.DataFrame(x, index=["c%d" % i for i in range(n)])


def _cases():
    from heracles.models import Cas9Model, PETracerModel, SequentialModel
    return [
        ("cas9_n300", Cas9Model(), _cas9_cm(300, 40, 8, 0.1, seed=1)),
        ("petracer_n300", PETracerModel(cassette_size=3),
         _cassette_cm(300, 12, 3, 8, 0.15, seed=2, ordered=False)),
        ("sequential_n150", SequentialModel(cassette_size=5),
         _cassette_cm(150, 8, 5, 6, 0.05, seed=3, ordered=True)),
    ]


def _git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.abspath(__file__))
                              ).stdout.strip()
    except Exception:
        return None


def run(reps):
    import torch
    from heracles import Heracles
    torch.set_num_threads(4)
    logging.getLogger("heracles").setLevel(logging.ERROR)
    warnings.simplefilter("ignore")
    out = {"_machine": {"platform": platform.platform(), "processor": platform.processor(),
                        "torch_threads": torch.get_num_threads(), "git": _git_rev(),
                        "steps": STEPS, "reps": reps}}
    for name, model, cm in _cases():
        rows = []
        for rep in range(reps):
            with tempfile.TemporaryDirectory() as d:
                m = Heracles(outdir=d, edit_model=model, max_iterations=STEPS,
                             min_iterations=0, convergence_thresh=None, seed=rep)
                t0 = time.perf_counter()
                m.fit(cm)
                fit_s = time.perf_counter() - t0
                diag = m.diagnostics_
                row = {"fit_s": fit_s, "init_s": diag["init_time"],
                       "optimizer_s": diag["optimizer_time"],
                       "ms_per_step": 1000.0 * diag["optimizer_time"] / m.n_iter_}
                if m.edit_model.supports_polish:
                    t0 = time.perf_counter()
                    m.polish(cm, max_rounds=1, write=False)
                    row["polish_s"] = time.perf_counter() - t0
            rows.append(row)
            print("%-16s rep %d  %s" % (name, rep, "  ".join(
                "%s=%.3f" % (k, v) for k, v in row.items())), flush=True)
        out[name] = {"reps": rows, "median": {k: statistics.median(r[k] for r in rows)
                                              for k in rows[0]}}
    return out


def compare(a, b, tol):
    worst, lines = 0.0, []
    for case in sorted(k for k in a if not k.startswith("_")):
        if case not in b:
            lines.append("%s: missing from the new run" % case)
            continue
        for metric, va in a[case]["median"].items():
            vb = b[case]["median"].get(metric)
            if vb is None:
                continue
            ratio = vb / va if va else float("nan")
            flag = "  SLOWER" if ratio > 1 + tol else ""
            if metric in ("ms_per_step", "polish_s"):
                worst = max(worst, ratio)
            lines.append("%-16s %-12s %9.3f -> %9.3f  x%.3f%s"
                         % (case, metric, va, vb, ratio, flag))
    return worst, lines


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--out", required=True)
    pr.add_argument("--reps", type=int, default=3)
    pc = sub.add_parser("compare")
    pc.add_argument("before")
    pc.add_argument("after")
    pc.add_argument("--tol", type=float, default=0.10)
    args = p.parse_args()
    if args.cmd == "run":
        res = run(args.reps)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", args.out)
    else:
        with open(args.before) as f:
            a = json.load(f)
        with open(args.after) as f:
            b = json.load(f)
        worst, lines = compare(a, b, args.tol)
        print("\n".join(lines))
        print("worst ms_per_step / polish ratio: x%.3f (tolerance x%.2f)" % (worst, 1 + args.tol))
        print("PASS" if worst <= 1 + args.tol else "SLOWER")


if __name__ == "__main__":
    main()
