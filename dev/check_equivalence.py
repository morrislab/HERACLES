"""Numerical equivalence harness for optimization work.

Runs a battery of short same-seed fits and records the per-step loss series
(and learned eps/delta series when tracked). Compare a baseline capture
against a post-change capture to verify a refactor did not change the math.

Short fits on purpose: step 1's loss is computed at the initial
configuration (pure forward equivalence); steps 2-5 include optimizer
updates (gradient equivalence) while staying short enough that float
round-off differences cannot amplify through trajectory divergence.

Usage:
    python dev/check_equivalence.py run --out new.json
    python dev/check_equivalence.py compare dev/equivalence_baseline.json new.json [--rtol 1e-9]

Two cases exercise alternative objectives that live with the paper's
supplementary figures (``experiments/paper/figures``); they are skipped, and
reported as skipped by ``compare``, when that checkout is not present.
"""

import argparse
import json
import sys
import tempfile

import numpy as np
import pandas as pd


def _make_cm(n, S, K, missing_frac, seed):
    rng = np.random.default_rng(seed)
    cm = pd.DataFrame(rng.integers(0, K + 1, (n, S)),
                      index=[f"c{i}" for i in range(n)])
    if missing_frac:
        cm.iloc[rng.random(cm.shape) < missing_frac] = -1
    return cm


import importlib.util
import os

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "experiments", "paper", "figures")


def _load(path, name):
    """Import ``name`` from a supplementary-figure module by file path."""
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def _factory_model(factory):
    """A Cas9 edit model whose likelihood is a supplement's loss factory
    (the plug-in route for the alternative objectives)."""
    from dataclasses import dataclass
    from typing import ClassVar
    from heracles.models import Cas9Model

    @dataclass
    class _Supplement(Cas9Model):
        name: ClassVar[str] = "supplement:" + getattr(factory, "__name__", "factory")

        def loss_function(self, states, allele_prior, *, seed_dists, batch_size,
                          root_weight, learn_allele_prior, config, device):
            return factory(states, allele_prior, states.shape[0], est_dists=seed_dists,
                           batch_size=batch_size, learn_allele_prior=learn_allele_prior,
                           device=device, config=config)
    return _Supplement()


# The baseline was captured under the pre-config defaults; pin them so the
# comparison tests the maths, not the changed defaults.
_BASELINE_DEFAULTS = dict(
    allele_prior="global_uniform", learn_allele_prior=False,
    heritable_eps=1e-5, stochastic_delta=0.2,
    learn_heritable_eps=False, learn_stochastic_delta=False,
    ultrametric_penalty=0, ultrametric_target_cv="auto",
    min_iterations=0, band_kmax_end=None, band_local_frac=0.0,
    band_local_mult_max=None,
)

CONFIGS = {
    "quartet_banded_full": dict(
        missing_frac=0.15,
        kwargs=dict(learn_heritable_eps=True, learn_stochastic_delta=True,
                    learn_allele_prior=True,
                    ultrametric_penalty="adaptive"),
    ),
    "quartet_soa": dict(
        missing_frac=0.15,
        kwargs=dict(learn_heritable_eps=True, learn_stochastic_delta=True,
                    learn_allele_prior=True,
                    ultrametric_penalty="adaptive", min_iterations=0,
                    band_kmin=2, band_kmax=16, band_kmax_end=128,
                    band_local_frac=1.0, band_local_mult_max=4),
    ),
    # alternative objectives / samplers from the paper supplement, run
    # through the loss_factory hook (skipped without the experiments checkout)
    "quartet_uniform_plain": dict(
        missing_frac=0.0,
        kwargs=dict(loss_factory=("supp_sampler_ablation/samplers.py",
                                  "uniform_loss_factory")),
    ),
    "triple_penalty_missing": dict(
        missing_frac=0.15,
        kwargs=dict(loss_factory=("supp_objective_ladder/objectives.py",
                                  "triple_loss_factory")),
    ),
    # the RapidNJ initialisation path (tree -> clamped tree distances -> MDS);
    # also records the init tree itself
    "quartet_rapidnj_init": dict(
        missing_frac=0.15, init_method="rapidnj_clamped",
        kwargs=dict(learn_heritable_eps=True, learn_stochastic_delta=True,
                    learn_allele_prior=True, ultrametric_penalty="adaptive"),
    ),
}


def run_battery(n=120, S=30, K=10, steps=5, seeds=(0, 1)):
    from heracles.fit import Heracles
    out = {}
    for name, spec in CONFIGS.items():
        for seed in seeds:
            cm = _make_cm(n, S, K, spec["missing_frac"], seed=100 + seed)
            kwargs = {**_BASELINE_DEFAULTS, **spec["kwargs"]}
            if "loss_factory" in kwargs:
                rel, factory_name = kwargs["loss_factory"]
                path = os.path.join(_FIG, rel)
                if not os.path.exists(path):
                    print(f"skipping {name}: {path} not found (needs the experiments checkout)")
                    break
                try:
                    kwargs["edit_model"] = _factory_model(_load(path, factory_name))
                except Exception as e:          # the figure code lags the package
                    print(f"skipping {name}: cannot load {path}: {e}")
                    break
                del kwargs["loss_factory"]
            with tempfile.TemporaryDirectory() as d:
                m = Heracles(dimension=spec.get("dimension", 8),
                             max_iterations=steps, convergence_thresh=None,
                             seed=seed, init_method=spec.get("init_method", "mds"),
                             outdir=d, **kwargs)
                m.fit(cm)
            rec = {"loss": m.history_["loss"]}
            for k in ("heritable_eps", "stochastic_delta",
                      "treelike_weight", "ultrametric_weight"):
                if k in m.history_:
                    rec[k] = m.history_[k]
            if m.diagnostics_.get("init_tree_newick"):
                rec["init_tree_newick"] = m.diagnostics_["init_tree_newick"]
            out[f"{name}/seed{seed}"] = rec
    return out


def compare(a, b, rtol):
    worst = 0.0
    failures, skipped = [], []
    for key in sorted(set(a) | set(b)):
        if key not in b:
            skipped.append(key)
            continue
        if key not in a:
            failures.append(f"{key}: not in the baseline")
            continue
        for series in sorted(set(a[key]) | set(b[key])):
            if series not in a[key] or series not in b[key]:
                failures.append(f"{key}/{series}: present in only one capture")
                continue
            if isinstance(a[key][series], str) or isinstance(b[key][series], str):
                if a[key][series] != b[key][series]:
                    failures.append(f"{key}/{series}: strings differ")
                continue
            va, vb = np.array(a[key][series]), np.array(b[key][series])
            if va.shape != vb.shape:
                failures.append(f"{key}/{series}: length {va.shape} vs {vb.shape}")
                continue
            denom = np.maximum(np.abs(va), 1e-30)
            rel = float(np.max(np.abs(va - vb) / denom))
            worst = max(worst, rel)
            if rel > rtol:
                failures.append(f"{key}/{series}: max rel diff {rel:.3e} > {rtol:g}")
    return worst, failures, skipped


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--out", required=True)
    pc = sub.add_parser("compare")
    pc.add_argument("baseline")
    pc.add_argument("new")
    pc.add_argument("--rtol", type=float, default=1e-9)
    args = p.parse_args()

    if args.cmd == "run":
        res = run_battery()
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"captured {len(res)} runs -> {args.out}")
    else:
        with open(args.baseline) as f:
            a = json.load(f)
        with open(args.new) as f:
            b = json.load(f)
        worst, failures, skipped = compare(a, b, args.rtol)
        print(f"worst relative difference: {worst:.3e} (rtol {args.rtol:g})")
        for key in skipped:
            print(f"  skipped {key} (not captured in the new run)")
        if failures:
            print("FAIL")
            for line in failures:
                print(" ", line)
            sys.exit(1)
        print("PASS")


if __name__ == "__main__":
    main()
