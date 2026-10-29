"""Tests for heracles CLI (heracles/__main__.py)."""

import json
import os

import numpy as np
import pandas as pd
import pytest


def _small_cm_path(tmp_path, n=10, sites=8, seed=42):
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 4, size=(n, sites))
    vals[rng.random((n, sites)) < 0.1] = -1
    labels = ["cell_%d" % i for i in range(n)]
    df = pd.DataFrame(vals, index=labels,
                      columns=["s%d" % j for j in range(sites)])
    p = tmp_path / "cm.csv"
    df.to_csv(p)
    return str(p)


@pytest.mark.slow
def test_fit_subcommand_runs(tmp_path):
    from heracles.__main__ import main

    cm = _small_cm_path(tmp_path)
    out = str(tmp_path / "out")
    rc = main(["fit", cm, "--outdir", out, "--allele-prior", "global_uniform",
               "--max-iterations", "10", "--dimension", "5",
               "--init-method", "mds", "--convergence-thresh", "none",
               "--readouts", "nj", "upgma"])
    assert rc == 0

    assert os.path.isfile(os.path.join(out, "model.npz"))
    assert os.path.isfile(os.path.join(out, "tree_nj.nwk"))
    assert os.path.isfile(os.path.join(out, "tree_upgma.nwk"))
    assert os.path.isfile(os.path.join(out, "config.json"))
    assert os.path.isfile(os.path.join(out, "report.html"))


@pytest.mark.slow
def test_fit_eps_auto(tmp_path):
    from heracles.__main__ import main

    cm = _small_cm_path(tmp_path)
    out = str(tmp_path / "out")
    rc = main(["fit", cm, "--outdir", out, "--allele-prior", "global_uniform",
               "--max-iterations", "10", "--dimension", "5",
               "--init-method", "mds", "--convergence-thresh", "none",
               "--eps", "auto", "--delta", "auto"])
    assert rc == 0

    cfg = json.loads(open(os.path.join(out, "config.json")).read())
    assert cfg["params"]["learn_heritable_eps"] is True
    assert cfg["params"]["learn_stochastic_delta"] is True


def test_inspect_subcommand(tmp_path):
    from heracles.__main__ import main
    from heracles import Heracles

    out = tmp_path / "run"
    cm_path = _small_cm_path(tmp_path)
    cm = pd.read_csv(cm_path, index_col=0)

    m = Heracles(allele_prior="global_uniform", outdir=str(out),
                 max_iterations=10, dimension=5, init_method="mds",
                 convergence_thresh=None)
    m.fit(cm)

    rc = main(["inspect", str(out / "model.npz")])
    assert rc == 0


def test_help_exits_cleanly():
    from heracles.__main__ import main

    with pytest.raises(SystemExit) as exc_info:
        main(["fit", "--help"])
    assert exc_info.value.code == 0


@pytest.mark.slow
def test_cli_polish(tmp_path):
    from conftest import _toy_states
    from heracles.__main__ import main
    states = _toy_states(n=10, S=6, K=3, seed=1)
    cm = pd.DataFrame(states[1:], index=["c%d" % i for i in range(10)],
                      columns=["s%d" % j for j in range(6)])
    csv = tmp_path / "cm.csv"
    cm.to_csv(csv)
    out = tmp_path / "run"
    assert main(["fit", str(csv), "--outdir", str(out), "--dimension", "4",
                 "--max-iterations", "30", "--init-method", "mds", "--no-report"]) == 0
    assert main(["polish", str(csv), "--model", str(out / "model.npz"),
                 "--max-rounds", "1"]) == 0
    assert (out / "tree_nj_polished.nwk").exists()


def test_cli_input_error_is_one_line(tmp_path, capsys):
    from heracles.__main__ import main
    csv = tmp_path / "cm.csv"
    pd.DataFrame([[1, -9], [0, 2]], index=["a", "b"]).to_csv(csv)
    rc = main(["fit", str(csv), "--outdir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "heracles fit: error:" in err and "missing code" in err
    assert "Traceback" not in err
