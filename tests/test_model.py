"""Tests for heracles.fit.Heracles."""

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import _toy_states

from heracles.fit import Heracles
from heracles.trees.tree import Tree
from heracles.models import Cas9Model


# Fixtures

def _small_cm(n=10, sites=8, seed=42):
    """A tiny character matrix with some missing data."""
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 4, size=(n, sites))
    vals[rng.random((n, sites)) < 0.1] = -1
    labels = ["cell_%d" % i for i in range(n)]
    return pd.DataFrame(vals, index=labels,
                        columns=["s%d" % j for j in range(sites)])


@pytest.fixture
def small_cm():
    return _small_cm()


# Constructor validation

def test_bad_readout(tmp_path):
    with pytest.raises(ValueError, match="unknown readout"):
        Heracles(readouts=["nj", "centroid"], allele_prior="global_uniform",
                 outdir=tmp_path)


def test_outdir_parent_must_exist(tmp_path):
    with pytest.raises(ValueError, match="parent directory does not exist"):
        Heracles(allele_prior="global_uniform",
                 outdir=tmp_path / "missing" / "leaf")
    # only the last layer missing is fine: fit() creates it
    Heracles(allele_prior="global_uniform", outdir=tmp_path / "leaf")


def test_default_readouts(tmp_path):
    m = Heracles(allele_prior="global_uniform", outdir=tmp_path)
    assert m.readouts == ["nj"]


def test_repr_unfitted(tmp_path):
    m = Heracles(heritable_eps=0.0, stochastic_delta=0.2,
                 learn_heritable_eps=False, learn_stochastic_delta=False,
                 allele_prior="global_uniform", outdir=tmp_path)
    r = repr(m)
    assert r.startswith("Heracles(")
    assert "fitted" not in r


# Fit (small and fast)

@pytest.fixture
def fitted_model(small_cm, tmp_path):
    m = Heracles(readouts=["nj", "upgma"],
                 dimension=5, max_iterations=50, convergence_thresh=None,
                 heritable_eps=0.01, stochastic_delta=0.01, seed=0,
                 learn_heritable_eps=False, learn_stochastic_delta=False,
                 allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    m.fit(small_cm)
    return m


def test_fit_sets_attributes(fitted_model):
    m = fitted_model
    n = 10
    assert m.leaf_dists_.shape == (n, n)
    assert m.full_dists_.shape == (n + 1, n + 1)
    assert len(m.labels_) == n
    assert len(m.history_["loss"]) == 50
    assert m.n_iter_ == 50
    assert m.mutation_rate_ is not None
    assert m.heritable_eps_ is not None
    assert m.stochastic_delta_ is not None


def test_fit_trees(fitted_model):
    m = fitted_model
    assert set(m.trees_.keys()) == {"nj", "upgma", "nj_collapsed"}
    for name, tree in m.trees_.items():
        assert isinstance(tree, Tree), name
        leaves = {l.taxon.label for l in tree.leaf_node_iter()}
        assert leaves == set(m.labels_), name


def test_fit_returns_self(small_cm, tmp_path):
    m = Heracles(dimension=5, max_iterations=10, convergence_thresh=None,
                 allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    ret = m.fit(small_cm)
    assert ret is m


def test_newick(small_cm, tmp_path):
    m = Heracles(readouts=["nj", "upgma"], dimension=5,
                 max_iterations=10, convergence_thresh=None,
                 allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path).fit(small_cm)
    for r in ("nj", "upgma"):
        nwk = m.newick(r)
        assert isinstance(nwk, str) and nwk.endswith(";")
        assert (tmp_path / ("tree_%s.nwk" % r)).read_text().strip() == nwk
    with pytest.raises(KeyError, match="no 'rapidnj' tree"):
        m.newick("rapidnj")


def test_collapsed_tree(small_cm, tmp_path):
    """fit() adds the first read-out with edit-free edges contracted, saved and reloaded."""
    from heracles.trees.collapse import sokal_collapse
    from heracles.trees.utils import as_newick
    m = Heracles(readouts=["nj", "upgma"], dimension=5, max_iterations=10,
                 convergence_thresh=None, allele_prior="global_uniform",
                 init_method="mds", outdir=tmp_path).fit(small_cm)
    assert "upgma_collapsed" not in m.trees_            # only the first read-out
    collapsed = m.trees_["nj_collapsed"]
    cm = small_cm.copy()
    cm.index = cm.index.astype(str)
    assert as_newick(collapsed) == as_newick(sokal_collapse(m.trees_["nj"], cm))
    n_internal = lambda t: sum(1 for v in t.preorder_node_iter() if not v.is_leaf())
    assert n_internal(collapsed) <= n_internal(m.trees_["nj"])
    assert ({l.taxon.label for l in collapsed.leaf_node_iter()}
            == {l.taxon.label for l in m.trees_["nj"].leaf_node_iter()})
    nwk = m.newick("nj_collapsed")
    assert (tmp_path / "tree_nj_collapsed.nwk").read_text().strip() == nwk
    assert Heracles.load(tmp_path / "model.npz").newick("nj_collapsed") == nwk


def test_not_fitted_raises(tmp_path):
    m = Heracles(allele_prior="global_uniform", outdir=tmp_path)
    with pytest.raises(RuntimeError, match="not fitted"):
        m.save(tmp_path / "nope.npz")


# Coords

def test_coords_present(fitted_model):
    assert fitted_model.coords_ is not None
    assert fitted_model.coords_.shape[0] == 11  # n+1 (founder + 10 leaves)


# Snapshots

def test_snapshot_every(small_cm, tmp_path):
    m = Heracles(dimension=5, max_iterations=100, convergence_thresh=None,
                 snapshot_every=25, seed=0, allele_prior="global_uniform",
                 init_method="mds", outdir=tmp_path)
    m.fit(small_cm)
    assert m.coords_history_ is not None
    assert 0 in m.coords_history_
    assert 25 in m.coords_history_
    for epoch, snap in m.coords_history_.items():
        assert snap.shape == m.coords_.shape


def test_no_snapshots_by_default(fitted_model):
    assert fitted_model.coords_history_ is None


# Small fit

def test_small_fit_nj_only(small_cm, tmp_path):
    m = Heracles(readouts=["nj"], dimension=5,
                 max_iterations=30, convergence_thresh=None,
                 heritable_eps=0.0, stochastic_delta=0.2, seed=0,
                 learn_heritable_eps=False, learn_stochastic_delta=False,
                 allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    m.fit(small_cm)
    assert m.leaf_dists_ is not None
    assert "nj" in m.trees_


# get_params

def test_get_params_unfitted(tmp_path):
    m = Heracles(heritable_eps=0.1, stochastic_delta=0.3,
                 learn_heritable_eps=False, learn_stochastic_delta=False,
                 allele_prior="global_uniform", outdir=tmp_path)
    p = m.get_params()
    assert p["edit_model"] == {"name": "cas9", "separate_dropout_rates": False}
    assert p["heritable_eps"] == 0.1
    assert p["stochastic_delta"] == 0.3
    assert "__version__" in p
    assert "fitted" not in p


def test_get_params_fitted(fitted_model):
    p = fitted_model.get_params()
    assert "fitted" in p
    assert p["fitted"]["n_iter_"] == 50
    assert p["fitted"]["mutation_rate_"] is not None


# Save / load round-trip

def test_save_load_roundtrip(fitted_model):
    m = fitted_model
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.npz")
        m.save(path)
        loaded = Heracles.load(path)

    assert set(loaded.history_.keys()) == set(m.history_.keys())
    np.testing.assert_array_almost_equal(loaded.history_["loss"],
                                         m.history_["loss"])

    assert loaded.edit_model == m.edit_model == Cas9Model()
    assert loaded.readouts == m.readouts
    assert loaded.dimension == m.dimension
    assert loaded.max_iterations == m.max_iterations
    np.testing.assert_array_almost_equal(loaded.leaf_dists_, m.leaf_dists_)
    np.testing.assert_array_almost_equal(loaded.full_dists_, m.full_dists_)
    assert loaded.labels_ == m.labels_
    assert loaded.n_iter_ == m.n_iter_
    assert loaded.converged_ == m.converged_
    assert loaded.mutation_rate_ == m.mutation_rate_
    assert loaded.heritable_eps_ == m.heritable_eps_
    assert loaded.stochastic_delta_ == m.stochastic_delta_
    assert set(loaded.trees_.keys()) == set(m.trees_.keys())
    for r in m.readouts:
        orig_nwk = m.trees_[r].as_string(schema="newick").strip()
        load_nwk = loaded.trees_[r].as_string(schema="newick").strip()
        assert orig_nwk == load_nwk

    assert loaded.diagnostics_["init_time"] == m.diagnostics_["init_time"]
    assert loaded.diagnostics_["optimizer_time"] == m.diagnostics_["optimizer_time"]
    assert loaded.diagnostics_["readout_times"] == m.diagnostics_["readout_times"]
    assert loaded.min_iterations == m.min_iterations
    assert loaded.diagnostics_["final_penalty_weight"] == m.diagnostics_["final_penalty_weight"]


def test_save_load_with_snapshots(small_cm, tmp_path):
    m = Heracles(readouts=["nj"], dimension=5, max_iterations=50,
                 convergence_thresh=None, snapshot_every=25, seed=0,
                 allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    m.fit(small_cm)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.npz")
        m.save(path)
        loaded = Heracles.load(path)
    assert loaded.coords_history_ is not None
    assert set(loaded.coords_history_.keys()) == set(m.coords_history_.keys())
    for epoch in m.coords_history_:
        np.testing.assert_array_almost_equal(
            loaded.coords_history_[epoch], m.coords_history_[epoch])


def test_history_tracks_learned_params_and_penalties(small_cm, tmp_path):
    m = Heracles(dimension=5, max_iterations=30, convergence_thresh=None,
                 learn_heritable_eps=True, learn_stochastic_delta=True,
                 ultrametric_penalty="adaptive",
                 seed=0, allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    m.fit(small_cm)

    expected = {"loss", "heritable_eps", "stochastic_delta",
                "treelike_weight", "gromov_delta",
                "ultrametric_weight", "depth_cv"}
    assert expected <= set(m.history_.keys())
    for name in expected:
        assert len(m.history_[name]) == m.n_iter_, name

    path = os.path.join(tmp_path, "model.npz")
    m.save(path)
    loaded = Heracles.load(path)
    assert set(loaded.history_.keys()) == set(m.history_.keys())
    np.testing.assert_array_almost_equal(
        loaded.history_["heritable_eps"], m.history_["heritable_eps"])


def test_save_load_coords(fitted_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.npz")
        fitted_model.save(path)
        loaded = Heracles.load(path)
    np.testing.assert_array_almost_equal(loaded.coords_, fitted_model.coords_)


# Convergence warning

def test_convergence_warning(small_cm, tmp_path):
    with pytest.warns(UserWarning, match="still improving"):
        Heracles(dimension=5, max_iterations=10, convergence_thresh=1e-12, allele_prior="global_uniform",
                 seed=0, init_method="mds",
                 outdir=tmp_path).fit(small_cm)


# In-loop early stopping

def test_early_stopping_fires(small_cm, tmp_path):
    # checks every 100 steps, stops after 3 consecutive ones below the threshold
    m = Heracles(dimension=5, max_iterations=3000, min_iterations=0,
                 convergence_thresh=0.5, seed=0,
                 allele_prior="global_uniform", init_method="mds", outdir=tmp_path)
    m.fit(small_cm)
    assert m.converged_ is True
    assert m.n_iter_ < 3000


def test_early_stopping_disabled_runs_all_steps(small_cm, tmp_path):
    m = Heracles(dimension=5, max_iterations=50, convergence_thresh=None,
                 seed=0, allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    m.fit(small_cm)
    assert m.n_iter_ == 50
    assert m.converged_ is False


# repr after fit

def test_repr_fitted(fitted_model):
    r = repr(fitted_model)
    assert "fitted" in r
    assert "50 iter" in r


# Run artifacts (config.json + heracles.log)

def test_fit_writes_run_artifacts(small_cm, tmp_path):
    out = tmp_path / "run1"
    m = Heracles(dimension=5, max_iterations=10, convergence_thresh=None,
                 allele_prior="global_uniform", init_method="mds", seed=0,
                 outdir=out)
    m.fit(small_cm)

    # fit() writes exactly these; plots and the report come from write_report()
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json", "heracles.log", "model.npz", "tree_nj.nwk", "tree_nj_collapsed.nwk"]
    cfg = json.loads((out / "config.json").read_text())
    assert "git_commit" in cfg          # key present; value may be None
    assert "package_version" in cfg and "timestamp" in cfg
    assert cfg["params"]["outdir"] == str(out)
    assert cfg["fitted"]["n_iter_"] == 10 and "mutation_rate_" in cfg["fitted"]
    report = m.write_report()
    assert os.path.exists(report) and (out / "loss_curve.png").exists()

    assert (out / "heracles.log").read_text().strip()

    hlog = logging.getLogger("heracles")  # no leaked handlers after fit
    assert all(isinstance(h, logging.NullHandler) for h in hlog.handlers)


# auto eps/delta, ultrametric coercion, files written by fit

def test_eps_auto_implies_learning(tmp_path):
    m = Heracles(heritable_eps="auto", allele_prior="global_uniform",
                 outdir=tmp_path)
    assert m.heritable_eps == 1e-3
    assert m.learn_heritable_eps is True


def test_delta_auto_implies_learning(tmp_path):
    m = Heracles(stochastic_delta="auto", allele_prior="global_uniform",
                 outdir=tmp_path)
    assert m.stochastic_delta == 1e-3
    assert m.learn_stochastic_delta is True


def test_eps_bad_string_rejected(tmp_path):
    with pytest.raises(ValueError, match="heritable_eps"):
        Heracles(heritable_eps="high", allele_prior="global_uniform",
                 outdir=tmp_path)


def test_delta_per_site_fit_and_roundtrip(small_cm, tmp_path):
    m = Heracles(dimension=5, max_iterations=30, convergence_thresh=None,
                 heritable_eps=0.0, learn_heritable_eps=False,
                 stochastic_delta=0.05, learn_stochastic_delta=True,
                 edit_model=Cas9Model(separate_dropout_rates=True),
                 seed=0, allele_prior="global_uniform", init_method="mds",
                 outdir=tmp_path)
    m.fit(small_cm)
    S = small_cm.shape[1]
    assert isinstance(m.stochastic_delta_sites_, list) and len(m.stochastic_delta_sites_) == S
    assert all(0.0 < v < 1.0 for v in m.stochastic_delta_sites_)
    assert m.stochastic_delta_ == pytest.approx(float(np.mean(m.stochastic_delta_sites_)))
    assert len(m.history_["stochastic_delta"]) == m.n_iter_
    fitted = m.get_params()["fitted"]
    assert len(fitted["stochastic_delta_sites_"]) == S
    path = os.path.join(tmp_path, "model.npz")
    m.save(path)
    loaded = Heracles.load(path)
    assert loaded.edit_model == Cas9Model(separate_dropout_rates=True)
    assert loaded.stochastic_delta_sites_ == pytest.approx(m.stochastic_delta_sites_)


def test_delta_per_site_requires_learning(tmp_path):
    with pytest.raises(ValueError, match="separate_dropout_rates"):
        Heracles(stochastic_delta=0.1, learn_stochastic_delta=False,
                 edit_model=Cas9Model(separate_dropout_rates=True),
                 allele_prior="global_uniform", outdir=tmp_path)


def test_delta_bad_string_rejected(tmp_path):
    with pytest.raises(ValueError, match="stochastic_delta"):
        Heracles(stochastic_delta="low", allele_prior="global_uniform",
                 outdir=tmp_path)


def test_ultrametric_penalty_string_coercion(tmp_path):
    m1 = Heracles(ultrametric_penalty="0", allele_prior="global_uniform",
                  outdir=tmp_path)
    assert m1.ultrametric_penalty == 0.0
    assert isinstance(m1.ultrametric_penalty, float)

    m2 = Heracles(ultrametric_penalty="1.5", allele_prior="global_uniform",
                  outdir=tmp_path)
    assert m2.ultrametric_penalty == 1.5

    m3 = Heracles(ultrametric_penalty="adaptive",
                  allele_prior="global_uniform", outdir=tmp_path)
    assert m3.ultrametric_penalty == "adaptive"


def test_ultrametric_penalty_bad_string_rejected(tmp_path):
    with pytest.raises(ValueError, match="ultrametric_penalty"):
        Heracles(ultrametric_penalty="strong", allele_prior="global_uniform",
                 outdir=tmp_path)


def test_ultrametric_target_cv_coercion(tmp_path):
    m1 = Heracles(ultrametric_target_cv="0.3", allele_prior="global_uniform",
                  outdir=tmp_path)
    assert m1.ultrametric_target_cv == 0.3
    assert isinstance(m1.ultrametric_target_cv, float)

    m2 = Heracles(ultrametric_target_cv="auto",
                  allele_prior="global_uniform", outdir=tmp_path)
    assert m2.ultrametric_target_cv == "auto"


@pytest.mark.slow
def test_fit_writes_expected_files(tmp_path, small_cm):
    out = tmp_path / "run"
    m = Heracles(allele_prior="global_uniform", outdir=out,
                 max_iterations=10, dimension=5, init_method="mds",
                 convergence_thresh=None, readouts=["nj", "upgma"])
    m.fit(small_cm)

    assert (out / "tree_nj.nwk").exists()
    assert (out / "tree_upgma.nwk").exists()
    assert (out / "model.npz").exists()

    # distances and coordinates are stored in model.npz, not as separate CSVs
    npz = np.load(out / "model.npz", allow_pickle=False)
    meta = json.loads(str(npz["_metadata_json"]))
    assert "coords" in npz.files
    assert "full_dists" in npz.files
    assert npz["coords"].shape[0] == len(small_cm) + 1
    assert npz["full_dists"].shape == (len(small_cm) + 1, len(small_cm) + 1)


# Two-stage tree-likeness penalty

def test_two_stage_penalty_records_history(tmp_path):
    """Two-stage penalty records weight and delta histories."""

    states = _toy_states(n=20, S=12, seed=42)
    cm = pd.DataFrame(states[1:], index=["c%d" % i for i in range(20)])
    m = Heracles(readouts=["nj"], dimension=10,
                 max_iterations=300, convergence_thresh=None, seed=0,
                 heritable_eps=1e-5, stochastic_delta=0.2,
                 learn_heritable_eps=False, learn_stochastic_delta=False,
                 init_method="mds", allele_prior="global_uniform",
                 outdir=tmp_path,
                 treelike_penalty_target_delta=0.005,
                 batch_size=60)
    m.fit(cm)
    assert m.history_["treelike_weight"] is not None
    assert m.history_["gromov_delta"] is not None
    assert m.diagnostics_["stage_transition_step"] is not None
    assert m.diagnostics_["final_penalty_weight"] > 0


def test_two_stage_penalty_stage1_zero_weight(tmp_path):
    """Stage 1 has zero penalty weight; stage 2 ramps up."""

    states = _toy_states(n=12, S=6, seed=7)
    cm = pd.DataFrame(states[1:], index=["c%d" % i for i in range(12)])
    m = Heracles(readouts=["nj"], dimension=5,
                 max_iterations=60, convergence_thresh=None, seed=0,
                 heritable_eps=0.0, stochastic_delta=0.0,
                 learn_heritable_eps=False, learn_stochastic_delta=False,
                 init_method="mds", allele_prior="global_uniform",
                 outdir=tmp_path,
                 treelike_penalty_target_delta=0.005,
                 batch_size=32)
    m.fit(cm)
    w = m.history_["treelike_weight"]
    ts = m.diagnostics_["stage_transition_step"]
    assert ts is not None
    assert all(x == 0.0 for x in w[:ts])
    assert max(w[ts:]) > 0


def test_telemetry_reports_init_breakdown(tmp_path):
    from heracles import Heracles

    rng = np.random.default_rng(0)
    cm = pd.DataFrame(rng.integers(0, 4, (16, 8)), index=["c%d" % i for i in range(16)])
    m = Heracles(outdir=str(tmp_path / "fit"), dimension=4, max_iterations=3,
                 min_iterations=0, init_method="mds", readouts=["nj"])
    m.fit(cm)
    b = m.diagnostics_["telemetry"]["init_breakdown_s"]
    assert set(b) == {"tree_creation", "distance_extraction", "embedding"}
    assert b["embedding"] >= 0.0


def test_default_dimension_is_five_root_n(tmp_path):
    import math
    from heracles import Heracles, HeraclesConfig
    from heracles.fit import embedding_dimension

    assert embedding_dimension(5000) == 354 and embedding_dimension(0) == 2
    with pytest.raises(ValueError, match="not supported"):
        HeraclesConfig(dimension="auto")
    cfg, ignored = HeraclesConfig.from_params({"dimension": "auto"})   # saved config: ignored with a note
    assert cfg.dimension is None and any("auto" in x for x in ignored)

    rng = np.random.default_rng(1)
    n = 100                       # 5*sqrt(100) = 50 axes fits inside 101 points
    cm = pd.DataFrame(rng.integers(0, 4, (n, 8)), index=["c%d" % i for i in range(n)])
    m = Heracles(outdir=str(tmp_path / "fit"), max_iterations=2, min_iterations=0,
                 init_method="mds", readouts=["nj"])
    m.fit(cm)
    d = math.ceil(5 * math.sqrt(n))
    assert m.dimension_ == d and m.coords_.shape[1] == d + 1     # Lorentz: +1 coordinate


def test_polish_method(fitted_model, small_cm):
    from heracles.polish.nni import RefitResult
    res = fitted_model.polish(small_cm, readout="nj", max_rounds=1)
    assert isinstance(res, RefitResult)
    assert "nj_polished" in fitted_model.trees_
    assert res.loglik_end >= res.loglik_start
    assert os.path.exists(os.path.join(fitted_model.outdir, "tree_nj_polished.nwk"))
    assert "nj_polished_collapsed" in fitted_model.trees_
    assert os.path.exists(os.path.join(fitted_model.outdir,
                                       "tree_nj_polished_collapsed.nwk"))
    leaves = sorted(n.taxon.label for n in res.tree.leaf_node_iter())
    assert leaves == sorted(str(x) for x in fitted_model.labels_)


def test_init_tree_path_and_string(fitted_model, small_cm, tmp_path):
    nwk = fitted_model.newick("nj")
    path = tmp_path / "start.nwk"
    path.write_text(nwk + "\n")
    kw = dict(dimension=5, max_iterations=10, convergence_thresh=None,
              allele_prior="global_uniform", seed=0)
    a = Heracles(outdir=tmp_path / "a", init_tree=path, **kw).fit(small_cm)
    b = Heracles(outdir=tmp_path / "b", init_tree=nwk, **kw).fit(small_cm)
    assert a.diagnostics_["init_method_used"] == "mds"      # tree distances embedded directly
    np.testing.assert_allclose(a.history_["loss"], b.history_["loss"])
    assert a.get_params()["init_tree"] == str(path)
    assert b.get_params()["init_tree"] == "<newick string>"
    partial = "(%s);" % ",".join(str(x) for x in small_cm.index[:4])
    with pytest.raises(ValueError, match="init_tree lacks"):
        Heracles(outdir=tmp_path / "c", init_tree=partial, **kw).fit(small_cm)


def test_load_maps_flat_fitted_keys_into_diagnostics(fitted_model, tmp_path):
    """load() reads optimiser internals stored as flat fitted keys into diagnostics_."""
    path = tmp_path / "m.npz"
    fitted_model.save(path)
    data = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(data["_metadata_json"]))
    diag = meta.pop("diagnostics")
    meta["telemetry"] = diag.pop("telemetry")
    meta["fitted"].update({k + "_": v for k, v in diag.items()})
    data["_metadata_json"] = np.array(json.dumps(meta))
    old = tmp_path / "old.npz"
    np.savez(old, **data)
    loaded = Heracles.load(old)
    assert loaded.diagnostics_["init_time"] == fitted_model.diagnostics_["init_time"]
    assert loaded.diagnostics_["telemetry"] == fitted_model.diagnostics_["telemetry"]


# Input checks and error messages

@pytest.mark.parametrize("cm, kwargs, match", [
    (pd.DataFrame([[1, np.nan]], index=["a"]), {}, "NaN"),
    (pd.DataFrame([[1, -9]], index=["a"]), {}, "not the missing code"),
    (pd.DataFrame([[1.5, 0]], index=["a"]), {}, "non-integer"),
    (pd.DataFrame({"cell": ["x"], "s0": [1]}), {}, "not numeric"),
    (pd.DataFrame([[1], [2]], index=["a", "a"]), {}, "more than once"),
])
def test_character_matrix_rejects_bad_input(cm, kwargs, match):
    from heracles.fit import load_character_matrix
    with pytest.raises(ValueError, match=match):
        load_character_matrix(cm, **kwargs)


def test_character_matrix_accepts_path_and_other_missing_code(tmp_path):
    from pathlib import Path
    from heracles.fit import load_character_matrix
    cm = pd.DataFrame([[1, -9], [0, 2]], index=["a", "b"])
    states, labels = load_character_matrix(cm, missing_state=-9)
    assert states.tolist() == [[1, -1], [0, 2]] and labels == ["a", "b"]
    cm.to_csv(tmp_path / "cm.csv")
    states, _ = load_character_matrix(Path(tmp_path / "cm.csv"), missing_state=-9)
    assert states.tolist() == [[1, -1], [0, 2]]


def test_fit_warns_without_cell_names(small_cm, tmp_path):
    with pytest.warns(UserWarning, match="no cell names"):
        Heracles(dimension=5, max_iterations=5, convergence_thresh=None,
                 init_method="mds", outdir=tmp_path).fit(small_cm.reset_index(drop=True))


def test_friendly_constructor_errors(tmp_path):
    assert Heracles(outdir=tmp_path, readouts="upgma").readouts == ["upgma"]
    with pytest.raises(TypeError, match="did you mean max_iterations"):
        Heracles(outdir=tmp_path, max_iter=5)
    with pytest.raises(ValueError, match="device must be"):
        Heracles(outdir=tmp_path, device="gpu")
    (tmp_path / "t.nwk").write_text("(a,b);\n")
    with pytest.raises(ValueError, match="not a model.npz"):
        Heracles.load(tmp_path / "t.nwk")


def test_config_json_omits_allele_prior_array(fitted_model):
    cfg = json.loads(Path(fitted_model.outdir, "config.json").read_text())
    assert "learned_allele_prior_" not in (cfg["fitted"] or {})
    assert "learned_allele_prior_" in fitted_model.get_params()["fitted"]


def test_refit_result_repr_is_short(fitted_model, small_cm):
    res = fitted_model.polish(small_cm, max_rounds=1, write=False)
    assert repr(res).startswith("RefitResult(") and len(repr(res)) < 200


def test_nj_falls_back_with_a_warning(small_cm, tmp_path, monkeypatch):
    import heracles.trees.readouts as ro
    from heracles.trees.utils import robinson_foulds
    kw = dict(dimension=5, max_iterations=10, convergence_thresh=None,
              init_method="mds", seed=0)
    fast = Heracles(outdir=tmp_path / "a", **kw).fit(small_cm)
    monkeypatch.setattr(ro, "rapidnj_available", lambda: False)
    with pytest.warns(UserWarning, match="falls back to the pure-Python"):
        slow = Heracles(outdir=tmp_path / "b", **kw).fit(small_cm)
    assert robinson_foulds(fast.trees_["nj"], slow.trees_["nj"]) == 0


def test_rapidnj_readout_name_points_to_nj(tmp_path):
    with pytest.raises(ValueError, match="'rapidnj' is now 'nj'"):
        Heracles(outdir=tmp_path, readouts=["nj", "rapidnj"])
