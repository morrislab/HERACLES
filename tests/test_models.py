"""Tests for the edit-model template (heracles.models.base)."""

from dataclasses import dataclass
from typing import ClassVar

import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import _toy_states
from heracles import Heracles
from heracles.models import (MODELS, PROCESSES, CassetteDropout, Cas9Model, Cas9Process,
                             CustomModel, EditProcess, PETracerModel, SequentialModel,
                             SequentialProcess, SiteDropout, get_model, model_from_legacy_params,
                             register_process)
from heracles.embedding.objective import quartet_loss_fn


def _cm(n=16, S=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, size=(n, S))
    return pd.DataFrame(x, index=["c%d" % i for i in range(n)])


_FAST = dict(dimension=4, max_iterations=10, convergence_thresh=None, init_method="mds",
             allele_prior="global_uniform", seed=0)


def test_registry_and_defaults():
    assert set(MODELS) >= {"cas9", "petracer", "sequential"}
    assert get_model(None) == Cas9Model()
    assert get_model("petracer") == PETracerModel()
    assert get_model({"name": "sequential", "cassette_size": 4}) == SequentialModel(cassette_size=4)
    with pytest.raises(ValueError, match="unknown edit model"):
        get_model("nonsense")


def test_capabilities_are_declared():
    seq = SequentialModel(cassette_size=3)
    assert Cas9Model().supports_low_memory and Cas9Model().supports_polish
    assert PETracerModel().supports_low_memory and PETracerModel().supports_polish
    assert not seq.supports_low_memory and not seq.supports_polish


def test_presets_are_compositions():
    assert Cas9Model().components() == (Cas9Process(), SiteDropout())
    assert PETracerModel(cassette_size=2).components() == (Cas9Process(), CassetteDropout(2))
    proc, miss = SequentialModel(cassette_size=4).components()
    assert isinstance(proc, SequentialProcess) and miss == CassetteDropout(4)


def test_unsupported_pairing_fails_early(tmp_path):
    with pytest.raises(ValueError, match="cannot score SiteDropout"):
        Heracles(edit_model=CustomModel(SequentialProcess(cassette_size=3), SiteDropout()),
                 outdir=tmp_path / "x")
    with pytest.raises(ValueError, match="one shared rate"):
        Heracles(edit_model=CustomModel(SequentialProcess(cassette_size=3), CassetteDropout(2)),
                 outdir=tmp_path / "y")


def test_template_requires_the_two_methods():
    @dataclass
    class Incomplete(EditProcess):
        name: ClassVar[str] = "incomplete"
    with pytest.raises(TypeError):
        Incomplete()


def test_custom_process_plugs_in(tmp_path):
    """A user process: Cas9 likelihood on a different seed matrix, in a CustomModel."""
    calls = {}

    @register_process
    @dataclass
    class PlainHamming(EditProcess):
        name: ClassVar[str] = "plain_hamming_test"
        supported_missing: ClassVar[tuple] = (SiteDropout,)
        scale: float = 1.0

        def seed_distances(self, states, *, dtype=np.float64):
            d = (states[:, None, :] != states[None, :, :]).mean(-1).astype(dtype) * self.scale
            np.fill_diagonal(d, 0.0)
            calls["seed"] = d.shape
            return d

        def loss_function(self, states, allele_prior, *, missing, seed_dists, batch_size,
                          root_weight, learn_allele_prior, config, device):
            calls["loss"] = True
            return quartet_loss_fn(states, allele_prior, states.shape[0], est_dists=seed_dists,
                                   batch_size=batch_size, root_weight=root_weight,
                                   learn_allele_prior=learn_allele_prior, config=config,
                                   device=device)
    try:
        model = CustomModel(PlainHamming(scale=2.0))
        m = Heracles(edit_model=model, outdir=tmp_path / "c", **_FAST).fit(_cm())
        assert calls == {"seed": (17, 17), "loss": True}
        assert m.get_params()["edit_model"] == {
            "name": "custom", "process": {"name": "plain_hamming_test", "scale": 2.0},
            "missing": {"name": "site", "separate_rates": False}}
        loaded = Heracles.load(os.path.join(m.outdir, "model.npz"))
        assert loaded.edit_model.process == PlainHamming(scale=2.0)
        with pytest.raises(ValueError, match="polish is not available"):
            m.polish(_cm())
    finally:
        PROCESSES.pop("plain_hamming_test", None)


def test_legacy_config_keys_map_to_models():
    assert model_from_legacy_params({}) == Cas9Model()
    assert model_from_legacy_params({"stochastic_delta_per_site": True}) == \
        Cas9Model(separate_dropout_rates=True)
    assert model_from_legacy_params({"dropout_unit": "cassette", "cassette_size": 3,
                                     "stochastic_delta_per_site": True}) == \
        PETracerModel(cassette_size=3, separate_dropout_rates=True)
    assert model_from_legacy_params({"likelihood": "sequential", "cassette_size": 5,
                                     "initiation_ratio": 2.0}) == \
        SequentialModel(cassette_size=5, initiation_ratio=2.0)


def test_load_maps_flat_model_keys_to_edit_model(tmp_path):
    """load() builds the edit model from the flat dropout_unit / likelihood keys
    that model.npz files without an edit_model entry carry."""
    m = Heracles(edit_model=PETracerModel(cassette_size=2), outdir=tmp_path / "old",
                 **_FAST).fit(_cm(S=6))
    path = os.path.join(m.outdir, "model.npz")
    data = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(data["_metadata_json"]))
    meta.pop("edit_model")
    meta.update(dropout_unit="cassette", cassette_size=2, likelihood="cas9",
                stochastic_delta_per_site=False)
    data["_metadata_json"] = np.array(json.dumps(meta))
    old = os.path.join(tmp_path, "old_format.npz")
    np.savez(old, **data)
    assert Heracles.load(old).edit_model == PETracerModel(cassette_size=2)


def test_cli_rejects_option_of_another_model():
    import argparse
    from heracles.models import add_model_arguments, model_from_args
    p = argparse.ArgumentParser()
    add_model_arguments(p)
    assert model_from_args(p.parse_args([])) == Cas9Model()
    assert model_from_args(p.parse_args(["--edit-model", "petracer"])) == PETracerModel()
    with pytest.raises(ValueError, match="option of"):
        model_from_args(p.parse_args(["--initiation-ratio", "2"]))


def test_cas9_quartet_loss_matches_tree_likelihood():
    """The template's consistency check: the Cas9 quartet kernel and the tree
    likelihood give the same probability for one quartet under one model."""
    import torch
    from heracles.models.missing import SiteDropout
    from heracles.embedding.objective import quartet_branch_lengths, quartet_site_logprob
    from heracles.trees.utils import FOUNDER_LABEL, parse_newick, tree_distances
    mu, eps, delta = 0.8, 0.1, 0.15
    prior = np.array([1.0, 0.6, 0.4])
    nwk = "((A:0.3,B:0.5):0.4,(C:0.2,D:0.6):0.35);"
    labels = ["A", "B", "C", "D"]
    founder_edge = 0.25
    tree = parse_newick(nwk, rooting="force-rooted")
    # exact founder-augmented distances: founder sits above the root
    with_founder = parse_newick("(%s:%g,(%s):%g);" % (FOUNDER_LABEL, 0.0, nwk[1:-2], founder_edge))
    D = tree_distances(with_founder, [FOUNDER_LABEL] + labels)
    rng = np.random.default_rng(0)
    states = rng.choice([0, 1, 2, -1], size=(4, 25), p=[0.4, 0.25, 0.25, 0.1])
    st_aug = torch.as_tensor(np.vstack([np.zeros((1, 25), dtype=np.int64), states]))
    d = torch.as_tensor(D)
    q = torch.tensor([[1, 2, 3, 4]])
    dists_6 = tuple(d[i, j].reshape(1) for i, j in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)))
    bl = quartet_branch_lengths(d[0], dists_6, q)
    kernel = quartet_site_logprob(st_aug, *bl, torch.tensor(mu), torch.as_tensor(prior),
                                  heritable_eps=torch.tensor(eps),
                                  stochastic_delta=torch.tensor(delta)).numpy().ravel()
    cm = pd.DataFrame(states, index=labels)
    lik = Cas9Process().tree_likelihood(cm, missing=SiteDropout(), mutation_rate=mu,
                                        heritable_eps=eps, stochastic_delta=delta,
                                        allele_prior=prior)
    from heracles.polish.ancestral import infer_ancestral_states
    per_site = infer_ancestral_states(tree, cm, likelihood=lik, root_edge_length=founder_edge,
                                      posteriors=False).loglik_per_site
    np.testing.assert_allclose(kernel, per_site, rtol=1e-8)
    assert lik.loglik(tree, root_edge_length=founder_edge) == pytest.approx(per_site.sum())


def test_docstring_example_process_fits_and_reloads(tmp_path):
    """The ToyProcess example in heracles.models.base runs as written."""
    import textwrap
    from dataclasses import dataclass
    from typing import ClassVar
    import torch
    import heracles.models.base as base
    from heracles.models import CustomModel, EditProcess, SiteDropout, register_process
    doc = base.__doc__
    start = doc.index("    from heracles.embedding.initialization import seed_distances")
    end = doc.index("    Heracles(outdir=...")
    ns = dict(np=np, torch=torch, dataclass=dataclass, ClassVar=ClassVar,
              EditProcess=EditProcess, SiteDropout=SiteDropout,
              register_process=register_process)
    exec(textwrap.dedent(doc[start:end]), ns)
    try:
        cm = pd.DataFrame(_toy_states(n=20, S=8, seed=3)[1:],
                          index=["c%d" % i for i in range(20)])
        m = Heracles(outdir=tmp_path, edit_model=CustomModel(ns["ToyProcess"]()),
                     max_iterations=30, convergence_thresh=None, init_method="mds",
                     dimension=4).fit(cm)
        assert np.isfinite(m.history_["loss"]).all()
        assert type(Heracles.load(tmp_path / "model.npz").edit_model.process).__name__ == "ToyProcess"
    finally:
        from heracles.models import PROCESSES
        PROCESSES.pop("toy", None)


def test_registering_a_taken_name_raises():
    from heracles.models import register_process

    with pytest.raises(ValueError, match="already registered to heracles.models.cas9.Cas9Process"):
        @register_process
        @dataclass
        class Impostor(Cas9Process):
            name: ClassVar[str] = "cas9"


def test_unregistered_custom_process_warns_at_construction(tmp_path):
    from heracles.models import CustomModel, SiteDropout

    @dataclass
    class Unlisted(Cas9Process):
        name: ClassVar[str] = "unlisted"

    with pytest.warns(UserWarning, match="Unlisted .decorate it with @register_process"):
        Heracles(outdir=tmp_path, edit_model=CustomModel(Unlisted(), SiteDropout()))


def test_polish_names_missing_tree_likelihood_parts(tmp_path):
    from heracles.models import CustomModel, SiteDropout, register_process, PROCESSES

    @register_process
    @dataclass
    class HalfDone(Cas9Process):
        name: ClassVar[str] = "half_done"

        def tree_likelihood(self, char_matrix, **kwargs):
            return object()

    try:
        cm = pd.DataFrame(_toy_states(n=12, S=6, seed=1)[1:],
                          index=["c%d" % i for i in range(12)])
        m = Heracles(outdir=tmp_path, edit_model=CustomModel(HalfDone(), SiteDropout()),
                     max_iterations=10, convergence_thresh=None, init_method="mds",
                     dimension=4).fit(cm)
        with pytest.raises(TypeError, match="without labels, prior_mass, emissions"):
            m.polish(cm)
    finally:
        PROCESSES.pop("half_done", None)
