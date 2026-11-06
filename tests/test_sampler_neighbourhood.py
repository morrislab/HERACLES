"""The quartet samplers: neighbourhood (the default) and uniform."""
import argparse
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import _toy_states
from heracles.config import HeraclesConfig, add_config_arguments, config_from_args
from heracles.embedding.initialization import seed_distances
from heracles.embedding.sampler import (SAMPLERS, build_quartet_sampler, build_tau_neighbours,
                                        build_tau_topk, register_sampler,
                                        sample_quartets_neighbourhood, sample_quartets_uniform,
                                        sampler_reach)
from tests.test_quartet_embedding import _tau_setup


def _ranks(q, order, col):
    """1-based rank of column ``col`` of each quartet in a's neighbour ranking (0 = absent)."""
    z = q.numpy() - 1
    rows = np.asarray(order)[z[:, 0]]
    hit = rows == z[:, col][:, None]
    return np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)


def _check_rows(q, tau, n):
    z = q.numpy() - 1
    assert q.shape[1] == 4 and int(q.min()) >= 1 and int(q.max()) <= n
    srt = np.sort(z, axis=1)
    assert (srt[:, 1:] > srt[:, :-1]).all(), "a row repeated a taxon"
    for a, b, c, d in z:
        assert max(tau[a, c], tau[a, d], tau[b, c], tau[b, d]) < tau[a, b]


def _tables(est, k):
    order, tau_knn, tau = build_tau_topk(est, k)
    tau_t = torch.as_tensor(tau)
    return dict(order=order, tau_knn=tau_knn, tau_fn=lambda i, j: tau_t[i, j]), tau


def _sha(q):
    return hashlib.sha256(q.numpy().astype("int64").tobytes()).hexdigest()


# sha256 of three 800-quartet batches (seed 3) from the neighbourhood sampler of
# the paper's runs (phylogeny 5c2d418, full (n, n) ranking): uncapped, and capped
# at 16. Pins the sampler's RNG stream.
_PAPER_SHA = {
    None: "c450a02ca03a8a99cf2d276b84d7262e306840b394a600c14b99787f1a080fea",
    16: "30594f7a47d9d597c1496547168c9f87f975c4a26880063f82bf3b2a52270857",
}


def test_topk_table_is_the_head_of_the_full_sort():
    _, _, _, est = _tau_setup(n=90)
    order_full, tau_full = build_tau_neighbours(est)
    for k in (5, 40, 89, 500):
        order, tau_knn, tau = build_tau_topk(est, k, row_chunk=32)
        kk = min(k, 89)
        np.testing.assert_array_equal(order, order_full[:, :kk])
        np.testing.assert_array_equal(tau, tau_full)
        np.testing.assert_array_equal(tau_knn, np.take_along_axis(tau_full, order, axis=1))


@pytest.mark.parametrize("max_k", [None, 16])
def test_neighbourhood_stream_matches_the_paper_runs(max_k):
    _, _, _, est = _tau_setup(n=80)
    sampler, cfg, _ = build_quartet_sampler(est, 81, sampler_max_k=max_k)
    assert cfg["strategy"] == "neighbourhood"
    assert cfg["table_k"] == (79 if max_k is None else max_k)
    g = torch.Generator().manual_seed(3)
    q = torch.cat([sampler(80, 800, generator=g) for _ in range(3)])
    assert _sha(q) == _PAPER_SHA[max_k]


def test_neighbourhood_draws_valid_local_quartets():
    order, tau, _, est = _tau_setup(n=80)
    kw, _ = _tables(est, 79)
    q, s = sample_quartets_neighbourhood(80, 800, generator=torch.Generator().manual_seed(3), **kw)
    _check_rows(q, tau, 80)
    assert s["fallback_frac"] == 0.0 and 0.0 < s["accept_rate"] <= 1.0
    rb, rc, rd = (_ranks(q, order, j) for j in (1, 2, 3))
    assert (rb > 0).all() and (rc > 0).all() and (rd > 0).all()   # b, c, d all neighbours of a
    assert ((rb < rc) & (rb < rd)).all()                          # b is the nearest of the three
    q2, _ = sample_quartets_neighbourhood(80, 800, generator=torch.Generator().manual_seed(3), **kw)
    assert torch.equal(q, q2)
    with pytest.raises(ValueError):
        sample_quartets_neighbourhood(80, 10, order=kw["order"], tau_knn=kw["tau_knn"])


def test_neighbourhood_concentrates_on_close_relatives():
    """The first pair of a neighbourhood quartet is closer than a uniformly drawn pair."""
    _, tau, _, est = _tau_setup(n=40)
    kw, _ = _tables(est, 39)
    qn, _ = sample_quartets_neighbourhood(40, 2000, generator=torch.Generator().manual_seed(1), **kw)
    qu, _ = sample_quartets_uniform(40, 2000, generator=torch.Generator().manual_seed(1))
    zn, zu = qn.numpy() - 1, qu.numpy() - 1
    assert tau[zn[:, 0], zn[:, 1]].mean() > tau[zu[:, 0], zu[:, 1]].mean()


def test_neighbourhood_size_cap():
    order, tau, _, est = _tau_setup(n=80)
    kw, _ = _tables(est, 79)
    q, s = sample_quartets_neighbourhood(80, 900, generator=torch.Generator().manual_seed(9),
                                         max_k=0.5, **kw)
    _check_rows(q, tau, 80)
    assert s["fallback_frac"] == 0.0
    assert max(_ranks(q, order, j).max() for j in (1, 2, 3)) <= 40     # within 0.5 n
    q8, _ = sample_quartets_neighbourhood(80, 900, generator=torch.Generator().manual_seed(9),
                                          max_k=8, **kw)
    assert max(_ranks(q8, order, j).max() for j in (1, 2, 3)) <= 8
    # a table of exactly max_k columns draws the same quartets as the full ranking
    kw8, _ = _tables(est, 8)
    q8t, _ = sample_quartets_neighbourhood(80, 900, generator=torch.Generator().manual_seed(9),
                                           max_k=8, **kw8)
    assert torch.equal(q8, q8t)


@pytest.mark.parametrize("max_k,knn", [(32, None), (32, 64), (0.2, None), (None, None)])
def test_capped_full_and_low_memory_draw_identical_quartets(max_k, knn):
    st = _toy_states(n=300, S=30, seed=2, missing_frac=0.1)
    est = seed_distances(st)
    reach = sampler_reach("neighbourhood", max_k, 300)
    s_full, cfg_full, _ = build_quartet_sampler(est, 301, sampler_max_k=max_k)
    s_low, cfg_low, _ = build_quartet_sampler(None, 301, sampler_max_k=max_k,
                                              sampler_knn=knn or reach, character_matrix=st)
    assert cfg_full["tau_source"] == "est_dists"
    assert cfg_low["tau_source"] == "knn_%d" % min(knn or reach, 299)
    g1, g2 = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    for _ in range(3):
        assert torch.equal(s_full(300, 1500, generator=g1), s_low(300, 1500, generator=g2))


def test_uniform_sampler_draws_distinct_quartets_without_distances():
    q, s = sample_quartets_uniform(30, 400, generator=torch.Generator().manual_seed(0))
    assert q.shape == (400, 4) and int(q.min()) >= 1 and int(q.max()) <= 30 and s == {}
    srt = q.sort(dim=1).values
    assert bool((srt[:, 1:] > srt[:, :-1]).all())
    q2, _ = sample_quartets_uniform(30, 400, generator=torch.Generator().manual_seed(0))
    assert torch.equal(q, q2)
    with pytest.raises(ValueError):
        sample_quartets_uniform(3, 10)
    sampler, cfg, stats = build_quartet_sampler(None, 31, sampler_strategy="uniform")
    assert cfg == {"strategy": "uniform"}
    assert sampler(30, 50, generator=torch.Generator().manual_seed(1)).shape == (50, 4)
    assert stats == {}
    # the ranking is not read at all, so a leaf-only matrix cannot shift the indices
    sampler, _, _ = build_quartet_sampler(np.zeros((30, 30)), 31, sampler_strategy="uniform")
    assert sampler(30, 5).shape == (5, 4)
    with pytest.raises(ValueError, match="sampler_strategy"):
        build_quartet_sampler(None, 31, sampler_strategy="banded")


def test_degenerate_tau_falls_back_to_uniform():
    """All-zero tau (e.g. an all-unedited matrix) must not hang or return junk."""
    with pytest.warns(UserWarning, match="entirely zero"):
        sampler, cfg, _ = build_quartet_sampler(np.zeros((13, 13)), 13)
    assert cfg == {"strategy": "uniform", "degenerate": True}
    q = sampler(12, 60, generator=torch.Generator().manual_seed(0))
    assert q.shape == (60, 4)
    srt = q.sort(dim=1).values
    assert bool((srt[:, 1:] > srt[:, :-1]).all())


def test_registered_sampler_is_reachable_by_name(tmp_path):
    from heracles.fit import Heracles

    @register_sampler("every_fourth_test")
    def build(est_dists, n_points, **_):
        def sampler(n_leaves, n_quartets, generator=None, device=None):
            q, _ = sample_quartets_uniform(n_leaves, n_quartets, generator=generator, device=device)
            return q
        return sampler, {"strategy": "every_fourth_test"}, {}
    try:
        assert HeraclesConfig(sampler_strategy="every_fourth_test").sampler_strategy == "every_fourth_test"
        st = _toy_states(n=20, S=10, K=4, seed=0)[1:]
        cm = pd.DataFrame(st, index=["c%d" % i for i in range(20)])
        m = Heracles(outdir=str(tmp_path / "fit"), sampler_strategy="every_fourth_test",
                     max_iterations=3, dimension=3, seed=0, init_method="mds")
        m.fit(cm)
        assert m.diagnostics_["init_method_used"]
    finally:
        SAMPLERS.pop("every_fourth_test")
    with pytest.raises(ValueError, match="sampler_strategy"):
        HeraclesConfig(sampler_strategy="every_fourth_test")


def test_reach_and_config_defaults():
    assert sampler_reach("neighbourhood", 1024, 50_000) == 1024
    assert sampler_reach("neighbourhood", 1024, 500) == 499
    assert sampler_reach("neighbourhood", None, 500) == 499
    assert sampler_reach("neighbourhood", 0.5, 500) == 250
    assert sampler_reach("uniform", 1024, 500) is None
    with pytest.raises(ValueError):
        sampler_reach("banded", 16, 500)
    cfg = HeraclesConfig()
    assert cfg.sampler_strategy == "neighbourhood" and cfg.sampler_max_k == 1024
    c = config_from_args(add_config_arguments(argparse.ArgumentParser()).parse_args(
        ["--sampler-strategy", "uniform", "--sampler-max-k", "0.5"]))
    assert c.sampler_strategy == "uniform" and c.sampler_max_k == 0.5
    for bad in (dict(sampler_max_k=0), dict(sampler_strategy="banded")):
        with pytest.raises(ValueError):
            HeraclesConfig(**bad)
    with pytest.raises(TypeError):
        HeraclesConfig(band_kmax=16)
    cfg, ignored = HeraclesConfig.from_params({"sampler_strategy": "uniform", "sampler_max_k": 64})
    assert cfg.sampler_strategy == "uniform" and cfg.sampler_max_k == 64 and ignored == []


@pytest.mark.parametrize("strategy", ["neighbourhood", "uniform"])
def test_low_memory_fit_with_each_sampler(tmp_path, strategy):
    from heracles.fit import Heracles
    st = _toy_states(n=60, S=20, K=4, seed=0, missing_frac=0.1)[1:]
    cm = pd.DataFrame(st, index=["c%d" % i for i in range(60)])
    m = Heracles(outdir=str(tmp_path / "fit"), low_memory=True, sampler_strategy=strategy,
                 sampler_max_k=16, max_iterations=5, dimension=4, seed=0)
    m.fit(cm)
    log = (tmp_path / "fit" / "heracles.log").read_text()
    if strategy == "neighbourhood":
        assert m.diagnostics_["sampler_knn"] == 16
        assert "neighbourhood sampler: top-16 table" in log and "on the fly" in log
    else:
        assert m.diagnostics_["sampler_knn"] is None
    assert m.config.sampler_strategy == strategy
