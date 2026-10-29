"""Low-memory mode: no n x n array during the fit, same answers as the full path."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.distance import squareform
from conftest import _toy_states

from heracles.trees.rapidnj import rapidnj_available
from heracles.embedding.initialization import (
    condensed_weighted_hamming, iter_weighted_hamming_rows, tree_init_dists,
    seed_distances,
)
from heracles.fit import Heracles, memory_plan
from heracles.embedding.optimize import iter_dist_rows, pairwise_dists
from heracles.embedding.sampler import (banded_sampler_reach, build_banded_sampler,
                                        build_tau_neighbours, build_tau_neighbours_from_cm)
from heracles.trees.readouts import heracles_upgma, upgma_condensed_height
from heracles.trees.utils import (
    FOUNDER_LABEL, as_newick, nj_tree, nj_tree_from_rows, robinson_foulds, tree_distances,
    parse_newick, write_phylip, write_phylip_rows,
)


def _cm(n=60, S=20, seed=0, missing=0.1):
    st = _toy_states(n=n, S=S, K=4, seed=seed, missing_frac=missing)[1:]
    return pd.DataFrame(st, index=["c%d" % i for i in range(n)],
                        columns=["s%d" % j for j in range(S)])


def test_seed_distance_rows_match_full_matrix():
    st = _toy_states(n=80, S=15, seed=3, missing_frac=0.15)
    full = seed_distances(st)
    rows = np.array([0, 5, 17, 42, 79])
    block = seed_distances(st, rows)
    assert block.shape == (5, 81)
    np.testing.assert_allclose(block, full[rows], rtol=0, atol=0)


def test_banded_sampler_reach():
    assert banded_sampler_reach(16) == 16
    assert banded_sampler_reach(16, 128) == 128
    assert banded_sampler_reach(16, 128, 1.0, 4) == 512
    assert banded_sampler_reach(16, None, 0.5, None) == 64


def test_knn_from_cm_matches_full_sort_order():
    st = _toy_states(n=150, S=25, seed=1, missing_frac=0.1)
    est = seed_distances(st)
    order_full, tau_full = build_tau_neighbours(est)
    k = 20
    order_knn, tau_knn, d_root = build_tau_neighbours_from_cm(st[1:], k)
    np.testing.assert_allclose(d_root, est[0, 1:])
    np.testing.assert_array_equal(order_knn, order_full[:, :k])
    tau_along_full = np.take_along_axis(tau_full, order_full[:, :k], axis=1)
    np.testing.assert_allclose(tau_knn, tau_along_full, atol=1e-12)


def test_knn_sampler_draws_same_quartets_as_full_sort():
    st = _toy_states(n=200, S=30, seed=2, missing_frac=0.05)
    est = seed_distances(st)
    kw = dict(band_kmin=2, band_kmax=8, band_kmax_end=None, band_local_frac=1.0,
              band_local_mult_max=4)
    reach = banded_sampler_reach(8, None, 1.0, 4)
    s_full, _, _ = build_banded_sampler(est, st.shape[0], **kw)
    s_knn, cfg_knn, _ = build_banded_sampler(None, st.shape[0], sampler_knn=reach,
                                             character_matrix=st, **kw)
    assert cfg_knn["tau_source"] == "knn_%d" % reach
    g1 = torch.Generator().manual_seed(11); g2 = torch.Generator().manual_seed(11)
    q_full = s_full(200, 500, generator=g1)
    q_knn = s_knn(200, 500, generator=g2)
    # the k-NN table reproduces the stable full sort, so the draws are identical
    assert torch.equal(q_full, q_knn)


def test_pairwise_dists_and_rows_agree():
    rng = np.random.default_rng(0)
    tang = rng.normal(scale=0.2, size=(70, 6))
    # lift to the Lorentz model
    x0 = np.sqrt(1.0 + (tang ** 2).sum(1, keepdims=True))
    coords = np.hstack([x0, tang])
    D = pairwise_dists(coords)
    assert D.shape == (70, 70) and np.allclose(D, D.T) and np.all(np.diag(D) == 0)
    rows = np.vstack([b for _, _, b in iter_dist_rows(coords, chunk=16, col_chunk=23)])
    assert np.all(np.diag(rows) == 0.0)
    np.testing.assert_allclose(rows, D, atol=1e-9)      # matmul vs broadcast summation order
    cond = upgma_condensed_height(iter_dist_rows(coords, chunk=16), coords.shape[0])
    d0 = D[0, 1:]; L = D[1:, 1:]
    tau = 0.5 * (d0[:, None] + d0[None, :] - L); np.clip(tau, 0, None, out=tau)
    np.fill_diagonal(tau, 0.0)
    height = tau.max() - tau; np.fill_diagonal(height, 0.0)
    np.testing.assert_allclose(cond, squareform(height, checks=False), atol=1e-9)


def test_low_memory_fit_end_to_end(tmp_path):
    cm = _cm(n=60)
    m = Heracles(outdir=tmp_path / "lm", readouts=["nj", "upgma"], low_memory=True,
                 init_method="mds", dimension=12, max_iterations=20,
                 convergence_thresh=None, allele_prior="global_uniform")
    m.fit(cm)
    assert m.dimension_ == 12 and m.coords_.shape == (61, 13)
    assert m.diagnostics_["init_method_used"] == "landmark_mds"
    assert m.diagnostics_["init_meta"]["n_landmarks"] == 61            # n < 500: every cell is a landmark
    assert m.diagnostics_["sampler_knn"] == min(banded_sampler_reach(
        m.band_kmax, m.band_kmax_end, m.band_local_frac, m.band_local_mult_max), 59)
    assert m._full_dists is None                           # nothing built during the fit
    data = np.load(tmp_path / "lm" / "model.npz", allow_pickle=False)
    assert "full_dists" not in data.files and "coords" in data.files
    # lazy distances: shape, dtype, consistency with the coordinates
    D = m.full_dists_
    assert D.shape == (61, 61) and D.dtype == np.float32 and m._full_dists is not None
    np.testing.assert_allclose(D, pairwise_dists(m.coords_), atol=1e-5)
    assert m.leaf_dists_.shape == (60, 60)
    for r in ("nj", "upgma"):
        taxa = sorted(t.label for t in m.trees_[r].leaf_nodes() for t in [t.taxon])
        assert taxa == sorted(cm.index)
    assert m.diagnostics_["telemetry"]["peak_rss_mb"] > 0
    loaded = Heracles.load(tmp_path / "lm" / "model.npz")
    assert loaded.dimension_ == 12 and loaded._full_dists is None
    np.testing.assert_allclose(loaded.full_dists_, D, atol=1e-6)


def test_low_memory_upgma_matches_full_path_readout(tmp_path):
    cm = _cm(n=50, seed=4)
    m = Heracles(outdir=tmp_path / "lm", readouts=["upgma"], low_memory=True,
                 init_method="mds", dimension=8, max_iterations=15, convergence_thresh=None,
                 allele_prior="global_uniform").fit(cm)
    emb_full = {"full_dists": pairwise_dists(m.coords_), "labels": m.labels_}
    t_full = heracles_upgma(emb_full)
    assert robinson_foulds(t_full, m.trees_["upgma"]) == 0


@pytest.mark.skipif(not rapidnj_available(), reason="RapidNJ binary not available")
def test_streamed_rapidnj_matches_full_matrix(tmp_path):
    rng = np.random.default_rng(1)
    tang = rng.normal(scale=0.3, size=(40, 5))
    coords = np.hstack([np.sqrt(1.0 + (tang ** 2).sum(1, keepdims=True)), tang])
    labels = [FOUNDER_LABEL] + ["c%d" % i for i in range(39)]
    D = pairwise_dists(coords)
    t1 = nj_tree(D, labels, rapidnj=True)
    t2 = nj_tree_from_rows(iter_dist_rows(coords, chunk=7), labels, 40)
    assert as_newick(t1) == as_newick(t2)


def test_memory_plan_lines():
    lines = memory_plan(150000, 1937, low_memory=True, dist_dtype=np.float32)
    assert any("low_memory" in l for l in lines) and any("upgma" in l for l in lines)
    lines = memory_plan(150000, 1937, low_memory=False)
    assert any("hint: low_memory" in l for l in lines)


@pytest.mark.skipif(not rapidnj_available(), reason="RapidNJ binary not available")
def test_low_memory_with_rapidnj_init_reproduces_standard_fit(tmp_path):
    """n < landmarks: every point is a landmark, so with the RapidNJ init the
    low-memory fit is the standard algorithm and must give the same losses."""
    cm = _cm(n=60, seed=6)
    kw = dict(readouts=["nj"], dimension=10, max_iterations=12, convergence_thresh=None,
              allele_prior="global_uniform")
    a = Heracles(outdir=tmp_path / "std", **kw).fit(cm)
    b = Heracles(outdir=tmp_path / "lm", low_memory=True, **kw).fit(cm)   # default init: rapidnj
    assert b.diagnostics_["init_method_used"] == "tree+landmark_mds"
    assert b.diagnostics_["init_tree_newick"] == a.diagnostics_["init_tree_newick"]
    np.testing.assert_allclose(b.history_["loss"], a.history_["loss"], rtol=1e-9)
    np.testing.assert_allclose(b.coords_, a.coords_, atol=1e-9)
    assert robinson_foulds(a.trees_["nj"], b.trees_["nj"]) == 0


def test_streamed_seed_distance_rows_match_full_matrix(tmp_path):
    st = _toy_states(n=90, S=12, seed=8, missing_frac=0.1)
    labels = ["t%d" % i for i in range(st.shape[0])]
    full = seed_distances(st)
    blocks = list(iter_weighted_hamming_rows(st, chunk=17))
    np.testing.assert_array_equal(np.vstack([b for _, _, b in blocks]), full)
    write_phylip(full, labels, tmp_path / "a.phy")
    write_phylip_rows(iter_weighted_hamming_rows(st, chunk=17), labels, tmp_path / "b.phy", len(labels))
    assert (tmp_path / "a.phy").read_bytes() == (tmp_path / "b.phy").read_bytes()


def test_tree_distances_rows_are_bit_identical_to_full_rows():
    nwk = "((a:0.1,(b:0.2,c:0.05):0.3):0.4,(d:0.7,(e:0.1,f:0.6):0.2):0.1,g:0.9);"
    tree = parse_newick(nwk)
    labels = list("gfedcba")
    full = tree_distances(tree, labels)
    rows = np.array([0, 3, 6])
    block = tree_distances(tree, labels, rows=rows)
    assert block.shape == (3, 7)
    np.testing.assert_array_equal(block, full[rows, :])


@pytest.mark.skipif(not rapidnj_available(), reason="RapidNJ binary not available")
def test_rapidnj_landmark_dists_matches_full_init_rows():
    st = _toy_states(n=120, S=20, seed=9, missing_frac=0.1)
    labels = ["t%d" % i for i in range(st.shape[0])]
    full, nwk_full, _ = tree_init_dists(
        st, labels, method="rapidnj",
        seed_dists=seed_distances(st))
    lm = np.array([0, 4, 17, 60, 119])
    block, nwk_lm, _ = tree_init_dists(st, labels, method="rapidnj", rows=lm)
    assert nwk_lm == nwk_full
    np.testing.assert_array_equal(block, full[lm, :])


def test_add_readouts_on_loaded_low_memory_model(tmp_path):
    cm = _cm(n=40, seed=11)
    m = Heracles(outdir=tmp_path / "lm", readouts=["nj"], low_memory=True, init_method="mds",
                 dimension=6, max_iterations=10, convergence_thresh=None,
                 allele_prior="global_uniform").fit(cm)
    loaded = Heracles.load(tmp_path / "lm" / "model.npz")
    assert "upgma" not in loaded.trees_
    new = loaded.add_readouts(["upgma"])
    assert set(new) == {"upgma"} and loaded._full_dists is None      # streamed from coords
    assert (tmp_path / "lm" / "tree_upgma.nwk").exists()
    again = Heracles.load(tmp_path / "lm" / "model.npz")
    assert set(again.trees_) == {"nj", "nj_collapsed", "upgma"}
    # same tree as the eager read-out on the materialised matrix
    emb_full = {"full_dists": pairwise_dists(m.coords_), "labels": m.labels_}
    assert robinson_foulds(heracles_upgma(emb_full), new["upgma"]) == 0


def test_grad_accum_matches_full_batch_gradient():
    """Summing k micro-batch losses / k gives the full-batch loss and gradient."""
    from heracles.embedding.initialization import lorentz_manifold, init_points
    from heracles.embedding.objective import quartet_loss_fn
    from heracles.embedding.sampler import _stratified_quartets
    from heracles.embedding.priors import build_allele_priors
    st = _toy_states(n=40, S=12, K=4, seed=21, missing_frac=0.1)
    est = seed_distances(st)
    prior = build_allele_priors(st[1:], prior="global_uniform")
    # root_weight fixed so micro and full batches optimise the same objective
    lf = quartet_loss_fn(st, prior, st.shape[0], est_dists=est, batch_size=100, root_weight=0.3)
    manifold, _ = lorentz_manifold(6)
    pts0, _ = init_points(manifold, 6, seed_dists=est, init_method="mds")
    g = torch.Generator().manual_seed(5)
    Q = _stratified_quartets(40, 200, generator=g)
    rad = torch.tensor(1.0, dtype=torch.float64); mu = torch.tensor(1.0, dtype=torch.float64)
    eps = torch.tensor(0.01, dtype=torch.float64); dl = torch.tensor(0.05, dtype=torch.float64)

    def grad_of(batches):
        p = pts0.clone().requires_grad_(True)
        total = 0.0
        for q in batches:
            loss, _ = lf(p, manifold, rad, mu, eps, dl, None, eval_batch=q)
            (loss / len(batches)).backward()
            total += loss.item() / len(batches)
        return total, p.grad.clone()
    l_full, g_full = grad_of([Q])
    l_half, g_half = grad_of([Q[:100], Q[100:]])
    assert abs(l_full - l_half) < 1e-10
    torch.testing.assert_close(g_full, g_half, rtol=1e-9, atol=1e-12)


def test_grad_accum_fit_runs_and_records_full_batch(tmp_path):
    cm = _cm(n=50, seed=12)
    m = Heracles(outdir=tmp_path / "ga", readouts=["nj"], dimension=6, max_iterations=6,
                 convergence_thresh=None, allele_prior="global_uniform", grad_accum=3).fit(cm)
    assert len(m.history_["loss"]) == 6 and np.isfinite(m.history_["loss"]).all()
    # 15 quartets per cell = 750 per step, drawn as 3 micro-batches of 250
    cfg = json.loads((tmp_path / "ga" / "config.json").read_text())["params"]
    assert cfg["grad_accum"] == 3
    assert m.config.grad_accum == 3


def test_condensed_weighted_hamming_matches_squareform():
    st = _toy_states(n=70, S=10, seed=13, missing_frac=0.1)
    full = seed_distances(st)
    np.testing.assert_array_equal(condensed_weighted_hamming(st, chunk=9), squareform(full, checks=False))


def test_upgma_landmark_dists_matches_full_tree_rows():
    from scipy.cluster.hierarchy import linkage, to_tree
    from heracles.trees.utils import linkage_to_newick
    st = _toy_states(n=90, S=15, seed=14, missing_frac=0.1)
    labels = ["t%d" % i for i in range(st.shape[0])]
    full = seed_distances(st)
    nwk = linkage_to_newick(to_tree(linkage(squareform(full, checks=False), "average")), labels)
    tree = parse_newick(nwk + ";", rooting="force-rooted")
    ref = tree_distances(tree, labels)
    lm = np.array([0, 3, 40, 89])
    block, nwk_lm, _ = tree_init_dists(st, labels, method="upgma", rows=lm)
    assert robinson_foulds(parse_newick(nwk_lm), tree) == 0
    np.testing.assert_array_equal(block, ref[lm, :])


def test_low_memory_upgma_init_fit(tmp_path):
    cm = _cm(n=45, seed=15)
    m = Heracles(outdir=tmp_path / "up", readouts=["nj"], low_memory=True, init_method="upgma",
                 dimension=6, max_iterations=8, convergence_thresh=None,
                 allele_prior="global_uniform").fit(cm)
    assert m.diagnostics_["init_method_used"] == "tree+landmark_mds"
    assert m.diagnostics_["init_tree_newick"] and m.dimension_ == 6


@pytest.mark.parametrize("n,S,K,miss,k", [(300, 20, 5, 0.1, 40), (500, 6, 2, 0.3, 64),
                                           (200, 60, 30, 0.0, 199), (120, 10, 3, 0.5, 20)])
def test_knn_table_matmul_matches_numpy_reference(n, S, K, miss, k):
    from heracles.embedding.sampler import _build_tau_neighbours_from_cm_numpy
    st = _toy_states(n=n, S=S, K=K, seed=n + S, missing_frac=miss)[1:]
    o1, t1, d1 = _build_tau_neighbours_from_cm_numpy(st, k)
    o2, t2, d2 = build_tau_neighbours_from_cm(st, k, row_chunk=77)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(d1, d2)


def test_knn_table_matmul_faster_than_numpy_at_5000():
    import time
    from heracles.embedding.sampler import _build_tau_neighbours_from_cm_numpy
    st = _toy_states(n=5000, S=60, K=30, seed=3, missing_frac=0.2)[1:]
    t = time.perf_counter(); build_tau_neighbours_from_cm(st, 512); t_matmul = time.perf_counter() - t
    t = time.perf_counter(); _build_tau_neighbours_from_cm_numpy(st, 512); t_numpy = time.perf_counter() - t
    assert t_matmul < t_numpy
