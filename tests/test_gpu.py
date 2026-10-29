"""CPU/GPU equivalence tests. Skipped automatically when CUDA is unavailable.

Sampling always runs on CPU (see ``sample_distinct``), so a given seed
draws identical batches on every device; the only CPU-vs-GPU differences
are floating-point reduction orders, which stay near machine epsilon over
short fits.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from heracles.fit import Heracles

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="CUDA not available")


def _cm(n=60, S=20, K=8, missing=0.15, seed=3):
    rng = np.random.default_rng(seed)
    cm = pd.DataFrame(rng.integers(0, K + 1, (n, S)),
                      index=[f"c{i}" for i in range(n)])
    cm.iloc[rng.random(cm.shape) < missing] = -1
    return cm


def _fit(device, outdir, cm, steps=30):
    m = Heracles(dimension=6, max_iterations=steps, convergence_thresh=None,
                 seed=0, init_method="mds", allele_prior="global_uniform",
                 learn_heritable_eps=True, learn_stochastic_delta=True,
                 learn_allele_prior=True,
                 ultrametric_penalty="adaptive",
                 device=device, outdir=outdir)
    m.fit(cm)
    return m


@cuda
def test_cpu_gpu_same_seed_equivalence(tmp_path):
    cm = _cm()
    a = _fit("cpu", tmp_path / "cpu", cm)
    b = _fit("cuda", tmp_path / "cuda", cm)
    la, lb = np.array(a.history_["loss"]), np.array(b.history_["loss"])
    assert la.shape == lb.shape
    rel = np.max(np.abs(la - lb) / np.maximum(np.abs(la), 1e-30))
    assert rel < 1e-8, f"CPU/GPU same-seed loss curves diverge: {rel:.3e}"
    np.testing.assert_allclose(a.leaf_dists_, b.leaf_dists_,
                               rtol=1e-6, atol=1e-8)


@cuda
def test_gpu_short_fit_runs(tmp_path):
    cm = _cm()
    m = Heracles(dimension=6, max_iterations=10,
                 convergence_thresh=None, seed=0, init_method="mds",
                 allele_prior="global_uniform",
                 device="cuda", outdir=tmp_path)
    m.fit(cm)
    assert np.isfinite(m.history_["loss"]).all()
