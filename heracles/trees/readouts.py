"""Tree read-outs: build a rooted tree from the fitted embedding distances.

Every read-out takes the dict returned by ``heracles.embed`` (``full_dists``
with the founder at row 0, or ``coords`` for a low-memory fit, and ``labels``)
and returns a tree rooted at the founder with the founder leaf pruned.
:data:`READOUTS` maps the names ``"nj"`` and ``"upgma"`` to them.

The neighbour-joining and Newick helpers used here are in
:mod:`heracles.trees.utils`.
"""

import logging
import warnings

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree

from heracles.constants import DIST_ROW_CHUNK, PHYLIP_BYTES_PER_ENTRY
from heracles.trees.rapidnj import rapidnj_available
from heracles.embedding.optimize import iter_dist_rows, pairwise_dists
from heracles.trees.tree import TaxonNamespace
from heracles.trees.utils import (
    FOUNDER_LABEL, linkage_to_newick, nj_tree, nj_tree_from_rows, parse_newick,
    root_on_founder,
)

logger = logging.getLogger("heracles")


__all__ = [
    'heracles_nj',
    'upgma_condensed_height',
    'heracles_upgma',
    'READOUTS',
]


def _row_blocks(emb, chunk=DIST_ROW_CHUNK):
    """``(i0, i1, block)`` rows of the ``(n+1, n+1)`` distance matrix, from the
    fitted matrix or streamed from the coordinates (low-memory fits)."""
    full = emb.get("full_dists")
    if full is not None:
        yield 0, full.shape[0], np.asarray(full)
        return
    yield from iter_dist_rows(emb["coords"], chunk=chunk, device=emb.get("device"))


def _n_points(emb):
    """Number of points including the founder."""
    full = emb.get("full_dists")
    return int(full.shape[0] if full is not None else emb["coords"].shape[0])


def _full_dists(emb):
    """The ``(n+1, n+1)`` matrix: as fitted, or built from the coordinates when
    the fit ran in low-memory mode (``full_dists`` is then ``None``)."""
    if emb.get("full_dists") is not None:
        return emb["full_dists"]
    m = _n_points(emb)
    logger.info("read-out: building the %d x %d distance matrix from the coordinates", m, m)
    return pairwise_dists(emb["coords"], dtype=emb.get("dist_dtype", np.float64),
                          device=emb.get("device"))


def heracles_nj(emb):
    """Neighbour joining on the embedding distances.

    Runs RapidNJ (:mod:`heracles.trees.rapidnj`), streaming distance rows to its input
    file so a low-memory fit holds no n x n matrix in Python (the RapidNJ
    process itself needs about 8 n^2 bytes).  Without a RapidNJ
    binary it warns and falls back to the pure-Python NJ, which gives the same
    kind of tree but needs the full matrix and O(n^3) time.
    """
    labels = emb["labels"]
    m = _n_points(emb)
    if not rapidnj_available():
        warnings.warn("RapidNJ is not available (it needs a C++ compiler at install or "
                      "first use); the nj read-out falls back to the pure-Python neighbour "
                      "joining, which is much slower for large n", stacklevel=2)
        tree = nj_tree(_full_dists(emb), [FOUNDER_LABEL] + labels, rapidnj=False)
        return root_on_founder(tree, labels)
    if emb.get("full_dists") is None:
        logger.info("read-out nj: streaming a %d x %d PHYLIP matrix to RapidNJ (~%.0f GB on disk)",
                    m, m, m * m * PHYLIP_BYTES_PER_ENTRY / 1024 ** 3)
    tree = nj_tree_from_rows(_row_blocks(emb), [FOUNDER_LABEL] + labels, m)
    return root_on_founder(tree, labels)


def upgma_condensed_height(row_blocks, n_points):
    """Condensed UPGMA input built from distance row blocks, without an n x n matrix.

    Args:
        row_blocks: ``(i0, i1, block)`` rows of the ``(n+1, n+1)`` matrix, founder first.
        n_points (int): matrix size ``n+1``, founder included.

    Returns:
        np.ndarray: condensed ``n(n-1)/2`` float64 vector of ``tau_max - tau_ij``,
        where ``tau_ij = (d(0,i) + d(0,j) - d(i,j)) / 2`` (clamped at 0) is the
        depth of the common ancestor of leaves i and j.
    """
    n = n_points - 1
    cond = np.empty(n * (n - 1) // 2, dtype=np.float64)
    d0 = None
    for i0, i1, block in row_blocks:
        block = np.asarray(block)
        if i0 == 0:
            d0 = block[0, 1:].astype(np.float64)       # founder row: depths
        for r in range(i1 - i0):
            i = i0 + r - 1                              # leaf index (founder excluded)
            if i < 0 or i >= n - 1:
                continue
            tau = 0.5 * (d0[i] + d0[i + 1:] - block[r, i + 2:])
            np.clip(tau, 0.0, None, out=tau)
            start = n * i - i * (i + 1) // 2
            cond[start:start + (n - 1 - i)] = tau
    tau_max = float(cond.max()) if cond.size else 0.0
    np.subtract(tau_max, cond, out=cond)
    return cond


def heracles_upgma(emb):
    """UPGMA (average linkage) on the common-ancestor depths from the founder.

    Uses :func:`upgma_condensed_height`; the tree is founder-rooted by construction.
    """
    labels = emb["labels"]
    cond = upgma_condensed_height(_row_blocks(emb), _n_points(emb))
    Z = linkage(cond, method="average")
    nwk = "[&R] " + linkage_to_newick(to_tree(Z), labels) + ";"
    return parse_newick(nwk, taxon_namespace=TaxonNamespace(labels),
                        rooting="force-rooted")


READOUTS = {"nj": heracles_nj, "upgma": heracles_upgma}
