"""Starting coordinates for the embedding.

:func:`init_points` is the entry point.  It computes weighted-Hamming seed
distances between cells (:func:`seed_distances`), by default builds a tree on
them (:func:`tree_init_dists`) and embeds its branch-length distances with
classical MDS, or with landmark MDS (:func:`landmark_mds_from_rows`) so that
no ``n x n`` matrix is built.  A starting tree given as
``Heracles(init_tree=...)`` is read by :func:`load_init_tree_dists`.
"""

import logging
import os
import time as _time

import numpy as np
import torch
import geoopt

from heracles.constants import HAMMING_ROW_CHUNK
from heracles.trees.rapidnj import rapidnj_available, rapidnj_executable
from heracles.trees.tree import Taxon
from heracles.trees.utils import (
    FOUNDER_LABEL, as_newick, nj_tree, parse_newick, prune_founder_newick, tree_distances,
)

logger = logging.getLogger("heracles")


__all__ = [
    'classical_mds',
    'landmark_mds_from_rows',
    'default_n_landmarks',
    'choose_landmarks',
    'lorentz_manifold',
    'seed_distances',
    'iter_weighted_hamming_rows',
    'condensed_weighted_hamming',
    'tree_init_dists',
    'TREE_INIT_METHODS',
    'INIT_METHODS',
    'init_points',
    'is_newick_string',
    'load_init_tree_dists',
]


#: scale of the tangent-space coordinates the MDS solution is mapped to
#: before projection onto the Lorentz model
_TANGENT_INIT_SCALE = 0.3


def _mds_factor(dists):
    """Eigendecomposition ``(w, v)`` of the double-centred Gram matrix, largest first."""
    n = dists.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * H.dot(dists ** 2).dot(H)
    w, v = np.linalg.eigh(b)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def _mds_from_factor(w, v, dim):
    """``(n, dim)`` MDS coordinates from ``_mds_factor``; zero-padded if ``dim > n``."""
    n, k = v.shape
    d_use = min(dim, k)
    coords = v[:, :d_use] * np.sqrt(np.clip(w[:d_use], 0.0, None))
    if d_use < dim:
        coords = np.hstack([coords, np.zeros((n, dim - d_use), dtype=coords.dtype)])
    return coords


def classical_mds(dists, dim):
    """Places points in flat space so their distances match *dists* (classical MDS).

    Takes an ``(n, n)`` symmetric *dists*; returns ``(n, dim)`` coordinates,
    with columns beyond ``n`` set to zero.
    """
    w, v = _mds_factor(dists)
    return _mds_from_factor(w, v, dim)


# Landmark MDS: eigenvalues below this fraction of the leading one are noise
# (the Gram matrix of k landmarks has rank <= k-1) and get zero coordinates
# instead of the 1/sqrt(w) amplification the Nystrom step would apply.
_LANDMARK_EIG_TOL = 1e-8


def default_n_landmarks(dim, floor=500):
    """Returns the landmark count for embedding dimension *dim*: ``max(floor, 2*dim + 1)``.

    Landmark MDS needs more landmarks than dimensions.
    """
    return max(int(floor), 2 * int(dim) + 1)


def choose_landmarks(n, n_landmarks, seed=0, always=(0,)):
    """Returns sorted landmark indices among *n* points.

    The indices in ``always`` (default: the founder, row 0) are always
    included; the rest are drawn uniformly without replacement using *seed*.
    """
    always = [i for i in always if 0 <= i < n]
    n_landmarks = min(int(n_landmarks), n)
    rng = np.random.RandomState(seed)
    pool = np.setdiff1d(np.arange(n), np.asarray(always, dtype=np.int64))
    extra = rng.choice(pool, max(n_landmarks - len(always), 0), replace=False)
    return np.sort(np.concatenate([np.asarray(always, dtype=np.int64), extra]))


def landmark_mds_from_rows(D_lx, landmarks, dim, eig_tol=_LANDMARK_EIG_TOL):
    """Embeds all points by landmark MDS, using only the landmark rows of the distances.

    Args:
        D_lx: ``(k, n)`` distances from the *k* landmarks to all *n* points.
        landmarks: ``(k,)`` indices of the landmarks, in the row order of ``D_lx``.
        eig_tol: Dimensions whose eigenvalue is below ``eig_tol`` times the
            largest (or is not positive) get zero coordinates.

    Returns ``(n, dim)`` coordinates.  Raises ``ValueError`` unless ``k > dim``
    and *landmarks* matches the rows of ``D_lx``.
    """
    D_lx = np.asarray(D_lx, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.int64)
    k, n = D_lx.shape
    if landmarks.shape != (k,):
        raise ValueError("landmarks must have one index per row of D_lx: %r vs %d rows"
                         % (landmarks.shape, k))
    if k <= dim:
        raise ValueError(
            "landmark MDS needs more landmarks than dimensions (rank of the "
            "landmark Gram matrix is at most k-1): k=%d, dim=%d" % (k, dim))
    D_ll = D_lx[:, landmarks]
    w, v = _mds_factor(D_ll)
    w_d = w[:dim]
    v_d = v[:, :dim]
    keep = w_d > max(eig_tol * float(w[0]), 0.0)

    D_ll_sq = D_ll ** 2
    col_mean = D_ll_sq.mean(axis=0)
    B = -0.5 * (D_lx ** 2 - col_mean[:, None])

    inv_sqrt_w = np.zeros(dim, dtype=np.float64)
    inv_sqrt_w[keep] = 1.0 / np.sqrt(w_d[keep])
    coords = (inv_sqrt_w[:, None] * (v_d.T @ B)).T
    if not keep.all():
        logger.info("landmark MDS: %d of %d dimensions below the eigenvalue "
                    "floor, zeroed", int((~keep).sum()), dim)
    return coords


def lorentz_manifold(dimension):
    """Returns ``(manifold, dimension + 1)``: the Lorentz model of curvature -1."""
    return geoopt.Lorentz(k=1.0), dimension + 1


def seed_distances(states, rows=None, *, missing_state=-1, dtype=np.float64):
    """Computes the weighted-Hamming dissimilarity between cells.

    Sites missing in either cell are skipped.  Each remaining site adds 0 if
    the states agree, 1 if exactly one cell is unedited, and 2 if both are
    edited to different alleles; the sum is divided by the number of sites
    present in both cells.

    Args:
        states: ``(n, S)`` integer state matrix (include the founder row if wanted).
        rows: ``(k,)`` row indices, or None for the full matrix.

    Returns:
        The ``(k, n)`` block ``out[a, j] = d(rows[a], j)``, or the symmetric
        ``(n, n)`` matrix with a zero diagonal.
    """
    mats = np.asarray(states)
    n = mats.shape[0]
    if rows is None:
        out = np.zeros((n, n), dtype=dtype)
        for i0 in range(0, n, HAMMING_ROW_CHUNK):
            i1 = min(i0 + HAMMING_ROW_CHUNK, n)
            block = _hamming_block(mats, np.arange(i0, i1), mats[i0:], missing_state)
            out[i0:i1, i0:] = block
            out[i0:, i0:i1] = block.T
        np.fill_diagonal(out, 0.0)
        return out
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros((len(rows), n), dtype=dtype)
    for a0 in range(0, len(rows), HAMMING_ROW_CHUNK):
        a1 = min(a0 + HAMMING_ROW_CHUNK, len(rows))
        out[a0:a1] = _hamming_block(mats, rows[a0:a1], mats, missing_state)
    out[np.arange(len(rows)), rows] = 0.0
    return out


def _hamming_block(mats, rows, others, missing_state):
    """``(len(rows), len(others))`` weighted-Hamming block (see :func:`seed_distances`)."""
    mi = mats[rows, None, :]
    mj = others[None, :, :]
    valid = (mi != missing_state) & (mj != missing_state)
    num_present = valid.sum(axis=2).astype(np.float64)
    differ = valid & (mi != mj)
    one_unedited = differ & ((mi == 0) | (mj == 0))
    both_edited = differ & ~one_unedited
    d = one_unedited.sum(axis=2).astype(np.float64) \
        + 2.0 * both_edited.sum(axis=2).astype(np.float64)
    block = np.zeros_like(d)
    np.divide(d, num_present, out=block, where=num_present > 0)
    return block


def iter_weighted_hamming_rows(mats, missing_state=-1, chunk=HAMMING_ROW_CHUNK):
    """Yields ``(i0, i1, block)`` row blocks of :func:`seed_distances`.

    Lets the matrix be streamed (e.g. to RapidNJ's input file) without building it.
    """
    mats = np.asarray(mats)
    n = mats.shape[0]
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        yield i0, i1, seed_distances(mats, np.arange(i0, i1), missing_state=missing_state)


def condensed_weighted_hamming(mats, missing_state=-1, chunk=HAMMING_ROW_CHUNK):
    """Returns :func:`seed_distances` as scipy's condensed ``n(n-1)/2`` float64 vector.

    Filled one row block at a time, so no ``n x n`` array is built.
    """
    mats = np.asarray(mats)
    n = mats.shape[0]
    cond = np.empty(n * (n - 1) // 2, dtype=np.float64)
    for i0, i1, block in iter_weighted_hamming_rows(mats, missing_state, chunk):
        for r in range(i1 - i0):
            i = i0 + r
            if i >= n - 1:
                break
            start = n * i - i * (i + 1) // 2
            cond[start:start + (n - 1 - i)] = block[r, i + 1:]
    return cond


TREE_INIT_METHODS = ("rapidnj", "nj", "upgma")


def tree_init_dists(states, labels, *, method, rows=None, seed_dists=None, missing_state=-1):
    """Builds a tree on the seed distances and returns its distances along the tree.

    Negative branch lengths are set to 0; distances sum branch lengths.

    Args:
        states: ``(n, S)`` state matrix, used when *seed_dists* is None.
        method: ``"rapidnj"`` (RapidNJ binary), ``"nj"`` (pure-Python neighbour
            joining) or ``"upgma"`` (average linkage); others raise ``ValueError``.
        rows: Optional row indices; only those rows are returned (landmark mode).
        seed_dists: Optional ``(n, n)`` seed matrix.  Without it the seed
            distances are streamed from *states* and, for ``"rapidnj"`` and
            ``"upgma"``, no ``n x n`` array is built in this process.

    Returns:
        ``(dists, tree_newick, timings)``, *dists* ``(n, n)`` or ``(len(rows), n)``.
    """
    from heracles.trees.utils import linkage_to_newick, nj_tree_from_rows
    if method not in TREE_INIT_METHODS:
        raise ValueError("method must be one of %s, got %r" % (TREE_INIT_METHODS, method))
    labels = list(labels)
    n = len(labels)
    t0 = _time.perf_counter()
    if method == "rapidnj":
        blocks = (((0, n, seed_dists),) if seed_dists is not None
                  else iter_weighted_hamming_rows(states, missing_state))
        tree = nj_tree_from_rows(blocks, labels, n)
    elif method == "nj":
        full = seed_dists if seed_dists is not None else seed_distances(states, missing_state=missing_state)
        tree = nj_tree(full, labels, rapidnj=False)
    else:
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform
        cond = (squareform(np.asarray(seed_dists, dtype=np.float64), checks=False)
                if seed_dists is not None else condensed_weighted_hamming(states, missing_state))
        Z = linkage(cond, method="average")
        del cond
        tree = parse_newick(linkage_to_newick(to_tree(Z), labels) + ";", rooting="force-rooted")
        del Z
    for edge in tree.preorder_edge_iter():
        if edge.length is not None and edge.length < 0:
            edge.length = 0.0
    tree_creation_s = _time.perf_counter() - t0
    tree_nwk = as_newick(tree, rooted=False)
    t1 = _time.perf_counter()
    dists = tree_distances(tree, labels, weighted=True,
                           rows=None if rows is None else np.asarray(rows))
    return dists, tree_nwk, {"tree_creation_s": tree_creation_s,
                             "distance_extraction_s": _time.perf_counter() - t1}


INIT_METHODS = ("rapidnj_clamped", "upgma", "mds")


def init_points(manifold, dimension, *, states=None, seed_dists=None, labels=None,
                init_dists=None, init_method="rapidnj_clamped", landmarks=None,
                missing_state=-1):
    """Computes starting coordinates for the embedding.

    The distances to embed are *init_dists* if given, else *seed_dists*, else
    seed distances computed from *states*.  ``"rapidnj_clamped"`` and
    ``"upgma"`` first build a tree on them and embed its distances (see
    :func:`tree_init_dists`); ``"mds"`` embeds them directly.  Without a
    RapidNJ binary, ``"rapidnj_clamped"`` falls back to ``"mds"`` with a
    warning.

    Args:
        states: ``(m, S)`` state matrix, founder row included.
        labels: Optional m point names (needed to drop the founder from
            ``init_tree_newick``).
        init_method: ``"rapidnj_clamped"`` (default), ``"upgma"`` or ``"mds"``;
            others raise ``ValueError``.
        landmarks: Optional landmark indices; switches to landmark MDS so no
            ``m x m`` matrix is built.

    Returns:
        ``(points, init_meta)``: ``(m, dimension + 1)`` torch tensor on the
        manifold, and a dict with the timings, ``init_tree_newick`` (or None),
        ``init_method_used``, ``init_dimension`` and ``n_landmarks``.
    """
    t_total_start = _time.perf_counter()
    if init_method not in INIT_METHODS:
        raise ValueError("unknown init_method %r (%s)" % (init_method, "|".join(INIT_METHODS)))
    source = init_dists if init_dists is not None else seed_dists
    if source is None and states is None:
        raise ValueError("init_points needs states, seed_dists or init_dists")
    n_pts = int(source.shape[0] if source is not None else np.asarray(states).shape[0])
    if labels is None:
        labels = ["_t%d" % i for i in range(n_pts)]
    if landmarks is not None:
        landmarks = np.asarray(landmarks, dtype=np.int64)

    tree_method = None
    if init_method == "rapidnj_clamped":
        if rapidnj_available():
            tree_method = "rapidnj"
            logger.info("init backend: rapidnj (%s)", rapidnj_executable())
        else:
            logger.warning(
                "RapidNJ not available; falling back to init_method='mds'. Build the "
                "vendored binary or set HERACLES_RAPIDNJ for the tree-based initialisation.")
            init_method = "mds"
    elif init_method == "upgma":
        tree_method = "upgma"

    tree_nwk, tree_timings = None, {}
    if tree_method is not None:
        D, tree_nwk, tree_timings = tree_init_dists(
            states, labels, method=tree_method, rows=landmarks,
            seed_dists=source, missing_state=missing_state)
        method_used = init_method if landmarks is None else "tree+landmark_mds"
    elif landmarks is None:
        D = source if source is not None else seed_distances(states, missing_state=missing_state)
        method_used = "mds"
    else:
        D = (np.asarray(source)[landmarks, :] if source is not None
             else seed_distances(states, landmarks, missing_state=missing_state))
        method_used = "landmark_mds"

    t_emb = _time.perf_counter()
    if landmarks is None:
        coords = classical_mds(np.asarray(D, dtype=np.float64), dimension)
    else:
        coords = landmark_mds_from_rows(D, landmarks, dimension)
    if coords.shape != (n_pts, dimension):
        raise RuntimeError(
            "initialisation produced coordinates of shape %r, expected (%d, %d): "
            "the embedding would silently run at the wrong dimension"
            % (coords.shape, n_pts, dimension))
    coords = coords / (np.abs(coords).max() + 1e-9) * _TANGENT_INIT_SCALE
    tangent = torch.tensor(coords, dtype=torch.float64)
    zeros = torch.zeros(tangent.shape[0], 1, dtype=torch.float64)
    result = manifold.expmap0(torch.cat([zeros, tangent], dim=1))
    embedding_s = _time.perf_counter() - t_emb

    if tree_nwk is not None and FOUNDER_LABEL in labels:
        tree_nwk = prune_founder_newick(tree_nwk, FOUNDER_LABEL)
    return result, {
        "tree_creation_s": tree_timings.get("tree_creation_s", 0.0),
        "distance_extraction_s": tree_timings.get("distance_extraction_s", 0.0),
        "embedding_s": embedding_s,
        "total_s": _time.perf_counter() - t_total_start,
        "init_tree_newick": tree_nwk,
        "init_method_used": method_used,
        "init_dimension": int(coords.shape[1]),
        "n_landmarks": None if landmarks is None else int(len(landmarks)),
    }


def is_newick_string(text):
    """True when *text* is a Newick tree rather than a file path."""
    t = str(text).strip()
    return t.endswith(";") or t.startswith(("(", "["))


def load_init_tree_dists(tree, labels):
    """Returns the distances along a starting tree, for ``Heracles(init_tree=...)``.

    *tree* is a Newick path or string containing every label (``ValueError``
    otherwise); negative branch lengths are set to 0.  Returns ``(n+1, n+1)``
    distances ordered ``[FOUNDER_LABEL] + labels``; the founder is the tree's
    ``__root__`` leaf if present, else its root.
    """
    tree = os.fspath(tree)
    if is_newick_string(tree):
        tree_nwk, source = tree.strip(), "a Newick string"
    else:
        with open(tree) as f:
            tree_nwk, source = f.read().strip(), tree

    tree = parse_newick(tree_nwk)
    for edge in tree.preorder_edge_iter():
        if edge.length is not None and edge.length < 0:
            edge.length = 0.0
    founder_is_leaf = any(
        t.label == FOUNDER_LABEL for t in tree.taxon_namespace)
    if not founder_is_leaf:
        tree.seed_node.taxon = Taxon(label=FOUNDER_LABEL)
    missing = sorted(set(labels) - {t.label for t in tree.taxon_namespace})
    if missing:
        raise ValueError("init_tree lacks %d of the cells, e.g. %s" % (len(missing), missing[:3]))
    dists = tree_distances(tree, [FOUNDER_LABEL] + list(labels), weighted=True)
    logger.info("init tree from %s (%d leaves, founder=%s)", source, len(labels),
                "__root__ leaf" if founder_is_leaf else "root node")
    return dists
