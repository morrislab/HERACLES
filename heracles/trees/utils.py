"""Tree helpers shared by initialisation, read-outs and the polish.

Newick reading and writing, rooting on the founder cell (:data:`FOUNDER_LABEL`),
all-pairs distances along a tree, neighbour joining from a distance matrix
(RapidNJ or native Python), scipy linkage to Newick, and the Robinson-Foulds
distance (:func:`robinson_foulds`).

Where tree code lives: :mod:`heracles.trees.tree` is the data structure and its
Newick format; :mod:`heracles.trees.utils` holds what you do with trees (read,
write, root, compare with :func:`~heracles.trees.utils.robinson_foulds`, build
by neighbour joining); :mod:`heracles.trees.readouts` turns a fitted embedding into
a tree; :mod:`heracles.trees.rapidnj` finds or builds the RapidNJ binary.  Start
from :mod:`heracles.trees.utils` unless you are working on the data structure.
"""

import logging
import os
import tempfile

import numpy as np

from heracles.constants import PHYLIP_BYTES_PER_ENTRY
from heracles.trees.tree import (
    quote_label,
    TaxonNamespace, Tree,
    native_nj,
    symmetric_difference as _symmetric_difference,
)

logger = logging.getLogger("heracles")


__all__ = [
    'FOUNDER_LABEL',
    'quote_label',
    'as_newick',
    'write_newick',
    'parse_newick',
    'copy_tree',
    'linkage_to_newick',
    'root_on_founder',
    'prune_founder_newick',
    'tree_distances',
    'write_phylip',
    'write_phylip_rows',
    'nj_tree_from_rows',
    'nj_tree',
    'robinson_foulds',
]


FOUNDER_LABEL = "__root__"        # synthetic all-unedited founder cell, always taxon 0


# Newick

def as_newick(tree, *, rooted=True, internal_labels=False):
    """Newick string of *tree*; internal node labels only if ``internal_labels=True``."""
    return tree.as_string(schema="newick", suppress_rooting=not rooted,
                          suppress_internal_node_labels=not internal_labels).strip()


def write_newick(path, tree):
    """Write ``as_newick(tree)`` (terminated with ``;``) to *path*."""
    nwk = as_newick(tree)
    if not nwk.endswith(";"):
        nwk += ";"
    with open(path, "w") as f:
        f.write(nwk + "\n")


def parse_newick(data, **kwargs):
    """Parse a Newick string into a :class:`~heracles.trees.tree.Tree`.

    Keyword arguments go to :meth:`~heracles.trees.tree.Tree.get` (for example
    ``taxon_namespace``, ``rooting``). Deep trees are fine (no recursion limit).
    """
    return Tree.get(data=data, schema="newick", **kwargs)


def copy_tree(tree):
    """Deep copy that keeps the taxon namespace (so RF comparisons still work)."""
    return tree.taxon_namespace_scoped_copy()


def linkage_to_newick(node, labels):
    """Newick (no trailing ``;``) from a scipy ``to_tree`` cluster node.

    Branch lengths are parent height minus child height, clamped at 0.
    """
    done = {}
    stack = [(node, False)]
    while stack:
        nd, expanded = stack.pop()
        if nd.is_leaf():
            done[nd.id] = quote_label(labels[nd.id])
        elif not expanded:
            stack.append((nd, True))
            stack.append((nd.right, False))
            stack.append((nd.left, False))
        else:
            left = done.pop(nd.left.id)
            right = done.pop(nd.right.id)
            done[nd.id] = "(%s:%.10g,%s:%.10g)" % (
                left, max(0.0, nd.dist - nd.left.dist),
                right, max(0.0, nd.dist - nd.right.dist))
    return done[node.id]


# Founder rooting

def _reroot_on_founder(tree, founder_label):
    """Root *tree* on the founder leaf and prune it, in place; False if no founder."""
    founder = tree.find_node_with_taxon_label(founder_label)
    if founder is None:
        return False
    tree.to_outgroup_position(founder, update_bipartitions=False)
    tree.is_rooted = True
    tree.prune_taxa_with_labels([founder_label], suppress_unifurcations=True)
    return True


def root_on_founder(tree, labels, founder_label=FOUNDER_LABEL):
    """Root *tree* on the founder leaf, prune it, and return a rooted tree over *labels*.

    *tree* is modified. *labels* are the leaf labels without the founder.
    """
    _reroot_on_founder(tree, founder_label)
    return parse_newick(as_newick(tree), taxon_namespace=TaxonNamespace(labels),
                        rooting="force-rooted")


def prune_founder_newick(tree_nwk, founder_label=FOUNDER_LABEL):
    """Newick rooted on the founder leaf with it pruned; unchanged if there is no founder."""
    tree = parse_newick(tree_nwk)
    return as_newick(tree) if _reroot_on_founder(tree, founder_label) else tree_nwk


# Distances and NJ

def tree_distances(tree, labels, weighted=True, rows=None):
    """All-pairs distances along *tree* between *labels*.

    Args:
        tree (Tree): tree whose leaves include *labels*.
        labels (list[str]): row and column order of the result.
        weighted (bool): sum branch lengths (True) or count edges (False).
        rows (array of int, optional): indices into *labels*; returns only those
            rows, without building the full matrix.

    Returns:
        np.ndarray: ``(n, n)`` matrix, or ``(len(rows), n)`` with *rows*. The
        summation order is fixed for reproducibility, so the matrix can differ
        from its transpose in the last bit.
    """
    nodes = list(tree.preorder_node_iter())
    idx = {id(nd): i for i, nd in enumerate(nodes)}
    n_nodes = len(nodes)
    parent = np.full(n_nodes, -1, dtype=np.int64)
    w = np.zeros(n_nodes, dtype=np.float64)
    children = [[] for _ in range(n_nodes)]
    rank = np.full(n_nodes, -1, dtype=np.int64)    # preorder rank among taxon-bearing nodes
    taxon_labels = []
    for i, nd in enumerate(nodes):
        if nd.parent_node is not None:
            p = idx[id(nd.parent_node)]
            parent[i] = p
            children[p].append(i)
            w[i] = (float(nd.edge_length)
                    if (weighted and nd.edge_length is not None) else 1.0)
        if nd.taxon is not None:
            rank[i] = len(taxon_labels)
            taxon_labels.append(nd.taxon.label)
    n_taxa = len(taxon_labels)

    # taxa of subtree(i) occupy ranks [lo[i], hi[i]) (preorder is contiguous)
    lo = np.zeros(n_nodes, dtype=np.int64)
    hi = np.zeros(n_nodes, dtype=np.int64)
    count = 0
    for i in range(n_nodes):
        lo[i] = count
        count += rank[i] >= 0
    for i in range(n_nodes - 1, -1, -1):
        h = lo[i] + (1 if rank[i] >= 0 else 0)
        for c in children[i]:
            h = max(h, hi[c])
        hi[i] = h

    # up pass: up[i][k] = distance from the taxon at rank lo[i]+k to node i
    up = [None] * n_nodes
    for i in range(n_nodes - 1, -1, -1):
        parts = []
        if rank[i] >= 0:
            parts.append(np.zeros(1, dtype=np.float64))
        for c in children[i]:
            parts.append(up[c] + w[c])
        up[i] = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    # down pass: col[i][r] = distance from the taxon at rank r to node i
    pos = {lab: r for r, lab in enumerate(taxon_labels)}
    order = np.array([pos[lbl] for lbl in labels])
    if rows is None:
        out = np.empty((n_taxa, n_taxa), dtype=np.float64)
    else:
        row_ranks = order[np.asarray(rows, dtype=np.int64)]
        out = np.empty((len(row_ranks), n_taxa), dtype=np.float64)
    col = [None] * n_nodes
    pending = [len(ch) for ch in children]
    for i in range(n_nodes):
        c = np.empty(n_taxa, dtype=np.float64)
        p = parent[i]
        if p < 0:
            c[:] = up[i]
        else:
            pc = col[p]
            c[:lo[i]] = pc[:lo[i]] + w[i]
            c[hi[i]:] = pc[hi[i]:] + w[i]
            c[lo[i]:hi[i]] = up[i]
            pending[p] -= 1
            if pending[p] == 0:
                col[p] = None
        up[i] = None
        if rank[i] >= 0:
            out[:, rank[i]] = c if rows is None else c[row_ranks]
        if children[i]:
            col[i] = c

    if rows is None:
        return out[np.ix_(order, order)]
    return out[:, order]


def write_phylip(dists, labels, path):
    """Write a square PHYLIP distance matrix (RapidNJ input), ``%.10g`` per entry."""
    n = dists.shape[0]
    write_phylip_rows(((0, n, dists),), labels, path, n)


def write_phylip_rows(row_blocks, labels, path, n):
    """:func:`write_phylip` from ``(i0, i1, block)`` row blocks of shape ``(i1-i0, n)``.

    The full matrix never has to be in memory; the output is identical.
    """
    row_fmt = " ".join(["%.10g"] * n)
    with open(path, "w", buffering=1 << 20) as f:
        f.write("%d\n" % n)
        for i0, i1, block in row_blocks:
            block = np.asarray(block)
            for r in range(i1 - i0):
                f.write("%s %s\n" % (labels[i0 + r], row_fmt % tuple(block[r].tolist())))


def nj_tree_from_rows(row_blocks, labels, n, *, timeout=None):
    """RapidNJ tree from streamed distance rows (see :func:`write_phylip_rows`).

    Needs the RapidNJ binary. The temporary PHYLIP file (about 11 bytes per entry)
    goes under ``$HERACLES_TMPDIR`` or the system temp dir and is removed afterwards.
    *timeout* is in seconds (None = no limit).
    """
    from heracles.trees.rapidnj import run_rapidnj
    tns = TaxonNamespace(labels)
    tmpdir = os.environ.get("HERACLES_TMPDIR") or None
    est_gb = n * n * PHYLIP_BYTES_PER_ENTRY / 1024 ** 3
    if tmpdir is None and est_gb > 10:
        logger.warning("RapidNJ input will be ~%.0f GB of PHYLIP text in the system temp dir "
                       "(%s); set HERACLES_TMPDIR to a large, node-independent disk",
                       est_gb, tempfile.gettempdir())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".phy", delete=False,
                                     dir=tmpdir) as f:
        tmp_path = f.name
    try:
        write_phylip_rows(row_blocks, labels, tmp_path, n)
        result = run_rapidnj(tmp_path, timeout=timeout)
    finally:
        os.unlink(tmp_path)
    return parse_newick(result.stdout.strip(), taxon_namespace=tns)


def nj_tree(dists, labels, *, rapidnj=None, timeout=None):
    """Neighbour-joining tree over *labels* from a distance matrix.

    Args:
        dists (np.ndarray): ``(n, n)`` distances in *labels* order.
        labels (list[str]): leaf labels.
        rapidnj (bool, optional): True requires RapidNJ, False uses the native
            Python NJ, None uses RapidNJ when available.
        timeout (float, optional): RapidNJ time limit in seconds.
    """
    from heracles.trees.rapidnj import rapidnj_available
    if rapidnj is None:
        rapidnj = rapidnj_available()
    if not rapidnj:
        return native_nj(dists, labels, taxon_namespace=TaxonNamespace(labels))
    n = np.asarray(dists).shape[0]
    return nj_tree_from_rows(((0, n, dists),), labels, n, timeout=timeout)


# Robinson-Foulds

def robinson_foulds(tree1, tree2, *, normalized=False):
    """Unrooted Robinson-Foulds distance between two trees over the same leaves.

    Args:
        tree1, tree2: :class:`~heracles.trees.tree.Tree` objects or Newick strings
            (not modified).
        normalized (bool): divide by the total number of internal edges of both
            trees (0 = identical topologies, 1 = no shared split).

    Returns:
        int or float: number of splits in one tree but not the other (not halved).
    """
    if isinstance(tree1, str) or isinstance(tree2, str):
        taxa = TaxonNamespace()
        tree1 = parse_newick(tree1, rooting="force-unrooted", taxon_namespace=taxa) \
            if isinstance(tree1, str) else tree1
        tree2 = parse_newick(tree2, rooting="force-unrooted", taxon_namespace=taxa) \
            if isinstance(tree2, str) else tree2
    tree1 = copy_tree(tree1)
    tree2 = copy_tree(tree2)
    tree1.deroot()
    tree2.deroot()
    rf = _symmetric_difference(tree1, tree2)
    if not normalized:
        return rf
    n_internal = (len(tree1.internal_edges(exclude_seed_edge=True))
                  + len(tree2.internal_edges(exclude_seed_edge=True)))
    return 0.0 if n_internal == 0 else rf / n_internal
