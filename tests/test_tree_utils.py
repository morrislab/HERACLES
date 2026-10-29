"""Tests for heracles.trees.utils."""

import numpy as np
import pytest

from heracles.trees.tree import Taxon, TaxonNamespace
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path as _sp_shortest_path

from heracles.trees.utils import robinson_foulds, tree_distances, parse_newick


# straightforward reference implementations the fast paths are checked against
def _tree_distances_shortest_path(tree, labels, weighted=True):
    """Reference ``tree_distances``: scipy shortest paths over all tree nodes
    (builds a dense (2n-1) x (2n-1) matrix)."""
    nodes = list(tree.preorder_node_iter())
    node_idx = {id(n): i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    rows, cols, data = [], [], []
    for node in nodes:
        if node.parent_node is not None:
            i = node_idx[id(node)]
            j = node_idx[id(node.parent_node)]
            w = (float(node.edge_length)
                 if (weighted and node.edge_length is not None) else 1.0)
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])

    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    all_dists = _sp_shortest_path(adj, directed=False, unweighted=(not weighted))

    label_to_node = {}
    for node in nodes:
        if node.taxon is not None:
            label_to_node[node.taxon.label] = node_idx[id(node)]
    leaf_idx = np.array([label_to_node[lbl] for lbl in labels])
    return all_dists[np.ix_(leaf_idx, leaf_idx)].astype(np.float64)


def _write_phylip_reference(dists, labels, path):
    """Reference PHYLIP writer, one formatted entry at a time."""
    n = dists.shape[0]
    with open(path, "w") as f:
        f.write("%d\n" % n)
        for i in range(n):
            f.write("%s %s\n" % (labels[i], " ".join("%.10g" % dists[i, j] for j in range(n))))


def test_robinson_foulds_identical():
    ns = TaxonNamespace(["a", "b", "c", "d"])
    t1 = parse_newick("((a,b),(c,d));", taxon_namespace=ns)
    t2 = parse_newick("((a,b),(c,d));", taxon_namespace=ns)
    assert robinson_foulds(t1, t2) == 0


def test_robinson_foulds_different():
    ns = TaxonNamespace(["a", "b", "c", "d"])
    t1 = parse_newick("((a,b),(c,d));", taxon_namespace=ns)
    t2 = parse_newick("((a,c),(b,d));", taxon_namespace=ns)
    assert robinson_foulds(t1, t2) == 2


def test_robinson_foulds_does_not_mutate():
    ns = TaxonNamespace(["a", "b", "c", "d"])
    t1 = parse_newick("((a,b),(c,d));", taxon_namespace=ns, rooting="force-rooted")
    t2 = parse_newick("((a,c),(b,d));", taxon_namespace=ns, rooting="force-rooted")
    robinson_foulds(t1, t2)
    assert t1.is_rooted
    assert t2.is_rooted


def test_tree_distances_match_reference():
    tree = parse_newick(
        "(((a:1.1, b:1):1, c:2):1, (d:2, (e:1,f:1):1):1):0;")
    labels = ["a", "b", "c", "d", "e", "f"]
    D = tree_distances(tree, labels, weighted=True)
    ref = _tree_distances_shortest_path(tree, labels, weighted=True)
    assert np.allclose(D, ref, atol=1e-9)


# deep trees, PHYLIP bytes, tree distances vs the shortest-path oracle
def _caterpillar(n, length=lambda i: 1.0, founder=None):
    s = "L0:%g" % length(0)
    for i in range(1, n):
        s = "(%s,L%d:%g):%g" % (s, i, length(i), length(i))
    if founder is not None:
        s = "(%s,'%s':1)" % (s, founder)
    return s + ";"


def test_parse_newick_deep_caterpillar_restores_limits():
    import sys
    import threading
    from heracles.trees.utils import FOUNDER_LABEL, as_newick, parse_newick, prune_founder_newick
    n = 5000
    limit, stack = sys.getrecursionlimit(), threading.stack_size()
    tree = parse_newick(_caterpillar(n, founder=FOUNDER_LABEL))
    assert len(tree.leaf_nodes()) == n + 1
    pruned = prune_founder_newick(as_newick(tree))
    assert FOUNDER_LABEL not in pruned
    assert robinson_foulds(pruned, pruned, normalized=True) == 0.0
    d = tree_distances(tree, [FOUNDER_LABEL] + ["L%d" % i for i in range(n)])
    assert d.shape == (n + 1, n + 1) and d[0, 1] == n + 1   # founder edge + the chain
    assert sys.getrecursionlimit() == limit and threading.stack_size() == stack


def test_parse_newick_reraises_worker_errors():
    from heracles.trees.utils import parse_newick
    with pytest.raises(Exception):
        parse_newick(_caterpillar(2000)[:-5] + ";")     # truncated string


def test_write_phylip_byte_identical_to_reference(tmp_path):
    from heracles.trees.utils import write_phylip
    rng = np.random.default_rng(0)
    sym = rng.random((300, 300))
    sym = (sym + sym.T) / 2
    np.fill_diagonal(sym, 0.0)
    edge = np.array([[-0.0, 1e-300, 1e300, 0.1 + 0.2, np.nan, np.inf,
                      -1.23456789012345, 7.0]] * 8)
    for k, arr in enumerate((sym, rng.random((40, 40)), edge)):
        labels = ["t%d" % i for i in range(arr.shape[0])]
        write_phylip(arr, labels, tmp_path / ("a%d" % k))
        _write_phylip_reference(arr, labels, tmp_path / ("b%d" % k))
        assert (tmp_path / ("a%d" % k)).read_bytes() == (tmp_path / ("b%d" % k)).read_bytes()


def _random_nj_tree(n, seed):
    from heracles.trees.utils import nj_tree
    rng = np.random.default_rng(seed)
    d = rng.random((n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    labels = ["c%d" % i for i in range(n)]
    tree = nj_tree(d, labels, rapidnj=False)
    for edge in tree.preorder_edge_iter():
        if edge.length is not None and edge.length < 0:
            edge.length = 0.0
    return tree, labels


@pytest.mark.parametrize("case", ["nj50", "nj300", "caterpillar", "founder_on_seed", "polytomy"])
def test_tree_distances_bit_identical_to_shortest_path(case):
    from heracles.trees.utils import FOUNDER_LABEL, parse_newick
    if case == "nj50":
        tree, labels = _random_nj_tree(50, 0)
    elif case == "nj300":
        tree, labels = _random_nj_tree(300, 1)
    elif case == "caterpillar":
        tree = parse_newick(_caterpillar(1000, length=lambda i: 0.1 * i + 0.05))
        labels = ["L%d" % i for i in range(1000)]
    elif case == "founder_on_seed":
        tree = parse_newick("((A:0.3,B:0.7):0.2,(C:1.1,D:0.4):0.9);")
        tree.seed_node.taxon = Taxon(label=FOUNDER_LABEL)
        labels = ["D", FOUNDER_LABEL, "A", "C", "B"]
    else:
        tree = parse_newick("((A:0,B:0.5,C:1):0,(D:2,E:0):1,F:3);")
        labels = ["F", "A", "E", "B", "D", "C"]
    for weighted in (True, False):
        fast = tree_distances(tree, labels, weighted=weighted)
        ref = _tree_distances_shortest_path(tree, labels, weighted=weighted)
        assert np.array_equal(fast, ref)


def test_linkage_to_newick_iterative_matches_recursive():
    import sys
    from scipy.cluster.hierarchy import linkage, to_tree
    from heracles.trees.utils import linkage_to_newick, quote_label
    labels = ["p%d" % i for i in range(3000)]
    root = to_tree(linkage(np.geomspace(1, 1e6, 3000)[:, None], "single"))

    def emit(nd, parent_dist=None):       # recursive reference
        if nd.is_leaf():
            s = quote_label(labels[nd.id])
        else:
            s = "(%s,%s)" % (emit(nd.left, nd.dist), emit(nd.right, nd.dist))
        return s if parent_dist is None else "%s:%.10g" % (s, max(0.0, parent_dist - nd.dist))

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 20000))
    try:
        assert linkage_to_newick(root, labels) == emit(root)
    finally:
        sys.setrecursionlimit(limit)


# tree structure and Robinson-Foulds inputs
def test_deroot_keeps_leaf_child_of_root():
    # the root's second child is a leaf: derooting must not drop it
    ns = TaxonNamespace(["a", "b", "c"])
    t = parse_newick("((a:1,b:1):1,c:2);", taxon_namespace=ns)
    t.deroot()
    assert sorted(n.taxon.label for n in t.leaf_node_iter()) == ["a", "b", "c"]
    assert len(t.seed_node.child_nodes()) == 3
    # founder-rooted shape with the leaf first: same outcome
    t2 = parse_newick("(c:2,(a:1,b:1):1);", taxon_namespace=ns)
    t2.deroot()
    assert sorted(n.taxon.label for n in t2.leaf_node_iter()) == ["a", "b", "c"]
    assert robinson_foulds(t, t2) == 0


def test_deroot_two_taxon_tree_unchanged():
    ns = TaxonNamespace(["a", "b"])
    t = parse_newick("(a:1,b:1);", taxon_namespace=ns)
    t.deroot()
    assert sorted(n.taxon.label for n in t.leaf_node_iter()) == ["a", "b"]


def test_internal_node_labels_round_trip():
    from heracles.trees.utils import as_newick
    t = parse_newick("((a:1,b:1)nd1:1,(c:1,d:1)nd2:1)nd0;")
    labels = {n.label for n in t.preorder_node_iter() if not n.is_leaf()}
    assert labels == {"nd0", "nd1", "nd2"}
    out = as_newick(t, internal_labels=True)
    assert "nd1" in out and "nd2" in out and "nd0" in out
    back = parse_newick(out)
    assert {n.label for n in back.preorder_node_iter() if not n.is_leaf()} == labels


def test_native_nj_registers_taxa_in_namespace():
    from heracles.trees.tree import native_nj
    ns = TaxonNamespace()
    D = np.array([[0, 2, 4, 4], [2, 0, 4, 4], [4, 4, 0, 2], [4, 4, 2, 0]], float)
    t = native_nj(D, ["a", "b", "c", "d"], taxon_namespace=ns)
    assert t.taxon_namespace is ns
    assert sorted(ns.labels()) == ["a", "b", "c", "d"]
    assert all(n.taxon in ns for n in t.leaf_node_iter())


def test_robinson_foulds_normalized_and_newick_inputs():
    a = "((a,b),(c,d),(e,f));"
    b = "((a,c),(b,d),(e,f));"
    assert robinson_foulds(a, a) == 0
    raw = robinson_foulds(a, b)
    assert raw == 4                                  # two splits each side
    assert robinson_foulds(a, b, normalized=True) == pytest.approx(raw / 6)
    ns = TaxonNamespace(list("abcdef"))
    ta = parse_newick(a, taxon_namespace=ns)
    tb = parse_newick(b, taxon_namespace=ns)
    assert robinson_foulds(ta, tb) == raw


def test_leaf_iter_handles_deep_trees():
    from heracles.trees.utils import parse_newick
    nwk = "(" * 3000 + "x0" + "".join(",x%d)" % i for i in range(1, 3001)) + ";"
    leaves = [l.taxon.label for l in parse_newick(nwk).seed_node.leaf_iter()]
    assert leaves == ["x%d" % i for i in range(3001)]
