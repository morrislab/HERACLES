"""Camin-Sokal collapse of mutationless edges.

A read-out tree resolves every split, including ones no edit supports.
:func:`sokal_collapse` contracts every edge that carries no inferred edit, so
unsupported splits become polytomies, as is usual before RF comparisons. It
matches Cassiopeia's ``collapse_mutationless_edges`` on integer character matrices.
"""

import numpy as np

from heracles.trees.utils import copy_tree, parse_newick


__all__ = [
    'MISSING_STATE',
    'reconstruct_lca_states',
    'sokal_collapse',
]


MISSING_STATE = -1


def reconstruct_lca_states(tree, cm, missing_state=MISSING_STATE):
    """Bottom-up Camin-Sokal ancestral states for every node, as ``{node: (S,) int64 array}``.

    Per site, a node gets the shared value if all non-missing children agree,
    missing if all children are missing, and 0 (unedited) otherwise.  *cm* is an
    ``(n_leaves, S)`` integer DataFrame indexed by leaf label; leaves absent from it
    (for example the founder) are treated as unedited.
    """
    S = cm.shape[1]
    leafvals = {str(i): row.to_numpy(dtype=np.int64) for i, row in cm.iterrows()}
    zeros = np.zeros(S, dtype=np.int64)
    states = {}
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            states[node] = leafvals.get(str(node.taxon.label), zeros)
            continue
        child_mat = np.stack([states[c] for c in node.child_nodes()])  # (n_child, S)
        m = np.ma.masked_equal(child_mat, missing_state)
        mn = m.min(axis=0)
        mx = m.max(axis=0)
        all_missing = m.mask.all(axis=0) if m.mask is not np.ma.nomask else np.zeros(S, bool)
        agree = (mn == mx)                          # all non-missing children equal
        out = np.where(np.asarray(agree), np.asarray(mn.filled(0)), 0).astype(np.int64)
        out[np.asarray(all_missing)] = missing_state
        states[node] = out
    return states


def sokal_collapse(tree, cm, *, missing_state=MISSING_STATE):
    """Return a copy of *tree* (a Tree or Newick string) with mutationless edges contracted.

    *cm* is an integer character matrix indexed by leaf label; tree leaves not in
    it (such as the founder) are pruned first.  A contracted edge's length is added
    to its children's edges, so distances from the root along the tree are
    preserved.  *tree* is not modified.
    """
    if isinstance(tree, str):
        tree = parse_newick(tree.strip(), rooting="force-rooted")
    else:
        tree = copy_tree(tree)
    tree.is_rooted = True   # keep the founder root; else the parser suppresses the basal split

    keep = set(str(i) for i in cm.index)
    extra = [tx for tx in tree.taxon_namespace
             if str(tx.label) not in keep and
             any(leaf.taxon is tx for leaf in tree.leaf_node_iter())]
    if extra:
        tree.prune_taxa(extra, suppress_unifurcations=True)

    states = reconstruct_lca_states(tree, cm, missing_state=missing_state)

    # Postorder so a node's children are finalised before it is contracted.
    for node in list(tree.postorder_node_iter()):
        parent = node.parent_node
        if node.is_leaf() or parent is None:
            continue
        if np.array_equal(states[node], states[parent]):
            elen = node.edge.length or 0.0
            for child in list(node.child_nodes()):
                child.edge.length = (child.edge.length or 0.0) + elen
                node.remove_child(child)
                parent.add_child(child)
            parent.remove_child(node)
    tree.update_bipartitions(suppress_unifurcations=False)
    return tree
