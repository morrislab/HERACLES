"""Tests for the native Camin-Sokal collapse (heracles.trees.collapse)."""

import pandas as pd

from heracles.trees.collapse import sokal_collapse, reconstruct_lca_states
from heracles.trees.collapse import MISSING_STATE
from heracles.trees.utils import parse_newick


def _leafset(node):
    return frozenset(l.taxon.label for l in node.leaf_iter())


def _tree(newick):
    return parse_newick(newick)


def test_collapses_mutationless_internal_edge():
    # X=(a,b) both [1,0]; Y=(c,d) with a conflict at site 0 -> Y reconstructs to
    # [0,0], same as the root -> edge root->Y is mutationless and must collapse,
    # lifting c and d up to the root. Edge X->root carries a mutation (kept).
    cm = pd.DataFrame({0: [1, 1, 2, 0], 1: [0, 0, 0, 0]},
                      index=["a", "b", "c", "d"])
    t = _tree("((a:1,b:1)X:1,(c:1,d:1)Y:2)root;")
    out = sokal_collapse(t, cm)

    internal = [n for n in out.preorder_internal_node_iter()]
    assert len(internal) == 2                      # root + X only (Y gone)
    root = out.seed_node
    child_leafsets = {_leafset(c) for c in root.child_nodes()}
    assert child_leafsets == {frozenset({"a", "b"}),
                              frozenset({"c"}), frozenset({"d"})}
    # branch-length preservation: c's edge (was 1) picks up Y's edge (2) -> 3
    c = next(l for l in out.leaf_node_iter() if l.taxon.label == "c")
    assert c.edge.length == 3.0


def test_shared_clade_edits_are_kept():
    # a,b share edit 1 at site0; c,d share edit 1 at site1. Those shared edits
    # sit on the X and Y internal edges, so neither collapses (root is unedited).
    cm = pd.DataFrame({0: [1, 1, 0, 0], 1: [0, 0, 1, 1]},
                      index=["a", "b", "c", "d"])
    t = _tree("((a:1,b:1)X:1,(c:1,d:1)Y:1)root;")
    out = sokal_collapse(t, cm)
    assert len([n for n in out.preorder_internal_node_iter()]) == 3


def test_tip_only_edits_collapse_to_star():
    # distinct leaf states but every internal LCA is unedited -> all internal
    # edges mutationless -> collapses to a star (one internal node).
    cm = pd.DataFrame({0: [1, 2, 3, 4], 1: [1, 2, 3, 4]},
                      index=["a", "b", "c", "d"])
    t = _tree("((a:1,b:1)X:1,(c:1,d:1)Y:1)root;")
    out = sokal_collapse(t, cm)
    assert len([n for n in out.preorder_internal_node_iter()]) == 1


def test_lca_rule_missing_and_conflict():
    # site0: a=1,b=missing -> impute 1; site1: a=1,b=2 -> conflict -> 0;
    # site2: both missing -> missing
    cm = pd.DataFrame({0: [1, MISSING_STATE], 1: [1, 2], 2: [MISSING_STATE, MISSING_STATE]},
                      index=["a", "b"])
    t = _tree("(a,b)r;")
    states = reconstruct_lca_states(t, cm)
    root = t.seed_node
    assert list(states[root]) == [1, 0, MISSING_STATE]


def test_founder_leaf_is_pruned():
    # a leaf absent from cm (the synthetic founder) is dropped before collapse
    cm = pd.DataFrame({0: [1, 1], 1: [0, 2]}, index=["a", "b"])
    t = _tree("((a:1,b:1):1,__root__:1)r;")
    out = sokal_collapse(t, cm)
    labels = {l.taxon.label for l in out.leaf_node_iter()}
    assert labels == {"a", "b"}


def test_input_tree_not_mutated():
    cm = pd.DataFrame({0: [1, 1, 0], 1: [0, 0, 0]}, index=["a", "b", "c"])
    t = _tree("((a:1,b:1)X:1,c:1)root;")
    before = t.as_string(schema="newick").strip()
    sokal_collapse(t, cm)
    assert t.as_string(schema="newick").strip() == before


def test_collapse_and_rf_are_exported_at_top_level():
    import heracles
    from heracles.trees.utils import robinson_foulds
    assert heracles.sokal_collapse is sokal_collapse
    assert heracles.robinson_foulds is robinson_foulds
    cm = pd.DataFrame({0: [1, 1, 0], 1: [0, 0, 0]}, index=["a", "b", "c"])
    collapsed = heracles.sokal_collapse("((a:1,b:1)X:1,c:1)root;", cm)
    assert heracles.robinson_foulds(collapsed, "((a,b),c);") == 0
