"""Tests for ``heracles.polish.nni`` (support-guided likelihood NNI refit).

Run with:

    .venv/bin/python -m pytest tests/test_polish.py -q

1. branch support is 1.0 everywhere when the distances are exactly the
   tree's own path lengths, and drops only around a swapped branch;
2. OLS branch lengths recover the generating lengths (root edge included);
3. rooted NNI yields two valid, distinct bifurcating neighbours per branch;
4. the fast likelihood path equals the full one;
5. a single wrong NNI on a simulated matrix is undone by the guided refit;
6. threshold 0 is a no-op.
"""


import numpy as np
import pandas as pd
import pytest

from heracles.polish.ancestral import infer_ancestral_states
from heracles.polish.nni import (
    _leafset,
    _rooted_nni,
    branch_support,
    guided_refit,
    ols_branch_lengths,
)
from heracles.trees.utils import copy_tree, robinson_foulds, tree_distances
from heracles.polish.ancestral import prepare_model


# whole-tree likelihood and NNI neighbours, used only by these tests
def _preorder_index(tree, node):
    for i, n in enumerate(tree.preorder_node_iter()):
        if n is node:
            return i
    raise ValueError("node not in tree")

def _node_at(tree, index):
    for i, n in enumerate(tree.preorder_node_iter()):
        if i == index:
            return n
    raise IndexError(index)

def _tree_loglik(tree, cm, *, rates, allele_prior, root_edge_length,
                 missing_state=-1):
    """Whole-tree log-likelihood of *cm* on *tree* under the Cas9 process.

    Args:
        rates (dict): keywords for :func:`~heracles.polish.ancestral.prepare_model`
            (``mutation_rate``, ``heritable_eps``, ``stochastic_delta``, optionally
            ``site_groups``).
    """
    return prepare_model(cm, allele_prior=allele_prior, missing_state=missing_state,
                         **rates).loglik(tree, root_edge_length=root_edge_length)

def _nni_candidates(tree, node):
    """The two rooted NNI neighbours of the branch above *node*, as copies.

    Returns:
        list of ``(candidate_tree, touched_nodes)`` pairs; empty if the
        branch is not bifurcating on both ends.
    """
    if node.parent_node is None or node.is_leaf():
        return []
    if len(node.parent_node.child_nodes()) != 2 or len(node.child_nodes()) != 2:
        return []
    k = _preorder_index(tree, node)
    out = []
    for which in (0, 1):
        t = copy_tree(tree)
        touched = _rooted_nni(t, _node_at(t, k), which)
        out.append((t, touched))
    return out

NEWICK = ("(((a:0.3,b:0.3):0.4,(c:0.2,d:0.2):0.5):0.6,"
          "((e:0.4,f:0.4):0.3,(g:0.1,h:0.1):0.6):0.7);")
LABELS = list("abcdefgh")
RATES = dict(mutation_rate=1.0, heritable_eps=0.05, stochastic_delta=0.1)


def _tree(newick=NEWICK):
    from heracles.trees.utils import parse_newick
    return parse_newick(newick, rooting="force-rooted")


def _full_dists(tree, labels, root_edge=0.5):
    """(n+1, n+1) path-length matrix with a founder ``root_edge`` above the seed."""
    D = tree_distances(tree, labels, weighted=True)
    depth = {}
    for node in tree.preorder_node_iter():
        depth[node] = 0.0 if node.parent_node is None else \
            depth[node.parent_node] + (node.edge_length or 0.0)
    leaf_depth = np.array([depth[tree.find_node_with_taxon_label(l)] for l in labels])
    full = np.zeros((len(labels) + 1,) * 2)
    full[1:, 1:] = D
    full[0, 1:] = full[1:, 0] = leaf_depth + root_edge
    return full


def _simulate(tree, labels, n_sites, *, seed=0, n_alleles=4, rates=RATES,
              root_edge=0.0):
    """Characters under the CTMC of ``heracles.polish.ancestral`` (no OTHER state)."""
    rng = np.random.default_rng(seed)
    mu, eps, delta = rates["mutation_rate"], rates["heritable_eps"], rates["stochastic_delta"]
    prior = np.ones(n_alleles + 1) / n_alleles
    prior[0] = 1.0
    state = {}
    for node in tree.preorder_node_iter():
        t = root_edge if node.parent_node is None else (node.edge_length or 0.0)
        parent = np.zeros(n_sites, dtype=np.int64) if node.parent_node is None \
            else state[node.parent_node].copy()
        s = parent.copy()
        deleted = rng.random(n_sites) < 1.0 - np.exp(-eps * t)
        mutate = (parent == 0) & (rng.random(n_sites) < 1.0 - np.exp(-mu * t))
        s[mutate] = rng.integers(1, n_alleles + 1, size=mutate.sum())
        s[deleted | (parent == -1)] = -1
        state[node] = s
    rows = []
    for l in labels:
        s = state[tree.find_node_with_taxon_label(l)].copy()
        s[rng.random(n_sites) < delta] = -1
        rows.append(s)
    cm = pd.DataFrame(np.array(rows), index=labels,
                      columns=["r%d" % j for j in range(n_sites)])
    return cm, prior


def test_branch_support_perfect_tree_and_after_swap():
    tree = _tree()
    full = _full_dists(tree, LABELS)
    supp = branch_support(tree, full, LABELS, n_pairs=50)
    assert supp and all(v == 1.0 for v in supp.values())
    # swap a with (c,d): the branch above (a,b) and around it must lose support
    moved = copy_tree(tree)
    ab = next(n for n in moved.preorder_internal_node_iter() if _leafset(n) == {"a", "b"})
    _rooted_nni(moved, ab, 0)
    supp2 = branch_support(moved, full, LABELS, n_pairs=50)
    clades = {_leafset(n): v for n, v in supp2.items()}
    assert clades[frozenset("efgh")] == 1.0
    assert clades[frozenset("ef")] == 1.0
    assert min(clades.values()) < 1.0


def test_ols_branch_lengths_recovers_truth():
    tree = _tree()
    truth = {_leafset(n) if not n.is_leaf() else n.taxon.label: n.edge_length
             for n in tree.preorder_node_iter() if n.parent_node is not None}
    full = _full_dists(tree, LABELS, root_edge=0.5)
    for n in tree.preorder_node_iter():
        if n.parent_node is not None:
            n.edge_length = 1.0                       # forget the lengths
    root_edge = ols_branch_lengths(tree, full, LABELS)
    assert abs(root_edge - 0.5) < 1e-8
    for n in tree.preorder_node_iter():
        if n.parent_node is None:
            continue
        key = _leafset(n) if not n.is_leaf() else n.taxon.label
        assert abs(n.edge_length - truth[key]) < 1e-8, key
    # negative solutions are clamped at zero
    full[1, 2] = full[2, 1] = 0.0                     # a and b coincide
    ols_branch_lengths(tree, full, LABELS)
    assert all((n.edge_length or 0.0) >= 0.0 for n in tree.preorder_node_iter())


def test_nni_candidates_count_and_validity():
    tree = _tree()
    for node in tree.preorder_internal_node_iter():
        if node is tree.seed_node:
            continue
        cands = _nni_candidates(tree, node)
        assert len(cands) == 2
        # rooted comparison: below the root the two NNIs are re-rootings of one
        # unrooted tree, so unrooted RF cannot tell them apart
        def clades(t):
            return {_leafset(n) for n in t.preorder_internal_node_iter()}
        c1, c2 = clades(cands[0][0]), clades(cands[1][0])
        for t, touched in cands:
            assert sorted(l.taxon.label for l in t.leaf_node_iter()) == LABELS
            assert all(len(n.child_nodes()) == 2 for n in t.preorder_internal_node_iter())
            assert len(clades(t) ^ clades(tree)) == 2
            assert len(touched) == 5
        assert c1 != c2
    # the input tree is untouched
    assert tree.as_string(schema="newick") == _tree().as_string(schema="newick")


def test_posteriors_false_matches_loglik():
    tree = _tree()
    cm, prior = _simulate(tree, LABELS, 40)
    full_res = infer_ancestral_states(copy_tree(tree), cm, allele_prior=prior,
                                      root_edge_length=0.2, **RATES)
    fast = infer_ancestral_states(copy_tree(tree), cm, allele_prior=prior,
                                  root_edge_length=0.2, posteriors=False, **RATES)
    assert fast.posteriors is None
    assert fast.loglik == pytest.approx(full_res.loglik, abs=1e-12)
    assert _tree_loglik(tree, cm, rates=RATES, allele_prior=prior,
                        root_edge_length=0.2) == pytest.approx(full_res.loglik)


@pytest.mark.parametrize("branch_lengths", ["ml", "ols"])
def test_guided_refit_recovers_one_nni(branch_lengths):
    truth = _tree()
    cm, prior = _simulate(truth, LABELS, 400, seed=1)
    full = _full_dists(truth, LABELS)
    wrong = copy_tree(truth)
    ab = next(n for n in wrong.preorder_internal_node_iter() if _leafset(n) == {"a", "b"})
    _rooted_nni(wrong, ab, 0)
    assert robinson_foulds(wrong, truth) == 2
    res = guided_refit(wrong, cm, full, LABELS, rates=RATES, allele_prior=prior,
                       support_threshold=1.0, branch_lengths=branch_lengths,
                       max_rounds=2)
    assert res.loglik_end > res.loglik_start
    assert robinson_foulds(res.tree, truth) == 0
    assert len(res.moves) == 1
    assert res.n_evaluated >= 2
    # a refit on the truth itself must not move away from it
    res2 = guided_refit(truth, cm, full, LABELS, rates=RATES, allele_prior=prior,
                        support_threshold=1.0, branch_lengths=branch_lengths,
                        max_rounds=1)
    assert robinson_foulds(res2.tree, truth) == 0


def test_guided_refit_noop_when_threshold_zero():
    tree = _tree()
    cm, prior = _simulate(tree, LABELS, 30)
    full = _full_dists(tree, LABELS)
    res = guided_refit(tree, cm, full, LABELS, rates=RATES, allele_prior=prior,
                       support_threshold=0.0, branch_lengths="keep")
    assert res.n_candidates == 0 and res.n_evaluated == 0 and res.moves == []
    assert robinson_foulds(res.tree, tree) == 0
    assert res.loglik_end == res.loglik_start


def test_incremental_nni_matches_full_recomputation():
    from heracles.polish.nni import _State, _apply_move
    truth = _tree()
    cm, prior = _simulate(truth, LABELS, 200, seed=5)
    tree = copy_tree(truth)
    state = _State(tree, prepare_model(cm, allele_prior=prior, missing_state=-1, **RATES), 0.3)
    assert state.loglik == pytest.approx(
        _tree_loglik(tree, cm, rates=RATES, allele_prior=prior, root_edge_length=0.3))
    for v in list(tree.preorder_internal_node_iter()):
        if v is tree.seed_node:
            continue
        for which in (0, 1):
            # (a) no length change: incremental == full pass on the moved copy
            ll0, lengths0 = state.evaluate_nni(v, which, n_sweeps=0)
            moved = copy_tree(tree)
            _rooted_nni(moved, next(n for n in moved.preorder_internal_node_iter()
                                    if _leafset(n) == _leafset(v)), which)
            full0 = _tree_loglik(moved, cm, rates=RATES, allele_prior=prior,
                                 root_edge_length=0.3)
            assert ll0 == pytest.approx(full0, abs=1e-8)
            # (b) local ML never lowers it and its total is exact at the refit lengths
            ll1, lengths1 = state.evaluate_nni(v, which, n_sweeps=2)
            assert ll1 >= ll0 - 1e-9
            cand = copy_tree(tree)
            # set the refit lengths on the copy (preorder index is stable under
            # copying), then swap: NNI moves nodes with their edge lengths
            orig = list(tree.preorder_node_iter())
            for k, n in enumerate(cand.preorder_node_iter()):
                if id(orig[k]) in lengths1:
                    n.edge_length = lengths1[id(orig[k])]
            cand_v = next(n for n in cand.preorder_internal_node_iter()
                          if _leafset(n) == _leafset(v))
            _rooted_nni(cand, cand_v, which)
            full1 = _tree_loglik(cand, cm, rates=RATES, allele_prior=prior,
                                 root_edge_length=0.3)
            assert ll1 == pytest.approx(full1, abs=1e-8)
    # applying a move and refreshing reproduces the incremental value
    v = next(n for n in tree.preorder_internal_node_iter() if _leafset(n) == {"c", "d"})
    ll, lengths = state.evaluate_nni(v, 0, n_sweeps=2)
    _apply_move(tree, v, 0, lengths)
    assert state.refresh() == pytest.approx(ll, abs=1e-8)
    # and the undo rule (swap child 1 back) restores the seed topology
    _rooted_nni(tree, v, 1)
    assert robinson_foulds(tree, truth) == 0


def test_guided_refit_candidate_clades():
    truth = _tree()
    cm, prior = _simulate(truth, LABELS, 400, seed=1)
    full = _full_dists(truth, LABELS)
    wrong = copy_tree(truth)
    ab = next(n for n in wrong.preorder_internal_node_iter() if _leafset(n) == {"a", "b"})
    _rooted_nni(wrong, ab, 0)
    # after the swap the branch to fix is the pivot {b, c, d}; swapping (c,d)
    # back with a restores the truth
    wrong_clade = frozenset("bcd")
    assert wrong_clade in {_leafset(n) for n in wrong.preorder_internal_node_iter()}
    # only the wrong branch is a candidate -> found; threshold ignored
    res = guided_refit(wrong, cm, full, LABELS, rates=RATES, allele_prior=prior,
                       support_threshold=0.0, candidate_clades={wrong_clade}, max_rounds=1)
    assert res.n_candidates == 1 and len(res.moves) == 1
    assert robinson_foulds(res.tree, truth) == 0
    # an unrelated well-supported branch only -> nothing accepted
    res2 = guided_refit(wrong, cm, full, LABELS, rates=RATES, allele_prior=prior,
                        support_threshold=0.0, candidate_clades={frozenset("gh")}, max_rounds=1)
    assert res2.n_candidates == 1 and res2.moves == []
