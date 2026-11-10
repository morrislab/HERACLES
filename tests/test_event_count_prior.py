"""Event-count allele priors: event sizes, event counts, the prior and the fit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heracles import Heracles, HeraclesConfig
from heracles.embedding.priors import (build_allele_priors, count_edit_events,
                                       estimate_event_sizes)
from heracles.fit import load_character_matrix
from heracles.trees.utils import parse_newick, tree_distances

DATA = Path(__file__).resolve().parent.parent / "tutorials" / "data" / "cas9_mixed_dropout"

# One site.  Cells 0-2 inherited allele 3 from one event; cell 3 is unedited;
# cell 4 carries allele 3 from a separate event; cell 5 is missing.
ONE_SITE = np.array([[3], [3], [3], [0], [3], [-1]])
NEAREST = np.array([[1, 2, 5, 3, 4],
                    [0, 2, 5, 3, 4],
                    [1, 0, 5, 3, 4],
                    [2, 1, 0, 5, 4],
                    [5, 3, 0, 1, 2],
                    [4, 3, 2, 1, 0]])


def test_event_size_counts_the_run_of_carriers_and_skips_missing_cells():
    sizes = estimate_event_sizes(ONE_SITE, NEAREST)[:, 0]
    # cells 0-2: themselves + the two others (cell 5 is missing and skipped, cell 3 ends the run)
    # cell 4: its nearest observed neighbour (cell 3) is unedited, so a run of one
    assert sizes.tolist() == [3, 3, 3, 0, 1, 0]


def test_carriers_of_one_event_add_up_to_one():
    events = count_edit_events(ONE_SITE, NEAREST)
    assert events.shape == (1, 4)
    assert events[0, 3] == pytest.approx(2.0)          # one event of three cells + one of a single cell
    assert events[0, 0] == 0.0


def test_a_run_that_fills_the_neighbour_list_counts_every_carrier_at_the_site():
    cm = np.full((6, 1), 2)
    nearest = np.array([[j for j in range(6) if j != i][:3] for i in range(6)])
    assert estimate_event_sizes(cm, nearest)[:, 0].tolist() == [6] * 6
    assert count_edit_events(cm, nearest)[0, 2] == pytest.approx(1.0)


@pytest.mark.parametrize("prior, shape", [("per_site_event_count", (2, 5)),
                                          ("global_event_count", (5,))])
def test_event_count_prior_is_normalised_over_observed_alleles(prior, shape):
    cm = np.array([[1, 4], [1, 4], [2, 0], [1, -1], [0, 4]])
    nearest = np.array([[1, 3, 2, 4], [0, 3, 2, 4], [4, 0, 1, 3], [0, 1, 2, 4], [2, 0, 1, 3]])
    q = build_allele_priors(cm, prior, neighbours=nearest)
    assert q.shape == shape
    assert np.all(q[..., 0] == 1.0)
    assert np.allclose(q[..., 1:].sum(axis=-1), 1.0)
    if q.ndim == 2:
        assert q[0, 3] == 0.0 and q[0, 4] == 0.0       # alleles never seen at site 0 stay 0
        # allele 1 arose once and three cells carry it, allele 2 arose once in one cell:
        # one event each, so the same prior however many cells inherited it
        assert q[0, 1] == pytest.approx(q[0, 2])


def test_event_count_prior_needs_neighbours():
    with pytest.raises(ValueError, match="nearest neighbours"):
        build_allele_priors(ONE_SITE, "per_site_event_count")


def test_neighbours_from_the_true_tree_recover_the_true_event_counts():
    """Ranked by the true tree, the tree-free count matches events counted on the
    tree closely (the residual comes from sister-lineage origins merging into one run)."""
    states, labels = load_character_matrix(DATA / "characters.csv")
    tree = parse_newick((DATA / "true_tree.nwk").read_text())
    dists = tree_distances(tree, labels, weighted=True)
    np.fill_diagonal(dists, np.inf)
    nearest = np.argsort(dists, axis=1, kind="stable")[:, :len(labels) - 1]
    counted = count_edit_events(states, nearest)
    on_tree = _events_on_tree(tree, labels, states, counted.shape)
    edited = on_tree > 0
    assert np.corrcoef(counted[edited], on_tree[edited])[0, 1] > 0.95


def _events_on_tree(tree, labels, states, shape):
    """Maximal clades whose observed cells all carry an allele, per site and allele."""
    row = {lab: i for i, lab in enumerate(labels)}
    value, events = {}, np.zeros(shape)
    order = list(tree.postorder_node_iter())
    for node in order:
        if node.is_leaf():
            value[id(node)] = states[row[node.taxon.label]]
            continue
        kids = np.stack([value[id(c)] for c in node.child_nodes()])
        seen = kids != -1
        first = np.where(seen.any(0), kids[seen.argmax(0), np.arange(kids.shape[1])], -1)
        agree = ((kids == first) | ~seen).all(0)
        value[id(node)] = np.where(first == -1, -1, np.where(agree & (first > 0), first, 0))
    for node in order:
        v = value[id(node)]
        parent = value[id(node.parent_node)] if node.parent_node is not None else np.zeros_like(v)
        for site in np.nonzero((v > 0) & (v != parent))[0]:
            events[site, v[site]] += 1
    return events


@pytest.mark.parametrize("low_memory", [False, True])
def test_fit_counts_the_prior_once_and_keeps_it(tmp_path, low_memory):
    cm = pd.read_csv(DATA / "characters.csv", index_col=0).iloc[:60]
    m = Heracles(outdir=tmp_path / "fit", allele_prior="per_site_event_count", dimension=6,
                 max_iterations=20, convergence_thresh=None, seed=0, readouts=["upgma"],
                 low_memory=low_memory).fit(cm)
    q = m.learned_allele_prior_
    assert q is not None and q.ndim == 2
    assert np.allclose(q[:, 1:].sum(axis=1)[q[:, 1:].sum(axis=1) > 0], 1.0)
    states, _ = load_character_matrix(cm)
    alleles = np.unique(states[states > 0])
    assert np.all(q[:, alleles].max(axis=0) > 0)
    loaded = Heracles.load(tmp_path / "fit" / "model.npz")
    assert np.array_equal(loaded.learned_allele_prior_, q)


def test_config_accepts_the_event_count_priors():
    for name in ("per_site_event_count", "global_event_count"):
        assert HeraclesConfig(allele_prior=name).allele_prior == name
    with pytest.raises(ValueError, match="allele_prior"):
        HeraclesConfig(allele_prior="event_count")
