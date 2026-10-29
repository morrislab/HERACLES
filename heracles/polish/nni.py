"""Likelihood NNI polish of a HERACLES read-out tree.

A local topology search that keeps an NNI move when it raises the whole-tree
log-likelihood of the character matrix under the fitted edit model (a
:class:`~heracles.polish.ancestral.TreeLikelihood`). :func:`branch_support` scores
branches by embedding quartet support; :func:`guided_refit` runs the NNI search;
:func:`polish_model` runs it with the published settings on a fitted
:class:`~heracles.Heracles`. Trees stay founder-rooted throughout, and distances
are the ``(n+1, n+1)`` ``full_dists`` with the founder at row 0.
"""

import dataclasses
import os
import time

import numpy as np

from heracles.trees.utils import copy_tree
from heracles.polish.ancestral import (SCALE_FLOOR, index_tree, optimize_branch_lengths,
                                       prepare_model, pruning_passes)


__all__ = [
    'branch_support',
    'ols_branch_lengths',
    'RefitResult',
    'estimate_root_edge',
    'guided_refit',
    'POLISH_MIN_GAIN',
    'POLISH_MAX_ROUNDS',
    'model_rates',
    'polish_model',
]


def _leafset(node):
    """Leaf labels below *node*, as a frozenset."""
    return frozenset(str(lf.taxon.label) for lf in node.leaf_iter())


def _internal_nonroot(tree):
    return [n for n in tree.preorder_internal_node_iter()
            if n is not tree.seed_node]


def _node_by_clade(tree):
    return {_leafset(n): n for n in _internal_nonroot(tree)}


def branch_support(tree, full_dists, labels, *, n_pairs=200, local_k=2, seed=0):
    """Quartet support of every internal branch under the embedding distances.

    Support is the fraction of sampled quartets ``(a, b | c, d)``, with ``a, b``
    inside the clade and ``c, d`` among the ``max(4, local_k * |C|)`` nearest
    outside leaves, whose four-point split agrees with the branch (1.0 = full
    agreement). Also written to ``node.annotations["support"]``; NaN for clades
    with fewer than 2 leaves inside or outside.

    Args:
        tree (Tree): rooted read-out tree over *labels*.
        full_dists (np.ndarray): ``(n+1, n+1)`` distances, founder at row 0.
        labels (list[str]): leaf labels in ``full_dists`` order (no founder).
        n_pairs (int): pairs sampled inside and outside the clade.
        local_k (int): outside neighbourhood size, as a multiple of the clade size.
        seed (int): random seed for the pair sampling.

    Returns:
        dict: ``{node: support}`` for every internal non-root node.
    """
    rng = np.random.default_rng(seed)
    D = np.asarray(full_dists)[1:, 1:]
    pos = {str(lab): i for i, lab in enumerate(labels)}
    n = len(labels)
    out = {}
    for node in _internal_nonroot(tree):
        inside = np.fromiter((pos[lab] for lab in _leafset(node)), dtype=np.int64)
        if len(inside) < 2 or len(inside) > n - 2:
            out[node] = np.nan
            continue
        mask = np.zeros(n, dtype=bool)
        mask[inside] = True
        outside = np.flatnonzero(~mask)
        near = outside[np.argsort(D[np.ix_(inside, outside)].mean(axis=0))[:max(4, local_k * len(inside))]]
        ab = np.column_stack(np.triu_indices(len(inside), 1))
        if len(ab) > n_pairs:
            ab = ab[rng.choice(len(ab), n_pairs, replace=False)]
        a = inside[ab[:, 0]][:, None]
        b = inside[ab[:, 1]][:, None]
        cd = rng.choice(len(near), size=(n_pairs, 2))
        cd = cd[cd[:, 0] != cd[:, 1]]
        c = near[cd[:, 0]][None, :]
        d = near[cd[:, 1]][None, :]
        s_ab = D[a, b] + D[c, d]
        s_ac = D[a, c] + D[b, d]
        s_ad = D[a, d] + D[b, c]
        supp = float((s_ab < np.minimum(s_ac, s_ad)).mean())
        out[node] = supp
        node.annotations["support"] = "%.3f" % supp
    return out


def ols_branch_lengths(tree, full_dists, labels):
    """Least-squares edge lengths so tree distances match the embedding, in place.

    Fits every leaf-leaf and founder-leaf distance, including the founder-to-seed
    edge; negative lengths are clamped to 0. Run time grows with ``n^2 x edges``.

    Args:
        tree (Tree): rooted tree over *labels*.
        full_dists (np.ndarray): ``(n+1, n+1)`` distances, founder at row 0.
        labels (list[str]): leaf labels in ``full_dists`` order.

    Returns:
        float: the fitted root-edge (founder to seed) length.
    """
    full_dists = np.asarray(full_dists, dtype=np.float64)
    pos = {str(lab): i + 1 for i, lab in enumerate(labels)}     # full_dists index
    edge_nodes = [n for n in tree.preorder_node_iter() if n is not tree.seed_node]
    e_of = {id(n): k for k, n in enumerate(edge_nodes)}
    E = len(edge_nodes) + 1                                     # + root edge
    root_e = E - 1
    n_tax = len(labels) + 1
    paths = np.zeros((n_tax, E), dtype=bool)
    paths[1:, root_e] = True         # every leaf path to the founder crosses the root edge
    for lf in tree.leaf_node_iter():  # founder row (index 0) stays empty: it is the origin
        r = pos[str(lf.taxon.label)]
        node = lf
        while node is not tree.seed_node:
            paths[r, e_of[id(node)]] = True
            node = node.parent_node
    iu, ju = np.triu_indices(n_tax, 1)
    d = full_dists[iu, ju]
    # Accumulate the normal equations in chunks over pairs: the full design
    # matrix (n_pairs x E) does not fit in memory for large n.
    n_pairs = iu.shape[0]
    XtX = np.zeros((E, E), dtype=np.float64)
    Xtd = np.zeros(E, dtype=np.float64)
    chunk = max(1, 20_000_000 // max(E, 1))
    for start in range(0, n_pairs, chunk):
        sl = slice(start, start + chunk)
        Xc = (paths[iu[sl]] != paths[ju[sl]]).astype(np.float64)
        XtX += Xc.T @ Xc
        Xtd += Xc.T @ d[sl]
    b = np.linalg.lstsq(XtX, Xtd, rcond=None)[0]
    b = np.maximum(b, 0.0)
    for n in edge_nodes:
        n.edge_length = float(b[e_of[id(n)]])
    return float(b[root_e])


def _rooted_nni(tree, node, which):
    """Swap child *which* (0/1) of *node* with its sibling, in place.

    Returns the five nodes whose edges the move touches.
    """
    parent = node.parent_node
    if parent is None:
        raise ValueError("cannot NNI around the root edge")
    sibs = [c for c in parent.child_nodes() if c is not node]
    kids = node.child_nodes()
    if len(sibs) != 1 or len(kids) != 2:
        raise ValueError("NNI needs a bifurcating pivot edge")
    sib, kid = sibs[0], kids[which]
    parent.remove_child(sib)
    node.remove_child(kid)
    node.add_child(sib)
    parent.add_child(kid)
    return [node, kids[1 - which], kid, sib, parent]


class _State:
    """Current tree and its pruning messages, for incremental NNI scoring."""

    def __init__(self, tree, likelihood, root_edge_length):
        self.tree = tree
        self.model = likelihood
        self.root_edge_length = float(root_edge_length)
        self.refresh()

    def refresh(self):
        """Recompute all pruning messages; returns the total log-likelihood."""
        self.nodes, _, self.row_of = index_tree(self.tree, self.model)
        self.pr = pruning_passes(self.tree, self.model, self.row_of, self.nodes,
                                  root_edge_length=self.root_edge_length,
                                  need_down=True)
        self.loglik = self.model.total(self.pr.loglik_per_site)
        return self.loglik

    def _P(self, node):
        P = self.pr.P.get(id(node))
        return P if P is not None else self.model.edge_P(node.edge_length)

    def _F(self, node):
        return self.pr.F[self.pr.idx_of[id(node)]]

    def _above(self, node):
        """Message arriving at *node*'s parent end from outside its subtree."""
        parent = node.parent_node
        if parent is None:
            return self.pr.pi
        above = self.pr.G[self.pr.idx_of[id(parent)]].copy()
        for other in parent.child_nodes():
            if other is not node:
                above *= self.pr.M[id(other)]
        return above

    def _path_loglik(self, node, F_node, new_sc):
        """Exact total log-likelihood after replacing *node*'s F, by redoing the
        up pass on the path to the root (*new_sc*: new scale factors by node id)."""
        pr, model = self.pr, self.model
        log_scale = pr.log_scale.copy()
        for nid, sc_new in new_sc.items():
            log_scale += np.log(sc_new) - np.log(pr.sc[pr.idx_of[nid]])
        cur, F_cur = node, F_node
        while cur.parent_node is not None:
            parent = cur.parent_node
            msg = np.einsum("sab,sb->sa", self._P(cur), F_cur)
            F_par = msg.copy()
            for other in parent.child_nodes():
                if other is not cur:
                    F_par *= pr.M[id(other)]
            sc = np.maximum(F_par.sum(axis=1), SCALE_FLOOR)
            F_par /= sc[:, None]
            log_scale += np.log(sc) - np.log(pr.sc[pr.idx_of[id(parent)]])
            cur, F_cur = parent, F_par
        site = np.log(np.maximum((pr.pi * F_cur).sum(axis=1), SCALE_FLOOR))
        return model.total(site + log_scale)

    def evaluate_nni(self, v, which, *, n_sweeps=2, tol=1e-4, t_max=None):
        """Log-likelihood of swapping child *which* of *v* with its sibling.

        The touched edges are refit by ML first (``n_sweeps=0`` keeps current
        lengths). Returns ``(loglik, {id(node): edge_length})``, or ``None`` if
        the pivot edge is not bifurcating on both ends.
        """
        from scipy.optimize import minimize_scalar

        p = v.parent_node
        if p is None or v.is_leaf():
            return None
        sibs = [c for c in p.child_nodes() if c is not v]
        kids = v.child_nodes()
        if len(sibs) != 1 or len(kids) != 2:
            return None
        s, kid, keep = sibs[0], kids[which], kids[1 - which]
        model = self.model
        if t_max is None:
            t_max = self._t_max
        above_p = self._above(p)                     # unchanged by the swap
        F_keep, F_s, F_kid = self._F(keep), self._F(s), self._F(kid)
        lengths = {id(n): max(0.0, float(n.edge_length or 0.0))
                   for n in (v, keep, s, kid)}
        p_is_root = p.parent_node is None
        if not p_is_root:
            lengths[id(p)] = max(0.0, float(p.edge_length or 0.0))
        P_cache = {k: model.edge_P(t) for k, t in lengths.items()}

        def local(P):
            """Unnormalised local likelihood per site for tensors *P*."""
            F_v = np.einsum("sab,sb->sa", P[id(keep)], F_keep) * \
                np.einsum("sab,sb->sa", P[id(s)], F_s)
            F_p = np.einsum("sab,sb->sa", P[id(v)], F_v) * \
                np.einsum("sab,sb->sa", P[id(kid)], F_kid)
            if p_is_root:
                return F_v, F_p, (above_p * F_p).sum(axis=1)
            return F_v, F_p, np.einsum("sa,sab,sb->s", above_p, P[id(p)], F_p)

        def total(P):
            return model.total(np.log(np.maximum(local(P)[2], SCALE_FLOOR)))

        cur = total(P_cache)
        for _ in range(n_sweeps):
            improved = False
            for k in list(lengths):
                def f(t, k=k):
                    P_try = dict(P_cache)
                    P_try[k] = model.edge_P(t)
                    return -total(P_try)
                r = minimize_scalar(f, bounds=(0.0, t_max), method="bounded",
                                    options={"xatol": tol})
                if -r.fun > cur + 1e-9:
                    lengths[k] = float(r.x)
                    P_cache[k] = model.edge_P(lengths[k])
                    cur = -r.fun
                    improved = True
            if not improved:
                break

        # exact total: new F at v and p, then the path above p
        F_v, F_p, _ = local(P_cache)
        sc_v = np.maximum(F_v.sum(axis=1), SCALE_FLOOR)
        F_v = F_v / sc_v[:, None]
        # F_p was built from the unnormalised F_v; rescale so that F_p is the
        # product of the normalised messages, as the stored pass defines it
        F_p = F_p / sc_v[:, None]
        sc_p = np.maximum(F_p.sum(axis=1), SCALE_FLOOR)
        F_p = F_p / sc_p[:, None]
        # the path pass reads p's transition tensor through _P; substitute the refit one
        saved, had = None, False
        if not p_is_root:
            had = id(p) in self.pr.P
            saved = self.pr.P.get(id(p))
            self.pr.P[id(p)] = P_cache[id(p)]
        try:
            ll = self._path_loglik(p, F_p, {id(v): sc_v, id(p): sc_p})
        finally:
            if not p_is_root:
                if had:
                    self.pr.P[id(p)] = saved
                else:
                    del self.pr.P[id(p)]
        return ll, lengths

    @property
    def _t_max(self):
        if not hasattr(self, "_t_max_cached"):
            h = self.tree.calc_node_root_distances(return_leaf_distances_only=True)
            self._t_max_cached = max(1.0, 4.0 * max(h or [0.0]))
        return self._t_max_cached


def _apply_move(tree, v, which, lengths):
    """In place: perform the NNI and set the refit lengths."""
    _rooted_nni(tree, v, which)
    for node in tree.preorder_node_iter():
        if id(node) in lengths:
            node.edge_length = lengths[id(node)]


@dataclasses.dataclass
class RefitResult:
    """Output of :func:`guided_refit`; ``tree`` is the polished founder-rooted tree."""

    tree: object
    root_edge_length: float
    loglik_start: float
    loglik_end: float
    support_start: dict            # clade frozenset -> support
    moves: list                    # accepted moves, in order
    evaluations: list              # per candidate branch: size, support, best gain
    n_candidates: int              # branches below threshold at the start
    n_evaluated: int               # candidate topologies scored
    n_rounds: int
    wall_s: float

    def __repr__(self):
        return ("RefitResult(%d moves in %d rounds, log-likelihood %.2f -> %.2f, "
                "%d candidate branches, %.1fs)"
                % (len(self.moves), self.n_rounds, self.loglik_start, self.loglik_end,
                   self.n_candidates, self.wall_s))


#: what the polish reads from an edit process's tree likelihood
TREE_LIKELIHOOD_API = ("labels", "prior_mass", "emissions", "edge_P", "root_prior", "total")

#: above this many leaves the dense OLS branch-length system is skipped
_DENSE_OLS_MAX_LEAVES = 3000


def estimate_root_edge(tree, full_dists):
    """Founder-to-seed branch length from the embedding distances.

    Returns the mean founder-leaf distance minus the mean seed-to-leaf distance
    along the tree, clamped at 0.
    """
    depth, total, count = {id(tree.seed_node): 0.0}, 0.0, 0
    for node in tree.preorder_node_iter():
        if node.parent_node is not None:
            t = node.edge_length
            t = 0.0 if t is None else max(0.0, float(t))
            depth[id(node)] = depth[id(node.parent_node)] + t
        if node.is_leaf():
            total += depth[id(node)]
            count += 1
    return max(0.0, float(np.mean(full_dists[0, 1:])) - total / max(count, 1))


def _seed_lengths(tree, cm, full_dists, labels, *, likelihood, branch_lengths,
                  root_edge_length, seed_sweeps):
    """Set starting branch lengths on *tree* in place; returns the root-edge length."""
    if branch_lengths == "keep":
        if root_edge_length is None:
            root_edge_length = estimate_root_edge(tree, full_dists)
        return float(root_edge_length)
    if len(labels) <= _DENSE_OLS_MAX_LEAVES:      # dense pair x edge system
        root_edge_length = ols_branch_lengths(tree, full_dists, labels)
    elif root_edge_length is None:
        root_edge_length = estimate_root_edge(tree, full_dists)
    if branch_lengths == "ml":
        _, root_edge_length, _ = optimize_branch_lengths(
            tree, cm, likelihood=likelihood, root_edge_length=root_edge_length,
            n_sweeps=seed_sweeps)
    return float(root_edge_length)


def guided_refit(tree, cm, full_dists, labels, *, rates=None, allele_prior=None,
                 support_threshold=0.9, max_rounds=3, min_gain=1e-6,
                 branch_lengths="ml", root_edge_length=None, n_pairs=200,
                 seed=0, missing_state=-1, rescore_each_round=False,
                 seed_sweeps=2, local_sweeps=2, candidate_clades=None,
                 candidate_frac=None, candidate_rank="support", prescreen_gain=-5.0,
                 log=None, likelihood=None):
    """NNI hill-climb on low-support branches, accepting likelihood gains.

    Each round scores both NNI neighbours of every candidate branch on the current
    tree and applies the non-overlapping improving moves, best first. Rounds stop
    when no move is accepted or after *max_rounds*.

    Args:
        tree (Tree): rooted read-out tree; not modified (a copy is returned).
        cm (pandas.DataFrame): character matrix indexed by leaf label.
        full_dists (np.ndarray): ``(n+1, n+1)`` embedding distances, founder at row 0.
        labels (list[str]): leaf labels in ``full_dists`` order.
        rates (dict): Cas9 rates for :func:`~heracles.polish.ancestral.prepare_model`;
            used only when *likelihood* is None.
        allele_prior: as in :func:`~heracles.polish.ancestral.infer_ancestral_states`;
            used only when *likelihood* is None.
        support_threshold (float): branches with support below this are candidates.
        max_rounds (int): maximum passes over the candidates.
        min_gain (float): minimum log-likelihood gain to accept a move.
        branch_lengths (str): starting lengths: ``"ml"`` (OLS then ML), ``"ols"``,
            or ``"keep"`` (input lengths, no per-candidate refit).
        root_edge_length (float, optional): founder-to-seed length when not fitted
            (default :func:`estimate_root_edge`).
        n_pairs (int): pairs per side for :func:`branch_support`.
        seed (int): random seed for :func:`branch_support`.
        missing_state (int): value in *cm* marking missing data.
        rescore_each_round (bool): recompute support at the start of each round.
        seed_sweeps (int): ML sweeps for the starting branch lengths.
        local_sweeps (int): ML sweeps over the touched edges per candidate.
        candidate_clades (set of frozenset, optional): explicit candidate leaf sets;
            replaces the threshold.
        candidate_frac (float, optional): try this fraction of branches instead,
            ranked by *candidate_rank*.
        candidate_rank (str): ``"support"`` (lowest first) or ``"gain"`` (largest
            fixed-length gain first).
        prescreen_gain (float or None): refit lengths only for candidates whose
            fixed-length gain exceeds this; None refits every candidate.
        log (callable, optional): receives progress strings.
        likelihood (TreeLikelihood, optional): scorer to use instead of building
            the Cas9 one from *rates* and *allele_prior*.

    Returns:
        RefitResult

    Raises:
        ValueError: if *branch_lengths* is not ``"ml"``, ``"ols"`` or ``"keep"``.
    """
    t0 = time.perf_counter()
    say = log or (lambda s: None)
    if branch_lengths not in ("ml", "ols", "keep"):
        raise ValueError("branch_lengths must be 'ml', 'ols' or 'keep'")
    if likelihood is None:
        likelihood = prepare_model(cm, allele_prior=allele_prior,
                                   missing_state=missing_state, **rates)
    cur = copy_tree(tree)
    root_edge_length = _seed_lengths(
        cur, cm, full_dists, labels, likelihood=likelihood, branch_lengths=branch_lengths,
        root_edge_length=root_edge_length, seed_sweeps=seed_sweeps)
    state = _State(cur, likelihood, root_edge_length)
    loglik_start = state.loglik
    sweeps = 0 if branch_lengths == "keep" else local_sweeps

    support = {_leafset(n): s for n, s in
               branch_support(cur, full_dists, labels, n_pairs=n_pairs, seed=seed).items()}
    support_start = dict(support)

    def _select(support):
        if candidate_clades is not None:
            return sorted((support.get(c, 1.0), c) for c in candidate_clades)
        if candidate_frac is not None:
            k = int(round(candidate_frac * len(support)))
            if candidate_rank == "gain":
                by_clade = _node_by_clade(cur)
                scored = []
                for c, s in support.items():
                    v = by_clade.get(c)
                    outs = [state.evaluate_nni(v, w, n_sweeps=0) for w in (0, 1)] if v is not None else []
                    g = [o[0] for o in outs if o is not None]
                    scored.append((max(g) - state.loglik if g else -np.inf, s, c))
                scored.sort(key=lambda x: -x[0])
                return [(s, c) for _, s, c in scored[:k]]
            ranked = sorted((s, c) for c, s in support.items() if not np.isnan(s))
            return ranked[:k]
        return sorted((s, c) for c, s in support.items()
                      if not np.isnan(s) and s < support_threshold)

    weak = _select(support)
    n_candidates = len(weak)
    say("seed loglik %.3f, %d/%d branches below support %.2f"
        % (state.loglik, n_candidates, len(support), support_threshold))

    moves, evals, n_eval, rounds = [], [], 0, 0
    for r in range(max_rounds):
        rounds += 1
        if rescore_each_round and r > 0:
            support = {_leafset(n): s for n, s in
                       branch_support(cur, full_dists, labels, n_pairs=n_pairs,
                                      seed=seed + r).items()}
            weak = _select(support)
        by_clade = _node_by_clade(cur)
        proposals = []                                   # (gain, v, which, lengths, s)
        for s, clade in weak:
            v = by_clade.get(clade)
            if v is None:
                continue
            best = None
            for which in (0, 1):
                if sweeps and prescreen_gain is not None:
                    quick = state.evaluate_nni(v, which, n_sweeps=0)
                    if quick is None:
                        break
                    if quick[0] - state.loglik <= prescreen_gain:
                        n_eval += 1
                        out = quick
                        if best is None or out[0] > best[0]:
                            best = (out[0], which, out[1])
                        continue
                out = state.evaluate_nni(v, which, n_sweeps=sweeps)
                if out is None:
                    break
                n_eval += 1
                if best is None or out[0] > best[0]:
                    best = (out[0], which, out[1])
            if best is None:
                continue
            gain = best[0] - state.loglik
            evals.append({"round": r, "clade_size": len(clade),
                          "support": float(s), "best_gain": float(gain)})
            if gain > min_gain:
                proposals.append((gain, v, best[1], best[2], s))

        # batch the non-overlapping improving moves, best first
        proposals.sort(key=lambda x: -x[0])
        taken, batch = set(), []
        for gain, v, which, lengths, s in proposals:
            touched = set(lengths) | {id(v.parent_node)}
            if touched & taken:
                continue
            taken |= touched
            batch.append((gain, v, which, lengths, s))
        accepted = 0
        if batch:
            before = state.loglik
            snapshot = {id(n): n.edge_length for n in cur.preorder_node_iter()}
            for gain, v, which, lengths, s in batch:
                _apply_move(cur, v, which, lengths)
            after = state.refresh()
            if after > before + min_gain:
                accepted = len(batch)
                for gain, v, which, lengths, s in batch:
                    moves.append({"round": r, "clade_size": len(_leafset(v)),
                                  "support": float(s), "gain": float(gain),
                                  "clade": sorted(_leafset(v))})
                say("  round %d: batch of %d moves, loglik %.3f -> %.3f"
                    % (r, len(batch), before, after))
            else:                                        # undo, go one at a time
                for gain, v, which, lengths, s in batch:
                    _rooted_nni(cur, v, 1)              # undo: moved sibling is child 1
                for n in cur.preorder_node_iter():
                    n.edge_length = snapshot[id(n)]
                state.refresh()
                for gain, v, which, lengths, s in batch:
                    before = state.loglik
                    snap = {id(n): n.edge_length for n in cur.preorder_node_iter()}
                    _apply_move(cur, v, which, lengths)
                    after = state.refresh()
                    if after > before + min_gain:
                        accepted += 1
                        moves.append({"round": r, "clade_size": len(_leafset(v)),
                                      "support": float(s),
                                      "gain": float(after - before),
                                      "clade": sorted(_leafset(v))})
                    else:
                        _rooted_nni(cur, v, 1)          # undo
                        for n in cur.preorder_node_iter():
                            n.edge_length = snap[id(n)]
                        state.refresh()
                say("  round %d: batch rejected, %d/%d moves kept one at a time"
                    % (r, accepted, len(batch)))
        say("round %d: %d moves accepted, %d candidates scored so far, loglik %.3f"
            % (r, accepted, n_eval, state.loglik))
        if accepted == 0:
            break
    return RefitResult(tree=cur, root_edge_length=float(root_edge_length),
                       loglik_start=float(loglik_start), loglik_end=float(state.loglik),
                       support_start=support_start, moves=moves, evaluations=evals,
                       n_candidates=n_candidates, n_evaluated=n_eval,
                       n_rounds=rounds, wall_s=time.perf_counter() - t0)


#: minimum log-likelihood gain for an accepted NNI move (the published setting)
POLISH_MIN_GAIN = 1.0
#: rounds over the candidate branches (the published setting)
POLISH_MAX_ROUNDS = 3


def model_rates(model):
    """Fitted rates of a ``Heracles`` model, as keywords for ``edit_model.tree_likelihood``.

    Returns ``mutation_rate``, ``heritable_eps`` and ``stochastic_delta`` (a
    per-site list when dropout was learned per site).
    """
    delta_sites = getattr(model, "stochastic_delta_sites_", None)
    rates = dict(mutation_rate=float(model.mutation_rate_),
                 heritable_eps=float(model.heritable_eps_ or 0.0),
                 stochastic_delta=([float(v) for v in delta_sites] if delta_sites
                                   else float(model.stochastic_delta_ or 0.0)))
    return rates


def polish_model(model, char_matrix, *, readout="nj", min_gain=POLISH_MIN_GAIN,
                 max_rounds=POLISH_MAX_ROUNDS, support_threshold=None, log=None,
                 **kwargs):
    """Run the likelihood NNI polish on one read-out tree of a fitted model.

    Args:
        model (Heracles): fitted model whose edit model supports the polish.
        char_matrix: the DataFrame (or CSV path) the model was fitted on.
        readout (str): which tree in ``model.trees_`` to polish.
        min_gain (float): minimum log-likelihood gain to accept a move.
        max_rounds (int): maximum rounds.
        support_threshold (float, optional): default None makes every branch a
            candidate.
        log (callable, optional): receives progress strings.
        **kwargs: passed to :func:`guided_refit`.

    Returns:
        RefitResult: ``tree`` is the polished, founder-rooted tree.

    Raises:
        ValueError: if the edit model has no tree likelihood, cells are missing
            from *char_matrix*, or *readout* is not on the model.
        TypeError: if the tree likelihood lacks an attribute the polish reads.
    """
    import pandas as pd
    if not model.edit_model.supports_polish:
        raise ValueError("the likelihood NNI polish is not available for %s"
                         % type(model.edit_model).__name__)
    if isinstance(char_matrix, (str, os.PathLike)):
        char_matrix = pd.read_csv(char_matrix, index_col=0)
    cm = char_matrix.copy()
    cm.index = cm.index.astype(str)
    labels = [str(x) for x in model.labels_]
    missing = [x for x in labels if x not in cm.index]
    if missing:
        raise ValueError("character matrix lacks %d of the model's cells (e.g. %r)"
                         % (len(missing), missing[0]))
    cm = cm.loc[labels]
    if readout not in model.trees_:
        raise ValueError("no %r read-out on the model (have %s)"
                         % (readout, sorted(model.trees_)))
    prior = model.learned_allele_prior_
    if prior is None:
        prior = model.config.allele_prior
    if not isinstance(prior, str):
        prior = np.asarray(prior, dtype=np.float64)
    threshold = 1.01 if support_threshold is None else float(support_threshold)
    likelihood = model.edit_model.tree_likelihood(
        cm, allele_prior=prior, missing_state=model.config.missing_state,
        **model_rates(model))
    lacking = [a for a in TREE_LIKELIHOOD_API if not hasattr(likelihood, a)]
    if lacking:
        raise TypeError("%s.tree_likelihood returned a %s without %s (the requirements "
                        "are listed in heracles.models.base)"
                        % (type(model.edit_model.process).__name__,
                           type(likelihood).__name__, ", ".join(lacking)))
    return guided_refit(model.trees_[readout], cm, model.full_dists_, labels,
                        likelihood=likelihood, support_threshold=threshold,
                        max_rounds=max_rounds, min_gain=min_gain, log=log, **kwargs)
