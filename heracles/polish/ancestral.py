"""Ancestral character states and tree likelihood under the Cas9 edit model.

Felsenstein pruning on a rooted tree with branch lengths, under the same model the
embedding uses: irreversible edits at rate ``mutation_rate`` landing on allele ``a``
with prior probability ``q_a``, heritable silencing at rate ``heritable_eps`` along
branches, and dropout with probability ``stochastic_delta`` at each leaf. Cassette-level
missing data is supported through ``site_groups``.

Main entry points: :func:`infer_ancestral_states` (node posteriors),
:func:`optimize_branch_lengths` (ML branch lengths), :func:`call_states` (hard calls),
and :class:`TreeLikelihood` (the tree scorer used by the NNI polish).
"""

import dataclasses
import json

import numpy as np


__all__ = [
    'PAD_STATE',
    'DEL_STATE',
    'OTHER_STATE',
    'SCALE_FLOOR',
    'AncestralResult',
    'prepare_model',
    'TreeLikelihood',
    'index_tree',
    'pruning_passes',
    'infer_ancestral_states',
    'optimize_branch_lengths',
    'call_states',
    'save_posteriors_npz',
    'load_posteriors_npz',
]


PAD_STATE = -100
DEL_STATE = -2
OTHER_STATE = -3

SCALE_FLOOR = 1e-300


@dataclasses.dataclass
class AncestralResult:
    """Output of :func:`infer_ancestral_states`.

    ``posteriors[i, s, :n_states[s]]`` is the posterior over
    ``state_values[s, :n_states[s]]`` for node ``node_labels[i]`` at site ``s``
    (zero on padding). ``argmax_calls`` is the argmax over the {0, alleles} states
    only, so it is always a real character value; use :func:`call_states` for
    thresholded calls. The posterior arrays are ``None`` when ``posteriors=False``.
    """

    node_labels: list
    is_leaf: np.ndarray          # (N,) bool
    posteriors: np.ndarray       # (N, S, A_max) float64
    state_values: np.ndarray     # (S, A_max) int64, PAD_STATE on padding
    n_states: np.ndarray         # (S,) int64, valid states per site
    argmax_calls: np.ndarray     # (N, S) int64, in {0, allele ids}
    del_posterior: np.ndarray    # (N, S) float64
    other_posterior: np.ndarray  # (N, S) float64
    loglik_per_site: np.ndarray  # (S,) float64
    loglik: float
    site_names: list
    tree_newick: str
    mutation_rate: float
    heritable_eps: float
    stochastic_delta: float          # site mean when a per-site vector was given
    root_edge_length: float


def _site_alphabets(values, allele_prior):
    """Per-site state layout ``[0, sorted alleles..., OTHER, DEL]`` and edit prior.

    Args:
        values (np.ndarray): (n, S) int64 character matrix, -1 = missing.
        allele_prior (np.ndarray): (K,) global or (S, K_max) per-site prior
            from ``build_allele_priors`` (index 0 is a placeholder, unused).

    Returns:
        tuple: ``(state_values (S, A_max) int64, prior_mass (S, A_max) float64,
        n_states (S,) int64)``. ``prior_mass`` is the edit landing distribution:
        ``q_a`` at allele slots, the residual ``max(0, 1 - sum q_a)`` at OTHER,
        zero at 0/DEL/padding.
    """
    n, S = values.shape
    site_alleles = []
    for s in range(S):
        col = values[:, s]
        site_alleles.append(np.unique(col[col > 0]))
    a_max = max((len(a) for a in site_alleles), default=0) + 3
    state_values = np.full((S, a_max), PAD_STATE, dtype=np.int64)
    prior_mass = np.zeros((S, a_max), dtype=np.float64)
    n_states = np.zeros(S, dtype=np.int64)
    for s in range(S):
        alleles = site_alleles[s]
        k = len(alleles)
        n_states[s] = k + 3
        state_values[s, 0] = 0
        state_values[s, 1:1 + k] = alleles
        state_values[s, 1 + k] = OTHER_STATE
        state_values[s, 2 + k] = DEL_STATE
        if k:
            if allele_prior.ndim == 1:
                q = allele_prior[alleles]
            else:
                q = allele_prior[s, alleles]
            prior_mass[s, 1:1 + k] = q
            prior_mass[s, 1 + k] = max(0.0, 1.0 - float(q.sum()))
        else:
            prior_mass[s, 1] = 1.0     # every edit lands on an unseen allele
    return state_values, prior_mass, n_states


def _edge_transition(t, prior_mass, n_states, mutation_rate, heritable_eps):
    """(S, A_max, A_max) row-stochastic transition matrices for one branch."""
    S, a_max = prior_mass.shape
    # mutation_rate: scalar, or (S,) per site (0 on stacked pattern sites)
    ne = np.broadcast_to(np.exp(-np.asarray(mutation_rate, dtype=np.float64) * t), (S,))
    nd = float(np.exp(-heritable_eps * t))
    s_idx = np.arange(S)
    del_idx = n_states - 1
    P = np.zeros((S, a_max, a_max), dtype=np.float64)
    # from unedited (slot 0): stay, edit (prior_mass is zero at 0/DEL/pad), or delete
    P[:, 0, :] = ((1.0 - ne) * nd)[:, None] * prior_mass
    P[s_idx, 0, 0] = ne * nd
    P[s_idx, 0, del_idx] = 1.0 - nd
    # edited states (slots 1 .. n_states-2): survive in place or delete
    edited = (np.arange(a_max)[None, :] >= 1) & (np.arange(a_max)[None, :] < del_idx[:, None])
    rows, cols = np.nonzero(edited)
    P[rows, cols, cols] = nd
    P[rows, cols, del_idx[rows]] = 1.0 - nd
    # DEL absorbing
    P[s_idx, del_idx, del_idx] = 1.0
    return P


def _leaf_emissions(values, state_values, n_states, stochastic_delta):
    """(n, S, A_max) emission likelihood L[x] = P(observed value | state x)."""
    n, S = values.shape
    a_max = state_values.shape[1]
    # scalar, (S,) per site, or (n, S) per cell and site
    delta = np.broadcast_to(np.asarray(stochastic_delta, dtype=np.float64), (n, S))
    E = np.zeros((n, S, a_max), dtype=np.float64)
    for s in range(S):
        ns = int(n_states[s])
        del_idx = ns - 1
        col = values[:, s]
        missing = col == -1
        E[missing, s, :del_idx] = delta[missing, s][:, None]
        E[missing, s, del_idx] = 1.0
        pos = np.searchsorted(state_values[s, :ns - 2], col[~missing])
        obs_rows = np.nonzero(~missing)[0]
        ok = (pos < ns - 2) & (state_values[s, pos] == col[~missing])
        if not ok.all():
            bad = np.unique(col[~missing][~ok])
            raise ValueError(
                "site %d: observed values %s not in the site alphabet" % (s, bad))
        E[obs_rows, s, pos] = 1.0 - delta[obs_rows, s]
    return E


@dataclasses.dataclass
class TreeLikelihood:
    """Tree scorer for one character matrix under a fitted edit model.

    Holds the topology-independent pieces of the pruning likelihood (per-site
    alphabets, rates, leaf emissions). Built by ``EditProcess.tree_likelihood``
    (for Cas9 via :func:`prepare_model`). :meth:`loglik` scores a tree; the NNI
    polish reuses its pruning messages to score moves incrementally.
    """

    values: np.ndarray           # (n, S) int64
    labels: list
    site_names: list
    state_values: np.ndarray
    prior_mass: np.ndarray
    n_states: np.ndarray
    emissions: np.ndarray        # (n, S_total, A_max)
    mutation_rate: float
    heritable_eps: float
    stochastic_delta: object     # float or (S,) vector
    n_data_sites: int = None     # S; S_total = 2S when pattern sites are stacked
    mu_site: np.ndarray = None   # (S_total,) per-site edit rate (0 on pattern sites)
    site_weight: np.ndarray = None   # (S_total,) or None (plain per-site model)

    def __post_init__(self):
        if self.n_data_sites is None:
            self.n_data_sites = int(self.prior_mass.shape[0])
        if self.mu_site is None:
            self.mu_site = self.mutation_rate

    @property
    def stacked(self):
        """True when cassette pattern sites are stacked after the data sites."""
        return self.site_weight is not None

    def total(self, per_site):
        """Whole-tree log-likelihood from a per-(stacked-)site vector."""
        if self.site_weight is None:
            return float(np.sum(per_site))
        return float(np.dot(per_site, self.site_weight))

    def fold(self, per_site):
        """Map ``(S_total,)`` stacked per-site values to ``(S,)`` data-site values."""
        if self.site_weight is None:
            return per_site
        S = self.n_data_sites
        return per_site[:S] + self.site_weight[S:] * per_site[S:]

    def edge_P(self, t):
        """``(S, A_max, A_max)`` transition matrices for a branch of length *t*."""
        t = 0.0 if t is None else max(0.0, float(t))
        return _edge_transition(t, self.prior_mass, self.n_states,
                                self.mu_site, self.heritable_eps)

    def loglik(self, tree, root_edge_length=0.0):
        """Whole-tree log-likelihood of the data on *tree* (not modified)."""
        tree = tree.taxon_namespace_scoped_copy()
        nodes, _, row_of = index_tree(tree, self)
        pr = pruning_passes(tree, self, row_of, nodes,
                            root_edge_length=root_edge_length, need_down=False)
        return self.total(pr.loglik_per_site)

    def root_prior(self, root_edge_length):
        """``(S, A_max)`` root distribution: unedited, evolved over the founder edge."""
        S, a_max = self.prior_mass.shape
        pi = np.zeros((S, a_max), dtype=np.float64)
        if root_edge_length > 0:
            pi[:] = self.edge_P(root_edge_length)[:, 0, :]
        else:
            pi[:, 0] = 1.0
        return pi


def prepare_model(cm, *, mutation_rate, allele_prior, heritable_eps,
                   stochastic_delta, missing_state, site_groups=None):
    """Build the Cas9 :class:`TreeLikelihood` for character matrix *cm*.

    Args:
        cm (pandas.DataFrame): ``(n, S)`` integer character matrix, cells as index.
        mutation_rate (float): edit rate, in branch-length units.
        allele_prior: ``"global_uniform"``, ``"per_site_uniform"`` or an explicit array.
        heritable_eps (float): heritable silencing rate.
        stochastic_delta: leaf dropout probability: scalar, ``(S,)`` or ``(n, S)``.
        missing_state (int): value in *cm* marking missing data.
        site_groups (array, optional): ``(S,)`` cassette index per site; switches
            the missing-data model to cassette level.

    Raises:
        ValueError: if *stochastic_delta* has the wrong shape.
    """
    from heracles.fit import load_character_matrix, build_allele_priors

    values, labels = load_character_matrix(cm, missing_state=missing_state)
    if isinstance(allele_prior, str):
        allele_prior = build_allele_priors(values, allele_prior)
    allele_prior = np.asarray(allele_prior, dtype=np.float64)
    S = values.shape[1]
    site_names = [str(c) for c in cm.columns] if hasattr(cm, "columns") \
        else ["s%d" % j for j in range(S)]
    delta = np.asarray(stochastic_delta, dtype=np.float64)
    if delta.ndim == 1 and delta.shape[0] != S:
        raise ValueError("stochastic_delta has %d entries for %d sites" % (delta.shape[0], S))
    if delta.ndim == 2 and delta.shape != values.shape:
        raise ValueError("per-cell stochastic_delta must be (n_cells, S) = %s in the row "
                         "order of the character matrix, got %s" % (values.shape, delta.shape))
    state_values, prior_mass, n_states = _site_alphabets(values, allele_prior)
    mu_site, site_weight = float(mutation_rate), None
    if site_groups is not None:
        # stack the 0/-1 missing pattern as S extra sites with edit rate 0 and
        # weight -(1 - 1/|cassette|)
        from heracles.models.missing import _check_site_groups
        groups, n_cassettes = _check_site_groups(site_groups, S)
        sizes = np.bincount(groups, minlength=n_cassettes)
        excess = 1.0 - 1.0 / sizes[groups].astype(np.float64)
        pattern = np.where(values < 0, -1, 0).astype(np.int64)
        sv_p, pm_p, ns_p = _site_alphabets(pattern, allele_prior)
        a_max = max(state_values.shape[1], sv_p.shape[1])
        def _pad(a, fill):
            out = np.full((a.shape[0], a_max), fill, dtype=a.dtype)
            out[:, :a.shape[1]] = a
            return out
        state_values = np.vstack([_pad(state_values, PAD_STATE), _pad(sv_p, PAD_STATE)])
        prior_mass = np.vstack([_pad(prior_mass, 0.0), _pad(pm_p, 0.0)])
        n_states = np.concatenate([n_states, ns_p])
        values = np.hstack([values, pattern])
        if delta.ndim >= 1:
            delta = np.concatenate([delta, delta], axis=-1)
        mu_site = np.concatenate([np.full(S, float(mutation_rate)), np.zeros(S)])
        site_weight = np.concatenate([np.ones(S), -excess])
    emissions = _leaf_emissions(values, state_values, n_states, delta)
    return TreeLikelihood(values=values, labels=labels, site_names=site_names,
                  state_values=state_values, prior_mass=prior_mass,
                  n_states=n_states, emissions=emissions,
                  mutation_rate=float(mutation_rate),
                  heritable_eps=float(heritable_eps),
                  stochastic_delta=(float(delta) if delta.ndim == 0 else delta),
                  n_data_sites=S, mu_site=mu_site, site_weight=site_weight)


def _likelihood_or_build(likelihood, cm, **kw):
    if likelihood is not None:
        return likelihood
    if kw["mutation_rate"] is None or kw["allele_prior"] is None:
        raise TypeError("pass mutation_rate and allele_prior, or a prebuilt likelihood=")
    return prepare_model(cm, **kw)


def index_tree(tree, model):
    """Return ``(preorder nodes, node labels, row_of)``; labels internal nodes in place.

    Internal nodes are named ``nd0, nd1, ...`` in preorder. ``row_of`` maps a leaf
    label to its row in the character matrix.

    Raises:
        ValueError: if the tree leaves and the matrix rows differ.
    """
    row_of = {lbl: i for i, lbl in enumerate(model.labels)}
    nodes = list(tree.preorder_node_iter())
    node_labels = []
    n_internal = 0
    for node in nodes:
        if node.is_leaf():
            if node.taxon is None or str(node.taxon.label) not in row_of:
                raise ValueError("tree leaf %r not found in the character matrix"
                                 % (node.taxon.label if node.taxon else None))
            node_labels.append(str(node.taxon.label))
        else:
            node.label = "nd%d" % n_internal
            node_labels.append(node.label)
            n_internal += 1
    seen = set(node_labels)
    for lbl in model.labels:
        if lbl not in seen:
            raise ValueError("character matrix row %r not found in the tree" % lbl)
    return nodes, node_labels, row_of


@dataclasses.dataclass
class _Pruning:
    """Messages of one pruning run on one (tree, model) pair.

    ``F[i]``: node i's normalised likelihood of the data below it. ``M[id(child)]``:
    the normalised message a child sends up its edge. ``G[i]``: normalised likelihood
    of everything outside node i's subtree given its state (``None`` without the down
    pass). Scale factors are summed into ``log_scale``; ``P`` caches per-edge
    transition matrices.
    """

    nodes: list
    idx_of: dict
    F: np.ndarray
    M: dict
    P: dict
    G: np.ndarray
    pi: np.ndarray
    log_scale: np.ndarray
    loglik_per_site: np.ndarray
    sc: np.ndarray = None        # (N, S) per-node scale factors of F


_P_CACHE_BUDGET = 2e9        # bytes; above this the transition tensors are recomputed


def pruning_passes(tree, model, row_of, nodes, *, root_edge_length,
                    need_down=True):
    """Run the up (and optionally down) pruning pass; returns a :class:`_Pruning`.

    Args:
        tree (Tree): rooted tree with branch lengths.
        model (TreeLikelihood): the scorer.
        row_of (dict), nodes (list): from :func:`index_tree`.
        root_edge_length (float): founder-to-seed branch length.
        need_down (bool): also compute the outside messages ``G``.
    """
    S, a_max = model.prior_mass.shape
    idx_of = {id(node): i for i, node in enumerate(nodes)}
    N = len(nodes)
    cache_P = N * S * a_max * a_max * 8 <= _P_CACHE_BUDGET

    F = np.zeros((N, S, a_max), dtype=np.float64)
    SC = np.zeros((N, S), dtype=np.float64)
    M, P = {}, {}
    log_scale = np.zeros(S, dtype=np.float64)
    for node in tree.postorder_node_iter():
        i = idx_of[id(node)]
        if node.is_leaf():
            F[i] = model.emissions[row_of[str(node.taxon.label)]]
        else:
            F[i] = 1.0
            for child in node.child_nodes():
                Pc = model.edge_P(child.edge_length)
                if cache_P:
                    P[id(child)] = Pc
                msg = np.einsum("sab,sb->sa", Pc, F[idx_of[id(child)]])
                M[id(child)] = msg
                F[i] *= msg
        sc = np.maximum(F[i].sum(axis=1), SCALE_FLOOR)
        F[i] /= sc[:, None]
        SC[i] = sc
        log_scale += np.log(sc)

    pi = model.root_prior(root_edge_length)
    root_i = idx_of[id(tree.seed_node)]
    loglik_per_site = np.log(np.maximum((pi * F[root_i]).sum(axis=1),
                                        SCALE_FLOOR)) + log_scale

    G = None
    if need_down:
        G = np.zeros((N, S, a_max), dtype=np.float64)
        G[root_i] = pi
        for node in tree.preorder_node_iter():
            if node.is_leaf():
                continue
            i = idx_of[id(node)]
            children = node.child_nodes()
            for child in children:
                above = G[i].copy()
                for other in children:
                    if other is not child:
                        above *= M[id(other)]
                Pc = P[id(child)] if cache_P else model.edge_P(child.edge_length)
                g = np.einsum("sa,sab->sb", above, Pc)
                sc = np.maximum(g.sum(axis=1), SCALE_FLOOR)
                G[idx_of[id(child)]] = g / sc[:, None]
    return _Pruning(nodes=nodes, idx_of=idx_of, F=F, M=M, P=P, G=G, pi=pi,
                    log_scale=log_scale, loglik_per_site=loglik_per_site, sc=SC)


def infer_ancestral_states(tree, cm, *, mutation_rate=None, allele_prior=None,
                           heritable_eps=0.0, stochastic_delta=0.0,
                           root_edge_length=0.0, missing_state=-1,
                           posteriors=True, site_groups=None, likelihood=None):
    """Marginal posteriors of every node's character states, by pruning.

    Internal nodes are relabelled ``nd0, nd1, ...`` in preorder **in place**, so
    callers can map nodes to rows of the result.

    Args:
        tree (Tree): rooted, with branch lengths (missing or negative lengths count
            as 0); polytomies allowed; leaf labels must match ``cm.index``.
        cm (pandas.DataFrame): ``(n_leaves, S)`` integer character matrix.
        mutation_rate (float): edit rate, in branch-length units. Required unless
            *likelihood* is given.
        allele_prior: ``"global_uniform"``, ``"per_site_uniform"`` or an explicit
            array. Required unless *likelihood* is given.
        heritable_eps (float): heritable silencing rate.
        stochastic_delta: leaf dropout probability: scalar, ``(S,)`` or ``(n, S)``.
        root_edge_length (float): founder-to-seed branch length (0 = seed unedited).
        missing_state (int): value in *cm* marking missing data.
        posteriors (bool): if False, only ``loglik`` / ``loglik_per_site`` are
            computed (about half the cost) and the posterior fields are ``None``.
        site_groups (array, optional): ``(S,)`` cassette index per site; switches
            the missing-data model to cassette level.
        likelihood (TreeLikelihood, optional): prebuilt scorer (for example from
            ``edit_model.tree_likelihood``); the rate arguments are then ignored.

    Returns:
        AncestralResult

    Raises:
        TypeError: if neither the rates nor *likelihood* are given.
        ValueError: if tree leaves and matrix rows differ.
    """
    model = _likelihood_or_build(likelihood, cm, mutation_rate=mutation_rate,
                                 allele_prior=allele_prior, heritable_eps=heritable_eps,
                                 stochastic_delta=stochastic_delta,
                                 missing_state=missing_state, site_groups=site_groups)
    nodes, node_labels, row_of = index_tree(tree, model)
    pr = pruning_passes(tree, model, row_of, nodes,
                         root_edge_length=root_edge_length, need_down=posteriors)
    is_leaf = np.array([node.is_leaf() for node in nodes])
    S = model.n_data_sites
    common = dict(
        node_labels=node_labels, is_leaf=is_leaf,
        state_values=model.state_values[:S], n_states=model.n_states[:S],
        loglik_per_site=model.fold(pr.loglik_per_site),
        loglik=model.total(pr.loglik_per_site), site_names=model.site_names,
        tree_newick=tree.as_string(schema="newick").strip(),
        mutation_rate=float(model.mutation_rate), heritable_eps=float(model.heritable_eps),
        stochastic_delta=float(np.mean(model.stochastic_delta)),
        root_edge_length=float(root_edge_length))
    if not posteriors:
        return AncestralResult(posteriors=None, argmax_calls=None,
                               del_posterior=None, other_posterior=None,
                               **common)

    s_idx = np.arange(S)
    del_idx = model.n_states[:S] - 1
    post = pr.F[:, :S] * pr.G[:, :S]
    post /= np.maximum(post.sum(axis=2, keepdims=True), SCALE_FLOOR)
    del_posterior = post[:, s_idx, del_idx]
    other_posterior = post[:, s_idx, del_idx - 1]

    called = post.copy()
    called[:, s_idx, del_idx] = 0.0
    called[:, s_idx, del_idx - 1] = 0.0
    argmax_slots = called.argmax(axis=2)                      # (N, S)
    argmax_calls = model.state_values[s_idx[None, :], argmax_slots]
    return AncestralResult(posteriors=post, argmax_calls=argmax_calls,
                           del_posterior=del_posterior,
                           other_posterior=other_posterior, **common)


def _edge_loglik(t, above, F_child, model):
    """Total log-likelihood of edge length *t*, up to a constant in *t*.

    ``above`` is the ``(S, A_max)`` message from outside the child's subtree,
    ``F_child`` the child's normalised partial likelihood.
    """
    P = model.edge_P(t)
    site = np.einsum("sa,sab,sb->s", above, P, F_child)
    return model.total(np.log(np.maximum(site, SCALE_FLOOR)))


def _root_loglik(t, F_root, model):
    """Total log-likelihood of founder edge length *t*, up to a constant."""
    site = (model.root_prior(t) * F_root).sum(axis=1)
    return model.total(np.log(np.maximum(site, SCALE_FLOOR)))


def optimize_branch_lengths(tree, cm, *, mutation_rate=None, allele_prior=None,
                            heritable_eps=0.0, stochastic_delta=0.0,
                            root_edge_length=0.0, missing_state=-1,
                            edges=None, n_sweeps=2, tol=1e-4, t_max=None,
                            optimize_root_edge=True, site_groups=None, likelihood=None):
    """Maximum-likelihood branch lengths on a fixed topology, in place.

    Coordinate ascent over edges; a sweep that lowers the log-likelihood is
    reverted and ends the search, so the result never gets worse.

    Args:
        tree (Tree): rooted, with branch lengths (missing or negative lengths start
            at 0). Internal nodes are relabelled in place.
        cm, mutation_rate, allele_prior, heritable_eps, stochastic_delta,
            missing_state, site_groups, likelihood: as in :func:`infer_ancestral_states`.
        root_edge_length (float): starting founder-to-seed length.
        edges (iterable of Node, optional): optimise only the edges above these
            nodes (default: every non-root node).
        n_sweeps (int): maximum number of sweeps.
        tol (float): absolute tolerance on each length.
        t_max (float, optional): upper bound per edge (default: 4x the tree
            height, at least 1).
        optimize_root_edge (bool): also fit ``root_edge_length``.

    Returns:
        tuple: ``(loglik, root_edge_length, n_sweeps_done)``.
    """
    from scipy.optimize import minimize_scalar

    model = _likelihood_or_build(likelihood, cm, mutation_rate=mutation_rate,
                                 allele_prior=allele_prior, heritable_eps=heritable_eps,
                                 stochastic_delta=stochastic_delta,
                                 missing_state=missing_state, site_groups=site_groups)
    nodes, _, row_of = index_tree(tree, model)
    for node in nodes:
        if node is not tree.seed_node and (node.edge_length is None
                                           or node.edge_length < 0):
            node.edge_length = 0.0
    if t_max is None:
        t_max = max(1.0, 4.0 * max(tree.calc_node_root_distances(
            return_leaf_distances_only=True) or [0.0]))
    targets = [n for n in (nodes if edges is None else edges)
               if n is not tree.seed_node]
    root_edge_length = max(0.0, float(root_edge_length))

    def best_t(fn, t0):
        r = minimize_scalar(lambda t: -fn(t), bounds=(0.0, t_max),
                            method="bounded", options={"xatol": tol})
        return float(r.x) if -r.fun >= fn(t0) else float(t0)

    pr = pruning_passes(tree, model, row_of, nodes,
                         root_edge_length=root_edge_length, need_down=True)
    loglik = model.total(pr.loglik_per_site)
    done = 0
    for _ in range(n_sweeps):
        old = {id(n): n.edge_length for n in targets}
        old_root = root_edge_length
        for node in targets:
            parent = node.parent_node
            above = pr.G[pr.idx_of[id(parent)]].copy()
            for other in parent.child_nodes():
                if other is not node:
                    above *= pr.M[id(other)]
            F_child = pr.F[pr.idx_of[id(node)]]
            node.edge_length = best_t(
                lambda t: _edge_loglik(t, above, F_child, model), node.edge_length)
        if optimize_root_edge:
            F_root = pr.F[pr.idx_of[id(tree.seed_node)]]
            root_edge_length = best_t(
                lambda t: _root_loglik(t, F_root, model), root_edge_length)
        pr_new = pruning_passes(tree, model, row_of, nodes,
                                 root_edge_length=root_edge_length, need_down=True)
        new_loglik = model.total(pr_new.loglik_per_site)
        if new_loglik < loglik - 1e-9:
            for n in targets:
                n.edge_length = old[id(n)]
            root_edge_length = old_root
            break
        done += 1
        improved = new_loglik - loglik > 1e-6
        pr, loglik = pr_new, new_loglik
        if not improved:
            break
    return loglik, root_edge_length, done


def call_states(result, *, min_call_posterior=0.5):
    """Hard state calls from stored posteriors, with low-confidence calls set to -1.

    A node/site is called ``-1`` when its DEL posterior exceeds 0.5, or when the
    largest renormalised posterior over the {0, alleles} states is below
    *min_call_posterior*; otherwise it gets the argmax over {0, alleles}.

    Args:
        result (AncestralResult): computed with ``posteriors=True``.
        min_call_posterior (float): confidence threshold for a call.

    Returns:
        tuple: ``(calls (N, S) int64, info)`` where ``info`` has boolean masks
        ``del_masked`` / ``lowconf_masked`` saying why each -1 was assigned
        (DEL-driven wins when both apply).
    """
    S = result.state_values.shape[0]
    s_idx = np.arange(S)
    del_idx = result.n_states - 1
    called = result.posteriors.copy()
    called[:, s_idx, del_idx] = 0.0
    called[:, s_idx, del_idx - 1] = 0.0
    total = np.maximum(called.sum(axis=2), SCALE_FLOOR)
    conf = called.max(axis=2) / total
    del_masked = result.del_posterior > 0.5
    lowconf_masked = ~del_masked & (conf < min_call_posterior)
    calls = result.argmax_calls.copy()
    calls[del_masked | lowconf_masked] = -1
    return calls, {"del_masked": del_masked, "lowconf_masked": lowconf_masked}


def save_posteriors_npz(path, result):
    """Save an :class:`AncestralResult` to a compressed ``.npz`` file."""
    arrays = {
        "node_labels": np.array(result.node_labels, dtype=str),
        "is_leaf": result.is_leaf,
        "posteriors": result.posteriors,
        "state_values": result.state_values,
        "argmax_calls": result.argmax_calls,
        "del_posterior": result.del_posterior,
        "other_posterior": result.other_posterior,
        "loglik_per_site": result.loglik_per_site,
        "site_names": np.array(result.site_names, dtype=str),
        "tree_newick": np.array(result.tree_newick),
        "_scalars_json": np.array(json.dumps({
            "mutation_rate": result.mutation_rate,
            "heritable_eps": result.heritable_eps,
            "stochastic_delta": result.stochastic_delta,
            "root_edge_length": result.root_edge_length,
            "loglik": result.loglik,
        })),
    }
    # arrays not computed (posteriors=False) are left out of the file
    np.savez_compressed(path, **{k: v for k, v in arrays.items() if v is not None})


def load_posteriors_npz(path):
    """Load an :class:`AncestralResult` written by :func:`save_posteriors_npz`."""
    with np.load(path, allow_pickle=False) as z:
        opt = {k: (z[k] if k in z.files else None)
               for k in ("posteriors", "argmax_calls", "del_posterior", "other_posterior")}
        scalars = json.loads(str(z["_scalars_json"]))
        state_values = z["state_values"]
        return AncestralResult(
            node_labels=[str(x) for x in z["node_labels"]],
            is_leaf=z["is_leaf"], posteriors=opt["posteriors"],
            state_values=state_values,
            n_states=(state_values != PAD_STATE).sum(axis=1).astype(np.int64),
            argmax_calls=opt["argmax_calls"],
            del_posterior=opt["del_posterior"],
            other_posterior=opt["other_posterior"],
            loglik_per_site=z["loglik_per_site"],
            loglik=scalars["loglik"],
            site_names=[str(x) for x in z["site_names"]],
            tree_newick=str(z["tree_newick"]),
            mutation_rate=scalars["mutation_rate"],
            heritable_eps=scalars["heritable_eps"],
            stochastic_delta=scalars["stochastic_delta"],
            root_edge_length=scalars["root_edge_length"])
