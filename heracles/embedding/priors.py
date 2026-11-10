"""Allele priors for the edit CTMC and MAP priors on the learned missing-data rates.

Event-count priors
------------------
The allele prior is the chance that one edit event produces a given allele, so
it should follow how often each allele *arises*.  An edit that happened early is
inherited by a whole clade, and anything that tallies alleles per cell counts
that one event once per descendant.  The event-count priors
(``"per_site_event_count"``, ``"global_event_count"``) count events instead,
without a tree:

1. For a cell carrying an allele at a site, its *event size* is the number of
   cells estimated to have inherited the edit from the same event: the cell
   itself plus the run of its nearest neighbours that carry the same allele.
   Neighbours missing at the site are skipped, and the run stops at the first
   neighbour observed without the allele (:func:`estimate_event_sizes`).
2. Each carrier adds ``1 / event size``, so the carriers of one event add up to
   one (:func:`count_edit_events`).
3. The prior is the events per allele plus a small pseudo-count, normalised
   over the alleles observed at the site, or over all sites for a global prior.
"""

import math

import numpy as np
import torch

from heracles.constants import EVENT_COUNT_PSEUDO_COUNT, LOG_FLOOR


__all__ = [
    "EVENT_COUNT_PRIORS",
    "build_allele_priors",
    "count_edit_events",
    "estimate_event_sizes",
    "resolve_prior",
    "missing_log_prior",
]


#: allele priors set from counted edit events; they are counted once, not learned
EVENT_COUNT_PRIORS = ("per_site_event_count", "global_event_count")

# entries of the (cells, neighbours, sites) comparison held in memory at once
_EVENT_SIZE_BLOCK = 1 << 25


def estimate_event_sizes(char_matrix, neighbours, missing_state=-1):
    """Estimate, for every edited entry, how many cells inherited that edit from one event.

    Cells that inherited one edit event are close relatives, so they come first
    in each other's neighbour lists.  The event size of cell ``i`` at site ``j``
    is one (the cell) plus the number of its nearest neighbours that carry the
    same allele before the first neighbour observed with something else;
    neighbours missing at the site are skipped.  A run that fills the whole
    neighbour list means a clade larger than the list, so the event size is then
    the number of cells carrying the allele at that site.

    Args:
        char_matrix: (n, S) integer character matrix (0 unedited, >0 allele).
        neighbours: (n, k) indices of each cell's nearest cells, nearest first.
        missing_state: value marking missing entries.

    Returns:
        np.ndarray: (n, S) int64 event sizes; 0 where the cell is unedited or missing.
    """
    states = torch.as_tensor(np.asarray(char_matrix), dtype=torch.int32)
    neighbours = torch.as_tensor(np.asarray(neighbours), dtype=torch.long)
    n, n_sites = states.shape
    k = neighbours.shape[1]
    edited = (states > 0) & (states != missing_state)
    carriers = _cells_per_allele(states, edited)                         # (S, A + 1)
    site = torch.arange(n_sites)
    rank = torch.arange(k)[None, :, None]
    sizes = torch.zeros((n, n_sites), dtype=torch.long)
    block = max(1, _EVENT_SIZE_BLOCK // max(k * n_sites, 1))
    for start in range(0, n, block):
        own = states[start:start + block]                                # (b, S)
        near = states[neighbours[start:start + block]]                   # (b, k, S)
        same = near == own[:, None, :]
        breaks_run = ~same & (near != missing_state)
        run_ends = torch.where(breaks_run.any(dim=1), breaks_run.int().argmax(dim=1),
                               torch.full_like(own, k).long())
        run = (same & (rank < run_ends[:, None, :])).sum(dim=1)
        fills_list = run_ends == k
        size = torch.where(fills_list, carriers[site, own.clamp(min=0).long()], 1 + run)
        sizes[start:start + block] = torch.where(edited[start:start + block], size.clamp(min=1), 0)
    return sizes.numpy()


def count_edit_events(char_matrix, neighbours, missing_state=-1):
    """Count the edit events behind each allele at each site, without a tree.

    Every carrier of an allele adds ``1 / event size``
    (:func:`estimate_event_sizes`), so the cells that inherited one event add up
    to one.

    Returns:
        np.ndarray: (S, A + 1) float64 events per site and allele code, A the
        largest allele code; column 0 (unedited) is 0.
    """
    sizes = torch.as_tensor(estimate_event_sizes(char_matrix, neighbours, missing_state))
    states = torch.as_tensor(np.asarray(char_matrix), dtype=torch.long)
    edited = sizes > 0
    weight = torch.where(edited, 1.0 / sizes.clamp(min=1).double(), 0.0)
    n_codes = int(states.max()) + 1 if edited.any() else 2
    events = torch.zeros((states.shape[1], n_codes), dtype=torch.float64)
    events.scatter_add_(1, torch.where(edited, states, 0).T.contiguous(), weight.T.contiguous())
    events[:, 0] = 0.0
    return events.numpy()


def _cells_per_allele(states, edited):
    """(S, A + 1) number of cells carrying each allele code at each site."""
    n_codes = int(states.max()) + 1 if edited.any() else 2
    cells = torch.zeros((states.shape[1], n_codes), dtype=torch.long)
    cells.scatter_add_(1, torch.where(edited, states, 0).long().T.contiguous(), edited.long().T.contiguous())
    return cells


def _event_count_prior(char_matrix, neighbours, missing_state, per_site):
    """Events per allele plus the pseudo-count, normalised over the observed alleles."""
    if neighbours is None:
        raise ValueError(
            "an event-count allele prior is counted from each cell's nearest neighbours; "
            "pass neighbours=, or use the prior a fitted model stored "
            "(Heracles.learned_allele_prior_)")
    events = count_edit_events(char_matrix, neighbours, missing_state)
    observed = events > 0
    if not per_site:
        events, observed = events.sum(axis=0), observed.any(axis=0)
    weights = np.where(observed, events + EVENT_COUNT_PSEUDO_COUNT, 0.0)
    total = weights.sum(axis=-1, keepdims=True)
    result = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    result[..., 0] = 1.0
    return result


def build_allele_priors(char_matrix, prior, missing_state=-1, neighbours=None):
    """Build the allele prior: the probability that an edit produces each allele.

    Args:
        char_matrix: (n, S) integer character matrix (0 unedited, >0 allele).
        prior: ``"global_uniform"`` (1/K for each of the K alleles observed
            anywhere; shape (A + 1,), A the largest allele code),
            ``"per_site_uniform"`` (1/K_j per allele observed at site j; shape
            (S, A + 1)), ``"global_event_count"`` or ``"per_site_event_count"``
            (the same shapes, each allele weighted by how many edit events
            produced it; see the module docstring), or an array of shape (K,) or
            (S, K) used as given (copied).  ``"uniform"`` or any other string
            raises ``ValueError``.
        missing_state: value marking missing entries.
        neighbours: (n, k) indices of each cell's nearest cells, nearest first;
            required by the event-count priors, ignored otherwise.

    Returns:
        float64 prior indexed by allele code, slot 0 (unedited) set to 1.0.
        Unobserved alleles get 0; a global prior with no observed edits is
        ``[1, 1]``.
    """
    if isinstance(prior, str) and prior == "uniform":
        raise ValueError(
            "allele_prior='uniform' is ambiguous. "
            "Use 'global_uniform' (one distribution shared across all sites) "
            "or 'per_site_uniform' (one distribution per site).")

    if isinstance(prior, np.ndarray):
        result = prior.copy().astype(np.float64)
        if result.ndim == 1:
            result[0] = 1.0
        else:
            result[:, 0] = 1.0
        return result

    if prior in EVENT_COUNT_PRIORS:
        return _event_count_prior(char_matrix, neighbours, missing_state,
                                  per_site=prior == "per_site_event_count")

    if prior == "per_site_uniform":
        n, S = char_matrix.shape
        K_max = int(char_matrix[char_matrix != missing_state].max()) + 1
        result = np.zeros((S, K_max), dtype=np.float64)
        result[:, 0] = 1.0
        for j in range(S):
            col = char_matrix[:, j]
            edited = col[(col > 0) & (col != missing_state)]
            if edited.size == 0:
                continue
            site_alleles = np.unique(edited)
            K_j = len(site_alleles)
            result[j, site_alleles] = 1.0 / K_j
        return result

    if prior != "global_uniform":
        raise ValueError(
            f"prior must be 'global_uniform', 'per_site_uniform', "
            f"'global_event_count', 'per_site_event_count' or a numpy array, got {prior!r}")

    edited = char_matrix[(char_matrix != 0) & (char_matrix != missing_state)]
    if edited.size == 0:
        return np.array([1.0, 1.0])
    vals = np.unique(edited)
    K = len(vals)
    result = np.zeros(int(vals.max()) + 1, dtype=np.float64)
    result[vals] = 1.0 / K
    result[0] = 1.0
    return result


def _gamma_from_mode_variance(mode, variance):
    """Return (shape, scale) for a Gamma distribution with the given mode and variance."""
    m, V = float(mode), float(variance)
    if m <= 0:
        raise ValueError("Gamma mode must be positive, got %g" % m)
    if V <= 0:
        raise ValueError("Gamma variance must be positive, got %g" % V)
    k = ((2 * V + m ** 2) + m * math.sqrt(m ** 2 + 4 * V)) / (2 * V)
    theta = m / (k - 1)
    return k, theta


#: search range of the Beta concentration kappa = alpha + beta (kappa > 2 keeps a
#: mode inside (0, 1)) and the bisection steps that pin it down
_BETA_KAPPA_BOUNDS = (2.0001, 1e8)
_BISECTION_STEPS = 200
#: "auto" Beta prior variance = mode (1 - mode) / this: a weak prior around the start value
_AUTO_BETA_VARIANCE_DIVISOR = 5


def _beta_from_mode_variance(mode, variance):
    """Return (alpha, beta) for a Beta distribution with the given mode and variance."""
    m = float(mode)
    V = float(variance)
    if not 0 < m < 1:
        raise ValueError("Beta mode must be in (0, 1), got %g" % m)
    max_var = m * (1 - m)
    if V <= 0 or V >= max_var:
        raise ValueError(
            "Beta variance must be in (0, mode*(1-mode)) = (0, %g), got %g"
            % (max_var, V))

    def _var_at_kappa(kappa):
        a = m * (kappa - 2) + 1
        b = (1 - m) * (kappa - 2) + 1
        return a * b / (kappa ** 2 * (kappa + 1))

    lo, hi = _BETA_KAPPA_BOUNDS
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2
        if _var_at_kappa(mid) > V:
            lo = mid
        else:
            hi = mid
    kappa = (lo + hi) / 2
    a = m * (kappa - 2) + 1
    b = (1 - m) * (kappa - 2) + 1
    return a, b


def resolve_prior(init_value, prior_variance, kind):
    """Turn an initial rate (the prior's mode) and a prior variance into hyperparameters.

    *prior_variance* is ``"auto"`` (``init_value**2`` for gamma, ``m(1-m)/5`` for
    beta), None (no prior) or a float; *kind* is ``"gamma"`` (``heritable_eps``) or
    ``"beta"`` (``stochastic_delta``).  Returns ``(shape, scale)`` for gamma,
    ``(alpha, beta)`` for beta, or None; ``ValueError`` if out of range.
    """
    if prior_variance is None:
        return None
    if prior_variance == "auto":
        if kind == "gamma":
            prior_variance = float(init_value) ** 2
        else:
            m = float(init_value)
            prior_variance = m * (1 - m) / _AUTO_BETA_VARIANCE_DIVISOR
    prior_variance = float(prior_variance)
    if kind == "gamma":
        return _gamma_from_mode_variance(init_value, prior_variance)
    return _beta_from_mode_variance(init_value, prior_variance)


def missing_log_prior(eps, delta, eps_gamma, delta_beta):
    """Log prior of the missing-data rates, up to an additive constant; a float64 scalar.

    *eps* / *delta* (scalar or ``(S,)``) are the current rates, or None to skip
    their term.  *eps_gamma* ``(k, theta)`` and *delta_beta* ``(a, b)`` are the
    hyperparameters (None = flat); the beta applies to every entry of *delta*.
    """
    target_device = eps.device if eps is not None else (
        delta.device if delta is not None else torch.device("cpu"))
    log_prior = torch.zeros((), dtype=torch.float64, device=target_device)
    if eps is not None and eps_gamma is not None:
        k, theta = eps_gamma
        log_prior = log_prior + (k - 1.0) * torch.log(eps.clamp_min(LOG_FLOOR)) - eps / theta
    if delta is not None and delta_beta is not None:
        a, b = delta_beta
        # delta is a scalar, or an (S,) per-site vector with an independent Beta
        # prior on every site: the sum reduces both cases to a scalar.
        log_prior = log_prior + ((a - 1.0) * torch.log(delta.clamp_min(LOG_FLOOR))
                                 + (b - 1.0) * torch.log((1.0 - delta).clamp_min(LOG_FLOOR))).sum()
    return log_prior
