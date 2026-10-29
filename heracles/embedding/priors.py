"""Allele priors for the edit CTMC and MAP priors on the learned missing-data rates."""

import math

import numpy as np
import torch

from heracles.constants import LOG_FLOOR


__all__ = [
    'build_allele_priors',
    'resolve_prior',
    'missing_log_prior',
]


def build_allele_priors(char_matrix, prior, missing_state=-1):
    """Build the allele prior: the probability that an edit produces each allele.

    Args:
        char_matrix: (n, S) integer character matrix (0 unedited, >0 allele).
        prior: ``"global_uniform"`` (1/K for each of the K alleles observed
            anywhere; shape (A + 1,), A the largest allele code),
            ``"per_site_uniform"`` (1/K_j per allele observed at site j; shape
            (S, A + 1)), or an array of shape (K,) or (S, K) used as given (copied).
            ``"uniform"`` or any other string raises ``ValueError``.

    Returns:
        float64 prior indexed by allele code, slot 0 (unedited) set to 1.0. The
        uniform options give unobserved alleles 0; a global prior with no observed
        edits is ``[1, 1]``.
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
            f"prior must be 'global_uniform', 'per_site_uniform', or a "
            f"numpy array, got {prior!r}")

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

    lo, hi = 2.0001, 1e8
    for _ in range(200):
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
            prior_variance = m * (1 - m) / 5
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
