"""Missing-data models: which sites go missing together.

* :class:`SiteDropout`: every site goes missing on its own (the Cas9 default).
* :class:`CassetteDropout`: blocks of consecutive sites (integration
  cassettes) go missing as a unit, as in PETracer barcodes.

Cassette dropout needs no new likelihood kernel.  Given its missing pattern, the
probability of a site factors into a missing-pattern term times an edit term, so
counting the pattern once per cassette instead of once per site is a per-site
correction: subtract ``(1 - 1/|c|)`` times the pattern log-probability, which is
the same kernel run with ``mutation_rate = 0`` (:class:`CassetteCorrection`).
Cassettes are assumed all-or-none missing within a cell; partial ones are
scored, counted and warned about.
"""

import logging

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from heracles.models.base import MissingData, model_option, register_missing

logger = logging.getLogger(__name__)


__all__ = [
    'SiteDropout',
    'CassetteDropout',
    'site_groups_from_size',
    'cassette_missingness_report',
    'CassetteCorrection',
]


def site_groups_from_size(n_sites, cassette_size):
    """Cassette index of each column, for blocks of ``cassette_size`` consecutive sites.

    Returns:
        ``(S,)`` int64 array ``[0, 0, .., 1, 1, .., C-1]``.

    Raises:
        ValueError: if ``cassette_size < 1`` or ``n_sites`` is not a multiple of it.
    """
    cassette_size = int(cassette_size)
    if cassette_size < 1:
        raise ValueError("cassette_size must be >= 1, got %r" % (cassette_size,))
    if n_sites % cassette_size:
        raise ValueError(
            "CassetteDropout needs the %d sites to split into whole "
            "cassettes of %d consecutive columns (remainder %d); pass the right "
            "cassette_size or drop the incomplete cassette from the matrix"
            % (n_sites, cassette_size, n_sites % cassette_size))
    return np.repeat(np.arange(n_sites // cassette_size, dtype=np.int64), cassette_size)


def _check_site_groups(site_groups, n_sites):
    groups = np.asarray(site_groups, dtype=np.int64).ravel()
    if groups.shape[0] != n_sites:
        raise ValueError("site_groups has %d entries for %d sites"
                         % (groups.shape[0], n_sites))
    if groups.min() < 0:
        raise ValueError("site_groups must be non-negative cassette indices")
    n_cassettes = int(groups.max()) + 1
    if len(np.unique(groups)) != n_cassettes:
        raise ValueError("site_groups must use every cassette index 0..C-1")
    return groups, n_cassettes


def cassette_missingness_report(missing, site_groups):
    """Count fully vs partially missing (cell, cassette) pairs.

    Args:
        missing: ``(n, S)`` boolean array; the founder row may be included.
        site_groups: ``(S,)`` cassette index per site.

    Returns:
        dict with ``n_cassettes``, ``cassette_sizes``, ``n_cell_cassettes``,
        ``n_all_missing``, ``n_partial`` and ``frac_partial``.
    """
    missing = np.asarray(missing, dtype=bool)
    groups, n_cassettes = _check_site_groups(site_groups, missing.shape[1])
    sizes = np.bincount(groups, minlength=n_cassettes)
    # (n, C) count of missing sites per cassette
    counts = np.zeros((missing.shape[0], n_cassettes), dtype=np.int64)
    for c in range(n_cassettes):
        counts[:, c] = missing[:, groups == c].sum(axis=1)
    all_missing = counts == sizes[None, :]
    partial = (counts > 0) & ~all_missing
    n_pairs = int(counts.size)
    return {
        "n_cassettes": int(n_cassettes),
        "cassette_sizes": sizes.tolist(),
        "n_cell_cassettes": n_pairs,
        "n_all_missing": int(all_missing.sum()),
        "n_partial": int(partial.sum()),
        "frac_partial": float(partial.sum() / n_pairs) if n_pairs else 0.0,
    }


class CassetteCorrection:
    """Per-site tensors the quartet loss needs to score cassette-level missing data.

    Built once per fit and passed to the quartet loss as ``cassette=``.

    Attributes:
        groups: ``(S,)`` long tensor, cassette index of each site.
        site_excess: ``(S,)`` float64 tensor ``1 - 1/|c|``, the weight of the
            pattern log-probability subtracted at each site.
        pattern_states: ``(m, S)`` long tensor, 0 where observed and -1 where
            missing; the input to the extra ``mutation_rate = 0`` kernel pass.
        report: output of :func:`cassette_missingness_report`.

    Args:
        site_groups: ``(S,)`` cassette index per site (see
            :func:`site_groups_from_size`).
        states: ``(m, S)`` founder-augmented hard states (``-1`` missing).
        device: torch device for the tensors.
    """

    def __init__(self, site_groups, states, device=None):
        states = np.asarray(states)
        groups, n_cassettes = _check_site_groups(site_groups, states.shape[1])
        self.n_sites = int(states.shape[1])
        self.n_cassettes = n_cassettes
        self.sizes = np.bincount(groups, minlength=n_cassettes)
        dev = torch.device(device) if device is not None else None

        self.groups = torch.as_tensor(groups, dtype=torch.long)
        excess = 1.0 - 1.0 / self.sizes[groups].astype(np.float64)
        self.site_excess = torch.as_tensor(excess, dtype=torch.float64)

        pat = np.where(states < 0, -1, 0).astype(np.int64)
        self.pattern_states = torch.as_tensor(pat, dtype=torch.long)
        missing = states[1:] < 0

        self.report = cassette_missingness_report(missing, groups)
        if dev is not None:
            self.to(dev)

    def to(self, device):
        """Move every tensor to ``device`` (in place); returns ``self``."""
        self.groups = self.groups.to(device)
        self.site_excess = self.site_excess.to(device)
        self.pattern_states = self.pattern_states.to(device)
        return self

    def correct(self, log_prob, log_prob_pattern):
        """Turn per-site log-probabilities into cassette-model ones.

        Args:
            log_prob: ``(B, S)`` terms evaluated on the data.
            log_prob_pattern: ``(B, S)`` the same terms on ``pattern_states``.

        Returns:
            ``(B, S)`` tensor ``log_prob - (1 - 1/|c|) * log_prob_pattern``.
        """
        return log_prob - self.site_excess * log_prob_pattern

    def expand(self, per_cassette):
        """``(C,)`` per-cassette values -> ``(S,)`` per-site values."""
        return per_cassette[self.groups]

    def log_report(self, log=logger):
        """Log the cassette counts; warn when any cassette is partially missing."""
        r = self.report
        log.info("cassette dropout: %d cassettes (sizes %s), %d/%d cell-cassettes "
                 "fully missing, %d partially missing",
                 r["n_cassettes"], sorted(set(r["cassette_sizes"])),
                 r["n_all_missing"], r["n_cell_cassettes"], r["n_partial"])
        if r["n_partial"]:
            log.warning(
                "CassetteDropout: %d of %d (cell, cassette) pairs are "
                "PARTIALLY missing (%.2f%%): the cassette model assumes all-or-none "
                "dropout; those pairs fall back to per-site pattern weights",
                r["n_partial"], r["n_cell_cassettes"], 100 * r["frac_partial"])

    def summary(self):
        """JSON-friendly summary recorded on the fit."""
        return {"missing_unit": "cassette", "n_cassettes": self.n_cassettes,
                "cassette_sizes": self.sizes.tolist(),
                "cassette_missingness": dict(self.report)}


@register_missing
@dataclass
class SiteDropout(MissingData):
    """Every site goes missing on its own.

    Args:
        separate_rates: learn one dropout probability per site.
    """

    name: ClassVar[str] = "site"
    separate_rates: bool = model_option(False, "one dropout probability per site", type=bool)

    def units(self, n_sites):
        return None


@register_missing
@dataclass
class CassetteDropout(MissingData):
    """Cassettes of ``cassette_size`` consecutive sites go missing as a unit.

    Dropout and heritable deletion blank every site of a cassette together,
    so a lost cassette is counted once rather than once per site.

    Args:
        cassette_size: consecutive sites per cassette (default 3). The number of
            sites must be a multiple of it.
        separate_rates: learn one dropout probability per cassette.
    """

    name: ClassVar[str] = "cassette"
    cassette_size: int = model_option(3, "consecutive sites per cassette", parse=int)
    separate_rates: bool = model_option(False, "one dropout probability per cassette",
                                        type=bool)

    def __post_init__(self):
        self.cassette_size = int(self.cassette_size)
        if self.cassette_size < 1:
            raise ValueError("cassette_size must be >= 1, got %r" % (self.cassette_size,))

    def units(self, n_sites):
        return site_groups_from_size(n_sites, self.cassette_size)

    def validate(self, n_sites, config):
        super().validate(n_sites, config)
        if n_sites is not None:
            site_groups_from_size(n_sites, self.cassette_size)   # raises on a remainder
