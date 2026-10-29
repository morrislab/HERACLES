r"""Likelihood for recorders that edit their sites in a fixed order (DNA
Typewriter, PASTE).

Sites are grouped into cassettes of size ``K`` whose positions fill in order,
so a cassette carries an *edit count* ``0..K`` (a Poisson-like walk that stops
at ``K``) and one allele per filled position (prior ``q_a``, as in the Cas9
model).  Over a branch of length ``t`` at rate ``lam``::

    P(i -> j, t) = e^{-lam t} (lam t)^{j-i} / (j-i)!    for i <= j < K
    P(i -> K, t) = gammainc(K - i, lam t)               for i < K
    P(K -> K, t) = 1

At ``K = 1`` every function here reduces to its Cas9 counterpart.  Pair,
triple and quartet likelihoods sum over the ancestor's edit count up to the
leaves' shared prefix length, all through one log-space reduction,
:func:`push_up_branch`.  :class:`SequentialModel` is the preset to pass to
``Heracles``.
"""

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

import torch

from heracles.constants import HAMMING_ROW_CHUNK

from heracles.embedding.ctmc import _index_prior, with_learnable_prior
from heracles.models.base import (EditModel, EditProcess, model_option, register_model,
                                  register_process)
from heracles.models.missing import CassetteDropout


__all__ = [
    'NEGATIVE_INFINITY',
    'MIN_RATE_TIME',
    'MIN_PROBABILITY',
    'MIN_LOG_PROBABILITY',
    'Cassettes',
    'validate_cassette_matrix',
    'cassette_summaries',
    'with_allele_prior',
    'shared_prefix_length',
    'log_transition',
    'leaf_column',
    'push_up_branch',
    'sequential_leaf_marginal_logprob',
    'sequential_triple_site_logprob',
    'sequential_quartet_site_logprob_balanced',
    'sequential_quartet_site_logprob_caterpillar',
    'sequential_quartet_site_logprob',
    'cassette_seed_distances',
    'sequential_quartet_loss_fn',
    'SequentialProcess',
    'SequentialModel',
]


# Stand-in for log(0) that is still a real number. Plain -inf must not be used:
# a log-sum-exp over a row that is entirely -inf returns -inf with a NaN
# gradient, and a whole subtree can genuinely be missing.
NEGATIVE_INFINITY = -1e30
# Smallest lam*t we take a logarithm or an incomplete gamma of. Branch lengths
# are clamped at zero further up, so t is often exactly 0, and the incomplete
# gamma has a NaN gradient there.
MIN_RATE_TIME = 1e-30
# Smallest probability we take a logarithm of.
MIN_PROBABILITY = 1e-300
# Floor on any log-probability this module returns, so the sentinel above never
# reaches the loss.
MIN_LOG_PROBABILITY = -700.0


# Reading cassettes out of a character matrix
@dataclass(frozen=True)
class Cassettes:
    """Per-cell cassette summaries, built once per fit.

    Attributes:
        cassette_size: positions per cassette (K above).
        n_cassettes: how many cassettes each cell has.
        alleles: (cells, cassettes, positions). Allele at each position, 0
            where the position is unfilled or the cassette is missing.
        edit_count: (cells, cassettes). Filled positions, 0 to cassette_size.
            0 for a missing cassette.
        is_missing: (cells, cassettes). True where nothing was observed.
        log_prior_upto: (cells, cassettes, cassette_size + 1). Running total of
            the allele-prior logs; entry ``p`` covers positions 0 to ``p - 1``,
            so entry 0 is always 0.
    """

    cassette_size: int
    n_cassettes: int
    alleles: torch.Tensor
    edit_count: torch.Tensor
    is_missing: torch.Tensor
    log_prior_upto: torch.Tensor


def validate_cassette_matrix(states, cassette_size, on_partial="promote"):
    """Check the cassette layout and tidy up half-observed cassettes.

    Cassette ``c`` owns columns ``[c * cassette_size, (c + 1) * cassette_size)``.
    A cassette with some positions missing and others observed is
    *half-observed*; by default the whole cassette is marked missing and the
    count is reported.

    Args:
        states: (cells, sites) integers. 0 unedited, positive allele, -1 missing.
        cassette_size: positions per cassette.
        on_partial: ``"promote"`` to blank half-observed cassettes (default),
            or ``"error"`` to refuse them.

    Returns:
        tuple: the tidied (cells, sites) state tensor, and a summary dict.

    Raises:
        ValueError: if the number of columns is not a whole number of
            cassettes, if a cassette has a filled position sitting above an
            empty one, or on a half-observed cassette under ``"error"``.
    """
    states = torch.as_tensor(states, dtype=torch.long)
    if states.dim() != 2:
        raise ValueError("states must be 2-D (cells, sites), got shape %r"
                         % (tuple(states.shape),))
    cassette_size = int(cassette_size)
    if cassette_size < 1:
        raise ValueError("cassette_size must be at least 1, got %d" % cassette_size)
    n_cells, n_sites = states.shape
    if n_sites % cassette_size:
        raise ValueError(
            "%d columns is not divisible by cassette_size=%d; the character "
            "matrix must hold a whole number of cassettes"
            % (n_sites, cassette_size))
    n_cassettes = n_sites // cassette_size

    blocks = states.view(n_cells, n_cassettes, cassette_size)
    some_missing = (blocks < 0).any(dim=-1)
    all_missing = (blocks < 0).all(dim=-1)
    half_observed = some_missing & ~all_missing
    n_half = int(half_observed.sum())
    total = n_cells * n_cassettes
    if n_half and on_partial == "error":
        raise ValueError(
            "%d of %d cassettes are partially observed (some positions -1, "
            "some not). Pass on_partial='promote' to blank them."
            % (n_half, total))
    if n_half:
        blocks = torch.where(half_observed.unsqueeze(-1),
                             torch.full_like(blocks, -1), blocks)
        states = blocks.reshape(n_cells, n_sites)
        all_missing = (blocks < 0).all(dim=-1)

    # Ordered editing means the filled positions must sit at the front.
    is_filled = blocks > 0
    edit_count = is_filled.sum(dim=-1)
    position = torch.arange(cassette_size, device=states.device).view(1, 1, -1)
    should_be_filled = position < edit_count.unsqueeze(-1)
    out_of_order = ((should_be_filled != is_filled)
                    & ~all_missing.unsqueeze(-1)).any(dim=-1)
    n_out_of_order = int(out_of_order.sum())
    if n_out_of_order:
        cell, cassette = out_of_order.nonzero(as_tuple=False)[0].tolist()
        raise ValueError(
            "%d cassettes have a filled position above an empty one, which "
            "ordered editing forbids (first: cell %d, cassette %d = %r). Check "
            "that cassette_size=%d matches how the data was generated."
            % (n_out_of_order, cell, cassette,
               blocks[cell, cassette].tolist(), cassette_size))

    report = {
        "n_cells": int(n_cells),
        "n_sites": int(n_sites),
        "n_cassettes": int(n_cassettes),
        "cassette_size": cassette_size,
        "n_partial_cassettes": n_half,
        "partial_frac": (n_half / total) if total else 0.0,
        "missing_cassette_frac": float(all_missing.to(torch.float64).mean()),
    }
    return states, report


def _cumulative_log_prior(states, allele_prior, n_cassettes, cassette_size):
    """Running total of allele-prior logs along each cassette."""
    n_cells = states.shape[0]
    prior = _index_prior(allele_prior, states)
    log_prior = torch.log(prior.clamp_min(MIN_PROBABILITY))
    # Unfilled and missing positions contribute nothing.
    log_prior = torch.where(states > 0, log_prior, torch.zeros_like(log_prior))
    log_prior = log_prior.view(n_cells, n_cassettes, cassette_size)
    leading_zero = torch.zeros(n_cells, n_cassettes, 1,
                               dtype=log_prior.dtype, device=log_prior.device)
    return torch.cat([leading_zero, log_prior.cumsum(dim=-1)], dim=-1)


def cassette_summaries(states, allele_prior, cassette_size, on_partial="promote"):
    """Build the :class:`Cassettes` record for a character matrix.

    Args:
        states: (cells, sites) integers, founder row included.
        allele_prior: (alleles,) shared prior, or (sites, alleles) per-site
            prior, with column 0 set to 1.
        cassette_size: positions per cassette.
        on_partial: passed to :func:`validate_cassette_matrix`.

    Returns:
        tuple: the :class:`Cassettes` record and the validation summary.
    """
    states, report = validate_cassette_matrix(states, cassette_size,
                                              on_partial=on_partial)
    n_cells, n_sites = states.shape
    n_cassettes = n_sites // cassette_size
    allele_prior = torch.as_tensor(allele_prior, dtype=torch.float64,
                                   device=states.device)
    blocks = states.view(n_cells, n_cassettes, cassette_size)
    return Cassettes(
        cassette_size=cassette_size,
        n_cassettes=n_cassettes,
        alleles=blocks.clamp_min(0),
        edit_count=(blocks > 0).sum(dim=-1),
        is_missing=(blocks < 0).all(dim=-1),
        log_prior_upto=_cumulative_log_prior(states, allele_prior,
                                             n_cassettes, cassette_size),
    ), report


def with_allele_prior(cassettes, allele_prior):
    """Return a copy of ``cassettes`` with ``log_prior_upto`` recomputed for *allele_prior*.

    Used each step when the allele prior is learned; nothing else changes.
    """
    states = cassettes.alleles.reshape(
        cassettes.alleles.shape[0], -1)
    log_prior_upto = _cumulative_log_prior(
        states, allele_prior, cassettes.n_cassettes,
        cassettes.cassette_size)
    return replace(cassettes, log_prior_upto=log_prior_upto)


@dataclass(frozen=True)
class _Leaves:
    """One leaf slot of a batch, sliced out of :class:`Cassettes`."""

    alleles: torch.Tensor        # (batch, cassettes, positions)
    edit_count: torch.Tensor     # (batch, cassettes)
    is_missing: torch.Tensor     # (batch, cassettes)
    log_prior_upto: torch.Tensor  # (batch, cassettes, positions + 1)


def _select_leaves(cassettes, leaf_index):
    """Gather (batch, slots) cell indices into one :class:`_Leaves` per slot."""
    alleles = cassettes.alleles[leaf_index]
    edit_count = cassettes.edit_count[leaf_index]
    is_missing = cassettes.is_missing[leaf_index]
    log_prior_upto = cassettes.log_prior_upto[leaf_index]
    return [_Leaves(alleles[:, s], edit_count[:, s], is_missing[:, s],
                    log_prior_upto[:, s])
            for s in range(leaf_index.shape[1])]


# How much of the edit history two or more cells share
def shared_prefix_length(alleles, is_missing):
    """How many leading positions every observed cell agrees on.

    Missing cells are skipped, not counted as disagreeing.

    Args:
        alleles: (batch, cassettes, positions, cells).
        is_missing: (batch, cassettes, cells).

    Returns:
        tuple: the shared length (batch, cassettes), equal to the cassette size
        when every cell is missing; and the slot of the first observed cell
        (batch, cassettes), whose allele priors serve as the reference.
    """
    n_batch, n_cassettes, n_positions, _ = alleles.shape
    is_observed = ~is_missing
    any_observed = is_observed.any(dim=-1)
    first_observed = is_observed.to(torch.uint8).argmax(dim=-1)
    reference = torch.gather(
        alleles, 3,
        first_observed.view(n_batch, n_cassettes, 1, 1)
        .expand(n_batch, n_cassettes, n_positions, 1)).squeeze(-1)

    # A position extends the run when every observed cell carries the same
    # filled allele there.
    agrees = (~is_observed).unsqueeze(2) | (alleles == reference.unsqueeze(-1))
    matches = agrees.all(dim=-1) & (reference > 0)
    run_length = (matches.logical_not().cumsum(dim=-1) == 0).sum(dim=-1)
    full = torch.full_like(run_length, n_positions)
    return torch.where(any_observed, run_length, full), first_observed


def _reference_log_prior(leaves, first_observed):
    """Pick each row's reference prior totals from a list of leaf slots."""
    stacked = torch.stack([leaf.log_prior_upto for leaf in leaves], dim=-1)
    n_batch, n_cassettes, n_states, _ = stacked.shape
    index = first_observed.view(n_batch, n_cassettes, 1, 1).expand(
        n_batch, n_cassettes, n_states, 1)
    return torch.gather(stacked, 3, index).squeeze(-1)


def _prefix_and_reference(leaves):
    """Shared prefix length and reference priors for a set of leaf slots."""
    length, first_observed = shared_prefix_length(
        torch.stack([leaf.alleles for leaf in leaves], dim=-1),
        torch.stack([leaf.is_missing for leaf in leaves], dim=-1))
    return length, _reference_log_prior(leaves, first_observed)


# Chance of the edit count moving along a branch
def log_transition(rate, branch_length, cassette_size, rate_model="equal"):
    """Log transition probabilities between edit counts along each branch.

    Args:
        rate: a single rate (``"equal"``), or a ``(cassette_size,)`` vector with
            one rate per step (``"matrix_exp"``, ``"two_rate"``).
        branch_length: (batch,) branch lengths.
        cassette_size: positions per cassette; the top count is absorbing.
        rate_model: ``"equal"``: closed form in the module docstring.
            ``"matrix_exp"``: exponential of the rate matrix, any rate per step.
            ``"two_rate"``: closed form when only the first step's rate differs
            (falls back to ``"matrix_exp"`` when that rate is the smaller one).

    Returns:
        torch.Tensor: (batch, 1, counts, counts) with ``counts = cassette_size + 1``;
        entry ``[.., i, j]`` is ``log P(i -> j)``.  The size-1 cassette axis
        broadcasts, since the values do not depend on the cassette.

    Raises:
        ValueError: unknown ``rate_model``.
    """
    if rate_model == "equal":
        return _log_transition_equal(rate, branch_length, cassette_size)
    if rate_model == "matrix_exp":
        return _log_transition_matrix_exp(rate, branch_length, cassette_size)
    if rate_model == "two_rate":
        return _log_transition_two_rate(rate, branch_length, cassette_size)
    raise ValueError("rate_model must be 'equal', 'matrix_exp', or 'two_rate', "
                     "got %r" % (rate_model,))


def _log_transition_equal(rate, branch_length, cassette_size):
    """Closed form for one rate throughout. See :func:`log_transition`."""
    branch_length = torch.as_tensor(branch_length, dtype=torch.float64)
    rate = torch.as_tensor(rate, dtype=torch.float64, device=branch_length.device)
    rate_time = (rate * branch_length).clamp_min(MIN_RATE_TIME).reshape(-1, 1, 1)
    device, dtype = rate_time.device, rate_time.dtype

    start = torch.arange(cassette_size + 1, device=device).view(-1, 1)
    end = torch.arange(cassette_size + 1, device=device).view(1, -1)

    # Both branches below are evaluated everywhere and picked between
    # afterwards. An unselected branch that came out as inf or NaN would still
    # be multiplied by zero on the way back through the gradient, and zero
    # times NaN is NaN, so the arguments are clamped into range first.
    n_steps = (end - start).clamp_min(0).to(dtype)
    log_poisson = (-rate_time + n_steps * torch.log(rate_time)
                   - torch.lgamma(n_steps + 1.0))

    steps_to_top = (cassette_size - start).clamp_min(1).to(dtype)
    log_reached_top = torch.log(
        torch.special.gammainc(steps_to_top, rate_time).clamp_min(MIN_PROBABILITY)
    ).expand(-1, cassette_size + 1, cassette_size + 1)

    forward = end >= start
    top_column = end == cassette_size
    already_at_top = start == cassette_size
    return torch.where(
        forward & already_at_top, torch.zeros_like(log_poisson),
        torch.where(
            forward & top_column, log_reached_top,
            torch.where(forward, log_poisson,
                        torch.full_like(log_poisson, NEGATIVE_INFINITY)))
    ).unsqueeze(1)


def _log_transition_matrix_exp(rate, branch_length, cassette_size):
    """Rate-matrix exponential, allowing a rate per step. See :func:`log_transition`."""
    branch_length = torch.as_tensor(branch_length, dtype=torch.float64)
    rate = torch.as_tensor(rate, dtype=torch.float64, device=branch_length.device)
    if rate.dim() == 0:
        rate = rate.expand(cassette_size)
    if rate.shape != (cassette_size,):
        raise ValueError("matrix_exp rate must be a single value or shape (%d,), "
                         "got %r" % (cassette_size, tuple(rate.shape)))
    # Leaving a count at its own rate, arriving at the next one; the top count
    # has no outgoing rate.
    leaving = torch.cat([-rate, torch.zeros(1, dtype=rate.dtype,
                                            device=rate.device)])
    rate_matrix = torch.diag_embed(leaving) + torch.diag_embed(rate, offset=1)
    probabilities = torch.linalg.matrix_exp(
        rate_matrix.unsqueeze(0) * branch_length.reshape(-1, 1, 1))
    return torch.log(probabilities.clamp_min(MIN_PROBABILITY)).unsqueeze(1)


def _log_transition_two_rate(rate, branch_length, cassette_size):
    """Closed form for ``rate = [alpha * lam, lam, ..., lam]``. See :func:`log_transition`.

    Uses :func:`_log_transition_matrix_exp` when ``alpha < 1``.
    """
    branch_length = torch.as_tensor(branch_length, dtype=torch.float64)
    rate = torch.as_tensor(rate, dtype=torch.float64, device=branch_length.device)

    alpha_rate = rate[0]
    cont_rate = rate[1]

    if alpha_rate < cont_rate:
        return _log_transition_matrix_exp(rate, branch_length, cassette_size)

    result = _log_transition_equal(cont_rate, branch_length, cassette_size)

    K = cassette_size
    t = branch_length.clamp_min(MIN_RATE_TIME)
    alpha = alpha_rate / cont_rate
    beta = alpha_rate - cont_rate

    exp_neg_at = torch.exp(-alpha_rate * t)
    p_row = [exp_neg_at]

    exp_bt = torch.exp(beta * t)
    # expm1(beta*t)/beta -> t as beta -> 0; guard the singularity for a
    # learned initiation_ratio passing through 1.0.
    beta_safe = beta + (beta == 0).to(beta.dtype)
    I_j = torch.where(beta.abs() < 1e-12, t, torch.expm1(beta * t) / beta_safe)
    coeff = alpha * exp_neg_at

    factorial = 1.0
    for j in range(1, K):
        p_row.append(coeff * cont_rate.pow(j) / factorial * I_j)
        if j < K - 1:
            I_j_rec = (t.pow(j) * exp_bt - j * I_j) / beta_safe
            I_j = torch.where(beta.abs() < 1e-12,
                              t.pow(j + 1) / (j + 1), I_j_rec)
        factorial *= j

    p_0_K = -torch.expm1(-alpha_rate * t)
    for pj in p_row[1:]:
        p_0_K = p_0_K - pj
    p_row.append(p_0_K.clamp_min(0.0))

    log_row_0 = torch.log(
        torch.stack(p_row, dim=-1).clamp_min(MIN_PROBABILITY))
    log_row_0 = log_row_0.unsqueeze(1).unsqueeze(2)
    return torch.cat([log_row_0, result[:, :, 1:, :]], dim=2)


# The two building blocks everything else is made of
def leaf_column(log_transitions, edit_count, log_prior_upto):
    """Log-probability of a leaf's cassettes for each edit count at the top of its branch.

    Entry ``s`` covers the count climbing from ``s`` to the leaf's own and an
    allele being drawn at every position past ``s``.

    Args:
        log_transitions: (batch, 1, counts, counts) for the leaf's branch.
        edit_count: (batch, cassettes) filled positions in the leaf.
        log_prior_upto: (batch, cassettes, counts) prior totals for the leaf.

    Returns:
        torch.Tensor: (batch, cassettes, counts), set to the sentinel where the
        count above already exceeds the leaf's own.
    """
    n_batch, n_cassettes = edit_count.shape
    n_states = log_prior_upto.shape[-1]
    transitions = log_transitions[:, 0]
    # Take column `edit_count` for every cassette without ever building the
    # full (batch, cassettes, counts, counts) tensor.
    index = edit_count.unsqueeze(1).expand(n_batch, n_states, n_cassettes)
    column = torch.gather(transitions, 2, index).transpose(1, 2)
    prior_at_end = torch.gather(log_prior_upto, 2, edit_count.unsqueeze(-1))
    result = column + prior_at_end - log_prior_upto
    state = torch.arange(n_states, device=edit_count.device).view(1, 1, -1)
    return torch.where(state <= edit_count.unsqueeze(-1), result,
                       torch.full_like(result, NEGATIVE_INFINITY))


def push_up_branch(log_transitions, child_factors, reference_log_prior,
                   shared_prefix):
    """Carry log factors from the bottom of a branch to the top.

    With ``ancestor`` the count at the top and ``mrca`` the count at the bottom::

        out[ancestor] = log-sum-exp over mrca of
              log_transitions[ancestor, mrca]
            + (reference priors from ancestor + 1 to mrca)
            + sum of the child factors at mrca

    where ``mrca`` runs from ``ancestor`` up to ``shared_prefix``.  Every pair,
    triple and quartet likelihood in this module is built from this step.

    Args:
        log_transitions: (batch, 1, counts, counts) for this branch.
        child_factors: one or more (batch, cassettes, counts) tensors, each
            indexed by the count at the bottom of the branch.
        reference_log_prior: (batch, cassettes, counts) prior totals.
        shared_prefix: (batch, cassettes) agreement length below the branch.

    Returns:
        torch.Tensor: (batch, cassettes, counts), indexed by the count at the
        top.  Peak memory is one (batch, cassettes, counts, counts) tensor.
    """
    below = child_factors[0]
    for factor in child_factors[1:]:
        below = below + factor
    below = below + reference_log_prior
    n_states = below.shape[-1]
    mrca = torch.arange(n_states, device=below.device).view(1, 1, -1)
    below = torch.where(mrca <= shared_prefix.unsqueeze(-1), below,
                        torch.full_like(below, NEGATIVE_INFINITY))
    # (batch, 1, ancestor, mrca) against (batch, cassettes, 1, mrca).
    combined = log_transitions + below.unsqueeze(2)
    return torch.logsumexp(combined, dim=3) - reference_log_prior


# Chances of actually seeing the data (deletion and dropout)
def _safe_log(probability):
    return torch.log(probability.clamp_min(MIN_PROBABILITY))


def _survives_deletion(deletion_rate, branch_length):
    """Chance a lineage is not deleted along a branch, shaped to broadcast."""
    return torch.exp(-deletion_rate * branch_length).reshape(-1, 1, 1)


def _is_observed(deletion_rate, dropout, branch_length):
    """Chance a leaf's cassette is actually read out, shaped to broadcast."""
    return torch.exp(-deletion_rate * branch_length).reshape(-1, 1, 1) * (1 - dropout)


def _weight_by_observation(column, is_missing, observed):
    """Fold a leaf's chance of being read into its column of counts."""
    return torch.where((~is_missing).unsqueeze(-1),
                       _safe_log(observed) + column,
                       _safe_log(1 - observed).expand_as(column))


def _add_deletion_alternative(factor, survives, everything_missing):
    """Add the "branch deleted, nothing below seen" term, as in the Cas9 kernels.

    Computes ``survives * factor + (1 - survives) * 1[all missing]`` in log space.
    """
    deleted = torch.where(everything_missing.unsqueeze(-1),
                          _safe_log(1 - survives).expand_as(factor),
                          torch.full_like(factor, NEGATIVE_INFINITY))
    return torch.logaddexp(_safe_log(survives) + factor, deleted)


# Pair, triple and quartet factors, one entry per possible ancestor count
def _pair_factor(cassette_size, leaf_i, leaf_j, stem, branch_i, branch_j,
                 rate, deletion_rate, dropout, rate_model, all_trans=None):
    """Cherry log factor, one entry per edit count at the top of the stem.

    Handles the four observed/missing cases of the two leaves as
    :func:`heracles.embedding.ctmc.pair_edit_joint_missing` does, per cassette.
    ``all_trans`` optionally supplies precomputed transitions for the branches
    ``[branch_i, branch_j, stem, stem + branch_i, stem + branch_j]``.
    """
    prefix, reference = _prefix_and_reference([leaf_i, leaf_j])

    B = stem.shape[0]
    if all_trans is None:
        all_trans = log_transition(
            rate,
            torch.cat([branch_i, branch_j, stem,
                       stem + branch_i, stem + branch_j]),
            cassette_size, rate_model)
    trans_i = all_trans[:B]
    trans_j = all_trans[B:2*B]
    trans_stem = all_trans[2*B:3*B]
    trans_sum_i = all_trans[3*B:4*B]
    trans_sum_j = all_trans[4*B:]

    column_i = leaf_column(trans_i, leaf_i.edit_count, leaf_i.log_prior_upto)
    column_j = leaf_column(trans_j, leaf_j.edit_count, leaf_j.log_prior_upto)
    both_seen = push_up_branch(trans_stem, [column_i, column_j], reference,
                               prefix)

    only_i = leaf_column(trans_sum_i, leaf_i.edit_count, leaf_i.log_prior_upto)
    only_j = leaf_column(trans_sum_j, leaf_j.edit_count, leaf_j.log_prior_upto)

    survives = _survives_deletion(deletion_rate, stem)
    seen_i = _is_observed(deletion_rate, dropout, branch_i)
    seen_j = _is_observed(deletion_rate, dropout, branch_j)

    observed_i = (~leaf_i.is_missing).unsqueeze(-1)
    observed_j = (~leaf_j.is_missing).unsqueeze(-1)
    neither = _safe_log((1 - survives) + survives * (1 - seen_i) * (1 - seen_j))
    return torch.where(
        observed_i & observed_j,
        _safe_log(survives * seen_i * seen_j) + both_seen,
        torch.where(
            observed_i,
            _safe_log(survives * seen_i * (1 - seen_j)) + only_i,
            torch.where(
                observed_j,
                _safe_log(survives * (1 - seen_i) * seen_j) + only_j,
                neither.expand_as(both_seen))))


def _triple_factor(cassette_size, leaves, root_to_split, split_to_cherry,
                   cherry_branch_1, cherry_branch_2, outgroup_branch,
                   rate, deletion_rate, dropout, rate_model,
                   cherry_pair_trans=None, outgroup_trans=None,
                   root_to_split_trans=None):
    """Rooted-triple log factor, one entry per count above ``root_to_split``.

    ``leaves`` are three :class:`_Leaves` ordered cherry, cherry, outgroup; the
    ``*_trans`` arguments optionally supply precomputed transitions.
    """
    cherry_1, cherry_2, outgroup = leaves
    cherry = _pair_factor(cassette_size, cherry_1, cherry_2, split_to_cherry,
                          cherry_branch_1, cherry_branch_2, rate,
                          deletion_rate, dropout, rate_model,
                          all_trans=cherry_pair_trans)

    if outgroup_trans is None:
        outgroup_trans = log_transition(rate, outgroup_branch, cassette_size,
                                        rate_model)
    outgroup_column = leaf_column(
        outgroup_trans, outgroup.edit_count, outgroup.log_prior_upto)
    outgroup_column = _weight_by_observation(
        outgroup_column, outgroup.is_missing,
        _is_observed(deletion_rate, dropout, outgroup_branch))

    prefix, reference = _prefix_and_reference(leaves)
    if root_to_split_trans is None:
        root_to_split_trans = log_transition(rate, root_to_split,
                                             cassette_size, rate_model)
    factor = push_up_branch(
        root_to_split_trans,
        [cherry, outgroup_column], reference, prefix)
    everything_missing = (cherry_1.is_missing & cherry_2.is_missing
                          & outgroup.is_missing)
    return _add_deletion_alternative(
        factor, _survives_deletion(deletion_rate, root_to_split),
        everything_missing)


def sequential_leaf_marginal_logprob(cassettes, depths, rate, leaf_index,
                                     heritable_eps=0.0, stochastic_delta=0.0,
                                     rate_model="equal"):
    """Log-probability of each leaf's cassettes given its depth below the founder.

    This root-to-leaf term fixes how far each leaf sits from the root, which
    quartets alone do not.

    Args:
        cassettes: the :class:`Cassettes` record.
        depths: (cells,) distance from the founder.
        rate: edit rate, as in :func:`log_transition`.
        leaf_index: (leaves,) cell indices to score.
        heritable_eps: deletion rate per unit branch length.
        stochastic_delta: per-cell dropout probability.

    Returns:
        torch.Tensor: (leaves, cassettes).
    """
    leaf, = _select_leaves(cassettes, leaf_index.unsqueeze(1))
    depth = depths[leaf_index]
    column = leaf_column(
        log_transition(rate, depth, cassettes.cassette_size, rate_model),
        leaf.edit_count, leaf.log_prior_upto)
    column = _weight_by_observation(
        column, leaf.is_missing,
        _is_observed(heritable_eps, stochastic_delta, depth))
    return column[..., 0].clamp_min(MIN_LOG_PROBABILITY)


def sequential_triple_site_logprob(cassettes, leaf_index, root_to_split,
                                   split_to_cherry, cherry_branch_1,
                                   cherry_branch_2, outgroup_branch, rate,
                                   heritable_eps=0.0, stochastic_delta=0.0,
                                   rate_model="equal"):
    """Log-probability of a rooted triple's cassettes, shape (triples, cassettes).

    ``leaf_index`` is (triples, 3), ordered cherry, cherry, outgroup.
    """
    factor = _triple_factor(
        cassettes.cassette_size, _select_leaves(cassettes, leaf_index),
        root_to_split, split_to_cherry, cherry_branch_1, cherry_branch_2,
        outgroup_branch, rate, heritable_eps, stochastic_delta, rate_model)
    return factor[..., 0].clamp_min(MIN_LOG_PROBABILITY)


def sequential_quartet_site_logprob_balanced(cassettes, leaf_index, root_branch,
                                             left_stem, left_branch_1,
                                             left_branch_2, right_stem,
                                             right_branch_1, right_branch_2,
                                             rate, heritable_eps=0.0,
                                             stochastic_delta=0.0,
                                             rate_model="equal"):
    """Log-probability of balanced quartets ((0,1),(2,3)), shape (quartets, cassettes)."""
    cassette_size = cassettes.cassette_size
    leaves = _select_leaves(cassettes, leaf_index)

    B = left_stem.shape[0]
    all_trans = log_transition(
        rate,
        torch.cat([left_branch_1, left_branch_2, left_stem,
                   left_stem + left_branch_1, left_stem + left_branch_2,
                   right_branch_1, right_branch_2, right_stem,
                   right_stem + right_branch_1, right_stem + right_branch_2,
                   root_branch]),
        cassette_size, rate_model)

    left = _pair_factor(cassette_size, leaves[0], leaves[1], left_stem,
                        left_branch_1, left_branch_2, rate, heritable_eps,
                        stochastic_delta, rate_model,
                        all_trans=all_trans[:5 * B])
    right = _pair_factor(cassette_size, leaves[2], leaves[3], right_stem,
                         right_branch_1, right_branch_2, rate, heritable_eps,
                         stochastic_delta, rate_model,
                         all_trans=all_trans[5 * B:10 * B])

    prefix, reference = _prefix_and_reference(leaves)
    factor = push_up_branch(
        all_trans[10 * B:],
        [left, right], reference, prefix)
    everything_missing = (leaves[0].is_missing & leaves[1].is_missing
                          & leaves[2].is_missing & leaves[3].is_missing)
    factor = _add_deletion_alternative(
        factor, _survives_deletion(heritable_eps, root_branch),
        everything_missing)
    return factor[..., 0].clamp_min(MIN_LOG_PROBABILITY)


def sequential_quartet_site_logprob_caterpillar(cassettes, outgroup_index,
                                                sub_triple_index, root_branch,
                                                outgroup_branch,
                                                sub_root_to_split,
                                                sub_split_to_cherry,
                                                sub_cherry_branch_1,
                                                sub_cherry_branch_2,
                                                sub_outgroup_branch, rate,
                                                heritable_eps=0.0,
                                                stochastic_delta=0.0,
                                                rate_model="equal"):
    """Log-probability of caterpillar quartets (outgroup plus a rooted triple).

    Stays in log space throughout, unlike the Cas9 caterpillar, so small
    cassette probabilities do not underflow.

    Returns:
        torch.Tensor: (quartets, cassettes).
    """
    cassette_size = cassettes.cassette_size
    outgroup, = _select_leaves(cassettes, outgroup_index)
    sub_leaves = _select_leaves(cassettes, sub_triple_index)

    B = root_branch.shape[0]
    all_trans = log_transition(
        rate,
        torch.cat([outgroup_branch,
                   sub_cherry_branch_1, sub_cherry_branch_2,
                   sub_split_to_cherry,
                   sub_split_to_cherry + sub_cherry_branch_1,
                   sub_split_to_cherry + sub_cherry_branch_2,
                   sub_outgroup_branch,
                   sub_root_to_split,
                   root_branch]),
        cassette_size, rate_model)

    outgroup_column = leaf_column(
        all_trans[:B], outgroup.edit_count, outgroup.log_prior_upto)
    outgroup_column = _weight_by_observation(
        outgroup_column, outgroup.is_missing,
        _is_observed(heritable_eps, stochastic_delta, outgroup_branch))

    sub_triple = _triple_factor(
        cassette_size, sub_leaves, sub_root_to_split, sub_split_to_cherry,
        sub_cherry_branch_1, sub_cherry_branch_2, sub_outgroup_branch, rate,
        heritable_eps, stochastic_delta, rate_model,
        cherry_pair_trans=all_trans[B:6 * B],
        outgroup_trans=all_trans[6 * B:7 * B],
        root_to_split_trans=all_trans[7 * B:8 * B])

    all_leaves = [outgroup] + sub_leaves
    prefix, reference = _prefix_and_reference(all_leaves)
    factor = push_up_branch(
        all_trans[8 * B:],
        [outgroup_column, sub_triple], reference, prefix)
    everything_missing = all_leaves[0].is_missing
    for leaf in all_leaves[1:]:
        everything_missing = everything_missing & leaf.is_missing
    factor = _add_deletion_alternative(
        factor, _survives_deletion(heritable_eps, root_branch),
        everything_missing)
    return factor[..., 0].clamp_min(MIN_LOG_PROBABILITY)


def sequential_quartet_site_logprob(cassettes, is_balanced,
                                    balanced_index, balanced_root,
                                    left_stem, left_branch_1, left_branch_2,
                                    right_stem, right_branch_1, right_branch_2,
                                    outgroup_index, sub_triple_index,
                                    caterpillar_root, outgroup_branch,
                                    sub_root_to_split, sub_split_to_cherry,
                                    sub_cherry_branch_1, sub_cherry_branch_2,
                                    sub_outgroup_branch, rate,
                                    heritable_eps=0.0, stochastic_delta=0.0,
                                    rate_model="equal"):
    """Log-probability of each quartet under its own topology, shape (quartets, cassettes).

    ``is_balanced`` selects, per quartet, the balanced or the caterpillar
    arguments; the remaining arguments are the branch lengths from
    :func:`heracles.embedding.objective.quartet_branch_lengths`.
    """
    n_bal = is_balanced.sum().item()
    n_batch = is_balanced.shape[0]

    if n_bal == n_batch:
        return sequential_quartet_site_logprob_balanced(
            cassettes, balanced_index, balanced_root,
            left_stem, left_branch_1, left_branch_2,
            right_stem, right_branch_1, right_branch_2, rate,
            heritable_eps=heritable_eps, stochastic_delta=stochastic_delta,
            rate_model=rate_model)

    if n_bal == 0:
        return sequential_quartet_site_logprob_caterpillar(
            cassettes, outgroup_index, sub_triple_index,
            caterpillar_root, outgroup_branch,
            sub_root_to_split, sub_split_to_cherry,
            sub_cherry_branch_1, sub_cherry_branch_2,
            sub_outgroup_branch, rate,
            heritable_eps=heritable_eps, stochastic_delta=stochastic_delta,
            rate_model=rate_model)

    bal_idx = is_balanced.nonzero(as_tuple=True)[0]
    cat_idx = (~is_balanced).nonzero(as_tuple=True)[0]

    bal_vals = sequential_quartet_site_logprob_balanced(
        cassettes, balanced_index[bal_idx], balanced_root[bal_idx],
        left_stem[bal_idx], left_branch_1[bal_idx], left_branch_2[bal_idx],
        right_stem[bal_idx], right_branch_1[bal_idx], right_branch_2[bal_idx],
        rate, heritable_eps=heritable_eps, stochastic_delta=stochastic_delta,
        rate_model=rate_model)

    cat_vals = sequential_quartet_site_logprob_caterpillar(
        cassettes, outgroup_index[cat_idx], sub_triple_index[cat_idx],
        caterpillar_root[cat_idx], outgroup_branch[cat_idx],
        sub_root_to_split[cat_idx], sub_split_to_cherry[cat_idx],
        sub_cherry_branch_1[cat_idx], sub_cherry_branch_2[cat_idx],
        sub_outgroup_branch[cat_idx], rate,
        heritable_eps=heritable_eps, stochastic_delta=stochastic_delta,
        rate_model=rate_model)

    all_idx = torch.cat([bal_idx, cat_idx])
    all_vals = torch.cat([bal_vals, cat_vals], dim=0)
    _, unsort = all_idx.sort()
    return all_vals[unsort]


# Starting distances for the embedding
def cassette_seed_distances(states, cassette_size, on_partial="promote",
                            chunk=HAMMING_ROW_CHUNK):
    """Rough pairwise distances to start the embedding from.

    Averages ``edits_i + edits_j - 2 * shared_prefix`` over the cassettes read
    in both cells: the edit steps the two cells do not share, in units of
    expected edits.  Values run from 0 to twice the cassette size.  At cassette
    size 1 this equals the Cas9 weighted Hamming distance.

    Returns:
        numpy.ndarray: (cells, cells) distances.
    """
    states, _ = validate_cassette_matrix(states, cassette_size,
                                         on_partial=on_partial)
    states = states.cpu()
    n_cells, n_sites = states.shape
    n_cassettes = n_sites // cassette_size
    blocks = states.view(n_cells, n_cassettes, cassette_size)
    is_missing = (blocks < 0).all(dim=-1)
    edit_count = (blocks > 0).sum(dim=-1)
    alleles = blocks.clamp_min(0)

    distances = torch.zeros(n_cells, n_cells, dtype=torch.float64)
    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        rows = alleles[start:stop].unsqueeze(1)
        both_read = (~is_missing[start:stop]).unsqueeze(1) & (~is_missing).unsqueeze(0)
        matches = (rows == alleles.unsqueeze(0)) & (rows > 0)
        shared = (matches.logical_not().cumsum(dim=-1) == 0).sum(dim=-1)
        unshared = (edit_count[start:stop].unsqueeze(1)
                    + edit_count.unsqueeze(0) - 2 * shared).to(torch.float64)
        n_usable = both_read.sum(dim=-1).to(torch.float64)
        distances[start:stop] = ((unshared * both_read.to(torch.float64)).sum(dim=-1)
                                 / n_usable.clamp_min(1.0))
    distances.fill_diagonal_(0.0)
    return distances.numpy()


# Loss and entry point
@with_learnable_prior
def sequential_quartet_loss_fn(states, get_allele_prior, m, *, est_dists=None,
                               batch_size=None,
                               cassette_size=None,
                               n_quartets=None, root_frac=1.0, root_weight=None,
                               quartet_sampler="banded",
                               band_kmin=None, band_kmax=None, band_kmax_end="default",
                               band_local_frac=None, band_local_mult_max="default",
                               rate_model="equal", initiation_ratio=1.0,
                               learn_initiation_ratio=False,
                               on_partial="promote",
                               device=None, config=None):
    """Build the sequential quartet loss (the :class:`SequentialModel` objective).

    Same contract and sampler options as
    :func:`heracles.embedding.objective.quartet_loss_fn`; the returned callable
    follows the ``loss_fn`` contract in :mod:`heracles.models.base`.  The loss is
    divided by ``cassette_size`` so it is on the same per-site scale as the Cas9
    loss.

    Args:
        states: (cells, sites) founder-augmented states.
        get_allele_prior: allele prior (wrapped by ``with_learnable_prior``).
        m: number of rows of *states*, founder included.
        est_dists: (m, m) seed distances that aim the quartet sampler.
        cassette_size: positions per cassette (required).
        initiation_ratio: rate of a cassette's first edit over the rate of the
            following ones; 1.0 keeps a single rate, anything else uses the
            ``"two_rate"`` transitions.
        learn_initiation_ratio: learn the ratio (exposed as
            ``loss_fn.initiation_ratio_param``, on the log scale).
        on_partial: passed to :func:`validate_cassette_matrix`.

    Raises:
        ValueError: missing ``cassette_size``, fewer than 4 leaves, an unknown
            sampler, or ``root_weight`` outside [0, 1].
    """
    from heracles.embedding.objective import quartet_objective

    if cassette_size is None:
        raise ValueError("cassette_size is required for sequential_quartet_loss_fn")
    if quartet_sampler != "banded":
        raise ValueError("only the banded quartet sampler is available, got %r"
                         % (quartet_sampler,))
    state_matrix = torch.as_tensor(states, dtype=torch.long)
    prior = torch.as_tensor(get_allele_prior(), dtype=torch.float64)
    if device is not None:
        target = torch.device(device)
        state_matrix, prior = state_matrix.to(target), prior.to(target)
    cassettes, cassette_report = cassette_summaries(
        state_matrix, prior, cassette_size, on_partial=on_partial)

    # A first edit at a different rate from the rest uses the "two_rate"
    # transitions; the kernels pass `rate` straight to log_transition, so
    # score() only has to build the per-step rate vector.
    initiation_ratio = float(initiation_ratio)
    step_scale = log_init_ratio = None
    if learn_initiation_ratio:
        ratio_floor = 1e-4
        log_init_ratio = torch.nn.Parameter(torch.log(torch.tensor(
            max(initiation_ratio, ratio_floor), dtype=torch.float64,
            device=state_matrix.device)))
        rate_model = "two_rate"
    elif initiation_ratio != 1.0:
        rate_model = "two_rate"
        step_scale = torch.ones(cassette_size, dtype=torch.float64, device=state_matrix.device)
        step_scale[0] = initiation_ratio

    def score(st, depths, branches, leaves, mutation_rate, heritable_eps, stochastic_delta):
        current_cassettes = with_allele_prior(cassettes, get_allele_prior())
        if log_init_ratio is not None:
            scale = torch.ones(cassette_size, dtype=torch.float64, device=mutation_rate.device)
            scale[0] = torch.exp(log_init_ratio)
            rate = mutation_rate * scale
        elif step_scale is not None:
            rate = mutation_rate * step_scale
        else:
            rate = mutation_rate
        log_prob_quartets = sequential_quartet_site_logprob(
            current_cassettes, *branches, rate, heritable_eps=heritable_eps,
            stochastic_delta=stochastic_delta, rate_model=rate_model)
        log_prob_root = sequential_leaf_marginal_logprob(
            current_cassettes, depths, rate, leaves, heritable_eps=heritable_eps,
            stochastic_delta=stochastic_delta, rate_model=rate_model)
        return log_prob_quartets, log_prob_root

    # one cassette log-probability covers cassette_size sites: dividing by it
    # puts the loss on the per-site scale of the Cas9 objective
    loss_fn = quartet_objective(
        states, m, score, est_dists=est_dists,
        batch_size=n_quartets if n_quartets is not None else batch_size,
        root_frac=root_frac, root_weight=root_weight, config=config, device=device,
        loss_scale=cassette_size, band_kmin=band_kmin, band_kmax=band_kmax,
        band_kmax_end=band_kmax_end, band_local_frac=band_local_frac,
        band_local_mult_max=band_local_mult_max)
    loss_fn.cassette_size = cassette_size
    loss_fn.cassette_report = cassette_report
    loss_fn.rate_model = rate_model
    loss_fn.initiation_ratio = initiation_ratio
    loss_fn.initiation_ratio_param = log_init_ratio
    loss_fn.meta.update({"objective": "sequential_quartet", "cassette_size": cassette_size,
                         "cassette_report": cassette_report, "rate_model": rate_model,
                         "initiation_ratio": initiation_ratio})
    return loss_fn


@register_process
@dataclass
class SequentialProcess(EditProcess):
    """Sites inside a cassette are edited in a fixed order (DNA Typewriter, PASTE).

    Each cassette is ``cassette_size`` consecutive columns whose positions fill
    one at a time, each with its own allele; cassettes are independent.  At
    ``cassette_size=1`` the likelihood equals the Cas9 one.  Scores only
    ``CassetteDropout`` of the same size with one shared dropout rate.

    Args:
        cassette_size: sites per cassette, filled in order (required).
        initiation_ratio: rate of a cassette's first edit over the rate of the
            following ones (1.0 = one rate throughout).
        learn_initiation_ratio: learn ``initiation_ratio`` during the fit.
    """

    name: ClassVar[str] = "sequential"
    supported_missing: ClassVar[tuple] = (CassetteDropout,)

    cassette_size: int = model_option(None, "sites per cassette (required for sequential)",
                                      parse=int)
    initiation_ratio: float = model_option(1.0, "first-edit rate over continuation rate",
                                           parse=float)
    learn_initiation_ratio: bool = model_option(False, "learn initiation_ratio", type=bool)

    def __post_init__(self):
        if self.cassette_size is None:
            raise ValueError("SequentialModel needs cassette_size")
        self.cassette_size = int(self.cassette_size)
        if self.cassette_size < 1:
            raise ValueError("cassette_size must be >= 1, got %r" % (self.cassette_size,))
        self.initiation_ratio = float(self.initiation_ratio)

    def validate(self, n_sites, config, missing):
        super().validate(n_sites, config, missing)
        if missing.cassette_size != self.cassette_size or missing.separate_rates:
            raise ValueError("SequentialProcess scores CassetteDropout(cassette_size=%d) with "
                             "one shared rate" % self.cassette_size)
        if n_sites is not None and n_sites % self.cassette_size:
            raise ValueError("SequentialModel needs the %d sites to split into whole cassettes "
                             "of %d consecutive columns" % (n_sites, self.cassette_size))

    def seed_distances(self, states, *, dtype=np.float64):
        """Unshared edit steps averaged over cassettes (:func:`cassette_seed_distances`)."""
        return cassette_seed_distances(states, self.cassette_size)

    def loss_function(self, states, allele_prior, *, missing, seed_dists, batch_size,
                      root_weight, learn_allele_prior, config, device):
        """The sequential quartet likelihood (:func:`sequential_quartet_loss_fn`)."""
        return sequential_quartet_loss_fn(
            states, allele_prior, states.shape[0], est_dists=seed_dists,
            batch_size=batch_size, root_weight=root_weight,
            cassette_size=self.cassette_size, initiation_ratio=self.initiation_ratio,
            learn_initiation_ratio=self.learn_initiation_ratio,
            learn_allele_prior=learn_allele_prior, config=config, device=device)


@register_model
@dataclass
class SequentialModel(EditModel):
    """:class:`SequentialProcess` with :class:`CassetteDropout` of the same size.

    Low-memory fits and the likelihood NNI polish are not available.

    Args:
        cassette_size: sites per cassette, filled in order (required).
        initiation_ratio: rate of a cassette's first edit over the rate of the
            following ones (1.0 = one rate throughout).
        learn_initiation_ratio: learn ``initiation_ratio`` during the fit.
    """

    name: ClassVar[str] = "sequential"

    cassette_size: int = model_option(None, "sites per cassette (required for sequential)",
                                      parse=int)
    initiation_ratio: float = model_option(1.0, "first-edit rate over continuation rate",
                                           parse=float)
    learn_initiation_ratio: bool = model_option(False, "learn initiation_ratio", type=bool)

    def __post_init__(self):
        proc = self.components()[0]            # validates the options
        self.cassette_size, self.initiation_ratio = proc.cassette_size, proc.initiation_ratio

    def components(self):
        return (SequentialProcess(cassette_size=self.cassette_size,
                                  initiation_ratio=self.initiation_ratio,
                                  learn_initiation_ratio=self.learn_initiation_ratio),
                CassetteDropout(cassette_size=self.cassette_size))
