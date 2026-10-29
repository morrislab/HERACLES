"""Quartet composite likelihood objective.

Each sampled leaf quartet is scored under the irreversible-edit CTMC of
:mod:`heracles.embedding.ctmc`. The unrooted topology is chosen from the four-point
sums and the rooted shape (balanced or caterpillar) from the Gromov
products. Batches come from the banded local-outgroup sampler in
:mod:`heracles.embedding.sampler`.

Main entry point: :func:`quartet_loss_fn`, the loss handed to the optimiser.
"""

import logging

import torch

from heracles.embedding.ctmc import (
    with_learnable_prior,
    LOG_FLOOR,
    _to_float_mask,
    _index_prior,
    observation_weight,
    leaf_dropout,
    pair_edit_joint_missing,
    leaf_marginal_logprob,
    triple_branch_lengths,
    _triple_site_logprob_parts,
    gromov_quartet_penalty_from_distances,
)
from heracles.embedding.sampler import (_stratified_quartets, build_banded_sampler,
                                       default_batch_size, n_distinct_quartets)

logger = logging.getLogger("heracles")


__all__ = [
    'quartet_branch_lengths',
    'quartet_site_logprob',
    'quartet_objective',
    'quartet_loss_fn',
]


# Unrooted topology index -> permutation of 4 leaf slots so that the
# selected pair occupies slots (0,1) and the other pair (2,3).
# Row 0: topology (01|23) -> identity
# Row 1: topology (02|13) -> swap 1<->2
# Row 2: topology (03|12) -> swap 1<->3
_QUARTET_PERM = torch.tensor(
    [[0, 1, 2, 3],   # (01|23): left=(0,1), right=(2,3)
     [0, 2, 1, 3],   # (02|13): left=(0,2), right=(1,3)
     [0, 3, 1, 2]],  # (03|12): left=(0,3), right=(1,2)
    dtype=torch.long)

_QUARTET_CONST_CACHE = {}


def _quartet_consts(device):
    """Per-device cache of the three constant lookup tables.

    Returns (perm, pair_to_flat, remaining):
    - perm: topology index -> slot permutation (see _QUARTET_PERM)
    - pair_to_flat: (slot_i, slot_j) -> index into a 6-vector of pair
      quantities ordered [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    - remaining: outgroup slot -> the 3 other slots, ascending
    """
    if device not in _QUARTET_CONST_CACHE:
        pair_to_flat = torch.tensor([
            [0, 0, 1, 2],
            [0, 0, 3, 4],
            [1, 3, 0, 5],
            [2, 4, 5, 0],
        ], dtype=torch.long, device=device)
        remaining = torch.tensor([
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ], dtype=torch.long, device=device)
        _QUARTET_CONST_CACHE[device] = (
            _QUARTET_PERM.to(device), pair_to_flat, remaining)
    return _QUARTET_CONST_CACHE[device]


def quartet_branch_lengths(depths, dists_6, quartets, quartet_depths=None):
    """Pick each quartet's topology and rooted shape and read off its branch lengths.

    The unrooted topology is the pairing with the smallest distance sum; the
    rooted shape is balanced ``((p,q),(r,s))`` or caterpillar
    ``(p,(q,(r,s)))``. Branch lengths for both shapes are returned for every
    quartet and ``is_balanced`` says which applies.

    Args:
        depths: (m,) root-to-taxon distances; entry 0 is the root.
        dists_6: tuple of six (B,) tensors ``(d01, d02, d03, d12, d13, d23)`` for
            the slot pairs of each quartet.
        quartets: (B, 4) long tensor of taxon indices.
        quartet_depths: optional pre-gathered ``depths[quartets]``, shape (B, 4).

    Returns:
        tuple: 18 entries, in this order.
        ``is_balanced``: (B,) bool.
        Balanced shape: ``idx_bal`` (B, 4), taxa as [left1, left2, right1,
        right2]; then (B,) lengths ``a`` (root to the top split), ``b1, c1, d1``
        (split to the left ancestor, left ancestor to left1 and to left2) and
        ``b2, c2, d2`` (the same on the right).
        Caterpillar shape: ``outgroup_idx`` (B, 1); ``sub_triple_idx`` (B, 3),
        cherry-ordered; (B,) lengths ``a_cat`` (root to where the outgroup branches
        off) and ``c_out`` (outgroup pendant branch); then the sub-triple's
        ``root_u, u_v, v_cherry1, v_cherry2, u_outgroup`` as in
        :func:`heracles.embedding.ctmc.triple_branch_lengths`, except that ``root_u`` starts
        at the outgroup's branch point rather than at the root.
    """
    d01, d02, d03, d12, d13, d23 = dists_6

    # Unrooted topology: argmin of the three distance sums
    s0 = d01 + d23   # topology (01|23)
    s1 = d02 + d13   # topology (02|13)
    s2 = d03 + d12   # topology (03|12)
    sums = torch.stack([s0, s1, s2], dim=1)
    topo = sums.argmin(dim=1)  # (B,)

    _perm_table, pair_to_flat, _remaining_table = _quartet_consts(quartets.device)

    # Permute leaf slots so left pair is (0,1), right pair is (2,3)
    perm = _perm_table[topo]
    idx = quartets.gather(1, perm)  # (B, 4)

    # Pair distances in the permuted ordering: d(left1, left2), d(right1, right2)
    # and the four cross distances, as one (B, 6) gather.
    all_dists = torch.stack([d01, d02, d03, d12, d13, d23], dim=1)  # (B, 6)
    p0, p1, p2, p3 = perm[:, 0], perm[:, 1], perm[:, 2], perm[:, 3]
    permuted_pairs = torch.stack([
        pair_to_flat[p0, p1], pair_to_flat[p2, p3],
        pair_to_flat[p0, p2], pair_to_flat[p0, p3],
        pair_to_flat[p1, p2], pair_to_flat[p1, p3],
    ], dim=1)  # (B, 6)
    permuted_dists = all_dists.gather(1, permuted_pairs)
    dist_left, dist_right = permuted_dists[:, 0], permuted_dists[:, 1]
    dist_02, dist_03 = permuted_dists[:, 2], permuted_dists[:, 3]
    dist_12, dist_13 = permuted_dists[:, 4], permuted_dists[:, 5]

    # Gromov products relative to root o. One dense gather of the pre-gathered
    # (B, 4) depths: every extra depths[...] fancy-index would cost its own
    # atomic scatter-add in the backward pass.
    if quartet_depths is None:
        quartet_depths = depths[quartets]                  # (B, 4)
    depths_perm = quartet_depths.gather(1, perm)           # (B, 4)
    depth_0, depth_1 = depths_perm[:, 0], depths_perm[:, 1]
    depth_2, depth_3 = depths_perm[:, 2], depths_perm[:, 3]

    tau_left = (0.5 * (depth_0 + depth_1 - dist_left)).clamp_min(0.0)
    tau_right = (0.5 * (depth_2 + depth_3 - dist_right)).clamp_min(0.0)
    tau_02 = (0.5 * (depth_0 + depth_2 - dist_02)).clamp_min(0.0)
    tau_03 = (0.5 * (depth_0 + depth_3 - dist_03)).clamp_min(0.0)
    tau_12 = (0.5 * (depth_1 + depth_2 - dist_12)).clamp_min(0.0)
    tau_13 = (0.5 * (depth_1 + depth_3 - dist_13)).clamp_min(0.0)
    tau_cross = 0.25 * (tau_02 + tau_03 + tau_12 + tau_13)

    # Rooted shape
    is_balanced = (tau_left >= tau_cross) & (tau_right >= tau_cross)

    # Balanced branch lengths
    a_bal = tau_cross.clamp_min(0.0)
    b1_bal = (tau_left - a_bal).clamp_min(0.0)
    c1_bal = (depth_0 - tau_left).clamp_min(0.0)
    d1_bal = (depth_1 - tau_left).clamp_min(0.0)
    b2_bal = (tau_right - a_bal).clamp_min(0.0)
    c2_bal = (depth_2 - tau_right).clamp_min(0.0)
    d2_bal = (depth_3 - tau_right).clamp_min(0.0)

    # Caterpillar branch lengths.
    # Identify the outgroup: the leaf with the shallowest connection.
    # If tau_left < tau_cross, outgroup is in the left pair {0,1}.
    # If tau_right < tau_cross, outgroup is in the right pair {2,3}.
    # Within the pair, the outgroup has smaller mean tau to the opposite pair.
    left_is_cat = tau_left < tau_cross
    mean_tau_0_to_right = 0.5 * (tau_02 + tau_03)
    mean_tau_1_to_right = 0.5 * (tau_12 + tau_13)
    left_outgroup_is_0 = mean_tau_0_to_right <= mean_tau_1_to_right

    mean_tau_2_to_left = 0.5 * (tau_02 + tau_12)
    mean_tau_3_to_left = 0.5 * (tau_03 + tau_13)
    right_outgroup_is_2 = mean_tau_2_to_left <= mean_tau_3_to_left

    # Outgroup column: 0, 1, 2, or 3: {left: 0/1, right: 2/3} as arithmetic
    out_col = ((~left_is_cat).long() * 2
               + torch.where(left_is_cat,
                             (~left_outgroup_is_0).long(),
                             (~right_outgroup_is_2).long()))

    # The 3 non-outgroup slots, via the cached lookup table
    remaining_cols = _remaining_table[out_col]  # (B, 3)

    outgroup_taxon = idx.gather(1, out_col.unsqueeze(1))  # (B, 1)
    sub_triple_taxa = idx.gather(1, remaining_cols)        # (B, 3)

    # Outgroup and sub-triple depths from the already-permuted (B, 4) depths
    out_depth = depths_perm.gather(1, out_col.unsqueeze(1)).squeeze(1)
    sub_taxa_depths = depths_perm.gather(1, remaining_cols)  # (B, 3)

    # Pair quantities in the permuted slot basis, ordered to match
    # pair_to_flat: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)].
    permuted_taus = torch.stack(
        [tau_left, tau_02, tau_03, tau_12, tau_13, tau_right], dim=1)  # (B, 6)
    permuted_pair_dists = torch.stack(
        [dist_left, dist_02, dist_03, dist_12, dist_13, dist_right], dim=1)              # (B, 6)

    # Taus from the outgroup slot to each remaining slot: one (B, 3) gather
    outgroup_taus = permuted_taus.gather(
        1, pair_to_flat[out_col.unsqueeze(1), remaining_cols])

    a_cat = outgroup_taus.min(dim=1).values.clamp_min(0.0)
    c_out_cat = (out_depth - a_cat).clamp_min(0.0)

    # The sub-triple's 3 pairwise distances: one (B, 3) gather
    rc0, rc1, rc2 = remaining_cols[:, 0], remaining_cols[:, 1], remaining_cols[:, 2]
    sub_pairs = torch.stack([pair_to_flat[rc0, rc1],
                             pair_to_flat[rc0, rc2],
                             pair_to_flat[rc1, rc2]], dim=1)  # (B, 3)
    sub_dists = permuted_pair_dists.gather(1, sub_pairs)
    sub_d01, sub_d02, sub_d12 = sub_dists[:, 0], sub_dists[:, 1], sub_dists[:, 2]

    sub_triple_idx, root_u_sub_abs, u_v_sub, v_cherry1_sub, v_cherry2_sub, u_outgroup_sub = triple_branch_lengths(
        depths, sub_d01, sub_d02, sub_d12, sub_triple_taxa,
        taxa_depths=sub_taxa_depths)
    # triple_branch_lengths measures root_u from the root o; the caterpillar
    # kernel needs the branch w1 -> w2 = root_u_sub_abs - a_cat. The other
    # sub-triple branches (u_v, v_cherry1, v_cherry2, u_outgroup) are already
    # measured from their parent node.
    root_u_sub = (root_u_sub_abs - a_cat).clamp_min(0.0)

    return (is_balanced,
            idx, a_bal, b1_bal, c1_bal, d1_bal, b2_bal, c2_bal, d2_bal,
            outgroup_taxon, sub_triple_idx, a_cat, c_out_cat,
            root_u_sub, u_v_sub, v_cherry1_sub, v_cherry2_sub, u_outgroup_sub)


def _quartet_site_logprob_balanced(states, idx, a, b1, c1, d1, b2, c2, d2,
                                   mutation_rate, allele_prior, heritable_eps=0.0, stochastic_delta=0.0):
    """Log probability for balanced quartets: two independent cherry factors."""
    states_0, states_1, states_2, states_3 = (states[idx[:, i]] for i in range(4))  # (B, S) each
    no_edit_root_to_internal = torch.exp(-mutation_rate * a).unsqueeze(1)
    no_edit_to_left_ancestor = torch.exp(-mutation_rate * b1).unsqueeze(1)
    no_edit_to_left_leaf_0 = torch.exp(-mutation_rate * c1).unsqueeze(1)
    no_edit_to_left_leaf_1 = torch.exp(-mutation_rate * d1).unsqueeze(1)
    no_edit_to_right_ancestor = torch.exp(-mutation_rate * b2).unsqueeze(1)
    no_edit_to_right_leaf_2 = torch.exp(-mutation_rate * c2).unsqueeze(1)
    no_edit_to_right_leaf_3 = torch.exp(-mutation_rate * d2).unsqueeze(1)

    prior_0 = _index_prior(allele_prior, states_0)
    prior_1 = _index_prior(allele_prior, states_1)
    prior_2 = _index_prior(allele_prior, states_2)
    prior_3 = _index_prior(allele_prior, states_3)

    missing_0, missing_1, missing_2, missing_3 = states_0 < 0, states_1 < 0, states_2 < 0, states_3 < 0

    # Left cherry (states_0, states_1) masks
    both_unedited_left = _to_float_mask((states_0 == 0) & (states_1 == 0))
    first_unedited_second_edited_left = _to_float_mask((states_0 == 0) & (states_1 > 0))
    first_edited_second_unedited_left = _to_float_mask((states_0 > 0) & (states_1 == 0))
    same_edit_left = _to_float_mask((states_0 > 0) & (states_0 == states_1))
    diff_edit_left = _to_float_mask((states_0 > 0) & (states_1 > 0) & (states_0 != states_1))

    # Right cherry (states_2, states_3) masks
    both_unedited_right = _to_float_mask((states_2 == 0) & (states_3 == 0))
    first_unedited_second_edited_right = _to_float_mask((states_2 == 0) & (states_3 > 0))
    first_edited_second_unedited_right = _to_float_mask((states_2 > 0) & (states_3 == 0))
    same_edit_right = _to_float_mask((states_2 > 0) & (states_2 == states_3))
    diff_edit_right = _to_float_mask((states_2 > 0) & (states_3 > 0) & (states_2 != states_3))

    # Deletion survival and observation weights
    no_del_root_to_internal = torch.exp(-heritable_eps * a).unsqueeze(1)
    no_del_to_left_ancestor = torch.exp(-heritable_eps * b1).unsqueeze(1)
    no_del_to_right_ancestor = torch.exp(-heritable_eps * b2).unsqueeze(1)
    observed_left_0 = torch.exp(-heritable_eps * c1).unsqueeze(1) * leaf_dropout(stochastic_delta, idx[:, 0])
    observed_left_1 = torch.exp(-heritable_eps * d1).unsqueeze(1) * leaf_dropout(stochastic_delta, idx[:, 1])
    observed_right_2 = torch.exp(-heritable_eps * c2).unsqueeze(1) * leaf_dropout(stochastic_delta, idx[:, 2])
    observed_right_3 = torch.exp(-heritable_eps * d2).unsqueeze(1) * leaf_dropout(stochastic_delta, idx[:, 3])

    # Left cherry factor (missing-aware)
    M_left = pair_edit_joint_missing(
        no_edit_to_left_ancestor, no_edit_to_left_leaf_0, no_edit_to_left_leaf_1, prior_0, prior_1, no_del_to_left_ancestor, observed_left_0, observed_left_1,
        both_unedited_left, first_unedited_second_edited_left, first_edited_second_unedited_left, same_edit_left, diff_edit_left,
        _to_float_mask((states_0 == 0) & missing_1), _to_float_mask((states_0 > 0) & missing_1),
        _to_float_mask(missing_0 & (states_1 == 0)), _to_float_mask(missing_0 & (states_1 > 0)),
        _to_float_mask(missing_0 & missing_1))

    # Right cherry factor (missing-aware)
    M_right = pair_edit_joint_missing(
        no_edit_to_right_ancestor, no_edit_to_right_leaf_2, no_edit_to_right_leaf_3, prior_2, prior_3, no_del_to_right_ancestor, observed_right_2, observed_right_3,
        both_unedited_right, first_unedited_second_edited_right, first_edited_second_unedited_right, same_edit_right, diff_edit_right,
        _to_float_mask((states_2 == 0) & missing_3), _to_float_mask((states_2 > 0) & missing_3),
        _to_float_mask(missing_2 & (states_3 == 0)), _to_float_mask(missing_2 & (states_3 > 0)),
        _to_float_mask(missing_2 & missing_3))

    # Root edited: all observed must be the same allele
    q_alpha = torch.where(states_0 > 0, prior_0, torch.where(states_1 > 0, prior_1,
              torch.where(states_2 > 0, prior_2, torch.where(states_3 > 0, prior_3,
              1.0))))
    edited_ok = _to_float_mask(((states_0 > 0) | missing_0) & ((states_1 > 0) | missing_1) &
                   ((states_2 > 0) | missing_2) & ((states_3 > 0) | missing_3))
    agree = _to_float_mask(
        (missing_0 | missing_1 | (states_0 == states_1)) & (missing_0 | missing_2 | (states_0 == states_2)) &
        (missing_0 | missing_3 | (states_0 == states_3)) & (missing_1 | missing_2 | (states_1 == states_2)) &
        (missing_1 | missing_3 | (states_1 == states_3)) & (missing_2 | missing_3 | (states_2 == states_3)))

    # Observation weights for the "root edited" case
    W_left = observation_weight(no_del_to_left_ancestor, observed_left_0, observed_left_1, missing_0, missing_1)
    W_right = observation_weight(no_del_to_right_ancestor, observed_right_2, observed_right_3, missing_2, missing_3)

    P = no_del_root_to_internal * (no_edit_root_to_internal * M_left * M_right
              + (1 - no_edit_root_to_internal) * q_alpha * W_left * W_right * edited_ok * agree) \
        + (1 - no_del_root_to_internal) * _to_float_mask(missing_0 & missing_1 & missing_2 & missing_3)
    return torch.log(P.clamp_min(LOG_FLOOR))


def _quartet_site_logprob_caterpillar(states, outgroup_idx, sub_triple_idx,
                                      a_cat, c_out,
                                      root_u_sub, u_v_sub, v_cherry1_sub, v_cherry2_sub, u_outgroup_sub,
                                      mutation_rate, allele_prior, heritable_eps=0.0, stochastic_delta=0.0):
    """Log probability for caterpillar quartets: outgroup + sub-triple."""
    states_outgroup = states[outgroup_idx.squeeze(1)]  # (B, S)

    # Outer level: root -> w1 (depth a_cat), then outgroup pendant c_out
    no_edit_root_to_split = torch.exp(-mutation_rate * a_cat).unsqueeze(1)

    missing_outgroup = states_outgroup < 0
    prior_outgroup = _index_prior(allele_prior, states_outgroup)

    # Outgroup leaf marginal given w1 = unedited
    no_del_root_to_split = torch.exp(-heritable_eps * a_cat).unsqueeze(1)
    no_edit_to_outgroup = torch.exp(-mutation_rate * c_out).unsqueeze(1)
    observed_outgroup = torch.exp(-heritable_eps * c_out).unsqueeze(1) * leaf_dropout(stochastic_delta, outgroup_idx.squeeze(1))

    L_out = observed_outgroup * (no_edit_to_outgroup * _to_float_mask(states_outgroup == 0)
                     + (1 - no_edit_to_outgroup) * prior_outgroup * _to_float_mask(states_outgroup > 0)) \
            + (1 - observed_outgroup) * _to_float_mask(missing_outgroup)

    # Sub-triple probability given w1 = unedited. The parts variant hands
    # back the intermediates this same sub-triple needs again below.
    logP_triple, sub = _triple_site_logprob_parts(
        states, sub_triple_idx, root_u_sub, u_v_sub, v_cherry1_sub, v_cherry2_sub, u_outgroup_sub,
        mutation_rate, allele_prior, heritable_eps=heritable_eps, stochastic_delta=stochastic_delta)
    P_triple = torch.exp(logP_triple)

    # Sub-triple leaf states for the "root edited" case (reused from the parts)
    states_sub_cherry_1, states_sub_cherry_2, states_sub_outgroup = sub["states"]
    missing_sub_cherry_1, missing_sub_cherry_2, missing_sub_outgroup = sub["missing"]
    prior_sub_cherry_1, prior_sub_cherry_2, prior_sub_outgroup = sub["priors"]

    # Root edited: alpha pinned by the first observed leaf (any of the 4)
    q_alpha = torch.where(states_outgroup > 0, prior_outgroup,
              torch.where(states_sub_cherry_1 > 0, prior_sub_cherry_1,
              torch.where(states_sub_cherry_2 > 0, prior_sub_cherry_2,
              torch.where(states_sub_outgroup > 0, prior_sub_outgroup, 1.0))))

    edited_ok = _to_float_mask(((states_outgroup > 0) | missing_outgroup) &
                   ((states_sub_cherry_1 > 0) | missing_sub_cherry_1) & ((states_sub_cherry_2 > 0) | missing_sub_cherry_2) & ((states_sub_outgroup > 0) | missing_sub_outgroup))

    agree = _to_float_mask(
        (missing_outgroup | missing_sub_cherry_1 | (states_outgroup == states_sub_cherry_1)) & (missing_outgroup | missing_sub_cherry_2 | (states_outgroup == states_sub_cherry_2)) &
        (missing_outgroup | missing_sub_outgroup | (states_outgroup == states_sub_outgroup)) & (missing_sub_cherry_1 | missing_sub_cherry_2 | (states_sub_cherry_1 == states_sub_cherry_2)) &
        (missing_sub_cherry_1 | missing_sub_outgroup | (states_sub_cherry_1 == states_sub_outgroup)) & (missing_sub_cherry_2 | missing_sub_outgroup | (states_sub_cherry_2 == states_sub_outgroup)))

    # Observation weights for the "w1 edited" case
    W_out = observed_outgroup * _to_float_mask(~missing_outgroup) + (1 - observed_outgroup) * _to_float_mask(missing_outgroup)

    # Sub-triple observation weights for the "w1 edited" case, all already
    # computed inside the sub-triple evaluation above.
    no_del_sub_root_to_internal = sub["no_del_root_to_internal"]
    W_triple = no_del_sub_root_to_internal * sub["W_cherry"] * sub["W_out"] \
        + (1 - no_del_sub_root_to_internal) * sub["all_missing_f"]

    all_missing = sub["all_missing_f"] * _to_float_mask(missing_outgroup)

    P = no_del_root_to_split * (no_edit_root_to_split * L_out * P_triple
              + (1 - no_edit_root_to_split) * q_alpha * W_out * W_triple * edited_ok * agree) \
        + (1 - no_del_root_to_split) * all_missing
    return torch.log(P.clamp_min(LOG_FLOOR))


def quartet_site_logprob(states, is_balanced,
                         idx_bal, a_bal, b1_bal, c1_bal, d1_bal,
                         b2_bal, c2_bal, d2_bal,
                         outgroup_idx, sub_triple_idx,
                         a_cat, c_out_cat,
                         root_u_sub, u_v_sub, v_cherry1_sub, v_cherry2_sub, u_outgroup_sub,
                         mutation_rate, allele_prior, heritable_eps=0.0, stochastic_delta=0.0):
    """Per-quartet, per-site log probability, using the balanced or caterpillar kernel per row.

    Takes ``states``, the outputs of :func:`quartet_branch_lengths`, and the
    CTMC parameters as in :func:`heracles.embedding.ctmc.triple_site_logprob`. Returns a
    (B, S) tensor. Each kernel runs only on the rows of its own shape.
    """
    B, S = is_balanced.shape[0], states.shape[1]
    bal_rows = is_balanced.nonzero(as_tuple=True)[0]
    cat_rows = (~is_balanced).nonzero(as_tuple=True)[0]
    out = torch.zeros(B, S, dtype=a_bal.dtype, device=a_bal.device)
    if bal_rows.numel():
        logP_bal = _quartet_site_logprob_balanced(
            states, idx_bal[bal_rows],
            a_bal[bal_rows], b1_bal[bal_rows], c1_bal[bal_rows],
            d1_bal[bal_rows], b2_bal[bal_rows], c2_bal[bal_rows],
            d2_bal[bal_rows],
            mutation_rate, allele_prior,
            heritable_eps=heritable_eps, stochastic_delta=stochastic_delta)
        out = out.index_copy(0, bal_rows, logP_bal)
    if cat_rows.numel():
        logP_cat = _quartet_site_logprob_caterpillar(
            states, outgroup_idx[cat_rows], sub_triple_idx[cat_rows],
            a_cat[cat_rows], c_out_cat[cat_rows],
            root_u_sub[cat_rows], u_v_sub[cat_rows],
            v_cherry1_sub[cat_rows], v_cherry2_sub[cat_rows],
            u_outgroup_sub[cat_rows],
            mutation_rate, allele_prior,
            heritable_eps=heritable_eps, stochastic_delta=stochastic_delta)
        out = out.index_copy(0, cat_rows, logP_cat)
    return out



def quartet_objective(states, n_points, score, *, est_dists=None, batch_size=None,
                      root_frac=1.0, root_weight=None, config=None, sampler=None,
                      knn_from_states=False, device=None, loss_scale=1.0,
                      band_kmin=None, band_kmax=None, band_kmax_end="default",
                      band_local_frac=None, band_local_mult_max="default"):
    """The quartet composite-likelihood loss around a model's scoring function.

    Shared by every edit process: each step draws a quartet batch from the
    banded sampler, computes the founder depths, the six pairwise distances and
    the rooted branch lengths, hands the four-point defect and the depths to the
    optimiser's penalties, picks the leaves of the root-to-leaf term, and asks
    *score* for the log probabilities.  The loss is
    ``-((1 - w) mean(quartets) + w mean(root terms)) / loss_scale``.

    Args:
        states: (n_points, S) states, founder at row 0.
        n_points: number of points including the founder.
        score: ``score(states, depths, branches, leaves, mutation_rate,
            heritable_eps, stochastic_delta) -> (logP_quartets, logP_root)``
            where ``branches`` is the tuple from :func:`quartet_branch_lengths`
            and ``leaves`` the leaf indices of the root-to-leaf term.
        knn_from_states: let the sampler build k-NN tables from ``states``
            (low-memory fits) when ``config.sampler_knn`` is set.
        loss_scale: divisor of the loss (e.g. sites per cassette, so a cassette
            log probability is on the per-site scale).
        Other arguments: as in :func:`quartet_loss_fn`.

    Returns:
        The ``loss_fn`` callable with the attributes the optimiser reads
        (``batch_gromov_penalty``, ``batch_depths``, ``make_eval_batch``) and the
        sampler and batch bookkeeping (``quartet_sampler``, ``n_quartets``,
        ``n_root_terms``, ``root_weight``, ``meta``).
    """
    n = n_points - 1
    if n < 4:
        raise ValueError(f"need at least 4 leaves for the quartet objective, got {n}")
    sampler_knn = None
    if config is not None:      # the fit's HeraclesConfig supplies unset settings
        band_kmin = config.band_kmin if band_kmin is None else band_kmin
        band_kmax = config.band_kmax if band_kmax is None else band_kmax
        band_local_frac = config.band_local_frac if band_local_frac is None else band_local_frac
        if band_kmax_end == "default":
            band_kmax_end = config.band_kmax_end
        if band_local_mult_max == "default":
            band_local_mult_max = config.band_local_mult_max
        sampler_knn = getattr(config, "sampler_knn", None)
        if batch_size is None:
            batch_size = config.batch_size
    if sampler is None:
        sampler_name = "banded"
        knn = (dict(sampler_knn=sampler_knn, character_matrix=states, device=device)
               if knn_from_states else {})
        sampler, sampler_config, sampler_stats = build_banded_sampler(
            est_dists, n_points, band_kmin=band_kmin, band_kmax=band_kmax,
            band_kmax_end=band_kmax_end, band_local_frac=band_local_frac,
            band_local_mult_max=band_local_mult_max, **knn)
    else:
        sampler_name = getattr(sampler, "__name__", "custom")
        sampler_config, sampler_stats = None, {}

    st = torch.as_tensor(states, dtype=torch.long)
    if device is not None:
        st = st.to(torch.device(device))
    batch_size = int(batch_size) if batch_size is not None else default_batch_size(n)
    batch_size = max(1, min(batch_size, n_distinct_quartets(n)))
    n_root = max(1, int(round(root_frac * n)))
    root_term_weight = (n_root / (batch_size + n_root)) if root_weight is None else float(root_weight)
    if not 0.0 <= root_term_weight <= 1.0:
        raise ValueError(f"root_weight must be in [0, 1], got {root_term_weight!r}")
    all_leaves = torch.arange(1, n_points, dtype=torch.long).to(st.device)

    def loss_fn(points, manifold, radius, mutation_rate, heritable_eps, stochastic_delta,
                rng, eval_batch=None):
        if eval_batch is not None:
            quartets = eval_batch
        else:
            quartets = sampler(n, batch_size, generator=rng).to(st.device)
        depths = radius * manifold.dist(points[0].unsqueeze(0), points)
        # One advanced-indexing op each for points and depths (every extra
        # fancy-index costs a scatter-add in backward), then unbind rather than
        # slice: four slices would each allocate and zero-fill a full (B, 4, dim)
        # gradient buffer in backward, unbind stacks the four into one.
        pq = points[quartets]                       # (B, 4, dim)
        quartet_depths = depths[quartets]           # (B, 4)
        points_0, points_1, points_2, points_3 = pq.unbind(1)
        d01 = radius * manifold.dist(points_0, points_1)
        d02 = radius * manifold.dist(points_0, points_2)
        d03 = radius * manifold.dist(points_0, points_3)
        d12 = radius * manifold.dist(points_1, points_2)
        d13 = radius * manifold.dist(points_1, points_3)
        d23 = radius * manifold.dist(points_2, points_3)
        # these distances and depths are already paid for: the optimiser's
        # tree-likeness and ultrametric penalties reuse them for this batch
        loss_fn.batch_gromov_penalty = gromov_quartet_penalty_from_distances(
            d01, d02, d03, d12, d13, d23)
        loss_fn.batch_depths = depths[1:]
        branches = quartet_branch_lengths(
            depths, (d01, d02, d03, d12, d13, d23), quartets,
            quartet_depths=quartet_depths)
        if n_root >= n:
            leaves = all_leaves
        else:
            perm = torch.randperm(n, generator=rng)[:n_root].to(all_leaves.device)
            leaves = all_leaves[perm]
        logP_q, logP_root = score(st, depths, branches, leaves, mutation_rate,
                                  heritable_eps, stochastic_delta)
        loss = -((1.0 - root_term_weight) * logP_q.mean()
                 + root_term_weight * logP_root.mean())
        if loss_scale != 1.0:
            loss = loss / loss_scale
        return loss, logP_q.numel() + logP_root.numel()

    loss_fn.quartet_sampler = sampler_name
    loss_fn.quartet_sampler_config = sampler_config
    loss_fn.quartet_sampler_stats = sampler_stats
    loss_fn.n_quartets = batch_size
    loss_fn.n_root_terms = n_root
    loss_fn.root_weight = root_term_weight
    loss_fn.batch_gromov_penalty = None
    loss_fn.batch_depths = None
    loss_fn.make_eval_batch = lambda generator: _stratified_quartets(
        n, batch_size, generator=generator)
    loss_fn.meta = {"n_quartets": batch_size, "quartet_sampler": sampler_name,
                    "quartet_sampler_config": sampler_config,
                    "quartet_sampler_stats": sampler_stats,
                    "n_root_terms": n_root, "root_weight": root_term_weight}
    return loss_fn


@with_learnable_prior
def quartet_loss_fn(states, get_allele_prior, n_points, *, est_dists=None, batch_size=None,
                     root_frac=1.0, root_weight=None,
                     band_kmin=None, band_kmax=None, band_kmax_end="default",
                     band_local_frac=None, band_local_mult_max="default",
                     device=None, sampler=None, config=None, cassette=None):
    """Return the quartet composite-likelihood loss used by :func:`heracles.fit.embed`.

    The loss is the negative mean log probability of a batch of quartets, mixed
    with the root-to-leaf term of :func:`heracles.embedding.ctmc.leaf_marginal_logprob`.
    Batches come from :func:`build_banded_sampler` built on ``est_dists``,
    unless ``sampler`` is given.

    Args:
        states: (n_points, S) states, founder at row 0.
        allele_prior: array from :func:`heracles.embedding.priors.build_allele_priors`.
        n_points: number of points including the founder.
        est_dists: (n_points, n_points) founder-augmented seed distances for the
            sampler.
        batch_size: quartets per step; default ``min(15 n, C(n, 4))``.
        root_frac: share of leaves in the root-to-leaf term each step (1 = all).
        root_weight: weight of that term, in [0, 1]; default
            ``n_root / (batch_size + n_root)``.
        learn_allele_prior: learn the allele prior (keeping its global or
            per-site shape).
        band_kmin, band_kmax, band_kmax_end, band_local_frac, band_local_mult_max:
            sampler options (see :func:`build_banded_sampler`); unset values come
            from ``config`` or the ``HeraclesConfig`` defaults.
        device: torch device for the states and the loss.
        sampler: optional callable ``(n_leaves, n_quartets, generator=None,
            device=None)`` that replaces the banded sampler.
        config: optional ``HeraclesConfig`` that supplies unset band settings and
            ``sampler_knn``.
        cassette: optional :class:`heracles.models.missing.CassetteDropout`; when
            given, the missing-data factors are charged once per cassette instead
            of once per site.

    Returns:
        Callable ``loss_fn(points, manifold, radius, mutation_rate, heritable_eps,
        stochastic_delta, rng, eval_batch=None) -> (loss, numel)``. It carries
        ``meta`` (recorded on the fit), ``make_eval_batch`` (fixed convergence
        batch), ``batch_gromov_penalty`` and ``batch_depths`` (reused by the
        optimiser's penalties) and ``allele_prior_param``.

    Raises:
        ValueError: if there are fewer than 4 leaves or ``root_weight`` is outside
            [0, 1].
    """
    if cassette is not None:
        st_device = torch.device(device) if device is not None else torch.device("cpu")
        cassette.to(st_device)
        zero_rate = torch.zeros((), dtype=torch.float64, device=st_device)

    def score(st, depths, branches, leaves, mutation_rate, heritable_eps, stochastic_delta):
        allele_prior = get_allele_prior()

        def quartet_logprob(hard_states, mu):
            return quartet_site_logprob(hard_states, *branches, mu, allele_prior,
                                        heritable_eps=heritable_eps,
                                        stochastic_delta=stochastic_delta)

        def root_logprob(hard_states, mu):
            return leaf_marginal_logprob(hard_states, depths, mu, allele_prior, leaves,
                                         heritable_eps=heritable_eps,
                                         stochastic_delta=stochastic_delta)

        # The same kernels on the missing pattern with no edits give the
        # missing-data factor alone; charge it once per cassette. The order
        # (quartet term, its correction, root term, its correction) sets the
        # gradient accumulation order and hence the round-off.
        logP_q = quartet_logprob(st, mutation_rate)
        if cassette is not None:
            logP_q = cassette.correct(logP_q, quartet_logprob(cassette.pattern_states, zero_rate))
        logP_root = root_logprob(st, mutation_rate)
        if cassette is not None:
            logP_root = cassette.correct(logP_root, root_logprob(cassette.pattern_states,
                                                                 zero_rate))
        return logP_q, logP_root

    loss_fn = quartet_objective(
        states, n_points, score, est_dists=est_dists, batch_size=batch_size,
        root_frac=root_frac, root_weight=root_weight, config=config, sampler=sampler,
        knn_from_states=True, device=device, band_kmin=band_kmin, band_kmax=band_kmax,
        band_kmax_end=band_kmax_end, band_local_frac=band_local_frac,
        band_local_mult_max=band_local_mult_max)
    loss_fn.meta["objective"] = "quartet"
    if cassette is not None:
        loss_fn.meta.update(cassette.summary())
    return loss_fn


