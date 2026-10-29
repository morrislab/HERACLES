"""Fixed numerical constants shared across modules.

User options live in :class:`heracles.config.HeraclesConfig`; these values
are not meant to be changed.
"""

#: floor applied to probabilities before taking logarithms
LOG_FLOOR = 1e-12

#: logit assigned to alleles outside a site's support before the softmax of a
#: learned allele prior (effectively -inf, finite for autograd)
MASK_LOGIT = -1e9

#: default quartets per optimiser step = min(QUARTETS_PER_CELL * n, C(n, 4))
QUARTETS_PER_CELL = 15

#: default embedding dimension = ceil(DIM_PER_SQRT_N * sqrt(n))
DIM_PER_SQRT_N = 5

#: steps over which the banded sampler's ``band_kmax`` ramps to ``band_kmax_end``
SAMPLER_RAMP_STEPS = 2000

#: fixed local-outgroup ring ``[k, LOCAL_RING_MULT * k)`` when
#: ``band_local_mult_max`` is None
LOCAL_RING_MULT = 4.0

#: bytes per entry of a PHYLIP distance matrix written as text (RapidNJ input)
PHYLIP_BYTES_PER_ENTRY = 11

#: rows per block when a distance matrix is streamed from the coordinates
DIST_ROW_CHUNK = 64

#: rows per block when weighted-Hamming distances are streamed
HAMMING_ROW_CHUNK = 128

#: offsets added to the run seed for the sampler, penalty and evaluation RNG
#: streams so they are independent of each other
SEED_OFFSET_SAMPLER = 1
SEED_OFFSET_PENALTY = 7
SEED_OFFSET_EVAL = 99

#: convergence check: every CONVERGENCE_CHECK_EVERY steps (after
#: ``min_iterations``), stop after CONVERGENCE_PATIENCE consecutive checks
#: below ``convergence_thresh``
CONVERGENCE_CHECK_EVERY = 100
CONVERGENCE_PATIENCE = 3
