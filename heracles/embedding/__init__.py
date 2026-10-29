"""The hyperbolic embedding and its fit.

=========================  ===================================================
:mod:`.ctmc`               edit-process likelihood kernels shared by the objectives
:mod:`.objective`          quartet composite likelihood (the loss)
:mod:`.sampler`            banded local-outgroup quartet sampler
:mod:`.initialization`     seed distances, initial tree and MDS start points
:mod:`.priors`             allele priors and priors on the dropout rates
:mod:`.optimize`           Riemannian Adam loop with the tree-likeness penalties
=========================  ===================================================

Most users only need :class:`heracles.Heracles`, which runs all of these.
"""
