"""The Cas9 edit process and the default Cas9 model.

Each site starts unedited (state 0), is edited once at rate ``mutation_rate``
to an allele drawn from the allele prior, and never changes again; sites are
independent.  :class:`Cas9Process` connects the kernels in :mod:`heracles.embedding.ctmc`,
the quartet loss in :mod:`heracles.embedding.objective` and the whole-tree
likelihood in :mod:`heracles.polish.ancestral` to the :class:`EditProcess` template.
"""

import logging
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from heracles.embedding.initialization import seed_distances
from heracles.models.base import (EditModel, EditProcess, model_option, register_model,
                                  register_process)
from heracles.models.missing import CassetteCorrection, CassetteDropout, SiteDropout
from heracles.embedding.objective import quartet_loss_fn

logger = logging.getLogger(__name__)


__all__ = ["Cas9Process", "Cas9Model"]


@register_process
@dataclass
class Cas9Process(EditProcess):
    """Independent sites, each edited at most once.

    Supports :class:`SiteDropout` and :class:`CassetteDropout`, low-memory fits
    and the likelihood NNI polish.
    """

    name: ClassVar[str] = "cas9"
    supported_missing: ClassVar[tuple] = (SiteDropout, CassetteDropout)
    supports_low_memory: ClassVar[bool] = True

    def seed_distances(self, states, *, dtype=np.float64):
        """Weighted Hamming distances (:func:`heracles.embedding.initialization.seed_distances`)."""
        return seed_distances(states, dtype=dtype)

    def loss_function(self, states, allele_prior, *, missing, seed_dists, batch_size,
                      root_weight, learn_allele_prior, config, device):
        """Quartet loss (:func:`heracles.embedding.objective.quartet_loss_fn`); with :class:`CassetteDropout`
        it adds the :class:`CassetteCorrection` and logs the cassette missingness counts.
        """
        extras = {}
        units = missing.units(states.shape[1])
        if units is not None:
            correction = CassetteCorrection(units, states, device=device)
            correction.log_report(logger)
            extras["cassette"] = correction
        return quartet_loss_fn(states, allele_prior, states.shape[0], est_dists=seed_dists,
                               batch_size=batch_size, root_weight=root_weight,
                               learn_allele_prior=learn_allele_prior, config=config,
                               device=device, **extras)

    def tree_likelihood(self, char_matrix, *, missing, mutation_rate, heritable_eps,
                        stochastic_delta, allele_prior, missing_state=-1):
        """Felsenstein pruning under the same model (:func:`heracles.polish.ancestral.prepare_model`)."""
        from heracles.polish.ancestral import prepare_model
        return prepare_model(char_matrix, mutation_rate=mutation_rate,
                             allele_prior=allele_prior, heritable_eps=heritable_eps,
                             stochastic_delta=stochastic_delta, missing_state=missing_state,
                             site_groups=missing.units(np.asarray(char_matrix).shape[1]))


@register_model
@dataclass
class Cas9Model(EditModel):
    """The default model: :class:`Cas9Process` with :class:`SiteDropout`.

    ``separate_dropout_rates=True`` learns one dropout probability per site (each
    under the same Beta prior) instead of a single shared one.
    """

    name: ClassVar[str] = "cas9"

    separate_dropout_rates: bool = model_option(
        False, "one dropout probability per site (per cassette for petracer)", type=bool)

    def components(self):
        return Cas9Process(), SiteDropout(separate_rates=self.separate_dropout_rates)
