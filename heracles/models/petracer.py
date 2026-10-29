"""The PETracer model: Cas9 edits, with sites grouped into cassettes lost as a unit."""

from dataclasses import dataclass
from typing import ClassVar

from heracles.models.base import EditModel, model_option, register_model
from heracles.models.cas9 import Cas9Process
from heracles.models.missing import CassetteDropout


__all__ = ["PETracerModel"]


@register_model
@dataclass
class PETracerModel(EditModel):
    """:class:`~heracles.models.Cas9Process` with :class:`~heracles.models.CassetteDropout`.

    Each cassette is ``cassette_size`` consecutive columns of the character
    matrix.  Edits stay per site; dropout and heritable deletion blank every
    site of a cassette together.  The likelihood NNI polish uses the same model.

    Args:
        cassette_size: consecutive sites per integration cassette (default 3,
            as in the PETracer barcode).
        separate_dropout_rates: learn one dropout probability per cassette.
    """

    name: ClassVar[str] = "petracer"

    separate_dropout_rates: bool = model_option(
        False, "one dropout probability per cassette", type=bool)
    cassette_size: int = model_option(3, "consecutive sites per cassette", parse=int)

    def __post_init__(self):
        self.cassette_size = int(self.cassette_size)
        if self.cassette_size < 1:
            raise ValueError("cassette_size must be >= 1, got %r" % (self.cassette_size,))

    def components(self):
        return Cas9Process(), CassetteDropout(cassette_size=self.cassette_size,
                                              separate_rates=self.separate_dropout_rates)
