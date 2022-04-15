"""Data assimilation example models."""

from dapy.models.linear_gaussian import DenseLinearGaussianModel
from dapy.models.netto_gimeno_mendes import NettoGimenoMendesModel
from dapy.models.lorenz_1963 import Lorenz1963Model
from dapy.models.lorenz_1996 import Lorenz1996Model
from dapy.models.majda_harlim import (
    FourierStochasticTurbulenceModel,
    SpatialStochasticTurbulenceModel,
)
from dapy.models.kuramoto_sivashinsky import (
    FourierLaminarFlameModel,
    SpatialLaminarFlameModel,
)
from dapy.models.navier_stokes import (
    FourierIncompressibleFluidModel,
    SpatialIncompressibleFluidModel,
)
from dapy.models.damped_advection import (
    FourierDampedAdvectionModel,
    SpatialDampedAdvectionModel,
)


__authors__ = "Matt Graham"
__license__ = "MIT"
