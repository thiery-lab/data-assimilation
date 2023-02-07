"""Data assimilation example models."""

from .linear_gaussian import DenseLinearGaussianModel
from .netto_gimeno_mendes import NettoGimenoMendesModel
from .lorenz_1963 import Lorenz1963Model
from .lorenz_1996 import Lorenz1996Model
from .majda_harlim import (
    FourierStochasticTurbulenceModel,
    SpatialStochasticTurbulenceModel,
)
from .kuramoto_sivashinsky import (
    FourierLaminarFlameModel,
    SpatialLaminarFlameModel,
)
from .navier_stokes import (
    FourierIncompressibleFluidModel,
    SpatialIncompressibleFluidModel,
)
from .damped_advection import (
    FourierDampedAdvectionModel,
    SpatialDampedAdvectionModel,
)


__authors__ = "Matt Graham"
__license__ = "MIT"
