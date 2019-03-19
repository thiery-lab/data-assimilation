"""Data assimilation example models."""

from dapy.models.linear_gaussian import DenseLinearGaussianModel
from dapy.models.netto79 import Netto79Model
from dapy.models.lorenz63 import Lorenz63Model
from dapy.models.lorenz96 import Lorenz96Model
from dapy.models.kuramoto_sivashinsky import (
    KuramotoSivashinskyModel, KuramotoSivashinskySPDEModel)
from dapy.models.navier_stokes import NavierStokes2dModel

__all__ = ['linear_gaussian', 'netto79', 'lorenz63', 'lorenz96',
           'kuramoto_sivashinsky', 'navier_stokes']
__authors__ = 'Matt Graham'
__license__ = 'MIT'
