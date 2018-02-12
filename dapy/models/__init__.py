"""Python package for data assimilation example models."""

from dapy.models.linear import LinearGaussianModel
from dapy.models.netto79 import Netto79Model
from dapy.models.lorenz63 import Lorenz63Model
from dapy.models.lorenz96 import Lorenz96Model
from dapy.models.kuramoto_sivashinsky import KuramotoSivashinskyModel
from dapy.models.fluidsim2d import FluidSim2DModel


__all__ = ['linear', 'netto79', 'lorenz63', 'lorenz96', 'fluidsim2d']
__authors__ = 'Matt Graham'
__license__ = 'MIT'
