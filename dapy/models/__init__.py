"""Python package for data assimilation example models."""

from dapy.models.linear import LinearGaussianModel
from dapy.models.netto79 import Netto79Model
from dapy.models.lorenz63 import Lorenz63Model


__all__ = ['linear', 'netto79', 'lorenz63', 'lorenz96']
__authors__ = 'Matt Graham'
__license__ = 'MIT'
