"""Filters for sequential inference in state-space models."""

from .ensemble_kalman import (
    EnsembleKalmanFilter,
    EnsembleTransformKalmanFilter,
)
from .kalman import MatrixKalmanFilter, FunctionKalmanFilter
from .particle import (
    BootstrapParticleFilter,
    OptimalProposalParticleFilter,
    EnsembleTransformParticleFilter,
)
from .local import (
    LocalEnsembleTransformKalmanFilter,
    LocalEnsembleTransformParticleFilter,
    ScalableLocalEnsembleTransformParticleFilter,
)

__authors__ = "Matt Graham"
__license__ = "MIT"
