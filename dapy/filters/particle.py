"""Particle filters for inference in state space models."""

import abc
from typing import Tuple
import numpy as np
from numpy.random import Generator
from scipy.special import logsumexp
from dapy.filters.base import AbstractEnsembleFilter
from dapy.models.base import AbstractModel
import dapy.ot as optimal_transport


class AbstractParticleFilter(AbstractEnsembleFilter):
    """Abstract base class for particle filters."""

    def _calculate_weights(
        self,
        model: AbstractModel,
        states: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> np.ndarray:
        """Calculate importance weights for particles given observations."""
        log_weights = model.log_density_observation_given_state(
            observation, states, time_index
        )
        log_sum_weights = logsumexp(log_weights)
        return np.exp(log_weights - log_sum_weights)

    @abc.abstractmethod
    def _assimilation_transform(
        self, rng: Generator, state_particles: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        pass

    def _assimilation_update(
        self,
        model: AbstractModel,
        rng: Generator,
        state_particles: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self._calculate_weights(
            model, state_particles, observation, time_index
        )
        state_mean = (weights[:, None] * state_particles).sum(0)
        state_std = (
            np.sum(weights[:, None] * (state_particles - state_mean) ** 2, axis=0)
            ** 0.5
        )
        state_particles = self._assimilation_transform(rng, state_particles, weights)
        return state_particles, state_mean, state_std


class BootstrapParticleFilter(AbstractParticleFilter):
    """Bootstrap particle filter (sequential importance resampling).

    The filtering distribution at each observation time index is approximated by
    alternating propagating an ensemble of state particles forward through time under
    the model dynamics and resampling according to weights calculated from the
    conditional probability densities of the observations at the current time index
    given the state particle values. Here the resampling step uses multinomial
    resampling.

    References:

        1. Gordon, N.J.; Salmond, D.J.; Smith, A.F.M. (1993). Novel approach to
           nonlinear / non-Gaussian Bayesian state estimation. Radar and Signal
           Processing, IEE Proceedings F. 140 (2): 107--113.
        2. Del Moral, Pierre (1996). Non Linear Filtering: Interacting Particle
           Solution. Markov Processes and Related Fields. 2 (4): 555--580.
    """

    def _assimilation_transform(self, rng, state_particles, weights):
        """Perform multinomial particle resampling given computed weights."""
        num_particle = state_particles.shape[0]
        resampled_indices = rng.choice(num_particle, num_particle, True, weights)
        return state_particles[resampled_indices]


class EnsembleTransformParticleFilter(AbstractParticleFilter):
    """Ensemble transform particle filter.

    The filtering distribution at each observation time index is approximated by
    alternating propagating an ensemble of state particles forward through time under
    the model dynamics and linearly transforming the ensemble with an optimal transport
    map computed to transform a uniform empirical distribution at the particle locations
    to an empirical distribution at the particle locations weighted according to the
    conditional probability densities of the observations at the current time index
    given the state particle values [1].

    References:

        1. Reich, S. (2013). A nonparametric ensemble transform method for
           Bayesian inference. SIAM Journal on Scientific Computing, 35(4),
           A2013-A2024.
    """

    def __init__(
        self,
        ot_solver=optimal_transport.solve_optimal_transport_exact,
        ot_solver_kwargs=None,
    ):
        """
        Args:
            ot_solver (function): Optimal transport solver function with
                call signature
                    ot_solver(source_dist, target_dist, cost_matrix,
                              **ot_solver_params)
                where source_dist and target_dist are the source and target
                distribution weights respectively as 1D arrays, cost_matrix is
                a 2D array of the distances between particles and
                ot_solver_params is any additional keyword parameter values
                for the solver.
            ot_solver_kwargs (dict): Any additional keyword parameters values
                for the optimal transport solver.
        """
        self.ot_solver = ot_solver
        self.ot_solver_kwargs = {} if ot_solver_kwargs is None else ot_solver_kwargs

    def _assimilation_transform(self, rng, state_particles, weights):
        """Solve optimal transport problem and transform ensemble."""
        num_particle = state_particles.shape[0]
        source_dist = np.ones(num_particle) / num_particle
        target_dist = weights
        target_dist /= target_dist.sum()
        # Cost matrix entries Euclidean distance between particles
        cost_matrix = optimal_transport.pairwise_euclidean_distance(
            state_particles, state_particles
        )
        transform_matrix = num_particle * self.ot_solver(
            source_dist, target_dist, cost_matrix, **self.ot_solver_kwargs
        )
        return transform_matrix @ state_particles
