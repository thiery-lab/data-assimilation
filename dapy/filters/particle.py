"""Particle filters for inference in state space models."""

import abc
from typing import Tuple, Dict, Callable, Any, Optional
import numpy as np
from numpy.random import Generator
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
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
        optimal_transport_solver: Callable[
            [np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ] = optimal_transport.solve_optimal_transport_exact,
        optimal_transport_solver_kwargs: Optional[Dict[str, Any]] = None,
        transport_cost: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ] = optimal_transport.pairwise_euclidean_distance,
        weight_threshold: float = 1e-8,
        use_sparse_matrix_multiply: bool = False,
    ):
        """
        Args:
            optimal_transport_solver: Optimal transport solver function with signature

                    transport_matrix = optimal_transport_solver(
                        source_dist, target_dist, cost_matrix,
                        **optimal_transport_solver_kwargs)

                where `source_dist` and `target_dist` are the source and target
                distribution weights respectively as 1D arrays, `cost_matrix` is a 2D
                array of the transport costs for each particle pair.
            optimal_transport_solver: Any additional keyword parameters values for the
                optimal transport solver.
            transport_cost: Function calculating transport cost matrix with signature

                    cost_matrix = transport_cost(source_particles, target_particles)

                where `source_particles` are the particles values of the source and
                target empirical distributions respecitively.
            weight_threshold: Threshold below which to set any particle weights to zero
                prior to solving the optimal transport problem. Using a small non-zero
                value can both improve the numerical stability of the optimal transport
                solves, with problems with many small weights sometimes failing to
                convergence, and also improve performance as some solvers (including)
                the default network simplex based algorithm) are able to exploit
                sparsity in the source / target distributions.
            use_sparse_matrix_multiply: Whether to conver the optimal transport based
                transform matrix used in the assimilation update to a sparse CSR format
                before multiplying by the state particle ensemble matrix. This may
                improve performance when the computed transport plan is sparse and the
                number of particles is large.
        """
        self.optimal_transport_solver = optimal_transport_solver
        self.optimal_transport_solver_kwargs = (
            {}
            if optimal_transport_solver_kwargs is None
            else optimal_transport_solver_kwargs
        )
        self.transport_cost = transport_cost
        self.weight_threshold = weight_threshold
        self.use_sparse_matrix_multiply = use_sparse_matrix_multiply

    def _assimilation_transform(self, rng, state_particles, weights):
        """Solve optimal transport problem and transform ensemble."""
        num_particle = state_particles.shape[0]
        source_dist = np.ones(num_particle) / num_particle
        target_dist = weights
        if self.weight_threshold > 0:
            target_dist[target_dist < self.weight_threshold] = 0
            target_dist /= target_dist.sum()
        cost_matrix = self.transport_cost(state_particles, state_particles)
        transform_matrix = num_particle * self.optimal_transport_solver(
            source_dist,
            target_dist,
            cost_matrix,
            **self.optimal_transport_solver_kwargs
        )
        if self.use_sparse_matrix_multiply:
            transform_matrix = csr_matrix(transform_matrix)
        return transform_matrix @ state_particles
