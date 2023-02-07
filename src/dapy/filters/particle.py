"""Particle filters for inference in state space models."""

import abc
from typing import Tuple, Dict, Callable, Any, Optional
import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from .base import AbstractEnsembleFilter
from ..models.base import AbstractModel, AbstractConditionallyGaussianModel
from .. import ot as optimal_transport


class AbstractParticleFilter(AbstractEnsembleFilter):
    """Abstract base class for particle filters.

    Particle filters approximate the filtering distribution at each observation time by
    alternating updates which (i) propose new values for an ensemble of state particles
    approximating the filtering distribution at the previous time step and compute
    weights for these proposed particles according to the relative densities of
    conditional distribution of the proposed state and observation given the previous
    state under the state space model and conditional distribution of the proposed state
    given the previous state and observation under the proposal and (ii) transform the
    resulting weighted ensemble to give a uniformly weighted ensemble representing an
    empirical approximation to the filtering distribution at the current observation
    time.

    References:

    1. Gordon, N.J.; Salmond, D.J.; Smith, A.F.M. (1993). Novel approach to nonlinear /
       non-Gaussian Bayesian state estimation. Radar and Signal Processing, IEE
       Proceedings F. 140 (2): 107--113.
    2. Del Moral, Pierre (1996). Non Linear Filtering: Interacting Particle Solution.
       Markov Processes and Related Fields. 2 (4): 555--580.
    """

    @abc.abstractmethod
    def _update_states_and_compute_log_weights(
        self,
        model: AbstractModel,
        rng: Generator,
        previous_states: Optional[ArrayLike],
        predicted_states: ArrayLike,
        observation: ArrayLike,
        time_index: int,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Compute state proposals and corresponding log weights.

        Compute proposed values for states in ensemble to represent empirical estimate
        of filtering distribution at time index `time_index` and corresponding (log)
        weights associated with particles.

        Args:
            model: State-space model to compute proposals and weights for.
            rng: NumPy random number generator to use for sampling any random variates.
            previous_states: Two-dimensional array of shape `(num_particle, dim_state)`
                with each row a state particle in ensemble representing empirical
                estimate of filtering distribution at time index `time_index - 1`.
            predicted_states: Two-dimensional array of shape
                `(num_particle, dim_state)` with each row a state particle in ensemble
                representing empirical estimate of predictive distribution at time index
                `time_index`.
            observation: Observations at time index `time_index`.
            time_index: Time index to compute filtering distribution estimate for.

        Returns:
            updated_states: Two-dimensional array of shape `(num_particle, dim_state)`
                with each row a state particle in ensemble representing empirical
                estimate of filtering distribution at time index `time_index`.
            log_weights: One-dimensional array of length `num_particle` with each
                element the logarithm of the weight associated with the corresponding
                state particle in `updated_states` in the empirical estimate of the
                filtering distribution at time index `time_index`.
        """

    @abc.abstractmethod
    def _transform_states_given_weights(
        self,
        rng: Generator,
        states: ArrayLike,
        weights: ArrayLike,
    ) -> ArrayLike:
        """Transform weighted to uniform empirical estimate of filtering distribution.

        Args:
            states: Two-dimensional array of shape `(num_particle, dim_state)` with each
                row a state particle in ensemble representing empirical estimate of
                filtering distribution.
            weights: One-dimensional array of length `num_particle` with each element
                the weight associated with the corresponding state particle in `states`
                in the empirical estimate of the filtering distribution.

        Returns:
            Two-dimensional array of shape `(num_particle, dim_state)` with each row a
                state particle in ensemble representing uniformly weighted empirical
                estimate of filtering distribution.
        """

    def _assimilation_update(
        self,
        model: AbstractModel,
        rng: Generator,
        previous_states: Optional[ArrayLike],
        predicted_states: ArrayLike,
        observation: ArrayLike,
        time_index: int,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        states, log_weights = self._update_states_and_compute_log_weights(
            model, rng, previous_states, predicted_states, observation, time_index
        )
        log_sum_weights = logsumexp(log_weights)
        weights = np.exp(log_weights - log_sum_weights)
        state_mean = (weights[:, None] * states).sum(0)
        state_std = np.sum(weights[:, None] * (states - state_mean) ** 2, axis=0) ** 0.5
        estimated_ess = 1 / (weights**2).sum()
        if not np.allclose(weights, 1 / weights.shape[0]):
            # Only apply weight transform / resampling update if particles are not
            # already uniformly weighted - this avoids introducing unnecessary variance
            # for cases where proposal produces exact samples from filtering
            # distribution with equal weight, for example the optimal proposal in
            # conditionally linear Gaussian models when assimilating observations on
            # the initial time step
            states = self._transform_states_given_weights(rng, states, weights)
        return (
            states,
            {
                "state_mean": state_mean,
                "state_std": state_std,
                "estimated_ess": estimated_ess,
            },
        )


class MultinomialResamplingMixIn:
    """Mix-in class implementing multinomial resampling ensemble transform update."""

    def _transform_states_given_weights(self, rng, states, weights):
        num_particle = states.shape[0]
        resampled_indices = rng.choice(num_particle, num_particle, True, weights)
        return states[resampled_indices]


class OptimalTransportTransformMixIn:
    """Mix-in class implementing optimal transport ensemble transform update.

    References:

    1. Reich, S. (2013). A nonparametric ensemble transform method for Bayesian
       inference. SIAM Journal on Scientific Computing, 35(4), A2013-A2024.
    """

    def __init__(
        self,
        optimal_transport_solver: Callable[
            [ArrayLike, ArrayLike, ArrayLike], ArrayLike
        ] = optimal_transport.solve_optimal_transport_exact,
        optimal_transport_solver_kwargs: Optional[Dict[str, Any]] = None,
        transport_cost: Callable[
            [ArrayLike, ArrayLike], ArrayLike
        ] = optimal_transport.pairwise_euclidean_distance,
        weight_threshold: float = 1e-8,
        use_sparse_matrix_multiply: bool = False,
        **kwargs
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
            optimal_transport_solver_kwargs: Any additional keyword parameters values
                for the optimal transport solver.
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
        super().__init__(**kwargs)
        self.optimal_transport_solver = optimal_transport_solver
        self.optimal_transport_solver_kwargs = (
            {}
            if optimal_transport_solver_kwargs is None
            else optimal_transport_solver_kwargs
        )
        self.transport_cost = transport_cost
        self.weight_threshold = weight_threshold
        self.use_sparse_matrix_multiply = use_sparse_matrix_multiply

    def _transform_states_given_weights(self, rng, states, weights):
        """Solve optimal transport problem and transform ensemble."""
        num_particle = states.shape[0]
        source_dist = np.ones(num_particle) / num_particle
        target_dist = weights
        if self.weight_threshold > 0:
            target_dist[target_dist < self.weight_threshold] = 0
            target_dist /= target_dist.sum()
        cost_matrix = self.transport_cost(states, states)
        transform_matrix = num_particle * self.optimal_transport_solver(
            source_dist,
            target_dist,
            cost_matrix,
            **self.optimal_transport_solver_kwargs
        )
        if self.use_sparse_matrix_multiply:
            transform_matrix = csr_matrix(transform_matrix)
        return transform_matrix @ states


class PriorProposalMixIn:
    """Mix-in class implementing proposals from prior state-space model.

    Samples proposal from prior dynamics of state space model without conditioning on
    observations, with log weight then corresponding to log density of conditional
    distribution on observations given proposed state at current time index.
    """

    def _update_states_and_compute_log_weights(
        self,
        model: AbstractModel,
        rng: Generator,
        previous_states: Optional[ArrayLike],
        predicted_states: ArrayLike,
        observation: ArrayLike,
        time_index: int,
    ) -> Tuple[ArrayLike, ArrayLike]:
        log_weights = model.log_density_observation_given_state(
            observation, predicted_states, time_index
        )
        return predicted_states, log_weights


class OptimalProposalMixIn:
    """Mix-in class implementing 'optimal' proposals for conditionally Gaussian models.

    Samples proposals from the conditional distribution on the state given the state
    at the previous time step and observations at the current time step under the
    generative state space model. This proposal is optimal is the sense of minimising
    the variances of the particle weights, which depend only on the values of the state
    particles at the previous time step, not the proposed states.

    References:

    1. Doucet, A., S. Godsill, and C. Andrieu (2000). On sequential Monte Carlo
       sampling methods for Bayesian filtering. Statistics and Computing, 10, 197-208.
    """

    def __init__(self, assume_time_homogeneous_model: bool = False, **kwargs):
        """
        Args:
            assume_time_homogeneous_model: Whether to assume the state space model that
                will be used to perform filtering is time-homogeneous and so has a
                fixed in time observation matrix and state and observation noise
                covariance matrices. If this is the case, linear algebra operations
                required for computing the optimal proposal can be performed once at the
                beginning of filtering rather than on each time step.
        """
        super().__init__(**kwargs)
        self._assume_time_homogeneous_model = assume_time_homogeneous_model

    def _get_covariance_matrices(
        self,
        model: AbstractConditionallyGaussianModel,
        time_index: Optional[int] = None,
    ):
        state_covar = (
            model.initial_state_covar if time_index == 0 else model.state_noise_covar
        )
        cov_observations_given_states = model.increment_by_observation_noise_covar(
            model.observation_mean(
                model.observation_mean(state_covar, time_index).T, time_index
            )
        )
        cho_factor_cov_observations_given_states = cho_factor(
            cov_observations_given_states
        )
        state_noise_covar_observation_matrix_T = model.observation_mean(
            state_covar, time_index
        )
        return (
            cho_factor_cov_observations_given_states,
            state_noise_covar_observation_matrix_T,
        )

    def _perform_model_specific_initialization(
        self,
        model: AbstractConditionallyGaussianModel,
        num_particle: int,
    ):
        if self._assume_time_homogeneous_model:
            (
                self._cho_factor_cov_observations_given_states,
                self._state_noise_covar_observation_matrix_T,
            ) = self._get_covariance_matrices(model, None)

    def _update_states_and_compute_log_weights(
        self,
        model: AbstractConditionallyGaussianModel,
        rng: Generator,
        previous_states: Optional[ArrayLike],
        predicted_states: ArrayLike,
        observation: ArrayLike,
        time_index: int,
    ) -> Tuple[ArrayLike, ArrayLike]:
        if time_index == 0 or not self._assume_time_homogeneous_model:
            (
                cho_factor_cov_observations_given_states,
                state_noise_covar_observation_matrix_T,
            ) = self._get_covariance_matrices(model, time_index)
        elif self._assume_time_homogeneous_model:
            cho_factor_cov_observations_given_states = (
                self._cho_factor_cov_observations_given_states
            )
            state_noise_covar_observation_matrix_T = (
                self._state_noise_covar_observation_matrix_T
            )
        if time_index != 0:
            # For optimal proposal weights depend only on particles from previous time
            # step not particles after assimilation update
            predicted_observation_means = model.observation_mean(
                model.next_state_mean(previous_states, time_index - 1), time_index
            )
            observation_diff = predicted_observation_means - observation[None]
            log_weights = (
                -(
                    observation_diff.T
                    * cho_solve(
                        cho_factor_cov_observations_given_states,
                        observation_diff.T,
                    )
                ).sum(0)
                / 2
            )
        else:
            # as initial state distribution is assumed to be Gaussian, optimal proposal
            # will produce exact samples from filtering distribution and particles
            # will all have equal weights
            log_weights = np.zeros(predicted_states.shape[0])
        simulated_observations = model.sample_observation_given_state(
            rng, predicted_states, time_index
        )
        states = (
            predicted_states
            + (
                state_noise_covar_observation_matrix_T
                @ (
                    cho_solve(
                        cho_factor_cov_observations_given_states,
                        (observation[None] - simulated_observations).T,
                    )
                )
            ).T
        )
        return states, log_weights


class BootstrapParticleFilter(
    MultinomialResamplingMixIn, PriorProposalMixIn, AbstractParticleFilter
):
    """Bootstrap particle filter (sequential importance resampling).

    The filtering distribution at each observation time index is approximated by
    alternating propagating an ensemble of state particles forward through time under
    the model dynamics and transforming according to weights calculated from the
    conditional probability densities of the observations at the current time index
    given the state particle values. Here the transform step uses multinomial
    resampling.

    References:

    1. Gordon, N.J.; Salmond, D.J.; Smith, A.F.M. (1993). Novel approach to nonlinear /
       non-Gaussian Bayesian state estimation. Radar and Signal Processing, IEE
       Proceedings F. 140 (2): 107--113.
    2. Del Moral, Pierre (1996). Non Linear Filtering: Interacting Particle Solution.
       Markov Processes and Related Fields. 2 (4): 555--580.
    """


class EnsembleTransformParticleFilter(
    OptimalTransportTransformMixIn, PriorProposalMixIn, AbstractParticleFilter
):
    """Ensemble transform particle filter.

    The filtering distribution at each observation time index is approximated by
    alternating propagating an ensemble of state particles forward through time under
    the model dynamics and linearly transforming the ensemble to a uniformly weighted
    empirical estimate of the filtering distribution with an optimal transport map.

    References:

    1. Reich, S. (2013). A nonparametric ensemble transform method for Bayesian
       inference. SIAM Journal on Scientific Computing, 35(4), A2013-A2024.
    """


class OptimalProposalParticleFilter(
    MultinomialResamplingMixIn, OptimalProposalMixIn, AbstractParticleFilter
):
    """Particle filter using 'optimal' proposal for conditionally Gaussian models.

    Samples proposals from the conditional distribution on the state given the state
    at the previous time step and observations at the current time step under the
    generative state space model. This proposal is optimal is the sense of minimising
    the variances of the particle weights, which depend only on the values of the state
    particles at the previous time step, not the proposed states. The proposed ensemble
    is resampled to a uniformly weighted empirical estimate of the filtering
    distribution with a multinomial scheme.

    References:

    1. Doucet, A., S. Godsill, and C. Andrieu (2000). On sequential Monte Carlo
       sampling methods for Bayesian filtering. Statistics and Computing, 10, 197-208.
    """


class OptimalProposalEnsembleTransformParticleFilter(
    OptimalTransportTransformMixIn, OptimalProposalMixIn, AbstractParticleFilter
):

    """Ensemble transform particle filter using 'optimal' proposal for conditionally
    Gaussian models.

    Samples proposals from the conditional distribution on the state given the state
    at the previous time step and observations at the current time step under the
    generative state space model. This proposal is optimal is the sense of minimising
    the variances of the particle weights, which depend only on the values of the state
    particles at the previous time step, not the proposed states. The proposed ensemble
    is linearly transformed to a uniformly weighted empirical estimate of the filtering
    distribution with an optimal transport map.

    References:

    1. Doucet, A., S. Godsill, and C. Andrieu (2000). On sequential Monte Carlo
       sampling methods for Bayesian filtering. Statistics and Computing, 10, 197-208.
    2. Reich, S. (2013). A nonparametric ensemble transform method for Bayesian
       inference. SIAM Journal on Scientific Computing, 35(4), A2013-A2024.
    """
