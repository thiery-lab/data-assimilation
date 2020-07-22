"""Model with linear dynamics and observations and additive Gaussian noise."""

import numpy as np
from numpy.random import Generator
from typing import Dict
import scipy.linalg as la
from dapy.models.base import AbstractModel


def generate_random_dense_parameters(
    rng: Generator, dim_state: int, dim_observation: int
) -> Dict[str, np.ndarray]:
    """Generate random parameters for dense linear Gaussian model.

    Args:
        rng: Numpy random number generator.
        dim_state: Dimension of model state vector.
        dim_observation: Dimension of observation vector.

    Returns:
        Dictionary of generated system parameters.
    """
    params = {}
    params["initial_state_mean"] = rng.normal(size=dim_state)
    temp_array = rng.standard_normal((dim_state, dim_state)) * (0.5 / dim_state) ** 0.5
    params["initial_state_covar"] = temp_array @ temp_array.T
    params["state_transition_matrix"] = (
        rng.standard_normal((dim_state, dim_state)) * (0.5 / dim_state) ** 0.5
    )
    temp_array = rng.standard_normal((dim_state, dim_state)) * (0.5 / dim_state) ** 0.5
    params["state_noise_covar"] = temp_array @ temp_array.T
    params["observation_matrix"] = (
        rng.standard_normal((dim_observation, dim_state))
        * (1.0 / (dim_state + dim_observation)) ** 0.5
    )
    temp_array = (
        rng.standard_normal((dim_observation, dim_observation))
        * (0.5 / dim_observation) ** 0.5
    )
    params["observation_noise_covar"] = temp_array @ temp_array.T
    return params


class DenseLinearGaussianModel(AbstractModel):
    """Model with linear dynamics and observations and additive Gaussian noise.

    The modelled system dynamics are of the form

        state_sequence[0] = (
            init_state_mean + chol(init_state_covar) @ rng.standard_normal(dim_state))
        observation_sequence[0] = (
            observation_matrix @ state_sequence[0] + 
            chol(observation_noise_covar) @ rng.standard_normal(dim_observation))
        for t in range(1, num_observation_time):
            state_sequence[t] = (
                state_transition_matrx @ state_sequence[t-1] + 
                chol(state_noise_covar) @ rng.standard_normal(dim_state))
            observation_sequence[t] = (
                observation_matrix  @ state_sequence[t]) +
                chol(observation_noise_covar) @ rng.standard_normal(dim_observation))

   where `chol` is a function computing the lower-triangular Cholesky factor of a
   positive-definite matrix.
   """

    def __init__(
        self,
        initial_state_mean: np.ndarray,
        initial_state_covar: np.ndarray,
        state_transition_matrix: np.ndarray,
        observation_matrix: np.ndarray,
        state_noise_covar: np.ndarray,
        observation_noise_covar: np.ndarray,
    ):
        """
        Args:
            initial_state_mean: Mean of initial state distribution. Shape `(dim_state)`.
            initial_state_covar: Positive-definite matrix defining covariance of initial
                state distribution. Shape `(dim_state, dim_state)`.
            state_transition_matrix: Matrix defining linear state transition operator.
                Shape `(dim_state, dim_state)`.
            observation_matrix: Matrix defining linear obervation operator. Shape
                `(dim_observation, dim_state)`.
            state_noise_covar: Positive-definite matrix defining covariance of additive
                state noise. Shape `(dim_state, dim_state)`.
            observation_noise_covar: Positive-definite matrix defining covariance of
                additive observation noise. Shape `(dim_observation, dim_observation)`.
        """
        # check for dimensional consistency
        assert state_transition_matrix.shape[0] == state_transition_matrix.shape[1]
        assert state_transition_matrix.shape[0] == observation_matrix.shape[1]
        assert state_transition_matrix.shape[0] == initial_state_mean.shape[0]
        assert state_transition_matrix.shape[0] == initial_state_covar.shape[0]
        assert state_transition_matrix.shape[0] == initial_state_covar.shape[1]
        dim_observation, dim_state = observation_matrix.shape
        self.state_transition_matrix = state_transition_matrix
        self.observation_matrix = observation_matrix
        self.initial_state_mean = initial_state_mean
        self.initial_state_covar = initial_state_covar
        self.chol_initial_state_covar = la.cholesky(initial_state_covar, lower=True)
        self.state_noise_covar = state_noise_covar
        self.chol_state_noise_covar = la.cholesky(state_noise_covar, lower=True)
        self.observation_noise_covar = observation_noise_covar
        self.chol_observation_noise_covar = la.cholesky(
            observation_noise_covar, lower=True
        )
        super().__init__(dim_state, dim_observation)

    def _sample_initial_state(self, rng: Generator, num_state: int) -> np.ndarray:
        return (
            self.initial_state_mean
            + rng.standard_normal((num_state, self.dim_state))
            @ self.chol_initial_state_covar.T
        )

    def next_state_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        return states @ self.state_transition_matrix.T

    def _sample_state_transition(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        return (
            self.next_state_mean(states, t)
            + rng.standard_normal((states.shape[0], self.dim_state))
            @ self.chol_state_noise_covar.T
        )

    def observation_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        return states @ self.observation_matrix.T

    def _sample_observation_given_state(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        return (
            self.observation_mean(states, t)
            + rng.standard_normal((states.shape[0], self.dim_observation))
            @ self.chol_observation_noise_covar.T
        )

    def log_density_initial_state(self, states: np.ndarray) -> np.ndarray:
        states_minus_mean = states - self.initial_state_mean
        return -(
            0.5
            * (
                states_minus_mean.T
                * la.cho_solve(
                    (self.chol_initial_state_covar, True), states_minus_mean.T
                )
            ).sum(0)
            + 0.5 * self.dim_state * np.log(2 * np.pi)
            + np.log(self.chol_initial_state_covar.diagonal()).sum()
        )

    def log_density_state_transition(
        self, next_states: np.ndarray, states: np.ndarray, t: int
    ) -> np.ndarray:
        next_states_minus_mean = next_states - self.next_state_mean(states, t)
        return -(
            0.5
            * (
                next_states_minus_mean.T
                * la.cho_solve(
                    (self.state_noise_covar_chol, True), next_states_minus_mean.T
                )
            ).sum(0)
            + 0.5 * self.dim_state * np.log(2 * np.pi)
            + np.log(self.chol_state_noise_covar.diagonal()).sum()
        )

    def log_density_observation_given_state(
        self, observations: np.ndarray, states: np.ndarray, t: int
    ) -> np.ndarray:
        observations_minus_mean = observations - self.observation_mean(states, t)
        return -(
            0.5
            * (
                observations_minus_mean.T
                * la.cho_solve(
                    (self.chol_observation_noise_covar, True), observations_minus_mean.T
                )
            ).sum(0)
            + 0.5 * self.dim_observation * np.log(2 * np.pi)
            + np.log(self.chol_observation_noise_covar.diagonal()).sum()
        )
