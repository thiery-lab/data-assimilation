"""Exact Kalman filter for inference in linear-Gaussian dynamical systems."""

import abc
from typing import Union, Optional, Sequence, Tuple, Dict
from numbers import Number
import numpy as np
from numpy.random import Generator
import numpy.linalg as nla
import scipy.linalg as sla
from dapy.models.linear_gaussian import AbstractLinearGaussianModel
import tqdm.auto as tqdm


class AbstractKalmanFilter(abc.ABC):
    """Abstract base class for exact Kalman filters for linear-Gaussian models.

    For linear-Gaussian models the Kalman filter (forward) updates [1] allow efficient
    exact calculation of the filtering distributions

        state_sequence[observation_time_indices[t]] | observation_sequence[0:t] ~
            Normal(state_mean_sequence[t], state_covar_sequence[t])

    with the Gaussian form of the filtering distribtions a result of the linear state
    transition and observation operators and Gaussianity of the state and observation
    noise.

    References:

        1. R. E. Kalman, A new approach to linear filtering and prediction
           problems, Transactions of the ASME -- Journal of Basic Engineering,
           Series D, 82 (1960), pp. 35--45.
    """

    def filter(
        self,
        model: AbstractLinearGaussianModel,
        observation_sequence: np.ndarray,
        observation_time_indices: Sequence[int],
        num_sample: int = 0,
        rng: Optional[Generator] = None,
        return_covar: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute means and covariances of Gaussian filtering distributions.

        Args:
            model: Linear-Gaussian generative model for observations.
            observation_sequence: Observation sequence with shape
                `(num_observation_time, dim_observation)` where `num_observation_time`
                is the number of observed time indices in the sequence and
                `dim_observation` is dimensionality of the observations.
            observation_time_indices: Sequence of time (step) indices at which state is
                observed. The sequence elements must be integers. The length of the
                sequence must correspond to the number of observation times
                `num_observation_time` represented in the `observation_sequences_array`.
            num_sample: Number of samples from per time index filtering
                distributions to return in addition to filtering distribution
                parameters.
            rng: NumPy random number generator object. Required if `num_sample > 0`.
            return_covar: Whether to return full covariance matrices for each filtering
                distribution in addition to the standard deviations which are always
                returned.
        Returns:
            Dictionary containing arrays of filtering distribution parameters -
                state_mean_sequence: Array of filtering distribution means at all
                    observation time indices. Shape `(num_observation_time, dim_state)`.
                state_std_sequence: Array of filtering distribution standard deviations
                    at all at all observation time indices. Shape
                    `(num_observation_time, dim_state)`.
                state_covar_sequence: Array of filtering distribution covariance
                    matrices at all observation time indices. Only returned if
                    `return_covar == True`. Shape
                    `(num_observation_time, dim_state, dim_state)`.
                state_samples_sequence: Array of samples from filtering distributions at
                    at all observation time indices. Only returned if `num_sample > 0`.
                    Shape `(num_observation_time, num_sample, dim_state)`.
        """
        num_obs_time, dim_observation = observation_sequence.shape
        observation_time_indices = np.sort(observation_time_indices)
        num_observation_time = len(observation_time_indices)
        assert observation_sequence.shape[0] == num_observation_time
        num_step = observation_time_indices[-1]
        state_mean_sequence = np.full((num_observation_time, model.dim_state), np.nan)
        state_std_sequence = np.full((num_observation_time, model.dim_state), np.nan)
        if return_covar:
            state_covar_sequence = np.full(
                (num_observation_time, model.dim_state, model.dim_state), np.nan
            )
        if num_sample is not None and num_sample > 0:
            state_samples_sequence = np.full(
                (num_observation_time, num_sample, model.dim_state), np.nan
            )
        for s in tqdm.trange(num_step + 1, desc="Filtering", unit="time steps"):
            if s == 0:
                state_mean = model.initial_state_mean
                state_covar = model.initial_state_covar
                t = 0
            else:
                state_mean, state_covar = self._prediction_update(
                    model, state_mean, state_covar, s
                )
            if s == observation_time_indices[t]:
                state_mean, state_covar = self._assimilation_update(
                    model, state_mean, state_covar, observation_sequence[t], s
                )
                state_mean_sequence[t] = state_mean
                state_std_sequence[t] = state_covar.diagonal() ** 0.5
                if return_covar:
                    state_covar_sequence[t] = state_covar
                if num_sample > 0:
                    try:
                        sqrt_state_covar = sla.cholesky(state_covar, lower=True)
                    except sla.LinAlgError:
                        eigval, eigvec = sla.eigh(state_covar)
                        sqrt_state_covar = eigvec * np.clip(eigval, 0, None) ** 0.5
                    state_samples_sequence[t] = (
                        state_mean[None, :]
                        + rng.standard_normal((num_sample, model.dim_state))
                        @ sqrt_state_covar.T
                    )
                t += 1
        results = {
            "state_mean_sequence": state_mean_sequence,
            "state_std_sequence": state_std_sequence,
        }
        if return_covar:
            results["state_covar_sequence"] = state_covar_sequence
        if num_sample > 0:
            results["state_samples_sequence"] = state_samples_sequence
            # Also alias under 'state_particles_sequence' key for consistency with
            # ensemble methods though technically not particle system
            results["state_particles_sequence"] = state_samples_sequence
        return results

    @abc.abstractmethod
    def _prediction_update(
        self,
        model: AbstractLinearGaussianModel,
        prev_state_mean: np.ndarray,
        prev_state_covar: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict state mean and covariance from values at previous time index.

        The prediction update only account for the free model dynamics without any
        adjustment for the observations.

        Args:
            prev_state_mean: State mean at previous time index `time_index - 1`.
            prev_state_covar: State covariance at previous time index `time_index - 1`.
            time_index: Current time index to compute the predicted statistics for.

        Returns:
            state_mean: Predicted state mean at current time index.
            state_covar: Predicted state covariance of state at current time index.
        """

    @abc.abstractmethod
    def _assimilation_update(
        self,
        model: AbstractLinearGaussianModel,
        state_mean: np.ndarray,
        state_covar: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray]:
        """Adjust state mean and covariance for observations at current time index.

        The assimilation update adjust the state mean and covariance from their prior
        predicted values to their posterior values given the observations at the current
        time index.

        Args:
            state_mean: Predicted state mean at current time index `time_index`.
            state_covar: Predicted state covariance at current time index `time_index`.
            observation: Observations at current time index `time_index`.
            time_index: Current time index to assimilate the observations for.

        Returns:
            post_state_mean: Posterior state mean at current time index.
            post_state_covar: Posterior state covariance of state at current time index.
        """


class MatrixKalmanFilter(AbstractKalmanFilter):
    """Exact Kalman filter for linear-Gaussian dynamical systems with matrix operators.

    This variant assumes the linear state transition and observation operators are
    specified by matrices (NumPy arrays) `state_transition_matrix` and
    `observation_matrix`.

    Assumes the model dynamics are of the form

        for s in range(num_step + 1):
            if s == 0:
                state_sequence[0] = (
                    model.initial_state_mean +
                    chol(model.initial_state_covar) @
                    rng.standard_normal(model.dim_state)
                )
                t = 0
            else:
                state_sequence[s] = (
                    model.state_transition_matrix @ state_sequence[s - 1] +
                    chol(model.state_noise_covar) @ rng.standard_normal(model.dim_state)
                )
            if s == observation_time_indices[t]:
                observation_sequence[t] = (
                    model.observation_matrix  @ state_sequence[s]) +
                    chol(model.observation_noise_covar) @
                    rng.standard_normal(model.dim_observation)
                )
                t += 1

    where `observation_time_indices` is a sequence of integer time indices specifying
    the observation times, `num_step = max(observation_time_indices)`, `rng` is
    a random number generator used to generate the required random variates and `chol`
    is a function computing the lower-triangular Cholesky factor of a positive-definite
    matrix.
    """

    def __init__(
        self, use_joseph_form: bool = True,
    ):
        """
        Args:
            use_joseph_form: Whether to use more numerically stable but more expensive
                Joseph's form for the covariance assimilation update.
        """
        self.use_joseph_form = use_joseph_form

    def _prediction_update(
        self,
        model: AbstractLinearGaussianModel,
        state_mean: np.ndarray,
        state_covar: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        state_mean = model.state_transition_matrix @ state_mean
        state_covar = model.increment_by_state_noise_covar(
            model.state_transition_matrix
            @ state_covar
            @ model.state_transition_matrix.T
        )
        return state_mean, state_covar

    def _assimilation_update(
        self,
        model: AbstractLinearGaussianModel,
        state_mean: np.ndarray,
        state_covar: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        observation_mean = model.observation_matrix @ state_mean
        observation_covar = model.increment_by_observation_noise_covar(
            model.observation_matrix @ state_covar @ model.observation_matrix.T
        )
        observation_matrix_state_covar = model.observation_matrix @ state_covar
        kalman_gain = nla.solve(observation_covar, observation_matrix_state_covar).T
        state_mean = state_mean + kalman_gain @ (observation - observation_mean)
        if self.use_joseph_form:
            # use more numerically stable but more expensive Joseph's form
            # for covariance update
            covar_transform = (
                np.identity(model.dim_state) - kalman_gain @ model.observation_matrix
            )
            state_covar = (
                covar_transform @ state_covar @ covar_transform.T
                + model.postmultiply_by_observation_noise_covar(kalman_gain)
                @ kalman_gain.T
            )
        else:
            state_covar = state_covar - kalman_gain @ observation_matrix_state_covar
        return state_mean, state_covar


class FunctionKalmanFilter(AbstractKalmanFilter):
    """Exact Kalman filter for linear-Gaussian dynamical systems with function operators

    This variant assumes the linear state transition and observation operators are
    specified by functions `next_state_mean` and `observation_mean` respectively rather
    than explicit matrices which can be more efficient when corresponding matrices are
    sparse and also allows for time-dependent operators.

    Assumes the model dynamics are of the form

        for s in range(num_step + 1):
            if s == 0:
                state_sequence[0] = (
                    model.initial_state_mean +
                    chol(model.initial_state_covar) @
                    rng.standard_normal(model.dim_state)
                )
                t = 0
            else:
                state_sequence[s] = (
                    model.next_state_mean(state_sequence[s - 1], s - 1) +
                    chol(model.state_noise_covar) @ rng.standard_normal(model.dim_state)
                )
            if s == observation_time_indices[t]:
                observation_sequence[t] = (
                    model.observation_mean(state_sequence[s], s) +
                    chol(model.observation_noise_covar) @
                    rng.standard_normal(model.dim_observation)
                )
                t += 1

    where `observation_time_indices` is a sequence of integer time indices specifying
    the observation times, `num_step = max(observation_time_indices)`, `rng` is
    a random number generator used to generate the required random variates and `chol`
    is a function computing the lower-triangular Cholesky factor of a positive-definite
    matrix.
    """

    def _prediction_update(
        self,
        model: AbstractLinearGaussianModel,
        state_mean: np.ndarray,
        state_covar: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        state_mean = model.next_state_mean(state_mean, time_index)
        state_covar = model.increment_by_state_noise_covar(
            model.next_state_mean(
                model.next_state_mean(state_covar, time_index).T, time_index,
            ),
        )
        return state_mean, state_covar

    def _assimilation_update(
        self,
        model: AbstractLinearGaussianModel,
        state_mean: np.ndarray,
        state_covar: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        observation_mean = model.observation_mean(state_mean, time_index)
        observation_covar = model.increment_by_observation_noise_covar(
            model.observation_mean(
                model.observation_mean(state_covar, time_index).T, time_index
            )
        )
        observation_matrix_state_covar = model.observation_mean(
            state_covar, time_index
        ).T
        kalman_gain = nla.solve(observation_covar, observation_matrix_state_covar).T
        state_mean = state_mean + kalman_gain @ (observation - observation_mean)
        state_covar = state_covar - kalman_gain @ observation_matrix_state_covar
        # Symmetrize covariance to improve numerical stability
        # https://stats.stackexchange.com/a/476713
        state_covar = (state_covar + state_covar.T) / 2
        return state_mean, state_covar
