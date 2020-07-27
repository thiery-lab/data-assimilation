"""Ensemble Kalman filters for inference in state space models."""

from typing import Tuple
import numpy as np
from numpy.random import Generator
import numpy.linalg as nla
import scipy.linalg as sla
from dapy.models.base import AbstractGaussianObservationModel, AbstractModel
from dapy.filters.base import AbstractEnsembleFilter


class EnsembleKalmanFilter(AbstractEnsembleFilter):
    """Ensemble Kalman filter with perturbed observations.

    The filtering distribution at each observation time index is approximated by
    alternating propagating an ensemble of state particles forward through time under
    the model dynamics and linearly transforming the ensemble according to a Monte Carlo
    estimate of the Kalman filter assimilation update due to the observations at the
    current time index. Here a 'perturbed observation' ensemble Kalman filter
    assimilation update is used with an observation particle sampled for each state
    particle from the conditional distribution on the observation given the state,
    and these observation particles as well as the original state particles used to
    approximate the covariance and mean statistics used in the Kalman update [1, 2].

    References:

        1. G. Evensen, Sequential data assimilation with nonlinear quasi-geostrophic
           model using Monte Carlo methods to forecast error statistics, Journal of
           Geophysical Research, 99 (C5) (1994), pp. 143--162

        2. P. Houtekamer and H. L. Mitchell, Data assimilation using an ensemble Kalman
           filter technique, Monthly Weather Review, 126 (1998), pp. 796--811
    """

    def _assimilation_update(
        self,
        model: AbstractModel,
        rng: Generator,
        state_particles: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation_particles = model.sample_observation_given_state(
            rng, state_particles, time_index
        )
        state_deviations = state_particles - state_particles.mean(0)
        observation_deviations = observation_particles - observation_particles.mean(0)
        observation_errors = observation - observation_particles
        state_particles = (
            state_particles
            + nla.lstsq(observation_deviations.T, observation_errors.T, rcond=None)[0].T
            @ state_deviations
        )
        return state_particles, state_particles.mean(0), state_particles.std(0)


class EnsembleTransformKalmanFilter(AbstractEnsembleFilter):
    """Ensemble Kalman filter with deterministic matrix square root updates.

    The filtering distribution at each observation time index is approximated by
    alternating propagating an ensemble of state particles forward through time under
    the model dynamics and linearly transforming the ensemble according to a Monte Carlo
    estimate of the Kalman filter assimilation update due to the observations at the
    current time index. Here a 'square-root' ensemble Kalman filter assimilation update
    is used, which requires that the model has Gaussian observation noise with a
    known covariance, but compared to the 'perturbed observation' variant avoids the
    additional variance associated with sampling pseudo-observations [1, 2].

    References:

        1. Bishop, C. H. Etherton, B. J. and  Majumdar, S. J. (2001).
           Adaptive sampling with the ensemble transform Kalman filter. Part I:
           Theoretical aspects.Mon. Wea. Rev., 129, 420–436.
        2. M. K. Tippett, J. L. Anderson, C. H. Bishop, T. M. Hamill,
           and J. S. Whitaker, Ensemble square root filters,
           Monthly Weather Review, 131 (2003), pp. 1485--1490.
    """

    def _assimilation_update(
        self,
        model: AbstractGaussianObservationModel,
        rng: Generator,
        state_particles: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_particle = state_particles.shape[0]
        state_mean = state_particles.mean(0)
        state_deviations = state_particles - state_mean
        # Note: compared to the `observation_particles` variable defined in the
        # perturbed observations EnKF implementation here these observation 'particles'
        # are pre addition of observation noise
        observation_particles = model.observation_mean(state_particles, time_index)
        observation_mean = observation_particles.mean(0)
        observation_deviations = observation_particles - observation_mean
        observation_error = observation - observation_mean
        # Let X = state_deviations, Y = observation_deviations, N = num_particle,
        # R = observation_noise_covar, Z = post_state_deviations, I = identity(N)
        # Then for consistency with Kalman filter covariance update we require
        # Z.T @ Z = X.T @ inv(I +  Y @ inv(R) @ Y.T / (N - 1))) @ X
        # If we find a N x N transform matrix T such that
        # T @ T = inv(I +  Y @ inv(R) @ Y.T / (N - 1))) then Z = T @ X.
        # Defining M = Y @ inv(chol(R / (N - 1)).T) then
        # I + M @ M.T = I + Y @ inv(R) @ Y.T / (N - 1)) and T @ T = inv(I + M @ M.T).
        # If U, s, V_T = svd(M) such that M = U @ diag(s) @ V_T then
        # I + M @ M.T = U @ U.T + U @ diag(s**2) @ U.T = U @ diag(1 + s**2) @ U.T
        # and so T = U @ diag(1 / (1 + s**2)**0.5) @ U.T
        transform_matrix_eigenvectors, non_zero_singular_values, _ = nla.svd(
            model.postmultiply_by_inv_chol_trans_observation_noise_covar(
                observation_deviations
            )
            / (num_particle - 1) ** 0.5,
        )
        squared_transform_matrix_eigenvalues = 1 / (1 + non_zero_singular_values ** 2)
        if model.dim_observation < num_particle:
            squared_transform_matrix_eigenvalues = np.concatenate(
                [
                    squared_transform_matrix_eigenvalues,
                    np.ones(num_particle - model.dim_observation),
                ]
            )
        transform_matrix = (
            transform_matrix_eigenvectors * squared_transform_matrix_eigenvalues ** 0.5
        ) @ transform_matrix_eigenvectors.T
        # Let e = observation_error, x = state_mean, z = post_state_mean and
        # X, Y, R, N, I as above
        # For consistency with the Kalman filter mean update we require that
        # z = x + X.T @ inv(I + Y @ inv(R) @ Y.T / (N - 1)) @ Y.T @ inv(R) @ e / (N - 1)
        # Reusing U and s from above we have that
        # inv(I + Y @ inv(R) @ Y.T / (N - 1)) = diag(1 / (1 + s**2)) @ U.T
        # and so
        # z = x + X.T @ U @ diag(1 / (1 + s**2)) @ U.T @ Y @ inv(R) @ e / (N - 1)
        kalman_gain_mult_observation_error = state_deviations.T @ (
            transform_matrix_eigenvectors
            @ (
                (
                    transform_matrix_eigenvectors.T
                    @ (
                        observation_deviations
                        @ model.postmultiply_by_inv_observation_noise_covar(
                            observation_error
                        )
                    )
                )
                * squared_transform_matrix_eigenvalues
            )
            / (num_particle - 1)
        )
        post_state_mean = state_mean + kalman_gain_mult_observation_error
        post_state_deviations = transform_matrix @ state_deviations
        return (
            post_state_mean + post_state_deviations,
            post_state_mean,
            (post_state_deviations ** 2).mean(0) ** 0.5,
        )
