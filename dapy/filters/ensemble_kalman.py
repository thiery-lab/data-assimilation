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


class EnsembleSquareRootFilter(AbstractEnsembleFilter):
    """Ensemble Kalman filter with deterministic matrix square root updates.

    The filtering distribution at each observation time index is approximated by
    alternating propagating an ensemble of state particles forward through time under
    the model dynamics and linearly transforming the ensemble according to a Monte Carlo
    estimate of the Kalman filter assimilation update due to the observations at the
    current time index. Here a 'square-root' ensemble Kalman filter assimilation update
    is used, which requires that the model has Gaussian observation noise with a
    known covariance, but compared to the 'perturbed observation' variant avoids the
    additional variance associated with sampling pseudo-observations [1].

    References:

        1. M. K. Tippett, J. L. Anderson, C. H. Bishop, T. M. Hamill,
           and J. S. Whitaker, Ensemble square root filters,
           Monthly Weather Review, 131 (2003), pp. 1485--1490.
    """

    def __init__(self, use_woodbury_identity=True):
        """
        Args:
            use_woodbury_identity: Use Woodbury matrix identity to avoid computing
                explicit inverse of a `(dim_observation, dim_observation)` shaped matrix
                within filtering loop, thus avoiding a `O(dim_observation**3)` cost in
                each operation, at the potential expense of having to perform a single
                `O(dim_observation**3)` decomposition of the observation noise
                covariance (if non-diagonal).
        """
        self.use_woodbury_identity = use_woodbury_identity

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
        # ----
        if self.use_woodbury_identity:
            # A = inv(R) @ Y.T
            a_matrix = model.premultiply_by_inv_observation_noise_covar(
                observation_deviations.T
            )
            # b = (y_obs - y_mean) @ A
            b_vector = observation_error @ a_matrix
            # C = Y @ inv(R) @ Y.T
            c_matrix = observation_deviations @ a_matrix
            # D = N * I + Y @ inv(R) @ Y.T
            d_matrix = num_particle * np.eye(num_particle) + c_matrix
            # E = inv(N * I + Y @ inv(R) @ Y.T) @ (Y @ inv(R) @ Y.T)
            e_matrix = nla.solve(d_matrix, c_matrix)
            kalman_gain_mult_observation_error = (
                b_vector - b_vector @ e_matrix) @ state_deviations / num_particle
            m_matrix = (c_matrix - c_matrix @ e_matrix) / num_particle
        else:
            prior_residual_covar = model.increment_by_observation_noise_covar(
                observation_deviations.T @ observation_deviations / num_particle
            )
            cho_factor_prior_residual_covar = sla.cho_factor(prior_residual_covar)
            inv_prior_residual_covar_observation_deviations_T = sla.cho_solve(
                cho_factor_prior_residual_covar, observation_deviations.T
            )
            kalman_gain_mult_observation_error = (
                state_deviations.T
                @ inv_prior_residual_covar_observation_deviations_T.T
                / num_particle
            ) @ observation_error
            m_matrix = (
                observation_deviations
                @ inv_prior_residual_covar_observation_deviations_T
                / num_particle)
        # ----
        post_state_mean = state_mean + kalman_gain_mult_observation_error
        sqrt_transform_matrix = nla.cholesky(np.identity(num_particle) - m_matrix)
        post_state_deviations = sqrt_transform_matrix.T @ state_deviations
        return (
            post_state_mean + post_state_deviations,
            post_state_mean,
            (post_state_deviations ** 2).mean(0) ** 0.5,
        )
