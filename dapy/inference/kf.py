"""Exact Kalman filter for inference in linear-Gaussian dynamical systems."""

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla


class KalmanFilter(object):
    """Exact Kalman filter for linear-Gaussian dynamical systems.

    Assumes the system dynamics are of the form

    z[0] = m + L.dot(u[0])
    x[0] = H.dot(z[0]) + J.dot(v[0])
    for t in range(1, T):
        z[t] = F.dot(z[t-1]) + G.dot(u[t])
        x[t] = H.dot(z[t]) + J.dot(v[t])

    where

       z[t]: unobserved system state at time index t,
       x[t]: observed system state at time index t,
       u[t]: zero-mean identity covariance Gaussian state noise vector at time
             index t,
       v[t]: zero-mean identity covariance Gaussian observation noise vector
             at time index t,
       m: Initial state distribution mean,
       L: Lower Cholesky factor of initial state covariance,
       H: linear observation matrix,
       J: observation noise transform matrix,
       F: linear state transition dynamics matrix,
       G: state noise transform matrix.

    For a model of this form the Kalman filter (forward) updates allow
    efficient exact calculation of the filtering densities

        p(z[t] = z_ | x[0:t]) = Normal(z_ | z_hat[t], P[t]),

    with the Gaussian form of the filtering densities a result of the
    linear dynamics and Gaussian noise assumptions.

    References:
         R. E. Kalman, A new approach to linear filtering and prediction
         problems, Transactions of the ASME -- Journal of Basic Engineering,
         Series D, 82 (1960), pp. 35--45.
    """

    def __init__(self, init_state_mean, init_state_covar, state_trans_matrix,
                 state_noise_matrix, observation_matrix, obser_noise_matrix,
                 rng=None):
        """
        Args:
            init_state_mean (array): Mean of initial state distribution.
            init_state_covar (array): Covariance matrix of initial state
                distribution.
            state_trans_matrix (array): Matrix defining linear state
                transition dynamics.
            state_noise_matrix (array): Matrix defining transformation of
                additive state noise.
            observation_matrix (array): Matrix defining linear obervation
                operator.
            obser_noise_matrix (array): Matrix defining transformation of
                additive observation noise.
            rng (RandomState): Numpy RandomState random number generator.
                Only needs to be specified if samples to be computed during
                filtering.
        """
        self.init_state_mean = init_state_mean
        self.dim_z = init_state_mean.shape[0]
        self.init_state_covar = init_state_covar
        self.state_trans_matrix = state_trans_matrix
        self.state_noise_matrix = state_noise_matrix
        self.observation_matrix = observation_matrix
        self.obser_noise_matrix = obser_noise_matrix
        self.state_noise_covar = state_noise_matrix.dot(state_noise_matrix.T)
        self.obser_noise_covar = obser_noise_matrix.dot(obser_noise_matrix.T)
        self.rng = rng

    def filter(self, x_observed, n_samples=None, return_covar=False):
        """Compute filtering density parameters.

        Args:
            x_observed (array): Observed state sequence with shape
                `(n_steps, dim_x)` where `n_steps` is number of time steps in
                sequence and `dim_x` is dimensionality of observations.
            n_samples (int): Number of samples from per-time step filtering
                distributions to return in addition to filtering distribution
                parameters.
            return_covar (bool): Whether to return full covariance matrices
                for each time steps filtering distribution in addition to
                standard deviations which are always returned.
        Returns:
            Dictionary containing arrays of filtering density parameters -
                z_mean_seq: Array of filtering distribution means at all time
                    steps.
                z_std_seq: Array of filtering distribution standard deviations
                    at all time steps.
                z_covar_seq: Array of filtering distribution full covariances
                    at all time steps (only returned if return_covar=True).
                z_samples: Array of samples from filtering distributions at
                    all time steps (only returned if n_samples != None).
        """
        n_steps, dim_x = x_observed.shape
        z_mean_seq = np.full((n_steps, self.dim_z), np.nan)
        z_std_seq = np.full((n_steps, self.dim_z), np.nan)
        if return_covar:
            z_covar_seq = np.full((n_steps, self.dim_z, self.dim_z), np.nan)
        if n_samples is not None and n_samples > 0:
            z_samples_seq = np.full((n_steps, n_samples, self.dim_z), np.nan)
        for t in range(n_steps):
            # forecast update
            if t == 0:
                z_f_mean = self.init_state_mean
                z_f_covar = self.init_state_covar
            else:
                z_f_mean = self.state_trans_matrix.dot(z_mean_seq[t-1])
                z_f_covar = (
                    self.state_trans_matrix.dot(z_a_covar).dot(
                        self.state_trans_matrix.T) +
                    self.state_noise_covar)
            # analysis update
            x_f_mean = self.observation_matrix.dot(z_f_mean)
            x_f_covar = (
                self.observation_matrix.dot(z_f_covar).dot(
                    self.observation_matrix.T) +
                self.obser_noise_covar)
            x_z_f_covar = z_f_covar.dot(self.observation_matrix.T)
            k_gain = nla.solve(x_f_covar, x_z_f_covar.T).T
            z_mean_seq[t] = z_f_mean + k_gain.dot(x_observed[t] - x_f_mean)
            # use more numerically stable but more expensive Joseph's form
            # for covariance update
            covar_trans = (
                np.eye(self.dim_z) - k_gain.dot(self.observation_matrix))
            z_a_covar = (
                covar_trans.dot(z_f_covar).dot(covar_trans.T) +
                k_gain.dot(self.obser_noise_covar).dot(k_gain.T))
            z_std_seq[t] = z_a_covar.diagonal()**0.5
            if return_covar:
                z_covar_seq[t] = z_a_covar
            if n_samples is not None and n_samples > 0:
                z_a_covar_chol = sla.cholesky(z_a_covar, lower=False)
                norm_smp = self.rng.normal(size=(n_samples, self.dim_z))
                z_samples_seq[t] = (
                    z_mean_seq[t][None, :] + norm_smp.dot(z_a_covar_chol))
        results = {'z_mean_seq': z_mean_seq, 'z_std_seq': z_std_seq}
        if return_covar:
            results['z_covar_seq'] = z_covar_seq
        if n_samples is not None and n_samples > 0:
            results['z_samples_seq'] = z_samples_seq
        return results
