"""Model with linear dynamics and observations and additive Gaussian noise."""

import numpy as np
import scipy.linalg as la
from dapy.models.base import AbstractModel, inherit_docstrings


def generate_random_parameters(dim_z, dim_x, rng):
    """Generate random parameters for linear-Gaussian model.

    Args:
        dim_z (integer): State dimension.
        dim_x (integer): Observation dimension.
        rng (RandomState): Numpy RandomState random number generator.

    Returns:
        Dictionary of generated system parameters.
    """
    params = {}
    params['init_state_mean'] = rng.normal(size=dim_z)
    temp = rng.normal(size=(dim_z, dim_z)) * (0.5 / dim_z)**0.5
    params['init_state_covar'] = temp.dot(temp.T)
    params['state_trans_matrix'] = rng.normal(
        size=(dim_z, dim_z)) * (0.5 / dim_z)**0.5
    params['state_noise_matrix'] = rng.normal(
        size=(dim_z, dim_z)) * (0.5 / dim_z)**0.5
    params['observation_matrix'] = rng.normal(
        size=(dim_x, dim_z)) * (1. / (dim_z + dim_x))**0.5
    params['obser_noise_matrix'] = rng.normal(
        size=(dim_x, dim_x)) * (0.5 / dim_x)**0.5
    return params


@inherit_docstrings
class LinearGaussianModel(AbstractModel):
    """Model with linear dynamics and observations and additive Gaussian noise.

    The model system dynamics are of the form

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
   """

    def __init__(self, init_state_mean, init_state_covar, state_trans_matrix,
                 observation_matrix, state_noise_matrix=None,
                 state_noise_covar=None, obser_noise_matrix=None,
                 obser_noise_covar=None, rng=None):
        """
        Args:
            init_state_mean (array): Mean of initial state distribution.
            init_state_covar (array): Covariance matrix of initial state
                distribution.
            state_trans_matrix (array): Matrix defining linear state
                transition dynamics.
            observation_matrix (array): Matrix defining linear obervation
                operator.
            state_noise_matrix (array): Matrix defining transformation of
                additive state noise. Either this or state_noise_covar should
                be defined but not both.
            state_noise_covar (array): Matrix defining covariance of additive
                state noise. Either this or state_noise_matrix should be
                defined but not both.
            obser_noise_matrix (array): Matrix defining transformation of
                additive observation noise. Either this or obser_noise_covar
                should be defined but not both.
            obser_noise_covar (array): Matrix defining covariance of additive
                observation noise. Either this or obser_noise_matrix should be
                defined but not both.
            rng (RandomState): Numpy RandomState random number generator.
        """
        # check for dimensional consistency
        assert state_trans_matrix.shape[0] == state_trans_matrix.shape[1]
        assert state_trans_matrix.shape[0] == observation_matrix.shape[1]
        assert state_trans_matrix.shape[0] == init_state_mean.shape[0]
        assert state_trans_matrix.shape[0] == init_state_covar.shape[0]
        assert state_trans_matrix.shape[0] == init_state_covar.shape[1]
        dim_x, dim_z = observation_matrix.shape
        self.state_trans_matrix = state_trans_matrix
        self.observation_matrix = observation_matrix
        self.init_state_mean = init_state_mean
        self.init_state_covar = init_state_covar
        self.init_state_covar_chol = la.cholesky(init_state_covar, lower=True)
        if (state_noise_matrix is None and state_noise_covar is None) or (
                state_noise_matrix is not None and
                state_noise_covar is not None):
            raise ValueError(
                'One and only one of state_noise_matrix and state_noise_covar '
                'should be specified.'
            )
        if state_noise_matrix is not None:
            state_noise_covar = state_noise_matrix.dot(state_noise_matrix.T)
        self.state_noise_covar_chol = la.cholesky(
            state_noise_covar, lower=True)
        if state_noise_covar is not None:
            state_noise_matrix = self.state_noise_covar_chol
        assert state_noise_matrix.shape == (dim_z, dim_z)
        self.state_noise_matrix = state_noise_matrix
        self.state_noise_covar = state_noise_covar
        if (obser_noise_matrix is None and obser_noise_covar is None) or (
                obser_noise_matrix is not None and
                obser_noise_covar is not None):
            raise ValueError(
                'One and only one of obser_noise_matrix and obser_noise_covar '
                'should be specified.'
            )
        if obser_noise_covar is None:
            obser_noise_covar = obser_noise_matrix.dot(obser_noise_matrix.T)
        self.obser_noise_covar_chol = la.cholesky(
            obser_noise_covar, lower=True)
        if obser_noise_matrix is None:
            obser_noise_matrix = self.obser_noise_covar_chol
        assert obser_noise_matrix.shape == (dim_x, dim_x)
        self.obser_noise_matrix = obser_noise_matrix
        self.obser_noise_covar = obser_noise_covar
        super(LinearGaussianModel, self).__init__(dim_z, dim_x, rng)

    def init_state_sampler(self, n=None):
        if n is None:
            return self.init_state_mean + self.init_state_covar_chol.dot(
                self.rng.normal(size=self.dim_z))
        else:
            return (
                self.init_state_mean +
                self.rng.normal(size=(n, self.dim_z)).dot(
                    self.init_state_covar_chol.T)
            )

    def next_state_sampler(self, z, t=None):
        if z.ndim == 1:
            return (
                self.state_trans_matrix.dot(z) +
                self.state_noise_matrix.dot(
                    self.rng.normal(size=self.state_noise_matrix.shape[1]))
            )
        else:
            return (
                z.dot(self.state_trans_matrix.T) +
                self.rng.normal(
                    size=(z.shape[0], self.state_noise_matrix.shape[1])).dot(
                        self.state_noise_matrix.T
                    )
            )

    def observation_sampler(self, z, t=None):
        if z.ndim == 1:
            return (
                self.observation_matrix.dot(z) +
                self.obser_noise_matrix.dot(
                    self.rng.normal(size=self.dim_x))
            )
        else:
            return (
                z.dot(self.observation_matrix.T) +
                self.rng.normal(size=(z.shape[0], self.dim_x)).dot(
                        self.obser_noise_matrix.T)
            )

    def log_prob_dens_init_state(self, z):
        n = z - self.init_state_mean
        return -(
            0.5 * (n.T * la.cho_solve(
                (self.init_state_covar_chol, True), n.T)).sum(0) +
            0.5 * self.dim_z * np.log(2 * np.pi) +
            np.log(self.init_state_covar_chol.diagonal()).sum()
        )

    def log_prob_dens_state_transition(self, z_n, z_c, t=None):
        n = z_n - z_c.dot(self.state_trans_matrix.T)
        return -(
            0.5 * (n.T * la.cho_solve(
                (self.state_noise_covar_chol, True), n.T)).sum(0) +
            0.5 * self.dim_z * np.log(2 * np.pi) +
            np.log(self.state_noise_covar_chol.diagonal()).sum()
        )

    def log_prob_dens_obs_gvn_state(self, x, z, t=None):
        n = x - z.dot(self.observation_matrix.T)
        return -(
            0.5 * (n.T * la.cho_solve(
                (self.obser_noise_covar_chol, True), n.T)).sum(0) +
            0.5 * self.dim_x * np.log(2 * np.pi) +
            np.log(self.obser_noise_covar_chol.diagonal()).sum()
        )
