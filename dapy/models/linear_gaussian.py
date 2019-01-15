"""Model with linear dynamics and observations and additive Gaussian noise."""

import numpy as np
import scipy.linalg as la
from dapy.utils.doc import inherit_docstrings
from dapy.models.base import AbstractModel


def generate_random_dense_parameters(dim_z, dim_x, rng):
    """Generate random parameters for dense linear Gaussian model.

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
class DenseLinearGaussianModel(AbstractModel):
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
        super(DenseLinearGaussianModel, self).__init__(dim_z, dim_x, rng)

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

    def next_state_func(self, z, t):
        return z.dot(self.state_trans_matrix.T)

    def next_state_sampler(self, z, t):
        if z.ndim == 1:
            return (
                self.next_state_func(z, t) +
                self.state_noise_matrix.dot(
                    self.rng.normal(size=self.state_noise_matrix.shape[1]))
            )
        else:
            return (
                self.next_state_func(z, t) +
                self.rng.normal(
                    size=(z.shape[0], self.state_noise_matrix.shape[1])).dot(
                        self.state_noise_matrix.T
                    )
            )

    def observation_func(self, z, t):
        return z.dot(self.observation_matrix.T)

    def observation_sampler(self, z, t):
        if z.ndim == 1:
            return (
                self.observation_func(z, t) +
                self.obser_noise_matrix.dot(
                    self.rng.normal(size=self.dim_x))
            )
        else:
            return (
                self.observation_func(z, t) +
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

    def log_prob_dens_state_transition(self, z_n, z_c, t):
        n = z_n - z_c.dot(self.state_trans_matrix.T)
        return -(
            0.5 * (n.T * la.cho_solve(
                (self.state_noise_covar_chol, True), n.T)).sum(0) +
            0.5 * self.dim_z * np.log(2 * np.pi) +
            np.log(self.state_noise_covar_chol.diagonal()).sum()
        )

    def log_prob_dens_obs_gvn_state(self, x, z, t):
        n = x - z.dot(self.observation_matrix.T)
        return -(
            0.5 * (n.T * la.cho_solve(
                (self.obser_noise_covar_chol, True), n.T)).sum(0) +
            0.5 * self.dim_x * np.log(2 * np.pi) +
            np.log(self.obser_noise_covar_chol.diagonal()).sum()
        )


class StochasticTurbulenceModel(DenseLinearGaussianModel):

    def __init__(self, dim_z, rng, obs_subsample, dt, grid_size,
                 damp_coeff, adv_coeff, diff_coeff, state_noise_std,
                 obs_noise_std, init_state_std):
        self.obs_subsample = obs_subsample
        self.dt = dt
        self.grid_size = grid_size
        self.damp_coeff = damp_coeff
        self.adv_coeff = adv_coeff
        self.diff_coeff = diff_coeff
        self.state_noise_std = state_noise_std * dt**0.5
        self.obs_noise_std = obs_noise_std
        self.init_state_std = init_state_std
        first_col = np.zeros(dim_z)
        alpha = diff_coeff * dt / grid_size**2
        beta = adv_coeff * dt / (2 * grid_size)
        first_col[0] = 1. - 2. * alpha - dt * self.damp_coeff
        first_col[1] = alpha + beta
        first_col[-1] = alpha - beta
        state_trans_matrix = la.circulant(first_col)
        dim_x = dim_z // obs_subsample
        observation_matrix = np.zeros((dim_x, dim_z))
        for i in range(dim_x):
            observation_matrix[i, i * obs_subsample] = 1.
        super(StochasticTurbulenceModel, self).__init__(
            init_state_mean=np.zeros(dim_z),
            init_state_covar=np.eye(dim_z) * init_state_std**2,
            state_trans_matrix=state_trans_matrix,
            observation_matrix=observation_matrix,
            state_noise_matrix=np.eye(dim_z) * state_noise_std * dt**0.5,
            state_noise_covar=None,
            obser_noise_matrix=np.eye(dim_x) * obs_noise_std,
            obser_noise_covar=None,
            rng=rng
        )


class OperatorStochasticTurbulenceModel(AbstractModel):

    def __init__(self, dim_z, rng, obs_subsample, dt, grid_size,
                 damp_coeff, adv_coeff, diff_coeff, state_noise_std,
                 obs_noise_std, init_state_std):
        self.obs_subsample = obs_subsample
        self.dt = dt
        self.grid_size = grid_size
        self.damp_coeff = damp_coeff
        self.adv_coeff = adv_coeff
        self.diff_coeff = diff_coeff
        self.state_noise_std = state_noise_std * dt**0.5
        self.obs_noise_std = obs_noise_std
        self.init_state_std = init_state_std
        alpha = diff_coeff * dt / grid_size**2
        beta = adv_coeff * dt / (2 * grid_size)
        self.coeff_0 = 1. - 2. * alpha - dt * self.damp_coeff
        self.coeff_p1 = alpha + beta
        self.coeff_m1 = alpha - beta
        dim_x = dim_z // obs_subsample
        self.init_state_mean = np.zeros(dim_z)
        self.init_state_covar = np.eye(dim_z) * init_state_std**2
        self.state_noise_matrix = np.eye(dim_z) * state_noise_std * dt**0.5
        self.obser_noise_matrix = np.eye(dim_x) * obs_noise_std
        super(OperatorStochasticTurbulenceModel, self).__init__(
            dim_z, dim_x, rng)

    def init_state_sampler(self, n=None):
        if n is None:
            return self.init_state_std * self.rng.normal(size=self.dim_z)
        else:
            return self.init_state_std * self.rng.normal(size=(n, self.dim_z))

    def next_state_func(self, z, t):
        return (
            z * self.coeff_0 +
            np.roll(z, shift=+1, axis=-1) * self.coeff_p1 +
            np.roll(z, shift=-1, axis=-1) * self.coeff_m1)

    def next_state_sampler(self, z, t):
        return (
            self.next_state_func(z, t) +
            self.state_noise_std * self.rng.normal(size=z.shape))

    def observation_func(self, z, t):
        return z.T[::self.obs_subsample].T

    def observation_sampler(self, z, t):
        x_mean = self.observation_func(z, t)
        return x_mean + self.obs_noise_std * self.rng.normal(size=x_mean.shape)

    def log_prob_dens_init_state(self, z):
        return -0.5 * (
            (((z - self.init_state_mean) / self.init_state_std)**2).sum(-1) +
            self.dim_z * np.log(2 * np.pi * self.init_state_std**2))

    def log_prob_dens_state_transition(self, z_n, z_c, t):
        z_n_mean = self.next_state_func(z_c)
        return -0.5 * (
            (((z_n - z_n_mean) / self.state_noise_std)**2).sum(-1) +
            self.dim_z * np.log(2 * np.pi * self.state_noise_std**2))

    def log_prob_dens_obs_gvn_state(self, x, z, t):
        x_mean = self.observation_func(z, t)
        return -0.5 * (
            (((x - x_mean) / self.obs_noise_std)**2).sum(-1) +
            self.dim_x * np.log(2 * np.pi * self.obs_noise_std**2))


class SpectralStochasticTurbulenceModel(AbstractModel):

    def __init__(self, dim_z, rng, obs_subsample, dt, grid_size,
                 damp_coeff, adv_coeff, diff_coeff, state_noise_ampl,
                 state_noise_length_scale, obs_noise_std, obs_offset=0):
        self.obs_subsample = obs_subsample
        self.obs_offset = obs_offset
        self.dt = dt
        self.grid_size = grid_size
        self.damp_coeff = damp_coeff
        self.adv_coeff = adv_coeff
        self.diff_coeff = diff_coeff
        self.state_noise_ampl = state_noise_ampl
        self.state_noise_length_scale = state_noise_length_scale
        self.obs_noise_std = obs_noise_std
        freqs = np.fft.rfftfreq(
            dim_z, grid_size / dim_z) * 2 * np.pi
        freqs_sq = freqs**2
        if dim_z % 2 == 0:
            freqs[dim_z // 2] = 0
        state_noise_scale = state_noise_ampl * dim_z**0.5 * np.exp(
            -freqs_sq * state_noise_length_scale**2)
        gamma = diff_coeff * freqs_sq + damp_coeff
        self.state_update_fourier_kernel = np.exp(
            (-gamma + 1j * adv_coeff * freqs) * dt)
        self.state_noise_fourier_kernel = state_noise_scale * (
            1. - np.exp(-2 * gamma * dt))**0.5 / (2 * gamma)**0.5
        self.state_noise_std_fourier = self.real_std_from_fourier_kernel(
            self.state_noise_fourier_kernel)
        self.init_state_fourier_kernel = state_noise_scale / (2 * gamma)**0.5
        self.init_state_std_fourier = self.real_std_from_fourier_kernel(
            self.init_state_fourier_kernel)
        dim_x = dim_z // obs_subsample
        self.init_state_mean = np.zeros(dim_z)
        fft_trans_mtx = np.fft.irfft(self.to_complex(np.eye(dim_z))).T
        self.init_state_covar = (
            fft_trans_mtx * self.init_state_std_fourier**2).dot(
                fft_trans_mtx.T)
        self.state_noise_matrix = fft_trans_mtx * self.state_noise_std_fourier
        self.obser_noise_matrix = np.eye(dim_x) * obs_noise_std
        super(SpectralStochasticTurbulenceModel, self).__init__(
            dim_z, dim_x, rng)

    def to_real(self, c):
        return np.concatenate([c.real, c[..., 1:-1].imag], axis=-1)

    def to_complex(self, r):
        c = r[..., :r.shape[-1] // 2 + 1] * (1 + 0j)
        c[..., 1:-1] += r[..., r.shape[-1] // 2 + 1:] * 1j
        return c

    def real_std_from_fourier_kernel(self, kernel):
        dim_z = (kernel.shape[0] - 1) * 2
        s = np.ones(dim_z) * dim_z**0.5 / 2**0.5
        s[0] *= 2**0.5
        s[dim_z // 2] *= 2**0.5
        s[:dim_z // 2 + 1] *= kernel
        s[dim_z // 2 + 1:] *= kernel[1:-1]
        return s

    def init_state_sampler(self, n=None):
        if n is None:
            u = self.rng.normal(size=self.dim_z)
        else:
            u = self.rng.normal(size=(n, self.dim_z))
        return np.fft.irfft(self.to_complex(self.init_state_std_fourier * u))

    def next_state_func(self, z, t):
        return np.fft.irfft(self.state_update_fourier_kernel * np.fft.rfft(z))

    def next_state_sampler(self, z, t):
        n = self.rng.normal(size=z.shape)
        return (
            self.next_state_func(z, t) +
            np.fft.irfft(self.to_complex(self.state_noise_std_fourier * n)))

    def observation_func(self, z, t):
        return z[..., self.obs_offset::self.obs_subsample]

    def observation_sampler(self, z, t):
        x_mean = self.observation_func(z, t)
        return x_mean + self.obs_noise_std * self.rng.normal(size=x_mean.shape)

    def log_prob_dens_obs_gvn_state(self, x, z, t):
        x_mean = self.observation_func(z, t)
        return -0.5 * (
            (((x - x_mean) / self.obs_noise_std)**2).sum(-1) +
            self.dim_x * np.log(2 * np.pi * self.obs_noise_std**2))
