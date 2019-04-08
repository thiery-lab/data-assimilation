"""One-dimensional Kuramoto-Sivashinsky PDE model on a periodic domain.

The equation exhibits spatio-temporal chaos.

The governing partial differential equation is

  dz/dt = -d^4z/ds^4 - d^2z/ds^2 - z * dz/ds

where `s` is the spatial coordinate, `t` the time coordinate and `z(s, t)` the
state field.

References:

  1. Kuramoto and Tsuzuki. Persistent propagation of concentration waves
     in dissipative media far from thermal equilibrium.
     Progress in Theoretical Physcs, 55 (1976) pp. 356–369.
  2. Sivashinsky. Nonlinear analysis of hydrodynamic instability in laminar
     flames I. Derivation of basic equations.
     Acta Astronomica, 4 (1977) pp. 1177–1206.
"""

import numpy as np
import pyfftw.interfaces.numpy_fft as fft
from dapy.utils.doc import inherit_docstrings
from dapy.models.base import IntegratorModel, DiagonalGaussianObservationModel
from dapy.integrators.etdrk4 import FourierETDRK4Integrator


@inherit_docstrings
class KuramotoSivashinskyModel(
        IntegratorModel, DiagonalGaussianObservationModel):

    def __init__(self, rng, n_grid=512, l_param=16, dt=0.25,
                 n_steps_per_update=4, obser_noise_std=0.1, obs_subsample=4,
                 init_state_ampl_scale=5., state_noise_ampl_scale=0.1,
                 state_noise_length_scale=1., n_roots=16):
        self.l_param = l_param
        self.timestep = n_steps_per_update * dt
        self.obs_subsample = obs_subsample
        self.init_state_ampl_scale = init_state_ampl_scale
        self.state_noise_ampl_scale = state_noise_ampl_scale
        self.state_noise_length_scale = state_noise_length_scale

        def linear_operator(freqs, freqs_sq):
            return freqs_sq - freqs_sq**2

        def nonlinear_operator(v, freqs, freqs_sq):
            return -0.5j * freqs * np.fft.rfft(np.fft.irfft(v)**2)

        integrator = FourierETDRK4Integrator(
            linear_operator=linear_operator,
            nonlinear_operator=nonlinear_operator, n_grid=n_grid,
            grid_size=l_param * 2 * np.pi, dt=dt, n_roots=n_roots)

        self.state_noise_kernel = np.exp(
            -0.5 * integrator.freqs_sq * state_noise_length_scale**2)

        super(KuramotoSivashinskyModel, self).__init__(
            integrator=integrator, n_steps_per_update=n_steps_per_update,
            obser_noise_std=obser_noise_std, dim_z=n_grid,
            dim_x=n_grid // obs_subsample, rng=rng)

    def init_state_sampler(self, n=None):
        if n is None:
            n = 1
            n_was_none = True
        else:
            n_was_none = False
        z_init = self.init_state_ampl_scale * fft.irfft(
            fft.rfft(self.rng.normal(size=(n, self.dim_z))) *
            self.state_noise_kernel)
        if n_was_none:
            return z_init[0]
        else:
            return z_init

    def next_state_sampler(self, z, t):
        if z.ndim == 1:
            z = z[None]
            z_was_one_dim = True
        else:
            z_was_one_dim = False
        n_particle = z.shape[0]
        z_next = self.next_state_func(z, t) + (
            self.timestep**0.5 * self.state_noise_ampl_scale * fft.irfft(
                fft.rfft(self.rng.normal(size=z.shape)) *
                self.state_noise_kernel))
        if z_was_one_dim:
            return z_next[0]
        else:
            return z_next

    def observation_func(self, z, t):
        return z.T[::self.obs_subsample].T


@inherit_docstrings
class KuramotoSivashinskySPDEModel(DiagonalGaussianObservationModel):

    def __init__(self, rng, n_grid=512, l_param=16, decay_coeff=1. / 6,
                 dt=0.25, n_steps_per_update=10, obs_noise_std=0.5,
                 obs_space_indices=slice(4, None, 8), obs_func=None,
                 init_state_ampl_scale=1., init_state_length_scale=1.,
                 state_noise_ampl_scale=1., state_noise_length_scale=1.,
                 n_roots=16):
        self.l_param = l_param
        self.decay_coeff = decay_coeff
        self.dt = dt
        self.n_steps_per_update = n_steps_per_update
        self.obs_space_indices = obs_space_indices
        if obs_func is None:
            def obs_func(z): return z
        self.obs_func = obs_func
        self.init_state_ampl_scale = init_state_ampl_scale
        self.init_state_length_scale = init_state_length_scale
        self.state_noise_ampl_scale = state_noise_ampl_scale
        self.state_noise_length_scale = state_noise_length_scale

        def linear_operator(freqs, freqs_sq):
            return freqs_sq - freqs_sq**2 - decay_coeff

        def nonlinear_operator(v, freqs, freqs_sq):
            return -0.5j * freqs * np.fft.rfft(np.fft.irfft(v)**2)

        grid_size = l_param * 2 * np.pi

        self.integrator = FourierETDRK4Integrator(
            linear_operator=linear_operator,
            nonlinear_operator=nonlinear_operator, n_grid=n_grid,
            grid_size=grid_size, dt=dt, n_roots=n_roots)

        self.init_state_kernel = init_state_ampl_scale * np.exp(
            -0.5 * self.integrator.freqs_sq * init_state_length_scale**2
        ) * (init_state_length_scale / grid_size)**0.5
        self.state_noise_kernel = state_noise_ampl_scale * np.exp(
            -0.5 * self.integrator.freqs_sq * state_noise_length_scale**2
        ) * (state_noise_length_scale / grid_size)**0.5

        super(KuramotoSivashinskySPDEModel, self).__init__(
            obser_noise_std=obs_noise_std, dim_z=n_grid,
            dim_x=len(range(n_grid)[obs_space_indices]), rng=rng)

    def _to_rfft_rep(self, u):
        return np.concatenate([
            self.dim_z * u[..., 0:1],
            self.dim_z * (u[..., 1:-1:2] + 1j * u[..., 2:-1:2]) / 2**0.5,
            self.dim_z * u[..., -1:]
        ], -1)

    def init_state_sampler(self, n=None):
        if n is None:
            n = 1
            n_was_none = True
        else:
            n_was_none = False
        v = (
            self.init_state_kernel *
            self._to_rfft_rep(self.rng.normal(size=(n, self.dim_z))))
        for s in range(self.n_steps_per_update):
            v = self.integrator.step_fft(v)
            v += (
                self.dt**0.5 * self.state_noise_kernel *
                self._to_rfft_rep(self.rng.normal(size=(n, self.dim_z))))
        z_init = fft.irfft(v)
        if n_was_none:
            return z_init[0]
        else:
            return z_init

    def next_state_sampler(self, z, t):
        if z.ndim == 1:
            z = z[None]
            z_was_one_dim = True
        else:
            z_was_one_dim = False
        n_particle = z.shape[0]
        v = fft.rfft(z)
        for s in range(self.n_steps_per_update):
            v = self.integrator.step_fft(v)
            v += (
                self.dt**0.5 * self.state_noise_kernel *
                self._to_rfft_rep(self.rng.normal(size=z.shape)))
        z_next = fft.irfft(v)
        if z_was_one_dim:
            return z_next[0]
        else:
            return z_next

    def observation_func(self, z, t):
        return self.obs_func(z[..., self.obs_space_indices])
