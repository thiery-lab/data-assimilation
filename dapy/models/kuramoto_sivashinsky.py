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
from dapy.utils import inherit_docstrings
from dapy.models.base import IntegratorModel, DiagonalGaussianObservationModel
from dapy.models.etdrk4integrator import FourierETDRK4Integrator


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
