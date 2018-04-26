"""Simple Navier-Stokes fluid simulation on two-dimensional grid."""

import numpy as np
import tqdm
from dapy.utils.doc import inherit_docstrings
from dapy.models.base import IntegratorModel, DiagonalGaussianObservationModel
from dapy.integrators.navier_stokes import StochasticNavierStokes2dIntegrator


@inherit_docstrings
class NavierStokes2dModel(IntegratorModel, DiagonalGaussianObservationModel):
    """Incompressible Navier-Stokes fluid simulation on 2D periodic grid.

    Simulates evolution of a fluid velocity field on a 2-torus using a Fourier
    spectral based implementation of the incompressible Navier-Stokes
    equations in two-dimensions. The divergence-free velocity field is
    parameterised by its curl - the voriticity. A semi-Lagrangian method is
    used for the advection updates with backwards-forwards error correction
    for improved accuracy [1].

    References:
      1. Kim, ByungMoon, Yingjie Liu, Ignacio Llamas, and Jarek Rossignac.
         FlowFixer: using BFECC for fluid simulation. Proceedings of the First
         Eurographics conference on Natural Phenomena. Eurographics
         Association, 2005.
    """

    def __init__(self, rng, grid_shape, visc_diff_coeff=1e-4,
                 vort_noise_length_scale=1e-2, init_vort_ampl_scale=5e-2,
                 vort_noise_ampl_scale=1e-2, init_vort_int_time=0.,
                 obser_noise_std=1e-1, obs_subsample=2, dt=0.01,
                 n_steps_per_update=5, grid_size=(2., 2.), obs_speed=False,
                 max_n_thread=4):
        """
        Args:
            rng (RandomState): Numpy RandomState random number generator.
            grid_shape (tuple): Grid dimensions as a 2-tuple.
            visc_diff_coeff (float): Viscous diffusion coefficient.
            vort_noise_length_scale (float): Length scale parameter for random
                noise used to generate initial vorticity and vorticity additive
                noise fields. Larger values correspond to smoother fields.
            init_vort_ampl_scale (float): Amplitude scale parameter for initial
                random vorticity field. Larger values correspond to larger
                magnitude vorticities (and so velocities).
            vort_noise_ampl_scale (float): Amplitude scale parameter for
                additive vorticity noise in model dynamics. Larger values
                correspond to larger magnitude additive noise
                in the vorticity field.
            init_vort_int_time (float): Time to integrate initial voriticity
                field from random field to reduce transient behaviour.
            obser_noise_std (float): Standard deviation of additive Gaussian
                noise in observations. Either a scalar or array of shape
                `(dim_x,)`. Noise in each dimension assumed to be independent
                i.e. a diagonal noise covariance.
            obs_subsample (int): Factor to subsample each spatial dimension by
                in observation operator.
            dt (float): Integrator time-step.
            n_steps_per_update (int): Number of integrator time-steps between
                successive observations and generated states.
            grid_size (tuple): Spatial extent of simulation grid as a 2-tuple.
            obs_speed (bool): Whether to observe speed (magnitude of velocity)
                field instead of vorticity.
        """
        dim_z = grid_shape[0] * grid_shape[1]
        dim_x = grid_shape[0] * grid_shape[1] // obs_subsample**2
        integrator = StochasticNavierStokes2dIntegrator(
            rng=rng, grid_shape=grid_shape, grid_size=grid_size, dt=dt,
            visc_diff_coeff=visc_diff_coeff,
            vort_noise_length_scale=vort_noise_length_scale,
            vort_noise_ampl_scale=vort_noise_ampl_scale,
            max_n_thread=max_n_thread)
        self.obs_subsample = obs_subsample
        self.vort_noise_length_scale = vort_noise_length_scale
        self.init_vort_ampl_scale = init_vort_ampl_scale
        self.vort_noise_ampl_scale = vort_noise_ampl_scale
        self.init_vort_int_time = init_vort_int_time
        self.obs_speed = obs_speed
        super(NavierStokes2dModel, self).__init__(
            integrator=integrator, n_steps_per_update=n_steps_per_update,
            obser_noise_std=obser_noise_std, dim_z=dim_z, dim_x=dim_x, rng=rng)

    def init_state_sampler(self, n=None):
        if n is None:
            n = 1
            n_was_none = True
        else:
            n_was_none = False
        noise_2d_fft = np.fft.rfft2(
            self.rng.normal(size=(n,) + self.integrator.grid_shape))
        vorticity = np.fft.irfft2(
            self.init_vort_ampl_scale * self.integrator.vort_noise_kernel *
            noise_2d_fft)
        n_step = int(self.init_vort_int_time / self.integrator.dt)
        # Integrate vorticity field forward in time.
        for s in tqdm.trange(n_step, desc='Integrating initial state',
                             unit='timestep'):
            vorticity = self.integrator.update_vorticity(vorticity)
        if n_was_none:
            return vorticity[0].flatten()
        else:
            return vorticity.reshape((n, -1))

    def next_state_sampler(self, z, t):
        return self.next_state_func(z, t)

    def observation_func(self, z, t):
        skip = self.obs_subsample
        if z.ndim == 1:
            vorticity = z.reshape(self.integrator.grid_shape)
            if self.obs_speed:
                velocity = self.integrator.velocity_from_vorticity(vorticity)
                speed = (velocity**2).sum(-3)**0.5
                return speed[::skip, ::skip].flatten()
            else:
                return vorticity[::skip, ::skip].flatten()

        else:
            vorticity = z.reshape((-1,) + self.integrator.grid_shape)
            if self.obs_speed:
                velocity = self.integrator.velocity_from_vorticity(vorticity)
                speed = (velocity**2).sum(-3)**0.5
                return speed[:, ::skip, ::skip].reshape((z.shape[0], -1))
            else:
                return vorticity[:, ::skip, ::skip].reshape((z.shape[0], -1))
