"""Simple Navier-Stokes fluid simulation on two-dimensional grid."""

import numpy as np
from dapy.utils import inherit_docstrings
from dapy.models.base import IntegratorModel, DiagonalGaussianObservationModel
from dapy.models.fluidsim2dintegrators import FourierFluidSim2dIntegrator


@inherit_docstrings
class FluidSim2DModel(IntegratorModel, DiagonalGaussianObservationModel):
    """Incompressible Navier-Stokes fluid simulation on 2D periodic grid.

    Simulates evolution of a fluid velocity field and density field of a
    carrier particle in a field on a 2-torus using a finite difference based
    implementation of the incompressible Navier-Stokes equations in
    two-dimensions. A semi-Lagrangian method is used for the advection updates
    and a FFT-based method used to implement the diffusion of the velocity and
    density fields and to project the velocity field to a divergence-free flow
    to respect the incompressibility condition [1,2].

    It is assumed the state (velocity and density fields on a 2D spatial grid)
    is potentially subject to additive Gaussian noise which is independent
    across each dimension. The observed variables are assumed to be the density
    field at all grid points with additive Gaussian noise, again assumed to be
    independent across each dimension.

    Each state vector `z` is assumed to be ordered such that if the spatial
    grid has a shape defined by the 2-tuple `grid_shape` the the following code
        z_grid = z.reshape((3,) + grid_shape)
        velocity = z_grid[:2]
        density = z_grid[2]
    will extract the velocity and density fields from the state vector as
    multi-dimensional arrays with the last two axes corresponding to the grid
    axes.


    References:
      1. Stam, Jos. A simple fluid solver based on the FFT.
         Journal of graphics tools 6.2 (2001): 43-52.
      2. Stam, Jos. Stable fluids. Proceedings of the 26th annual conference on
         Computer graphics and interactive techniques. ACM Press/Addison-Wesley
         Publishing Co., 1999.
    """

    def __init__(self, rng, grid_shape, stream_noise_scale=1e-2,
                 init_vel_ampl_scale=5e-2, vel_noise_ampl_scale=1e-2,
                 n_init_dens_region=5, dens_region_radius_min=2e-2,
                 dens_region_radius_max=2e-1, init_dens_region_val=10,
                 log_dens_noise_std=1e-2, obser_noise_std=1e-1,
                 obs_subsample=2, dt=0.01, n_steps_per_update=5,
                 density_source=0., grid_size=(2., 2.), dens_diff_coeff=2e-4,
                 visc_diff_coeff=1e-4, vort_coeff=5., use_vort_conf=False,
                 use_bfecc=True, dens_min=1e-8):
        """
        Args:
            rng (RandomState): Numpy RandomState random number generator.
            grid_shape (tuple): Grid dimensions as a 2-tuple.
            stream_noise_scale (float): Length scale parameter for random
                stream function field used to generate initial divergence-free
                velocity field. Larger values correspond to smoother fields.
            init_vel_amp_scale (float): Amplitude scale parameter for initial
                random divergence-free velocity field. Larger values correspond
                to larger magnitude velocities.
            vel_noise_ampl_scale (float): Amplitude scale parameter for
                additive divergence-free velocity noise in model dynamics.
                Larger values correspond to larger magnitude additive noise
                in the velocity field.
            n_init_dens_region (int): Number of circular regions to place in
                initial random density field.
            dens_region_radius_min (float): Positive scalar parameter defining
                minimum of uniform random radius variables used to define the
                radii of the circular regions placed in the initial density
                fields.
            dens_region_radius_min (float): Positive scalar parameter defining
                maximum of uniform random radius variables used to define the
                radii of the circular regions placed in the initial density
                fields.
            init_dens_region_val (float): Positive scalar parameter defining
                the density value within the randomly sized and placed circular
                regions in the initial density fields.
            log_dens_noise_std (float): Standard deviation of the additive
                normal noise to the logarithm of the density fields or
                equivalently multiplicative log-normal noise to the density
                fields.
            obser_noise_std (float): Standard deviation of additive Gaussian
                noise in observations. Either a scalar or array of shape
                `(dim_x,)`. Noise in each dimension assumed to be independent
                i.e. a diagonal noise covariance.
            obs_subsample (int): Factor to subsample each spatial dimension by
                in observation operator.
            dt (float): Integrator time-step.
            n_steps_per_update (int): Number of integrator time-steps between
                successive observations and generated states.
            density_source (array or None): Array defining density source field
                used to increment density field on each integrator time step.
                Should be of shape `grid_shape`. If `None` no density source
                update is applied.
            grid_size (tuple): Spatial extent of simulation grid as a 2-tuple.
            dens_diff_coeff (float): Density diffusion coefficient.
            visc_diff_coeff (float): Velocity viscous diffusion coefficient.
            vort_coeff (float): Vorticity coeffient for voriticity confinement.
            use_vort_conf (bool): Whether to apply vorticity confinement
                update on each time step to velocity field.
            use_bfecc (bool): Whether to use BFECC advection steps instead of
                first-order semi-Lagrangian method.
            dens_min (float): Lower bound for density field values.
        """
        dim_z = grid_shape[0] * grid_shape[1] * 3
        dim_x = grid_shape[0] * grid_shape[1] // obs_subsample**2
        integrator = FourierFluidSim2dIntegrator(
            grid_shape=grid_shape,
            density_source=density_source, grid_size=grid_size, dt=dt,
            dens_diff_coeff=dens_diff_coeff, visc_diff_coeff=visc_diff_coeff,
            vort_coeff=vort_coeff, use_vort_conf=use_vort_conf,
            dens_min=dens_min
        )
        self.n_init_dens_region = n_init_dens_region
        self.dens_region_radius_min = dens_region_radius_min
        self.dens_region_radius_max = dens_region_radius_max
        self.init_dens_region_val = init_dens_region_val
        self.log_dens_noise_std = log_dens_noise_std
        self.obs_subsample = obs_subsample
        self.stream_noise_scale = stream_noise_scale
        self.init_vel_ampl_scale = init_vel_ampl_scale
        self.vel_noise_ampl_scale = vel_noise_ampl_scale
        # Kernels in frequency space for generating random divergence-free
        # initial velocity and additive noise fields
        stream_noise_kernel = np.exp(
            -stream_noise_scale * integrator.wavnum_sq_grid)
        self.vel_fft_noise_kernel = np.stack([
            stream_noise_kernel * integrator.grad_1_kernel[None, :],
            stream_noise_kernel * -integrator.grad_0_kernel[:, None]
        ], axis=0) / integrator.cell_size[:, None, None]
        super(FluidSim2DModel, self).__init__(
            integrator=integrator, n_steps_per_update=n_steps_per_update,
            obser_noise_std=obser_noise_std, dim_z=dim_z, dim_x=dim_x, rng=rng)

    def init_state_sampler(self, n=None):
        if n is None:
            n = 1
            n_was_none = True
        else:
            n_was_none = False
        init_density = np.ones(
            (n,) + self.integrator.grid_shape) * self.integrator.dens_min
        u = np.stack(np.meshgrid(
            np.linspace(0, 1, self.integrator.grid_shape[0]),
            np.linspace(0, 1, self.integrator.grid_shape[1]), indexing='ij'),
            axis=-1)
        centres = self.rng.uniform(size=(n, self.n_init_dens_region, 2))
        radii = self.rng.uniform(
                self.dens_region_radius_min, self.dens_region_radius_max,
                size=(n, self.n_init_dens_region,))
        dists = np.abs(u[None, None] - centres[:, :, None, None])
        dists = (np.minimum(dists, 1 - dists)**2).sum(-1)
        init_density[(dists < radii[:, :, None, None]**2).sum(1) > 0] = (
            self.init_dens_region_val)
        noise_2d_fft = np.fft.rfft2(
            self.rng.normal(size=(n, 1) + self.integrator.grid_shape))
        init_velocity = np.fft.irfft2(
                self.init_vel_ampl_scale * noise_2d_fft *
                self.vel_fft_noise_kernel[None])
        init_state = np.concatenate([
            init_velocity.reshape((n, -1)), init_density.reshape((n, -1))
        ], axis=1)
        if n_was_none:
            return init_state[0]
        else:
            return init_state

    def next_state_sampler(self, z, t):
        if z.ndim == 1:
            z = z[None]
            z_was_one_dim = True
        else:
            z_was_one_dim = False
        n_particle = z.shape[0]
        z_next = self.next_state_func(z, t)
        velocity = z_next[:, :2 * self.integrator.n_grid].reshape(
            (n_particle, 2,) + self.integrator.grid_shape)
        density = z_next[:, 2 * self.integrator.n_grid:].reshape(
            (n_particle,) + self.integrator.grid_shape)
        noise_2d_fft = np.fft.rfft2(self.rng.normal(
                size=(n_particle, 1) + self.integrator.grid_shape))
        timestep = self.integrator.dt * self.n_steps_per_update
        vel_noise_fft = (
                timestep**0.5 * self.vel_noise_ampl_scale *
                noise_2d_fft * self.vel_fft_noise_kernel[None])
        velocity_new = np.fft.irfft2(np.fft.rfft2(velocity) + vel_noise_fft)
        density_new = density * np.exp(
            self.log_dens_noise_std *
            self.rng.normal(size=(n_particle,) + self.integrator.grid_shape))
        if z_was_one_dim:
            return np.concatenate([
                velocity_new.flatten(), density_new.flatten()])
        else:
            return np.concatenate([
                    velocity_new.reshape((n_particle, -1)),
                    density_new.reshape((n_particle, -1))], axis=-1)

    def observation_func(self, z, t):
        if z.ndim == 1:
            dens = z[-self.integrator.n_grid:].reshape(
                self.integrator.grid_shape)
            return dens[::self.obs_subsample, ::self.obs_subsample].flatten()
        else:
            dens = z[:, -self.integrator.n_grid:].reshape(
                (-1,) + self.integrator.grid_shape)
            return dens[:, ::self.obs_subsample, ::self.obs_subsample].reshape(
                    (z.shape[0], -1))
