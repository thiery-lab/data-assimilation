"""Simple Navier-Stokes fluid simulation on two-dimensional grid."""

import numpy as np
from dapy.utils import inherit_docstrings
from dapy.models.base import DiagonalGaussianIntegratorModel
from dapy.models.fluidsim2dintegrators import FourierFluidSim2dIntegrator


@inherit_docstrings
class FluidSim2DModel(DiagonalGaussianIntegratorModel):
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

    def __init__(self, rng, grid_shape, init_state_mean, init_state_std,
                 state_noise_std, obser_noise_std, dt=0.05,
                 n_steps_per_update=1, density_source=0.,
                 grid_size=(2., 2.), dens_diff_coeff=2e-4,
                 visc_diff_coeff=1e-4, vort_coeff=5., use_vort_conf=True,
                 use_bfecc=True):
        """
        Args:
            rng (RandomState): Numpy RandomState random number generator.
            grid_shape (tuple): Grid dimensions as a 2-tuple.
            init_state_mean (float or array): Initial state distribution mean.
                Either a scalar or array of shape `(dim_z,)`.
            init_state_std (float or array): Initial state distribution
                standard deviation. Either a scalar or array of shape
                `(dim_z,)`.
            state_noise_std (float or array): Standard deviation of additive
                Gaussian noise in state update. Either a scalar or array of
                shape `(dim_z,)`. Noise in each dimension assumed to be independent i.e. a diagonal noise covariance. If zero or None deterministic dynamics are assumed.
            obser_noise_std (float): Standard deviation of additive Gaussian
                noise in observations. Either a scalar or array of shape
                `(dim_x,)`. Noise in each dimension assumed to be independent
                i.e. a diagonal noise covariance.
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
        """
        dim_z = grid_shape[0] * grid_shape[1] * 3
        dim_x = grid_shape[0] * grid_shape[1]
        integrator = FourierFluidSim2dIntegrator(
            grid_shape=grid_shape,
            density_source=density_source, grid_size=grid_size, dt=dt,
            dens_diff_coeff=dens_diff_coeff, visc_diff_coeff=visc_diff_coeff,
            vort_coeff=vort_coeff, use_vort_conf=use_vort_conf
        )
        super(FluidSim2DModel, self).__init__(
            integrator=integrator, n_steps_per_update=n_steps_per_update,
            dim_z=dim_z, dim_x=dim_x, rng=rng,
            init_state_mean=init_state_mean, init_state_std=init_state_std,
            state_noise_std=state_noise_std, obser_noise_std=obser_noise_std
        )

    def observation_func(self, z, t):
        return z.T[-self.dim_x:].T
