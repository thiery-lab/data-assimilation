import numpy as np
from dapy.models.base import (
    DiagonalGaussianIntegratorModel, inherit_docstrings)
from dapy.models.fluidsim2dintegrators import FourierFluidSim2dIntegrator


@inherit_docstrings
class FluidSim2DModel(DiagonalGaussianIntegratorModel):

    def __init__(self, rng, grid_shape, init_state_mean, init_state_std,
                 state_noise_std, obser_noise_std, density_source=0.,
                 grid_size=(2., 2.), dt=0.05, dens_diff_coeff=2e-4,
                 visc_diff_coeff=1e-4, vort_coeff=5., n_threads=1):
        dim_z = grid_shape[0] * grid_shape[1] * 3
        dim_x = grid_shape[0] * grid_shape[1]
        self.dt = dt
        self.n_steps_per_update = 1
        integrator = FourierFluidSim2dIntegrator(
            grid_shape=grid_shape,
            density_source=density_source, grid_size=grid_size, dt=dt,
            dens_diff_coeff=dens_diff_coeff, visc_diff_coeff=visc_diff_coeff,
            vort_coeff=vort_coeff, n_threads=n_threads
        )
        super(FluidSim2DModel, self).__init__(
            integrator=integrator, dim_z=dim_z, dim_x=dim_x, rng=rng,
            init_state_mean=init_state_mean, init_state_std=init_state_std,
            state_noise_std=state_noise_std, obser_noise_std=obser_noise_std
        )

    def observation_func(self, z):
        return z.T[-self.dim_x:].T
