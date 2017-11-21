"""Simple Navier-Stokes fluid simulation on two-dimensional grid.

Based on methods proposed in

> Stam, Jos. Stable fluids. Proceedings of the 26th annual conference on
> Computer graphics and interactive techniques. ACM Press/Addison-Wesley
> Publishing Co., 1999.
> Fedkiw, Ronald, Jos Stam, and Henrik Wann Jensen. Visual simulation of smoke.
> Proceedings of the 28th annual conference on Computer graphics and
> interactive techniques. ACM, 2001.
"""

import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft


class AbstractFluidSim2dIntegrator(object):

    def __init__(self, grid_shape, grid_size=(2., 2.), dt=0.05,
                 dens_diff_coeff=2e-4, visc_diff_coeff=1e-4, vort_coeff=5.):
        self.grid_shape = grid_shape
        self.grid_size = grid_size
        self.dt = dt
        self.cell_size = np.array([
            self.grid_size[0] / float(self.grid_shape[0]),
            self.grid_size[0] / float(self.grid_shape[1])
        ])
        self.dt_cell_size_ratio = np.array([
            self.dt / self.cell_size[0],
            self.dt / self.cell_size[1]
        ])
        self.dens_diff_coeff = dens_diff_coeff
        self.visc_diff_coeff = visc_diff_coeff
        self.vort_coeff = vort_coeff

    def forward_integrate(self, z_curr, z_next, start_time):
        n = self.grid_shape[0] * self.grid_shape[1]
        for z_c, z_n in zip(z_curr, z_next):
            velocity = z_c[:2*n].reshape((2,) + self.grid_shape)
            density = z_c[2*n:].reshape(self.grid_shape)
            velocity = self.update_velocity(velocity)
            density = self.update_density(density, velocity)
            z_n[:2*n] = velocity.flatten()
            z_n[2*n:] = density.flatten()


class FourierFluidSim2dIntegrator(AbstractFluidSim2dIntegrator):

    def __init__(self, grid_shape, density_source=0., grid_size=(2., 2.),
                 dt=0.05, dens_diff_coeff=2e-4, visc_diff_coeff=1e-4,
                 vort_coeff=5., n_threads=1):
        super(FourierFluidSim2dIntegrator, self).__init__(
            grid_shape=grid_shape, grid_size=grid_size, dt=dt,
            dens_diff_coeff=dens_diff_coeff, visc_diff_coeff=visc_diff_coeff,
            vort_coeff=vort_coeff
        )
        self.n_threads = n_threads
        self.density_source = np.zeros(
            (self.grid_shape[0], self.grid_shape[1]))
        self.cell_indices = np.array(np.meshgrid(
                np.arange(self.grid_shape[0]),
                np.arange(self.grid_shape[1]), indexing='ij'))
        self.freq_grid_0 = fft.fftfreq(grid_shape[0], 1. / grid_shape[0])
        self.freq_grid_1 = np.arange(grid_shape[1] // 2 + 1)
        self.wavnum_sq_grid = (
            self.freq_grid_0[:, None]**2 + self.freq_grid_1[None, :]**2)
        # Clip wave number square values to above small positive constant
        # to avoid divide by zero warnings
        wavnum_sq_grid_clipped = np.maximum(self.wavnum_sq_grid, 1e-8)
        self.fft_proj_coeff_00 = (
            1. - self.freq_grid_0[:, None]**2 / wavnum_sq_grid_clipped)
        self.fft_proj_coeff_11 = (
            1. - self.freq_grid_1[None, :]**2 / wavnum_sq_grid_clipped)
        self.fft_proj_coeff_01 = (
            self.freq_grid_0[:, None] * self.freq_grid_1[None, :] /
            wavnum_sq_grid_clipped)
        self.visc_diff_kernel = np.exp(
            -self.wavnum_sq_grid * dt * visc_diff_coeff)
        self.dens_diff_kernel = np.exp(
            -self.wavnum_sq_grid * dt * dens_diff_coeff)

    def project_and_diffuse_velocity(self, velocity):
        velocity_fft = fft.rfft2(velocity, threads=self.n_threads)
        velocity_fft_new = self.visc_diff_kernel * np.stack([
            self.fft_proj_coeff_00 * velocity_fft[0] -
            self.fft_proj_coeff_01 * velocity_fft[1],
            self.fft_proj_coeff_11 * velocity_fft[1] -
            self.fft_proj_coeff_01 * velocity_fft[0]
        ])
        velocity_fft_new[:, 0, 0] = velocity_fft[:, 0, 0]
        return fft.irfft2(
            velocity_fft_new, overwrite_input=True, threads=self.n_threads)

    def diffuse_density(self, density):
        density_fft = fft.rfft2(density, threads=self.n_threads)
        density_fft *= self.dens_diff_kernel
        return fft.irfft2(
            density_fft,  overwrite_input=True, threads=self.n_threads)

    def advect(self, field, velocity):
        particle_centers = (
            self.cell_indices - velocity *
            self.dt_cell_size_ratio[:, None, None]
        ).reshape((2, -1))
        lt_ix = np.floor(particle_centers).astype(int)
        rb_ix = lt_ix + 1
        weights = particle_centers - lt_ix
        lt_ix[0] = np.mod(lt_ix[0], self.grid_shape[0])
        rb_ix[0] = np.mod(rb_ix[0], self.grid_shape[0])
        lt_ix[1] = np.mod(lt_ix[1], self.grid_shape[1])
        rb_ix[1] = np.mod(rb_ix[1], self.grid_shape[1])
        return (
            (1 - weights[0]) * (1 - weights[1]) * field[lt_ix[0], lt_ix[1]] +
            (1 - weights[0]) * weights[1] * field[lt_ix[0], rb_ix[1]] +
            weights[0] * (1 - weights[1]) * field[rb_ix[0], lt_ix[1]] +
            weights[0] * weights[1] * field[rb_ix[0], rb_ix[1]]
        ).reshape(self.grid_shape)

    def vorticity_confinement_accel(self, velocity):
        curl = 0.5 * (
            (np.roll(velocity[0], 1, 0) -
             np.roll(velocity[0], -1, 0)) / self.cell_size[0] +
            (np.roll(velocity[1], 1, 1) -
             np.roll(velocity[1], -1, 1)) / self.cell_size[1]
        )
        grad_curl_0 = 0.5 * (
            np.roll(curl, 1, 0) - np.roll(curl, -1, 0)) / self.cell_size[0]
        grad_curl_1 = 0.5 * (
            np.roll(curl, 1, 1) - np.roll(curl, -1, 1)) / self.cell_size[1]
        grad_curl_mag = (grad_curl_0**2 + grad_curl_1**2)**0.5 + 1e-8
        return -self.vort_coeff * curl * np.stack(([
            grad_curl_0 * self.cell_size[0],
            grad_curl_1 * self.cell_size[1]])) / grad_curl_mag

    def update_velocity(self, velocity):
        velocity = (
            velocity + self.dt * self.vorticity_confinement_accel(velocity))
        velocity = np.stack([
            self.advect(velocity[0], velocity),
            self.advect(velocity[1], velocity)
        ])
        return self.project_and_diffuse_velocity(velocity)

    def update_density(self, density, velocity):
        density = density + self.dt * self.density_source
        density = self.advect(density, velocity)
        return self.diffuse_density(density)
