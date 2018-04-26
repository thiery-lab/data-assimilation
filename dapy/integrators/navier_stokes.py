"""Simple Navier-Stokes fluid simulation on periodic two-dimensional grid.

Based on methods proposed in

> Stam, Jos. A simple fluid solver based on the FFT.
> Journal of graphics tools 6.2 (2001): 43-52.
>
> Kim, ByungMoon, Yingjie Liu, Ignacio Llamas, and Jarek Rossignac.
> FlowFixer: using BFECC for fluid simulation.
> Proceedings of the First Eurographics conference on Natural Phenomena.
> Eurographics Association, 2005.
"""

import math
import numpy as np
try:
    import pyfftw.interfaces.numpy_fft as fft
    PYFFTW_AVAILABLE = True
except ImportError:
    import numpy.fft as fft
    PYFFTW_AVAILABLE = False
from dapy.integrators.interpolate import batch_bilinear_interpolate


class NavierStokes2dIntegrator:
    """Incompressible Navier-Stokes fluid simulation on 2D periodic grid.

    Simulates evolution of an incompressible fluid velocity field on a 2-torus
    using a finite difference based implementation of the incompressible
    Navier-Stokes equations in two-dimensions. To enforce the
    incompressibility condition the velocity field is parameterised as a
    scalar vorticity field corresponding to the curl of the velocity field. A
    semi-Lagrangian method is used for the advection updates and a FFT-based
    method used to implement the viscous diffusion.

    A second-order accurate BFECC (back and forward error correction and
    compensation) method is used to simulate the advection steps [1] which
    decreases numerical dissipation of voriticity.

    References:
      1. Kim, ByungMoon, Yingjie Liu, Ignacio Llamas, and Jarek Rossignac.
         FlowFixer: using BFECC for fluid simulation. Proceedings of the First
         Eurographics conference on Natural Phenomena. Eurographics
         Association, 2005.
    """

    def __init__(self, grid_shape, grid_size=(2., 2.), dt=0.05,
                 visc_diff_coeff=1e-4, max_n_thread=1):
        """
        Incompressible Navier-Stokes fluid simulation on 2D periodic grid.

        Args:
            grid_shape (tuple): Grid dimensions as a 2-tuple.
            grid_size (tuple): Spatial extent of simulation grid as a 2-tuple.
            dt (float): Integrator time-step.
            visc_diff_coeff (float): Velocity viscous diffusion coefficient.
            max_n_thread (int): Maximum number of threads to use for FFT and
                interpolation operations.
        """
        self.grid_shape = grid_shape
        self.grid_size = grid_size
        self.n_grid = grid_shape[0] * grid_shape[1]
        self.visc_diff_coeff = visc_diff_coeff
        self.dt = dt
        self.max_n_thread = max_n_thread
        # Calculate spatial size of each cell in grid.
        self.cell_size = np.array([
            self.grid_size[0] / float(self.grid_shape[0]),
            self.grid_size[1] / float(self.grid_shape[1])
        ])
        # Coordinate indices of grid cell corners.
        self.cell_indices = np.array(np.meshgrid(
                np.arange(self.grid_shape[0]),
                np.arange(self.grid_shape[1]), indexing='ij'))
        # Spatial angular frequency values for real-valued FFT grid layout.
        freq_grid_0 = np.fft.fftfreq(
                grid_shape[0], grid_size[0] / grid_shape[0]) * 2 * np.pi
        freq_grid_1 = np.fft.rfftfreq(
                grid_shape[1], grid_size[1] / grid_shape[1]) * 2 * np.pi
        # Squared wavenumbers on FFT grid.
        self.wavnum_sq_grid = freq_grid_0[:, None]**2 + freq_grid_1[None, :]**2
        # Kernel in frequency space to simulate viscous diffusion term.
        # Corresponds to solving diffusion equation in 2D exactly in time with
        # spectral method to approximate second-order spatial derivatives.
        self.visc_diff_kernel = np.exp(
            -visc_diff_coeff * dt * self.wavnum_sq_grid)
        # For first derivative expressions zero Nyquist frequency for even
        # number of grid points:
        # > Notes on FFT-based differentiation.
        # > Steven G. Johnson, MIT Applied Mathematics.
        # > http://math.mit.edu/~stevenj/fft-deriv.pdf
        grad_0_kernel = freq_grid_0 * 1j
        grad_1_kernel = freq_grid_1 * 1j
        if grid_shape[0] % 2 == 0:
            grad_0_kernel[grid_shape[0] // 2] = 0
        if grid_shape[1] % 2 == 0:
            grad_1_kernel[grid_shape[1] // 2] = 0
        # Clip zero wave number square values to small positive constant to
        # avoid divide by zero warnings.
        wavnum_sq_grid_clip = np.maximum(self.wavnum_sq_grid, 1e-8)
        # Coefficients of vector field frequency components to solve Poisson's
        # equation to project to divergence-free field.
        self.fft_vel_coeff_0 = grad_1_kernel[None, :] / wavnum_sq_grid_clip
        self.fft_vel_coeff_1 = -grad_0_kernel[:, None] / wavnum_sq_grid_clip

    def rfft2(self, field):
        """Convenience wrapper for real-valued 2D FFT."""
        if PYFFTW_AVAILABLE:
            n_thread = min(field.shape[0], self.max_n_thread)
            return fft.rfft2(field, threads=n_thread)
        else:
            return fft.rfft2(field)

    def irfft2(self, x):
        """Convenience wrapper for inverse real-valued 2D FFT."""
        if PYFFTW_AVAILABLE:
            n_thread = min(x.shape[0], self.max_n_thread)
            return fft.irfft2(x, threads=n_thread)
        else:
            return fft.irfft2(field)

    def velocity_from_vorticity(self, vorticity=None, vorticity_fft=None):
        """Compute velocity vector field from vorticity scalar field."""
        if vorticity_fft is None:
            if vorticity is None:
                raise ValueError(
                    'One of voriticity or voriticity_fft must be provided.')
            vorticity_fft = self.rfft2(vorticity)
        # Solve for velocity field in terms of vorticity in frequency space.
        velocity_fft = np.stack([
            self.fft_vel_coeff_0 * vorticity_fft,
            self.fft_vel_coeff_1 * vorticity_fft
        ], axis=-3)
        # Perform inverse 2D real-valued FFT to map back to spatial fields.
        return self.irfft2(velocity_fft)

    def semi_lagrangian_advect(self, field, velocity):
        """Use semi-Lagrangian method to advect a given field a single step."""
        # Set number of threads to parallelise interpolation across
        # conservatively such that never more than number of independent fields
        n_thread = min(field.shape[0], self.max_n_thread)
        # Estimate grid coordinates of particles which end up at grid corners
        # points when following current velocity field.
        particle_centers = self.cell_indices[None] - (
            velocity * self.dt / self.cell_size[None, :, None, None])
        # Calculate advected field values by bilinearly interpolating field
        # values at back traced particle locations.
        return batch_bilinear_interpolate(field, particle_centers, n_thread)

    def bfecc_advect(self, field, velocity):
        """Use BFECC method to advect a given field a single step."""
        # Initial forwards step
        field_1 = self.semi_lagrangian_advect(field, velocity)
        # Backwards step from field_1
        field_2 = self.semi_lagrangian_advect(field_1, -velocity)
        # Compute error corrected original field
        field_3 = (3 * field - field_2) / 2.
        # Final forwards step
        return self.semi_lagrangian_advect(field_3, velocity)

    def update_vorticity(self, vorticity):
        """Perform single time step update of vorticity field."""
        vorticity_fft = self.rfft2(vorticity)
        # Diffuse vorticity in spectral domain
        vorticity_fft *= self.visc_diff_kernel
        # Calculate velocity and vorticity in spatial domain
        velocity = self.velocity_from_vorticity(vorticity_fft=vorticity_fft)
        vorticity = self.irfft2(vorticity_fft)
        # Advect vorticity
        vorticity = self.bfecc_advect(vorticity, velocity)
        return vorticity

    def forward_integrate(self, z_curr, start_time_index, n_step=1):
        """Forward integrate system state one or more time steps.

        State vectors are flattened 2D vorticity fields.

        Args:
            z_curr (array): Two dimensional array of current system states of
                shape `(n_state, dim_z)` where `n_state` is the number of
                independent system states being simulated and `dim_z` is the
                state dimension.
            z_next (array): Empty two dimensional array to write forward
                integrated states two. Should be of same shape as `z_curr`.
            start_time_index (int): Integer indicating current time index
                associated with the `z_curr` states (i.e. number of previous
                `forward_integrate` calls) to allow for calculate of time for
                non-homogeneous systems.
            n_step (int): Number of integrator time steps to perform.
        """
        n_batch = z_curr.shape[0]
        # Reshape state vector to vorticity field.
        vorticity = z_curr.reshape((n_batch,) + self.grid_shape)
        for s in range(n_step):
            # Update vorticity field by a single time step.
            vorticity = self.update_vorticity(vorticity)
        # Return flattened voriticity field as new state vector.
        return vorticity.reshape((n_batch, -1))


class StochasticNavierStokes2dIntegrator(NavierStokes2dIntegrator):

    def __init__(self, rng, grid_shape, grid_size=(2., 2.), dt=0.05,
                 visc_diff_coeff=1e-4, vort_noise_length_scale=1e-2,
                 vort_noise_ampl_scale=1e-2, max_n_thread=2):
        """
        Stochastic incompr. Navier-Stokes fluid simulation on 2D periodic grid.

        Additive noise scaled by square root of timestep added to vorticity
        field at each time step.

        Args:
            rng (RandomState): Seeded random number generator object.
            grid_shape (tuple): Grid dimensions as a 2-tuple.
            grid_size (tuple): Spatial extent of simulation grid as a 2-tuple.
            dt (float): Integrator time-step.
            visc_diff_coeff (float): Velocity viscous diffusion coefficient.
            vort_noise_length_scale (float): Length scale parameter for random
                noise used to generate vorticity additive noise fields. Larger
                values correspond to smoother fields.
            vort_noise_ampl_scale (float): Amplitude scale parameter for
                additive vorticity noise in model dynamics. Larger values
                correspond to larger magnitude additive noise in the vorticity
                field.
            max_n_thread (int): Maximum number of threads to use for FFT and
                interpolation operations.
        """
        super(StochasticNavierStokes2dIntegrator, self).__init__(
            grid_shape=grid_shape, grid_size=grid_size, dt=dt,
            visc_diff_coeff=visc_diff_coeff, max_n_thread=max_n_thread)
        self.rng = rng
        self.vort_noise_length_scale = vort_noise_ampl_scale
        self.vort_noise_ampl_scale = vort_noise_ampl_scale
        self.vort_noise_kernel = np.exp(
            -vort_noise_length_scale * self.wavnum_sq_grid) / (
                self.cell_size[0] * self.cell_size[1])**0.5

    def update_vorticity(self, vorticity):
        """Perform single time step update of vorticity field."""
        vorticity_fft = self.rfft2(vorticity)
        # Add noise in spectral domain
        vorticity_fft += (
            self.dt**0.5 * self.vort_noise_ampl_scale *
            self.vort_noise_kernel *
            self.rfft2(self.rng.normal(size=vorticity.shape)))
        # Diffuse vorticity in spectral domain
        vorticity_fft *= self.visc_diff_kernel
        # Calculate velocity and vorticity in spatial domain
        velocity = self.velocity_from_vorticity(vorticity_fft=vorticity_fft)
        vorticity = self.irfft2(vorticity_fft)
        # Advect vorticity
        vorticity = self.bfecc_advect(vorticity, velocity)
        return vorticity
