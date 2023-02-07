"""Incompressible Navier-Stokes integrator on periodic two-dimensional domain.

Based on methods proposed in

> Stam, Jos. A simple fluid solver based on the FFT.
> Journal of graphics tools 6.2 (2001): 43-52.
>
> Kim, ByungMoon, Yingjie Liu, Ignacio Llamas, and Jarek Rossignac.
> FlowFixer: using BFECC for fluid simulation.
> Proceedings of the First Eurographics conference on Natural Phenomena.
> Eurographics Association, 2005.
"""

from typing import Tuple
import numpy as np

try:
    import pyfftw.interfaces.numpy_fft as fft

    PYFFTW_AVAILABLE = True
except ImportError:
    import numpy.fft as fft

    PYFFTW_AVAILABLE = False
from ..integrators.interpolate import batch_bilinear_interpolate


class FourierNavierStokesIntegrator:
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

    def __init__(
        self,
        mesh_shape: Tuple[int, int],
        domain_size: Tuple[float, float] = (2.0, 2.0),
        time_step: float = 0.05,
        viscous_diffusion_coeff: float = 1e-4,
        max_num_thread: int = 1,
    ):
        """
        Incompressible Navier-Stokes fluid simulation on 2D periodic grid.

        Args:
            mesh_shape: Mesh dimensions as a 2-tuple `(dim_0, dim_1)`.
            domain_size: Spatial domain size a 2-tuple `(size_0, size_1)`.
            time_step: Integrator time-step.
            viscous_diffusion_coeff: Velocity viscous diffusion coefficient.
            max_num_thread: Maximum number of threads to use for FFT and
                interpolation operations.
        """
        self.mesh_shape = mesh_shape
        self.domain_size = domain_size
        self.dim_state = mesh_shape[0] * mesh_shape[1]
        self.viscous_diffusion_coeff = viscous_diffusion_coeff
        self.time_step = time_step
        self.max_num_thread = max_num_thread
        # Calculate spatial size of each cell in mesh.
        self.mesh_cell_size = np.array(
            [
                self.domain_size[0] / self.mesh_shape[0],
                self.domain_size[1] / self.mesh_shape[1],
            ]
        )
        # Coordinate indices of mesh cell corners.
        self.mesh_corner_indices = np.array(
            np.meshgrid(
                np.arange(self.mesh_shape[0]),
                np.arange(self.mesh_shape[1]),
                indexing="ij",
            )
        )
        # Spatial angular frequency values for rfft2 frequency grid layout i.e.
        # FFT along axis 0 and RFFT along axis 1
        # Always use numpy.fft module here as pyfftw interface does not provide
        # fftfreq functions
        freq_grid_0 = np.fft.fftfreq(mesh_shape[0], self.mesh_cell_size[0]) * 2 * np.pi
        freq_grid_1 = np.fft.rfftfreq(mesh_shape[1], self.mesh_cell_size[1]) * 2 * np.pi
        # Squared wavenumbers
        self.wavnums_sq = freq_grid_0[:, None] ** 2 + freq_grid_1[None, :] ** 2
        # Kernel in frequency space to simulate viscous diffusion term.
        # Corresponds to solving diffusion equation in 2D exactly in time with
        # spectral method to approximate second-order spatial derivatives.
        self.viscous_diffusion_kernel = np.exp(
            -viscous_diffusion_coeff * time_step * self.wavnums_sq
        )
        # For first derivative expressions zero Nyquist frequency for even
        # number of grid points:
        # > Notes on FFT-based differentiation.
        # > Steven G. Johnson, MIT Applied Mathematics.
        # > http://math.mit.edu/~stevenj/fft-deriv.pdf
        grad_0_kernel = freq_grid_0 * 1j
        grad_1_kernel = freq_grid_1 * 1j
        if mesh_shape[0] % 2 == 0:
            grad_0_kernel[mesh_shape[0] // 2] = 0
        if mesh_shape[1] % 2 == 0:
            grad_1_kernel[mesh_shape[1] // 2] = 0
        # Clip zero wave number square values to small positive constant to
        # avoid divide by zero warnings.
        wavnums_sq_clip = np.maximum(self.wavnums_sq, 1e-8)
        # Coefficients of vector field frequency components to solve Poisson's
        # equation to project to divergence-free field.
        self.fft_vel_coeff_0 = grad_1_kernel[None, :] / wavnums_sq_clip
        self.fft_vel_coeff_1 = -grad_0_kernel[:, None] / wavnums_sq_clip

    def rfft2(self, field):
        """Convenience wrapper for real-valued 2D FFT."""
        if PYFFTW_AVAILABLE:
            num_thread = min(field.shape[0], self.max_num_thread)
            return fft.rfft2(field, norm="ortho", threads=num_thread)
        else:
            return fft.rfft2(field, norm="ortho")

    def irfft2(self, field):
        """Convenience wrapper for inverse real-valued 2D FFT."""
        if PYFFTW_AVAILABLE:
            num_thread = min(field.shape[0], self.max_num_thread)
            return fft.irfft2(field, norm="ortho", threads=num_thread)
        else:
            return fft.irfft2(field, norm="ortho")

    def velocity_from_fft_vorticity(self, fft_vorticity):
        """Compute velocity vector field from FFT of vorticity scalar field."""
        # Solve for velocity field in terms of vorticity in frequency space.
        fft_velocity = np.stack(
            [
                self.fft_vel_coeff_0 * fft_vorticity,
                self.fft_vel_coeff_1 * fft_vorticity,
            ],
            axis=-3,
        )
        # Perform inverse 2D real-valued FFT to map back to spatial fields.
        return self.irfft2(fft_velocity)

    def semi_lagrangian_advect(self, field, velocity):
        """Use semi-Lagrangian method to advect a given field a single step."""
        # Set number of threads to parallelise interpolation across
        # conservatively such that never more than number of independent fields
        num_thread = min(field.shape[0], self.max_num_thread)
        # Estimate mesh coordinates of particles which end up at mesh corner
        # points when following current velocity field.
        particle_centers = self.mesh_corner_indices[None] - (
            velocity * self.time_step / self.mesh_cell_size[None, :, None, None]
        )
        # Calculate advected field values by bilinearly interpolating field
        # values at back traced particle locations.
        return batch_bilinear_interpolate(field, particle_centers, num_thread)

    def bfecc_advect(self, field, velocity):
        """Use BFECC method to advect a given field a single step."""
        # Initial forwards step
        field_1 = self.semi_lagrangian_advect(field, velocity)
        # Backwards step from field_1
        field_2 = self.semi_lagrangian_advect(field_1, -velocity)
        # Compute error corrected original field
        field_3 = (3 * field - field_2) / 2.0
        # Final forwards step
        return self.semi_lagrangian_advect(field_3, velocity)

    def step(self, fft_vorticity):
        """Perform single time step update of FFT of vorticity field."""
        # Diffuse vorticity in spectral domain
        fft_vorticity *= self.viscous_diffusion_kernel
        # Calculate velocity and vorticity in spatial domain
        velocity = self.velocity_from_fft_vorticity(fft_vorticity)
        vorticity = self.irfft2(fft_vorticity)
        # Advect vorticity
        vorticity = self.bfecc_advect(vorticity, velocity)
        return self.rfft2(vorticity)
