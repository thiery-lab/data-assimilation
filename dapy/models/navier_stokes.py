"""Incompressible Navier-Stokes fluid simulation on periodic two-dimensional domain."""

from typing import Optional, Tuple, Callable
import numpy as np
from dapy.models.base import AbstractDiagonalGaussianModel
from dapy.integrators.navier_stokes import FourierNavierStokesIntegrator
from dapy.models.transforms import (
    TwoDimensionalFourierTransformedDiagonalGaussianModelMixIn,
    fft,
    real_array_to_rfft2_coeff,
    rfft2_coeff_to_real_array,
)


class FourierIncompressibleFluidModel(AbstractDiagonalGaussianModel):
    """Incompressible Navier-Stokes fluid simulation on two-dimensional periodic domain.

    This model class represents the 2D state field by its the Fourier coefficients
    rather than values of the state field at the spatial mesh points.

    Simulates evolution of a fluid velocity field on a 2-torus using a Fourier spectral
    based implementation [1] of the incompressible Navier-Stokes equations in
    two-dimensions. The divergence-free velocity field is parameterised by its curl -
    the voriticity. A semi-Lagrangian method is used for the advection updates with
    backwards-forwards error correction for improved accuracy [2].

    References:
      1. Stam, Jos. A simple fluid solver based on the FFT.
         Journal of graphics tools 6.2 (2001): 43-52.
      2. Kim, ByungMoon, Yingjie Liu, Ignacio Llamas, and Jarek Rossignac.
         FlowFixer: using BFECC for fluid simulation. Proceedings of the First
         Eurographics conference on Natural Phenomena. Eurographics
         Association, 2005.
    """

    def __init__(
        self,
        mesh_shape: Tuple[int, int] = (128, 128),
        time_step: float = 0.25,
        domain_size: Tuple[float, float] = (5.0, 5.0),
        viscous_diffusion_coeff: float = 1e-4,
        observation_noise_std: float = 1e-1,
        state_noise_length_scale: float = 2e-3,
        initial_state_amplitude: float = 5e-2,
        state_noise_amplitude: float = 4e-3,
        observation_subsample: int = 4,
        observe_speed: bool = False,
        observation_function: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        max_num_thread: int = 4,
    ):
        """
        Args:
            mesh_shape: Mesh dimensions as a 2-tuple `(dim_0, dim_1)`.
            time_step: Integrator time-step.
            domain_size: Spatial domain size a 2-tuple `(size_0, size_1)`.
            viscous_diffusion_coeff: Velocity viscous diffusion coefficient.
            state_noise_length_scale: Length scale parameter for random noise used to
                generate initial vorticity and vorticity additive noise fields. Larger
                values correspond to smoother fields.
            initial_state_amplitude: Amplitude scale parameter for initial random
                vorticity field. Larger values correspond to larger magnitude
                vorticities (and so velocities).
            state_noise_amplitude: Amplitude scale parameter for additive vorticity
                noise in model dynamics. Larger values correspond to larger magnitude
                additive noise in the vorticity field.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_observation,)`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance.
            observation_subsample: Factor to subsample each spatial dimension by
                in observation operator.
            observe_speed: Whether to observe speed (magnitude of velocity) field
                instead of vorticity.
            observation_function: Function to apply to subsampled vorticity / speed
                field to compute mean of observation(s) given state(s) at a given time
                index. Defaults to identity function in first argument.
            max_num_thread: Maximum number of threads to use for FFT and interpolation
                operations.
        """
        self.mesh_shape = mesh_shape
        assert mesh_shape[0] % 2 == 0 and mesh_shape[1] % 2 == 0, (
            "Mesh dimensions must both be even")
        dim_state = mesh_shape[0] * mesh_shape[1]
        dim_observation = mesh_shape[0] * mesh_shape[1] // observation_subsample ** 2
        self.integrator = FourierNavierStokesIntegrator(
            mesh_shape=mesh_shape,
            domain_size=domain_size,
            time_step=time_step,
            viscous_diffusion_coeff=viscous_diffusion_coeff,
            max_num_thread=max_num_thread,
        )
        smoothing_kernel = (
            np.exp(-state_noise_length_scale * self.integrator.wavnums_sq)
            * ((mesh_shape[0] * mesh_shape[1]) / (domain_size[0] * domain_size[1]))
            ** 0.5
        )
        state_noise_kernel = time_step ** 0.5 * state_noise_amplitude * smoothing_kernel
        initial_state_kernel = initial_state_amplitude * smoothing_kernel
        state_noise_std = rfft2_coeff_to_real_array(
            state_noise_kernel + 1j * state_noise_kernel, False
        )
        initial_state_std = rfft2_coeff_to_real_array(
            initial_state_kernel + 1j * initial_state_kernel, False
        )
        self.observation_subsample = observation_subsample
        self.state_noise_length_scale = state_noise_length_scale
        self.initial_state_amplitude = initial_state_amplitude
        self.state_noise_amplitude = state_noise_amplitude
        self.observe_speed = observe_speed
        self.observation_function = observation_function
        super().__init__(
            dim_state=dim_state,
            dim_observation=dim_observation,
            initial_state_std=initial_state_std,
            initial_state_mean=np.zeros(dim_state),
            state_noise_std=state_noise_std,
            observation_noise_std=observation_noise_std,
        )

    def _next_state_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        return rfft2_coeff_to_real_array(
            self.integrator.step(real_array_to_rfft2_coeff(states, self.mesh_shape))
        )

    def _observation_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        fft_vorticity_fields = real_array_to_rfft2_coeff(states, self.mesh_shape)
        if self.observe_speed:
            velocity_fields = self.integrator.velocity_from_fft_vorticity(
                fft_vorticity_fields
            )
            fields = (velocity_fields ** 2).sum(-3) ** 0.5
        else:
            fields = fft.irfft2(fft_vorticity_fields, norm="ortho")
        subsampled_and_flattened_fields = fields[
            ..., :: self.observation_subsample, :: self.observation_subsample
        ].reshape(states.shape[:-1] + (-1,))
        if self.observation_function is None:
            return subsampled_and_flattened_fields
        else:
            return self.observation_function(subsampled_and_flattened_fields)


class SpatialIncompressibleFluidModel(
    TwoDimensionalFourierTransformedDiagonalGaussianModelMixIn,
    FourierIncompressibleFluidModel,
):
    """Incompressible Navier-Stokes fluid simulation on two-dimensional periodic domain.

    This model class represents the 2D state field by its values at the spatial mesh
    points rather than the corresponding Fourier coefficients. For more details see the
    docstring of `FourierIncompressibleFluidModel`.
    """
