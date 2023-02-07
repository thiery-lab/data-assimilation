"""Linear damped advection model on periodic two-dimensional domain."""

from typing import Optional, Tuple, Callable
import numpy as np
from .base import AbstractDiagonalGaussianModel
from .spatial import SpatiallyExtendedModelMixIn
from .transforms import (
    TwoDimensionalFourierTransformedDiagonalGaussianModelMixIn,
    fft,
    real_array_to_rfft2_coeff,
    rfft2_coeff_to_real_array,
)


class FourierDampedAdvectionModel(AbstractDiagonalGaussianModel):
    """Linear damped advection SPDE model on a periodic 2D spatial domain.

    This model class represents the state field by its the Fourier coefficients rather
    than values of the state field at the spatial mesh points.

    Based on the stochastic model for turbulent signals described in Chapter 5 of [1].

    The governing stochastic partial differential equation (SPDE) is

        dX(s₀,s₁,t) = -(β₀∂₀ + β₁∂₁ + γ) X(s₀,s₁,t) dt + √(2γ) κ(s₀,s₁) ⊛ dW(s₀,s₁,t)

    where `s₀, s₁` are the spatial coordinates on a periodic domain `[0, S₀) × [0, S₁)`,
    `t` the time coordinate, `X(s₀,s₁,t)` the state field process, `β₀` and `β₁` are
    coefficient controlling the degree of linear advection along respectively the `s₀`
    and `s₁` spatial coordinates,  `γ` a coefficient controlling the degree of damping
    in the dynamics, `W(s₀,s₁,t)` a space-time white noise process, `κ(s₀,s₁)` a spatial
    smoothing kernel and `⊛` indicates circular convolution in the spatial coordinates.

    Using a spectral spatial discretisation with a `M₀ × M₁` mesh, this corresponds to a
    linear system of stochastic differential equations (SDEs) in the Fourier
    coefficients `X̃ⱼₖ ∀ j ∈ {0, ..., M₀/2}, k ∈ {0, ..., M₁/2}`

        dX̃ⱼₖ(t) = -(γ + iβ₀ωⱼ + iβ₁ωₖ) X̃ⱼₖ(t) + √(2γ) κ̃ⱼₖ dW̃ⱼₖ(t)

    where `W̃ⱼₖ` is a complex-valued Wiener process, `κ̃ⱼₖ` the (j,k)th Fourier
    coefficient of the smoothing kernel `κ`, `ωⱼ = 2πj / S₀` and `ωₖ = 2πk / S₁` the jth
    and kth spatial frequencies along the two spatial coordinates and `i` the imaginary
    unit.

    This time-homogenous linear system of SDES in the Fourier coefficients can be solved
    exactly, giving a complex normal transition probability

        X̃ⱼₖ(t) | X̃ⱼₖ(0) ~
            ComplexNormal(exp(-(γ + iβ₀ωⱼ + iβ₁ωₖ)t) X̃ⱼₖ(0), (1 - exp(-2γt)) κ̃ⱼₖ²)

    and a stationary distribution

        X̃ⱼₖ(∞) ~ ComplexNormal(0, κ̃ⱼₖ²)

    The smoothing kernel Fourier coefficients are assumed to be

        κ̃ⱼₖ = α² exp(-(ωⱼ² + ωₖ²) ℓ²) ℓ² (M₀M₁) / (S₀S₁)

    where `α` is a parameter controlling the amplitude and `ℓ` a parameter controlling
    the length scale.

    References:

        1. Majda, A. J., & Harlim, J. (2012).
           Filtering complex turbulent systems. Cambridge University Press.
    """

    def __init__(
        self,
        spatial_mesh_shape: Tuple[int, int] = (64, 64),
        time_step: float = 1.0,
        domain_size: Tuple[float, float] = (1.0, 1.0),
        advection_coeff: Tuple[float, float] = (0.0625, 0.0625),
        damping_coeff: float = 0.2231,
        observation_noise_std: float = 0.5,
        state_noise_length_scale: float = 5e-2,
        state_noise_amplitude: float = 1.5625,
        observation_grid_shape: Tuple[int, int] = (16, 16),
        observation_function: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ):
        """
        Args:
            spatial_mesh_shape: Spatial mesh dimensions  `(M₀, M₁)`.
            time_step: Integrator time step.
            domain_size: Spatial domain size `(S₀, S₁).
            advection_coeff: Coefficients `(β₀, β₁)` controlling degree of linear
                advection in dynamics along two spatial dimensions.
            damping_coeff: Coefficient `γ` controlling degree of damping in dynamics.
            state_noise_length_scale: Length scale parameter `ℓ` for smoothing kernel
                used to general initial state and additive noise fields. Larger values
                correspond to smoother fields.
            state_noise_amplitude: Amplitude scale parameter `α` for smoothing kernel
                used to general initial state and additive noise fields. Larger values
                correspond to larger magnitude fields.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_observation,)`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance.
            observation_grid_shape: Dimensions of equispaced rectilinear grid of spatial
                mesh nodes that state is observed at.
            observation_function: Function to apply to subsampled state field to compute
                mean of observation(s) given state(s) at a given time index. Defaults to
                identity function in first argument.
        """
        self.spatial_mesh_shape = spatial_mesh_shape
        self.domain_size = domain_size
        self.time_step = time_step
        self.advection_coeff = advection_coeff
        self.damping_coeff = damping_coeff
        self.state_noise_length_scale = state_noise_length_scale
        self.state_noise_amplitude = state_noise_amplitude
        self.observation_grid_shape = observation_grid_shape
        self.observation_function = observation_function
        self.observation_steps = (
            spatial_mesh_shape[0] // observation_grid_shape[0],
            spatial_mesh_shape[1] // observation_grid_shape[1],
        )
        dim_state = spatial_mesh_shape[0] * spatial_mesh_shape[1]
        dim_observation = observation_grid_shape[0] * observation_grid_shape[1]
        cell_size = np.array(
            [
                domain_size[0] / spatial_mesh_shape[0],
                domain_size[1] / spatial_mesh_shape[1],
            ]
        )
        freq_grid_0 = np.fft.fftfreq(spatial_mesh_shape[0], cell_size[0]) * 2 * np.pi
        freq_grid_1 = np.fft.rfftfreq(spatial_mesh_shape[1], cell_size[1]) * 2 * np.pi
        wavnums_sq = freq_grid_0[:, None] ** 2 + freq_grid_1[None, :] ** 2
        smoothing_kernel = (
            state_noise_amplitude ** 2
            * np.exp(-(state_noise_length_scale ** 2) * wavnums_sq)
            * (
                state_noise_length_scale ** 2
                * (spatial_mesh_shape[0] * spatial_mesh_shape[1])
                / (domain_size[0] * domain_size[1])
            )
        )
        self.state_transition_kernel = np.exp(
            -(
                damping_coeff
                + 1j
                * (
                    freq_grid_0[:, None] * advection_coeff[0]
                    + freq_grid_1[None, :] * advection_coeff[1]
                )
            )
            * time_step
        )
        stationary_var = rfft2_coeff_to_real_array(
            smoothing_kernel + 1j * smoothing_kernel, spatial_mesh_shape, False
        )
        initial_state_std = stationary_var ** 0.5
        state_noise_std = (
            (1 - np.exp(-2 * time_step * damping_coeff)) * stationary_var
        ) ** 0.5
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
            self.state_transition_kernel
            * real_array_to_rfft2_coeff(states, self.spatial_mesh_shape),
            self.spatial_mesh_shape,
        )

    def _observation_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        fields = fft.irfft2(
            real_array_to_rfft2_coeff(states, self.spatial_mesh_shape), norm="ortho"
        )
        subsampled_and_flattened_fields = fields[
            ...,
            self.observation_steps[0] // 2 :: self.observation_steps[0],
            self.observation_steps[1] // 2 :: self.observation_steps[1],
        ].reshape(states.shape[:-1] + (-1,))
        if self.observation_function is None:
            return subsampled_and_flattened_fields
        else:
            return self.observation_function(subsampled_and_flattened_fields, t)


class SpatialDampedAdvectionModel(
    SpatiallyExtendedModelMixIn,
    TwoDimensionalFourierTransformedDiagonalGaussianModelMixIn,
    FourierDampedAdvectionModel,
):
    """Linear damped advection SPDE model on a periodic 2D spatial domain.

    This model class represents the 2D state field by its values at the spatial mesh
    points rather than the corresponding Fourier coefficients. For more details see the
    docstring of `FourierDampedAdvectionModel`.
    """

    def __init__(
        self,
        spatial_mesh_shape: Tuple[int, int] = (64, 64),
        domain_size: Tuple[float, float] = (1.0, 1.0),
        observation_grid_shape: Tuple[int, int] = (16, 16),
        **kwargs
    ):
        """
        Args:
            spatial_mesh_shape: Spatial mesh dimensions  `(M₀, M₁)`.
            time_step: Integrator time step.
            domain_size: Spatial domain size `(S₀, S₁).
            advection_coeff: Coefficients `(β₀, β₁)` controlling degree of linear
                advection in dynamics along two spatial dimensions.
            damping_coeff: Coefficient `γ` controlling degree of damping in dynamics.
            state_noise_length_scale: Length scale parameter `ℓ` for smoothing kernel
                used to general initial state and additive noise fields. Larger values
                correspond to smoother fields.
            state_noise_amplitude: Amplitude scale parameter `α` for smoothing kernel
                used to general initial state and additive noise fields. Larger values
                correspond to larger magnitude fields.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_observation,)`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance.
            observation_grid_shape: Dimensions of equispaced rectilinear grid of spatial
                mesh nodes that state is observed at.
            observation_function: Function to apply to subsampled state field to compute
                mean of observation(s) given state(s) at a given time index. Defaults to
                identity function in first argument.
        """
        observation_steps = (
            spatial_mesh_shape[0] // observation_grid_shape[0],
            spatial_mesh_shape[1] // observation_grid_shape[1],
        )
        observation_space_indices = (
            np.arange(spatial_mesh_shape[0] * spatial_mesh_shape[1])
            .reshape(spatial_mesh_shape)[
                observation_steps[0] // 2 :: observation_steps[0],
                observation_steps[1] // 2 :: observation_steps[1],
            ]
            .flatten()
        )
        super().__init__(
            spatial_mesh_shape=spatial_mesh_shape,
            observation_grid_shape=observation_grid_shape,
            observation_node_indices=observation_space_indices,
            mesh_shape=spatial_mesh_shape,
            domain_extents=domain_size,
            domain_is_periodic=True,
            **kwargs
        )
