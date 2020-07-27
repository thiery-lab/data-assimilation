"""Linear SPDE model on a periodic 1D spatial domain for turbulent signals.

Based on the stochastic model for turbulent signals described in Chapter 5 of [1].

References:

    1. Majda, A. J., & Harlim, J. (2012).
        Filtering complex turbulent systems. Cambridge University Press.
"""

from typing import Union, Sequence
import numpy as np

from dapy.models.base import AbstractDiagonalGaussianModel, AbstractLinearModel
from dapy.models.spatial import SpatiallyExtendedModelMixIn
from dapy.models.transforms import (
    OneDimensionalFourierTransformedDiagonalGaussianModelMixIn,
    fft,
    real_array_to_rfft_coeff,
    rfft_coeff_to_real_array,
)


class FourierStochasticTurbulenceModel(
    AbstractDiagonalGaussianModel, AbstractLinearModel
):
    """Linear SPDE model on a periodic 1D spatial domain for turbulent signals.

    This model class represents the state field by its the Fourier coefficients rather
    than values of the state field at the spatial mesh points.

    Based on the stochastic model for turbulent signals described in Chapter 5 of [1].

    The governing stochastic partial differential equation (SPDE) is

        dX = (α * ∂²X/∂s² + β * ∂X/∂s - γ * X) dt + κ ⊛ dW

    where `s` is the spatial coordinate in a periodic domain `[0, S)`, `t` the time
    coordinate, `X(s, t)` the state field process, `α` a coefficient controlling the
    degree of diffusion in the dynamics, `β` a coefficient controlling the degree of
    linear advection in the dynamics,  `γ` a coefficient controlling the degree of
    damping in the dynamics, `W(s, t)` a space-time white noise process, `κ(s)` a
    spatial smoothing kernel and `⊛` indicates circular convolution in the spatial
    coordinate.

    Using a spectral spatial discretisation with `M` mesh points, this corresponds to a
    linear system of stochastic differential equations (SDEs) in the Fourier
    coefficients X̃ₖ

        dX̃ₖ = (-α * ωₖ² - γ + i * β * ωₖ) * X̃ₖ + κ̃ₖ * dW̃ₖ  ∀ k ∈ {0, ..., M/2}

    where `W̃ₖ` is a complex-valued Wiener process, `κ̃ₖ` the kth Fourier coefficient of
    the smoothing kernel `κ`, `ωₖ = 2 * π * k / S` the kth spatial frequency and `i`
    the imaginary unit.

    This time-homogenous linear system of SDES in the Fourier coefficients can be solved
    exactly, giving a complex normal transition probability

        X̃ₖ(t) | X̃ₖ(0) ~ ComplexNormal(exp((-ψₖ + i * β * ωₖ) * t) * X̃ₖ(0),
                                      (1 - exp(-2 * ψₖ * t)) * κ̃ₖ² / (2 * ψₖ))

    where `ψₖ = α * ωₖ² + γ`, and a stationary distribution

        X̃ₖ(∞) ~ ComplexNormal(0, κ̃ₖ² / (2 * ψₖ))

    The smoothing kernel Fourier coefficients are assumed to be

        κ̃ₖ = σ * exp(-ωₖ² * ℓ²) * √(M / S)

    where `σ` is a parameter controlling the amplitude and `ℓ` a parameter controlling
    the length scale.

    References:

        1. Majda, A. J., & Harlim, J. (2012).
           Filtering complex turbulent systems. Cambridge University Press.
    """

    def __init__(
        self,
        dim_state: int = 512,
        observation_space_indices: Union[slice, Sequence[int]] = slice(4, None, 8),
        time_step: float = 0.25,
        domain_extent: float = 1.0,
        damping_coeff: float = 0.1,
        advection_coeff: float = 0.1,
        diffusion_coeff: float = 4e-5,
        observation_noise_std: float = 0.5,
        state_noise_amplitude: float = 0.1,
        state_noise_length_scale: float = 4e-3,
        **kwargs
    ):
        """
        Args:
            dim_state: Dimension of state which is equivalent here to number of mesh
                points in spatial discretization.
            observation_space_indices: Slice or sequence of integers to be used to
                subsample spatial dimension of state when computing observations.
            time_step: Integrator time step.
            domain_extent: Extent (size) of spatial domain.
            advection_coeff: Coefficient (`α` in description above) controlling degree
                of advection in dynamics.
            diffusion_coeff: Coefficient (`β` in description above) controlling degree
                of diffusion in dynamics.
            damping_coeff: Coefficient (`γ` in description above) controlling degree of
                damping in dynamics.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_observation,)`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance.
            state_noise_amplitude: Amplitude scale parameter for additive state noise
                in model dynamics. Larger values correspond to larger magnitude
                additive noise in the state field.
            state_noise_length_scale: Length scale parameter for smoothed noise used to
                generate initial state and additive state noise fields. Larger values
                correspond to smoother fields.
        """
        assert dim_state % 2 == 0, "State dimension `dim_state` must be even"
        self.observation_space_indices = observation_space_indices
        spatial_freqs = np.arange(dim_state // 2 + 1) * 2 * np.pi / domain_extent
        spatial_freqs_sq = spatial_freqs ** 2
        spatial_freqs[dim_state // 2] = 0
        smoothing_kernel = (
            state_noise_amplitude
            * np.exp(-spatial_freqs_sq * state_noise_length_scale ** 2)
            * (dim_state / domain_extent) ** 0.5
        )
        psi = diffusion_coeff * spatial_freqs_sq + damping_coeff
        self.state_transition_kernel = np.exp(
            (-psi + 1j * advection_coeff * spatial_freqs) * time_step
        )
        state_noise_kernel = (
            smoothing_kernel
            * (1.0 - np.exp(-2 * psi * time_step)) ** 0.5
            / (2 * psi) ** 0.5
        )
        state_noise_std = np.concatenate([state_noise_kernel, state_noise_kernel[1:-1]])
        initial_state_kernel = smoothing_kernel / (2 * psi) ** 0.5
        initial_state_std = np.concatenate(
            [initial_state_kernel, initial_state_kernel[1:-1]]
        )
        dim_observation = len(np.arange(dim_state)[observation_space_indices])
        super().__init__(
            dim_state=dim_state,
            dim_observation=dim_observation,
            initial_state_std=initial_state_std,
            initial_state_mean=np.zeros(dim_state),
            state_noise_std=state_noise_std,
            observation_noise_std=observation_noise_std,
            **kwargs
        )

    def _next_state_mean(self, states, t):
        return rfft_coeff_to_real_array(
            self.state_transition_kernel * real_array_to_rfft_coeff(states)
        )

    def _observation_mean(self, states, t):
        return fft.irfft(real_array_to_rfft_coeff(states), norm="ortho")[
            ..., self.observation_space_indices
        ]


class SpatialStochasticTurbulenceModel(
    SpatiallyExtendedModelMixIn,
    OneDimensionalFourierTransformedDiagonalGaussianModelMixIn,
    FourierStochasticTurbulenceModel,
):
    """Linear SPDE model on a periodic 1D spatial domain for turbulent signals.

    This model class represents the state field by its values at the spatial mesh points
    rather than the corresponding Fourier coefficients. For more details see the
    docstring of `FourierStochasticTurbulenceModel`.
    """

    def __init__(
        self,
        dim_state: int = 512,
        observation_space_indices: Union[slice, Sequence[int]] = slice(4, None, 8),
        time_step: float = 0.25,
        domain_extent: float = 1.0,
        damping_coeff: float = 0.1,
        advection_coeff: float = 0.1,
        diffusion_coeff: float = 4e-5,
        observation_noise_std: float = 0.5,
        state_noise_amplitude: float = 0.1,
        state_noise_length_scale: float = 4e-3,
        **kwargs
    ):
        """
        Args:
            dim_state: Dimension of state which is equivalent here to number of mesh
                points in spatial discretization.
            observation_space_indices: Slice or sequence of integers to be used to
                subsample spatial dimension of state when computing observations.
            time_step: Integrator time step.
            domain_extent: Extent (size) of spatial domain.
            advection_coeff: Coefficient (`α` in description above) controlling degree
                of advection in dynamics.
            diffusion_coeff: Coefficient (`β` in description above) controlling degree
                of diffusion in dynamics.
            damping_coeff: Coefficient (`γ` in description above) controlling degree of
                damping in dynamics.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_observation,)`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance.
            state_noise_amplitude: Amplitude scale parameter for additive state noise
                in model dynamics. Larger values correspond to larger magnitude
                additive noise in the state field.
            state_noise_length_scale: Length scale parameter for smoothed noise used to
                generate initial state and additive state noise fields. Larger values
                correspond to smoother fields.
        """
        super().__init__(
            dim_state=dim_state,
            observation_space_indices=observation_space_indices,
            time_step=time_step,
            domain_extent=domain_extent,
            advection_coeff=advection_coeff,
            diffusion_coeff=diffusion_coeff,
            damping_coeff=damping_coeff,
            observation_noise_std=observation_noise_std,
            state_noise_amplitude=state_noise_amplitude,
            state_noise_length_scale=state_noise_length_scale,
            mesh_shape=(dim_state,),
            domain_extents=(domain_extent,),
            domain_is_periodic=True,
            observation_node_indices=observation_space_indices,
        )
