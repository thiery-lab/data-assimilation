"""Non-linear SPDE model on a periodic 1D spatial domain for laminar wave fronts.

Based on the Kuramato--Sivashinsky PDE model [1, 2] which exhibits spatio-temporally
chaotic dynamics.

References:

    1. Kuramoto and Tsuzuki. Persistent propagation of concentration waves
       in dissipative media far from thermal equilibrium.
       Progress in Theoretical Physcs, 55 (1976) pp. 356–369.
    2. Sivashinsky. Nonlinear analysis of hydrodynamic instability in laminar
       flames I. Derivation of basic equations.
       Acta Astronomica, 4 (1977) pp. 1177–1206.
"""

from typing import Union, Optional, Sequence, Callable
import numpy as np
from dapy.models.base import AbstractDiagonalGaussianModel
from dapy.models.spatial import SpatiallyExtendedModelMixIn
from dapy.integrators.etdrk4 import FourierETDRK4Integrator
from dapy.models.transforms import (
    OneDimensionalFourierTransformedDiagonalGaussianModelMixIn,
    fft,
    real_array_to_rfft_coeff,
    rfft_coeff_to_real_array,
)


class FourierLaminarFlameModel(AbstractDiagonalGaussianModel):
    """Non-linear SPDE model on a periodic 1D spatial domain for laminar flame fronts.

    This model class represents the state field by its the Fourier coefficients rather
    than values of the state field at the spatial mesh points.

    Based on the Kuramato--Sivashinsky PDE model [1, 2] which exhibits spatio-temporally
    chaotic dynamics.

    The governing stochastic partial differential equation (SPDE) is

        dX = -(∂⁴X/∂s⁴ + ∂²X/∂s² + X * ∂X/∂s + γ * X) dt + κ ⊛ dW

    where `s` is the spatial coordinate in a periodic domain `[0, S)`, `t` the time
    coordinate, `X(s, t)` the state field process, `γ` a coefficient controlling the
    degree of damping in the dynamics, `W(s, t)` a space-time white noise process,
    `κ(s)` a spatial smoothing kernel and `⊛` indicates circular convolution in the
    spatial coordinate.

    Using a spectral spatial discretisation, this corresponds to a non-linear system of
    stochastic differential equations (SDEs) in the Fourier coefficients X̃ₖ

        dX̃ₖ = (ωₖ² - ωₖ⁴ - γ) * X̃ₖ + (i * ωₖ / 2) * DFTₖ(IDFT(X̃)²) + κ̃ₖ * dW̃ₖ

    where `W̃ₖ` is a complex-valued Wiener process, `κ̃ₖ` the kth Fourier coefficient of
    the smoothing kernel `κ`, `ωₖ = 2 * pi * k / S` the kth spatial frequency and `i`
    the imaginary unit.

    A Fourier-domain exponential time-differencing integrator with 4th order Runge--
    Kutta updates for non-linear terms [3, 4] is used to integrate the deterministic
    component of the SDE dynamics and an Euler-Maruyama discretisation used for the
    Wiener process increment.

    The smoothing kernel Fourier coefficients are assumed to be

        κ̃ₖ = σ * exp(-ωₖ² * ℓ²) * √(M / S)

    where `σ` is a parameter controlling the amplitude and `ℓ` a parameter controlling
    the length scale.

    References:

        1. Kuramoto and Tsuzuki. Persistent propagation of concentration waves
           in dissipative media far from thermal equilibrium.
           Progress in Theoretical Physcs, 55 (1976) pp. 356–369.
        2. Sivashinsky. Nonlinear analysis of hydrodynamic instability in laminar
           flames I. Derivation of basic equations. Acta Astronomica, 4 (1977)
           pp. 1177–1206.
        3. Kassam, Aly-Khan and Trefethen, Lloyd N.
            Fourth-order time-stepping for stiff PDEs.
            SIAM Journal on Scientific Computing 26.4 (2005): 1214-1233.
        4. Cox, Steven M. and Matthews, Paul C.
            Exponential time differencing for stiff systems.
            Journal of Computational Physics 176.2 (2002): 430-455.
    """

    def __init__(
        self,
        dim_state: int = 512,
        observation_space_indices: Union[slice, Sequence[int]] = slice(4, None, 8),
        observation_function: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        time_step: float = 0.25,
        domain_extent: float = 32 * np.pi,
        damping_coeff: float = 1.0 / 6,
        observation_noise_std: float = 0.5,
        initial_state_amplitude: float = 1.0,
        state_noise_amplitude: float = 1.0,
        state_noise_length_scale: float = 1.0,
        num_roots_of_unity_etdrk4_integrator: int = 16,
        **kwargs
    ):
        """
        Args:
            dim_state: Dimension of state which is equivalent here to number of mesh
                points in spatial discretization.
            observation_space_indices: Slice or sequence of integers specifying spatial
                mesh node indices (indices in to state vector) corresponding to
                observation points.
            observation_function: Function to apply to subsampled state field to compute
                mean of observation(s) given state(s) at a given time index. Defaults to
                identity function in first argument.
            time_step: Integrator time step.
            domain_extent: Extent (size) of spatial domain.
            damping_coeff: Coefficient (`γ` in description above) controlling degree of
                damping in dynamics.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_observation,)`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance.
            initial_state_amplitude: Amplitude scale parameter for initial random
                state field. Larger values correspond to larger magnitude values for the
                initial state.
            state_noise_amplitude: Amplitude scale parameter for additive state noise
                in model dynamics. Larger values correspond to larger magnitude
                additive noise in the state field.
            state_noise_length_scale: Length scale parameter for smoothed noise used to
                generate initial state and additive state noise fields. Larger values
                correspond to smoother fields.
            num_roots_of_unity_etdrk4_integrator: Number of roots of unity to use in
                approximating contour integrals in exponential time-differencing plus
                fourth-order Runge Kutta integrator.
        """
        assert dim_state % 2 == 0, "State dimension `dim_state` must be even"
        self.time_step = time_step
        self.observation_space_indices = observation_space_indices
        self.observation_function = observation_function
        spatial_freqs = np.arange(dim_state // 2 + 1) * 2 * np.pi / domain_extent
        spatial_freqs_sq = spatial_freqs ** 2
        spatial_freqs[dim_state // 2] = 0
        state_noise_kernel = (
            (time_step) ** 0.5
            * state_noise_amplitude
            * np.exp(-0.5 * spatial_freqs_sq * state_noise_length_scale ** 2)
            * (dim_state / domain_extent) ** 0.5
        )
        state_noise_std = rfft_coeff_to_real_array(
            state_noise_kernel + 1j * state_noise_kernel, False
        )
        initial_state_kernel = (
            initial_state_amplitude
            * np.exp(-0.5 * spatial_freqs_sq * state_noise_length_scale ** 2)
            * (dim_state / domain_extent) ** 0.5
        )
        initial_state_std = rfft_coeff_to_real_array(
            initial_state_kernel + 1j * initial_state_kernel, False
        )

        def linear_operator(freqs, freqs_sq):
            return freqs_sq - freqs_sq ** 2 - damping_coeff

        def nonlinear_operator(v, freqs, freqs_sq):
            return (
                -0.5j * freqs * fft.rfft(fft.irfft(v, norm="ortho") ** 2, norm="ortho")
            )

        self.integrator = FourierETDRK4Integrator(
            linear_operator=linear_operator,
            nonlinear_operator=nonlinear_operator,
            num_mesh_point=dim_state,
            domain_size=domain_extent,
            time_step=time_step,
            num_roots_of_unity=num_roots_of_unity_etdrk4_integrator,
        )
        if observation_function is None:
            dim_observation = np.zeros(dim_state)[observation_space_indices].shape[0]
        else:
            dim_observation = observation_function(
                np.zeros(dim_state)[observation_space_indices], 0
            ).shape[0]
        super().__init__(
            dim_state=dim_state,
            dim_observation=dim_observation,
            initial_state_std=initial_state_std,
            initial_state_mean=np.zeros(dim_state),
            state_noise_std=state_noise_std,
            observation_noise_std=observation_noise_std,
            **kwargs
        )

    def _next_state_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        return rfft_coeff_to_real_array(
            self.integrator.step(real_array_to_rfft_coeff(states))
        )

    def _observation_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        subsampled_states = fft.irfft(real_array_to_rfft_coeff(states), norm="ortho")[
            ..., self.observation_space_indices
        ]
        if self.observation_function is None:
            return subsampled_states
        else:
            return self.observation_function(subsampled_states, t)


class SpatialLaminarFlameModel(
    SpatiallyExtendedModelMixIn,
    OneDimensionalFourierTransformedDiagonalGaussianModelMixIn,
    FourierLaminarFlameModel,
):
    """Non-linear SPDE model on a periodic 1D spatial domain for laminar flame fronts.

    This model class represents the state field by its values at the spatial mesh points
    rather than the corresponding Fourier coefficients. For more details see the
    docstring of `FourierLaminarFlameModel`.
    """

    def __init__(
        self,
        dim_state: int = 512,
        observation_space_indices: Union[slice, Sequence[int]] = slice(4, None, 8),
        observation_function: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        time_step: float = 0.25,
        domain_extent: float = 32 * np.pi,
        damping_coeff: float = 1.0 / 6,
        observation_noise_std: float = 0.5,
        initial_state_amplitude: float = 1.0,
        state_noise_amplitude: float = 1.0,
        state_noise_length_scale: float = 1.0,
        num_roots_of_unity_etdrk4_integrator: int = 16,
    ):
        """
        Args:
            dim_state: Dimension of state which is equivalent here to number of mesh
                points in spatial discretization.
            observation_space_indices: Slice or sequence of integers specifying spatial
                mesh node indices (indices in to state vector) corresponding to
                observation points.
            observation_function: Function to apply to subsampled state field to compute
                mean of observation(s) given state(s) at a given time index. Defaults to
                identity function in first argument.
            time_step: Integrator time step.
            domain_extent: Extent (size) of spatial domain.
            damping_coeff: Coefficient (`γ` in description above) controlling degree of
                damping in dynamics.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_observation,)`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance.
            initial_state_amplitude: Amplitude scale parameter for initial random
                state field. Larger values correspond to larger magnitude values for the
                initial state.
            state_noise_amplitude: Amplitude scale parameter for additive state noise
                in model dynamics. Larger values correspond to larger magnitude
                additive noise in the state field.
            state_noise_length_scale: Length scale parameter for smoothed noise used to
                generate initial state and additive state noise fields. Larger values
                correspond to smoother fields.
            num_roots_of_unity_etdrk4_integrator: Number of roots of unity to use in
                approximating contour integrals in exponential time-differencing plus
                fourth-order Runge Kutta integrator.
        """
        super().__init__(
            dim_state=dim_state,
            observation_space_indices=observation_space_indices,
            observation_function=observation_function,
            time_step=time_step,
            domain_extent=domain_extent,
            damping_coeff=damping_coeff,
            observation_noise_std=observation_noise_std,
            initial_state_amplitude=initial_state_amplitude,
            state_noise_amplitude=state_noise_amplitude,
            state_noise_length_scale=state_noise_length_scale,
            num_roots_of_unity_etdrk4_integrator=num_roots_of_unity_etdrk4_integrator,
            mesh_shape=(dim_state,),
            domain_extents=(domain_extent,),
            domain_is_periodic=True,
            observation_node_indices=observation_space_indices,
        )
