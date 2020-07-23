"""Mix-in classes and functions for transformed models."""

import abc
from typing import Union, Tuple
import numpy as np
from numpy.random import Generator

try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    import numpy.fft as fft


class AbstractTransformedModelMixIn(abc.ABC):
    """Base for mix-ins for models in which state is transform of state of a base model.

    Given functions `forward_map` and `backward_map` which are mutual inverses, i.e.
    `forward_map(backward_map(x)) = x` and `backward_map(forward_map(x)) = x` for all
    `x`, and `sample_initial_state`, `sample_observation_given_state` and
    `sample_state_transition` functions (methods) of a 'base' model the model dynamics
    are described by

        state_sequence[0] = forward_map(sample_initial_state(rng))
        observation_sequence[0] = sample_observation_given_state(
            rng, bacward_map(state_sequence[0]), 0)
        for t in range(1, num_observation_time):
            state_sequence[t] = forward_map(
                sample_state_transition(rng, backward_map(state_sequence[t-1]), t-1))
            observation_sequence[t] = sample_observation_given_state(
                rng, backward_map(state_sequence[t]), t)

    """

    @abc.abstractmethod
    def forward_map(self, base_states: np.ndarray) -> np.ndarray:
        """Compute forward map of transform bijection.

        Args:

            base_states: Array of state(s) of *base model*. Should be of shape
                `(dim_state,)` or `(num_state, dim_state)` if multiple states to be
                mapped.

        Returns:

            Array of corresponding state(s) of *transformed model*, of shape
            `(dim_state,)` if `base_states` is one-dimensional otherwise of shape
            `(num_state, dim_state)`.
        """

    @abc.abstractmethod
    def backward_map(self, transformed_states: np.ndarray) -> np.ndarray:
        """Compute backward map of transform bijection.

        Args:

            transformed_states: Array of state(s) of *transformed model*. Should be of
                shape `(dim_state,)` or `(num_state, dim_state)` if multiple states to
                be mapped.

        Returns:

            Array of corresponding state(s) of *base model*, of shape `(dim_state,)` if
            `base_states` is one-dimensional otherwise `(num_state, dim_state)`.
        """

    @abc.abstractmethod
    def log_det_jacobian_backward_map(
        self, transformed_states: np.ndarray,
    ) -> Union[float, np.ndarray]:
        """Compute Jacobian log determinant of backward map of transform bijection.

        Args:

            transformed_states: Array of state(s) of *transformed model*. Should be of
                shape `(dim_state,)` or `(num_state, dim_state)` if multiple states to
                be mapped.

        Returns:

            transformed_states: Array of corresponding state(s) of *transformed model*,
                of shape `(dim_state,)` if `base_states` is one-dimensional otherwise
                `(num_state, dim_state)`.
        """

    def _sample_initial_state(self, rng: Generator, num_state: int) -> np.ndarray:
        return self.forward_map(super()._sample_initial_state(rng, num_state))

    def _sample_state_transition(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        return self.forward_map(
            super()._sample_state_transition(rng, self.backward_map(states), t)
        )

    def _sample_observation_given_state(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        return super()._sample_observation_given_state(
            rng, self.backward_map(states), t
        )

    def log_density_observation_given_state(
        self, observations: np.ndarray, states: np.ndarray, t: int
    ) -> np.ndarray:
        return super().log_density_observation_given_state(
            observations, self.backward_map(states), t
        )

    def log_density_initial_state(self, states: np.ndarray) -> np.ndarray:
        return super().log_density_initial_state(
            self.backward_map(states)
        ) - self.log_det_jacobian_backward_map(states)

    def log_density_state_transition(
        self, next_states: np.ndarray, states: np.ndarray, t: int
    ) -> np.ndarray:
        return super().log_density_state_transition(
            self.backward_map(next_states), self.backward_map(states), t
        ) - self.log_det_jacobian_backward_map(next_states)


class AbstractTransformedDiagonalGaussianModelMixin(AbstractTransformedModelMixIn):
    """Specialisation of `AbstractTransformedModelMixIn` for diagonal Gaussian models.

    Additionally applies transformations to `observation_mean` and `next_state_mean`
    methods so that these methods operate on *transformed* state to allow consistent
    external use.
    """

    def observation_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        return self._observation_mean(self.backward_map(states), t)

    def next_state_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        return self.forward_map(self._next_state_mean(self.backward_map(states), t))


class AbstractLinearTransformedDiagonalGaussianModelMixin(
    AbstractTransformedDiagonalGaussianModelMixin
):
    """Linear specialisation of `AbstractTransformedDiagonalGaussianModelMixin`.

    Assumes transformation is linear.

    Additionally provides `initial_state_covar` and `state_noise_covar` properties
    which exploit linearity of transformation to compute covariance matrices from
    linear transform matrix and base model diagonal covariances.
    methods so that these methods operate on *transformed* state to allow consistent
    external use.
    """

    @property
    def initial_state_mean(self) -> np.ndarray:
        return self.forward_map(self._initial_state_mean)

    @property
    def forward_transform_matrix(self) -> np.ndarray:
        if not hasattr(self, "_transform_matrix"):
            self._forward_transform_matrix = self.forward_map(
                np.identity(self.dim_state)
            )
        return self._forward_transform_matrix

    @property
    def initial_state_covar(self) -> np.ndarray:
        if not hasattr(self, "__initial_state_covar"):
            self.__initial_state_covar = (
                self.forward_transform_matrix * self._initial_state_std ** 2
            ) @ self.forward_transform_matrix.T
        return self.__initial_state_covar

    @property
    def state_noise_covar(self) -> np.ndarray:
        if not hasattr(self, "__state_noise_covar"):
            self.__state_noise_covar = (
                self.forward_transform_matrix * self._state_noise_std ** 2
            ) @ self.forward_transform_matrix.T
        return self.__state_noise_covar


def rfft_coeff_to_real_array(
    rfft_coeff: np.ndarray, orth_scale: bool = True
) -> np.ndarray:
    """Maps a batch of complex `rfft` coefficients array(s) to a batch of real arrays.

    Args:
        rfft_coeffs: Array of complex `rfft` coefficients of real 1D array(s), either a
           one dimensional array of shape `(length_array // 2 + 1,)` where
           `length_array` is the *even* length of the real-valued 1D array the `rfft`
           was applied to, or a two dimensional array of shape `(num_array, length_array
           // 2 + 1)` if coefficients are being provided for multiple distinct 1D
           arrays, with `num_array` being the number of 1D arrays.
        orth_scale: Whether to rescale the complex coefficients such that when composed
            with `fft.rfft` this function forms an orthogonal linear map, or whether
            to simply concatenate the the coefficients without rescaling.

    Returns:
        An real-value array of shape `(length_array,)` if `rfft_coeffs` is
        one-dimensional or of shape `(num_array, length_array)`, which concatenates the
        real and imaginary components of the `rfft` coefficients and rescales the zero
        and Nyquist frequence components so that

            rfft_coeff_to_real_array(fft.rfft(array, norm='ortho'), orth_scale=True)

        is an orthogonal linear map on real valued arrays of length `len(array)`.
    """
    multiplier = 2 ** 0.5 if orth_scale else 1
    return np.concatenate(
        [
            rfft_coeff[..., :1].real,
            rfft_coeff[..., 1:-1].real * multiplier,
            rfft_coeff[..., -1:].real,
            rfft_coeff[..., 1:-1].imag * multiplier,
        ],
        axis=-1,
    )


def real_array_to_rfft_coeff(
    real_array: np.ndarray, orth_scale: bool = True
) -> np.ndarray:
    """Maps a batch of real arrays to a batch of complex `rfft` coefficients array(s).

    Args:
        real_array: Array of real-mapped transforms of complex `rfft` coefficients of
            real 1D arrays, either a one dimensional array of shape `(length_array,)`
            where `length_array` is the *even* length of the real-valued 1D array `rfft`
            was applied to, or a two dimensional array of shape `(num_array,
            length_array)` if real-mapped coefficients are being provided for multiple
            distinct sequences, with `num_array` being the number of 1D arrays.
        orth_scale: Whether to rescale the complex coefficients such that when composed
            with `fft.irfft` this function forms an orthogonal linear map, or whether
            to simply concatenate the the coefficients without rescaling.

    Returns:
        A complex-value array of shape `(length_array // 2 + 1,)` if `real_array` is
        one-dimensional or of shape `(num_array, length_array // 2 + 1)`
        otherwise, which corresponds to the original complex valued `rfft` coefficients
        and is scaled such that

            fft.irfft(real_array_to_rfft_coeff(array, orth_scale=True), norm='ortho'))

        is an orthogonal linear map on real valued arrays of length `len(array)`.
    """
    rfft_coeffs = real_array[..., : real_array.shape[-1] // 2 + 1] * (1 + 0j)
    rfft_coeffs[..., 1:-1] += real_array[..., real_array.shape[-1] // 2 + 1 :] * 1j
    if orth_scale:
        rfft_coeffs[..., 1:-1] /= 2 ** 0.5
    return rfft_coeffs


class OneDimensionalFourierTransformedDiagonalGaussianModelMixIn(
    AbstractLinearTransformedDiagonalGaussianModelMixin
):
    """Transformed model mix-in for diagonal Gaussian 1D Fourier domain base models.

    Applies a 1D FFT based transform to model state to map from Fourier domain
    representation to a 1D spatial domain representation. The base model in the Fourier
    domain representation is assumed to have an initial state distribution, state noise
    distribution and observation noise distribution which are all Gaussian with a
    diagonal covariance matrix.
    """

    def forward_map(self, states: np.ndarray) -> np.ndarray:
        return fft.irfft(real_array_to_rfft_coeff(states), norm="ortho")

    def backward_map(self, states: np.ndarray) -> np.ndarray:
        return rfft_coeff_to_real_array(fft.rfft(states, norm="ortho"))

    def log_det_jacobian_backward_map(
        self, states: np.ndarray
    ) -> Union[float, np.ndarray]:
        return 0


def rfft2_coeff_to_real_array(rfft2_coeff: np.ndarray, orth_scale=True) -> np.ndarray:
    """Maps a batch of complex `rfft2` coefficients array(s) to a batch of real arrays.

    Args:
        rfft2_coeffs: Array of complex `rfft2` coefficients of real 2D array(s), either a
           two dimensional array of shape `(dim_0, dim_1 // 2 + 1,)` where
           `(dim_0, dim_1)` is the shape of the real-valued 2D array(s) `rfft2`
           was applied to, or a three dimensional array of shape `(num_array,
           dim_0, dim_1 // 2 + 1)` if coefficients are being provided for multiple
           distinct 2D arrays, with `num_array` being the number of 2D arrays.
        orth_scale: Whether to rescale the complex coefficients such that when composed
            with `fft.rfft2` this function forms an orthogonal linear map, or whether
            to simply concatenate the the coefficients without rescaling.

    Returns:
        An real-value array of shape `(dim_0 * dim_1,)` if `rfft2_coeffs` is
        one-dimensional or of shape `(num_array, dim_0 * dim_1)`, which
        concatenates the real and imaginary components of the `rfft2` coefficients and
        rescales the zero and Nyquist frequence components so that

            rfft2_coeff_to_real_array(
                fft.rfft2(array.reshape((dim_0, dim_1)), norm='ortho'), orth_scale=True)

        is an orthogonal linear map on real-valued arrays of length `dim_0 * dim_1`.
    """
    dim_0 = rfft2_coeff.shape[-2]
    multiplier = 2 ** 0.5 if orth_scale else 1
    return np.concatenate(
        [
            rfft2_coeff[..., 0:1, 0].real,
            rfft2_coeff[..., 1 : dim_0 // 2, 0].real * multiplier,
            rfft2_coeff[..., dim_0 // 2 : dim_0 // 2 + 1, 0].real,
            rfft2_coeff[..., 1 : dim_0 // 2, 0].imag * multiplier,
            rfft2_coeff[..., :, 1:-1].reshape(rfft2_coeff.shape[:-2] + (-1,)).real
            * multiplier,
            rfft2_coeff[..., :, 1:-1].reshape(rfft2_coeff.shape[:-2] + (-1,)).imag
            * multiplier,
            rfft2_coeff[..., 0:1, -1].real,
            rfft2_coeff[..., 1 : dim_0 // 2, -1].real * multiplier,
            rfft2_coeff[..., dim_0 // 2 : dim_0 // 2 + 1, -1].real,
            rfft2_coeff[..., 1 : dim_0 // 2, -1].imag * multiplier,
        ],
        -1,
    )


def real_array_to_rfft2_coeff(
    real_array: np.ndarray, mesh_shape: Tuple[int, int], orth_scale=True
) -> np.ndarray:
    """Maps a batch of real arrays to a batch of complex `rfft2` coefficients array(s).

    Args:
        real_array: Array of real-mapped transforms of complex `rfft2` coefficients of
            real two-dimensional array(s), either a one dimensional array of shape
            `(dim_0 * dim_1,)` where `(dim_0, dim_1)` is the shape of the real-valued
            array `rfft2` was computed on, or a two dimensional array of shape
            `(num_array, dim_0 * dim_1)` if real-mapped coefficients are being provided
            for multiple distinct 2D arrays, with `num_array` being the number of
            arrays.
        mesh_shape: Shape of original two-dimensional array(s) fed to `rfft2`,
            equivalent to `(dim_0, dim_1)` in the description above.
        orth_scale: Whether to rescale the complex coefficients such that when composed
            with `fft.irfft2` this function forms an orthogonal linear map, or whether
            to simply concatenate the the coefficients without rescaling.

    Returns:
        A complex-value array of shape `(dim_0, dim_1 // 2 + 1,)` if `real_array` is
        one-dimensional or of shape `(num_array, dim_0, dim_1 // 2 + 1)` otherwise,
        which corresponds to the original complex valued (R)FFT coefficients and is
        scaled such that

            fft.irfft2(real_array_to_rfft2_coeff(
                array, (dim_0, dim_1), orth_scale=True), norm='ortho'))

        is an orthogonal linear map on real-valued arrays of length `dim_0 * dim_1`.
    """
    dim_0, dim_1 = mesh_shape
    rfft2_coeff = np.empty(
        real_array.shape[:-1] + (dim_0, dim_1 // 2 + 1), dtype=np.complex128
    )
    rfft2_coeff[..., 0, 0] = real_array[..., 0]
    rfft2_coeff[..., 1 : dim_0 // 2, 0] = (
        real_array[..., 1 : dim_0 // 2] + real_array[..., dim_0 // 2 + 1 : dim_0] * 1j
    ) / 2 ** 0.5
    rfft2_coeff[..., dim_0 // 2, 0] = real_array[..., dim_0 // 2]
    rfft2_coeff[..., :, 1:-1] = (
        real_array[..., dim_0 : dim_0 * dim_1 // 2]
        + real_array[..., dim_0 * dim_1 // 2 : dim_0 * (dim_1 - 1)] * 1j
    ).reshape(real_array.shape[:-1] + (dim_0, dim_1 // 2 - 1)) / 2 ** 0.5
    rfft2_coeff[..., 0, -1] = real_array[..., dim_0 * (dim_1 - 1)]
    rfft2_coeff[..., 1 : dim_0 // 2, -1] = (
        real_array[..., dim_0 * (dim_1 - 1) + 1 : dim_0 * (2 * dim_1 - 1) // 2]
        + real_array[..., dim_0 * (2 * dim_1 - 1) // 2 + 1 :] * 1j
    ) / 2 ** 0.5
    rfft2_coeff[..., dim_0 // 2, -1] = real_array[..., dim_0 * (2 * dim_1 - 1) // 2 + 1]
    rfft2_coeff[..., dim_0 // 2 + 1 :, 0] = rfft2_coeff[..., dim_0 // 2 - 1 : 0 : -1, 0]
    rfft2_coeff[..., dim_0 // 2 + 1 :, -1] = rfft2_coeff[
        ..., dim_0 // 2 - 1 : 0 : -1, -1
    ]
    return rfft2_coeff


class TwoDimensionalFourierTransformedDiagonalGaussianModelMixIn(
    AbstractLinearTransformedDiagonalGaussianModelMixin
):
    """Transformed model mix-in for diagonal Gaussian 2D Fourier domain base models.

    Applies a 2D FFT based transform to model state to map from Fourier domain
    representation to a 2D spatial domain representation. The base model in the Fourier
    domain representation is assumed to have an initial state distribution, state noise
    distribution and observation noise distribution which are all Gaussian with a
    diagonal covariance matrix.
    """

    def forward_map(self, states: np.ndarray) -> np.ndarray:
        return fft.irfft2(
            real_array_to_rfft2_coeff(states, self.mesh_shape), norm="ortho"
        ).reshape(states.shape[:-1] + (-1,))

    def backward_map(self, states: np.ndarray) -> np.ndarray:
        return rfft2_coeff_to_real_array(
            fft.rfft2(states.reshape(states.shape[:-1] + self.mesh_shape), norm="ortho")
        )

    def log_det_jacobian_backward_map(
        self, states: np.ndarray
    ) -> Union[float, np.ndarray]:
        return 0
