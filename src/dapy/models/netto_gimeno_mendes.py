"""One-dimensional model with non-linear dynamics and observation operator.

Model originally proposed in:

> M. L. A. Netto, L. Gimeno, and M. J. Mendes.
> A new spline algorithm for non-linear filtering of discrete time systems.
> Proceedings of the 7th Triennial World Congress, 1979.
"""

import numpy as np
from dapy.models.base import AbstractDiagonalGaussianModel


class NettoGimenoMendesModel(AbstractDiagonalGaussianModel):
    """One-dimensional model with non-linear dynamics and observation operator.

    State update defined as

        state_sequence[t+1] = (
            alpha * state_sequence[t] +
            beta * (state_sequence[t] / (1 + state_sequence[t]**2)) +
            gamma  * cos(delta * t) + state_noise_std * rng.standard_normal())

    with initial state

        state_sequence[0] = (
            initial_state_mean + initial_state_std * rng.standard_normal())

    Observed process defined by

        observation_sequence[t] = (
            epsilon * state_sequence[t]**2 +
            observation_noise_std * rng.standard_normal())

    References:

        1. M. L. A. Netto, L. Gimeno, and M. J. Mendes.
           A new spline algorithm for non-linear filtering of discrete time
           systems. Proceedings of the 7th Triennial World Congress, 1979.
    """

    def __init__(
        self,
        initial_state_mean: float = 10.0,
        initial_state_std: float = 5.0,
        state_noise_std: float = 1.0,
        observation_noise_std: float = 10.0 ** 0.5,
        alpha: float = 0.5,
        beta: float = 25.0,
        gamma: float = 8.0,
        delta: float = 1.2,
        epsilon: float = 0.05,
    ):
        """
        Args:
            initial_state_mean: Initial state distribution mean.
            initial_state_std: Initial state distribution standard deviation.
            state_noise_std: Standard deviation of additive Gaussian noise in state
                update.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations.
            alpha: Coefficient in non-linear state update.
            beta: Coefficient in non-linear state update.
            gamma: Coefficient in non-linear state update.
            delta: Coefficient in non-linear state update.
            epsilon: Coefficient in non-linear observation function.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        super().__init__(
            dim_state=1,
            dim_observation=1,
            initial_state_mean=initial_state_mean,
            initial_state_std=initial_state_std,
            state_noise_std=state_noise_std,
            observation_noise_std=observation_noise_std,
        )

    def _next_state_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        return (
            self.alpha * states
            + self.beta * states / (1.0 + states ** 2)
            + self.gamma * np.cos(self.delta * t)
        )

    def _observation_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        return self.epsilon * states ** 2
