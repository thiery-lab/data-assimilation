"""One-dimensional model with non-linear dynamics and observation operator.

Model originally proposed in:

> M. L. A. Netto, L. Gimeno, and M. J. Mendes.
> A new spline algorithm for non-linear filtering of discrete time systems.
> Proceedings of the 7th Triennial World Congress, 1979.
"""

import numpy as np
from dapy.utils.doc import inherit_docstrings
from dapy.models.base import DiagonalGaussianModel


@inherit_docstrings
class Netto79Model(DiagonalGaussianModel):
    """One-dimensional model with non-linear dynamics and observation operator.

    State update defined as

      z[t+1] = alpha * z[t] + beta * (z[t] / (1 + z[t]**2)) +
               gamma  * cos(delta * t) + sigma_z * u[t]

    with u[t] ~ N(0, 1) and z[0] ~ N(m, s**2).

    Observed process defined by

      x[t] = epsilon * z[t]**2 + sigma_x * v[t]

    with v[t] ~ N(0, 1).

    Standard parameter values assumed here are alpha = 0.5, beta = 25,
    gamma = 8, delta = 1.2, epsilon = 0.05, m=10, s=5, sigma_z = 1,
    sigma_x = 10**0.5 and T = 100 simulated time steps.

    References:

        M. L. A. Netto, L. Gimeno, and M. J. Mendes.
        A new spline algorithm for non-linear filtering of discrete time
        systems. Proceedings of the 7th Triennial World Congress, 1979.
    """

    def __init__(self, rng, init_state_mean=10., init_state_std=5.,
                 state_noise_std=1., obser_noise_std=10.**0.5, alpha=0.5,
                 beta=25., gamma=8., delta=1.2, epsilon=0.05):
        """
        Args:
            rng (RandomState): Numpy RandomState random number generator.
            init_state_mean (float): Initial state distribution mean.
            init_state_std (float): Initial state distribution standard
                deviation.
            state_noise_std (float): Standard deviation of additive Gaussian
                noise in state update.
            obser_noise_std (float): Standard deviation of additive Gaussian
                noise in observations.
            alpha (float): Coefficient in non-linear state update.
            beta (float): Coefficient in non-linear state update.
            gamma (float): Coefficient in non-linear state update.
            delta (float): Coefficient in non-linear state update.
            epsilon (float): Coefficient in non-linear observation function.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        super(Netto79Model, self).__init__(
            dim_z=1, dim_x=1, rng=rng, init_state_mean=init_state_mean,
            init_state_std=init_state_std, state_noise_std=state_noise_std,
            obser_noise_std=obser_noise_std)

    def next_state_func(self, z, t):
        return (
            self.alpha * z + self.beta * z / (1. + z**2) +
            self.gamma * np.cos(self.delta * t)
        )

    def observation_func(self, z, t):
        return self.epsilon * z**2
