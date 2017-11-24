"""Particle filters for inference in state space models."""

import numpy as np
import ot


class BootstrapParticleFilter(object):
    """Bootstrap particle filter (sequential importance resampling).

    Resampling step uses multinomial resampling.

    Assumes the system dynamics are of the form

    z[0] = init_state_sampler()
    x[0] = observation_sampler(z[0], 0)
    for t in range(1, T):
        z[t] = next_state_sampler(z[t-1], t-1)
        x[t] = observation_sampler(z[t], t)

    It is further assumed the conditional distribution on the observations
    given system states has a well-defined density with respect to the
    Lebesgue measure

        p(x[t] = x_ | z[t] = z_)

    which can be evaluated pointwise up to a potentially unknown normalising
    constant.

    The distribution of the system state at each time step is approximated
    by propagating a system of particles each representing a single system
    forward through time. In the limit of an infinite number of particles and
    assuming stable dynamics the particle filter converges to giving exact
    results.

    References:
        Gordon, N.J.; Salmond, D.J.; Smith, A.F.M. (1993). Novel approach to
        nonlinear / non-Gaussian Bayesian state estimation. Radar and Signal
        Processing, IEE Proceedings F. 140 (2): 107--113.

        Del Moral, Pierre (1996). Non Linear Filtering: Interacting Particle
        Solution. Markov Processes and Related Fields. 2 (4): 555--580.
    """

    def __init__(self, init_state_sampler, next_state_sampler,
                 log_prob_dens_obs_gvn_state, rng):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as
                arguments.
            log_prob_dens_obs_gvn_state (function): Function returning log
                probability density (up to an additive constant) of
                observation vector given state vector at the corresponding
                time index.
            rng (RandomState): Numpy RandomState random number generator.
        """
        self.init_state_sampler = init_state_sampler
        self.next_state_sampler = next_state_sampler
        self.log_prob_dens_obs_gvn_state = log_prob_dens_obs_gvn_state
        self.rng = rng

    def calculate_weights(self, z, x, t):
        """Calculate importance weights for particles given observations."""
        log_w = self.log_prob_dens_obs_gvn_state(x, z, t)
        log_w_max = log_w.max()
        log_sum_w = log_w_max + np.log(np.exp(log_w - log_w_max).sum())
        return np.exp(log_w - log_sum_w)

    def resample(self, z, w):
        """Perform multinomial particle resampling given computed weights."""
        n_particles = z.shape[0]
        idx = self.rng.choice(n_particles, n_particles, True, w)
        return z[idx]

    def filter(self, x_observed, n_particles):
        """Compute filtering distribution approximations.

        Args:
            x_observed (array): Observed state sequence with shape
                `(n_steps, dim_x)` where `n_steps` is number of time steps in
                sequence and `dim_x` is dimensionality of observations.
            n_particles (integer): Number of particles to use to represent
                filtering distribution at each time step.

        Returns:
            Dictionary containing arrays of filtering density parameters -
                z_mean_seq: Array of filtering density means at all time steps.
                z_particles_seq: Array of particles representing filtering
                    distribution at each time step.
        """
        n_steps, dim_x = x_observed.shape
        for t in range(n_steps):
            if t == 0:
                z_forecast = self.init_state_sampler(n_particles)
                dim_z = z_forecast.shape[1]
                z_mean_seq = np.full((n_steps, dim_z), np.nan)
                z_particles_seq = np.full(
                    (n_steps, n_particles, dim_z),  np.nan)
            else:
                z_forecast = self.next_state_sampler(z_particles_seq[t-1], t-1)
            w = self.calculate_weights(z_forecast, x_observed[t], t)
            z_mean_seq[t] = (w[:, None] * z_forecast).sum(0)
            z_particles_seq[t] = self.resample(z_forecast, w)
        return {'z_mean_seq': z_mean_seq, 'z_particles_seq': z_particles_seq}


class EnsembleTransformParticleFilter(BootstrapParticleFilter):
    """Ensemble transform particle filter.

    Ensemble transform step uses optimal transport to calculate linear
    ensemble transform which adjusts for non-uniform weights of forecast
    ensemble due to observations.

    Assumes the system dynamics are of the form

    z[0] = init_state_sampler()
    x[0] = observation_sampler(z[0], 0)
    for t in range(1, T):
        z[t] = next_state_sampler(z[t-1], t-1)
        x[t] = observation_sampler(z[t], t)

    It is further assumed the conditional distribution on the observations
    given system states has a well-defined density with respect to the
    Lebesgue measure

        p(x[t] = x_ | z[t] = z_)

    which can be evaluated pointwise up to a potentially unknown normalising
    constant.

    The distribution of the system state at each time step is approximated
    by propagating a system of particles each representing a single system
    forward through time. In the limit of an infinite number of particles and
    assuming stable dynamics the particle filter converges to giving exact
    results.

    References:
        Reich, S. (2013). A nonparametric ensemble transform method for
        Bayesian inference. SIAM Journal on Scientific Computing, 35(4),
        A2013-A2024.
    """

    def resample(self, z, w):
        """Solve optimal transport problem and transform ensemble."""
        n_particles = z.shape[0]
        u = np.ones(n_particles) / n_particles
        # Cost matrix entries Euclidean distance between particles
        cost_mtx = ot.dist(z, z)
        trans_mtx = ot.emd(w, u, cost_mtx) * n_particles
        return trans_mtx.T.dot(z)
