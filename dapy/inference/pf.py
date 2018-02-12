"""Particle filters for inference in state space models."""

import numpy as np
import ot
from dapy.inference.base import (
        AbstractEnsembleFilter, AbstractLocalEnsembleFilter)
from dapy.utils import inherit_docstrings


def log_sum_exp(x):
    """Compute logarithm of sum of exponents with improved numerical stability.

    Args:
        x (array): One-dimension arrray of values to calculate log-sum-exp of.

    Returns:
        Scalar corresponding to logarith of sum of exponents of x values.
    """
    x_max = x.max()
    return x_max + np.log(np.exp(x - x_max).sum())


@inherit_docstrings
class BootstrapParticleFilter(AbstractEnsembleFilter):
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
        super(BootstrapParticleFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler, rng=rng
        )
        self.log_prob_dens_obs_gvn_state = log_prob_dens_obs_gvn_state

    def calculate_weights(self, z, x, t):
        """Calculate importance weights for particles given observations."""
        log_w = self.log_prob_dens_obs_gvn_state(x, z, t)
        log_sum_w = log_sum_exp(log_w)
        return np.exp(log_w - log_sum_w)

    def resample(self, z, w):
        """Perform multinomial particle resampling given computed weights."""
        n_particles = z.shape[0]
        idx = self.rng.choice(n_particles, n_particles, True, w)
        return z[idx]

    def analysis_update(self, z_forecast, x_observed, time_index):
        w = self.calculate_weights(z_forecast, x_observed, time_index)
        z_analysis_mean = (w[:, None] * z_forecast).sum(0)
        z_analysis_std = (
                w[:, None] * (z_forecast - z_analysis_mean)**2).sum(0)**0.5
        z_analysis = self.resample(z_forecast, w)
        return z_analysis, z_analysis_mean, z_analysis_std


@inherit_docstrings
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


class LocalEnsembleTransformParticleFilter(AbstractLocalEnsembleFilter):

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_func, obser_noise_std, n_grid, localisation_func,
                 inflation_factor, rng, use_sinkhorn=False, sinkhorn_reg=1e-3):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as
                arguments.
            observation_func (function): Function returning pre-noise
                observations given current state(s). Takes array of current
                state(s) and current time index as arguments.
            obser_noise_std (array): One-dimensional array defining standard
                deviations of additive Gaussian observation noise on each
                dimension with it assumed that the noise is independent across
                dimensions i.e. a diagonal observation noise covariance matrix.
            n_grid (integer): Number of spatial points over which state is
                defined. Typically points will be on a rectilinear grid though
                this is not actually required. It is assumed that if `z` is a
                state vector of size `dim_z` then `dim_z % n_grid == 0` and
                that `z` is ordered such that iterating over the last
                dimension of a reshaped array
                    z_grid = z.reshape((dim_z // n_grid, n_grid))
                will correspond to iterating over the state component values
                across the different spatial (grid) locations.
            localisation_func (function): Function (or callable object) which
                given an index corresponding to a spatial grid point (i.e.
                the iteration index over the last dimension of a reshaped
                array `z_grid` as described above) will return an array of
                integer indices into an observation vector and corresponding
                array of weight coefficients specifying the observation vector
                entries 'local' to state grid point described by the index and
                there corresponding weights (with closer observations
                potentially given larger weights).
            inflation_factor (float): A value greater than or equal to one used
                to inflate the analysis ensemble on each update as a heuristic
                to overcome the underestimation of the uncertainty in the
                system state by ensemble Kalman filter methods.
            rng (RandomState): Numpy RandomState random number generator.
            use_sinkhorn (bool): Flag indicating whether to use entropic
                regularised optimal transport solution using Sinkhorn-Knopp
                algorithm rather than non-regularised earth mover distance.
            sinkhorn_reg (float): Positive entropic regularisation coefficient
                if using Sinkhorn optimal transport solver.
        """
        super(LocalEnsembleTransformParticleFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler,
                observation_func=observation_func,
                obser_noise_std=obser_noise_std,
                n_grid=n_grid, localisation_func=localisation_func, rng=rng
        )
        self.inflation_factor = inflation_factor
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_reg = sinkhorn_reg

    def local_analysis_update(self, z_forecast, x_forecast, x_observed,
                              obs_noise_std, localisation_weights):
        n_particles = z_forecast.shape[0]
        dx_error = x_forecast - x_observed
        log_particle_weights = -0.5 * (
            dx_error * (localisation_weights / obs_noise_std**2) * dx_error
        ).sum(-1)
        log_particle_weights_sum = log_sum_exp(log_particle_weights)
        particle_weights = np.exp(
            log_particle_weights - log_particle_weights_sum)
        u = ot.unif(n_particles)
        cost_mtx = ot.dist(z_forecast, z_forecast)
        if self.use_sinkhorn:
            trans_mtx = ot.sinkhorn(
                particle_weights, u, cost_mtx, self.sinkhorn_reg) * n_particles
        else:
            trans_mtx = ot.emd(particle_weights, u, cost_mtx) * n_particles
        z_analysis = trans_mtx.T.dot(z_forecast)
        z_analysis_mean = z_analysis.mean(0)
        dz_analysis = z_analysis - z_analysis_mean
        return z_analysis_mean + dz_analysis * self.inflation_factor
