"""Particle filters for inference in state space models."""

import numpy as np
from scipy.special import logsumexp
from dapy.inference.base import (
        AbstractEnsembleFilter, AbstractLocalEnsembleFilter)
from dapy.utils.doc import inherit_docstrings
import dapy.ot as ot


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
        log_sum_w = logsumexp(log_w)
        return np.exp(log_w - log_sum_w)

    def analysis_transform(self, z_forecast, weights):
        """Perform multinomial particle resampling given computed weights."""
        n_particles = z_forecast.shape[0]
        idx = self.rng.choice(n_particles, n_particles, True, weights)
        return z_forecast[idx]

    def analysis_update(self, z_forecast, x_observed, time_index):
        w = self.calculate_weights(z_forecast, x_observed, time_index)
        z_analysis_mean = (w[:, None] * z_forecast).sum(0)
        z_analysis_std = (
                w[:, None] * (z_forecast - z_analysis_mean)**2).sum(0)**0.5
        z_analysis = self.analysis_transform(z_forecast, w)
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

    def __init__(self, init_state_sampler, next_state_sampler,
                 log_prob_dens_obs_gvn_state, rng,
                 ot_solver=ot.solve_optimal_transport_exact,
                 ot_solver_params={}):
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
            ot_solver (function): Optimal transport solver function with
                call signature
                    ot_solver(source_dist, target_dist, cost_matrix,
                              **ot_solver_params)
                where source_dist and target_dist are the source and target
                distribution weights respectively as 1D arrays, cost_matrix is
                a 2D array of the distances between particles and
                ot_solver_params is any additional keyword parameter values
                for the solver.
            ot_solver_params (dict): Any additional keyword parameters values
                for the optimal transport solver.
        """
        super(EnsembleTransformParticleFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler,
                log_prob_dens_obs_gvn_state=log_prob_dens_obs_gvn_state,
                rng=rng)
        self.ot_solver = ot_solver
        self.ot_solver_params = ot_solver_params

    def analysis_transform(self, z_forecast, weights):
        """Solve optimal transport problem and transform ensemble."""
        n_particles = z_forecast.shape[0]
        source_dist = np.ones(n_particles) / n_particles
        target_dist = weights
        target_dist /= target_dist.sum()
        # Cost matrix entries Euclidean distance between particles
        cost_matrix = ot.pairwise_euclidean_distance(z_forecast, z_forecast)
        trans_matrix = n_particles * self.ot_solver(
            source_dist, target_dist, cost_matrix, **self.ot_solver_params)
        return trans_matrix.dot(z_forecast)


class LocalEnsembleTransformParticleFilter(AbstractLocalEnsembleFilter):

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_func, obser_noise_std, n_grid, localisation_func,
                 rng=None, inflation_factor=1.,
                 ot_solver=ot.solve_optimal_transport_exact,
                 ot_solver_params={}):
        """
        Args:
            init_state_sampler (callable): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (callable): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as
                arguments.
            observation_func (callable): Function returning pre-noise
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
            localisation_func (callable): Function (or callable object) which
                given an index corresponding to a spatial grid point (i.e.
                the iteration index over the last dimension of a reshaped
                array `z_grid` as described above) will return an array of
                integer indices into an observation vector and corresponding
                array of weight coefficients specifying the observation vector
                entries 'local' to state grid point described by the index and
                there corresponding weights (with closer observations
                potentially given larger weights).
            rng (RandomState): Numpy RandomState random number generator.
            inflation_factor (float): A value greater than or equal to one used
                to inflate the analysis ensemble on each update as a heuristic
                to overcome the underestimation of the uncertainty in the
                system state by ensemble methods.
            ot_solver (callable): Optimal transport solver function with
                call signature
                    ot_solver(source_dist, target_dist, cost_matrix,
                              **ot_solver_params)
                where source_dist and target_dist are the source and target
                distribution weights respectively as 1D arrays, cost_matrix is
                a 2D array of the distances between particles and
                ot_solver_params is any additional keyword parameter values
                for the solver.
            ot_solver_params (dict): Any additional keyword parameters values
                for the optimal transport solver.
        """
        super(LocalEnsembleTransformParticleFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler,
                observation_func=observation_func,
                obser_noise_std=obser_noise_std,
                n_grid=n_grid, localisation_func=localisation_func, rng=rng
        )
        self.inflation_factor = inflation_factor
        self.ot_solver = ot_solver
        self.ot_solver_params = ot_solver_params

    def local_analysis_update(self, z_forecast, x_forecast, x_observed,
                              obs_noise_std, localisation_weights):
        n_particles = z_forecast.shape[0]
        dx_error = x_forecast - x_observed
        log_particle_weights = -0.5 * (
            dx_error * (localisation_weights / obs_noise_std**2) * dx_error
        ).sum(-1)
        target_dist = np.exp(
            log_particle_weights - logsumexp(log_particle_weights))
        target_dist /= target_dist.sum()
        source_dist = np.ones(n_particles) / n_particles
        cost_matrix = ot.pairwise_euclidean_distance(z_forecast, z_forecast)
        trans_matrix = n_particles * self.ot_solver(
            source_dist, target_dist, cost_matrix, **self.ot_solver_params)
        z_analysis = trans_matrix.dot(z_forecast)
        if self.inflation_factor > 1.:
            z_analysis_mean = z_analysis.mean(0)
            dz_analysis = z_analysis - z_analysis_mean
            return z_analysis_mean + dz_analysis * self.inflation_factor
        else:
            return z_analysis


class ScalableLocalEnsembleTransportParticleFilter(AbstractEnsembleFilter):
    """Scalable local ensemble transport particle filter."""

    def __init__(self, init_state_sampler, next_state_sampler, rng,
                 log_obs_dens_per_loc_func, obs_coords, loc_func, loc_radius,
                 pou, calculate_cost_matrices_func, ot_solver,
                 ot_solver_params={}):
        super().__init__(init_state_sampler, next_state_sampler, rng)
        self.log_obs_dens_per_loc_func = log_obs_dens_per_loc_func
        self.loc_func = loc_func
        self.pou = pou
        self.ot_solver = ot_solver
        self.ot_solver_params = ot_solver_params
        self.loc_kernel_obs_coord = np.stack(
            [loc_func(pou.patch_distance(p, obs_coords), loc_radius)
             for p in range(pou.n_patch)], -1)
        self.calculate_cost_matrices_func = calculate_cost_matrices_func

    def analysis_update(self, z_forecast, x_observed, time_index):
        n_particle = z_forecast.shape[0]
        log_obs_dens_per_loc = self.log_obs_dens_per_loc_func(
            z_forecast, x_observed, time_index)
        log_target_dists = log_obs_dens_per_loc.dot(self.loc_kernel_obs_coord)
        log_target_dists -= logsumexp(log_target_dists, axis=0)
        target_dists = np.exp(log_target_dists.T)
        source_dists = np.ones_like(target_dists) / n_particle
        z_forecast = z_forecast.reshape(
            (z_forecast.shape[0], -1, self.pou.n_node))
        if self.calculate_cost_matrices_func is not None:
            cost_matrices = self.calculate_cost_matrices_func(z_forecast)
        else:
            z_dist_matrices = np.sum(
                (z_forecast[:, None] - z_forecast[None, :])**2, -2)
            cost_matrices = self.pou.split_into_patches(
                z_dist_matrices).sum(-1)
            cost_matrices = np.moveaxis(cost_matrices, 2, 0)
        trans_matrices = self.ot_solver(
            source_dists, target_dists, cost_matrices,
            **self.ot_solver_params) * n_particle
        scaled_z_forecast_patches = (
            self.pou.split_into_patches_and_scale(z_forecast))
        z_analysis_patches = np.einsum(
            'kij,jlkm->ilkm', trans_matrices, scaled_z_forecast_patches)
        z_analysis = self.pou.combine_patches(
            z_analysis_patches).reshape((z_forecast.shape[0], -1))
        z_analysis_mean = z_analysis.mean(0)
        return z_analysis, z_analysis_mean, z_analysis.std(0)
