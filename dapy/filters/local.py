import abc
from typing import Tuple, Dict, Callable, Any, Optional
import numpy as np
from numpy.random import Generator
from scipy.special import logsumexp
from dapy.filters.base import AbstractEnsembleFilter
from dapy.models.base import AbstractModel
import dapy.ot as optimal_transport


class AbstractLocalEnsembleFilter(AbstractEnsembleFilter):
    """Localised ensemble filter base class for spatially extended models.

    Assumes system state and observations are defined over a fixed set of
    points in a spatial domain and that dependencies between state values at a
    point and observations are signficant only for observations in a localised
    region around the state location. It is further assumed here that the
    observations at a time point are conditionally independent given the state
    with a diagonal covariance Gaussian conditional distribution. Under these
    assumptions, when performing the analysis update to the forecasted state
    ensemble to take in to account the observations at a given time index, the
    ensemble state values at each grid point can each be updated independently
    based only a local subset of the observations.
    """

    def __init__(
        self,
        n_grid,
        localisation_func,
    ):
        """
        Args:
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
        """
        self.n_grid = n_grid
        self.localisation_func = localisation_func

    def _assimilation_update(
        self,
        model: AbstractModel,
        rng: Generator,
        state_particles: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_particle = state_particles.shape[0]
        z_forecast_grid = z_forecast.reshape((n_particles, -1, self.n_grid))
        observation_means = self.observation_mean(state_particles, time_index)
        z_analysis_grid = np.empty(z_forecast_grid.shape)
        for grid_index in range(self.n_grid):
            obs_indices, obs_weights = self.localisation_func(grid_index)
            z_forecast_local = z_forecast_grid[:, :, grid_index]
            x_forecast_local = x_forecast[:, obs_indices]
            x_observed_local = x_observed[obs_indices]
            obs_noise_std_local = self.obser_noise_std[obs_indices]
            z_analysis_grid[:, :, grid_index] = self._local_assimilation_update(
                z_forecast_local,
                x_forecast_local,
                x_observed_local,
                obs_noise_std_local,
                obs_weights,
            )
        z_analysis = z_analysis_grid.reshape((n_particles, -1))
        return state_particles, state_particles.mean(0), state_particles.std(0)

    @abc.abstractmethod
    def _local_assimilation_update(
        self, z_forecast, x_forecast, x_observed, obs_noise_std, localisation_weights
    ):
        """Perform a local analysis update for the state at a grid point.

        Args:
            z_forecast (array): Two-dimensional array of shape
                `(n_particles, dim_z_local)` where `n_particles` is the number
                of particles in the ensemble and `dim_z_local` is the dimension
                of the local state at each spatial location / grid point, with
                each row the local state values of an ensemble member.
            x_forecast (array): Two-dimensional array of shape
                `(n_particles, dim_x_local)` where `n_particles` is the number
                of particles in the ensemble and `dim_x_local` is the dimension
                of the vector of observations local to the current state
                spatial location / grid point, with each row the forecasted
                local observation values for an ensemble member.
            x_observed (array): One-dimensional array of shape `(dim_x_local)`
                where `dim_x_local` is the dimension of the vector of
                observations local to the current state spatial location /
                grid point, with entries corresponding to the local values of
                the observations at the current time point.
            obs_noise_std (array): One-dimensional array of shape
                `(dim_x_local)` where `dim_x_local` is the dimension of the
                vector of observations local to the current state spatial
                location / grid point, with entries corresponding to the
                standard deviations of each local observed variable given the
                current state variable values.
            localisation_weights (array): One-dimensional array of shape
                `(dim_x_local)` where `dim_x_local` is the dimension of the
                vector of observations local to the current state spatial
                location / grid point, with entries corresponding to weights
                for each local observed variable in [0, 1] to modulate the
                strength of the effect of each local observation on the
                updated state values based on the distance between the state
                spatial location / grid point and observation point.

        Returns:
            Two-dimensional array of shape `(n_particles, dim_z_local)` where
            `n_particles` is the number of particles in the ensemble and
            `dim_z_local` is the dimension of the local state at each spatial
            location / grid point, with each row the local updated analysis
            state values of each ensemble member.
        """


class LocalEnsembleTransformParticleFilter(AbstractLocalEnsembleFilter):
    def __init__(
        self,
        n_grid,
        localisation_func,
        inflation_factor=1.0,
        ot_solver=optimal_transport.solve_optimal_transport_exact,
        ot_solver_params={},
    ):
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
            n_grid=n_grid, localisation_func=localisation_func,
        )
        self.inflation_factor = inflation_factor
        self.ot_solver = ot_solver
        self.ot_solver_params = ot_solver_params

    def _local_assimilation_update(
        self, z_forecast, x_forecast, x_observed, obs_noise_std, localisation_weights
    ):
        n_particles = z_forecast.shape[0]
        dx_error = x_forecast - x_observed
        log_particle_weights = -0.5 * (
            dx_error * (localisation_weights / obs_noise_std ** 2) * dx_error
        ).sum(-1)
        target_dist = np.exp(log_particle_weights - logsumexp(log_particle_weights))
        target_dist /= target_dist.sum()
        source_dist = np.ones(n_particles) / n_particles
        cost_matrix = optimal_transport.pairwise_euclidean_distance(
            z_forecast, z_forecast
        )
        trans_matrix = n_particles * self.ot_solver(
            source_dist, target_dist, cost_matrix, **self.ot_solver_params
        )
        z_analysis = trans_matrix.dot(z_forecast)
        if self.inflation_factor > 1.0:
            z_analysis_mean = z_analysis.mean(0)
            dz_analysis = z_analysis - z_analysis_mean
            return z_analysis_mean + dz_analysis * self.inflation_factor
        else:
            return z_analysis


class ScalableLocalEnsembleTransportParticleFilter(AbstractEnsembleFilter):
    """Scalable local ensemble transport particle filter."""

    def __init__(
        self,
        init_state_sampler,
        next_state_sampler,
        log_obs_dens_per_loc_func,
        obs_coords,
        loc_func,
        loc_radius,
        pou,
        calculate_cost_matrices_func,
        ot_solver,
        ot_solver_params={},
        target_dist_callback=None,
    ):
        super().__init__(init_state_sampler, next_state_sampler, rng)
        self.log_obs_dens_per_loc_func = log_obs_dens_per_loc_func
        self.loc_func = loc_func
        self.pou = pou
        self.ot_solver = ot_solver
        self.ot_solver_params = ot_solver_params
        self.loc_kernel_obs_coord = np.stack(
            [
                loc_func(pou.patch_distance(p, obs_coords), loc_radius)
                for p in range(pou.n_patch)
            ],
            -1,
        )
        self.calculate_cost_matrices_func = calculate_cost_matrices_func
        self.target_dist_callback = target_dist_callback

    def analysis_update(self, z_forecast, x_observed, time_index):
        n_particle = z_forecast.shape[0]
        log_obs_dens_per_loc = self.log_obs_dens_per_loc_func(
            z_forecast, x_observed, time_index
        )
        log_target_dists = log_obs_dens_per_loc.dot(self.loc_kernel_obs_coord)
        log_target_dists -= logsumexp(log_target_dists, axis=0)
        target_dists = np.exp(log_target_dists.T)
        target_dists /= target_dists.sum(-1)[:, None]
        if self.target_dist_callback is not None:
            self.target_dist_callback(time_index, target_dists)
        source_dists = np.ones_like(target_dists) / n_particle
        z_forecast = z_forecast.reshape((z_forecast.shape[0], -1, self.pou.n_node))
        if self.calculate_cost_matrices_func is not None:
            cost_matrices = self.calculate_cost_matrices_func(z_forecast)
        else:
            z_dist_matrices = np.sum(
                (z_forecast[:, None] - z_forecast[None, :]) ** 2, -2
            )
            cost_matrices = self.pou.split_into_patches(z_dist_matrices).sum(-1)
            cost_matrices = np.moveaxis(cost_matrices, 2, 0)
        trans_matrices = (
            self.ot_solver(
                source_dists, target_dists, cost_matrices, **self.ot_solver_params
            )
            * n_particle
        )
        scaled_z_forecast_patches = self.pou.split_into_patches_and_scale(z_forecast)
        z_analysis_patches = np.einsum(
            "kij,jlkm->ilkm", trans_matrices, scaled_z_forecast_patches
        )
        z_analysis = self.pou.combine_patches(z_analysis_patches).reshape(
            (z_forecast.shape[0], -1)
        )
        z_analysis_mean = z_analysis.mean(0)
        return z_analysis, z_analysis_mean, z_analysis.std(0)


class LocalEnsembleTransformKalmanFilter(AbstractLocalEnsembleFilter):
    """
    Localised ensemble transform Kalman filter for spatially extended models.

    References:
        1. Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007).
           Efficient data assimilation for spatiotemporal chaos:
           A local ensemble transform Kalman filter.
           Physica D: Nonlinear Phenomena, 230(1), 112-126.
    """

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_func, obser_noise_std, n_grid, localisation_func,
                 rng=None, inflation_factor=1.):
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
            rng (RandomState): Numpy RandomState random number generator.
            inflation_factor (float): A value greater than or equal to one used
                to inflate the analysis ensemble on each update as a heuristic
                to overcome the underestimation of the uncertainty in the
                system state by ensemble Kalman filter methods.
        """
        super(LocalEnsembleTransformKalmanFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler,
                observation_func=observation_func,
                obser_noise_std=obser_noise_std,
                n_grid=n_grid, localisation_func=localisation_func, rng=rng
        )
        self.inflation_factor = inflation_factor

    def local_analysis_update(self, z_forecast, x_forecast, x_observed,
                              obs_noise_std, localisation_weights):
        # Number of particles
        n_p = z_forecast.shape[0]
        # Compute local state ensemble mean vector and deviations matrix
        z_mean_forecast = z_forecast.mean(0)
        dz_forecast = z_forecast - z_mean_forecast
        # Compute local observation ensemble mean vector and deviations matrix
        x_mean_forecast = x_forecast.mean(0)
        dx_forecast = x_forecast - x_mean_forecast
        # Compute reciprocal of effective per observation standard variances
        # by scaling by the inverse variances by the localisation weights
        eff_inv_obs_var = localisation_weights / obs_noise_std**2
        # The local analysis covariance in the reduced n_p dimensional subspace
        # spanned by the the ensemble members (denoted $\tilde{\mathbf{P}}^a$
        # in Hunt et al. (2007)) is calculated as the inverse of
        #    identity(n_p) * (n_p - 1) / inflation_factor +
        #    dx_forecast @ diag(eff_inv_obs_var) @ dx_forecast.T
        # where identity(n_p) is the n_p dimensional identity matrix. If we
        # calculate a singular value decomposition
        #    u, s, vh = svd(dx_forecast @ diag(eff_inv_obs_var**0.5))
        # then u corresponds to a set of orthonormal eigenvectors for this
        # local analysis covariance and
        #    (ones(n_p) * (n_p - 1) / inflation_factor +
        #     concatenate([s**2, zeros(n_p - n_o)]))**(-1)
        # to a vector of eigenvalues of the reduced subspace local analysis
        # covariance matrix, where n_o is the number of local observations.
        eigvec_p, sing_val, _ = la.svd(dx_forecast * eff_inv_obs_var**0.5)
        eigval_p_inv = np.ones(n_p) * (n_p - 1) / self.inflation_factor
        eigval_p_inv[:dx_forecast.shape[1]] += sing_val**2
        eigval_p = 1. / eigval_p_inv
        # The 'deviations' (from mean) of the n_particles * n_particles
        # matrix used to weight the forecast state ensemble deviations when
        # calculating the analysis state ensemble is calculated as a scaled
        # symmetric matrix square root of the local analysis covariance matrix
        # an eigendecomposition was calculated for above
        dw_matrix = (n_p - 1)**0.5 * (eigvec_p * eigval_p**0.5).dot(eigvec_p.T)
        # Mean of weightings matrix rows is calculated from the observed data
        # and the inverse of the local analysis covariance which we have an
        # eigendecomposition for
        d_vector = (dx_forecast * eff_inv_obs_var).dot(
            x_observed - x_mean_forecast)
        w_mean = eigvec_p.dot(eigval_p * eigvec_p.T.dot(d_vector))
        # Calculate weighting matrix by adding mean vector to each row of the
        # weighting deviations matrix
        w_matrix = w_mean[None] + dw_matrix
        # Local analysis state ensemble calculated as a weighted linear
        # combination of local forecast state ensemble deviations shifted by
        # the local forecast state mean
        return z_mean_forecast + w_matrix.dot(dz_forecast)
