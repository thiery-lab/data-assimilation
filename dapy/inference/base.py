"""Base classes for ensemble filters implementing common interface."""

import numpy as np
from dapy.utils.doc import inherit_docstrings


class AbstractEnsembleFilter(object):
    """Abstract base class for ensemble filters defining standard interface."""

    def __init__(self, init_state_sampler, next_state_sampler, rng):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as arguments.
            rng (RandomState): Numpy RandomState random number generator.
        """
        self.init_state_sampler = init_state_sampler
        self.next_state_sampler = next_state_sampler
        self.rng = rng

    def analysis_update(self, z_forecast, x_observed, time_index):
        """Perform analysis update to forecasted states given observations.

        Args:
            z_forecast (array): Two-dimensional array of shape
                `(n_particle, dim_z)` with each row a state particle generated
                by simulating model dynamics forward from analysis ensemble at
                previous time step.
            x_observed (array): One-dimensional array of shape `(dim_x, )`
                corresponding to current observations vector.
            time_index (integer): Current time index.

        Returns:
            z_analysis (array): Two-dimensional array of shape
                `(n_particle, dim_z)` with each row a state particle in
                analysis ensemble.
            z_analysis_mean (array): One-dimensional array of shape `(dim_z, )`
                corresponding to estimated mean of state analysis distribution.
            z_analysis_std (array): One-dimensional array of shape `(dim_z, )`
                corresponding to estimated per-dimension standard deviations
                of analysis distribution.
        """
        raise NotImplementedError()

    def filter(self, x_observed_seq, n_particles, return_particles=False):
        """Compute filtering distribution approximations.

        Args:
            x_observed_seq (array): Observed state sequence with shape
                `(n_steps, dim_x)` where `n_steps` is number of time steps in
                sequence and `dim_x` is dimensionality of observations.
            n_particles (integer): Number of particles to use to represent
                filtering distribution at each time step.
            return_particles (boolean): Whether to return two-dimensional
                array of shape `(n_steps, n_particles, dim_z)` containing all
                state particles at each time step. Potentially memory-heavy
                for system with large state dimensions.

        Returns:
            Dictionary containing arrays of filtering density parameters -
                z_mean_seq: Array of filtering density mean for each
                    dimension at all time steps.
                z_std_seq: Array of filtering density standard deviation for
                    each dimension at all time steps.
                z_particles_seq: Array of particles representing filtering
                    distribution at each time step (if return_particles==True).
        """
        n_steps, dim_x = x_observed_seq.shape
        for t in range(n_steps):
            # Forecast update.
            if t == 0:
                z_forecast = self.init_state_sampler(n_particles)
                dim_z = z_forecast.shape[1]
                z_mean_seq = np.full((n_steps, dim_z), np.nan)
                z_std_seq = np.full((n_steps, dim_z), np.nan)
                if return_particles:
                    z_particles_seq = np.full(
                        (n_steps, n_particles, dim_z), np.nan)
            else:
                z_forecast = self.next_state_sampler(z_analysis, t-1)
            # Analysis update.
            z_analysis, z_analysis_mean, z_analysis_std = self.analysis_update(
                z_forecast, x_observed_seq[t], t)
            # Record updated ensemble statistics.
            z_mean_seq[t] = z_analysis_mean
            z_std_seq[t] = z_analysis_std
            if return_particles:
                z_particles_seq[t] = z_analysis
        results = {'z_mean_seq': z_mean_seq, 'z_std_seq': z_std_seq}
        if return_particles:
            results['z_particles_seq'] = z_particles_seq
        return results


@inherit_docstrings
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

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_func, obser_noise_std, n_grid, localisation_func,
                 rng):
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
        """
        super(AbstractLocalEnsembleFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler, rng=rng
        )
        self.observation_func = observation_func
        self.obser_noise_std = obser_noise_std
        self.n_grid = n_grid
        self.localisation_func = localisation_func

    def analysis_update(self, z_forecast, x_observed, time_index):
        n_particles = z_forecast.shape[0]
        z_forecast_grid = z_forecast.reshape((n_particles, -1, self.n_grid))
        x_forecast = self.observation_func(z_forecast, time_index)
        z_analysis_grid = np.empty(z_forecast_grid.shape)
        for grid_index in range(self.n_grid):
            obs_indices, obs_weights = self.localisation_func(grid_index)
            z_forecast_local = z_forecast_grid[:, :, grid_index]
            x_forecast_local = x_forecast[:, obs_indices]
            x_observed_local = x_observed[obs_indices]
            obs_noise_std_local = self.obser_noise_std[obs_indices]
            z_analysis_grid[:, :, grid_index] = self.local_analysis_update(
                z_forecast_local, x_forecast_local, x_observed_local,
                obs_noise_std_local, obs_weights)
        z_analysis = z_analysis_grid.reshape((n_particles, -1))
        return z_analysis, z_analysis.mean(0), z_analysis.std(0)

    def local_analysis_update(self, z_forecast, x_forecast, x_observed,
                              obs_noise_std, localisation_weights):
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
        raise NotImplementedError()
