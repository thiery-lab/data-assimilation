"""Base class for filters implementing common interfaces."""

import numpy as np


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
             Two-dimensional array of shape `(n_particle, dim_z)` with each
             row a state particle in analysis ensemble.
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
            z_analysis = self.analysis_update(z_forecast, x_observed_seq[t], t)
            # Record updated ensemble statistics.
            z_mean_seq[t] = z_analysis.mean(0)
            z_std_seq[t] = z_analysis.std(0)
            if return_particles:
                z_particles_seq[t] = z_analysis
        results = {'z_mean_seq': z_mean_seq, 'z_std_seq': z_std_seq}
        if return_particles:
            results['z_particles_seq'] = z_particles_seq
        return results
