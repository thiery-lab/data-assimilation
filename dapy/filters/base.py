"""Base classes for ensemble filters implementing common interface."""

import abc
from typing import Sequence, Dict
import numpy as np
from numpy.random import Generator
from dapy.models.base import AbstractModel
import tqdm.auto as tqdm


class AbstractEnsembleFilter(abc.ABC):
    """Abstract base class for ensemble filters defining standard interface."""

    @abc.abstractmethod
    def _assimilation_update(
        self,
        model: AbstractModel,
        rng: Generator,
        state_particles: np.ndarray,
        observation: np.ndarray,
        time_index: int,
    ) -> np.ndarray:
        """Adjust state particle ensemble for observations at current time index.

        The assimilation update transforms the prior predicted empirical distribution
        represented by the state particle ensemble to a particle ensemble corresponding
        to an empricial estimate of the posterior (filtering) distribution given the
        observations at the current time index.

        Args:
            model: State-space model to perform assimilation update with.
            rng: NumPy random number geneator.
            state_particles: Two-dimensional array of shape
                `(num_particle, dim_state)` with each row a state particle generated
                by simulating model dynamics forward from analysis ensemble at
                previous time step.
            observation: Observations at current time index `time_index`.
            time_index: Current time index to assimilate the observations for.

        Returns:
            post_state_particles (array): Two-dimensional array of shape
                `(n_particle, dim_z)` with each row a state particle in
                analysis ensemble.
            post_state_mean (array): One-dimensional array of shape `(dim_z, )`
                corresponding to estimated mean of state analysis distribution.
            post_state_std (array): One-dimensional array of shape `(dim_z, )`
                corresponding to estimated per-dimension standard deviations
                of analysis distribution.
        """

    def filter(
        self,
        model: AbstractModel,
        observation_sequence: np.ndarray,
        observation_time_indices: Sequence[int],
        num_particle: int,
        rng: Generator,
        return_particles: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute particle ensemble approximations of filtering distributions.

        Args:
            model: Generative state-space model for observations.
            observation_sequence: Observation sequence with shape
                `(num_observation_time, dim_observation)` where `num_observation_time`
                is the number of observed time indices in the sequence and
                `dim_observation` is dimensionality of the observations.
            observation_time_indices: Sequence of time (step) indices at which state is
                observed. The sequence elements must be integers. The length of the
                sequence must correspond to the number of observation times
                `num_observation_time` represented in the `observation_sequences_array`.
            num_particle: Number of particles to use in ensembles used to approximate
                filtering distributions at each time index.
            rng: NumPy random number generator object.
            return_particles: Whether to return two-dimensional array of shape
                `(num_observation_time, num_particle, dim_state)` containing all state
                particles at each observation time index. Potentially memory-heavy for
                for models with large state dimensions.
        Returns:
            Dictionary containing arrays of filtering distribution parameters -
                state_mean_sequence: Array of filtering distribution means at all
                    observation time indices. Shape `(num_observation_time, dim_state)`.
                state_std_sequence: Array of filtering distribution standard deviations
                    at all at all observation time indices. Shape
                    `(num_observation_time, dim_state)`.
                state_particles_sequence: Array of state particle ensemble
                    approximations to filtering distributions at at all observation time
                    indices. Only returned if `return_particles == True`. Shape
                    `(num_observation_time, num_particle, dim_state)`.
        """
        num_obs_time, dim_observation = observation_sequence.shape
        observation_time_indices = np.sort(observation_time_indices)
        num_observation_time = len(observation_time_indices)
        assert observation_sequence.shape[0] == num_observation_time
        num_step = observation_time_indices[-1]
        state_mean_sequence = np.full((num_observation_time, model.dim_state), np.nan)
        state_std_sequence = np.full((num_observation_time, model.dim_state), np.nan)
        if return_particles:
            state_particles_sequence = np.full(
                (num_obs_time, num_particle, model.dim_state), np.nan
            )
        for s in tqdm.trange(num_step + 1, desc="Filtering", unit="time steps"):
            if s == 0:
                state_particles = model.sample_initial_state(rng, num_particle)
                t = 0
            else:
                state_particles = model.sample_state_transition(rng, state_particles, s)
            if s == observation_time_indices[t]:
                state_particles, state_mean, state_std = self._assimilation_update(
                    model, rng, state_particles, observation_sequence[t], s
                )
                state_mean_sequence[t] = state_mean
                state_std_sequence[t] = state_std
                if return_particles:
                    state_particles_sequence[t] = state_particles
                t += 1
        results = {
            "state_mean_sequence": state_mean_sequence,
            "state_std_sequence": state_std_sequence,
        }
        if return_particles:
            results["state_particles_sequence"] = state_particles_sequence
        return results
