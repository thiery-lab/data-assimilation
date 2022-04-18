"""Abstract base classes for ensemble filters for inference in state space models."""

import abc
from typing import Sequence, Dict
import numpy as np
from numpy.random import Generator
from dapy.models.base import AbstractModel
from dapy.utils.progressbar import ProgressBar


class AbstractEnsembleFilter(abc.ABC):
    """Abstract base class for ensemble filters defining standard interface.

    The filtering distribution at each observation time index is approximated by
    an ensemble of state particles, with the particles sequentially updated by
    alternating prediction updates propagating the particles forward in time under
    the model dynamics and assimilation updates which transform particles representing
    a prior predictive distribution to a posterior filtering distribution given the
    observations at the current time index.
    """

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

        The assimilation update transforms the prior predictive empirical distribution
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

    def _perform_model_specific_initialization(
        self, model: AbstractModel, num_particle: int
    ):
        """Hook to allow performing model-specific initialization before filtering."""

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
        results = {}
        if return_particles:
            results["state_particles_sequence"] = np.full(
                (num_obs_time, num_particle, model.dim_state), np.nan
            )
        self._perform_model_specific_initialization(model, num_particle)
        with ProgressBar(range(num_step + 1), "Filtering", unit="time-steps") as pb:
            for s in pb:
                if s == 0:
                    state_particles = model.sample_initial_state(rng, num_particle)
                    t = 0
                else:
                    state_particles = model.sample_state_transition(
                        rng, state_particles, s
                    )
                if s == observation_time_indices[t]:
                    state_particles, statistics = self._assimilation_update(
                        model, rng, state_particles, observation_sequence[t], s
                    )
                    for key, statistic_array in statistics.items():
                        if t == 0:
                            results[key + "_sequence"] = np.full(
                                (num_observation_time,) + statistic_array.shape, np.nan
                            )
                        results[key + "_sequence"][t] = statistic_array
                    if return_particles:
                        results["state_particles_sequence"][t] = state_particles
                    t += 1
        return results
