"""Abstract base classes for ensemble filters for inference in state space models."""

import abc
from typing import Sequence, Dict, Tuple
import numpy as np
from numpy.typing import ArrayLike
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
        previous_states: ArrayLike,
        predicted_states: ArrayLike,
        observation: ArrayLike,
        time_index: int,
    ) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """Adjust state particle ensemble for observations at current time index.

        The assimilation update transforms a particle ensemble `predicted_states`
        corresponding to an empirical estimate of the predictive distribution at time
        index `time_index` for state space model `model` to an empirical estimate of
        the filtering distribution at the same time index given the observation vector
        at this time index `observation`.

        Args:
            model: State-space model to perform assimilation update with.
            rng: NumPy random number generator to use for sampling any random variates.
            previous_states: Two-dimensional array of shape `(num_particle, dim_state)`
                with each row a state particle in ensemble representing empirical
                estimate of filtering distribution at time index `time_index - 1`.
            predicted_states: Two-dimensional array of shape
                `(num_particle, dim_state)` with each row a state particle in ensemble
                representing empirical estimate of predictive distribution at time index
                `time_index`.
            observation: Observations at time index `time_index`.
            time_index: Time index to assimilate the observations for.

        Returns:
            updated_states: Two-dimensional array of shape `(num_particle, dim_state)`
                with each row a state particle in ensemble representing empirical
                estimate of filtering distribution at time index `time_index`.
            statistics: Dictionary of ensemble statistics with string keys and array
                values.
        """

    def _perform_model_specific_initialization(
        self, model: AbstractModel, num_particle: int
    ):
        """Hook to allow performing model-specific initialization before filtering."""

    def filter(
        self,
        model: AbstractModel,
        observation_sequence: ArrayLike,
        observation_time_indices: Sequence[int],
        num_particle: int,
        rng: Generator,
        return_particles: bool = False,
    ) -> Dict[str, ArrayLike]:
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
                `num_observation_time` represented in the `observation_sequence`.
            num_particle: Number of particles to use in ensembles used to approximate
                filtering distributions at each time index.
            rng: NumPy random number generator object to use to sample random variates.
            return_particles: Whether to return two-dimensional array of shape
                `(num_observation_time, num_particle, dim_state)` containing all state
                particles at each observation time index. Potentially memory-heavy for
                for models with large state dimensions.
        Returns:
            Dictionary containing arrays of filtering distribution statistics -
                state_mean_sequence: Array of filtering distribution means at all
                    observation time indices. Shape `(num_observation_time, dim_state)`.
                state_std_sequence: Array of filtering distribution standard deviations
                    at all at all observation time indices. Shape
                    `(num_observation_time, dim_state)`.
                state_particles_sequence: Array of state particles representing
                    empirical estimates of filtering distributions at at all observation
                     time indices. Only returned if `return_particles == True`. Shape
                    `(num_observation_time, num_particle, dim_state)`.
        """
        observation_time_indices = np.sort(observation_time_indices)
        num_observation_time = len(observation_time_indices)
        assert observation_sequence.shape[0] == num_observation_time
        num_step = observation_time_indices[-1]
        results = {}
        if return_particles:
            results["state_particles_sequence"] = np.full(
                (num_observation_time, num_particle, model.dim_state), np.nan
            )
        self._perform_model_specific_initialization(model, num_particle)
        observation_index = 0
        states = None
        with ProgressBar(range(num_step + 1), "Filtering", unit="time-steps") as pb:
            for time_index in pb:
                previous_states = states
                if time_index == 0:
                    predicted_states = model.sample_initial_state(rng, num_particle)
                else:
                    predicted_states = model.sample_state_transition(
                        rng, previous_states, time_index - 1
                    )
                if time_index == observation_time_indices[observation_index]:
                    states, statistics = self._assimilation_update(
                        model,
                        rng,
                        previous_states,
                        predicted_states,
                        observation_sequence[observation_index],
                        time_index,
                    )
                    for key, statistic_array in statistics.items():
                        if observation_index == 0:
                            results[key + "_sequence"] = np.full(
                                (num_observation_time,) + statistic_array.shape, np.nan
                            )
                        results[key + "_sequence"][observation_index] = statistic_array
                    if return_particles:
                        results["state_particles_sequence"][observation_index] = states
                    observation_index += 1
                else:
                    states = predicted_states
        return results
