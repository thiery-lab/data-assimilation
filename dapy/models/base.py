"""Abstract base classes for state-space models."""

import abc
from typing import Optional, Union, Sequence
from numbers import Number
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from numpy.random import Generator
from dapy.utils.progressbar import ProgressBar


class DensityNotDefinedError(Exception):
    """Raised on calling a method to evaluate a undefined probability density.

    For some models the state transition or observation given state conditional
    probability densities will not be definedm for example in the case of
    deterministic state transitions or observations.
    """


class AbstractModel(abc.ABC):
    """Abstract state space model base class.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = model.sample_initial_state(rng)
                t = 0
            else:
                state_sequence[s] = model.sample_state_transition(
                    rng, state_sequence[s - 1], s - 1)
            if s == observation_time_indices[t]:
                observation_sequence[t] = model.sample_observation_given_state(
                    rng, state_sequence[s], s)
                t += 1

    where `observation_time_indices` is a sequence of integer time indices specifying
    the observation times, `num_step = max(observation_time_indices)` and `rng` is
    a random number generator used to generate the required random variates.
    """

    def __init__(self, dim_state: int, dim_observation: int, **kwargs):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_observation: Dimension of observation vector.
        """
        self._dim_state = dim_state
        self._dim_observation = dim_observation
        super().__init__(**kwargs)

    @property
    def dim_state(self) -> int:
        """Dimension of model state vector."""
        return self._dim_state

    @property
    def dim_observation(self) -> int:
        """Dimension of observation vector."""
        return self._dim_observation

    def sample_initial_state(
        self, rng: Generator, num_state: Optional[int] = None
    ) -> np.ndarray:
        """Independently sample initial state(s) at time index 0.

        Args:
            rng: NumPy random number generator.
            num_state: Number of states to sample. If `None` a one-dimensional state
                array corresponding to a single sample is returned.

        Returns:
            Array containing independent initial state sample(s). Either of shape
            `(num_state, dim_state)` if `num_state` is not `None` or of shape
            `(dim_state,)` otherwise.
        """
        if num_state is None:
            num_state = 1
            num_state_was_none = True
        else:
            num_state_was_none = False
        initial_states = self._sample_initial_state(rng, num_state)
        if num_state_was_none:
            return initial_states[0]
        else:
            return initial_states

    @abc.abstractmethod
    def _sample_initial_state(self, rng: Generator, num_state: int) -> np.ndarray:
        """Internal implementation of `sample_initial_state` method.

        Unlike `sample_initial_state` only accepts / returns 2D arrays.

        Args:
            rng: NumPy random number generator.
            num_state: Number of states to sample.

        Returns:
            Array containing independent initial state samples of shape
            `(num_state, dim_state)`.
        """

    def sample_state_transition(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Independently sample next state(s) given current state(s).

        Args:
            rng: NumPy random number generator.
            states: Array of model state(s) at time index `time_index`. Either of shape
                `(num_state, dim_state)` if multiple states are to be propagated or
                shape `(dim_state,)` if a single state is to be propagated.
            time_index: Current time index.

        Returns:
            Array containing samples of state(s) at time index `time_index + 1`. Either
            of shape `(num_state, dim_state)` if multiple states were propagated or
            shape `(dim_state,)` if a single state was propagated.
        """
        if states.ndim == 1:
            states = states[None]
            states_was_one_dimensional = True
        else:
            states_was_one_dimensional = False
        next_states = self._sample_state_transition(rng, states, time_index)
        if states_was_one_dimensional:
            return next_states[0]
        else:
            return next_states

    @abc.abstractmethod
    def _sample_state_transition(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Internal implementation of `sample_state_transition` method.

        Unlike `sample_state_transition` only accepts / returns 2D arrays.

        Args:
            rng: NumPy random number generator.
            states: Array of model states at time index `time_index`  of shape
                `(num_state, dim_state)`.
            time_index: Current time index.

        Returns:
            Array containing samples of states at time index `time_index + 1` of shape
            `(num_state, dim_state)`.
        """

    def sample_observation_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Independently sample observation(s) given current state(s).

        Args:
            rng: NumPy random number generator.
            states: Array of model state(s) at time index `time_index`. Either of shape
                `(num_state, dim_state)` if multiple observations are to be generated or
                shape `(dim_state,)` if a single observation is to be generated.
            time_index: Current time index.

        Returns:
            Array containing samples of observations(s) at time index `time_index`.
            Either of shape `(num_state, dim_observation)` if multiple observations were
            generated or shape `(dim_observation,)` if a single observation was
            generated.
        """
        if states.ndim == 1:
            states = states[None]
            states_was_one_dimensional = True
        else:
            states_was_one_dimensional = False
        observations = self._sample_observation_given_state(rng, states, time_index)
        if states_was_one_dimensional:
            return observations[0]
        else:
            return observations

    @abc.abstractmethod
    def _sample_observation_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Internal implementation of `sample_observation_given_state` method.

        Unlike `sample_observation_given_state` only accepts / returns 2D arrays.

        Args:
            rng: NumPy random number generator.
            states: Array of model states at time index `time_index` of shape
                `(num_state, dim_state)`.
            time_index: Current time index.

        Returns:
            Array containing samples of observations at time index `time_index` of shape
            `(num_state, dim_observation)`.
        """

    def sample_state_and_observation_sequences(
        self,
        rng: Generator,
        observation_time_indices: Sequence[int],
        num_sample: Optional[int] = None,
        return_states_at_all_times: bool = False,
    ) -> np.ndarray:
        """Generate state and observation sequences from model.

        Args:
            rng: NumPy random number generator.
            observation_time_indices: Sequence of time indices at which state is
                observed. The sequence elements must be integers. The maximal element
                of the sequence corresponds to the total number of time steps to
                simulate `num_step`, while the length of the sequence corresponds to
                the number of observation times `num_observation_time`.
            num_sample: Number of independent sequence pairs to sample. If equal
                to `None` (default) a single sample is generated.
            return_states_at_all_times: Whether to return states at all time indices
                including those not corresponding to observation times (`True`) or to
                only return the sequence(s) of states at the observation time indices
                (`False`).

        Returns:
            state_sequences: Generated state sequence(s) of shape
                `(num_state_time, dim_state)` if `num_sample` is None or
                `(num_state_time, num_sample, dim_state)` otherwise where
                `num_state_time = num_step + 1` if `return_state_at_all_times == True`
                and `num_state_time = num_observation_time` otherwise.
            observation_sequences: Generated observation sequence(s) of shape
                `(num_observation_time, dim_observation)` if `num_sample` is None or
                `(num_observation_time, num_sample, dim_observation)` otherwise.
        """
        observation_time_indices = np.sort(observation_time_indices)
        num_step = observation_time_indices[-1]
        num_observation_time = len(observation_time_indices)
        num_state_time = (
            num_step + 1 if return_states_at_all_times else num_observation_time
        )
        if num_sample is None:
            state_sequences = np.full((num_state_time, self.dim_state), np.nan)
            observation_sequences = np.full(
                (num_observation_time, self.dim_observation), np.nan
            )
        else:
            state_sequences = np.full(
                (num_state_time, num_sample, self.dim_state), np.nan
            )
            observation_sequences = np.full(
                (num_observation_time, num_sample, self.dim_observation), np.nan
            )
        with ProgressBar(range(num_step + 1), "Sampling", unit="time-steps") as pb:
            for s in pb:
                if s == 0:
                    states = self.sample_initial_state(rng, num_sample)
                    t = 0
                else:
                    states = self.sample_state_transition(rng, states, s - 1)
                if return_states_at_all_times:
                    state_sequences[s] = states
                if s == observation_time_indices[t]:
                    if not return_states_at_all_times:
                        state_sequences[t] = states
                    observation_sequences[t] = self.sample_observation_given_state(
                        rng, states, s
                    )
                    t += 1
            return state_sequences, observation_sequences

    @abc.abstractmethod
    def log_density_initial_state(self, states: np.ndarray) -> np.ndarray:
        """Evaluate the log probability densities of initial states at time index 0.

        Args:
            states: Array of model state(s) at time index 0. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                states or shape `(dim_state,)` if density is to be evaluated for a
                single state.

        Returns:
            Array of log probability densities for each state of shape `(num_state,)` or
            a scalar / array of shape `()` if a single state is provided.
        """

    @abc.abstractmethod
    def log_density_state_transition(
        self, next_states: np.ndarray, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Evaluate the log probability densities of transitions between states.

        Args:
            next_states: Array of model state(s) at time index `time_indext + 1`. Either
                of shape `(num_state, dim_state)` if the density is to be evalulated at
                multiple state pairs or shape `(dim_state,)` if density is to be
                evaluated for a single state pair.
            states: Array of model state(s) at time index `time_index`. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                state pairs or shape `(dim_state,)` if density is to be evaluated for a
                single state pair.
            time_index: Current time index.

        Returns:
            Array of log conditional probability densities of next state given current
            state for each state pair of shape `(num_state,)` or a scalar / array of
            shape `()` if a single state pair is provided.
        """

    @abc.abstractmethod
    def log_density_observation_given_state(
        self, observations: np.ndarray, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Evaluate log probability densities of observations given states.

        Args:
            observations: Array of model observation(s) at time index `time_index`.
                Either of shape `(num_state, dim_observation)` if density is to be
                evalulated at multiple observations or shape `(dim_observation,)` if
                density is to be evaluated for a single observation.
            states: Array of model state(s) at time index `time_index`. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                observations or shape `(dim_state,)` if density is to be evaluated for
                a single observation.
            time_index: Current time index.

        Returns:
            Array of log conditional probability densities of observation given state
            for each state and observation pair of shape `(num_state,)` or a scalar /
            array of shape `()` if a single state and observation pair is provided.
        """

    def log_density_state_sequence(self, state_sequences: np.ndarray) -> np.ndarray:
        """Evaluate log probability densities of state sequences.

        Args:
           state_sequences: Array of model state sequence(s). Either of shape
               `(num_step + 1, num_sequence, dim_state)` if density is to be evaluated
               for multiple sequences or shape `(num_step + 1, dim_state)` if density
               is to be evaluated for a single sequence.

        Returns:
            Array of log joint probability densities of state sequences of shape
            `(num_sequence,)` or a scalar / array of shape `()` if a single state
            sequence is provided.
        """
        log_density = self.log_density_initial_state(state_sequences[0])
        for t in range(1, state_sequences.shape[0]):
            log_density += self.log_density_state_transition(
                state_sequences[t], state_sequences[t - 1], t - 1
            )
        return log_density

    def log_density_state_and_observation_sequence(
        self,
        state_sequence: np.ndarray,
        observation_sequence: np.ndarray,
        observation_time_indices: Sequence[int],
    ) -> np.ndarray:
        """Evaluate log probability densities of state and observation sequences.

        Args:
            state_sequences: Array of model state sequence(s). Either of shape
                `(num_step + 1, num_sequence, dim_state)` if density is to be evaluated
                form multiple sequences or shape `(num_step + 1, dim_state)` if density
                is to be evaluated for a single sequence.
            observation_sequences: Array of model observation sequence(s). Either of
                shape `(num_observation_time, num_sequence, dim_observation)` if density
                is to be evaluated for multiple sequences or shape
                `(num_observation_time, dim_observation)` if density is to be evaluated
                for a single sequence.
            observation_time_indices: Sequence of time (step) indices at which state is
                observed. The sequence elements must be integers. The maximal element of
                the sequence must corresponds to the total number of time steps
                `num_step` represented in the `state_sequences` array while the length
                of the sequence must correspond to the number of observation times
                `num_observation_time` represented in the `observation_sequences_array`.

        Returns:
            Array of log probability densities of state and observation sequences of
            shape `(num_sequence,)` or a scalar / array of shape `()` if a single state
            and observation sequence pair is provided.
        """
        observation_time_indices = np.sort(observation_time_indices)
        num_step = observation_time_indices[-1]
        num_observation_time = len(observation_time_indices)
        assert state_sequence.shape[0] == num_step + 1
        assert observation_sequence.shape[0] == num_observation_time
        for s in range(0, num_step + 1):
            if s == 0:
                log_density = self.log_density_initial_state(state_sequence[0])
                t = 0
            else:
                log_density += self.log_density_state_transition(
                    state_sequence[s], state_sequence[s - 1], s - 1
                )
            if s == observation_time_indices[t]:
                log_density += self.log_density_observation_given_state(
                    observation_sequence[t], state_sequence[s], s
                )
        return log_density


class AbstractAdditiveObservationNoiseModel(AbstractModel):
    """Abstract base class for state-space models with additive observation noise.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = model.sample_initial_state(rng)
                t = 0
            else:
                state_sequence[s] = model.sample_state_transition(
                    rng, state_sequence[s - 1], s - 1)
            if s == observation_time_indices[t]:
                observation_sequence[t] = (
                    model.observation_mean(state_sequence[s], s)
                    + model.sample_observation_noise_given_state(
                        rng, state_sequence[s], s)
                t += 1

    where `observation_mean` computes the mean of the observation given the current
    state and `sample_observation_noise_given_state` samples a zero-mean observation
    noise vector given the current state.
    """

    def observation_mean(self, states: np.ndarray, time_index: int) -> np.ndarray:
        """Computes mean of observation(s) given state(s) at time index `time_index`.

        Args:
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if the mean is to be evalulated for multiple
                observations or shape `(dim_state,)` if mean is to be evaluated for
                a single observation.
            time_index: Current time index.

        Returns:
            Array corresponding to mean of observations at time index `t`, of shape
            `(num_state, dim_observation)` if a 2D `states` array is passed or of shape
            `(dim_observation,`) otherwise.
        """
        return self._observation_mean(states, time_index)

    @abc.abstractmethod
    def _observation_mean(self, states: np.ndarray, time_index: int) -> np.ndarray:
        """Internal implementation of `observation_mean` method.

        Should be called in preference to `observation_mean` internally by other
        methods to allow use of mix-ins to change behaviour of `observation_mean`
        without affecting internal use.

        Args:
            states: Array of model state(s) at time index `t` of shape
                `(num_state, dim_state)` if the mean is to be evalulated for multiple
                observations or shape `(dim_state,)` if mean is to be evaluated for
                a single observation.
            time_index: Current time index.

        Returns:
            Array corresponding to mean of observations at time index `t`, of shape
            `(num_state, dim_observation)` if a 2D `states` array is passed or of shape
            `(dim_observation,`) otherwise.
        """

    def sample_observation_noise_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Independently sample observation noise vector(s) given current state(s).

        Args:
            rng: NumPy random number generator.
            states: Array of model state(s) at time index `time_index`. Either of shape
                `(num_state, dim_state)` if multiple observation noise vectors are to be
                generated or shape `(dim_state,)` if a single observation noise vector
                is to be generated.
            time_index: Current time index.

        Returns:
            Array containing samples of observations noise vector(s) at time index
            `time_index`. Either of shape `(num_state, dim_observation)` if multiple
            observation noise vectors were generated or shape `(dim_observation,)` if a
            single observation noise vector was generated.
        """
        if states.ndim == 1:
            states = states[None]
            states_was_one_dimensional = True
        else:
            states_was_one_dimensional = False
        observation_noise_vectors = self._sample_observation_noise_given_state(
            rng, states, time_index
        )
        if states_was_one_dimensional:
            return observation_noise_vectors[0]
        else:
            return observation_noise_vectors

    @abc.abstractmethod
    def _sample_observation_noise_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Internal implementation of `sample_observation_noise_given_state` method.

        Unlike `sample_observation_noise_given_state` only accepts / returns 2D arrays.

        Args:
            rng: NumPy random number generator.
            states: Array of model states at time index `time_index` of shape
                `(num_state, dim_state)`.
            time_index: Current time index.

        Returns:
            Array containing sample(s) of additive observation noise vector(s) at time
            index `time_index` of shape `(num_state, dim_observation)`.
        """

    def _sample_observation_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        return self._observation_mean(
            states, time_index
        ) + self._sample_observation_noise_given_state(rng, states, time_index)

    @abc.abstractmethod
    def log_density_observation_noise_given_state(
        self, observation_noise_vectors: np.ndarray, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Evaluate the log densities of observation noise vectors given states.

        Args:
            observation_noise_vectors: Array of additive observation noise vector(s) at
                time index `time_index`. Either of shape `(num_state, dim_observation)`
                if density is to be evalulated at multiple observation noise vectors or
                shape `(dim_observation,)` if density is to be evaluated for a single
                observation noise vector.
            states: Array of model state(s) at time index `time_index`. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                observation noise vectors or shape `(dim_state,)` if density is to be
                evaluated for a single observation noise vector.
            time_index: Current time index.

        Returns:
            Array of log conditional probability densities of observation noise vectors
            given state for each state and observation noise vector pair of shape
            `(num_state,)` or a scalar / array of shape `()` if a single state and
            observation noise vector pair is provided.
        """

    def log_density_observation_given_state(
        self, observations: np.ndarray, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        return self.log_density_observation_noise_given_state(
            observations - self._observation_mean(states, time_index),
            states,
            time_index,
        )


class AbstractAdditiveStateNoiseModel(AbstractModel):
    """Abstract base class for state-space models with additive state noise.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = model.sample_initial_state(rng)
                t = 0
            else:
                state_sequence[s] = (
                    model.next_state_mean(state_sequence[s - 1], s - 1)
                    + model.sample_state_noise_given_state(
                        rng, state_sequence[s - 1], s - 1)
                )
            if s == observation_time_indices[t]:
                observation_sequence[t] = (
                    model.observation_mean(state_sequence[s], s)
                    + model.sample_observation_noise_given_state(
                        rng, state_sequence[s], s)
                t += 1

    where `next_state_mean` computes the mean of the next_state given the current
    state and `sample_state_noise_given_state` samples a zero-mean state noise vector
    given the current state. In the case of deterministic state updates, i.e. zero state
    noise, the conditional distribution of the next state given the current state will
    be a Dirac measure with all mass located at the forward map of the current state
    through the state dynamics function and so will not have a density with respect to
    the Lebesgue measure.
    """

    def __init__(
        self, dim_state: int, dim_observation: int, deterministic_state_update: bool
    ):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_observation: Dimension of observation vector.
            deterministic_state_update: Whether state transitions are deterministic i.e.
                there is zero state noise.
        """
        self._deterministic_state_update = deterministic_state_update
        super().__init__(dim_state=dim_state, dim_observation=dim_observation)

    def next_state_mean(self, states: np.ndarray, time_index: int) -> np.ndarray:
        """Computes mean of next state(s) given current state(s).

        Implements determinstic component of state update dynamics with new state
        calculated by output of this function plus additive zero-mean Gaussian noise.
        For models with fully-determinstic state dynamics no noise is added so this
        function exactly calculates the next state.

        Args:
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if the mean is to be evalulated for multiple
                states or shape `(dim_state,)` if mean is to be evaluated for a single
                state.
            time_index: Current time index.

        Returns:
            Array corresponding to mean of states at time index `t + 1`, of shape
            `(num_state, dim_state)` if a 2D `states` array is passed or of shape
            `(dim_state,`) otherwise.
        """
        return self._next_state_mean(states, time_index)

    @abc.abstractmethod
    def _next_state_mean(self, states: np.ndarray, time_index: int) -> np.ndarray:
        """Internal implementation of `next_state_mean` method.

        Should be called in preference to `next_state_mean` internally by other
        methods to allow use of mix-ins to change behaviour of `next_state_mean`
        without affecting internal use.

        Args:
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if the mean is to be evalulated for multiple
                states or shape `(dim_state,)` if mean is to be evaluated for a single
                state.
            time_index: Current time index.

        Returns:
            Array corresponding to mean of states at time index `t + 1`, of shape
            `(num_state, dim_state)` if a 2D `states` array is passed or of shape
            `(dim_state,`) otherwise.
        """

    def sample_state_noise_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Independently sample state noise vector(s) given current state(s).

        Args:
            rng: NumPy random number generator.
            states: Array of model state(s) at time index `time_index`. Either of shape
                `(num_state, dim_state)` if multiple state noise vectors are to be
                generated or shape `(dim_state,)` if a single state noise vector is to
                be generated.
            time_index: Current time index.

        Returns:
            Array containing samples of state noise vector(s) at time index
            `time_index`. Either of shape `(num_state, dim_state)` if multiple state
            noise vectors were generated or shape `(dim_state,)` if a single state noise
            vector was generated.
        """
        if states.ndim == 1:
            states = states[None]
            states_was_one_dimensional = True
        else:
            states_was_one_dimensional = False
        state_noise_vectors = self._sample_state_noise_given_state(
            rng, states, time_index
        )
        if states_was_one_dimensional:
            return state_noise_vectors[0]
        else:
            return state_noise_vectors

    @abc.abstractmethod
    def _sample_state_noise_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Internal implementation of `sample_state_noise_given_state` method.

        Unlike `sample_state_noise_given_state` only accepts / returns 2D arrays.

        Args:
            rng: NumPy random number generator.
            states: Array of model states at time index `time_index` of shape
                `(num_state, dim_state)`.
            time_index: Current time index.

        Returns:
            Array containing sample(s) of additive state noise vector(s) at time index
            `time_index` of shape `(num_state, dim_state)`.
        """

    def _sample_state_transition(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        if self._deterministic_state_update:
            return self._next_state_mean(states, time_index)
        else:
            return self._next_state_mean(
                states, time_index
            ) + self._sample_state_noise_given_state(rng, states, time_index)

    @abc.abstractmethod
    def log_density_state_noise_given_state(
        self, state_noise_vectors: np.ndarray, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        """Evaluate the log densities of state noise vectors given states.

        Args:
            state_noise_vectors: Array of additive state noise vector(s) at time index
                `time_index`. Either of shape `(num_state, dim_state)` if density is to
                be evalulated at multiple state noise vectors or shape `(dim_state,)` if
                density is to be evaluated for a single state noise vector.
            states: Array of model state(s) at time index `time_index`. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                observation noise vectors or shape `(dim_state,)` if density is to be
                evaluated for a single observation noise vector.
            time_index: Current time index.

        Returns:
            Array of log conditional probability densities of state noise vectors given
            states for each state and state noise vector pair of shape `(num_state,)` or
            a scalar / array of shape `()` if a single state and state noise vector pair
            is provided.
        """

    def log_density_state_transition(
        self, next_states: np.ndarray, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        if self._deterministic_state_update:
            raise DensityNotDefinedError("Deterministic state transition.")
        return self.log_density_state_noise_given_state(
            next_states - self._next_state_mean(states, time_index), states, time_index,
        )


def _increment_matrix(
    matrix: np.ndarray, scalar_or_vector_or_matrix: Union[Number, np.ndarray]
) -> np.ndarray:
    """Increment matrix by another which may be represented as a scalar or vector.

    Args:
        matrix: Matrix to be incremented, a 2D array. Updated in-place where possible.
        scalar_or_vector_or_matrix: Representation of second matrix to add to `matrix`.
            Either a scalar, vector (1D array) or matrix (2D array). If a scalar the
            matrix to be added is assumed to be the identity matrix of the same shape as
            `matrix` multiplied by the scalar. If a vector the matrix to be added is
            assumed to be a diagonal matrix with the values of the vector along its
            diagonal.

    Returns:
        Computed matrix sum.
    """
    if (
        isinstance(scalar_or_vector_or_matrix, np.ndarray)
        and scalar_or_vector_or_matrix.ndim == 2
    ):
        matrix += scalar_or_vector_or_matrix
        return matrix
    elif (
        isinstance(scalar_or_vector_or_matrix, np.ndarray)
        and scalar_or_vector_or_matrix.ndim == 1
    ) or isinstance(scalar_or_vector_or_matrix, Number):
        matrix_diagonal = np.einsum("ii->i", matrix)
        matrix_diagonal += scalar_or_vector_or_matrix
        return matrix
    else:
        raise ValueError(
            f"Second argument is of unrecognised type: "
            f"{type(scalar_or_vector_or_matrix)}."
        )


def _postmultiply_matrix(
    matrix_or_vector: np.ndarray, scalar_or_vector_or_matrix: Union[Number, np.ndarray]
) -> np.ndarray:
    """Postmultiply an array by a matrix which may be represented as a scalar or vector.

    Args:
        matrix_or_vector: Matrix / vector to be multiplied, a 2D or 1D array.
        scalar_or_vector_or_matrix: Representation of second matrix to Postmultiply
            `matrix` by. Either a scalar, vector (1D array) or matrix (2D array). If a
            scalar the matrix to be postmultiplied by is assumed to be the identity
            matrix of the same shape as `matrix` multiplied by the scalar. If a vector
            the matrix to be postmultiplied by is assumed to be a diagonal matrix with
            the values of the vector along its diagonal.

    Returns:
        Computed matrix product.
    """
    if (
        isinstance(scalar_or_vector_or_matrix, np.ndarray)
        and scalar_or_vector_or_matrix.ndim == 2
    ):
        return matrix_or_vector @ scalar_or_vector_or_matrix
    elif (
        isinstance(scalar_or_vector_or_matrix, np.ndarray)
        and scalar_or_vector_or_matrix.ndim == 1
    ) or isinstance(scalar_or_vector_or_matrix, Number):
        return matrix_or_vector * scalar_or_vector_or_matrix
    else:
        raise ValueError(
            f"Second argument is of unrecognised type: "
            f"{type(scalar_or_vector_or_matrix)}."
        )


class AbstractGaussianObservationModel(AbstractAdditiveObservationNoiseModel):
    """Abstract model base class with Gaussian observation noise distributions.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = model.sample_initial_state(rng)
                t = 0
            else:
                state_sequence[s] = model.sample_state_transition(
                    rng, state_sequence[s - 1], s - 1)
            if s == observation_time_indices[t]:
                observation_sequence[t] = rng.multivariate_normal(
                    model.observation_mean(state_sequence[t], t),
                    model.observation_noise_covar)
                t += 1

    This corresponds to assuming the conditional distribution of the current observation
    given current state takes the form of a multivariate Gaussian distribution.
    """

    def __init__(
        self,
        dim_state: int,
        dim_observation: int,
        observation_noise_covar: Union[Number, np.ndarray],
        **kwargs,
    ):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_observation: Dimension of observation vector.
            observation_noise_covar: Covariance of additive Gaussian noise in
                observations. Either a scalar, 1D array of shape `(dim_observation,)` or
                2D array of shape `(dim_observation, dim_observation)`.. If a scalar the
                covariance matrix will assume to be the identity scaled by this value.
                If a 1D array the covariance matrix will be assumed to be diagonal with
                the array specifying the diagonal values. If a 2D array the covariance
                will be assumed to be specified directly by the array.
        """
        self._observation_noise_covar = observation_noise_covar
        super().__init__(dim_state=dim_state, dim_observation=dim_observation, **kwargs)

    @property
    def observation_noise_covar(self) -> np.ndarray:
        """Covariance of Gaussian observation noise.

        This will return a 2D array of shape `(dim_observation, dim_observation)`
        corresponding to the observation noise covariance matrix. Where possible the
        methods `increment_by_observation_noise_covar`,
        `postmultiply_by_observation_noise_covar` and
        `premultiply_by_inv_observation_noise_covar` should be used preferentially to
        this property as they will avoid explicitly constructing the covariance matrix
        if it is of a sparse form.
        """
        if isinstance(self._observation_noise_covar, Number):
            return np.identity(self.dim_observation) * self._observation_noise_covar
        elif (
            isinstance(self._observation_noise_covar, np.ndarray)
            and self._observation_noise_covar.ndim == 1
        ):
            return np.diag(self._observation_noise_covar)
        elif (
            isinstance(self._observation_noise_covar, np.ndarray)
            and self._observation_noise_covar.ndim == 2
        ):
            return self._observation_noise_covar
        else:
            raise NotImplementedError()

    def increment_by_observation_noise_covar(self, matrix: np.ndarray) -> np.ndarray:
        """Adds observation noise covariance to another matrix.

        Args:
            matrix: Matrix to be incremented, a 2D array. Updated in-place where
                possible.

        Returns:
            Computed matrix sum.
        """
        return _increment_matrix(matrix, self._observation_noise_covar)

    def postmultiply_by_observation_noise_covar(
        self, matrix_or_vector: np.ndarray
    ) -> np.ndarray:
        """Postmultiply another matrix by observation noise covariance matrix.

        Args:
            matrix_or_vector: Matrix or vector to be postmultiplied, a 2D / 1D array.

        Returns:
            Computed matrix product `matrix_or_vector @ observation_noise_covar`.
        """
        return _postmultiply_matrix(matrix_or_vector, self._observation_noise_covar)

    def postmultiply_by_inv_observation_noise_covar(
        self, matrix_or_vector: np.ndarray
    ) -> np.ndarray:
        """Postmultiply a matrix / vector by obs. noise covariance matrix inverse.

        Args:
            matrix_or_vector: Matrix or vector to be postmultiplied, a 2D / 1D array.

        Returns:
            Computed product `matrix_or_vector @ inv(observation_noise_covar)`.
        """
        if isinstance(self._observation_noise_covar, Number) or (
            isinstance(self._observation_noise_covar, np.ndarray)
            and self._observation_noise_covar.ndim == 1
        ):
            return matrix_or_vector / self._observation_noise_covar
        else:
            if not hasattr(self, "_chol_observation_noise_covar"):
                self._chol_observation_noise_covar = nla.cholesky(
                    self._observation_noise_covar
                )
            return sla.cho_solve(
                (self._chol_observation_noise_covar, True), matrix_or_vector
            )

    def postmultiply_by_inv_chol_trans_observation_noise_covar(
        self, matrix_or_vector: np.ndarray
    ) -> np.ndarray:
        """Postmultiply matrix / vector by obs. noise covar. matrix inverse of factor.

        Args:
            matrix_or_vector: Matrix or vector to be postmultiplied, a 2D / 1D array.

        Returns:
            Computed matrix product `matrix @ inv(chol(observation_noise_covar).T)`.
        """
        if isinstance(self._observation_noise_covar, Number) or (
            isinstance(self._observation_noise_covar, np.ndarray)
            and self._observation_noise_covar.ndim == 1
        ):
            return matrix_or_vector / self._observation_noise_covar ** 0.5
        else:
            if not hasattr(self, "_chol_observation_noise_covar"):
                self._chol_observation_noise_covar = sla.cholesky(
                    self._observation_noise_covar
                )
            return sla.solve_triangular(
                self._chol_observation_noise_covar, matrix_or_vector.T, lower=True
            ).T


class AbstractGaussianStateModel(AbstractAdditiveStateNoiseModel):
    """Abstract model class with Gaussian initial state and state noise distributions.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = rng.multivariate_normal(
                    model.initial_state_mean, model.initial_state_covar)
                t = 0
            else:
                state_sequence[s] = rng.multivariate_normal(
                    model.next_state_mean(state_sequence[s - 1], s - 1),
                    model.state_noise_covar
                )
            if s == observation_time_indices[t]:
                observation_sequence[t] = model.sample_observation_given_state(
                    rng, state_sequence[s], s)
                t += 1

    This corresponds to assuming the initial state distribution and conditional
    distribution of the next state given current take the form of multivariate Gaussian
    distributions.
    """

    def __init__(
        self,
        dim_state: int,
        dim_observation: int,
        initial_state_mean: Union[Number, np.ndarray],
        initial_state_covar: Union[Number, np.ndarray],
        state_noise_covar: Union[Number, np.ndarray],
        **kwargs,
    ):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_obs: Dimension of observation vector.
            initial_state_mean: Initial state distribution mean. Either a scalar or
                array of shape `(dim_state,)`. If a scalar the mean will be assumed to
                be the scalar multiplied by a vector of ones.
            initial_state_covar: Covariance of Gaussian initial state distribution.
                Either a scalar, 1D array of shape `(dim_state,)` or 2D array of shape
                `(dim_state, dim_state)`. If a scalar the covariance matrix will assume
                to be the identity scaled by this value. If a 1D array the covariance
                matrix will be assumed to be diagonal with the array specifying the
                diagonal values. If a 2D array the covariance will be assumed to be
                specified directly by the array.
            state_noise_covar: Covariance of Gaussian state noise distribution. Either
                a scalar, 1D array of shape `(dim_state,)` or 2D array of shape
                `(dim_state, dim_state)`. If a scalar the covariance matrix will assume
                to be the identity scaled by this value. If a 1D array the covariance
                matrix will be assumed to be diagonal with the array specifying the
                diagonal values. If a 2D array the covariance will be assumed to be
                specified directly by the array.
        """
        self._initial_state_mean = initial_state_mean
        self._initial_state_covar = initial_state_covar
        self._state_noise_covar = state_noise_covar
        super().__init__(
            dim_state=dim_state,
            dim_observation=dim_observation,
            deterministic_state_update=np.all(state_noise_covar == 0.0),
            **kwargs,
        )

    @property
    def state_noise(self) -> np.ndarray:
        """Covariance of Gaussian state noise.

        This will return a 2D array of shape `(dim_state, dim_state)` corresponding to
        the state noise covariance matrix. Where possible the method
        `increment_by_state_noise_covar` should be used preferentially to this property
        as they will avoid explicitly constructing the covariance matrix if it is of a
        sparse form.
        """
        if isinstance(self._state_noise_covar, Number):
            return np.identity(self.dim_state) * self._state_noise_covar
        elif (
            isinstance(self._state_noise_covar, np.ndarray)
            and self._state_noise_covar.ndim == 1
        ):
            return np.diag(self._state_noise_covar)
        elif (
            isinstance(self._state_noise_covar, np.ndarray)
            and self._state_noise_covar.ndim == 2
        ):
            return self._state_noise_covar
        else:
            raise NotImplementedError()

    def increment_by_state_noise_covar(self, matrix: np.ndarray) -> np.ndarray:
        """Adds state noise covariance to another matrix.

        Args:
            matrix: Matrix to be incremented, a 2D array. Updated in-place where
                possible.

        Returns:
            Computed matrix sum.
        """
        return _increment_matrix(matrix, self._state_noise_covar)

    @property
    def initial_state_mean(self) -> np.ndarray:
        """Mean of initial state distribution."""
        if isinstance(self._initial_state_mean, np.ndarray):
            return self._initial_state_mean
        elif isinstance(self._initial_state_mean, Number):
            return self._initial_state_mean * np.ones(self.dim_state)
        else:
            raise NotImplementedError()

    @property
    def initial_state_covar(self) -> np.ndarray:
        """Covariance of initial state distribution."""
        if (
            isinstance(self._initial_state_covar, np.ndarray)
            and self._initial_state_covar.ndim == 2
        ):
            return self._initial_state_covar
        elif (
            isinstance(self._initial_state_covar, np.ndarray)
            and self._initial_state_covar.ndim == 1
        ):
            return np.diag(self._initial_state_covar)
        elif isinstance(self._initial_state_mean, Number):
            return self._initial_state_covar * np.identity(self.dim_state)
        else:
            raise NotImplementedError()


def _centred_diagonal_covariance_gaussian_log_density(
    vectors: np.array, variances: Union[Number, np.ndarray]
) -> np.ndarray:
    """Evaluate log densities of Gaussian distributed vectors.

    Gaussian distribution is assumed to be centred (zero-mean) and with a diagonal
    covariance matrix.

    Args:
        vectors: Array of Gaussian random vector realization(s) to evaluate log density
            at. Either of shape `(num_vector, dim_vector)`
            if density is to be evalulated at multiple vectors or
            shape `(dim_vector,)` if density is to be evaluated for a single
            vector.
        variances: Array of shape `(dim_vector,`) of per-component variances
            corresponding to diagonal elements of covariance matrix or scalar if all
            components share the same variance (corresponding to a scaled identity
            covariance matrix).

    Returns:
        Array of log probability densities of vectors of shape `(num_vector,)` or a
        scalar / array of shape `()` if a single vector is provided.
    """
    return (
        -(vectors ** 2 / variances + np.log(2 * np.pi) + np.log(variances)).sum(-1) / 2
    )


class AbstractDiagonalGaussianObservationModel(AbstractGaussianObservationModel):
    """Abstract model base class with diagonal Gaussian observation noise distributions.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = model.sample_initial_state(rng)
                t = 0
            else:
                state_sequence[s] = model.sample_state_transition(
                    rng, state_sequence[s - 1], s - 1)
            if s == observation_time_indices[t]:
                observation_sequence[t] = (
                    model.observation_mean(state_sequence[t], t)
                    + model.observation_noise_std *
                    rng.standard_normal(model.dim_observation)
                )
                t += 1

    This corresponds to assuming the conditional distribution of the current observation
    given current state takes the form of a multivariate Gaussian distributions with
    diagonal covariance.
    """

    def __init__(
        self,
        dim_state: int,
        dim_observation: int,
        observation_noise_std: Union[Number, np.ndarray],
        **kwargs,
    ):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_observation: Dimension of observation vector.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_state,)`. Noise in
                each dimension assumed to be independent i.e. diagonal noise covariance.
        """
        self._observation_noise_std = observation_noise_std
        super().__init__(
            dim_state=dim_state,
            dim_observation=dim_observation,
            observation_noise_covar=observation_noise_std ** 2,
            **kwargs,
        )

    @property
    def observation_noise_std(self) -> np.ndarray:
        """Standard deviation of additive Gaussian noise in observations."""
        if isinstance(self._observation_noise_std, Number):
            self._observation_noise_std = self._observation_noise_std * np.ones(
                self.dim_observation
            )
        return self._observation_noise_std

    def _sample_observation_noise_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        return self._observation_noise_std * rng.standard_normal(
            (states.shape[0], self.dim_observation)
        )

    def log_density_observation_noise_given_state(
        self, observation_noise_vectors: np.ndarray, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        return _centred_diagonal_covariance_gaussian_log_density(
            observation_noise_vectors, self._observation_noise_covar
        )


class AbstractDiagonalGaussianStateModel(AbstractGaussianStateModel):
    """Abstract model base class with diagonal Gaussian state distributions.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = (
                    model.initial_state_mean
                    + model.initial_state_std * rng.standard_normal(model.dim_state)
                )
                t = 0
            else:
                state_sequence[s] = (
                    model.next_state_mean(state_sequence[s - 1], s - 1)
                    + model.state_noise_std * rng.standard_normal(model.dim_state)
                )
            if s == observation_time_indices[t]:
                observation_sequence[t] = model.sample_observation_given_state(
                    rng, state_sequence[s], s)
                t += 1

    This corresponds to assuming the initial state distribution and conditional
    distribution of the next state given current take the form of multivariate Gaussian
    distributions with diagonal covariances.
    """

    def __init__(
        self,
        dim_state: int,
        dim_observation: int,
        initial_state_mean: Union[float, np.ndarray],
        initial_state_std: Union[float, np.ndarray],
        state_noise_std: Union[float, np.ndarray] = 0,
        **kwargs,
    ):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_obs: Dimension of observation vector.
            initial_state_mean: Initial state distribution mean.
                Either a scalar or array of shape `(dim_state,)`.
            initial_state_std: Initial state distribution standard deviation. Either a
                scalar or array of shape `(dim_state,)`. Each state dimension is assumed
                to be independent i.e. a diagonal covariance.
            state_noise_std: Standard deviation of additive Gaussian noise in state
                update. Either a scalar or array of shape `(dim_state,)`. Noise in each
                dimension assumed to be independent i.e. a diagonal noise covariance. If
                zero deterministic dynamics are assumed.
        """
        self._initial_state_mean = initial_state_mean
        self._initial_state_std = initial_state_std
        self._state_noise_std = state_noise_std
        super().__init__(
            dim_state=dim_state,
            dim_observation=dim_observation,
            initial_state_mean=initial_state_mean,
            initial_state_covar=initial_state_std ** 2,
            state_noise_covar=state_noise_std ** 2,
            **kwargs,
        )

    def _sample_initial_state(self, rng: Generator, num_state: int) -> np.ndarray:
        return (
            self._initial_state_mean
            + rng.standard_normal((num_state, self.dim_state)) * self._initial_state_std
        )

    def log_density_initial_state(self, states: np.ndarray) -> np.ndarray:
        return -(
            0.5 * ((states - self._initial_state_mean) / self._initial_state_std) ** 2
            + 0.5 * np.log(2 * np.pi)
            + np.log(self._initial_state_std)
        ).sum(-1)

    def _sample_state_noise_given_state(
        self, rng: Generator, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        return self._state_noise_std * rng.standard_normal(states.shape)

    def log_density_state_noise_given_state(
        self, state_noise_vectors: np.ndarray, states: np.ndarray, time_index: int
    ) -> np.ndarray:
        return _centred_diagonal_covariance_gaussian_log_density(
            state_noise_vectors, self._state_noise_covar
        )


class AbstractDiagonalGaussianModel(
    AbstractDiagonalGaussianObservationModel, AbstractDiagonalGaussianStateModel
):
    """Abstract model base class with diagonal Gaussian noise distributions.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = (
                    model.initial_state_mean
                    + model.initial_state_std * rng.standard_normal(model.dim_state)
                )
                t = 0
            else:
                state_sequence[s] = (
                    model.next_state_mean(state_sequence[s - 1], s - 1)
                    + model.state_noise_std * rng.standard_normal(model.dim_state)
                )
            if s == observation_time_indices[t]:
                observation_sequence[t] = (
                    model.observation_mean(state_sequence[t], t)
                    + model.observation_noise_std *
                    rng.standard_normal(model.dim_observation)
                )
                t += 1

    This corresponds to assuming the initial state distribution, conditional
    distribution of the next state given current and conditional distribution of the
    current observation given current state all take the form of multivariate Gaussian
    distributions with diagonal covariances. In the case of deterministic state update
    dynamics the conditional distribution of the next state given the current state will
    be a Dirac measure with all mass located at the forward map of the current state
    through the state dynamics function and so will not have a density with respect to
    the Lebesgue measure.
    """

    def __init__(
        self,
        dim_state: int,
        dim_observation: int,
        observation_noise_std: Union[float, np.ndarray],
        initial_state_mean: Union[float, np.ndarray],
        initial_state_std: Union[float, np.ndarray],
        state_noise_std: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_obs: Dimension of observation vector.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_state,)`. Noise in
                each dimension assumed to be independent i.e. diagonal noise covariance.
            initial_state_mean: Initial state distribution mean.
                Either a scalar or array of shape `(dim_state,)`.
            initial_state_std: Initial state distribution standard deviation. Either a
                scalar or array of shape `(dim_state,)`. Each state dimension is assumed
                to be independent i.e. a diagonal covariance.
            state_noise_std: Standard deviation of additive Gaussian noise in state
                update. Either a scalar or array of shape `(dim_state,)` or `None`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance. If zero or `None` deterministic dynamics are assumed.

        """
        super().__init__(
            dim_state=dim_state,
            dim_observation=dim_observation,
            observation_noise_std=observation_noise_std,
            initial_state_mean=initial_state_mean,
            initial_state_std=initial_state_std,
            state_noise_std=state_noise_std,
            **kwargs,
        )


class AbstractLinearModel(
    AbstractAdditiveObservationNoiseModel, AbstractAdditiveStateNoiseModel
):
    """Abstract base class for linear models.

    Here linear models refers to models with linear state transition and observation
    operators and additive state and observation noise.

    The modelled system dynamics are of the form

        for s in range(num_step + 1):
            if s == 0:
                state_sequence[0] = model.sample_initial_state(rng)
                t = 0
            else:
                state_sequence[s] = (
                    model.state_transition_matrix @ state_sequence[s - 1] +
                    model.sample_state_noise(rng, state_sequence[s - 1], s - 1)
                )
            if s == observation_time_indices[t]:
                observation_sequence[t] = (
                    model.observation_matrix  @ state_sequence[s]) +
                    model.sample_observation_noise(rng, state_sequence[s - 1], s - 1)
                )
                t += 1

    where `state_transition_matrix` and `observation_matrix` are matrices representing
    respectively the linear state transition and observation operators. In practice
    these may be implemented in a functional form via the `next_state_mean` and
    `observation_mean` methods.
    """

    @property
    def state_transition_matrix(self) -> np.ndarray:
        """Matrix representing linear state transition operator.

        This will return a 2D array of shape `(dim_state, dim_state)` corresponding to
        the matrix representation of the linear state transition operator. Where
        possible the method `next_state_mean` should be used preferentially to this
        property as they will avoid explicitly constructing the transition matrix if it
        is of a sparse form.
        """
        if hasattr(self, "_state_transition_matrix"):
            return self._state_transition_matrix
        elif hasattr(self, "next_state_mean"):
            self._state_transition_matrix = self.next_state_mean(
                np.identity(self.dim_state), None
            ).T
            return self._state_transition_matrix
        else:
            raise NotImplementedError()

    @property
    def observation_matrix(self) -> np.ndarray:
        """Matrix representing linear observation operator.

        This will return a 2D array of shape `(dim_observation, dim_state)`
        corresponding to the matrix representation of the linear observation operator.
        Where possible the method `observation_mean` should be used preferentially to
        this property as they will avoid explicitly constructing the observation matrix
        if it is of a sparse form.
        """
        if hasattr(self, "_observation_matrix"):
            return self._observation_matrix
        elif hasattr(self, "observation_mean"):
            self._observation_matrix = self.observation_mean(
                np.identity(self.dim_state), None
            ).T
            return self._observation_matrix
        else:
            raise NotImplementedError()


class AbstractLinearGaussianModel(
    AbstractLinearModel, AbstractGaussianStateModel, AbstractGaussianObservationModel
):
    """Abstract base class for linear-Gaussian models defining expected interface."""

    @property
    def state_transition_matrix(self) -> np.ndarray:
        """Matrix representing linear state transition operator."""
        if hasattr(self, "_state_transition_matrix"):
            return self._state_transition_matrix
        elif hasattr(self, "next_state_mean"):
            self._state_transition_matrix = self.next_state_mean(
                np.identity(self.dim_state), None
            ).T
            return self._state_transition_matrix
        else:
            raise NotImplementedError()

    @property
    def observation_matrix(self) -> np.ndarray:
        """Matrix representing linear observation operator."""
        if hasattr(self, "_observation_matrix"):
            return self._observation_matrix
        elif hasattr(self, "observation_mean"):
            self._observation_matrix = self.observation_mean(
                np.identity(self.dim_state), None
            ).T
            return self._observation_matrix
        else:
            raise NotImplementedError()


class AbstractIntegratorModel(AbstractAdditiveStateNoiseModel):
    """Abstract base class for models using integrator to perform state updates.

    The modelled system dynamics are of the form

        for s in range(num_step):
            if s == 0:
                state_sequence[0] = model.sample_initial_state(rng)
                t = 0
            else:
                state_sequence[s] = (
                    model.integrate.forward_integrate(
                        state_sequence[s-1], s - 1, num_step)
                + model.sample_state_noise(rng, states, s - 1)
            )
            if s == observation_time_indices[t]:
                observation_sequence[t] = model.sample_observation_given_state(
                    rng, state_sequence[s], s)
                t += 1

    Where `model.integrator.forward_integrate` integrates the deterministic model
    dynamics forward `num_step` integrator time steps.
    """

    def __init__(
        self,
        dim_state: int,
        dim_observation: int,
        integrator,
        num_integrator_step_per_update: int = 1,
        **kwargs,
    ):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_obs: Dimension of observation vector.
            integrator: Integrator for model state dynamics. Object should define a
                `forward_integrate` method with signature

                    integrator.forward_integrate(states, start_time_index, num_step)

                with `states` an array of a batch of state vectors at the current time
                index, `start_time_index` an integer defining the current time index and
                `num_step` the number of integrator time steps to perform, with the
                method returning an array of state vectors at the next time index. The
                integrator object should also have a float attribute `dt` corresponding
                to the integrator time step.
            num_integrator_step_per_update: Number of integrator time-steps between
                successive observations and generated states.
        """
        self.integrator = integrator
        self.num_integrator_step_per_update = num_integrator_step_per_update
        super().__init__(dim_state=dim_state, dim_observation=dim_observation, **kwargs)

    def _next_state_mean(self, states: np.ndarray, time_index: int) -> np.ndarray:
        """Computes mean of next state(s) given current state(s).

        Implements determinstic component of state update dynamics with new state
        calculated by output of this function plus additive zero-mean Gaussian noise.
        For models with fully-determinstic state dynamics no noise is added so this
        function exactly calculates the next state.

        Args:
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if the mean is to be evalulated for multiple
                states or shape `(dim_state,)` if mean is to be evaluated for a single
                state.
            time_index: Current time index.

        Returns:
            Array corresponding to mean of states at time index `t + 1`, of shape
            `(num_state, dim_state)` if a 2D `states` array is passed or of shape
            `(dim_state,`) otherwise.
        """
        if states.ndim == 1:
            return np.asarray(
                self.integrator.forward_integrate(
                    states[None], time_index, self.num_integrator_step_per_update
                )[0]
            )
        else:
            return np.asarray(
                self.integrator.forward_integrate(
                    states, time_index, self.num_integrator_step_per_update
                )
            )

