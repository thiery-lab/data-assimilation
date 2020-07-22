"""Base classes for models."""

import abc
from typing import Optional, Union, Sequence
import numpy as np
from numpy.random import Generator
import tqdm.auto as tqdm


class DensityNotDefinedError(Exception):
    """Raised on calling a method to evaluate a undefined probability density.

    For some models the state transition or observation given state conditional
    probability densities will not be definedm for example in the case of
    deterministic state transitions or observations.
    """


class AbstractModel(abc.ABC):
    """Abstract dynamical model base class.

    The modelled system dynamics are of the form

        state_sequence[0] = sample_initial_state(rng)
        observation_sequence[0] = sample_observation_given_state(
            rng, state_sequence[0], 0)
        for t in range(1, num_observation_time):
            state_sequence[t] = sample_state_transition(rng, state_sequence[t-1], t-1)
            observation_sequence[t] = sample_observation_given_state(
                rng, state_sequence[t], t)
    """

    def __init__(self, dim_state: int, dim_observation: int):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_observation: Dimension of observation vector.
        """
        self.dim_state = dim_state
        self.dim_observation = dim_observation

    def sample_initial_state(
        self, rng: Generator, num_state: Optional[int] = None
    ) -> np.ndarray:
        """Independently sample initial state(s).

        Args:
            rng: NumPy random number generator object.
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
            rng: NumPy random generator object.
            num_state: Number of states to sample.

        Returns:
            Array containing independent initial state samples of shape
            `(num_state, dim_state)`.
        """

    def sample_state_transition(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        """Independently sample next state(s) given current state(s).

        Args:
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if multiple states are to be propagated or
                shape `(dim_state,)` if a single state is to be propagated.
            t: Current time index.

        Returns:
            Array containing samples of state(s) at time index `t+1`. Either of shape
            `(num_state, dim_state)` if multiple states were propagated or shape
            `(dim_state,)` if a single state was propagated.
        """
        if states.ndim == 1:
            states = states[None]
            states_was_one_dimensional = True
        else:
            states_was_one_dimensional = False
        next_states = self._sample_state_transition(rng, states, t)
        if states_was_one_dimensional:
            return next_states[0]
        else:
            return next_states

    @abc.abstractmethod
    def _sample_state_transition(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        """Internal implementation of `sample_state_transition` method.

        Unlike `sample_state_transition` only accepts / returns 2D arrays.

        Args:
            states: Array of model states at time index `t`  of shape
                `(num_state, dim_state)`.
            t: Current time index.

        Returns:
            Array containing samples of states at time index `t+1` of shape
                `(num_state, dim_state)`.
        """

    def sample_observation_given_state(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        """Independently sample observation(s) given current state(s).

        Args:
            states: Array of model state(s) at time index `t`. Either of shape
            `(num_state, dim_state)` if multiple observations are to be generated or
            shape `(dim_state,)` if a single observation is to be generated.
            t: Current time index.

        Returns:
            Array containing samples of state(s) at time index `t+1`. Either of
            shape `(num_state, dim_state)` if multiple observations were generated or
            shape `(dim_state,)` if a single observation was generated.
        """
        if states.ndim == 1:
            states = states[None]
            states_was_one_dimensional = True
        else:
            states_was_one_dimensional = False
        observations = self._sample_observation_given_state(rng, states, t)
        if states_was_one_dimensional:
            return observations[0]
        else:
            return observations

    @abc.abstractmethod
    def _sample_observation_given_state(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        """Internal implementation of `sample_observation_given_state` method.

        Unlike `sample_observation_given_state` only accepts / returns 2D arrays.

        Args:
            states: Array of model states at time index `t` of shape
                `(num_state, dim_state)`.
            t: Current time index.

        Returns:
            Array containing samples of states at time index `t+1` of shape
            `(num_state, dim_state)`.
        """

    def sample_state_and_observation_sequences(
        self,
        rng: Generator,
        observation_time_indices: Sequence[int],
        num_sample: Optional[int] = None,
    ) -> np.ndarray:
        """Generate state and observation sequences from model.

        Args:
            rng: NumPy random number generator.
            observation_time_indices: Sequence of time (step) indices at which state is
                observed. The sequence elements must be integers. The maximal element
                of the sequence corresponds to the total number of time steps to
                simulate `num_step`, while the length of the sequence corresponds to
                the number of observation times `num_observation_time`.
            num_sample: Number of independent sequence pairs to sample. If equal
                to `None` (default) a single sample is generated.

        Returns:
            state_sequences: Generated state sequence(s) of shape
                `(num_step, dim_state)` if `num_sample` is None or
                `(num_step, num_sample, dim_state)` otherwise.
            observation_sequences: Generated observation sequence(s) of shape
                `(num_observation_time, dim_observation)` if `num_sample` is None or
                `(num_observation_time, num_sample, dim_observation)` otherwise.
        """
        observation_time_indices = np.sort(observation_time_indices)
        num_step = observation_time_indices[-1]
        num_observation_time = len(observation_time_indices)
        if num_sample is None:
            state_sequences = np.full((num_step, self.dim_state), np.nan)
            observation_sequences = np.full(
                (num_observation_time, self.dim_observation), np.nan
            )
        else:
            state_sequences = np.full((num_step, num_sample, self.dim_state), np.nan)
            observation_sequences = np.full(
                (num_observation_time, num_sample, self.dim_observation), np.nan
            )
        for s in tqdm.trange(0, num_step, desc="Sampling", unit="time steps"):
            if s == 0:
                state_sequences[s] = self.sample_initial_state(rng, num_sample)
                t = 0
            else:
                state_sequences[s] = self.sample_state_transition(
                    rng, state_sequences[s - 1], s - 1
                )
            if s == observation_time_indices[t]:
                observation_sequences[t] = self.sample_observation_given_state(
                    rng, state_sequences[s], s
                )
                t += 1
        return state_sequences, observation_sequences

    @abc.abstractmethod
    def log_density_initial_state(self, states: np.ndarray) -> np.ndarray:
        """Calculate log probability density of initial state(s).

        Args:
            states: Array of model state(s) at time index 0. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                states or shape `(dim_state,)` if density is to be evaluated for a
                single state.

        Returns:
            Array of log probability densities for each state or a scalar / array of
            shape `()` if a single state is provided.
        """

    @abc.abstractmethod
    def log_density_state_transition(
        self, next_states: np.ndarray, states: np.ndarray, t: int
    ) -> np.ndarray:
        """Calculate log probability density of a transition between states.

        Args:
            next_states: Array of model state(s) at time index `t + 1`. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                state pairs or shape `(dim_state,)` if density is to be evaluated for a
                single state pair.
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                state pairs or shape `(dim_state,)` if density is to be evaluated for a
                single state pair.
            t: Current time index.

        Returns:
            Array of log conditional probability densities of next state given current
            state for each state pair or a scalar / array of shape `()` if a single
            state pair is provided.
        """

    @abc.abstractmethod
    def log_density_observation_given_state(
        self, observations: np.ndarray, states: np.ndarray, t: int
    ) -> np.ndarray:
        """Calculate log probability density of observation(s) given state(s).

        Args:
            observations: Array of model observation(s) at time index `t`. Either of
                shape `(num_state, dim_observation)` if density is to be evalulated at
                multiple observations or shape `(dim_observation,)` if density is to be
                evaluated for a single observation.
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if the density is to be evalulated at multiple
                observations or shape `(dim_state,)` if density is to be evaluated for
                a single observation.
            t: Current time index.

        Returns:
            Array of log conditional probability densities of observation given state
            for each state and observation pair of shape `(n_state,)` or a scalar /
            array of shape `()` if a single state and observation pair is provided.
        """

    def log_density_state_sequence(self, state_sequences: np.ndarray) -> np.ndarray:
        """Calculate the log joint probability density of state sequence(s).

        Args:
           state_sequences: Array of model state sequence(s). Either of shape
               `(num_step, num_sequence, dim_state)` if density is to be evaluated for
               multiple sequences or shape `(num_step, dim_state)` if density is to be
               evaluated for a single sequence.

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
        """Evaluate the log density of a state and observation sequence pair.

        Args:
            state_sequences: Array of model state sequence(s). Either of shape
                `(num_step, num_sequence, dim_state)` if density is to be evaluated for
                multiple sequences or shape `(num_step, dim_state)` if density is to be
                evaluated for a single sequence.
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
            Array of log joint probability densities of state and observation sequences
            of shape `(num_sequence,)` or a scalar / array of shape `()` if a single
            state and observation sequence pair is provided.
        """
        observation_time_indices = np.sort(observation_time_indices)
        num_step = observation_time_indices[-1]
        num_observation_time = len(observation_time_indices)
        assert state_sequence.shape[0] == num_step
        assert observation_sequence.shape[0] == num_observation_time
        for s in range(0, state_sequence.shape[0]):
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


class DiagonalGaussianObservationModel(AbstractModel):
    """Abstract model base class with diagonal Gaussian observation noise distributions.

    The modelled system dynamics are of the form

        state_sequence[0] = sample_initial_state(rng)
        observation_sequence[0] = (
            observation_mean(state_sequence[0], 0)
            + observation_noise_std * rng.standard_normal(dim_observation))
        for t in range(1, num_step):
            state_sequence[t] = sample_state_transition(rng, state_sequence[t-1], t-1)
            observation_sequence[t] = (
                observation_mean(state_sequence[t], t)
                + observation_noise_std * rng.standard_normal(dim_observation))

    This corresponds to assuming the conditional distribution of the current observation
    given current state takes the form of a multivariate Gaussian distributions with
    diagonal covariance.
    """

    def __init__(
        self,
        dim_state: int,
        dim_observation: int,
        observation_noise_std: Union[float, np.ndarray],
        **kwargs
    ):
        """
        Args:
            dim_state: Dimension of model state vector.
            dim_observation: Dimension of observation vector.
            observation_noise_std: Standard deviation of additive Gaussian noise in
                observations. Either a scalar or array of shape `(dim_state,)`. Noise in
                each dimension assumed to be independent i.e. diagonal noise covariance.
        """
        self.observation_noise_std = observation_noise_std
        super().__init__(dim_state=dim_state, dim_observation=dim_observation, **kwargs)

    def observation_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        """Computes mean of observation(s) given state(s) at time index `t`.

        Args:
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if the mean is to be evalulated for multiple
                observations or shape `(dim_state,)` if mean is to be evaluated for
                a single observation.
            t: Current time index.

        Returns:
            Array corresponding to mean of observations at time index `t`, of shape
            `(num_state, dim_observation)` if a 2D `states` array is passed or of shape
            `(dim_observation,`) otherwise.
        """
        return self._observation_mean(states, t)

    @abc.abstractmethod
    def _observation_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        """Internal implementation of `observation_mean` method.

        Should be called in preference to `observation_mean` internally by other
        methods to allow use of mix-ins to change behaviour of `observation_mean`
        without affecting internal use.

        Args:
            states: Array of model state(s) at time index `t` of shape
                `(num_state, dim_state)` if the mean is to be evalulated for multiple
                observations or shape `(dim_state,)` if mean is to be evaluated for
                a single observation.
            t: Current time index.

        Returns:
            Array corresponding to mean of observations at time index `t`, of shape
            `(num_state, dim_observation)` if a 2D `states` array is passed or of shape
            `(dim_observation,`) otherwise.
        """

    def _sample_observation_given_state(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        return self._observation_mean(
            states, t
        ) + self.observation_noise_std * rng.standard_normal(
            (states.shape[0], self.dim_observation)
        )

    def log_density_observation_given_state(
        self, observations: np.ndarray, states: np.ndarray, t: int
    ) -> np.ndarray:
        return -(
            0.5
            * (
                (observations - self._observation_mean(states, t))
                / self.observation_noise_std
            )
            ** 2
            + 0.5 * np.log(2 * np.pi)
            + np.log(self.observation_noise_std)
        ).sum(-1)


class DiagonalGaussianStateNoiseModel(AbstractModel):
    """Abstract model base class with diagonal Gaussian state distributions.

    The modelled system dynamics are of the form

        state_sequence[0] = (
            init_state_mean + init_state_std * rng.standard_normal(dim_state))
        observation_sequence[0] = sample_observation_given_state(
            rng, state_sequence[0], 0)
        for t in range(1, n_steps):
            state_sequence[t] = (
                next_state_mean(state_sequence[t-1], t-1)
                + state_noise_std * rng.standard_normal(dim_state))
            observation_sequence[t] = sample_observation_given_state(
                rng, state_sequence[t], t)

    This corresponds to assuming the initial state distribution and conditional
    distribution of the next state given current take the form of multivariate Gaussian
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
        initial_state_mean: Union[float, np.ndarray],
        initial_state_std: Union[float, np.ndarray],
        state_noise_std: Optional[Union[float, np.ndarray]] = None,
        **kwargs
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
                update. Either a scalar or array of shape `(dim_state,)` or `None`.
                Noise in each dimension assumed to be independent i.e. a diagonal noise
                covariance. If zero or `None` deterministic dynamics are assumed.
        """
        self.initial_state_mean = initial_state_mean
        self.initial_state_std = initial_state_std
        if state_noise_std is None or np.all(state_noise_std == 0.0):
            self._deterministic_state_update = True
        else:
            self._deterministic_state_update = False
            self.state_noise_std = state_noise_std
        super().__init__(dim_state=dim_state, dim_observation=dim_observation, **kwargs)

    def _sample_initial_state(self, rng: Generator, num_state: int) -> np.ndarray:
        return (
            self.initial_state_mean
            + rng.standard_normal((num_state, self.dim_state)) * self.initial_state_std
        )

    def next_state_mean(self, states: np.ndarray, t: int) -> np.ndarray:
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
            t: Current time index.

        Returns:
            Array corresponding to mean of states at time index `t + 1`, of shape
            `(num_state, dim_state)` if a 2D `states` array is passed or of shape
            `(dim_state,`) otherwise.
        """
        return self._next_state_mean(self, states, t)

    @abc.abstractmethod
    def _next_state_mean(self, states: np.ndarray, t: int) -> np.ndarray:
        """Internal implementation of `next_state_mean` method.

        Should be called in preference to `next_state_mean` internally by other
        methods to allow use of mix-ins to change behaviour of `next_state_mean`
        without affecting internal use.

        Args:
            states: Array of model state(s) at time index `t`. Either of shape
                `(num_state, dim_state)` if the mean is to be evalulated for multiple
                states or shape `(dim_state,)` if mean is to be evaluated for a single
                state.
            t: Current time index.

        Returns:
            Array corresponding to mean of states at time index `t + 1`, of shape
            `(num_state, dim_state)` if a 2D `states` array is passed or of shape
            `(dim_state,`) otherwise.
        """

    def _sample_state_transition(
        self, rng: Generator, states: np.ndarray, t: int
    ) -> np.ndarray:
        if self._deterministic_state_update:
            return self._next_state_mean(states, t)
        else:
            return self._next_state_mean(
                states, t
            ) + self.state_noise_std * rng.standard_normal(states.shape)

    def log_density_initial_state(self, states: np.ndarray) -> np.ndarray:
        return -(
            0.5 * ((states - self.initial_state_mean) / self.initial_state_std) ** 2
            + 0.5 * np.log(2 * np.pi)
            + np.log(self.initial_state_std)
        ).sum(-1)

    def log_density_state_transition(
        self, next_states: np.ndarray, states: np.ndarray, t: int
    ) -> np.ndarray:
        if self._deterministic_state_update:
            raise DensityNotDefinedError("Deterministic state transition.")
        else:
            return -(
                0.5
                * (
                    (next_states - self._next_state_mean(states, t))
                    / self.state_noise_std
                )
                ** 2
                + 0.5 * np.log(2 * np.pi)
                + np.log(self.state_noise_std)
            ).sum(-1)


class DiagonalGaussianModel(
    DiagonalGaussianObservationModel, DiagonalGaussianStateNoiseModel
):
    """Abstract model base class with diagonal Gaussian noise distributions.

    Assumes the model dynamics take the form

        state_sequence[0] = (
            init_state_mean + init_state_std * rng.standard_normal(dim_state))
        observation_sequence[0] = (
            observation_mean(state_sequence[0], 0)
            + observation_noise_std * rng.standard_normal(dim_observation))
        for t in range(1, n_steps):
            state_sequence[t] = (
                next_state_mean(state_sequence[t-1], t-1)
                + state_noise_std * rng.standard_normal(dim_state))
            observation_sequence[t] = (
                observation_mean(state_sequence[t], t)
                + observation_noise_std * rng.standard_normal(dim_observation))

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
        **kwargs
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
            **kwargs
        )


class IntegratorModel(AbstractModel):
    """Model using integrator to perform determinstic state updates."""

    def __init__(
        self,
        dim_state: int,
        dim_observation: int,
        integrator,
        num_integrator_step_per_update: int = 1,
        **kwargs
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

    def _next_state_mean(self, states: np.ndarray, t: int) -> np.ndarray:
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
            t: Current time index.

        Returns:
            Array corresponding to mean of states at time index `t + 1`, of shape
            `(num_state, dim_state)` if a 2D `states` array is passed or of shape
            `(dim_state,`) otherwise.
        """
        if states.ndim == 1:
            return self.integrator.forward_integrate(
                states[None], t, self.num_integrator_step_per_update
            )[0]
        else:
            return self.integrator.forward_integrate(
                states, t, self.num_integrator_step_per_update
            )
