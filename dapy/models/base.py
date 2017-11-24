"""Base model class for data assimilation example models."""

import numpy as np
import types


class DensityNotDefinedError(Exception):
    """Raised on calling a method to evaluate a undefined probability density.

    For some (non-dominated) models the state transition or observation given
    state conditional probability densities will not be defined for example
    in the case of deterministic state transitions.
    """
    pass


def inherit_docstrings(cls):
    """Class decorator to inherit parent class docstrings if not specified.

    Taken from https://stackoverflow.com/a/38601305/4798943.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
        elif isinstance(func, property) and not func.fget.__doc__:
            for parent in cls.__bases__:
                parprop = getattr(parent, name, None)
                if parprop and getattr(parprop.fget, '__doc__', None):
                    newprop = property(fget=func.fget,
                                       fset=func.fset,
                                       fdel=func.fdel,
                                       doc=parprop.fget.__doc__)
                    setattr(cls, name, newprop)
                    break

    return cls


class AbstractModel(object):
    """Abstract model base class."""

    def __init__(self, dim_z, dim_x, rng):
        """
        Args:
            dim_z (integer): Dimension of model state vector.
            dim_x (integer): Dimension of observation vector.
            rng (RandomState): Numpy RandomState random number generator.
        """
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.rng = rng

    def init_state_sampler(self, n=None):
        """Independently sample initial state(s).

        Args:
            n (integer): Number of state vectors to sample. If `None` a single
            one-dimensional state vector sample is returned.

        Returns:
            Array containing independent initial state sample(s). Either of
            shape `(n, dim_z)` if `n` is not equal to `None` and of shape
            `(dim_z,)` otherwise.
        """
        raise NotImplementedError()

    def next_state_sampler(self, z, t):
        """Independently sample next state(s) given current state(s).

        Args:
            z (array): Array of model state(s) at time index `t`. Either of
            shape `(n, dim_z)` if multiple states are to be propagated or
            shape `(dim_z,)` if a single state is to be propagated.
            t (integer): Current time index for time-inhomogeneous systems.

        Returns:
            Array containing samples of state(s) at time index `t+1`. Either of
            shape `(n, dim_z)` if multiple states were propagated or shape
            `(dim_z,)` if a single state was propagated.
        """
        raise NotImplementedError()

    def observation_sampler(self, z, t):
        """Independently sample observation(s) given current state(s).

        Args:
            z (array): Array of model state(s) at time index `t`. Either of
            shape `(n, dim_z)` if multiple observations are to be generated or
            shape `(dim_z,)` if a single observation is to be generated.
            t (integer): Current time index for time-inhomogeneous systems.

        Returns:
            Array containing samples of state(s) at time index `t+1`. Either of
            shape `(n, dim_z)` if multiple observations were generated or shape
            `(dim_z,)` if a single observation was generated.
        """
        raise NotImplementedError()

    def log_prob_dens_init_state(self, z):
        """Calculate log probability density of initial state(s).

        Args:
            z (array): Array of model state(s) at time index 0. Either of
            shape `(n, dim_z)` if the density is to be evalulated at multiple
            states or shape `(dim_z,)` if density is to be evaluated for a
            single state.

        Returns:
            Array of log probability densities for each state (or a single
            scalar if a single state and observation pair is provided).
        """
        raise NotImplementedError()

    def log_prob_dens_state_trans(self, z_n, z_c, t):
        """Calculate log probability density of a transition between states.

        Args:
            z_n (array): Array of model state(s) at time index `t + 1`. Either
            of shape `(n, dim_z)` if the density is to be evalulated at
            multiple state pairs or shape `(dim_z,)` if density is to be
            evaluated for a single state pair.
            z_c (array): Array of model state(s) at time index `t`. Either of
            shape `(n, dim_z)` if the density is to be evalulated at multiple
            state pairs or shape `(dim_z,)` if density is to be evaluated for
            a single state pair.
            t (integer): Current time index for time-inhomogeneous systems.

        Returns:
            Array of log conditional probability densities for each state pair
            (or a single scalar if a single state pair is provided).
        """
        raise NotImplementedError()

    def log_prob_den_obs_gvn_state(self, x, z, t):
        """Calculate log probability density of observation(s) given state(s).

        Args:
            z (array): Array of model state(s) at time index `t`. Either of
            shape `(n, dim_z)` if the density is to be evalulated at multiple
            observations or shape `(dim_z,)` if density is to be evaluated for
            a single observation.
            x (array): Array of model observation(s) at time index `t`. Either
            of shape `(n, dim_x)` if the density is to be evalulated at
            multiple observations or shape `(dim_x,)` if density is to be
            evaluated for a single observation.
            t (integer): Current time index for time-inhomogeneous systems.

        Returns:
            Array of log conditional probability densities for each state and
            observation pair (or a single scalar if a single state and
            observation pair is provided).
        """
        raise NotImplementedError()

    def log_prob_dens_state_seq(self, z_seq):
        """Evaluate the log joint probability density of a state sequence.

        Args:
           z_seq (array): State sequence array of shape `(n_step, dim_z)`
           where `n_step` is the number of time steps in the state sequence
           and `dim_z` the state dimensionality.

        Returns:
           Log joint probability density of overall state sequence.
        """
        log_dens = self.log_prob_dens_init_state(z_seq[0])
        for t in range(1, z_seq.shape[0]):
            log_dens += self.log_prob_dens_state_trans(
                    z_seq[t], z_seq[t-1], t-1)
            return log_dens

    def log_prob_dens_state_and_obs_seq(self, z_seq, x_seq):
        """Evaluate the log density of a state and observation sequence pair.

        Args:
           z_seq (array): State sequence array of shape `(n_step, dim_z)`
           where `n_step` is the number of time steps in the sequence and
           `dim_z` the state dimensionality.
           z_seq (array): Observation sequence array of shape `(n_step, dim_x)`
           where `n_step` is the number of time steps in the sequence and
           `dim_x` the observation dimensionality.

        Returns:
           Log joint probability density of state and observation sequence
           pair.
        """
        log_dens = self.log_prob_dens_init_state(z_seq[0])
        log_dens += self.log_prob_dens_obs_gvn_state(x_seq[0], 0)
        for t in range(1, z_seq.shape[0]):
            log_dens += self.log_prob_dens_state_trans(
                    z_seq[t], z_seq[t-1], t-1)
            log_dens += self.log_prob_dens_obs_gvn_state(x_seq[t], z_seq[t], t)
        return log_dens

    def generate(self, n_step):
        """Generate state and observation sequences from model.

        Args:
            n_step: Number of time steps to generate sequences over.

        Returns:
            z_seq (array): Generated state sequence of shape `(n_step, dim_z)`.
            x_seq (array): Generated obs. sequence of shape `(n_step, dim_x)`.
        """
        z_seq = np.full((n_step, self.dim_z), np.nan)
        x_seq = np.full((n_step, self.dim_x), np.nan)
        z_seq[0] = self.init_state_sampler()
        x_seq[0] = self.observation_sampler(z_seq[0], 0)
        for t in range(1, n_step):
            z_seq[t] = self.next_state_sampler(z_seq[t-1], t-1)
            x_seq[t] = self.observation_sampler(z_seq[t-1], t)
        return z_seq, x_seq


@inherit_docstrings
class DiagonalGaussianModel(AbstractModel):
    """Abstract model base class with diagonal Gaussian noise distributions.

    Assumes the model dynamics take the form

        z[0] = init_state_mean + init_state_std * u[0]
        x[0] = observation_func(x[0], 0) + obser_noise_std * v[0]
        for t in range(1, n_steps):
            z[t] = next_state_func(z[t-1], t-1) + state_noise_std * u[t]
            x[t] = observation_func(x[t], t) + obser_noise_std * v[t]

    where

       z[t]: unobserved system state vector at time index t,
       x[t]: observations vector at time index t,
       u[t]: zero-mean identity covariance Gaussian state noise vector at time
             index t,
       v[t]: zero-mean identity covariance Gaussian observation noise vector
             at time index t,
       init_state_mean: initial state distribution mean vector,
       init_state_std: initial state distribution standard deviation vector,
       observation_func: function mapping from states to (pre-noise)
           observations,
       obser_noise_std: observation noise distribution standard deviation
           vector,
       next_state_func: potentially non-linear function specifying state
           update dynamics for example forward integration of a ODE system,
       state_noise_std: state noise distribution standard deviation vector.
           May be all zeros which corresponds to a model with deterministic
           state update dynamics.

    This corresponds to assuming the initial state distribution, conditional
    distribution of the next state given current and conditional distribution
    of the current observation given current state all take the form of
    multivariate Gaussian distributions with diagonal covariances. In the case
    of deterministic state update dynamics the conditional distribution of the
    next state given the current state will be a Dirac measure with all mass
    located at the forward map of the current state through the state dynamics
    function and so will not have a well-defined probability density function.
    """

    def __init__(self, dim_z, dim_x, rng, init_state_mean, init_state_std,
                 state_noise_std, obser_noise_std):
        """
        Args:
            dim_z (integer): Dimension of model state vector.
            dim_x (integer): Dimension of observation vector.
            rng (RandomState): Numpy RandomState random number generator.
            init_state_mean (float or array): Initial state distribution mean.
                Either a scalar or array of shape `(dim_z,)`.
            init_state_std (float or array): Initial state distribution
                standard deviation. Either a scalar or array of shape
                `(dim_z,)`. Each state dimension is assumed to be independent
                i.e. a diagonal covariance.
            state_noise_std (float or array): Standard deviation of additive
                Gaussian noise in state update. Either a scalar or array of
                shape `(dim_z,)`. Noise in each dimension assumed to be
                independent i.e. a diagonal noise covariance. If zero or None
                deterministic dynamics are assumed.
            obser_noise_std (float): Standard deviation of additive Gaussian
                noise in observations. Either a scalar or array of shape
                `(dim_z,)`. Noise in each dimension assumed to be independent
                i.e. a diagonal noise covariance.
        """
        self.init_state_mean = init_state_mean
        self.init_state_std = init_state_std
        if state_noise_std is None or np.all(state_noise_std == 0.):
            self.deterministic_state_update = True
        else:
            self.deterministic_state_update = False
            self.state_noise_std = state_noise_std
        self.obser_noise_std = obser_noise_std
        super(DiagonalGaussianModel, self).__init__(
            dim_z=dim_z, dim_x=dim_x, rng=rng)

    def init_state_sampler(self, n=None):
        if n is None:
            return (
                self.init_state_mean +
                self.rng.normal(size=(self.dim_z,)) * self.init_state_std
            )
        else:
            return (
                self.init_state_mean +
                self.rng.normal(size=(n, self.dim_z)) * self.init_state_std
            )

    def next_state_func(self, z, t):
        """Computes mean of next state given current state.

        Implements determinstic component of state update dynamics with new
        state calculated by output of this function plus additive zero-mean
        Gaussian noise. For models with fully-determinstic state dynamics
        no noise is added so this function exactly calculates the next state.

        Args:
            z (array): Current state vector.
            t (integer): Current time index.

        Returns:
            Array corresponding to (mean of) state at next time index t + 1.
        """
        raise NotImplementedError()

    def observation_func(self, z, t):
        """Computes mean of current observation state given current state.

        Implements determinstic component of observation process with
        observation calculated by output of this function plus additive
        zero-mean Gaussian noise.

        Args:
            z (array): Current state vector.
            t (integer): Current time index.

        Returns:
            Array corresponding to (mean of) observations at time index t.
        """
        raise NotImplementedError()

    def next_state_sampler(self, z, t):
        if self.deterministic_state_update:
            return self.next_state_func(z, t)
        else:
            return (
                self.next_state_func(z, t) +
                self.state_noise_std * self.rng.normal(size=z.shape)
            )

    def observation_sampler(self, z, t):
        if z.ndim == 2:
            return (
                self.observation_func(z, t) +
                self.rng.normal(size=(z.shape[0], self.dim_x)) *
                self.obser_noise_std
            )
        else:
            return (
                self.observation_func(z, t) +
                self.rng.normal(size=(self.dim_x)) * self.obser_noise_std
            )

    def log_prob_dens_init_state(self, z):
        return -(
            0.5 * ((z - self.init_state_mean) / self.init_state_std)**2 +
            0.5 * np.log(2 * np.pi) + np.log(self.init_state_std)
        ).sum(-1)

    def log_prob_dens_state_trans(self, z_n, z_c, t):
        if self.deterministic_state_update:
            raise DensityNotDefinedError('Deterministic state transition.')
        else:
            return -(
                0.5 * ((z_n - self.next_state_func(z_c, t)) /
                       self.state_noise_std)**2 +
                0.5 * np.log(2 * np.pi) + np.log(self.state_noise_std)
            ).sum(-1)

    def log_prob_dens_obs_gvn_state(self, x, z, t):
        return -(
            0.5 * ((x - self.observation_func(z)) /
                   self.obser_noise_std)**2 +
            0.5 * np.log(2 * np.pi) + np.log(self.obser_noise_std)
        ).sum(-1)


@inherit_docstrings
class DiagonalGaussianIntegratorModel(DiagonalGaussianModel):
    """Model with integrator state update and diagonal Gaussian distributions.

    Assumes the model dynamics take the form

        z[0] = init_state_mean + init_state_std * u[0]
        x[0] = observation_func(x[0], 0) + obser_noise_std * v[0]
        for t in range(1, n_steps):
            z[t] = next_state_func(z[t-1], t-1) + state_noise_std * u[t]
            x[t] = observation_func(x[t], t) + obser_noise_std * v[t]

    where

       z[t]: unobserved system state vector at time index t,
       x[t]: observations vector at time index t,
       u[t]: zero-mean identity covariance Gaussian state noise vector at time
             index t,
       v[t]: zero-mean identity covariance Gaussian observation noise vector
             at time index t,
       init_state_mean: initial state distribution mean vector,
       init_state_std: initial state distribution standard deviation vector,
       observation_func: function mapping from states to (pre-noise)
           observations,
       obser_noise_std: observation noise distribution standard deviation
           vector,
       next_state_func: function impleting state update dynamics by forward
           integration of a ODE system,
       state_noise_std: state noise distribution standard deviation vector.
           May be all zeros which corresponds to a model with deterministic
           state update dynamics.

    This corresponds to assuming the initial state distribution, conditional
    distribution of the next state given current and conditional distribution
    of the current observation given current state all take the form of
    multivariate Gaussian distributions with diagonal covariances. In the case
    of deterministic state update dynamics the conditional distribution of the
    next state given the current state will be a Dirac measure with all mass
    located at the forward map of the current state through the state dynamics
    function and so will not have a well-defined probability density function.
    """

    def __init__(self, integrator, dim_z, dim_x, rng,
                 init_state_mean, init_state_std,
                 state_noise_std, obser_noise_std):
        """
        Args:
            integrator (object): Integrator for model state dynamics. Object
                should define a `forward_integrate` function with function
                signature
                    def forward_integrate(self, z_curr, z_next, time)
                with `z_curr` an input array of a batch of state vectors at
                current time index, `z_next` an output array to write the
                values of the batch of state vectors at the next time index
                and `time` a float defining the current real time (*not*
                time index).
            dim_z (integer): Dimension of model state vector.
            dim_x (integer): Dimension of observation vector.
            rng (RandomState): Numpy RandomState random number generator.
            init_state_mean (float or array): Initial state distribution mean.
                Either a scalar or array of shape `(dim_z,)`.
            init_state_std (float or array): Initial state distribution
                standard deviation. Either a scalar or array of shape
                `(dim_z,)`. Each state dimension is assumed to be independent
                i.e. a diagonal covariance.
            state_noise_std (float or array): Standard deviation of additive
                Gaussian noise in state update. Either a scalar or array of
                shape `(dim_z,)`. Noise in each dimension assumed to be
                independent i.e. a diagonal noise covariance. If zero or None
                deterministic dynamics are assumed.
            obser_noise_std (float): Standard deviation of additive Gaussian
                noise in observations. Either a scalar or array of shape
                `(dim_z,)`. Noise in each dimension assumed to be independent
                i.e. a diagonal noise covariance.
        """
        self.integrator = integrator
        super(DiagonalGaussianIntegratorModel, self).__init__(
            dim_z=dim_z, dim_x=dim_x, rng=rng,
            init_state_mean=init_state_mean, init_state_std=init_state_std,
            state_noise_std=state_noise_std, obser_noise_std=obser_noise_std
        )

    def next_state_func(self, z, t):
        z_next = np.empty(z.shape)
        time = t * self.dt * self.n_steps_per_update
        if z.ndim == 1:
            self.integrator.forward_integrate(z[None], z_next[None], time)
        else:
            self.integrator.forward_integrate(z, z_next, time)
        return z_next
