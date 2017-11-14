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
            log_dens += self.log_prob_dens_state_trans(z_seq[t], z_seq[t-1], t)
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
            log_dens += self.log_prob_dens_state_trans(z_seq[t], z_seq[t-1], t)
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
        z_seq = np.empty((n_step, self.dim_z)) * np.nan
        x_seq = np.empty((n_step, self.dim_x)) * np.nan
        z_seq[0] = self.init_state_sampler()
        x_seq[0] = self.observation_sampler(z_seq[0], 0)
        for t in range(1, n_step):
            z_seq[t] = self.next_state_sampler(z_seq[t-1], t)
            x_seq[t] = self.observation_sampler(z_seq[t-1], t)
        return z_seq, x_seq
