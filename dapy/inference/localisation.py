"""Helper functions for localisation of filtering spatially extended models."""

import numpy as np
import numpy.linalg as la
import numba as nb


@nb.njit(nb.double[:](nb.double[:], nb.double), parallel=True)
def gaspari_and_cohn_weighting(d, radius):
    """Compactly supported smooth weighting kernel function.

    Args:
        d (array): One-dimensional array of distances.
        radius (float): Positive scaling parameter.

    Returns:
        One-dimensional array of weight values in [0, 1].

    References:
        1. Gaspari, G., & Cohn, S. E. (1999).
           Construction of correlation functions in two and three dimensions.
           Quarterly Journal of the Royal Meteorological Society,
           125(554), 723-757.
    """
    w = np.empty(d.shape[0])
    for i in nb.prange(d.shape[0]):
        u = abs(d[i]) / radius
        if u <= 1:
            w[i] = -u**5 / 4. + u**4 / 2. + 5 * u**3 / 8. - 5 * u**2 / 3. + 1
        elif u <= 2:
            w[i] = (u**5 / 12. - u**4 / 2. + 5 * u**3 / 8. + 5 * u**2 / 3. -
                    5 * u + 4 - 2 / (3 * u))
        else:
            w[i] = 0
    return w


@nb.njit(nb.double[:](nb.double[:], nb.double), parallel=True)
def triangular_weighting(d, radius):
    """Compactly supported piecewise linear weighting kernel function.

    Args:
        d (array): One-dimensional array of distances.
        radius (float): Positive scaling parameter.

    Returns:
        One-dimensional array of weight values in [0, 1].
    """
    w = np.empty(d.shape[0])
    for i in nb.prange(d.shape[0]):
        u = abs(d[i]) / radius
        if u <= 2.:
            w[i] = 1. - u / 2.
        else:
            w[i] = 0.
    return w


class LocalisationFunction(object):
    """Callable object for localisation of models on compact spatial domains.

    Assumes a Euclidean distance function between points.
    """

    def __init__(self, state_coords, obser_coords, localisation_radius,
                 weighting_function=gaspari_and_cohn_weighting,
                 use_cache=True):
        """
        Args:
            state_coords (array): Two-dimensional array of shape
                `(n_state_points, n_dim)` where `n_state_points` is the number
                of spatial points over which the system state is defined and
                `n_dim` is the spatial dimensionality. Each row corresponds to
                a set of spatial coordinates.
            obser_coords (array): Two-dimensional array of shape
                `(n_obser_points, n_dim)` where `n_obser_points` is the number
                of spatial points at which the system is observed and  `n_dim`
                is the spatial dimensionality. Each row corresponds to
                a set of spatial coordinates.
            localisation_radius (float): Positive scale parameter used to
                define distance over which observations are considered 'local'
                to a state spatial point.
            weighting_function (function): Function which given an array of
                distances and a scale parameter returns an array of weights
                between 0 and 1 with 1 corresponding to a zero distance.
            use_cache (boolean): Whether to cache the calculate observation
                indices and weights for each index to avoid recalculation on
                future calls. For large grids this may become memory-heavy.
        """
        self.state_coords = state_coords
        self.obser_coords = obser_coords
        self.localisation_radius = localisation_radius
        self.weighting_function = weighting_function
        self.use_cache = use_cache
        if use_cache:
            self._cache = [None] * state_coords.shape[0]

    def observation_distances(self, point):
        return ((point - self.obser_coords)**2).sum(-1)**0.5

    def __call__(self, index):
        if self.use_cache and self._cache[index] is not None:
            return self._cache[index]
        state_point = self.state_coords[index]
        dists = self.observation_distances(state_point)
        weights = self.weighting_function(dists, self.localisation_radius)
        nz_weights = weights > 0.
        indices = np.nonzero(nz_weights)[0]
        weights = weights[nz_weights]
        if self.use_cache:
            self._cache[index] = (indices, weights)
        return indices, weights


class PeriodicLocalisationFunction(LocalisationFunction):
    """Callable object for localisation of models on periodic spatial domains.

    Assumes a (wrapped) Euclidean distance function between points.
    """

    def __init__(self, state_coords, state_extents, obser_coords,
                 localisation_radius,
                 weighting_function=gaspari_and_cohn_weighting,
                 use_cache=True):
        """
        Args:
            state_coords (array): Two-dimensional array of shape
                `(n_state_points, n_dim)` where `n_state_points` is the number
                of spatial points over which the system state is defined and
                `n_dim` is the spatial dimensionality. Each row corresponds to
                a set of spatial coordinates.
            state_extents (tuple): Tuple giving extents of cuboidal spatial
                domain state is defined over with domain assumed to wrap at
                boundaries.
            obser_coords (array): Two-dimensional array of shape
                `(n_obser_points, n_dim)` where `n_obser_points` is the number
                of spatial points at which the system is observed and  `n_dim`
                is the spatial dimensionality. Each row corresponds to
                a set of spatial coordinates.
            localisation_radius (float): Positive scale parameter used to
                define distance over which observations are considered 'local'
                to a state spatial point.
            weighting_function (function): Function which given an array of
                distances and a scale parameter returns an array of weights
                between 0 and 1 with 1 corresponding to a zero distance.
            use_cache (boolean): Whether to cache the calculate observation
                indices and weights for each grid index to avoid recalculation
                on future calls. For large grids this may become memory-heavy.
        """
        super(PeriodicLocalisationFunction, self).__init__(
            state_coords=state_coords,
            obser_coords=obser_coords,
            localisation_radius=localisation_radius,
            weighting_function=weighting_function,
            use_cache=use_cache
        )
        self.state_extents = state_extents

    def observation_distances(self, point):
        deltas = np.abs(point - self.obser_coords)
        return (np.minimum(deltas,
                           self.state_extents - deltas)**2).sum(-1)**0.5
