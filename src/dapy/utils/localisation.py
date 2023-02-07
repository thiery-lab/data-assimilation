"""Locally supported weighting functions."""

import numpy as np
import numba as nb


@nb.njit(nb.double[:](nb.double[:], nb.double), parallel=True)
def gaspari_and_cohn_weighting(distances, radius):
    """Compactly supported smooth weighting kernel function.

    Args:
        distances: One-dimensional array of distances.
        radius: Positive scaling parameter determining distance over which weights are
            non-zero, with distances greater than `radius` mapping to zero weights and
            distances in `[0, radius]` mapping to non-zero weights that smoothly
            decrease from 1 to 0.

    Returns:
        One-dimensional array of weight values in [0, 1] with zero weights for all
        distances greater than radius i.e. `all(weights[d > radius] == 0) == True`.

    References:
        1. Gaspari, G., & Cohn, S. E. (1999).
           Construction of correlation functions in two and three dimensions.
           Quarterly Journal of the Royal Meteorological Society,
           125(554), 723-757.
    """
    weights = np.empty(distances.shape[0])
    for i in nb.prange(distances.shape[0]):
        u = 2 * abs(distances[i]) / radius
        if u <= 1:
            weights[i] = (
                -(u ** 5) / 4.0 + u ** 4 / 2.0 + 5 * u ** 3 / 8.0 - 5 * u ** 2 / 3.0 + 1
            )
        elif u <= 2:
            weights[i] = (
                u ** 5 / 12.0
                - u ** 4 / 2.0
                + 5 * u ** 3 / 8.0
                + 5 * u ** 2 / 3.0
                - 5 * u
                + 4
                - 2 / (3 * u)
            )
        else:
            weights[i] = 0
    return weights


@nb.njit(nb.double[:](nb.double[:], nb.double), parallel=True)
def triangular_weighting(distances, radius):
    """Compactly supported piecewise linear weighting kernel function.

    Args:
        distances: One-dimensional array of distances.
        radius: Positive scaling parameter determining distance over which weights are
            non-zero, with distances greater than `radius` mapping to zero weights and
            distances in `[0, radius]` mapping to non-zero weights that linearly
            decrease from 1 to 0.

    Returns:
        One-dimensional array of weight values in [0, 1] with zero weights for all
        distances greater than radius i.e. `all(weights[d > radius] == 0) == True`.
    """
    weights = np.empty(distances.shape[0])
    for i in nb.prange(distances.shape[0]):
        u = abs(distances[i]) / radius
        if u <= 1.0:
            weights[i] = 1.0 - u
        else:
            weights[i] = 0.0
    return weights


@nb.njit(nb.double[:](nb.double[:], nb.double), parallel=True)
def uniform_weighting(distances, radius):
    """Compactly supported uniform (rectangular) kernel function.

    Args:
        distances: One-dimensional array of distances.
        radius: Positive scaling parameter determining distance over which weights are
            non-zero, with distances greater than `radius` mapping to zero weights and
            distances in `[0, radius]` mapping to unit weights.

    Returns:
        One-dimensional array of weight values in [0, 1] with zero weights for all
        distances greater than radius i.e. `all(weights[d > radius] == 0) == True`.
    """
    weights = np.empty(distances.shape[0])
    for i in nb.prange(distances.shape[0]):
        u = abs(distances[i]) / radius
        if u <= 1.0:
            weights[i] = 1.0
        else:
            weights[i] = 0.0
    return weights
