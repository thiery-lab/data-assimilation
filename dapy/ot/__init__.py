"""Optimal transport solver functions."""

import warnings
import ot
import numpy as np
from dapy.ot.batch_solvers import sequential_ot_solve, parallel_ot_solve

DEFAULT_MAX_ITER = 100000


def log_sum_exp(x):
    """Compute logarithm of sum of exponents with improved numerical stability.

    Args:
        x (array): One-dimension arrray of values to calculate log-sum-exp of.

    Returns:
        Scalar corresponding to logarith of sum of exponents of x values.
    """
    x_max = x.max()
    return x_max + np.log(np.exp(x - x_max).sum())


def solve_optimal_transport_exact_batch(
        cost_matrices, target_marginals, max_iter=DEFAULT_MAX_ITER,
        num_threads=1):
    n_particle, n_bases = target_marginals.shape
    target_marginals /= target_marginals.sum(0)[None, :]
    mu = target_marginals.T
    nu = np.ones(mu.shape) / n_particle
    cost_matrices = np.rollaxis(cost_matrices, 2, 0)
    if num_threads == 1:
        trans_matrices, costs, results_codes = sequential_ot_solve(
            np.ascontiguousarray(mu), nu, np.ascontiguousarray(cost_matrices))
    else:
        trans_matrices, costs, results_codes = parallel_ot_solve(
            np.ascontiguousarray(mu), nu, np.ascontiguousarray(cost_matrices),
            max_iter=max_iter, num_threads=num_threads)
    return np.ascontiguousarray(trans_matrices.T * n_particle)


def solve_optimal_transport_sinkhorn_batch(
        cost_matrices, target_marginals, epsilon, n_iter):
    n_particle, n_bases = target_marginals.shape

    def modified_cost(u, v):
        return (-cost_matrices + u[:, None, :] + v[None, :, :]) / epsilon

    log_mu = -np.ones((n_particle, n_bases)) * np.log(n_particle)
    log_nu = (
        np.log(target_marginals) - np.log(target_marginals.sum(0))[None])
    u = 0 * log_mu
    v = 0 * log_nu
    for i in range(n_iter):
        u = epsilon * (log_mu - log_sum_exp(modified_cost(u, v), 1)) + u
        v = epsilon * (log_nu - log_sum_exp(modified_cost(u, v), 0)) + v
        pi = np.exp(modified_cost(u, v))
    max_marginal_error = np.max(np.abs(pi.sum(1) - 1. / n_particle))
    if max_marginal_error > 1e-8:
        warnings.warn('Poor Sinkhorn--Knopp convergence. '
                      'Max absolute marginal difference: ({0:.2e})'
                      .format(max_marginal_error))
    return pi * n_particle


def solve_optimal_transport_exact_pot(
        cost_matrices, target_marginals, max_iter=DEFAULT_MAX_ITER):
    n_particle, n_bases = target_marginals.shape
    u = ot.unif(n_particle)
    trans_matrices = np.empty((n_particle, n_particle, n_bases))
    target_marginals /= target_marginals.sum(0)[None, :]
    for r in range(n_bases):
        trans_matrices[:, :, r] = ot.emd(
            np.ascontiguousarray(target_marginals[:, r]), u,
            np.ascontiguousarray(cost_matrices[:, :, r]),
            numItermax=max_iter).T
    return trans_matrices * n_particle


def solve_optimal_transport_sinkhorn_pot(
        cost_matrices, target_marginals, epsilon, max_iter=DEFAULT_MAX_ITER):
    n_particle, n_bases = target_marginals.shape
    u = ot.unif(n_particle)
    trans_matrices = np.empty((n_particle, n_particle, n_bases))
    target_marginals /= target_marginals.sum(0)[None, :]
    for r in range(n_bases):
        trans_matrices[:, :, r] = ot.sinkhorn(
            np.ascontiguousarray(target_marginals[:, r]), u,
            np.ascontiguousarray(cost_matrices[:, :, r]),
            epsilon, 'sinkhorn_stabilized',
            numItermax=max_iter).T
    return trans_matrices * n_particle
