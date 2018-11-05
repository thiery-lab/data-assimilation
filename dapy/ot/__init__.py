"""Optimal transport solver functions."""

import warnings
import numpy as np
from scipy.special import logsumexp
from dapy.ot.solvers import (
    get_result_code_strings,
    solve_optimal_transport_network_simplex,
    solve_optimal_transport_network_simplex_batch)
try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False

DEFAULT_MAX_ITER = 100000
DEFAULT_SUM_DIFF_TOLERANCE = 1e-8


class ConvergenceError(Exception):
    """Error raised when optimal transport solver fails to converge."""


def pairwise_euclidean_distance(z1, z2):
    z1_sq_norms = np.sum(z1**2, -1)[:, None]
    z2_sq_norms = z1_sq_norms.T if z2 is z1 else np.sum(z2**2, -1)[None, :]
    return np.clip(
        z1_sq_norms - 2 * np.dot(z1, z2.T) + z2_sq_norms, 0, None)


def solve_optimal_transport_exact(
        source_dist, target_dist, cost_matrix, max_iter=DEFAULT_MAX_ITER,
        sum_diff_tolerance=DEFAULT_SUM_DIFF_TOLERANCE):
    trans_matrix, cost, results_code = (
        solve_optimal_transport_network_simplex(
            source_dist, target_dist, cost_matrix, max_iter=max_iter,
            sum_diff_tolerance=sum_diff_tolerance))
    if not results_code == 1:
        raise ConvergenceError('OT solution did not converge. {0}: {1}'
                               .format(*get_result_code_strings(results_code)))
    return trans_matrix


def solve_optimal_transport_exact_batch(
        source_dists, target_dists, cost_matrices, max_iter=DEFAULT_MAX_ITER,
        sum_diff_tolerance=DEFAULT_SUM_DIFF_TOLERANCE, n_thread=1):
    trans_matrices, costs, results_codes = (
        solve_optimal_transport_network_simplex_batch(
            source_dists, target_dists, cost_matrices, max_iter=max_iter,
            sum_diff_tolerance=sum_diff_tolerance, n_thread=n_thread))
    if not np.all(results_codes == 1):
        if POT_AVAILABLE:
            is_err = results_codes != 1
            trans_matrices[is_err] = solve_optimal_transport_exact_batch_pot(
                source_dists[is_err], target_dists[is_err],
                cost_matrices[is_err], max_iter)
        else:
            err_codes = np.unique(results_codes[results_codes != 1])
            raise ConvergenceError(
                '{0}/{1} of OT solutions did not converge. '
                'Errors: {2}.'
                .format(np.sum(results_codes != 1), source_dists.shape[0],
                        ['{0}: {1}'.format(*get_result_code_strings(e))
                         for e in err_codes]))
    return trans_matrices


def solve_optimal_transport_sinkhorn_batch(
        source_dists, target_dists, cost_matrices, epsilon, n_iter):

    def modified_cost(u, v):
        return (-cost_matrices + u[:, None, :] + v[:, :, None]) / epsilon

    log_mu = np.log(source_dists)
    log_nu = np.log(target_dists)
    u = 0 * log_mu
    v = 0 * log_nu
    for i in range(n_iter):
        u = epsilon * (log_mu - logsumexp(modified_cost(u, v), 2)) + u
        v = epsilon * (log_nu - logsumexp(modified_cost(u, v), 1)) + v
    trans_matrices = np.exp(modified_cost(u, v))
    max_marginal_error = np.max(np.abs(trans_matrices.sum(1) - source_dists))
    if max_marginal_error > 1e-8:
        warnings.warn('Poor Sinkhorn--Knopp convergence. '
                      'Max absolute marginal difference: ({0:.2e})'
                      .format(max_marginal_error))
    return trans_matrices


if POT_AVAILABLE:

    def solve_optimal_transport_exact_pot(
            source_dist, target_dist, cost_matrix, max_iter=DEFAULT_MAX_ITER):
        trans_matrix, log = ot.emd(
            np.ascontiguousarray(source_dist),
            np.ascontiguousarray(target_dist),
            np.ascontiguousarray(cost_matrix), numItermax=max_iter, log=True)
        if not log['result_code'] == 1:
            warnings.warn('OT solution did not converge. Error codes: {0}'
                          .format(log['result_code']))
        return trans_matrix

    def solve_optimal_transport_sinkhorn_pot(
            source_dist, target_dist, cost_matrix, epsilon,
            max_iter=DEFAULT_MAX_ITER):
        trans_matrix = ot.sinkhorn(
                np.ascontiguousarray(source_dists[p]),
                np.ascontiguousarray(target_dists[p]),
                np.ascontiguousarray(cost_matrices[p]), reg=epsilon,
                method='sinkhorn_stabilized', numItermax=max_iter)
        return trans_matrix

    def solve_optimal_transport_exact_batch_pot(
            source_dists, target_dists, cost_matrices,
            max_iter=DEFAULT_MAX_ITER):
        n_problem, n_particle = source_dists.shape
        trans_matrices = np.empty((n_problem, n_particle, n_particle))
        result_codes = np.empty((n_problem,))
        for p in range(n_problem):
            trans_matrices[p], log = ot.emd(
                np.ascontiguousarray(source_dists[p]),
                np.ascontiguousarray(target_dists[p]),
                np.ascontiguousarray(cost_matrices[p]), numItermax=max_iter,
                log=True)
            result_codes[p] = log['result_code']
        if not np.all(result_codes == 1):
            err_codes = np.unique(result_codes[result_codes != 1])
            warnings.warn('{0}/{1} of OT solutions did not converge. '
                          'Error codes: {2}'
                          .format(np.sum(result_codes != 1),
                                  source_dists.shape[0], err_codes))
        return trans_matrices

    def solve_optimal_transport_sinkhorn_batch_pot(
            source_dists, target_dists, cost_matrices, epsilon,
            max_iter=DEFAULT_MAX_ITER):
        n_problem, n_particle = source_dists.shape
        trans_matrices = np.empty((n_problem, n_particle, n_particle))
        for p in range(n_problem):
            trans_matrices[p] = ot.sinkhorn(
                np.ascontiguousarray(source_dists[p]),
                np.ascontiguousarray(target_dists[p]),
                np.ascontiguousarray(cost_matrices[p]), reg=epsilon,
                method='sinkhorn_stabilized', numItermax=max_iter)
        return trans_matrices
