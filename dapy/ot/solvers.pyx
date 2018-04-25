# coding: utf-8
# cython: language=c++
"""Cython wrapper for network simplex algorithm optimal transport solver.

Derived from files emd_wrapper.pyx and EMD_wrapper.cpp in the Python
Optimal Transport (POT) library by Remi Flammary
    https://github.com/rflamary/POT
The EMD_wrapper.cpp file was itself originally written by Antoine Rolet as
a wrapper around code written by Nicolas Boneel available at
    https://perso.liris.cnrs.fr/nicolas.bonneel/FastTransport/
which itself is based on an implementation of the network simplex algorithm
extracted from the C++ graph-based optimisation library LEMON
    http://lemon.cs.elte.hu/trac/lemon

Compared to the POT implementation this adds support for calculating a batch
of OT problems of the same dimensions in parallel using OpenMP and allows use
of non-C-contiguous arrays for the source and target histograms and cost
matrices by using memoryview interfaces to abstract indexing of the arrays.

License: MIT License
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free

ctypedef int Node
ctypedef long long Arc

cdef extern from "<iostream>":
    pass

# Default tolerance for checking equality of target and source dist. sums
cdef double DEFAULT_SUM_DIFF_TOLERANCE = 1e-8
# Default maximum number of iterations of network simplex algorithm
cdef int DEFAULT_MAX_ITER = 100000

cdef extern from "full_bipartitegraph.h" namespace "lemon" nogil:
    cdef cppclass FullBipartiteDigraph:
        FullBipartiteDigraph() except +
        FullBipartiteDigraph(int, int) except +
        @staticmethod
        Arc arcFromId(int)
        @staticmethod
        void next(Arc&)
        void first(Arc&)
        Node source(Arc)
        Node target(Arc)

ctypedef FullBipartiteDigraph Digraph

cdef extern from "network_simplex_simple.h":
    cdef int INVALID

cdef extern from "network_simplex_simple.h" namespace "lemon" nogil:
    cdef cppclass NetworkSimplexSimple[G, V, C, N]:
        NetworkSimplexSimple(G, bool, int, long long, int) except +
        NetworkSimplexSimple& supplyMap(V*, int, V*, int)
        NetworkSimplexSimple& setCost(Arc, C)
        int run()
        V flow(Arc&)
        C potential(Node&)

ctypedef NetworkSimplexSimple[
    Digraph, double, double, long] NetworkSimplex

# Enumeration of network simplex algorithm return codes
cdef enum ProblemType:
    INFEASIBLE  # The problem has no feasible solution
    OPTIMAL  # The algorithm converged to an optimal solution
    UNBOUNDED  # The objective function of the problem is unbounded
    MAX_ITER_REACHED  # The maximum number of iterations has been reached


def get_result_code_strings(result_code):
    """
    Translates network simplex algorithm result code to description string.

    Args:
        result_code (int): Integer result code from running network simplex
            algorithm on optimal transport problem.

    Returns:
        Tuple with first entry a single word description of result code and the
        second entry a longer textual description.
    """
    if result_code == INFEASIBLE:
        return ('Infeasible',
                'The problem has no feasible solution. Source or target '
                'distribution entries may be non-negative or sum to different '
                'values.')
    elif result_code == OPTIMAL:
        return 'Optimal', 'The algorithm converged to an optimal solution.'
    elif result_code == UNBOUNDED:
        return 'Unbounded', 'The problem objective function is unbounded.'
    elif result_code == MAX_ITER_REACHED:
        return ('Unconverged',
                'The maximum number of iterations was reached in the solver '
                'before convergence. Computed transport matrix and cost may '
                'be non-optimal.')


# Wrapper for external C++ code setting up optimal transport problems and
# running solvers. Not for direct use in Python code - convenience Functions
# defined below should be used instead.
cdef int _solve_optimal_transport_network_simplex(
        double[:] source_dist, double[:] target_dist,
        double[:, :] cost_matrix, double[:, :] trans_matrix,
        double* cost, int max_iter, double sum_diff_tolerance) nogil:

    cdef int i, j, result_code
    cdef double val, flow
    cdef Arc a = 0
    cdef int n_nonzero_source = 0, n_nonzero_target = 0
    cdef double sum_diff = 0.

    # Get number of sources and targets
    cdef int n_source = source_dist.shape[0]
    cdef int n_target = target_dist.shape[0]

    # Initialise transport matrix to zeros (only non-zero values later updated)
    for i in range(n_source):
        for j in range(n_target):
            trans_matrix[i, j] = 0.

    # Compute the number of non-zero entries in source and target arrays,
    # check all entries are non-negative and compute difference of sums
    for i in range(n_source):
        val = source_dist[i];
        if val > 0:
            sum_diff += val
            n_nonzero_source += 1
        elif val < 0:
            return INFEASIBLE

    for i in range(n_target):
        val = target_dist[i]
        if val > 0:
            sum_diff -= val
            n_nonzero_target += 1
        elif val < 0:
            return INFEASIBLE

    # check difference of sums of source and target arrays within threshhold
    if sum_diff > sum_diff_tolerance or sum_diff < -sum_diff_tolerance:
        return INFEASIBLE

    # Construct the flow graph on the stack
    cdef Digraph graph = Digraph(n_nonzero_source, n_nonzero_target)

    # Construct the network simplex algorithm object on the heap
    # (Cython cannot construct object without default no-argument constructor
    # on stack)
    cdef NetworkSimplex *net = new NetworkSimplex(
        graph, True, n_nonzero_source + n_nonzero_target,
        n_nonzero_source * n_nonzero_target, max_iter)

    # Raise memory error if returned null pointer
    if not net:
        with gil:
            raise MemoryError()

    # Allocate arrays for non-zero source / target indices and weights
    cdef int* indices_source = <int*> malloc(sizeof(int) * n_nonzero_source)
    cdef int* indices_target = <int*> malloc(sizeof(int) * n_nonzero_target)
    cdef double* weights_source = <double*> malloc(
        sizeof(double) * n_nonzero_source)
    cdef double* weights_target = <double*> malloc(
        sizeof(double) * n_nonzero_target)

    # Raise memory error if any allocations returned null pointer
    if not (indices_source and indices_target and
            weights_source and weights_target):
        with gil:
            raise MemoryError()

    try:

        # Set supply and demand, not accounting for zero values for efficiency
        j = 0
        for i in range(n_source):
            val = source_dist[i];
            if val > 0:
                weights_source[j] = val
                indices_source[j] = i
                j += 1
        j = 0
        for i in range(n_target):
            val = target_dist[i]
            if val > 0:
                weights_target[j] = -val  # demand is negative supply
                indices_target[j] = i
                j += 1
        net.supplyMap(weights_source, n_nonzero_source,
                      weights_target, n_nonzero_target)

        # Set the cost of each edge
        for i in range(n_nonzero_source):
            for j in range(n_nonzero_target):
                val = cost_matrix[indices_source[i], indices_target[j]]
                net.setCost(graph.arcFromId(i * n_nonzero_target + j), val)

        # Solve the problem with the network simplex algorithm
        result_code = net.run()

        # If algorithm converged or reached iteration limit get
        # computed cost and transport matric values and return
        if result_code == OPTIMAL or result_code == MAX_ITER_REACHED:
            cost[0] = 0
            graph.first(a)
            while a != INVALID:
                i = indices_source[graph.source(a)]
                j = indices_target[graph.target(a) - n_nonzero_source]
                flow = net.flow(a)
                cost[0] += flow * cost_matrix[i, j]
                trans_matrix[i, j] = flow
                graph.next(a)
        return result_code

    finally:

        # Ensure heap allocated object memory always freed
        del net
        free(indices_source)
        free(indices_target)
        free(weights_source)
        free(weights_target)


def solve_optimal_transport_network_simplex(
        double[:] source_dist, double[:] target_dist,
        double[:, :] cost_matrix, int max_iter=DEFAULT_MAX_ITER,
        double sum_diff_tolerance=DEFAULT_SUM_DIFF_TOLERANCE):
    """
    Solve optimal transport problem using network simplex algorithm.

    Exactly solves a discrete optimal transport problem of the form

        minimise expected_cost = sum(trans_matrix * cost_matrix)
        with respect to trans_matrix
        subject to all(trans_matrix.sum(1) == source_dist))
               and all(trans_matrix.sum(0) == target_dist))
               and all(trans_matrix >= 0)

    using an implementation of the network simplex algorithm [1,2] from the C++
    graph template library LEMON [3].

    source_dist and target_dist are 1D arrays of shape (n_source,) and
    (n_target,) respectively with non-negative entries which specify the
    source and target distributions to solve an optimal transport problem for.
    For the optimal transport problem to have a solutions it is necessary that
        sum(target_dist) == sum(source_dist)
    i.e. that the source and target distributions sum to the same value.
    Typically both arrays will represent probability distributions and sum to
    one however technically the only requirement is that the total 'mass' in
    both arrays is the same such that there exists a valid transport plan to
    transport all the mass from the sources to targets.

    cost_matrix is a 2D array of shape (n_source, n_target) with the entry
    cost_matrix[s, t] specifying the cost of moving the s'th source to the
    t'th target. Typically the entries will correspond to some distrance
    measure between the underlying points representing the source and targets.

    Args:
        source_dists (1D array): Array of source distribution to solve for of
            shape (n_source,).
        target_dists (1D array): Array of target distribution to solve for of
            shape (n_target,).
        cost_matrices (2D array): Array of cost matrix to solve for of shape
            (n_source, n_target).
        max_iter (int): Maximum number of iterations to run network simplex
            algorithm for.
        sum_diff_tolerance (double): Tolerance for check on equality of sums of
            target and source distribution entries.

    Returns:
        trans_matrix (2D array): Computed optimal transport matrix of shape
            (n_source, n_target).
        cost (double): Computed optimal expected transport cost.
        results_code (int): Integer results code indicating outcome of running
            network simplex algorithm on optimal transport problem. The
            integer code can be translated to a string description using the
            get_result_code_strings function. A value of 1 indicates the
            solver converged to an optimal solution for the problem.

    References:
        1. James B. Orlin (1997). A polynomial time primal network simplex
           algorithm for minimum cost flows. Mathematical Programming,
           78 (2): 109–129. doi:10.1007/BF02614365.
        2. Nicolas Bonneel, Michiel van De Panne, Sylvain Paris and Wolfgang
           Heidrich (2011). Displacement interpolation using Lagrangian mass
           transport. ACM Transactions on Graphics, 30 (6): 158:1-158:12.
           doi:10.1145/2070781.2024192.
        3. Balázs Dezső, Alpár Jüttner and Péter Kovács (2011). LEMON – an Open
           Source C++ Graph Template Library. Electronic Notes in Theoretical
           Computer Science, 264:23-45. http://lemon.cs.elte.hu/trac/lemon
    """
    cdef int result_code
    cdef double cost
    cdef int n_source = cost_matrix.shape[0]
    cdef int n_target = cost_matrix.shape[1]

    # Check input arguments are dimensionally consistent
    assert source_dist.shape[0] == n_source, (
        'source_dist shape incompatible with cost_matrix.')
    assert target_dist.shape[0] == n_target, (
        'target_dist shape incompatible with cost_matrix.')

    # Define numpy array to store computed transport matrix
    cdef np.ndarray[double, ndim=2, mode='c'] trans_matrix = np.empty(
        (n_source, n_target), dtype='double', order='C')

    # Solve optimal transport problem
    result_code = _solve_optimal_transport_network_simplex(
        source_dist, target_dist, cost_matrix, trans_matrix, &cost, max_iter,
        sum_diff_tolerance)

    return trans_matrix, cost, result_code


def solve_optimal_transport_network_simplex_batch(
        double[:, :] source_dists, double[:, :] target_dists,
        double[:, :, :] cost_matrices, int max_iter=DEFAULT_MAX_ITER,
        double sum_diff_tolerance=DEFAULT_SUM_DIFF_TOLERANCE, int n_thread=1):
    """
    Solve batch of optimal transport problems using network simplex algorithm.

    Exactly solves multiple discrete optimal transport problems of the form

        for source_dist, target_dist, cost_matrix in zip(
                source_dists, target_dists, cost_matrices):
            minimise expected_cost = sum(trans_matrix * cost_matrix)
            with respect to trans_matrix
            subject to all(trans_matrix.sum(1) == source_dist))
                   and all(trans_matrix.sum(0) == target_dist))
                   and all(trans_matrix >= 0)

    using an implementation of the network simplex algorithm [1,2] from the C++
    graph template library LEMON [3].

    source_dists and target_dists are 2D arrays of shape (n_problem, n_source)
    and (n_problem, n_target) respectively with non-negative entries which
    specify the set of n_problem (equal-dimensioned) source and target
    distributions to solve optimal transport problems for. For the optimal
    transport problems to have solutions it is necessary that
        all(sum(target_dists, 1) == sum(source_dists, 1))
    i.e. each pair of source and target distributions sum to the same value.
    Typically both arrays will represent probability distributions and sum to
    one however technically the only requirement is that the total 'mass' in
    both arrays is the same such that there exists a valid transport plan to
    transport all the mass from the sources to targets.

    cost_matrices is a 3D array of shape (n_problem, n_source, n_target) with
    the entry cost_matrix[p, s, t] specifying the cost of moving the s'th
    source to the t'th target in the p'th problem. Typically the entries will
    correspond to some distrance measure between the underlying points
    representing the source and targets.

    Args:
        source_dists (2D array): Array of source distributions to solve for of
            shape (n_problem, n_source).
        target_dists (2D array): Array of target distributions to solve for of
            shape (n_problem, n_target).
        cost_matrices (3D array): Array of cost matrices to solve for of shape
            (n_problem, n_source, n_target).
        max_iter (int): Maximum number of iterations to run network simplex
            algorithm instances for.
        sum_diff_tolerance (double): Tolerance for check on equality of sums of
            target and source distribution entries for each problem.
        n_thread (int): Number of parallel threads to distribute solving of
            independent optimal transport problems over.

    Returns:
        trans_matrices (3D array): Array of computed optimal transport
            matrices of shape (n_problem, n_source, n_target).
        costs (1D array): Array of computed optimal expected transport costs
            of shape (n_problem,).
        results_codes (1D array): Array of results codes indicating outcome of
            running network simplex algorithm on optimal transport problems of
            shape (n_problem,). The integer codes can be translated to string d
            descriptions using the get_result_code_strings function. A value
            of 1 indicates the solver converged to an optimal solution for the
            problem.

    References:
        1. James B. Orlin (1997). A polynomial time primal network simplex
           algorithm for minimum cost flows. Mathematical Programming,
           78 (2): 109–129. doi:10.1007/BF02614365.
        2. Nicolas Bonneel, Michiel van De Panne, Sylvain Paris and Wolfgang
           Heidrich (2011). Displacement interpolation using Lagrangian mass
           transport. ACM Transactions on Graphics, 30 (6): 158:1-158:12.
           doi:10.1145/2070781.2024192.
        3. Balázs Dezső, Alpár Jüttner and Péter Kovács (2011). LEMON – an Open
           Source C++ Graph Template Library. Electronic Notes in Theoretical
           Computer Science, 264:23-45. http://lemon.cs.elte.hu/trac/lemon
    """
    cdef int n_problem = cost_matrices.shape[0]
    cdef int n_source = cost_matrices.shape[1]
    cdef int n_target = cost_matrices.shape[2]
    cdef int p = 0

    # Check input arguments are dimensionally consistent
    assert (source_dists.shape[0] == n_problem and
            source_dists.shape[1] == n_source), (
            'source_dists shape incompatible with cost_matrices')
    assert (target_dists.shape[0] == n_problem and
            target_dists.shape[1] == n_target), (
            'target_dists shape incompatible with cost_matrices')

    # Define numpy arrays to store computed results
    cdef np.ndarray[int, ndim=1, mode='c'] result_codes = np.empty(
        n_problem, dtype='int32')
    cdef np.ndarray[double, ndim=1, mode='c'] costs = np.empty(
        n_problem, dtype='double')
    cdef np.ndarray[double, ndim=3, mode='c'] trans_matrices = np.empty(
        (n_problem, n_source, n_target), dtype='double', order='C')

    # Define memoryviews of numpy arrays to allow slicing without GIL in
    # parallel loop
    cdef int[:] result_codes_mv = result_codes
    cdef double[:] costs_mv = costs
    cdef double[:, :, :] trans_matrices_mv = trans_matrices

    # Iterate over optimal transport problems, potentially distributing over
    # multiple parallel threads
    for p in prange(n_problem, num_threads=n_thread, schedule='static',
                    nogil=True):
        result_codes_mv[p] = _solve_optimal_transport_network_simplex(
            source_dists[p], target_dists[p], cost_matrices[p],
            trans_matrices_mv[p], &costs_mv[p], max_iter, sum_diff_tolerance)

    return trans_matrices, costs, result_codes
