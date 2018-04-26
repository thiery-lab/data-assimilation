# coding: utf-8
"""Cython wrapper for basic batched exact optimal transport problem solvers.

Based on `emd_wrapper.pyx` + `emd.cpp` by Remi Flamary <remi.flamary@unice.fr>

Adapted by Matt Graham

License: MIT License
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


cdef extern from "emd.h":
    int emd(int n1,int n2, double *X, double *Y, double *D, double *G,
            double* alpha, double* beta, double *cost, int maxIter) nogil
    cdef enum ProblemType: INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED


def parallel_ot_solve(double[:, ::1] mu_vectors, double[:, ::1] nu_vectors,
                      double[:, :, ::1] cost_matrices, int max_iter=100000,
                      int num_threads=4):
    cdef int n_problem = cost_matrices.shape[0]
    cdef int mu_dim = cost_matrices.shape[1]
    cdef int nu_dim = cost_matrices.shape[2]
    cdef int p = 0

    cdef np.ndarray[int, ndim=1, mode='c'] result_codes = np.zeros(
        n_problem, dtype='int32')
    cdef np.ndarray[double, ndim=1, mode='c'] costs = np.zeros(
        n_problem, dtype='double')
    cdef np.ndarray[double, ndim=3, mode='c'] transport_matrices = np.zeros(
        (n_problem, mu_dim, nu_dim), dtype='double', order='C')
    cdef np.ndarray[double, ndim=2, mode='c'] alpha_vectors = np.zeros(
        (n_problem, mu_dim), dtype='double', order='C')
    cdef np.ndarray[double, ndim=2, mode='c'] beta_vectors = np.zeros(
        (n_problem, nu_dim), dtype='double', order='C')

    cdef int[::1] result_codes_mv = result_codes
    cdef double[::1] costs_mv = costs
    cdef double[:, :, ::1] transport_matrices_mv = transport_matrices
    cdef double[:, ::1] alpha_vectors_mv = alpha_vectors
    cdef double[:, ::1] beta_vectors_mv = beta_vectors

    loop_range = range(n_problem)

    for p in prange(n_problem, num_threads=num_threads, schedule='static',
                    nogil=True):
        result_codes_mv[p] = emd(
            mu_dim, nu_dim, &mu_vectors[p, 0], &nu_vectors[p, 0],
            &cost_matrices[p, 0, 0], &transport_matrices_mv[p, 0, 0],
            &alpha_vectors_mv[p, 0], &beta_vectors_mv[p, 0], &costs_mv[p],
            max_iter)

    return transport_matrices, costs, result_codes


def sequential_ot_solve(double[:, ::1] mu_vectors, double[:, ::1] nu_vectors,
                        double[:, :, ::1] cost_matrices, int max_iter=100000):
    cdef int n_problem = cost_matrices.shape[0]
    cdef int mu_dim = cost_matrices.shape[1]
    cdef int nu_dim = cost_matrices.shape[2]
    cdef int p = 0

    cdef np.ndarray[int, ndim=1, mode='c'] result_codes = np.zeros(
        n_problem, dtype='int32')
    cdef np.ndarray[double, ndim=1, mode='c'] costs = np.zeros(
        n_problem, dtype='double')
    cdef np.ndarray[double, ndim=3, mode='c'] transport_matrices = np.zeros(
        (n_problem, mu_dim, nu_dim), dtype='double', order='C')
    cdef np.ndarray[double, ndim=2, mode='c'] alpha_vectors = np.zeros(
        (n_problem, mu_dim), dtype='double', order='C')
    cdef np.ndarray[double, ndim=2, mode='c'] beta_vectors = np.zeros(
        (n_problem, nu_dim), dtype='double', order='C')

    cdef int[::1] result_codes_mv = result_codes
    cdef double[::1] costs_mv = costs
    cdef double[:, :, ::1] transport_matrices_mv = transport_matrices
    cdef double[:, ::1] alpha_vectors_mv = alpha_vectors
    cdef double[:, ::1] beta_vectors_mv = beta_vectors

    for p in range(n_problem):
        result_codes_mv[p] = emd(
            mu_dim, nu_dim, &mu_vectors[p, 0], &nu_vectors[p, 0],
            &cost_matrices[p, 0, 0], &transport_matrices_mv[p, 0, 0],
            &alpha_vectors_mv[p, 0], &beta_vectors_mv[p, 0], &costs_mv[p],
            max_iter)

    return transport_matrices, costs, result_codes
