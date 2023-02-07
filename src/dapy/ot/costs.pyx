import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void calculate_cost_matrices_1d_in_place(
    double[:, :, :] cost_matrices,
    double[:, :, :] meshed_state_particles,
    int num_patch,
    int half_overlap,
    int subsample,
    int num_thread
):
    cdef int num_particle = meshed_state_particles.shape[0]
    cdef int dim_field = meshed_state_particles.shape[1]
    cdef int mesh_size = meshed_state_particles.shape[2]
    cdef int block_width = mesh_size // num_patch
    cdef int patch_width = block_width + 2 * half_overlap
    cdef int num_subsample_node = patch_width // subsample
    cdef int p, i, j, k, l, start_node_index, node_index
    for p in prange(num_patch, num_threads=num_thread, schedule='static', nogil=True):
        start_node_index = p * block_width - half_overlap
        if start_node_index < 0:
            start_node_index = start_node_index + mesh_size
        for i in range(num_particle):
            for j in range(i):
                for k in range(num_subsample_node):
                    node_index = (start_node_index + k * subsample) % mesh_size
                    for l in range(dim_field):
                        cost_matrices[p, i, j] += (
                            meshed_state_particles[i, l, node_index] -
                            meshed_state_particles[j, l, node_index])**2
                cost_matrices[p, j, i] = cost_matrices[p, i, j]


def calculate_cost_matrices_1d(
    double[:, :, :] meshed_state_particles,
    int num_patch,
    int half_overlap,
    int subsample=1,
    int num_thread=1
):
    cdef int num_particle = meshed_state_particles.shape[0]
    cdef np.ndarray[double, ndim=3, mode='c'] cost_matrices = np.zeros(
        (num_patch, num_particle, num_particle), dtype='double', order='C')
    cdef double[:, :, :] cost_matrices_mv = cost_matrices
    calculate_cost_matrices_1d_in_place(
        cost_matrices_mv, meshed_state_particles, num_patch,
        half_overlap, subsample, num_thread
    )
    return cost_matrices


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef calculate_cost_matrices_2d_in_place(
        double[:, :, :] cost_matrices,
        double[:, :, :] meshed_state_particles,
        int mesh_shape_0,
        int mesh_shape_1,
        int pou_shape_0,
        int pou_shape_1,
        int half_overlap_0,
        int half_overlap_1,
        int subsample,
        int num_thread,
):
    cdef int num_particle = meshed_state_particles.shape[0]
    cdef int dim_field = meshed_state_particles.shape[1]
    cdef int mesh_size = meshed_state_particles.shape[2]
    cdef int num_patch = pou_shape_0 * pou_shape_1
    cdef int block_shape_0 = mesh_shape_0 // pou_shape_0
    cdef int block_shape_1 = mesh_shape_1 // pou_shape_1
    cdef int patch_shape_0 = block_shape_0 + 2 * half_overlap_0
    cdef int patch_shape_1 = block_shape_1 + 2 * half_overlap_1
    cdef int num_subsample_node_0 = patch_shape_0 // subsample
    cdef int num_subsample_node_1 = patch_shape_1 // subsample
    cdef int i, j, b, p, q, k, l, m, n
    cdef int node_index_0, start_node_index_0, start_node_index_1
    for b in prange(num_patch, num_threads=num_thread, schedule='static', nogil=True):
        i = b // pou_shape_1
        j = b % pou_shape_1
        start_node_index_0 = i * block_shape_0 - half_overlap_0
        if start_node_index_0 < 0:
            start_node_index_0 = start_node_index_0 + mesh_shape_0
        start_node_index_1 = j * block_shape_1 - half_overlap_1
        if start_node_index_1 < 0:
            start_node_index_1 = start_node_index_1 + mesh_shape_1
        for p in range(num_particle):
            for q in range(p):
                for k in range(num_subsample_node_0):
                    node_index_0 = (start_node_index_0 + k * subsample) % mesh_shape_0
                    for l in range(num_subsample_node_1):
                        n = (node_index_0 * mesh_shape_1 + (
                            start_node_index_1 + l * subsample) % mesh_shape_1)
                        for m in range(dim_field):
                            cost_matrices[b, p, q] += (
                                meshed_state_particles[p, m, n] -
                                meshed_state_particles[q, m, n])**2
                cost_matrices[b, q, p] = cost_matrices[b, p, q]


def calculate_cost_matrices_2d(
    double[:, :, :] meshed_state_particles,
    int mesh_shape_0,
    int mesh_shape_1,
    int pou_shape_0,
    int pou_shape_1,
    int half_overlap_0,
    int half_overlap_1,
    int subsample=1,
    int num_thread=1
):
    cdef int num_particle = meshed_state_particles.shape[0]
    cdef int num_patch = pou_shape_0 * pou_shape_1
    cdef np.ndarray[double, ndim=3, mode='c'] cost_matrices = np.zeros(
        (num_patch, num_particle, num_particle), dtype='double', order='C')
    cdef double[:, :, :] cost_matrices_mv = cost_matrices
    calculate_cost_matrices_2d_in_place(
        cost_matrices_mv, meshed_state_particles, mesh_shape_0, mesh_shape_1,
        pou_shape_0, pou_shape_1, half_overlap_0, half_overlap_1, subsample, num_thread
    )
    return cost_matrices
