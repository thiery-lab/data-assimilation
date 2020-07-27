import numpy as np
import numba as nb


@nb.njit(nb.float64[:, :, :](
            nb.float64[:, :, :], nb.int64, nb.int64, nb.int64), parallel=True)
def calculate_cost_matrices_1d(z, num_patch, half_overlap, subsample):
    n_particle, n_field, n_node = z.shape
    block_width = n_node // num_patch
    cost_matrices = np.zeros((num_patch, n_particle, n_particle), np.float64)
    for p in range(num_patch):
        for i in range(n_particle):
            for j in range(n_particle):
                for k in range(p * block_width - half_overlap,
                               (p+1) * block_width + half_overlap, subsample):
                    for l in range(n_field):
                        cost_matrices[p, i, j] += (
                            z[i, l, k % n_node] - z[j, l, k % n_node])**2
    return cost_matrices


@nb.njit(nb.float64[:, :, :](
    nb.float64[:, :, ::1], nb.types.Tuple((nb.int64, nb.int64)),
    nb.types.Tuple((nb.int64, nb.int64)), nb.types.Tuple((nb.int64, nb.int64)),
    nb.int64), parallel=True)
def calculate_cost_matrices_2d(
        z, mesh_shape, pou_shape, half_overlap, subsample):
    m_0, m_1 = mesh_shape
    b_0, b_1 = pou_shape
    h_0, h_1 = half_overlap
    n_particle, n_field, n_node = z.shape
    z_2d = np.reshape(z, (n_particle, n_field,) + mesh_shape)
    num_patch = pou_shape[0] * pou_shape[1]
    cost_matrices = np.zeros((num_patch, n_particle, n_particle), np.float64)
    for i in range(pou_shape[0]):
        for j in range(pou_shape[1]):
            b = i * b_1 + j
            for p in range(n_particle):
                for q in range(n_particle):
                    for k in range(i*b_0 - h_0, (i+1) * b_0 + h_0, subsample):
                        for l in range(j * b_1 - h_1, (j+1) * b_1 + h_1,
                                       subsample):
                            for m in range(n_field):
                                cost_matrices[b, p, q] += (
                                    z_2d[p, m, k % m_0, l % m_1] -
                                    z_2d[q, m, k % m_0, l % m_1])**2
    return cost_matrices
