"""Partition of unity bases functions for scalable localisation."""

import numpy as np
import numpy.fft as fft
import numba as nb


class PerGridPointPartitionOfUnityBasis(object):
    """PoU on spatial grid with constant basis at each grid point."""

    def __init__(self, n_grid):
        self.n_grid = n_grid

    def integrate_against_bases(self, f):
        return f

    def split_into_patches_and_scale(self, f):
        return f.reshape(f.shape + (1,))

    def combine_patches(self, f_patches):
        return f_patches.reshape(f_patches.shape[:-1])


class ConstantPartitionOfUnityBasis(object):
    """Trivial single constant basis partition of unity."""

    def __init__(self, n_grid):
        self.n_grid = n_grid

    def split_into_patches_and_scale(self, f):
        return f.reshape(f.shape[:-1] + (1, -1))

    def integrate_against_bases(self, f):
        return self.split_into_patches_and_scale(f).sum(-1)

    def combine_patches(self, f_patches):
        return f_patches.reshape(f_patches.shape[:-2] + (-1,))


class Blocked1dPartitionOfUnityBasis(object):
    """PoU on 1D grid using constant bases on non-overlapping blocks."""

    def __init__(self, n_grid, n_bases):
        assert n_grid % n_bases == 0, 'n_bases should be factor of n_grid'
        self.n_grid = n_grid
        self.n_bases = n_bases

    def split_into_patches_and_scale(self, f):
        return f.reshape(f.shape[:-1] + (self.n_bases, -1))

    def integrate_against_bases(self, f):
        return self.split_into_patches_and_scale(f).sum(-1)

    def combine_patches(self, f_patches):
        return f_patches.reshape(f_patches.shape[:-2] + (self.n_grid,))


class Blocked2dPartitionOfUnityBasis(object):
    """PoU on 2D grid using constant bases on non-overlapping blocks."""

    def __init__(self, grid_shape, bases_grid_shape):
        assert (grid_shape[0] % bases_grid_shape[0] == 0 and
                grid_shape[1] % bases_grid_shape[1] == 0), (
                    'bases_grid_shape should be factors of grid_shape')
        self.grid_shape = grid_shape
        self.bases_grid_shape = bases_grid_shape
        self.n_grid = grid_shape[0] * grid_shape[1]
        self.n_bases = bases_grid_shape[0] * bases_grid_shape[1]
        self.n_block = self.n_grid // self.n_bases
        self.block_shape = (grid_shape[0] // bases_grid_shape[0],
                            grid_shape[1] // bases_grid_shape[1])

    def split_into_patches_and_scale(self, f):
        return f.reshape(
            f.shape[:-1] +
            (self.bases_grid_shape[0], self.block_shape[0],
             self.bases_grid_shape[1], self.block_shape[1])
        ).swapaxes(-3, -2).reshape(f.shape[:-1] + (self.n_bases, self.n_block))

    def integrate_against_bases(self, f):
        return self.split_into_patches_and_scale(f).sum(-1)

    def combine_patches(self, f_patches):
        return f_patches.reshape(
            f_patches.shape[:-2] + self.bases_grid_shape + self.block_shape
        ).swapaxes(-3, -2).reshape(f_patches.shape[:-2] + (self.n_grid,))


class GaspariAndCohn1dPartitionOfUnityBasis(object):
    """PoU on 1D grid using Gaspari and Cohn compact smooth basis functions."""

    def __init__(self, n_grid, n_bases):
        assert n_grid % n_bases == 0, 'n_bases should be a factor of n_grid'
        self.n_grid = n_grid
        self.n_bases = n_bases
        self.basis_spacing = n_grid // n_bases
        self.basis_width = 4 * self.basis_spacing
        u = ((np.arange(self.basis_spacing) + 0.5) / self.basis_spacing)
        corr = 0.5 * (u**2 - u) - 1. / (1 - 0.5 * (u**2 - u)) + 58. / 24
        self.basis_weights = np.empty(self.basis_width) * np.nan
        self.basis_weights[:self.basis_spacing] = (
            (-2 * u**5 + 8 * u**4 + u**3 + 2 * u**2 + 4 * u + 8) / 24. -
            2 / (6 - 3 * u)) / corr
        self.basis_weights[self.basis_spacing:2 * self.basis_spacing] = (
            (6 * u**5 - 18 * u**4 - 3 * u**3 + 17 * u**2 + 17 * u + 5) / 24.
            ) / corr
        self.basis_weights[2 * self.basis_spacing:3 * self.basis_spacing] = (
            (-6 * u**5 + 12 * u**4 + 15 * u**3 - 40 * u**2 + 24) / 24.) / corr
        self.basis_weights[3 * self.basis_spacing:] = (
            (2 * u**5 - 2 * u**4 - 13 * u**3 + 33 * u**2 - 33 * u + 21) / 24. -
            2 / (3 + 3 * u)) / corr

    def split_into_patches_and_scale(self, f):
        f_shifted_stack = np.stack([
            f,
            np.roll(f, shift=self.basis_spacing, axis=-1),
            np.roll(f, shift=2 * self.basis_spacing, axis=-1),
            np.roll(f, shift=3 * self.basis_spacing, axis=-1)
        ], axis=-2)
        return f_shifted_stack.reshape(
                    f.shape[:-1] + (self.n_bases, self.basis_width)
                ) * self.basis_weights

    def integrate_against_bases(self, f):
        return self.split_into_patches_and_scale(f).sum(-1)

    def combine_patches(self, f_patches):
        f_stack = f_patches.reshape(f_patches.shape[:-2] + (4, self.n_grid))
        return (
            f_stack[..., 0, :] +
            np.roll(f_stack[..., 1, :], shift=-self.basis_spacing, axis=-1) +
            np.roll(f_stack[..., 2, :], shift=-2*self.basis_spacing, axis=-1) +
            np.roll(f_stack[..., 3, :], shift=-3*self.basis_spacing, axis=-1)
        ).reshape(f_patches.shape[:-2] + (self.n_grid,))


def corrected_gaspari_and_cohn_weights(u, basis_spacing):
    corr = 0.5 * (u**2 - u) - 1. / (1 - 0.5 * (u**2 - u)) + 58. / 24
    basis_weights = np.empty(basis_spacing * 4) * np.nan
    basis_weights[:basis_spacing] = (
        (-2 * u**5 + 8 * u**4 + u**3 + 2 * u**2 + 4 * u + 8) / 24. -
        2 / (6 - 3 * u)) / corr
    basis_weights[basis_spacing:2 * basis_spacing] = (
        (6 * u**5 - 18 * u**4 - 3 * u**3 + 17 * u**2 + 17 * u + 5) / 24.
        ) / corr
    basis_weights[2 * basis_spacing:3 * basis_spacing] = (
        (-6 * u**5 + 12 * u**4 + 15 * u**3 - 40 * u**2 + 24) / 24.) / corr
    basis_weights[3 * basis_spacing:] = (
        (2 * u**5 - 2 * u**4 - 13 * u**3 + 33 * u**2 - 33 * u + 21) / 24. -
        2 / (3 + 3 * u)) / corr
    return basis_weights


class GaspariAndCohn2dPartitionOfUnityBasis(object):
    """PoU on 2D grid using Gaspari and Cohn compact smooth basis functions."""

    def __init__(self, grid_shape, bases_grid_shape):
        self.grid_shape = grid_shape
        self.n_grid = grid_shape[0] * grid_shape[1]
        self.bases_grid_shape = bases_grid_shape
        self.n_bases = bases_grid_shape[0] * bases_grid_shape[1]
        self.basis_spacing = np.array([
            grid_shape[0] // bases_grid_shape[0],
            grid_shape[1] // bases_grid_shape[1]])
        self.basis_dim = 4 * self.basis_spacing
        u0 = ((np.arange(self.basis_spacing[0]) + 0.5) / self.basis_spacing[0])
        u1 = ((np.arange(self.basis_spacing[1]) + 0.5) / self.basis_spacing[1])
        basis_weights_0 = corrected_gaspari_and_cohn_weights(
            u0, self.basis_spacing[0])
        basis_weights_1 = corrected_gaspari_and_cohn_weights(
            u1, self.basis_spacing[1])
        self.basis_weights = (
            basis_weights_0[:, None] * basis_weights_1[None, :]).flatten()

    def split_into_patches_and_scale(self, f):
        f_2d = np.reshape(f, (-1,) + self.grid_shape)
        f_2d_shifted_stack = []
        for i in range(4):
            for j in range(4):
                shift = (self.basis_spacing[0] * i, self.basis_spacing[1] * j)
                f_2d_shifted_stack.append(
                    np.roll(f_2d, shift=shift, axis=(-2, -1)))
        f_2d_shifted_stack = np.stack(f_2d_shifted_stack, axis=1)
        f_patches = f_2d_shifted_stack.reshape(
            f.shape[:-1] + (
                16,
                self.bases_grid_shape[0] // 4, self.basis_dim[0],
                self.bases_grid_shape[1] // 4, self.basis_dim[1])
        ).swapaxes(-3, -2).reshape(f.shape[:-1] + (self.n_bases, -1))
        return f_patches * self.basis_weights

    def integrate_against_bases(self, f):
        # return self.split_into_patches_and_scale(f).sum(-1)
        f_2d = np.reshape(f, (-1,) + self.grid_shape)
        integrals = []
        for i in range(4):
            for j in range(4):
                shift = (self.basis_spacing[0] * i, self.basis_spacing[1] * j)
                f_2d_shifted = np.roll(f_2d, shift=shift, axis=(-2, -1))
                f_patches = f_2d_shifted.reshape(f.shape[:-1] + (
                    self.bases_grid_shape[0] // 4, self.basis_dim[0],
                    self.bases_grid_shape[1] // 4, self.basis_dim[1])
                ).swapaxes(-3, -2).reshape(
                    f.shape[:-1] + (self.n_bases // 16, -1))
                integrals.append((f_patches * self.basis_weights).sum(-1))
        return np.concatenate(integrals, -1)

    def combine_patches(self, f_patches):
        f_2d_shifted_stack = f_patches.reshape((
            -1, 16,
            self.bases_grid_shape[0] // 4, self.bases_grid_shape[1] // 4,
            self.basis_dim[0], self.basis_dim[1]
        )).swapaxes(-3, -2).reshape((-1, 16) + self.grid_shape)
        f_2d_unshifted = 0
        for i in range(4):
            for j in range(4):
                shift = (-self.basis_spacing[0] * i,
                         -self.basis_spacing[1] * j)
                f_2d_unshifted += np.roll(
                    f_2d_shifted_stack[:, i*4+j], shift=shift, axis=(-2, -1))
        return f_2d_unshifted.reshape(f_patches.shape[:-2] + (self.n_grid,))


class AltGaspariAndCohn2dPartitionOfUnityBasis(object):
    """PoU on 2D grid using Gaspari and Cohn compact smooth basis functions."""

    def __init__(self, grid_shape, bases_grid_shape):
        self.grid_shape = grid_shape
        self.n_grid = grid_shape[0] * grid_shape[1]
        self.bases_grid_shape = bases_grid_shape
        self.n_bases = bases_grid_shape[0] * bases_grid_shape[1]
        self.basis_spacing = np.array([
            grid_shape[0] // bases_grid_shape[0],
            grid_shape[1] // bases_grid_shape[1]])
        self.basis_dim = 4 * self.basis_spacing
        u0 = ((np.arange(self.basis_spacing[0]) + 0.5) / self.basis_spacing[0])
        u1 = ((np.arange(self.basis_spacing[1]) + 0.5) / self.basis_spacing[1])
        basis_weights_0 = corrected_gaspari_and_cohn_weights(
            u0, self.basis_spacing[0])
        basis_weights_1 = corrected_gaspari_and_cohn_weights(
            u1, self.basis_spacing[1])
        basis_patches = []
        for i in range(4):
            for j in range(4):
                b0 = np.roll(basis_weights_0, (i - 1) * self.basis_spacing[0])
                b1 = np.roll(basis_weights_1, (j - 1) * self.basis_spacing[1])
                basis_patches.append(b0[:, None] * b1[None, :])
        self.basis_patches = np.array(basis_patches)

    def split_into_patches_and_scale(self, f):
        f_2d_patches = f.reshape((
            -1, 1,
            self.bases_grid_shape[0] // 4, self.basis_dim[0],
            self.bases_grid_shape[1] // 4, self.basis_dim[1])).swapaxes(3, 4)
        f_2d_scaled_patches = f_2d_patches * self.basis_patches[:, None, None]
        return f_2d_scaled_patches.reshape(f.shape[:-1] + (self.n_bases, -1))

    def integrate_against_bases(self, f):
        return self.split_into_patches_and_scale(f).sum(-1)

    def combine_patches(self, f_patches):
        return f_patches.reshape((
            -1, 16,
            self.bases_grid_shape[0] // 4, self.bases_grid_shape[1] // 4,
            self.basis_dim[0], self.basis_dim[1]
        )).swapaxes(3, 4).sum(1).reshape(f_patches.shape[:-2] + (self.n_grid,))


class SquaredCosine1dPartitionOfUnityBasis(object):
    """PoU on 1D grid using squared cosine compact smooth basis functions."""

    def __init__(self, n_grid, n_bases):
        self.n_grid = n_grid
        self.n_bases = n_bases
        self.patch_half_dim = n_grid // n_bases
        u = (((np.arange(n_grid) + 0.5) / n_grid)[None, :] *
             np.pi * self.n_bases / 2.)
        self.s = np.sin(u)**2
        self.c = np.cos(u)**2

    def split_into_patches_and_scale(self, f):
        f_1d = np.reshape(f, (-1, self.n_grid))
        f_1d_s = f_1d * self.s
        f_1d_c = np.roll(f_1d * self.c, shift=self.patch_half_dim, axis=1)
        f_1d_scaled_stack = np.stack([f_1d_s, f_1d_c], axis=1)
        return f_1d_scaled_stack.reshape(
            f.shape[:-1] + (self.n_bases,  2 * self.patch_half_dim))

    def integrate_against_bases(self, f):
        f_1d = np.reshape(f, (-1, self.n_grid))
        f_1d_s = f_1d * self.s
        f_1d_c = np.roll(f_1d * self.c, shift=self.patch_half_dim, axis=1)
        f_1d_scaled_stack = np.stack([f_1d_s, f_1d_c], axis=1)
        return self.split_into_patches_and_scale(f).sum(-1)

    def combine_patches(self, f_patches):
        n_particle, dim_field = f_patches.shape[:2]
        f_1d_stacked = f_patches.reshape(
            (n_particle * dim_field, 2, self.n_grid))
        f_1d_s = f_1d_stacked[:, 0, :]
        f_1d_c = np.roll(
            f_1d_stacked[:, 1, :], shift=-self.patch_half_dim, axis=1)
        return (f_1d_s + f_1d_c).reshape((n_particle, dim_field, self.n_grid))


class SquaredCosine2dPartitionOfUnityBasis(object):

    def __init__(self, grid_shape, bases_grid_shape):
        self.grid_shape = grid_shape
        self.n_grid = grid_shape[0] * grid_shape[1]
        self.bases_grid_shape = bases_grid_shape
        self.n_bases = bases_grid_shape[0] * bases_grid_shape[1]
        self.patch_half_dim = (
            grid_shape[0] // bases_grid_shape[0],
            grid_shape[1] // bases_grid_shape[1])
        u0 = (
            ((np.arange(grid_shape[0]) + 0.5) / grid_shape[0])[None, :, None] *
            np.pi * self.bases_grid_shape[0] / 2.)
        u1 = (
            ((np.arange(grid_shape[1]) + 0.5) / grid_shape[1])[None, None, :] *
            np.pi * self.bases_grid_shape[1] / 2.)
        self.s0 = np.sin(u0)**2
        self.s1 = np.sin(u1)**2
        self.c0 = np.cos(u0)**2
        self.c1 = np.cos(u1)**2

    def integrate_against_bases(self, f):
        f_2d = np.reshape(f, (-1,) + self.grid_shape)
        f_2d_ss = (f_2d * self.s0) * self.s1
        f_2d_sc = np.roll(
            (f_2d * self.s0) * self.c1, shift=self.patch_half_dim[1], axis=2)
        f_2d_cs = np.roll(
            (f_2d * self.c0) * self.s1, shift=self.patch_half_dim[0], axis=1)
        f_2d_cc = np.roll(
            (f_2d * self.c0) * self.c1, shift=self.patch_half_dim, axis=(1, 2))
        f_2d_scaled_stack = np.stack(
            [f_2d_ss, f_2d_sc, f_2d_cs, f_2d_cc], axis=1)
        return f_2d_scaled_stack.reshape(
            (-1, 4, self.bases_grid_shape[0] // 2,  2 * self.patch_half_dim[0],
             self.bases_grid_shape[1] // 2, 2 * self.patch_half_dim[1])
        ).sum((3, 5)).reshape(f.shape[:-1] + (self.n_bases,))

    def split_into_patches_and_scale(self, f):
        n_particle, dim_field, n_grid = f.shape
        f_2d = f.reshape((-1,) + self.grid_shape)
        f_2d_ss = (f_2d * self.s0) * self.s1
        f_2d_sc = np.roll(
            (f_2d * self.s0) * self.c1, shift=self.patch_half_dim[1], axis=2)
        f_2d_cs = np.roll(
            (f_2d * self.c0) * self.s1, shift=self.patch_half_dim[0], axis=1)
        f_2d_cc = np.roll(
            (f_2d * self.c0) * self.c1, shift=self.patch_half_dim, axis=(1, 2))
        f_2d_scaled_stack = np.stack(
            [f_2d_ss, f_2d_sc, f_2d_cs, f_2d_cc], axis=1)
        return f_2d_scaled_stack.reshape(
            (n_particle * dim_field, 4,
             self.bases_grid_shape[0] // 2, 2 * self.patch_half_dim[0],
             self.bases_grid_shape[1] // 2, 2 * self.patch_half_dim[1])
        ).swapaxes(3, 4).reshape(
            (n_particle, dim_field, self.n_bases, 4 * n_grid // self.n_bases))

    def combine_patches(self, f_patches):
        n_particle, dim_field = f_patches.shape[:2]
        f_2d_stacked = f_patches.reshape((
            n_particle * dim_field, 4,
            self.bases_grid_shape[0] // 2, self.bases_grid_shape[1] // 2,
            2 * self.patch_half_dim[0], 2 * self.patch_half_dim[1]
        )).swapaxes(3, 4).reshape(
            (n_particle * dim_field, 4) + self.grid_shape)
        f_2d_ss = f_2d_stacked[:, 0, :, :]
        f_2d_sc = np.roll(
            f_2d_stacked[:, 1, :, :], shift=-self.patch_half_dim[1], axis=2)
        f_2d_cs = np.roll(
            f_2d_stacked[:, 2, :, :], shift=-self.patch_half_dim[0], axis=1)
        f_2d_cc = np.roll(
            f_2d_stacked[:, 3, :, :],
            shift=(-self.patch_half_dim[0], -self.patch_half_dim[1]),
            axis=(1, 2))
        return sum([f_2d_ss, f_2d_sc, f_2d_cs, f_2d_cc]).reshape(
            (n_particle, dim_field, self.n_grid))


@nb.njit(nb.float64[:, :, :](nb.float64[:, :, :, :], nb.int64,
                             nb.int64, nb.int64, nb.int64))
def _sum_overlapping_patches_1d(
        f_patches, block_width, kernel_width, n_patch, n_node):
    w = block_width
    k = kernel_width
    f_padded = np.zeros(f_patches.shape[:-2] + (n_node + k - 1,))
    for b in range(n_patch):
        f_padded[..., b*w:(b+1)*w+(k-1)] += f_patches[..., b, :]
    f_padded[..., (k-1)//2:(k-1)] += f_padded[..., -(k-1)//2:]
    f_padded[..., -(k-1):-(k-1)//2] += f_padded[..., :(k-1)//2]
    return f_padded[..., (k-1)//2:-(k-1)//2]


class SmoothedBlock1dPartitionOfUnityBasis(object):
    """PoU on 1D grid using block PoU convolved with smooth kernel."""

    def __init__(self, n_node, n_patch, kernel, offset=0):
        self.n_node = n_node
        self.n_patch = n_patch
        self.mesh_coords = (0.5 + np.arange(n_node)) / n_node
        self.block_width = n_node // n_patch
        self.kernel_width = kernel.shape[0]
        self.offset = offset
        kernel = kernel / kernel.sum()
        self.kernel = np.zeros(n_node)
        if self.kernel_width > 1:
            self.kernel[-(self.kernel_width - 1) // 2:] = kernel[
                :self.kernel_width // 2]
        self.kernel[:(self.kernel_width + 1) // 2] = kernel[
            self.kernel_width // 2:]
        self.rfft_kernel = fft.rfft(self.kernel)
        block = np.zeros(n_node)
        block[(n_node - self.block_width) // 2:
              (n_node + self.block_width) // 2] = 1
        self.patch_width = self.block_width + self.kernel_width - 1
        self.smoothed_bump = fft.irfft(self.rfft_kernel * fft.rfft(block))[
            (n_node - self.patch_width) // 2:
            (n_node + self.patch_width) // 2]
        self.patch_lims = np.full((self.n_patch, 2), np.nan)
        half_width = (self.kernel_width - 1) // 2
        patch_lindex = (
            offset + np.arange(n_patch) * self.block_width - half_width)
        self.patch_lims = np.stack(
            [patch_lindex, patch_lindex + self.patch_width], -1
        ) / self.n_node % 1.

    def split_into_patches(self, f):
        if self.offset != 0:
            f = np.roll(f, -self.offset, axis=-1)
        if self.kernel_width > 1:
            f_padded = np.zeros(
                f.shape[:-1] + (self.n_node + self.kernel_width - 1,))
            f_padded[..., :(self.kernel_width - 1) // 2] = f[
                ..., -(self.kernel_width - 1) // 2:]
            f_padded[..., -(self.kernel_width - 1) // 2:] = f[
                ..., :(self.kernel_width - 1) // 2]
            f_padded[..., (self.kernel_width - 1) // 2:
                     -(self.kernel_width - 1) // 2] = f
        else:
            f_padded = f.copy()
        shape = f.shape[:-1] + (self.n_patch, self.patch_width)
        strides = f_padded.strides[:-1] + (
            self.block_width * f.strides[-1], f.strides[-1])
        return np.lib.stride_tricks.as_strided(f_padded, shape, strides)

    def split_into_patches_and_scale(self, f):
        f_patches = self.split_into_patches(f)
        return f_patches * self.smoothed_bump

    def integrate_against_bases(self, f):
        f_1d = np.reshape(f, (-1, self.n_node))
        if self.offset != 0:
            f_1d = np.roll(f_1d, -self.offset, axis=-1)
        f_1d_smooth = fft.irfft(self.rfft_kernel * fft.rfft(f_1d))
        return f_1d_smooth.reshape(f.shape[:-1] + (self.n_patch, -1)).sum(-1)

    def combine_patches(self, f_patches):
        w = self.block_width
        k = self.kernel_width
        if k > 1:
            f_1d = _sum_overlapping_patches_1d(
                f_patches, w, k, self.n_patch, self.n_node)
        else:
            f_1d = f_patches.reshape(f_patches.shape[:-2] + (self.n_node,))
        if self.offset != 0:
            f_1d = np.roll(f_1d, self.offset, axis=-1)
        return f_1d

    def patch_distance(self, patch_index, coords):
        lim = self.patch_lims[patch_index]
        edge_dist = np.minimum((lim[0] - coords) % 1., (coords - lim[1]) % 1.)
        if lim[1] > lim[0]:
            not_in_patch = (coords < lim[0]) | (coords > lim[1])
            return not_in_patch * edge_dist
        else:
            not_in_patch = (coords < lim[0]) & (coords > lim[1])
            return not_in_patch * edge_dist


def _separable_fft_convolve_2d(x, fft_k_0, fft_k_1):
    x_ = fft.irfft(fft_k_0[:, None] * fft.rfft(x, axis=-2), axis=-2)
    return fft.irfft(fft_k_1 * fft.rfft(x_, axis=-1), axis=-1)


class SmoothedBlock2dPartitionOfUnityBasis(object):
    """PoU on 2D grid using block PoU convolved with smooth kernel."""

    def __init__(self, mesh_shape, pou_shape, kernel_weights):
        self.mesh_shape = mesh_shape
        self.pou_shape = pou_shape
        self.mesh_coords = np.stack((c.flat for c in np.meshgrid(
            (0.5 + np.arange(mesh_shape[0])) / mesh_shape[0],
            (0.5 + np.arange(mesh_shape[1])) / mesh_shape[1])), -1)
        self.n_node = mesh_shape[0] * mesh_shape[1]
        self.n_grid = self.n_node
        self.n_patch = pou_shape[0] * pou_shape[1]
        self.block_shape = (
            mesh_shape[0] // pou_shape[0], mesh_shape[1] // pou_shape[1])
        self.kernel_width = kernel_weights.shape[0]
        kernel_weights = kernel_weights / kernel_weights.sum()
        self.kernel_0 = np.zeros(mesh_shape[0])
        self.kernel_0[-(self.kernel_width - 1) // 2:] = kernel_weights[
            :self.kernel_width // 2]
        self.kernel_0[:(self.kernel_width + 1) // 2] = kernel_weights[
            self.kernel_width // 2:]
        self.rfft_kernel_0 = fft.rfft(self.kernel_0)
        self.kernel_1 = np.zeros(mesh_shape[1])
        self.kernel_1[-(self.kernel_width - 1) // 2:] = kernel_weights[
            :self.kernel_width // 2]
        self.kernel_1[:(self.kernel_width + 1) // 2] = kernel_weights[
            self.kernel_width // 2:]
        self.rfft_kernel_1 = fft.rfft(self.kernel_1)
        block = np.zeros(mesh_shape)
        block[(mesh_shape[0] - self.block_shape[0]) // 2:
              (mesh_shape[0] + self.block_shape[0]) // 2,
              (mesh_shape[1] - self.block_shape[1]) // 2:
              (mesh_shape[1] + self.block_shape[1]) // 2] = 1
        self.patch_shape = (
            self.block_shape[0] + self.kernel_width - 1,
            self.block_shape[1] + self.kernel_width - 1)
        self.smoothed_bump = _separable_fft_convolve_2d(
            block, self.rfft_kernel_0, self.rfft_kernel_1)[
                (mesh_shape[0] - self.patch_shape[0]) // 2:
                (mesh_shape[0] + self.patch_shape[0]) // 2,
                (mesh_shape[1] - self.patch_shape[1]) // 2:
                (mesh_shape[1] + self.patch_shape[1]) // 2]
        self.slice_mid = slice((self.kernel_width - 1) // 2,
                               -(self.kernel_width - 1) // 2)
        self.slice_e_0 = slice(None, (self.kernel_width - 1) // 2)
        self.slice_e_1 = slice(-(self.kernel_width - 1) // 2, None)
        self.patch_lims = np.full((2, self.n_patch, 2), np.nan)
        half_width = (self.kernel_width - 1) // 2
        for i in range(pou_shape[0]):
            for j in range(pou_shape[1]):
                self.patch_lims[0, i * pou_shape[0] + j] = (
                    ((i * self.block_shape[0] - half_width) /
                     self.mesh_shape[0]) % 1.,
                    (((i + 1) * self.block_shape[0] + half_width) /
                     self.mesh_shape[0]) % 1.)
                self.patch_lims[1, i * pou_shape[0] + j] = (
                    ((j * self.block_shape[1] - half_width) /
                     self.mesh_shape[1]) % 1.,
                    (((j + 1) * self.block_shape[1] + half_width) /
                     self.mesh_shape[1]) % 1.)

    def split_into_patches_and_scale(self, f):
        f_2d = np.reshape(f, f.shape[:-1] + self.mesh_shape)
        f_padded = np.zeros(
            f_2d.shape[:-2] + (self.mesh_shape[0] + self.kernel_width - 1,
                               self.mesh_shape[1] + self.kernel_width - 1))
        f_padded[..., self.slice_mid, self.slice_e_0] = (
            f_2d[..., :, self.slice_e_1])
        f_padded[..., self.slice_mid, self.slice_e_1] = (
            f_2d[..., :, self.slice_e_0])
        f_padded[..., self.slice_e_0, self.slice_mid] = (
            f_2d[..., self.slice_e_1, :])
        f_padded[..., self.slice_e_1, self.slice_mid] = (
            f_2d[..., self.slice_e_0, :])
        f_padded[..., self.slice_e_0, self.slice_e_0] = (
            f_2d[..., self.slice_e_1, self.slice_e_1])
        f_padded[..., self.slice_e_0, self.slice_e_1] = (
            f_2d[..., self.slice_e_1, self.slice_e_0])
        f_padded[..., self.slice_e_1, self.slice_e_0] = (
            f_2d[..., self.slice_e_0, self.slice_e_1])
        f_padded[..., self.slice_e_1, self.slice_e_1] = (
            f_2d[..., self.slice_e_0, self.slice_e_0])
        f_padded[..., self.slice_mid, self.slice_mid] = f_2d
        shape = f_padded.shape[:-2] + self.pou_shape + self.patch_shape
        strides = f_padded.strides[:-2] + (
            self.block_shape[0] * f_padded.strides[-2],
            self.block_shape[1] * f_padded.strides[-1]) + f_padded.strides[-2:]
        f_patches = np.lib.stride_tricks.as_strided(f_padded, shape, strides)
        return (f_patches * self.smoothed_bump).reshape(
            f.shape[:-1] + (self.n_patch, -1))

    def integrate_against_bases(self, f):
        f_2d = np.reshape(f, (-1,) + self.mesh_shape)
        f_2d_smooth = _separable_fft_convolve_2d(
            f_2d, self.rfft_kernel_0, self.rfft_kernel_1)
        return f_2d_smooth.reshape(f.shape[:-1] + (
            self.pou_shape[0], self.block_shape[0],
            self.pou_shape[1], self.block_shape[1])).sum((-1, -3)).reshape(
                f.shape[:-1] + (self.n_patch,))

    def combine_patches(self, f_patches):
        b0, b1 = self.block_shape
        k = self.kernel_width
        f_padded = np.zeros(
            f_patches.shape[:-2] + (self.mesh_shape[0] + k - 1,
                                    self.mesh_shape[1] + k - 1))
        for p in range(self.n_patch):
            i, j = p // self.pou_shape[0], p % self.pou_shape[0]
            f_padded[..., i*b0:(i+1)*b0+(k-1), j*b1:(j+1)*b1+(k-1)] += (
                f_patches[..., p, :].reshape(
                    f_patches.shape[:-2] + self.patch_shape))
        f_2d = f_padded[..., self.slice_mid, self.slice_mid] * 1.
        f_2d[..., :, self.slice_e_0] += (
            f_padded[..., self.slice_mid, self.slice_e_1])
        f_2d[..., :, self.slice_e_1] += (
            f_padded[..., self.slice_mid, self.slice_e_0])
        f_2d[..., self.slice_e_0, :] += (
            f_padded[..., self.slice_e_1, self.slice_mid])
        f_2d[..., self.slice_e_1, :] += (
            f_padded[..., self.slice_e_0, self.slice_mid])
        f_2d[..., self.slice_e_0, self.slice_e_0] += (
            f_padded[..., self.slice_e_1, self.slice_e_1])
        f_2d[..., self.slice_e_0, self.slice_e_1] += (
            f_padded[..., self.slice_e_1, self.slice_e_0])
        f_2d[..., self.slice_e_1, self.slice_e_0] += (
            f_padded[..., self.slice_e_0, self.slice_e_1])
        f_2d[..., self.slice_e_1, self.slice_e_1] += (
            f_padded[..., self.slice_e_0, self.slice_e_0])
        return f_2d.reshape(f_patches.shape[:-2] + (-1,))

    def patch_distance(self, patch_index, coords):
        lim_0, lim_1 = self.patch_lims[:, patch_index]
        m_00 = np.maximum(0, lim_0[0] - coords[:, 0])
        m_01 = np.maximum(0, coords[:, 0] - lim_0[1])
        m_10 = np.maximum(0, lim_1[0] - coords[:, 1])
        m_11 = np.maximum(0, coords[:, 1] - lim_1[1])
        d_0 = (np.maximum(m_00, m_01) if lim_0[0] < lim_0[1] else
               np.minimum(m_00, m_01))
        d_1 = (np.maximum(m_10, m_11) if lim_1[0] < lim_1[1] else
               np.minimum(m_10, m_11))
        return (d_0**2 + d_1**2)**0.5


class PerMeshNodePartitionOfUnityBasis(object):
    """PoU on spatial mesh with constant bump function at each mesh node."""

    def __init__(self, mesh_shape):
        if hasattr(mesh_shape, '__len__'):
            self.mesh_shape = mesh_shape
            self.n_node = mesh_shape[0] * mesh_shape[1]
            self.mesh_coords = np.stack((c.flat for c in np.meshgrid(
                (0.5 + np.arange(mesh_shape[0])) / mesh_shape[0],
                (0.5 + np.arange(mesh_shape[1])) / mesh_shape[1])), -1)
        else:
            self.mesh_shape = (mesh_shape,)
            self.n_node = mesh_shape
            self.mesh_coords = (0.5 + np.arange(self.n_node)) / self.n_node
        self.n_patch = self.n_node

    def integrate_against_bases(self, f):
        return f

    def split_into_patches_and_scale(self, f):
        return f.reshape(f.shape + (1,))

    def combine_patches(self, f_patches):
        return f_patches.reshape(f_patches.shape[:-1])

    def patch_distance(self, patch_index, coords):
        node_coord = self.mesh_coords[patch_index]
        if len(self.mesh_shape) == 1:
            return np.abs(coords - node_coord)
        else:
            return np.sum((coords - node_coord)**2, -1)**0.5
