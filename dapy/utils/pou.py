"""Partition of unity bases functions for scalable localisation."""

import numpy as np


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
class SquaredCosine1dPartitionOfUnityBasis(object):

    def __init__(self, n_grid, n_bases):
        self.n_grid = n_grid
        self.n_bases = n_bases
        self.patch_half_dim = n_grid // n_bases
        u = (((np.arange(n_grid) + 0.5) / n_grid)[None, :] *
             np.pi * self.n_bases / 2.)
        self.s = np.sin(u)**2
        self.c = np.cos(u)**2
        self.norm = self.patch_half_dim

    def integrate_against_bases(self, f):
        f_1d = np.reshape(f, (-1, self.n_grid))
        f_1d_s = f_1d * self.s
        f_1d_c = np.roll(f_1d * self.c, shift=self.patch_half_dim, axis=1)
        f_1d_scaled_stack = np.stack([f_1d_s, f_1d_c], axis=1)
        return f_1d_scaled_stack.reshape(
            (-1, 2, self.n_bases // 2,  2 * self.patch_half_dim)
        ).sum(-1).reshape(f.shape[:-1] + (self.n_bases,)) / self.norm

    def split_into_patches_and_scale(self, f):
        n_particle, dim_field, n_grid = f.shape
        f_1d = f.reshape((-1, self.n_grid))
        f_1d_s = f_1d * self.s
        f_1d_c = np.roll(f_1d * self.c, shift=self.patch_half_dim, axis=1)
        f_1d_scaled_stack = np.stack([f_1d_s, f_1d_c], axis=1)
        return f_1d_scaled_stack.reshape(
            (n_particle, dim_field, self.n_bases, 2 * self.patch_half_dim))

    def combine_patches(self, f_patches):
        n_particle, dim_field = f_patches.shape[:2]
        f_1d_stacked = f_patches.reshape(
            (n_particle * dim_field, 2, self.n_grid))
        f_1d_s = f_1d_stacked[:, 0, ]
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
        self.norm = self.patch_half_dim[0] * self.patch_half_dim[1]

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
        ).sum((3, 5)).reshape(f.shape[:-1] + (self.n_bases,)) / self.norm

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
