"""Partition of unity bases functions for scalable localisation."""

import abc
from typing import Tuple, Callable
import numpy as np
import numpy.fft as fft
import numba as nb
from dapy.models.base import AbstractModel
import dapy.utils.localisation as localisation


@nb.njit(
    nb.float64[:, :, :](nb.float64[:, :, :, :], nb.int64, nb.int64, nb.int64, nb.int64)
)
def _sum_overlapping_patches_1d(
    f_patches, block_width, kernel_width, num_patch, num_node
):
    w = block_width
    k = kernel_width
    f_padded = np.zeros(f_patches.shape[:-2] + (num_node + k - 1,))
    for b in range(num_patch):
        f_padded[..., b * w : (b + 1) * w + (k - 1)] += f_patches[..., b, :]
    f_padded[..., (k - 1) // 2 : (k - 1)] += f_padded[..., -(k - 1) // 2 :]
    f_padded[..., -(k - 1) : -(k - 1) // 2] += f_padded[..., : (k - 1) // 2]
    return f_padded[..., (k - 1) // 2 : -(k - 1) // 2]


def _separable_fft_convolve_2d(x, fft_k_0, fft_k_1):
    x_ = fft.irfft(fft_k_0[:, None] * fft.rfft(x, axis=-2), axis=-2)
    return fft.irfft(fft_k_1 * fft.rfft(x_, axis=-1), axis=-1)


class AbstractPartitionOfUnity(abc.ABC):
    """Abstract base class for partitions of unity on spatial domains."""

    @property
    @abc.abstractmethod
    def num_patch(self) -> int:
        """Number of patches patial domain is partitioned in to."""

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple specifying number of patches along each spatial axis."""

    @property
    def patch_half_overlap(self) -> Tuple[int, ...]:
        """Tuple specifying half node overlap between patches along each spatial axis"""

    @abc.abstractmethod
    def split_into_patches_and_scale(self, field: np.ndarray) -> np.ndarray:
        """Split a (batch) of spatial fields into patches and scale by bump function."""

    @abc.abstractmethod
    def combine_patches(self, f_patches: np.ndarray) -> np.ndarray:
        """Combine (bump function) scaled patches of a (batch) of spatial fields."""

    @abc.abstractmethod
    def patch_distance(self, patch_index: int, coords: np.ndarray) -> np.ndarray:
        """Compute the distance from a patch to points in the spatial domain."""


class PerMeshNodePartitionOfUnityBasis(AbstractPartitionOfUnity):
    """Partition of unity on spatial mesh with unit bump function at each mesh node."""

    def __init__(self, model: AbstractModel):
        self.model = model

    @property
    def num_patch(self) -> int:
        return self.model.mesh_size

    @property
    def patch_half_overlap(self) -> Tuple[int, ...]:
        return (0,) * self.model.spatial_dimension

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.model.mesh_shape

    def split_into_patches_and_scale(self, f):
        return f.reshape(f.shape + (1,))

    def combine_patches(self, f_patches):
        return f_patches.reshape(f_patches.shape[:-1])

    def patch_distance(self, patch_index, coords):
        return self.model.distances_from_mesh_node_to_points(patch_index, coords)


class SmoothedBlock1dPartitionOfUnityBasis(AbstractPartitionOfUnity):
    """Partition of unity on 1D spatial mesh formed by convolving with smooth kernel."""

    def __init__(
        self,
        model: AbstractModel,
        num_patch: int,
        kernel_halfwidth: int,
        offset: int = 0,
        kernel_weighting_function: Callable[
            [np.ndarray, float], np.ndarray
        ] = localisation.gaspari_and_cohn_weighting,
    ):
        self.model = model
        self._num_patch = num_patch
        self.block_width = model.mesh_size // num_patch
        self.offset = offset
        kernel = kernel_weighting_function(
            abs(np.arange(-kernel_halfwidth + 1, kernel_halfwidth, dtype=np.float64)),
            kernel_halfwidth
        )
        kernel = kernel / kernel.sum()
        self.kernel_width = kernel.shape[0]
        self.kernel = np.zeros(model.mesh_size)
        if self.kernel_width > 1:
            self.kernel[-(self.kernel_width - 1) // 2 :] = kernel[
                : self.kernel_width // 2
            ]
        self.kernel[: (self.kernel_width + 1) // 2] = kernel[self.kernel_width // 2 :]
        self.rfft_kernel = fft.rfft(self.kernel)
        block = np.zeros(model.mesh_size)
        block[
            (model.mesh_size - self.block_width)
            // 2 : (model.mesh_size + self.block_width)
            // 2
        ] = 1
        self.patch_width = self.block_width + self.kernel_width - 1
        self.smoothed_bump = fft.irfft(self.rfft_kernel * fft.rfft(block))[
            (model.mesh_size - self.patch_width)
            // 2 : (model.mesh_size + self.patch_width)
            // 2
        ]
        self.patch_lims = np.full((self.num_patch, 2), np.nan)
        half_width = (self.kernel_width - 1) // 2
        patch_lindex = offset + np.arange(num_patch) * self.block_width - half_width
        self.patch_lims = (
            np.stack([patch_lindex, patch_lindex + self.patch_width], -1)
            / self.model.mesh_size
            % 1.0
        ) * model.domain_extents[0]

    @property
    def num_patch(self):
        return self._num_patch

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self._num_patch,)

    @property
    def patch_half_overlap(self) -> Tuple[int, ...]:
        return ((self.kernel_width - 1) // 2,)

    def split_into_patches(self, f):
        if self.offset != 0:
            f = np.roll(f, -self.offset, axis=-1)
        if self.kernel_width > 1:
            f_padded = np.zeros(
                f.shape[:-1] + (self.model.mesh_size + self.kernel_width - 1,)
            )
            f_padded[..., : (self.kernel_width - 1) // 2] = f[
                ..., -(self.kernel_width - 1) // 2 :
            ]
            f_padded[..., -(self.kernel_width - 1) // 2 :] = f[
                ..., : (self.kernel_width - 1) // 2
            ]
            f_padded[
                ..., (self.kernel_width - 1) // 2 : -(self.kernel_width - 1) // 2
            ] = f
        else:
            f_padded = f
        shape = f.shape[:-1] + (self.num_patch, self.patch_width)
        strides = f_padded.strides[:-1] + (
            self.block_width * f_padded.strides[-1],
            f_padded.strides[-1],
        )
        return np.lib.stride_tricks.as_strided(
            f_padded, shape, strides, writeable=False
        )

    def split_into_patches_and_scale(self, f):
        f_patches = self.split_into_patches(f)
        return f_patches * self.smoothed_bump

    def combine_patches(self, f_patches):
        w = self.block_width
        k = self.kernel_width
        if k > 1:
            f_1d = _sum_overlapping_patches_1d(
                f_patches, w, k, self.num_patch, self.model.mesh_size
            )
        else:
            f_1d = f_patches.reshape(f_patches.shape[:-2] + self.model.mesh_shape)
        if self.offset != 0:
            f_1d = np.roll(f_1d, self.offset, axis=-1)
        return f_1d

    def patch_distance(self, patch_index, coords):
        lim = self.patch_lims[patch_index]
        coords = coords[:, 0]
        edge_dist = np.minimum(
            (lim[0] - coords) % self.model.domain_extents[0],
            (coords - lim[1]) % self.model.domain_extents[0],
        )
        if lim[1] > lim[0]:
            not_in_patch = (coords < lim[0]) | (coords > lim[1])
            return not_in_patch * edge_dist
        else:
            not_in_patch = (coords < lim[0]) & (coords > lim[1])
            return not_in_patch * edge_dist


class SmoothedBlock2dPartitionOfUnityBasis(AbstractPartitionOfUnity):
    """PoU on 2D grid using block PoU convolved with smooth kernel."""

    def __init__(
        self,
        model: AbstractModel,
        shape: Tuple[int, int],
        kernel_halfwidth: int,
        kernel_weighting_function: Callable[
            [np.ndarray, float], np.ndarray
        ] = localisation.gaspari_and_cohn_weighting,
    ):
        self.model = model
        self._shape = shape
        self.block_shape = (
            model.mesh_shape[0] // shape[0],
            model.mesh_shape[1] // shape[1],
        )
        kernel = kernel_weighting_function(
            np.abs(np.arange(-kernel_halfwidth + 1, kernel_halfwidth, dtype=np.float64))
        )
        self.kernel_width = kernel.shape[0]
        kernel = kernel / kernel.sum()
        self.kernel_0 = np.zeros(model.mesh_shape[0])
        self.kernel_0[-(self.kernel_width - 1) // 2 :] = kernel[
            : self.kernel_width // 2
        ]
        self.kernel_0[: (self.kernel_width + 1) // 2] = kernel[self.kernel_width // 2 :]
        self.rfft_kernel_0 = fft.rfft(self.kernel_0)
        self.kernel_1 = np.zeros(model.mesh_shape[1])
        self.kernel_1[-(self.kernel_width - 1) // 2 :] = kernel[
            : self.kernel_width // 2
        ]
        self.kernel_1[: (self.kernel_width + 1) // 2] = kernel[self.kernel_width // 2 :]
        self.rfft_kernel_1 = fft.rfft(self.kernel_1)
        block = np.zeros(model.mesh_shape)
        block[
            (model.mesh_shape[0] - self.block_shape[0])
            // 2 : (model.mesh_shape[0] + self.block_shape[0])
            // 2,
            (model.mesh_shape[1] - self.block_shape[1])
            // 2 : (model.mesh_shape[1] + self.block_shape[1])
            // 2,
        ] = 1
        self.patch_shape = (
            self.block_shape[0] + self.kernel_width - 1,
            self.block_shape[1] + self.kernel_width - 1,
        )
        self.smoothed_bump = _separable_fft_convolve_2d(
            block, self.rfft_kernel_0, self.rfft_kernel_1
        )[
            (model.mesh_shape[0] - self.patch_shape[0])
            // 2 : (model.mesh_shape[0] + self.patch_shape[0])
            // 2,
            (model.mesh_shape[1] - self.patch_shape[1])
            // 2 : (model.mesh_shape[1] + self.patch_shape[1])
            // 2,
        ]
        self.slice_mid = slice(
            (self.kernel_width - 1) // 2, -(self.kernel_width - 1) // 2
        )
        self.slice_e_0 = slice(None, (self.kernel_width - 1) // 2)
        self.slice_e_1 = slice(-(self.kernel_width - 1) // 2, None)
        self.patch_lims = np.full((2, self.num_patch, 2), np.nan)
        half_width = (self.kernel_width - 1) // 2
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.patch_lims[0, i * shape[0] + j] = (
                    ((i * self.block_shape[0] - half_width) / self.mesh_shape[0]) % 1.0,
                    (((i + 1) * self.block_shape[0] + half_width) / self.mesh_shape[0])
                    % 1.0,
                ) * model.domain_extents[0]
                self.patch_lims[1, i * shape[0] + j] = (
                    ((j * self.block_shape[1] - half_width) / self.mesh_shape[1]) % 1.0,
                    (((j + 1) * self.block_shape[1] + half_width) / self.mesh_shape[1])
                    % 1.0,
                ) * model.domain_extents[1]

    @property
    def num_patch(self):
        return self._num_patch

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def patch_half_overlap(self) -> Tuple[int, ...]:
        return ((self.kernel_width - 1) // 2,) * 2

    def split_into_patches_and_scale(self, f):
        f_2d = np.reshape(f, f.shape[:-1] + self.model.mesh_shape)
        f_padded = np.zeros(
            f_2d.shape[:-2]
            + (
                self.model.mesh_shape[0] + self.kernel_width - 1,
                self.model.mesh_shape[1] + self.kernel_width - 1,
            )
        )
        f_padded[..., self.slice_mid, self.slice_e_0] = f_2d[..., :, self.slice_e_1]
        f_padded[..., self.slice_mid, self.slice_e_1] = f_2d[..., :, self.slice_e_0]
        f_padded[..., self.slice_e_0, self.slice_mid] = f_2d[..., self.slice_e_1, :]
        f_padded[..., self.slice_e_1, self.slice_mid] = f_2d[..., self.slice_e_0, :]
        f_padded[..., self.slice_e_0, self.slice_e_0] = f_2d[
            ..., self.slice_e_1, self.slice_e_1
        ]
        f_padded[..., self.slice_e_0, self.slice_e_1] = f_2d[
            ..., self.slice_e_1, self.slice_e_0
        ]
        f_padded[..., self.slice_e_1, self.slice_e_0] = f_2d[
            ..., self.slice_e_0, self.slice_e_1
        ]
        f_padded[..., self.slice_e_1, self.slice_e_1] = f_2d[
            ..., self.slice_e_0, self.slice_e_0
        ]
        f_padded[..., self.slice_mid, self.slice_mid] = f_2d
        shape = f_padded.shape[:-2] + self.shape + self.patch_shape
        strides = (
            f_padded.strides[:-2]
            + (
                self.block_shape[0] * f_padded.strides[-2],
                self.block_shape[1] * f_padded.strides[-1],
            )
            + f_padded.strides[-2:]
        )
        f_patches = np.lib.stride_tricks.as_strided(f_padded, shape, strides)
        return (f_patches * self.smoothed_bump).reshape(
            f.shape[:-1] + (self.num_patch, -1)
        )

    def integrate_against_bases(self, f):
        f_2d = np.reshape(f, (-1,) + self.model.mesh_shape)
        f_2d_smooth = _separable_fft_convolve_2d(
            f_2d, self.rfft_kernel_0, self.rfft_kernel_1
        )
        return (
            f_2d_smooth.reshape(
                f.shape[:-1]
                + (
                    self.shape[0],
                    self.block_shape[0],
                    self.shape[1],
                    self.block_shape[1],
                )
            )
            .sum((-1, -3))
            .reshape(f.shape[:-1] + (self.num_patch,))
        )

    def combine_patches(self, f_patches):
        b0, b1 = self.block_shape
        k = self.kernel_width
        f_padded = np.zeros(
            f_patches.shape[:-2]
            + (self.model.mesh_shape[0] + k - 1, self.model.mesh_shape[1] + k - 1)
        )
        for p in range(self.num_patch):
            i, j = p // self.shape[0], p % self.shape[0]
            f_padded[
                ..., i * b0 : (i + 1) * b0 + (k - 1), j * b1 : (j + 1) * b1 + (k - 1)
            ] += f_patches[..., p, :].reshape(f_patches.shape[:-2] + self.patch_shape)
        f_2d = f_padded[..., self.slice_mid, self.slice_mid] * 1.0
        f_2d[..., :, self.slice_e_0] += f_padded[..., self.slice_mid, self.slice_e_1]
        f_2d[..., :, self.slice_e_1] += f_padded[..., self.slice_mid, self.slice_e_0]
        f_2d[..., self.slice_e_0, :] += f_padded[..., self.slice_e_1, self.slice_mid]
        f_2d[..., self.slice_e_1, :] += f_padded[..., self.slice_e_0, self.slice_mid]
        f_2d[..., self.slice_e_0, self.slice_e_0] += f_padded[
            ..., self.slice_e_1, self.slice_e_1
        ]
        f_2d[..., self.slice_e_0, self.slice_e_1] += f_padded[
            ..., self.slice_e_1, self.slice_e_0
        ]
        f_2d[..., self.slice_e_1, self.slice_e_0] += f_padded[
            ..., self.slice_e_0, self.slice_e_1
        ]
        f_2d[..., self.slice_e_1, self.slice_e_1] += f_padded[
            ..., self.slice_e_0, self.slice_e_0
        ]
        return f_2d.reshape(f_patches.shape[:-2] + (-1,))

    def patch_distance(self, patch_index, coords):
        lim_0, lim_1 = self.patch_lims[:, patch_index]
        m_00 = np.maximum(0, lim_0[0] - coords[:, 0])
        m_01 = np.maximum(0, coords[:, 0] - lim_0[1])
        m_10 = np.maximum(0, lim_1[0] - coords[:, 1])
        m_11 = np.maximum(0, coords[:, 1] - lim_1[1])
        d_0 = np.maximum(m_00, m_01) if lim_0[0] < lim_0[1] else np.minimum(m_00, m_01)
        d_1 = np.maximum(m_10, m_11) if lim_1[0] < lim_1[1] else np.minimum(m_10, m_11)
        return (d_0 ** 2 + d_1 ** 2) ** 0.5

