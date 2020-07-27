"""Mix-in for spatially extended state-space models."""

from typing import Tuple, Sequence, Optional, Union
import numpy as np


class SpatiallyExtendedModelMixIn:
    """Mix-in class for spatially extended state-space models."""

    def __init__(
        self,
        mesh_shape: Tuple[int, ...],
        domain_extents: Tuple[float, ...],
        domain_is_periodic: bool,
        observation_coords: Optional[np.ndarray] = None,
        observation_node_indices: Optional[Union[slice, Sequence[int]]] = None,
        **kwargs,
    ):
        """
        Args:
            mesh_shape: Tuple of integers specifying dimensions (number of nodes along
                each axis) of rectilinear mesh used to discretize spatial domain. For
                example `mesh_shape=(64,)` would represent a 1D spatial domain with
                64 (equispaced) mesh nodes along the extents of the domain while
                `mesh_shape=(32, 64)` would represent a 2D spatial domain with 32
                equispaced mesh nodes along the first spatial axis and 64 equispaced
                mesh noes along the second spatial axis with there being in total
                `2048 = 32 * 64` mesh nodes in this case.
            domain_extents: Tuple of (positive) floats specifying spatial extent (size)
                of domain along each spatial axis, for example `domain_size=(1, 1)`
                would specify a 2D spatial domain of unit length along both axes.
            domain_is_periodic: Whether the spatial domain should be assumed to have
                periodic boundary conditions or equivalently to be a D-torus where D is
                the spatial dimension.
            observation_coords: Two-dimensional array of shape
                `(dim_observation, spatial_dimension)` specifying coordinates of
                observation points in order corresponding to values in observation
                vectors. Either this or `observation_indices` should be specified but
                not both.
            observation_node_indices: Sequence of integers or slice specifying indices
                of mesh nodes corresponding to observation points. Either this or
                `observation_coords` should be specified but not both.
        """
        self._mesh_shape = mesh_shape
        self._mesh_size = np.product(mesh_shape)
        self._domain_extents = domain_extents
        self._domain_is_periodic = domain_is_periodic
        self._mesh_node_coords = np.stack(
            np.meshgrid(
                *(
                    np.linspace(
                        0, domain_extents[d], mesh_shape[d], not domain_is_periodic
                    )
                    for d in range(self.spatial_dimension)
                )
            ),
            axis=-1,
        ).reshape((self.mesh_size, self.spatial_dimension))
        if observation_coords is None and observation_node_indices is None:
            raise ValueError(
                "One of observation_coords or observation_node_indices must be "
                "specified"
            )
        elif observation_coords is not None and observation_node_indices is not None:
            raise ValueError(
                "Only one of observation_coords or observation_node_indices must be "
                "specified"
            )
        elif observation_node_indices is not None:
            self._observation_coords = self._mesh_node_coords[observation_node_indices]
        else:
            self._observation_coords = observation_coords
        super().__init__(**kwargs)

    @property
    def mesh_shape(self) -> Tuple[int, ...]:
        """Number of nodes along each axis of spatial mesh."""
        return self._mesh_shape

    @property
    def mesh_size(self) -> int:
        """Total number of nodes in spatial mesh."""
        return self._mesh_size

    @property
    def spatial_dimension(self) -> int:
        """Number of dimensions of spatial domain."""
        return len(self._mesh_shape)

    @property
    def domain_extents(self) -> Tuple[float, ...]:
        """Spatial extents of domain along each spatial axis."""
        return self._domain_extents

    @property
    def domain_is_periodic(self) -> bool:
        """Whether domain has periodic boundary conditions or not."""
        return self._domain_is_periodic

    @property
    def mesh_node_coords(self) -> np.ndarray:
        """Two-dimensional array containing coordinates of spatial mesh nodes.

        Of shape `(mesh_size, spatial_dimension)` with each row representing the
        coordinates for one mesh node, with the row ordering following the ordering of
        the mesh nodes in the state vectors.
        """
        return self._mesh_node_coords

    @property
    def observation_coords(self) -> np.ndarray:
        """Two-dimensional array containing coordinates of observation points.

        Of shape `(mesh_size, spatial_dimension)` with each row representing the
        coordinates for one observation point, with the row ordering following the
        ordering of the observation points in the observation vectors.
        """
        return self._observation_coords

    def distances_from_mesh_node_to_observation_points(
        self, mesh_node_index: int
    ) -> np.ndarray:
        """Compute distance between mesh node and observation points.

        Args:
            mesh_node_index: Integer index of mesh node in order represented in state
                vector.

        Returns:
            One-dimensional array of spatial distances from specified mesh node to each
            of observation points, in order in which observation points are represented
            in the observation vectors.
        """
        return self.distances_from_mesh_node_to_points(
            mesh_node_index, self.observation_coords)

    def distances_from_mesh_node_to_points(
        self, mesh_node_index: int, coords: np.ndarray
    ) -> np.ndarray:
        """Compute distance between mesh node and points in spatial domain.

        Args:
            mesh_node_index: Integer index of mesh node in order represented in state
                vector.
            coords: Two-dimensional array of spatial coordinates of points to compute
                distances for.

        Returns:
            One-dimensional array of spatial distances from specified mesh node to each
            of points with coordinates specified in `coords`.
        """
        mesh_node_coord = self.mesh_node_coords[mesh_node_index]
        if self.domain_is_periodic:
            deltas = np.abs(mesh_node_coord - coords)
            return (np.minimum(deltas, self.domain_extents - deltas) ** 2).sum(
                -1
            ) ** 0.5
        else:
            return ((mesh_node_coord - coords) ** 2).sum(-1) ** 0.5
