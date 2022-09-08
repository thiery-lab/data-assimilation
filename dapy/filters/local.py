import abc

"""Localised ensemble filters for inference in spatially extended state-space models."""

from typing import Tuple, Dict, Callable, Any, Optional, Sequence
from functools import partial
import numpy as np
import numpy.linalg as nla
from numpy.typing import ArrayLike
from numpy.random import Generator
from scipy.special import logsumexp
from dapy.filters.base import AbstractEnsembleFilter
from dapy.models.base import AbstractDiagonalGaussianObservationModel
import dapy.ot as optimal_transport
from dapy.utils.localisation import gaspari_and_cohn_weighting
from dapy.utils.pou import AbstractPartitionOfUnity, PerMeshNodePartitionOfUnityBasis
from dapy.ot.costs import calculate_cost_matrices_1d, calculate_cost_matrices_2d


class AbstractLocalEnsembleFilter(AbstractEnsembleFilter):
    """Localised ensemble filter base class for spatially extended state-space models.

    Assumes model state and observations are defined over a fixed set of points in a
    spatial domain and that dependencies between state values at a point and
    observations are signficant only for observations in a localised region around the
    state location. It is further assumed here that the observations at a time point are
    conditionally independent given the state with a diagonal covariance Gaussian
    conditional distribution. Under these assumptions, when performing the assimilation
    update to the prior (predictive) state ensemble to take in to account the
    observations at a given time index, the ensemble state values at each spatial mesh
    node can each be updated independently based only a local subset of the
    observations.
    """

    def __init__(
        self,
        localisation_radius: float,
        localisation_weighting_func: Callable[
            [ArrayLike, float], ArrayLike
        ] = gaspari_and_cohn_weighting,
        inflation_factor: float = 1.0,
    ):
        """
        Args:
            localisation_radius: Positive value specifing maximum distance from a mesh
                node to observation point to assign a non-zero localisation weight to
                the observation point for that mesh node. Observation points within a
                distance of the localisation radius of the mesh node will be assigned
                localisation weights in the range `[0, 1]`.
            localisation_weighting_func: Function which given a one-dimensional array of
                distances and positive localisation radius computes a set of
                localisation weights in the range `[0, 1]` with distances greater than
                the localisation radius mapping to zero weights and distances between
                zero and the localisation radius mapping monotonically from weight one
                at distance zero to weight zero at distance equal to the localisation
                radius.
            inflation_factor: A value greater than or equal to one used to inflate the
                posterior ensemble deviations on each update as a heuristic to overcome
                the underestimation of the uncertainty in the system state by ensemble
                methods.
        """
        self.localisation_radius = localisation_radius
        self.localisation_weighting_func = localisation_weighting_func
        self.inflation_factor = inflation_factor

    def _perform_model_specific_initialization(
        self,
        model: AbstractDiagonalGaussianObservationModel,
        num_particle: int,
    ):
        self._observation_indices_and_weights_cache = [None] * model.mesh_size

    def _observation_indices_and_weights(
        self, node_index: int, model: AbstractDiagonalGaussianObservationModel
    ) -> Tuple[Sequence[int], ArrayLike]:
        if self._observation_indices_and_weights_cache[node_index] is not None:
            return self._observation_indices_and_weights_cache[node_index]
        observation_distances = model.distances_from_mesh_node_to_observation_points(
            node_index
        )
        localisation_weights = self.localisation_weighting_func(
            observation_distances, self.localisation_radius
        )
        non_zero_localisation_weights = localisation_weights > 0.0
        non_zero_indices = np.nonzero(non_zero_localisation_weights)[0]
        localisation_weights = localisation_weights[non_zero_localisation_weights]
        self._observation_indices_and_weights_cache[node_index] = (
            non_zero_indices,
            localisation_weights,
        )
        return non_zero_indices, localisation_weights

    def _assimilation_update(
        self,
        model: AbstractDiagonalGaussianObservationModel,
        rng: Generator,
        previous_states: Optional[ArrayLike],
        predicted_states: ArrayLike,
        observation: ArrayLike,
        time_index: int,
    ) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        num_particle = predicted_states.shape[0]
        predicted_states_mesh = predicted_states.reshape(
            (num_particle, -1, model.mesh_size)
        )
        observation_means = model.observation_mean(predicted_states, time_index)
        post_states_mesh = np.full(predicted_states_mesh.shape, np.nan)
        for node_index in range(model.mesh_size):
            local_indices, local_weights = self._observation_indices_and_weights(
                node_index, model
            )
            post_states_mesh[:, :, node_index] = self._local_assimilation_update(
                predicted_states_mesh[:, :, node_index],
                observation_means[:, local_indices],
                observation[local_indices],
                model.observation_noise_std[local_indices],
                local_weights,
            )
        post_state_particles = post_states_mesh.reshape((num_particle, -1))
        return (
            post_state_particles,
            {
                "state_mean": post_state_particles.mean(0),
                "state_std": post_state_particles.std(0),
            },
        )

    @abc.abstractmethod
    def _local_assimilation_update(
        self,
        node_state_particles: ArrayLike,
        local_observation_particles: ArrayLike,
        local_observation: ArrayLike,
        local_observation_noise_std: ArrayLike,
        local_observation_weights: ArrayLike,
    ) -> ArrayLike:
        """Perform a local analysis update for the state at a grid point.

        Args:
            node_state_particles: Two-dimensional array of shape
                `(num_particle, dim_per_node_state)` where `num_particle` is the number
                of particles in the ensemble and `dim_per_node_state` is the dimension
                of the local state at each spatial mesh node, with each row the local
                state values of an ensemble member at a particular mesh node.
            local_observation_particles: Two-dimensional array of shape
                `(num_particle, dim_observation_local)` where `num_particle` is the
                number of particles in the ensemble and `dim_observation_local` is the
                dimension of the vector of observations local to the current state
                spatial mesh node, with each row the predicted local observation means
                for a particle in the ensemble.
            local_observation: One-dimensional array of shape `(dim_observation_local)`
                where `dim_observation_local` is the dimension of the vector of
                observations local to the current state spatial mesh node, with entries
                corresponding to the local values of the observations at the current
                time point.
            local_observation_noise_std: One-dimensional array of shape
                `(dim_observation_local)` where `dim_observation_local` is the dimension
                of the vector of observations local to the current state spatial mesh
                node, with entries corresponding to the standard deviations of each
                local observed variable given the current state variable values.
            local_observation_weights: One-dimensional array of shape
                `(dim_observation_local)` where `dim_observation_local` is the dimension
                of the vector of observations local to the current state spatial mesh
                node, with entries corresponding to weights for each local observed
                variable in [0, 1] to modulate the strength of the effect of each local
                observation on the updated state values based on the distance between
                the state spatial mesh node and observation location.

        Returns:
            Two-dimensional array of shape `(num_particle, dim_per_node_state)` where
            `num_particle` is the number of particles in the ensemble and
            `dim_per_node_state` is the dimension of the local state at each spatial
            mesh node, with each row the local updated posterior state values of each
            particle in the ensemble.
        """


class LocalEnsembleTransformParticleFilter(AbstractLocalEnsembleFilter):
    """Localised ensemble transform particle filter for spatially extended models.

    References:

        1. Reich, S. (2013). A nonparametric ensemble transform method for
           Bayesian inference. SIAM Journal on Scientific Computing, 35(4),
           A2013-A2024.
    """

    def __init__(
        self,
        localisation_radius: float,
        localisation_weighting_func: Callable[
            [ArrayLike, float], ArrayLike
        ] = gaspari_and_cohn_weighting,
        inflation_factor: float = 1.0,
        optimal_transport_solver: Callable[
            [ArrayLike, ArrayLike, ArrayLike], ArrayLike
        ] = optimal_transport.solve_optimal_transport_exact,
        optimal_transport_solver_kwargs: Optional[Dict[str, Any]] = None,
        transport_cost: Callable[
            [ArrayLike, ArrayLike], ArrayLike
        ] = optimal_transport.pairwise_euclidean_distance,
        weight_threshold: float = 1e-8,
    ):
        """
        Args:
            localisation_radius: Positive value specifing maximum distance from a mesh
                node to observation point to assign a non-zero localisation weight to
                the observation point for that mesh node. Observation points within a
                distance of the localisation radius of the mesh node will be assigned
                localisation weights in the range `[0, 1]`.
            localisation_weighting_func: Function which given a one-dimensional array of
                distances and positive localisation radius computes a set of
                localisation weights in the range `[0, 1]` with distances greater than
                the localisation radius mapping to zero weights and distances between
                zero and the localisation radius mapping monotonically from weight one
                at distance zero to weight zero at distance equal to the localisation
                radius.
            inflation_factor: A value greater than or equal to one used to inflate the
                posterior ensemble deviations on each update as a heuristic to overcome
                the underestimation of the uncertainty in the system state by ensemble
                methods.
            optimal_transport_solver: Optimal transport solver function with signature

                    transport_matrix = optimal_transport_solver(
                        source_dist, target_dist, cost_matrix,
                        **optimal_transport_solver_kwargs)

                where `source_dist` and `target_dist` are the source and target
                distribution weights respectively as 1D arrays, `cost_matrix` is a 2D
                array of the transport costs for each particle pair.
            optimal_transport_solver_kwargs: Any additional keyword parameters values
                for the optimal transport solver.
            transport_cost: Function calculating transport cost matrix with signature

                    cost_matrix = transport_cost(source_particles, target_particles)

                where `source_particles` are the particles values of the source and
                target empirical distributions respecitively.
            weight_threshold: Threshold below which to set any particle weights to zero
                prior to solving the optimal transport problem. Using a small non-zero
                value can both improve the numerical stability of the optimal transport
                solves, with problems with many small weights sometimes failing to
                convergence, and also improve performance as some solvers (including)
                the default network simplex based algorithm) are able to exploit
                sparsity in the source / target distributions.
        """
        super().__init__(
            localisation_radius=localisation_radius,
            localisation_weighting_func=localisation_weighting_func,
            inflation_factor=inflation_factor,
        )
        self.optimal_transport_solver = optimal_transport_solver
        self.optimal_transport_solver_kwargs = (
            {}
            if optimal_transport_solver_kwargs is None
            else optimal_transport_solver_kwargs
        )
        self.transport_cost = transport_cost
        self.weight_threshold = weight_threshold

    def _local_assimilation_update(
        self,
        node_state_particles: ArrayLike,
        local_observation_particles: ArrayLike,
        local_observation: ArrayLike,
        local_observation_noise_std: ArrayLike,
        local_observation_weights: ArrayLike,
    ) -> ArrayLike:
        num_particle = node_state_particles.shape[0]
        local_observation_errors = local_observation_particles - local_observation
        node_log_particle_weights = -0.5 * (
            local_observation_errors
            * (local_observation_weights / local_observation_noise_std**2)
            * local_observation_errors
        ).sum(-1)
        node_source_dist = np.ones(num_particle) / num_particle
        node_target_dist = np.exp(
            node_log_particle_weights - logsumexp(node_log_particle_weights)
        )
        if self.weight_threshold > 0:
            node_target_dist[node_target_dist < self.weight_threshold] = 0
            node_target_dist /= node_target_dist.sum()
        node_cost_matrix = self.transport_cost(
            node_state_particles, node_state_particles
        )
        node_transform_matrix = num_particle * self.optimal_transport_solver(
            node_source_dist,
            node_target_dist,
            node_cost_matrix,
            **self.optimal_transport_solver_kwargs
        )
        node_post_state_particles = node_transform_matrix @ node_state_particles
        if self.inflation_factor > 1.0:
            node_post_state_mean = node_post_state_particles.mean(0)
            node_post_state_devs = node_post_state_particles - node_post_state_mean
            return node_post_state_mean + node_post_state_devs * self.inflation_factor
        else:
            return node_post_state_particles


class LocalEnsembleTransformKalmanFilter(AbstractLocalEnsembleFilter):
    """Localised ensemble transform Kalman filter for spatially extended models.

    References:
        1. Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007).
           Efficient data assimilation for spatiotemporal chaos:
           A local ensemble transform Kalman filter.
           Physica D: Nonlinear Phenomena, 230(1), 112-126.
    """

    def _local_assimilation_update(
        self,
        node_state_particles: ArrayLike,
        local_observation_particles: ArrayLike,
        local_observation: ArrayLike,
        local_observation_noise_std: ArrayLike,
        local_observation_weights: ArrayLike,
    ) -> ArrayLike:
        num_particle = node_state_particles.shape[0]
        dim_observation_local = local_observation.shape[0]
        # Compute local state ensemble mean vector and deviations matrix
        node_state_mean = node_state_particles.mean(0)
        node_state_deviations = node_state_particles - node_state_mean
        # Compute local observation ensemble mean vector and deviations matrix
        local_observation_mean = local_observation_particles.mean(0)
        local_observation_deviations = (
            local_observation_particles - local_observation_mean
        )
        local_observation_error = local_observation - local_observation_mean
        # Compute reciprocal of effective per observation variances
        # by scaling by the inverse variances by the localisation weights
        effective_inv_observation_variance = (
            local_observation_weights / local_observation_noise_std**2
        )
        transform_matrix_eigenvectors, non_zero_singular_values, _ = nla.svd(
            local_observation_deviations
            * effective_inv_observation_variance**0.5
            / (num_particle - 1) ** 0.5,
        )
        squared_transform_matrix_eigenvalues = 1 / (1 + non_zero_singular_values**2)
        if dim_observation_local < num_particle:
            squared_transform_matrix_eigenvalues = np.concatenate(
                [
                    squared_transform_matrix_eigenvalues,
                    np.ones(num_particle - dim_observation_local),
                ]
            )
        transform_matrix = (
            transform_matrix_eigenvectors * squared_transform_matrix_eigenvalues**0.5
        ) @ transform_matrix_eigenvectors.T
        kalman_gain_mult_observation_error = node_state_deviations.T @ (
            transform_matrix_eigenvectors
            @ (
                (
                    transform_matrix_eigenvectors.T
                    @ (
                        local_observation_deviations
                        @ (local_observation_error * effective_inv_observation_variance)
                    )
                )
                * squared_transform_matrix_eigenvalues
            )
            / (num_particle - 1)
        )
        node_post_state_mean = node_state_mean + kalman_gain_mult_observation_error
        node_post_state_deviations = transform_matrix @ node_state_deviations
        return node_post_state_mean + self.inflation_factor * node_post_state_deviations


class ScalableLocalEnsembleTransformParticleFilter(AbstractEnsembleFilter):
    """Scalable local ensemble transform particle filter.

    References:

      1. Graham, M.M. and Thiery, A. H. (2019). A scalable optimal-transport based local
         particle filter. arXiv preprint 1906.00507.

    """

    def __init__(
        self,
        localisation_radius: float,
        partition_of_unity: Optional[AbstractPartitionOfUnity] = None,
        calculate_cost_matrices_func: Optional[Callable[[ArrayLike], ArrayLike]] = None,
        localisation_weighting_func: Callable[
            [ArrayLike, float], ArrayLike
        ] = gaspari_and_cohn_weighting,
        optimal_transport_solver: Callable[
            [ArrayLike, ArrayLike, ArrayLike], ArrayLike
        ] = optimal_transport.solve_optimal_transport_exact_batch,
        optimal_transport_solver_kwargs: Optional[Dict[str, Any]] = None,
        calculate_cost_matrices_func_kwargs: Optional[Dict[str, Any]] = None,
        weight_threshold: float = 1e-8,
    ):
        """
        Args:
            localisation_radius: Positive value specifing maximum distance from a mesh
                node to observation point to assign a non-zero localisation weight to
                the observation point for that mesh node. Observation points within a
                distance of the localisation radius of the mesh node will be assigned
                localisation weights in the range `[0, 1]`.
            partition_of_unity: Object defining partition of unity on spatial domain.
            calculate_cost_matrices_func: Function returning the per-patch optimal
                transport cost matrices as a 3D array of shape
                `(num_patch, num_particle, num_particle)` give a 2D array of meshed
                state particles of shape `(num_particle, dim_node_state, mesh_size)`
                where `dim_node_state` is the dimension of the per spatial mesh node
                state and `mesh_size` is the number of nodes in the spatial mesh.
            localisation_weighting_func: Function which given a one-dimensional array of
                distances and positive localisation radius computes a set of
                localisation weights in the range `[0, 1]` with distances greater than
                the localisation radius mapping to zero weights and distances between
                zero and the localisation radius mapping monotonically from weight one
                at distance zero to weight zero at distance equal to the localisation
                radius.
            optimal_transport_solver: Optimal transport solver function with signature

                    transport_matrix = optimal_transport_solver(
                        per_patch_source_dists, per_patch_target_dists,
                        per_patch_cost_matrices, **optimal_transport_solver_kwargs)

                where `per_patch_source_dists` and `per_patch_target_dists` are the
                per-patch source and target distribution weights respectively as 2D
                arrays of shape `(num_patch, num_particle)`, `per_patch_cost_matrices`
                is a 3D array of shape `(num_patch, num_particle, num_particle)` the
                per-patch transport costs for each particle pair.
            optimal_transport_solver_kwargs: Any additional keyword argument values
                for the optimal transport solver.
            calculate_cost_matrices_func_kwargs: Any additional keyword argument values
                for the transport cost matrix function.
            weight_threshold: Threshold below which to set any particle weights to zero
                prior to solving the optimal transport problem. Using a small non-zero
                value can both improve the numerical stability of the optimal transport
                solves, with problems with many small weights sometimes failing to
                convergence, and also improve performance as some solvers (including)
                the default network simplex based algorithm) are able to exploit
                sparsity in the source / target distributions.
        """
        self.localisation_radius = localisation_radius
        self.localisation_weighting_func = localisation_weighting_func
        self.partition_of_unity = partition_of_unity
        self.optimal_transport_solver = optimal_transport_solver
        self.optimal_transport_solver_kwargs = (
            {}
            if optimal_transport_solver_kwargs is None
            else optimal_transport_solver_kwargs
        )
        self.weight_threshold = weight_threshold
        self.calculate_cost_matrices_func = calculate_cost_matrices_func
        self.calculate_cost_matrices_func_kwargs = (
            {}
            if calculate_cost_matrices_func_kwargs is None
            else calculate_cost_matrices_func_kwargs
        )

    def _perform_model_specific_initialization(
        self,
        model: AbstractDiagonalGaussianObservationModel,
        num_particle: int,
    ):
        if self.partition_of_unity is None:
            self.partition_of_unity = PerMeshNodePartitionOfUnityBasis(model)
        if self.calculate_cost_matrices_func is None:
            if model.spatial_dimension == 1:
                self.calculate_cost_matrices_func = partial(
                    calculate_cost_matrices_1d,
                    num_patch=self.partition_of_unity.num_patch,
                    half_overlap=self.partition_of_unity.patch_half_overlap[0],
                )
            elif model.spatial_dimension == 2:
                self.calculate_cost_matrices_func = partial(
                    calculate_cost_matrices_2d,
                    mesh_shape_0=model.mesh_shape[0],
                    mesh_shape_1=model.mesh_shape[1],
                    pou_shape_0=self.partition_of_unity.shape[0],
                    pou_shape_1=self.partition_of_unity.shape[1],
                    half_overlap_0=self.partition_of_unity.patch_half_overlap[0],
                    half_overlap_1=self.partition_of_unity.patch_half_overlap[1],
                )
            else:
                raise NotImplementedError()
        self._per_patch_localisation_weights = np.stack(
            [
                self.localisation_weighting_func(
                    self.partition_of_unity.patch_distance(p, model.observation_coords),
                    self.localisation_radius,
                )
                for p in range(self.partition_of_unity.num_patch)
            ],
            axis=0,
        )

    def _assimilation_update(
        self,
        model: AbstractDiagonalGaussianObservationModel,
        rng: Generator,
        previous_states: Optional[ArrayLike],
        predicted_states: ArrayLike,
        observation: ArrayLike,
        time_index: int,
    ) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        num_particle = predicted_states.shape[0]
        observation_log_densities = (
            -0.5
            * (model.observation_mean(predicted_states, time_index) - observation) ** 2
            / (model.observation_noise_std**2)
        )
        per_patch_log_target_dists = (
            self._per_patch_localisation_weights @ observation_log_densities.T
        )
        per_patch_target_dists = np.exp(
            per_patch_log_target_dists
            - logsumexp(per_patch_log_target_dists, axis=-1)[:, None]
        )
        per_patch_source_dists = np.ones_like(per_patch_target_dists) / num_particle
        predicted_states_mesh = predicted_states.reshape(
            (num_particle, -1, model.mesh_size)
        )
        per_patch_cost_matrices = self.calculate_cost_matrices_func(
            predicted_states_mesh, **self.calculate_cost_matrices_func_kwargs
        )
        if self.weight_threshold > 0:
            per_patch_target_dists[per_patch_target_dists < self.weight_threshold] = 0
            per_patch_target_dists /= per_patch_target_dists.sum(-1)[:, None]
        per_patch_transform_matrices = (
            self.optimal_transport_solver(
                per_patch_source_dists,
                per_patch_target_dists,
                per_patch_cost_matrices,
                **self.optimal_transport_solver_kwargs
            )
            * num_particle
        )
        post_states_patches = np.einsum(
            "kij,jlkm->ilkm",
            per_patch_transform_matrices,
            self.partition_of_unity.split_into_patches_and_scale(predicted_states_mesh),
        )
        post_states = self.partition_of_unity.combine_patches(
            post_states_patches
        ).reshape((num_particle, model.dim_state))
        return (
            post_states,
            {
                "state_mean": post_states.mean(0),
                "state_std": post_states.std(0),
                "per_patch_estimated_ess": 1 / (per_patch_target_dists**2).sum(-1),
            },
        )
