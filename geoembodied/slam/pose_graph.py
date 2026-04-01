# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Differentiable Pose Graph for SLAM.

A pose graph represents:
    - Nodes: SE(3) poses (robot positions over time)
    - Edges: Relative pose measurements (odometry, loop closures)

Optimization minimizes the total factor graph cost:
    minimize  Σ_k  ‖e_k(x)‖²_{Σ_k}

where e_k are factor residuals and x are SE(3) poses.

This module provides a high-level API that wraps factors and the
Gauss-Newton optimizer into a clean, SLAM-friendly interface.

Key features:
    - Pure PyTorch: fully differentiable for end-to-end learning
    - Manifold-aware: uses Exp/Log maps, never violates SE(3) constraint
    - Robust: supports Huber/Cauchy kernels for outlier loop closures
    - Composable: mix BetweenFactor, PriorFactor, LandmarkFactor freely
"""

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from geoembodied.lietensor.se3 import SE3
from geoembodied.optim.factor import Factor, BetweenFactor, PriorFactor
from geoembodied.optim.gauss_newton import ManifoldGaussNewton
from geoembodied.optim.robust_kernel import RobustKernel


class PoseGraph(nn.Module):
    """Differentiable pose graph for SLAM and multi-view geometry.

    High-level API for constructing and optimizing pose graphs.

    Example::

        >>> graph = PoseGraph()
        >>> # Add nodes with initial pose estimates
        >>> graph.add_node(0, SE3.identity())
        >>> graph.add_node(1, SE3.exp(torch.tensor([0,0,0.1, 1,0,0])))
        >>> graph.add_node(2, SE3.exp(torch.tensor([0,0,0.2, 2,0,0])))
        >>>
        >>> # Add odometry factors (relative pose measurements)
        >>> graph.add_between(0, 1, odom_01, info_matrix=torch.eye(6)*100)
        >>> graph.add_between(1, 2, odom_12, info_matrix=torch.eye(6)*100)
        >>>
        >>> # Add loop closure (node 2 sees node 0 again)
        >>> graph.add_between(2, 0, loop_20, info_matrix=torch.eye(6)*50)
        >>>
        >>> # Fix first pose (gauge freedom)
        >>> graph.fix_node(0)
        >>>
        >>> # Optimize
        >>> optimized = graph.optimize(max_iterations=20)
        >>> print(f"Final cost: {graph.total_cost():.6f}")

    Args:
        default_information: Default information matrix for factors
    """

    def __init__(
        self,
        default_information: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self._nodes: Dict[int, Tensor] = {}
        self._factors: List[Factor] = nn.ModuleList()
        self._fixed_nodes: Set[int] = set()
        self._default_info = default_information

    @property
    def num_nodes(self) -> int:
        """Number of pose nodes."""
        return len(self._nodes)

    @property
    def num_factors(self) -> int:
        """Number of factors (constraints)."""
        return len(self._factors)

    @property
    def node_ids(self) -> List[int]:
        """Sorted list of node IDs."""
        return sorted(self._nodes.keys())

    def add_node(
        self,
        node_id: int,
        initial_pose: Tensor,
    ) -> None:
        """Add a pose node to the graph.

        Args:
            node_id: Unique integer ID for this node
            initial_pose: Initial SE(3) estimate
                shape: [7] (compact SE(3): qw,qx,qy,qz,tx,ty,tz)
                Can be SE3 LieTensor or plain Tensor.
        """
        if isinstance(initial_pose, SE3):
            pose = initial_pose.as_subclass(Tensor).detach().clone()
        else:
            pose = initial_pose.detach().clone()
        self._nodes[node_id] = pose

    def add_between(
        self,
        node_i: int,
        node_j: int,
        measurement: Tensor,
        information: Optional[Tensor] = None,
    ) -> None:
        """Add a relative pose constraint between two nodes.

        Args:
            node_i: Source node ID
            node_j: Target node ID
            measurement: Measured relative transform T_ij
                shape: [7] (compact SE(3))
            information: Information matrix Σ^{-1}
                shape: [6, 6]. Uses default if None.
        """
        if isinstance(measurement, SE3):
            measurement = measurement.as_subclass(Tensor)

        info = information if information is not None else self._default_info
        factor = BetweenFactor(node_i, node_j, measurement, info)
        self._factors.append(factor)

    def add_prior(
        self,
        node_id: int,
        prior_pose: Tensor,
        information: Optional[Tensor] = None,
    ) -> None:
        """Add a prior constraint on a single node.

        Args:
            node_id: Node ID
            prior_pose: Prior pose value
                shape: [7] (compact SE(3))
            information: Information matrix Σ^{-1}
                shape: [6, 6]. Uses default if None.
        """
        if isinstance(prior_pose, SE3):
            prior_pose = prior_pose.as_subclass(Tensor)

        info = information if information is not None else self._default_info
        factor = PriorFactor(node_id, prior_pose, info)
        self._factors.append(factor)

    def add_factor(self, factor: Factor) -> None:
        """Add an arbitrary factor to the graph.

        Args:
            factor: Any Factor subclass
        """
        self._factors.append(factor)

    def fix_node(self, node_id: int) -> None:
        """Fix a node (do not optimize it).

        Commonly used for the first pose to eliminate gauge freedom.

        Args:
            node_id: Node ID to fix
        """
        self._fixed_nodes.add(node_id)

    def get_pose(self, node_id: int) -> SE3:
        """Get current pose estimate for a node.

        Args:
            node_id: Node ID

        Returns:
            SE3 pose
        """
        return SE3(self._nodes[node_id])

    def get_poses(self) -> Dict[int, Tensor]:
        """Get all current poses.

        Returns:
            Dict mapping node_id → SE(3) pose [7]
        """
        return {k: v.clone() for k, v in self._nodes.items()}

    def total_cost(self) -> float:
        """Compute total cost over all factors.

        Returns:
            Total cost as Python float
        """
        total = 0.0
        for factor in self._factors:
            e = factor.residual(self._nodes)
            total += (e @ factor.information @ e).item()
        return total

    def optimize(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-6,
        damping: float = 1e-4,
        kernel: Optional[RobustKernel] = None,
        verbose: bool = False,
    ) -> dict:
        """Run Gauss-Newton optimization on the pose graph.

        Args:
            max_iterations: Maximum GN iterations
            tolerance: Convergence tolerance
            damping: LM damping parameter
            kernel: Robust kernel (None = standard least-squares)
            verbose: Print iteration info

        Returns:
            info dict with 'iterations', 'costs', 'converged'
        """
        optimizer = ManifoldGaussNewton(
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            kernel=kernel,
        )

        optimized_poses, info = optimizer.optimize(
            initial_poses=self._nodes,
            factors=list(self._factors),
            fixed_nodes=self._fixed_nodes,
        )

        if verbose:
            for i, c in enumerate(info['costs']):
                print(f"  GN iter {i:3d}: cost = {c:.6f}")
            status = "✓ converged" if info['converged'] else "✗ max iter"
            print(f"  [{status}] in {info['iterations']} iterations")

        # Update internal state
        self._nodes = optimized_poses

        return info

    def extract_trajectory(self) -> Tuple[Tensor, Tensor]:
        """Extract trajectory as arrays of positions and rotations.

        Returns:
            positions: [N, 3] — xyz positions in world frame
            quaternions: [N, 4] — rotation quaternions (wxyz)
        """
        ids = sorted(self._nodes.keys())
        positions = []
        quaternions = []
        for nid in ids:
            T = self._nodes[nid]
            quaternions.append(T[:4])
            positions.append(T[4:7])
        return torch.stack(positions), torch.stack(quaternions)
