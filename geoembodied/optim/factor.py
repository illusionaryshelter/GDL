# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Factor graph factors for SLAM and geometric optimization.

A factor encodes a probabilistic constraint between variables.
In SLAM, each factor corresponds to a measurement (odometry,
loop closure, landmark observation) with associated uncertainty.

Mathematical framework:
    Given variables x and measurement z with noise model Σ,
    the factor computes:
        residual e(x) = measurement_model(x) ⊖ z
        cost = e(x)^T Σ^{-1} e(x)

    For SE(3) variables, ⊖ is the logarithm map:
        e = Log(T_z^{-1} ∘ T_pred)

    This is the proper Lie-group residual — NOT T_pred - T_z.

All factors are nn.Modules for torch.autograd compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from geoembodied.lietensor.se3 import SE3
from geoembodied.functional.se3_ops import (
    se3_multiply,
    se3_inverse,
    se3_log,
)


class Factor(nn.Module, ABC):
    """Abstract base class for all factor types.

    Subclasses must implement `residual()` which computes the
    error vector given current variable values.

    Attributes:
        connected_nodes: List of node IDs this factor connects
        information: Information matrix (Σ^{-1})
            shape: [dim, dim] where dim = residual dimension
    """

    def __init__(
        self,
        connected_nodes: list,
        information: Optional[Tensor] = None,
        residual_dim: int = 6,
    ) -> None:
        super().__init__()
        self.connected_nodes = connected_nodes
        self.residual_dim = residual_dim

        if information is not None:
            self.register_buffer('information', information)
        else:
            self.register_buffer(
                'information',
                torch.eye(residual_dim),
            )

    @abstractmethod
    def residual(self, poses: Dict[int, Tensor]) -> Tensor:
        """Compute the residual error vector.

        Args:
            poses: Dict mapping node_id → SE(3) pose [7]

        Returns:
            Residual vector ∈ R^{residual_dim}
        """
        ...

    def weighted_residual(self, poses: Dict[int, Tensor]) -> Tensor:
        """Compute information-weighted residual.

        √(Σ^{-1}) @ e(x)   (for use in least-squares: ‖√Σ^{-1} e‖² = e^T Σ^{-1} e)

        Args:
            poses: Dict mapping node_id → SE(3) pose [7]

        Returns:
            Weighted residual, shape: [residual_dim]
        """
        e = self.residual(poses)
        # Cholesky for √(Σ^{-1})
        L = torch.linalg.cholesky(self.information)
        return L @ e

    def cost(self, poses: Dict[int, Tensor]) -> Tensor:
        """Compute cost = e^T Σ^{-1} e.

        Args:
            poses: Dict mapping node_id → SE(3) pose [7]

        Returns:
            Scalar cost
        """
        e = self.residual(poses)
        return e @ self.information @ e


class BetweenFactor(Factor):
    """Relative pose constraint between two SE(3) nodes.

    The most fundamental SLAM factor: encodes a measured relative
    transform T_ij between pose i and pose j.

    Residual:
        e = Log( T_ij^{-1} ∘ T_i^{-1} ∘ T_j )

    This is zero when T_j = T_i ∘ T_ij (perfect measurement).

    Args:
        node_i: Source pose node ID
        node_j: Target pose node ID
        measurement: Measured relative transform T_ij
            shape: [7] (compact SE(3): qw,qx,qy,qz,tx,ty,tz)
        information: Information matrix Σ^{-1}
            shape: [6, 6]
    """

    def __init__(
        self,
        node_i: int,
        node_j: int,
        measurement: Tensor,
        information: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            connected_nodes=[node_i, node_j],
            information=information,
            residual_dim=6,
        )
        self.node_i = node_i
        self.node_j = node_j
        # Store measurement and its inverse
        self.register_buffer('measurement', measurement.detach().clone())
        self.register_buffer(
            'measurement_inv',
            se3_inverse(measurement.detach().clone()),
        )

    def residual(self, poses: Dict[int, Tensor]) -> Tensor:
        """Compute relative pose residual.

        e = Log( T_ij^{-1} ∘ T_i^{-1} ∘ T_j )

        Args:
            poses: Dict mapping node_id → SE(3) pose [7]

        Returns:
            Residual twist ∈ R^6
        """
        T_i = poses[self.node_i]  # [7]
        T_j = poses[self.node_j]  # [7]

        # T_i^{-1} ∘ T_j
        T_i_inv = se3_inverse(T_i)
        T_ij_pred = se3_multiply(T_i_inv, T_j)

        # T_ij_meas^{-1} ∘ T_ij_pred
        error_se3 = se3_multiply(self.measurement_inv, T_ij_pred)

        # Log map → twist vector in R^6
        return se3_log(error_se3)


class PriorFactor(Factor):
    """Prior constraint on a single SE(3) node.

    Anchors a pose to a given value. Commonly used for:
        - First pose in SLAM (gauge fixing)
        - GPS/IMU prior
        - Localization constraint

    Residual:
        e = Log( T_prior^{-1} ∘ T_i )

    Args:
        node_id: Pose node ID
        prior: Prior pose value
            shape: [7] (compact SE(3))
        information: Information matrix Σ^{-1}
            shape: [6, 6]
    """

    def __init__(
        self,
        node_id: int,
        prior: Tensor,
        information: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            connected_nodes=[node_id],
            information=information,
            residual_dim=6,
        )
        self.node_id = node_id
        self.register_buffer('prior', prior.detach().clone())
        self.register_buffer('prior_inv', se3_inverse(prior.detach().clone()))

    def residual(self, poses: Dict[int, Tensor]) -> Tensor:
        """Compute prior residual.

        e = Log( T_prior^{-1} ∘ T_i )

        Args:
            poses: Dict mapping node_id → SE(3) pose [7]

        Returns:
            Residual twist ∈ R^6
        """
        T_i = poses[self.node_id]
        error_se3 = se3_multiply(self.prior_inv, T_i)
        return se3_log(error_se3)


class LandmarkFactor(Factor):
    """3D landmark observation factor.

    Constrains a pose T_i by observing a 3D landmark L at a known
    position in the camera/sensor frame.

    Residual:
        e = T_i^{-1} ∘ L - z

    where z is the observed landmark position in the sensor frame.

    This is a 3-DOF residual (x, y, z in sensor frame).

    Args:
        pose_node: Pose node ID
        landmark: 3D landmark position in world frame
            shape: [3]
        observation: Observed position in sensor frame
            shape: [3]
        information: Information matrix Σ^{-1}
            shape: [3, 3]
    """

    def __init__(
        self,
        pose_node: int,
        landmark: Tensor,
        observation: Tensor,
        information: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            connected_nodes=[pose_node],
            information=information,
            residual_dim=3,
        )
        self.pose_node = pose_node
        self.register_buffer('landmark', landmark.detach().clone())
        self.register_buffer('observation', observation.detach().clone())

    def residual(self, poses: Dict[int, Tensor]) -> Tensor:
        """Compute landmark observation residual.

        e = R_i^T (L - t_i) - z

        Args:
            poses: Dict mapping node_id → SE(3) pose [7]

        Returns:
            Residual ∈ R^3
        """
        from geoembodied.functional.se3_ops import se3_act

        T_i = poses[self.pose_node]
        T_i_inv = se3_inverse(T_i)

        # Transform landmark to sensor frame
        L_sensor = se3_act(T_i_inv, self.landmark)

        return L_sensor - self.observation
