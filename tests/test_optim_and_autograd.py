# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for manifold optimizers and autograd through Lie group operations.

Verifies:
    1. ManifoldSGD/Adam converge on a simple pose regression task
    2. Autograd flows correctly through exp/log/multiply/act
    3. Mixed LieTensor + Tensor parameter groups work
    4. Periodic stabilization maintains manifold constraint
"""

import pytest
import torch
from torch import Tensor


class TestAutograd:
    """Test gradient flow through Lie group operations."""

    def test_so3_exp_grad(self) -> None:
        """Gradient flows through SO(3) exp map."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        omega = torch.randn(5, 3, requires_grad=True)
        q = so3_exp(omega)
        loss = q.sum()
        loss.backward()
        assert omega.grad is not None
        assert not torch.isnan(omega.grad).any()

    def test_se3_exp_grad(self) -> None:
        """Gradient flows through SE(3) exp map."""
        from geoembodied.functional.se3_ops import se3_exp

        xi = torch.randn(5, 6, requires_grad=True)
        T = se3_exp(xi)
        loss = T.sum()
        loss.backward()
        assert xi.grad is not None
        assert not torch.isnan(xi.grad).any()

    def test_se3_act_grad(self) -> None:
        """Gradient flows through SE(3) action on points."""
        from geoembodied.functional.se3_ops import se3_exp, se3_act

        xi = torch.randn(3, 6, requires_grad=True)
        T = se3_exp(xi)
        points = torch.randn(3, 50, 3, requires_grad=True)
        result = se3_act(T, points)
        loss = result.sum()
        loss.backward()

        assert xi.grad is not None
        assert points.grad is not None
        assert not torch.isnan(xi.grad).any()
        assert not torch.isnan(points.grad).any()

    def test_se3_multiply_grad(self) -> None:
        """Gradient flows through SE(3) group multiplication."""
        from geoembodied.functional.se3_ops import se3_exp, se3_multiply

        xi1 = torch.randn(4, 6, requires_grad=True)
        xi2 = torch.randn(4, 6, requires_grad=True)
        T1 = se3_exp(xi1)
        T2 = se3_exp(xi2)
        T3 = se3_multiply(T1, T2)
        loss = T3.sum()
        loss.backward()

        assert xi1.grad is not None
        assert xi2.grad is not None
        assert not torch.isnan(xi1.grad).any()
        assert not torch.isnan(xi2.grad).any()

    def test_so3_log_grad(self) -> None:
        """Gradient flows through SO(3) log map."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        omega = torch.randn(5, 3, requires_grad=True)
        q = so3_exp(omega)
        omega_back = so3_log(q)
        loss = omega_back.sum()
        loss.backward()

        assert omega.grad is not None
        assert not torch.isnan(omega.grad).any()

    def test_se3_log_grad(self) -> None:
        """Gradient flows through SE(3) log map."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        xi = torch.randn(5, 6, requires_grad=True)
        T = se3_exp(xi)
        xi_back = se3_log(T)
        loss = xi_back.sum()
        loss.backward()

        assert xi.grad is not None
        assert not torch.isnan(xi.grad).any()

    def test_chordal_distance_grad(self) -> None:
        """Gradient flows through chordal distance."""
        from geoembodied.functional.so3_ops import so3_exp
        from geoembodied.functional.distance import so3_chordal_distance

        omega1 = torch.randn(10, 3, requires_grad=True)
        omega2 = torch.randn(10, 3, requires_grad=True)
        q1 = so3_exp(omega1)
        q2 = so3_exp(omega2)
        d = so3_chordal_distance(q1, q2)
        loss = d.sum()
        loss.backward()

        assert omega1.grad is not None
        assert omega2.grad is not None
        assert not torch.isnan(omega1.grad).any()

    def test_lietensor_wrapper_grad(self) -> None:
        """Gradient flows through LieTensor wrapper @ operator."""
        from geoembodied.lietensor import SE3

        xi = torch.randn(3, 6, requires_grad=True)
        T = SE3.exp(xi)
        points = torch.randn(3, 20, 3)
        result = T @ points
        loss = result.sum()
        loss.backward()

        assert xi.grad is not None
        assert not torch.isnan(xi.grad).any()


class TestManifoldSGD:
    """Test ManifoldSGD optimizer."""

    def test_so3_rotation_regression(self) -> None:
        """ManifoldSGD can regress a SO(3) rotation.

        Task: Given target rotation R_target, optimize R to minimize
        chordal distance d(R, R_target).
        """
        from geoembodied.lietensor import SO3
        from geoembodied.optim import ManifoldSGD
        from geoembodied.functional.distance import so3_chordal_distance

        torch.manual_seed(42)

        # Target rotation (fixed)
        target = SO3.exp(torch.tensor([0.5, -0.3, 0.8]))

        # Initial rotation (optimizable via .parameter())
        current = SO3.exp(torch.randn(3) * 0.1).parameter()

        optimizer = ManifoldSGD([current], lr=0.1, momentum=0.9)

        initial_dist = so3_chordal_distance(
            current.detach().as_subclass(Tensor),
            target.as_subclass(Tensor),
        ).item()

        for _ in range(100):
            optimizer.zero_grad()
            dist = so3_chordal_distance(
                current.as_subclass(Tensor),
                target.as_subclass(Tensor),
            )
            dist.backward()
            optimizer.step()

        final_dist = so3_chordal_distance(
            current.detach().as_subclass(Tensor),
            target.as_subclass(Tensor),
        ).item()

        assert final_dist < initial_dist * 0.1, \
            f"SGD didn't converge: initial={initial_dist:.4f}, final={final_dist:.4f}"

    def test_se3_pose_regression(self) -> None:
        """ManifoldSGD can regress a SE(3) pose via point cloud alignment.

        Task: Given target SE(3) transform T_target and source points,
        optimize T to minimize ‖T @ points - T_target @ points‖².
        """
        from geoembodied.lietensor import SE3
        from geoembodied.optim import ManifoldSGD

        torch.manual_seed(7)

        # Source points
        points = torch.randn(50, 3)

        # Target transform
        target = SE3.exp(torch.tensor([0.3, -0.2, 0.5, 1.0, -0.5, 0.2]))
        target_points = (target @ points).detach()

        # Initial estimate (identity, optimizable)
        current = SE3.identity().parameter()

        optimizer = ManifoldSGD([current], lr=0.01, momentum=0.9)

        initial_error = (current @ points - target_points).norm().item()

        for _ in range(500):
            optimizer.zero_grad()
            transformed = current @ points
            loss = ((transformed - target_points) ** 2).sum()
            loss.backward()
            optimizer.step()

        final_error = (current @ points - target_points).detach().norm().item()

        assert final_error < initial_error * 0.5, \
            f"SE3 SGD didn't converge: initial={initial_error:.4f}, final={final_error:.4f}"


class TestManifoldAdam:
    """Test ManifoldAdam optimizer."""

    def test_so3_regression_adam(self) -> None:
        """ManifoldAdam converges faster than SGD on rotation regression."""
        from geoembodied.lietensor import SO3
        from geoembodied.optim import ManifoldAdam
        from geoembodied.functional.distance import so3_chordal_distance

        torch.manual_seed(42)

        target = SO3.exp(torch.tensor([0.5, -0.3, 0.8]))
        current = SO3.exp(torch.randn(3) * 0.1).parameter()

        optimizer = ManifoldAdam([current], lr=0.05)

        for _ in range(50):
            optimizer.zero_grad()
            dist = so3_chordal_distance(
                current.as_subclass(Tensor),
                target.as_subclass(Tensor),
            )
            dist.backward()
            optimizer.step()

        final_dist = so3_chordal_distance(
            current.detach().as_subclass(Tensor),
            target.as_subclass(Tensor),
        ).item()

        assert final_dist < 0.01, f"Adam didn't converge: final_dist={final_dist:.6f}"

    def test_se3_regression_adam(self) -> None:
        """ManifoldAdam on SE(3) pose regression."""
        from geoembodied.lietensor import SE3
        from geoembodied.optim import ManifoldAdam

        torch.manual_seed(13)

        points = torch.randn(100, 3)
        target = SE3.exp(torch.tensor([0.2, -0.1, 0.4, 0.5, -0.3, 0.1]))
        target_points = (target @ points).detach()

        current = SE3.identity().parameter()

        optimizer = ManifoldAdam([current], lr=0.02)

        for _ in range(150):
            optimizer.zero_grad()
            loss = ((current @ points - target_points) ** 2).sum()
            loss.backward()
            optimizer.step()

        final_error = (current @ points - target_points).detach().norm().item()

        assert final_error < 0.5, \
            f"SE3 Adam didn't converge: final_error={final_error:.4f}"

    def test_mixed_params(self) -> None:
        """ManifoldAdam handles mixed LieTensor + Tensor params."""
        from geoembodied.lietensor import SO3
        from geoembodied.optim import ManifoldAdam

        rot = SO3.identity().parameter()
        vec = torch.randn(3, requires_grad=True)

        optimizer = ManifoldAdam([rot, vec], lr=0.01)

        # Dummy loss using both
        loss = rot.as_subclass(Tensor).sum() + vec.sum()
        loss.backward()
        optimizer.step()  # Should not crash

        # Verify rotation is still normalized (unit quaternion)
        q_norm = rot.as_subclass(Tensor).norm().item()
        assert abs(q_norm - 1.0) < 0.1, f"Quaternion norm drifted: {q_norm}"

    def test_stabilize_maintains_manifold(self) -> None:
        """Periodic stabilization keeps quaternion on unit sphere."""
        from geoembodied.lietensor import SO3
        from geoembodied.optim import ManifoldAdam

        torch.manual_seed(99)

        rot = SO3.exp(torch.randn(3) * 0.5).parameter()

        optimizer = ManifoldAdam([rot], lr=0.1, stabilize_every=5)

        for i in range(20):
            optimizer.zero_grad()
            loss = rot.as_subclass(Tensor).sum()
            loss.backward()
            optimizer.step()

        # After stabilization, norm should be very close to 1
        q_norm = rot.as_subclass(Tensor).detach().norm().item()
        assert abs(q_norm - 1.0) < 0.01, \
            f"After stabilization, quaternion norm = {q_norm}, should be ≈ 1.0"
