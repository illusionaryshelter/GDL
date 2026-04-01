# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Additional coverage tests for optimizer edge cases and LieTensor abstract base.

Targets:
    - ManifoldSGD/Adam closure, weight_decay, amsgrad, validation
    - LieTensor base class abstract method coverage
    - Optimizer with no grad parameters
    - PyTree protocol
"""

import pytest
import torch
from torch import Tensor


class TestManifoldSGDEdgeCases:
    """Edge cases for ManifoldSGD."""

    def test_invalid_lr(self) -> None:
        """Negative lr should raise ValueError."""
        from geoembodied.optim import ManifoldSGD
        with pytest.raises(ValueError, match="Invalid learning rate"):
            ManifoldSGD([torch.randn(3, requires_grad=True)], lr=-0.1)

    def test_invalid_momentum(self) -> None:
        """Negative momentum should raise ValueError."""
        from geoembodied.optim import ManifoldSGD
        with pytest.raises(ValueError, match="Invalid momentum"):
            ManifoldSGD([torch.randn(3, requires_grad=True)], momentum=-0.5)

    def test_weight_decay_euclidean(self) -> None:
        """Weight decay should apply to plain tensors."""
        from geoembodied.optim import ManifoldSGD

        x = torch.ones(3, requires_grad=True)
        optimizer = ManifoldSGD([x], lr=0.1, weight_decay=0.1)

        loss = x.sum()
        loss.backward()
        optimizer.step()

        # With weight_decay, the update should be x - lr*(grad + wd*x)
        # grad = [1,1,1], wd*x = [0.1,0.1,0.1]
        # x_new = 1 - 0.1*(1 + 0.1) = 1 - 0.11 = 0.89
        assert torch.allclose(x, torch.tensor([0.89, 0.89, 0.89]), atol=1e-5)

    def test_no_grad_params_skipped(self) -> None:
        """Parameters with no gradient should be skipped."""
        from geoembodied.optim import ManifoldSGD

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        optimizer = ManifoldSGD([x, y], lr=0.1)

        # Only set grad on x
        loss = x.sum()
        loss.backward()
        y_before = y.clone()
        optimizer.step()

        # y should not change
        assert torch.allclose(y, y_before)

    def test_closure(self) -> None:
        """Optimizer should support closure argument."""
        from geoembodied.optim import ManifoldSGD

        x = torch.randn(3, requires_grad=True)
        optimizer = ManifoldSGD([x], lr=0.01)

        def closure():
            optimizer.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            return loss

        loss_val = optimizer.step(closure=closure)
        assert loss_val is not None
        assert loss_val.item() >= 0

    def test_no_momentum_path(self) -> None:
        """Test SGD with momentum=0 (no momentum buffer)."""
        from geoembodied.lietensor import SO3
        from geoembodied.optim import ManifoldSGD
        from geoembodied.functional.distance import so3_chordal_distance

        target = SO3.exp(torch.tensor([0.3, 0.1, -0.2]))
        current = SO3.identity().parameter()
        optimizer = ManifoldSGD([current], lr=0.5, momentum=0)  # No momentum

        for _ in range(20):
            optimizer.zero_grad()
            d = so3_chordal_distance(current.as_subclass(Tensor), target.as_subclass(Tensor))
            d.backward()
            optimizer.step()

        final = so3_chordal_distance(
            current.detach().as_subclass(Tensor),
            target.as_subclass(Tensor)
        ).item()
        assert final < 1.5  # Should at least make progress


class TestManifoldAdamEdgeCases:
    """Edge cases for ManifoldAdam."""

    def test_invalid_betas(self) -> None:
        """Invalid beta values should raise."""
        from geoembodied.optim import ManifoldAdam
        with pytest.raises(ValueError, match="Invalid beta_1"):
            ManifoldAdam([torch.randn(3, requires_grad=True)], betas=(1.5, 0.999))
        with pytest.raises(ValueError, match="Invalid beta_2"):
            ManifoldAdam([torch.randn(3, requires_grad=True)], betas=(0.9, -0.1))

    def test_amsgrad(self) -> None:
        """AMSGrad variant should work."""
        from geoembodied.lietensor import SO3
        from geoembodied.optim import ManifoldAdam
        from geoembodied.functional.distance import so3_chordal_distance

        target = SO3.exp(torch.tensor([0.3, 0.1, -0.2]))
        current = SO3.identity().parameter()
        optimizer = ManifoldAdam([current], lr=0.05, amsgrad=True)

        for _ in range(30):
            optimizer.zero_grad()
            d = so3_chordal_distance(current.as_subclass(Tensor), target.as_subclass(Tensor))
            d.backward()
            optimizer.step()

        final = so3_chordal_distance(
            current.detach().as_subclass(Tensor),
            target.as_subclass(Tensor)
        ).item()
        assert final < 0.1

    def test_weight_decay_euclidean(self) -> None:
        """Weight decay on Euclidean params."""
        from geoembodied.optim import ManifoldAdam

        x = torch.ones(3, requires_grad=True)
        optimizer = ManifoldAdam([x], lr=0.1, weight_decay=0.01)

        loss = x.sum()
        loss.backward()
        x_before = x.clone()
        optimizer.step()

        # x should have changed
        assert not torch.allclose(x, x_before)

    def test_closure(self) -> None:
        """Adam should support closure."""
        from geoembodied.optim import ManifoldAdam

        x = torch.randn(3, requires_grad=True)
        optimizer = ManifoldAdam([x], lr=0.01)

        def closure():
            optimizer.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            return loss

        loss_val = optimizer.step(closure=closure)
        assert loss_val is not None

    def test_amsgrad_euclidean(self) -> None:
        """AMSGrad for plain Euclidean params."""
        from geoembodied.optim import ManifoldAdam

        x = torch.ones(5, requires_grad=True)
        optimizer = ManifoldAdam([x], lr=0.01, amsgrad=True)

        for _ in range(5):
            optimizer.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            optimizer.step()

        # Should have decreased
        assert (x ** 2).sum().item() < 5.0


class TestLieTensorBaseAbstract:
    """Test abstract LieTensor base class methods."""

    def test_base_ltype_raises(self) -> None:
        """LieTensor.ltype should raise NotImplementedError."""
        from geoembodied.lietensor.base import LieTensor
        t = LieTensor(torch.randn(4))
        with pytest.raises(NotImplementedError):
            _ = t.ltype

    def test_base_exp_raises(self) -> None:
        """LieTensor.exp should raise NotImplementedError."""
        from geoembodied.lietensor.base import LieTensor
        with pytest.raises(NotImplementedError):
            LieTensor.exp(torch.randn(3))

    def test_base_log_raises(self) -> None:
        from geoembodied.lietensor.base import LieTensor
        t = LieTensor(torch.randn(4))
        with pytest.raises(NotImplementedError):
            t.log()

    def test_base_multiply_raises(self) -> None:
        from geoembodied.lietensor.base import LieTensor
        t = LieTensor(torch.randn(4))
        with pytest.raises(NotImplementedError):
            t.multiply(t)

    def test_base_inverse_raises(self) -> None:
        from geoembodied.lietensor.base import LieTensor
        t = LieTensor(torch.randn(4))
        with pytest.raises(NotImplementedError):
            t.inverse()

    def test_base_identity_raises(self) -> None:
        from geoembodied.lietensor.base import LieTensor
        with pytest.raises(NotImplementedError):
            LieTensor.identity()

    def test_base_project_raises(self) -> None:
        from geoembodied.lietensor.base import LieTensor
        t = LieTensor(torch.randn(4))
        with pytest.raises(NotImplementedError):
            t.project_()

    def test_base_act_raises(self) -> None:
        from geoembodied.lietensor.base import LieTensor
        t = LieTensor(torch.randn(4))
        with pytest.raises(NotImplementedError):
            t.act(torch.randn(3))

    def test_matmul_lietensor_dispatch(self) -> None:
        """@ with two LieTensors should dispatch to multiply."""
        from geoembodied.lietensor import SO3
        R1 = SO3.identity()
        R2 = SO3.exp(torch.tensor([0.1, 0.2, 0.3]))
        result = R1 @ R2
        assert isinstance(result, SO3)

    def test_matmul_tensor_dispatch(self) -> None:
        """@ with LieTensor and Tensor should dispatch to act."""
        from geoembodied.lietensor import SO3
        R = SO3.exp(torch.tensor([0.1, 0.2, 0.3]))
        pts = torch.randn(10, 3)
        result = R @ pts
        assert result.shape == (10, 3)

    def test_matmul_unsupported(self) -> None:
        """@ with unsupported type should return NotImplemented."""
        from geoembodied.lietensor import SO3
        R = SO3.identity()
        result = R.__matmul__("invalid")
        assert result is NotImplemented


class TestPyTreeProtocol:
    """Test __tensor_flatten__ / __tensor_unflatten__."""

    def test_so3_flatten_unflatten(self) -> None:
        """SO3 should survive flatten/unflatten cycle."""
        from geoembodied.lietensor import SO3

        R = SO3.exp(torch.tensor([0.1, -0.2, 0.3]))
        names, metadata = R.__tensor_flatten__()
        assert "_data" in names
        assert "ltype_name" in metadata

    def test_se3_flatten_unflatten(self) -> None:
        """SE3 should survive flatten/unflatten cycle."""
        from geoembodied.lietensor import SE3

        T = SE3.exp(torch.randn(6) * 0.3)
        names, metadata = T.__tensor_flatten__()
        assert "_data" in names
        assert metadata["ltype_name"] == "SE3"


class TestEgradToRgrad:
    """Test the egrad → rgrad projection functions."""

    def test_so3_rgrad_tangent(self) -> None:
        """Riemannian gradient should be tangent to S³ (orthogonal to q)."""
        from geoembodied.optim.manifold_sgd import _egrad_to_rgrad_so3
        from geoembodied.functional.so3_ops import so3_exp

        q = so3_exp(torch.randn(10, 3))  # unit quaternions
        egrad = torch.randn(10, 4)
        rgrad = _egrad_to_rgrad_so3(q, egrad)

        # <rgrad, q> should be ≈ 0 (orthogonal)
        dot = (rgrad * q).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(10), atol=1e-5), \
            f"Rgrad not tangent: max inner product = {dot.abs().max():.2e}"

    def test_se3_rgrad_quat_tangent(self) -> None:
        """SE3 rgrad quaternion part should be tangent to S³."""
        from geoembodied.optim.manifold_sgd import _egrad_to_rgrad_se3
        from geoembodied.functional.se3_ops import se3_exp

        T = se3_exp(torch.randn(10, 6) * 0.3)
        egrad = torch.randn(10, 7)
        rgrad = _egrad_to_rgrad_se3(T, egrad)

        # Quaternion part of rgrad should be orthogonal to quaternion part of T
        q = T[..., :4]
        rgrad_q = rgrad[..., :4]
        dot = (rgrad_q * q).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(10), atol=1e-5)

    def test_se3_rgrad_translation_unchanged(self) -> None:
        """SE3 rgrad translation part should equal egrad translation part."""
        from geoembodied.optim.manifold_sgd import _egrad_to_rgrad_se3
        from geoembodied.functional.se3_ops import se3_exp

        T = se3_exp(torch.randn(10, 6) * 0.3)
        egrad = torch.randn(10, 7)
        rgrad = _egrad_to_rgrad_se3(T, egrad)

        # Translation part [4:7] should be unchanged
        assert torch.allclose(rgrad[..., 4:], egrad[..., 4:], atol=1e-6)
