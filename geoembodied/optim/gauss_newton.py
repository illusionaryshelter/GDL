# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Gauss-Newton optimizer on SE(3) manifold.

Second-order optimizer for nonlinear least-squares problems on Lie groups.
Achieves quadratic convergence near the optimum — orders of magnitude
faster than first-order methods (ManifoldAdam/SGD) for SLAM problems.

Mathematical framework:
    minimize  Σ_k  e_k(x)^T Σ_k^{-1} e_k(x)

    where e_k are factor residuals and x are SE(3) poses.

    Gauss-Newton update (in tangent space):
        J = de/dξ     (Jacobian w.r.t. Lie algebra perturbation)
        H ≈ J^T W J   (Gauss-Newton Hessian approximation)
        g = J^T W e   (gradient)
        δξ = -H^{-1} g   (update step)

    Retraction to manifold (AGENTS.md Rule 2):
        T_new = Exp(δξ) ∘ T_old

    The Jacobian is computed via torch.autograd — no manual derivation
    needed (but analytic Jacobians can be provided for speed).

Supports:
    - Damping (Levenberg-Marquardt style): H + λI
    - Robust kernels (Huber, Cauchy) via IRLS
    - Batched optimization of independent problems
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from geoembodied.functional.se3_ops import se3_exp, se3_multiply
from geoembodied.functional.quaternion_ops import quaternion_normalize
from geoembodied.optim.robust_kernel import RobustKernel, TrivialKernel


class ManifoldGaussNewton:
    """Gauss-Newton optimizer for SE(3) pose graph problems.

    Iteratively linearizes the factors, solves the normal equations
    in the tangent space (Lie algebra), and retracts to the manifold.

    Args:
        max_iterations: Maximum GN iterations
        tolerance: Convergence tolerance on step norm ‖δξ‖
        damping: Initial Levenberg-Marquardt damping λ
            λ > 0 regularizes toward gradient descent
        damping_increase: Factor to increase damping on bad step
        damping_decrease: Factor to decrease damping on good step
        kernel: Robust kernel for outlier handling

    Example::

        >>> from geoembodied.optim.factor import BetweenFactor, PriorFactor
        >>> optimizer = ManifoldGaussNewton(max_iterations=20)
        >>> poses, info = optimizer.optimize(
        ...     initial_poses={0: T0, 1: T1, 2: T2},
        ...     factors=[prior, odom_01, odom_12, loop_20],
        ...     fixed_nodes={0},  # anchor first pose
        ... )
    """

    def __init__(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-6,
        damping: float = 1e-4,
        damping_increase: float = 10.0,
        damping_decrease: float = 0.1,
        kernel: Optional[RobustKernel] = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.damping_increase = damping_increase
        self.damping_decrease = damping_decrease
        self.kernel = kernel or TrivialKernel()

    def optimize(
        self,
        initial_poses: Dict[int, Tensor],
        factors: list,
        fixed_nodes: Optional[set] = None,
    ) -> Tuple[Dict[int, Tensor], dict]:
        """Run Gauss-Newton optimization.

        Args:
            initial_poses: Dict mapping node_id → SE(3) pose [7]
            factors: List of Factor objects (constraints)
            fixed_nodes: Set of node IDs to keep fixed (not optimized)

        Returns:
            Tuple of:
                - optimized_poses: Dict mapping node_id → optimized SE(3) pose [7]
                - info: Dict with convergence information
                    'iterations': number of iterations
                    'costs': list of total costs per iteration
                    'converged': whether tolerance was reached
        """
        if fixed_nodes is None:
            fixed_nodes = set()

        # Copy poses to avoid modifying originals
        poses = {k: v.clone() for k, v in initial_poses.items()}

        # Identify optimization variables (non-fixed nodes)
        free_nodes = sorted(set(poses.keys()) - fixed_nodes)
        n_free = len(free_nodes)
        node_to_idx = {node: i for i, node in enumerate(free_nodes)}

        if n_free == 0:
            return poses, {'iterations': 0, 'costs': [], 'converged': True}

        costs = []
        current_damping = self.damping

        for iteration in range(self.max_iterations):
            # 1. Compute total cost
            total_cost = self._compute_total_cost(poses, factors)
            costs.append(total_cost)

            # 2. Build linear system: H δξ = -g
            H, g = self._build_linear_system(
                poses, factors, free_nodes, node_to_idx,
            )

            # 3. Solve with damping: (H + λI) δξ = -g
            n_vars = n_free * 6
            H_damped = H + current_damping * torch.eye(n_vars, device=H.device)

            try:
                delta_xi = torch.linalg.solve(H_damped, -g)
            except torch.linalg.LinAlgError:
                # Singular — increase damping
                current_damping *= self.damping_increase
                continue

            # 4. Check convergence
            step_norm = delta_xi.norm().item()
            if step_norm < self.tolerance:
                return poses, {
                    'iterations': iteration + 1,
                    'costs': costs,
                    'converged': True,
                }

            # 5. Retract: T_new = Exp(δξ) ∘ T_old (AGENTS.md Rule 2)
            new_poses = self._retract(poses, delta_xi, free_nodes, node_to_idx)

            # 6. Check if step improved the cost
            new_cost = self._compute_total_cost(new_poses, factors)

            if new_cost < total_cost:
                # Good step — accept and decrease damping
                poses = new_poses
                current_damping = max(
                    current_damping * self.damping_decrease,
                    1e-10,
                )
            else:
                # Bad step — reject and increase damping
                current_damping *= self.damping_increase

        # Max iterations reached
        total_cost = self._compute_total_cost(poses, factors)
        costs.append(total_cost)

        return poses, {
            'iterations': self.max_iterations,
            'costs': costs,
            'converged': False,
        }

    def _compute_total_cost(
        self,
        poses: Dict[int, Tensor],
        factors: list,
    ) -> float:
        """Compute total cost over all factors.

        Returns:
            Total cost as Python float
        """
        total = 0.0
        for factor in factors:
            e = factor.residual(poses)
            sq = (e @ factor.information @ e).item()
            total += self.kernel.evaluate(
                torch.tensor(sq)
            ).item()
        return total

    def _build_linear_system(
        self,
        poses: Dict[int, Tensor],
        factors: list,
        free_nodes: list,
        node_to_idx: dict,
    ) -> Tuple[Tensor, Tensor]:
        """Build the Gauss-Newton normal equations.

        Uses autograd to compute Jacobians of residuals w.r.t.
        Lie algebra perturbation δξ.

        The key trick: parameterize T(δξ) = Exp(δξ) ∘ T_current,
        then differentiate residual w.r.t. δξ at δξ = 0.

        Returns:
            H: Approximate Hessian [6N, 6N]
            g: Gradient [6N]
        """
        n_free = len(free_nodes)
        n_vars = n_free * 6
        device = next(iter(poses.values())).device

        H = torch.zeros(n_vars, n_vars, device=device)
        g = torch.zeros(n_vars, device=device)

        for factor in factors:
            # Create perturbation variables
            perturbations = {}  # node_id → δξ_i
            for node_id in factor.connected_nodes:
                if node_id in node_to_idx:
                    xi = torch.zeros(6, device=device, requires_grad=True)
                    perturbations[node_id] = xi

            # Build perturbed poses: T(δξ) = Exp(δξ) ∘ T_current
            perturbed_poses = {}
            for node_id in factor.connected_nodes:
                T_current = poses[node_id]
                if node_id in perturbations:
                    xi = perturbations[node_id]
                    T_perturbed = se3_multiply(se3_exp(xi), T_current)
                    perturbed_poses[node_id] = T_perturbed
                else:
                    perturbed_poses[node_id] = T_current

            # Compute residual at perturbation
            e = factor.residual(perturbed_poses)

            # Compute Jacobian via autograd: J_i = de/dξ_i
            jacobians = {}  # node_id → [res_dim, 6]
            for node_id, xi in perturbations.items():
                J_rows = []
                for k in range(e.shape[0]):
                    grads = torch.autograd.grad(
                        e[k], xi,
                        retain_graph=True,
                        create_graph=False,
                    )[0]
                    J_rows.append(grads)
                jacobians[node_id] = torch.stack(J_rows)

            # Residual at current point (δξ = 0)
            e_val = e.detach()

            # Robust kernel weight
            sq_residual = (e_val @ factor.information @ e_val)
            w = self.kernel.weight(sq_residual)

            # Accumulate into H and g
            # H += J_i^T W J_j  (for all pairs i,j in factor)
            # g += J_i^T W e
            W = w * factor.information  # weighted information

            for node_i, J_i in jacobians.items():
                idx_i = node_to_idx[node_i]
                s_i = idx_i * 6
                e_i = s_i + 6

                # g += J_i^T @ W @ e
                g[s_i:e_i] += J_i.T @ W @ e_val

                for node_j, J_j in jacobians.items():
                    idx_j = node_to_idx[node_j]
                    s_j = idx_j * 6
                    e_j = s_j + 6

                    # H += J_i^T @ W @ J_j
                    H[s_i:e_i, s_j:e_j] += J_i.T @ W @ J_j

        return H, g

    def _retract(
        self,
        poses: Dict[int, Tensor],
        delta_xi: Tensor,
        free_nodes: list,
        node_to_idx: dict,
    ) -> Dict[int, Tensor]:
        """Retract update to manifold.

        T_new = Exp(δξ) ∘ T_old  (AGENTS.md Rule 2)

        Args:
            poses: Current poses
            delta_xi: Stacked update [6N]
            free_nodes: List of free node IDs
            node_to_idx: Node ID → index mapping

        Returns:
            Updated poses
        """
        new_poses = {k: v.clone() for k, v in poses.items()}

        for node_id in free_nodes:
            idx = node_to_idx[node_id]
            xi = delta_xi[idx * 6: (idx + 1) * 6]

            T_current = poses[node_id]
            # T_new = Exp(δξ) ∘ T_old
            delta_T = se3_exp(xi)
            T_new = se3_multiply(delta_T, T_current)

            # Normalize quaternion (AGENTS.md Rule 5)
            T_new[..., :4] = quaternion_normalize(T_new[..., :4])

            new_poses[node_id] = T_new

        return new_poses
