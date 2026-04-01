# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Pose Graph SLAM demo — differentiable loop closure optimization.

Demonstrates GeoEmbodied's PoseGraph + ManifoldGaussNewton on a
synthetic 2D SLAM problem:

    1. Robot drives in an oval trajectory (50 poses)
    2. Odometry measurements accumulate drift
    3. Loop closure detected when robot returns to start
    4. Gauss-Newton optimizes the pose graph on SE(3) manifold
    5. Before vs After: drift eliminated

The demo showcases:
    - Proper Lie group residuals (Log map, not Euclidean subtraction)
    - Quadratic convergence of manifold Gauss-Newton
    - Robust kernels filtering outlier loop closures
    - All computations differentiable via torch.autograd

Usage:
    python examples/slam/pose_graph_slam.py
    python examples/slam/pose_graph_slam.py --num_poses 100 --noise 0.03
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geoembodied.lietensor.se3 import SE3
from geoembodied.slam import PoseGraph
from geoembodied.optim import HuberKernel
from geoembodied.functional.se3_ops import (
    se3_exp, se3_multiply, se3_inverse,
)


def generate_oval_trajectory(
    num_poses: int = 50,
    a: float = 5.0,
    b: float = 3.0,
) -> list:
    """Generate ground truth poses on an oval trajectory.

    Robot drives counterclockwise on an ellipse in the XY plane.
    Each pose includes heading tangent to the ellipse.

    Args:
        num_poses: Number of poses
        a: Semi-major axis (X)
        b: Semi-minor axis (Y)

    Returns:
        List of SE(3) poses [7] (ground truth)
    """
    poses = []
    for i in range(num_poses):
        t = 2.0 * math.pi * i / num_poses

        # Position on ellipse
        x = a * math.cos(t)
        y = b * math.sin(t)

        # Heading: tangent to ellipse
        dx = -a * math.sin(t)
        dy = b * math.cos(t)
        yaw = math.atan2(dy, dx)

        # Create SE(3) pose (rotation about Z axis + translation)
        xi = torch.tensor([0.0, 0.0, yaw, x, y, 0.0])
        poses.append(se3_exp(xi))

    return poses


def generate_noisy_odometry(
    gt_poses: list,
    noise_std: float = 0.02,
) -> list:
    """Generate noisy odometry measurements.

    T_ij = T_i^{-1} ∘ T_j + noise

    Args:
        gt_poses: Ground truth poses
        noise_std: Noise standard deviation on twist

    Returns:
        List of relative transforms (noisy odometry)
    """
    odometries = []
    N = len(gt_poses)
    for i in range(N):
        j = (i + 1) % N
        T_ij = se3_multiply(se3_inverse(gt_poses[i]), gt_poses[j])

        # Add noise in tangent space
        noise = se3_exp(torch.randn(6) * noise_std)
        T_ij_noisy = se3_multiply(T_ij, noise)
        odometries.append(T_ij_noisy)

    return odometries


def dead_reckon(start_pose: torch.Tensor, odometries: list) -> list:
    """Compute dead-reckoned trajectory (accumulates drift).

    Args:
        start_pose: Initial SE(3) pose [7]
        odometries: List of relative transforms

    Returns:
        List of dead-reckoned poses
    """
    poses = [start_pose.clone()]
    for odom in odometries[:-1]:  # Don't use loop closure odom
        next_pose = se3_multiply(poses[-1], odom)
        poses.append(next_pose)
    return poses


def main():
    parser = argparse.ArgumentParser(
        description="GeoEmbodied SLAM Demo: Pose Graph Optimization"
    )
    parser.add_argument('--num_poses', type=int, default=50,
                        help='Number of poses in the trajectory')
    parser.add_argument('--noise', type=float, default=0.03,
                        help='Odometry noise standard deviation')
    parser.add_argument('--outlier_ratio', type=float, default=0.0,
                        help='Fraction of outlier loop closures')
    parser.add_argument('--robust', action='store_true',
                        help='Use Huber robust kernel')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  GeoEmbodied SLAM Demo: Pose Graph Optimization")
    print(f"{'='*60}")
    print(f"  Poses: {args.num_poses}")
    print(f"  Odometry noise: {args.noise}")
    print(f"  Robust kernel: {'Huber' if args.robust else 'None'}")
    print(f"{'='*60}\n")

    torch.manual_seed(42)

    # ── 1. Generate Ground Truth ────────────────────────────
    print("1. Generating oval trajectory...")
    gt_poses = generate_oval_trajectory(args.num_poses)
    print(f"   {len(gt_poses)} ground truth poses on ellipse")

    # ── 2. Generate Noisy Odometry ──────────────────────────
    print("2. Generating noisy odometry...")
    odometries = generate_noisy_odometry(gt_poses, args.noise)

    # ── 3. Dead Reckoning (before optimization) ─────────────
    print("3. Computing dead-reckoned trajectory...")
    dr_poses = dead_reckon(gt_poses[0], odometries)

    # Compute drift
    dr_errors = []
    for i, (dr, gt) in enumerate(zip(dr_poses, gt_poses)):
        err = (dr[4:7] - gt[4:7]).norm().item()
        dr_errors.append(err)

    max_drift = max(dr_errors)
    end_drift = dr_errors[-1]
    avg_drift = sum(dr_errors) / len(dr_errors)
    print(f"   Dead reckoning drift:")
    print(f"     Max: {max_drift:.4f}m, End: {end_drift:.4f}m, Avg: {avg_drift:.4f}m")

    # ── 4. Build Pose Graph ─────────────────────────────────
    print("4. Building pose graph...")
    graph = PoseGraph()

    # Add nodes (initialized from dead reckoning)
    for i, pose in enumerate(dr_poses):
        graph.add_node(i, pose)

    # Information matrix for odometry (higher = more trustworthy)
    odom_info = torch.eye(6) * (1.0 / (args.noise ** 2 + 1e-8))
    loop_info = torch.eye(6) * (1.0 / (args.noise ** 2 + 1e-8)) * 0.5

    # Add odometry factors
    for i in range(args.num_poses - 1):
        graph.add_between(i, i + 1, odometries[i], odom_info)

    # Add loop closure (last → first)
    graph.add_between(args.num_poses - 1, 0, odometries[-1], loop_info)

    # Add optional outlier loop closures
    n_outliers = int(args.num_poses * args.outlier_ratio)
    for k in range(n_outliers):
        i = torch.randint(0, args.num_poses, (1,)).item()
        j = torch.randint(0, args.num_poses, (1,)).item()
        if i != j:
            # Random bogus measurement
            fake_odom = se3_exp(torch.randn(6) * 2.0)
            graph.add_between(i, j, fake_odom, loop_info * 0.1)

    # Fix first pose (gauge freedom)
    graph.add_prior(0, gt_poses[0], odom_info * 1000)
    graph.fix_node(0)

    n_total_factors = graph.num_factors
    print(f"   {graph.num_nodes} nodes, {n_total_factors} factors")
    print(f"   (including {n_outliers} outlier loop closures)")

    cost_before = graph.total_cost()
    print(f"   Initial cost: {cost_before:.4f}")

    # ── 5. Optimize ─────────────────────────────────────────
    print("5. Running Gauss-Newton optimization...")
    kernel = HuberKernel(delta=0.5) if args.robust else None

    info = graph.optimize(
        max_iterations=30,
        tolerance=1e-8,
        damping=1e-4,
        kernel=kernel,
        verbose=True,
    )

    cost_after = graph.total_cost()

    # ── 6. Evaluate ─────────────────────────────────────────
    print(f"\n6. Results:")
    print(f"   Converged: {'✓' if info['converged'] else '✗'}")
    print(f"   Iterations: {info['iterations']}")
    print(f"   Cost: {cost_before:.4f} → {cost_after:.6f}")

    # Compute optimized errors
    opt_errors = []
    for i in range(args.num_poses):
        T_opt = graph._nodes[i]
        err = (T_opt[4:7] - gt_poses[i][4:7]).norm().item()
        opt_errors.append(err)

    max_opt_err = max(opt_errors)
    end_opt_err = opt_errors[-1]
    avg_opt_err = sum(opt_errors) / len(opt_errors)

    print(f"\n  {'Metric':<25} | {'Dead Reckon':>12} | {'Optimized':>12} | {'Improvement':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Max position error':<25} | {max_drift:>11.4f}m | {max_opt_err:>11.4f}m | {max_drift/max(max_opt_err,1e-8):>11.1f}×")
    print(f"  {'Endpoint error':<25} | {end_drift:>11.4f}m | {end_opt_err:>11.4f}m | {end_drift/max(end_opt_err,1e-8):>11.1f}×")
    print(f"  {'Average error':<25} | {avg_drift:>11.4f}m | {avg_opt_err:>11.4f}m | {avg_drift/max(avg_opt_err,1e-8):>11.1f}×")

    print(f"\n{'='*60}")
    print(f"  Key: Gauss-Newton on SE(3) manifold eliminates drift.")
    print(f"  Loop closure propagates correction to ALL poses.")
    print(f"  Convergence: {info['iterations']} iters (quadratic rate).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
