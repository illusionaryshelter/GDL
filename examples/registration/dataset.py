# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Synthetic paired point cloud dataset for registration.

Generates structured point clouds with **geometrically distinctive** primitives
that provide well-conditioned rotation constraints:
    - Box: 8 corners + 12 edges → full 3-DoF rotation constraint
    - L-shape: asymmetric orthogonal structure → breaks all rotational symmetry
    - Torus: curved surface with non-trivial curvature gradient
    - Cylinder: 1 axis locked by curved surface + 2 flat caps with edges

Deliberately avoids degenerate primitives:
    ❌ Perfect spheres (rotation-unobservable, 0 DoF constraint)
    ❌ Infinite planes (only 1 DoF constraint, in-plane sliding)
    ❌ Gaussian blobs (no structure, no normal consistency)

Partial overlap uses single-sided cropping:
    Target = complete "map" point cloud
    Source = partial "scan" (random spatial crop of the SAME cloud)
    Correspondences are tracked exactly through the crop.
"""

import math
from typing import Tuple, Optional, List

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticRegistrationDataset(Dataset):
    """Point cloud registration dataset with geometrically rich primitives.

    Each sample generates a structured 3D shape, applies a random SE(3)
    transform, and optionally crops the source cloud to simulate partial
    overlap (single-sided: target stays complete).

    Args:
        num_samples: Number of training pairs
        num_points: Points per cloud (N)
        rotation_range: Maximum rotation angle in degrees
        translation_range: Maximum translation magnitude
        noise_std: Gaussian noise added to target cloud
        partial_ratio: Fraction of source points visible (1.0 = full)
            Uses single-sided spatial crop (source only).
            Target always has N points. Source has int(N * partial_ratio).
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_points: int = 512,
        rotation_range: float = 180.0,
        translation_range: float = 0.5,
        noise_std: float = 0.01,
        partial_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_points = num_points
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.partial_ratio = partial_ratio
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Generate a single registration pair.

        Returns:
            dict with keys:
                'source': [N_src, 3] source point cloud (possibly cropped)
                'target': [N, 3] target point cloud (always complete)
                'gt_rotation': [3, 3] ground truth rotation matrix
                'gt_translation': [3] ground truth translation
                'gt_quat': [4] ground truth quaternion (wxyz)
                'correspondences': [N_src] index mapping source[i] → target[j]
        """
        # 1. Generate geometrically distinctive point cloud
        cloud = self._generate_structured_cloud()  # [N, 3]

        # 2. Generate random SE(3) transform
        R, t, q = self._random_se3()

        # 3. Apply transform: target = R @ source + t + noise
        target = (R @ cloud.T).T + t.unsqueeze(0)
        if self.noise_std > 0:
            target = target + torch.randn_like(target) * self.noise_std

        # 4. Single-sided partial overlap (source crop only)
        N = self.num_points
        if self.partial_ratio < 1.0:
            n_visible = int(N * self.partial_ratio)
            # Spatial crop: pick a random center, keep closest points
            # This simulates a LiDAR scan seeing only one side
            crop_dir = torch.randn(3, generator=self.generator)
            crop_dir = crop_dir / crop_dir.norm().clamp(min=1e-8)
            # Project points onto crop direction
            projections = cloud @ crop_dir  # [N]
            # Keep the n_visible points with highest projection
            # (= points on one side of the object)
            _, sorted_idx = projections.sort(descending=True)
            src_idx = sorted_idx[:n_visible]

            source = cloud[src_idx]
            # Correspondences: source[i] corresponds to target[src_idx[i]]
            correspondences = src_idx
        else:
            source = cloud
            correspondences = torch.arange(N)

        return {
            'source': source,
            'target': target,
            'gt_rotation': R,
            'gt_translation': t,
            'gt_quat': q,
            'correspondences': correspondences,
        }

    def _generate_structured_cloud(self) -> Tensor:
        """Generate a point cloud from geometrically distinctive primitives.

        Combines 2-4 primitives chosen from:
            - Box (edges + corners): Full 3-DoF rotation constraint
            - L-shape: Asymmetric structure breaks all symmetry
            - Torus: Non-trivial curvature with asymmetric cross-section
            - Cylinder: Axis + edge constraint
            - Cone: Tip + curved surface

        Each primitive has corners, edges, or curvature gradients that
        provide strong geometric features for equivariant networks.

        Returns:
            Point cloud, shape: [N, 3]
        """
        N = self.num_points
        parts: List[Tensor] = []
        remaining = N

        # 2-4 random primitives
        n_parts = torch.randint(2, 5, (1,), generator=self.generator).item()

        for i in range(n_parts):
            if i == n_parts - 1:
                n_pts = remaining
            else:
                n_pts = max(1, remaining // (n_parts - i))
                n_pts = torch.randint(
                    n_pts // 2, n_pts + 1, (1,), generator=self.generator
                ).item()
            remaining -= n_pts
            if n_pts <= 0:
                continue

            # Choose from geometrically distinctive primitives
            ptype = torch.randint(0, 5, (1,), generator=self.generator).item()

            if ptype == 0:
                pts = self._sample_box(n_pts)
            elif ptype == 1:
                pts = self._sample_lshape(n_pts)
            elif ptype == 2:
                pts = self._sample_torus(n_pts)
            elif ptype == 3:
                pts = self._sample_cylinder(n_pts)
            else:
                pts = self._sample_cone(n_pts)

            # Random position offset
            center = torch.randn(3, generator=self.generator) * 0.3
            pts = pts + center

            parts.append(pts)

        cloud = torch.cat(parts, dim=0)[:N]

        # Center the cloud
        cloud = cloud - cloud.mean(dim=0, keepdim=True)

        # Random global scale (variety)
        scale = 0.5 + torch.rand(1, generator=self.generator).item() * 0.5
        cloud = cloud * scale

        return cloud

    def _sample_box(self, n_pts: int) -> Tensor:
        """Sample points on box surface (edges + faces with thickness).

        A box has 8 corners, 12 edges, and 6 faces — maximum geometric
        distinctiveness. Points are distributed with higher density on
        edges/corners to ensure they appear even at low N.

        Args:
            n_pts: Number of points to sample

        Returns:
            Points on box surface, shape: [n_pts, 3]
        """
        # Random box dimensions (asymmetric to break symmetry)
        dims = torch.rand(3, generator=self.generator) * 0.3 + 0.1  # [0.1, 0.4]

        pts_list = []

        # 30% on edges (high geometric value)
        n_edge = max(1, int(n_pts * 0.3))
        edges = []
        for axis in range(3):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    start = torch.zeros(3)
                    end = torch.zeros(3)
                    other_axes = [a for a in range(3) if a != axis]
                    start[other_axes[0]] = s1 * dims[other_axes[0]]
                    start[other_axes[1]] = s2 * dims[other_axes[1]]
                    end = start.clone()
                    start[axis] = -dims[axis]
                    end[axis] = dims[axis]
                    edges.append((start, end))

        for _ in range(n_edge):
            e = torch.randint(0, len(edges), (1,), generator=self.generator).item()
            t_val = torch.rand(1, generator=self.generator).item()
            pt = edges[e][0] + t_val * (edges[e][1] - edges[e][0])
            pts_list.append(pt)

        # 70% on faces (with slight thickness for realism)
        n_face = n_pts - n_edge
        for _ in range(n_face):
            face_axis = torch.randint(0, 3, (1,), generator=self.generator).item()
            face_sign = (-1) ** torch.randint(0, 2, (1,), generator=self.generator).item()
            pt = torch.zeros(3)
            pt[face_axis] = face_sign * dims[face_axis]
            other = [a for a in range(3) if a != face_axis]
            pt[other[0]] = (torch.rand(1, generator=self.generator).item() * 2 - 1) * dims[other[0]]
            pt[other[1]] = (torch.rand(1, generator=self.generator).item() * 2 - 1) * dims[other[1]]
            pts_list.append(pt)

        pts = torch.stack(pts_list)
        # Add slight surface noise (manufacturing imperfection)
        pts = pts + torch.randn_like(pts) * 0.005
        return pts

    def _sample_lshape(self, n_pts: int) -> Tensor:
        """Sample points on an L-shaped structure.

        An L-shape is maximally asymmetric: it has no rotational symmetry
        whatsoever (unlike a box which has 180° symmetries about each axis).

        Args:
            n_pts: Number of points

        Returns:
            Points on L-shape, shape: [n_pts, 3]
        """
        # L-shape = two joined rectangular blocks
        w = 0.08 + torch.rand(1, generator=self.generator).item() * 0.06
        h1 = 0.2 + torch.rand(1, generator=self.generator).item() * 0.2
        h2 = 0.15 + torch.rand(1, generator=self.generator).item() * 0.15
        l1 = 0.3 + torch.rand(1, generator=self.generator).item() * 0.2

        # Split points between the two arms
        n1 = n_pts // 2
        n2 = n_pts - n1

        # Vertical arm: x ∈ [-w, w], y ∈ [0, h1], z ∈ [-w, w]
        arm1 = torch.zeros(n1, 3)
        arm1[:, 0] = (torch.rand(n1, generator=self.generator) * 2 - 1) * w
        arm1[:, 1] = torch.rand(n1, generator=self.generator) * h1
        arm1[:, 2] = (torch.rand(n1, generator=self.generator) * 2 - 1) * w

        # Horizontal arm: x ∈ [0, l1], y ∈ [-w, w], z ∈ [-w, w]
        arm2 = torch.zeros(n2, 3)
        arm2[:, 0] = torch.rand(n2, generator=self.generator) * l1
        arm2[:, 1] = (torch.rand(n2, generator=self.generator) * 2 - 1) * w
        arm2[:, 2] = (torch.rand(n2, generator=self.generator) * 2 - 1) * w

        pts = torch.cat([arm1, arm2], dim=0)
        pts = pts + torch.randn_like(pts) * 0.005
        return pts

    def _sample_torus(self, n_pts: int) -> Tensor:
        """Sample points on a torus surface.

        A torus has non-trivial curvature that varies smoothly from
        convex (outer ring) to saddle (inner ring). This curvature
        gradient provides strong geometric features.

        Args:
            n_pts: Number of points

        Returns:
            Points on torus, shape: [n_pts, 3]
        """
        R_major = 0.2 + torch.rand(1, generator=self.generator).item() * 0.15
        r_minor = 0.05 + torch.rand(1, generator=self.generator).item() * 0.05

        theta = torch.rand(n_pts, generator=self.generator) * 2 * math.pi
        phi = torch.rand(n_pts, generator=self.generator) * 2 * math.pi

        x = (R_major + r_minor * torch.cos(phi)) * torch.cos(theta)
        y = (R_major + r_minor * torch.cos(phi)) * torch.sin(theta)
        z = r_minor * torch.sin(phi)

        pts = torch.stack([x, y, z], dim=-1)
        pts = pts + torch.randn_like(pts) * 0.005
        return pts

    def _sample_cylinder(self, n_pts: int) -> Tensor:
        """Sample points on a cylinder (curved surface + flat caps).

        The junction between the curved surface and flat caps creates
        a sharp circular edge — excellent geometric feature.

        Args:
            n_pts: Number of points

        Returns:
            Points on cylinder, shape: [n_pts, 3]
        """
        radius = 0.1 + torch.rand(1, generator=self.generator).item() * 0.1
        height = 0.2 + torch.rand(1, generator=self.generator).item() * 0.3

        # 20% on caps (including edge ring), 80% on curved surface
        n_cap = max(2, int(n_pts * 0.2))
        n_body = n_pts - n_cap

        # Curved surface
        theta = torch.rand(n_body, generator=self.generator) * 2 * math.pi
        h = torch.rand(n_body, generator=self.generator) * height - height / 2
        body = torch.stack([
            radius * torch.cos(theta),
            radius * torch.sin(theta),
            h,
        ], dim=-1)

        # Caps (with edge concentration)
        cap_pts = []
        for _ in range(n_cap):
            # 50% chance of being on the edge ring
            if torch.rand(1, generator=self.generator).item() < 0.5:
                # Edge ring
                a = torch.rand(1, generator=self.generator).item() * 2 * math.pi
                z_sign = (-1) ** torch.randint(0, 2, (1,), generator=self.generator).item()
                pt = torch.tensor([radius * math.cos(a), radius * math.sin(a),
                                   z_sign * height / 2])
            else:
                # Face interior
                r = radius * torch.rand(1, generator=self.generator).item() ** 0.5
                a = torch.rand(1, generator=self.generator).item() * 2 * math.pi
                z_sign = (-1) ** torch.randint(0, 2, (1,), generator=self.generator).item()
                pt = torch.tensor([r * math.cos(a), r * math.sin(a),
                                   z_sign * height / 2])
            cap_pts.append(pt)

        caps = torch.stack(cap_pts)
        pts = torch.cat([body, caps], dim=0)
        pts = pts + torch.randn_like(pts) * 0.005
        return pts

    def _sample_cone(self, n_pts: int) -> Tensor:
        """Sample points on a cone (tip + slanted surface).

        The tip is a singular point of maximum curvature — extremely
        distinctive. The base circle provides edge features.

        Args:
            n_pts: Number of points

        Returns:
            Points on cone, shape: [n_pts, 3]
        """
        radius = 0.15 + torch.rand(1, generator=self.generator).item() * 0.1
        height = 0.3 + torch.rand(1, generator=self.generator).item() * 0.2

        # 10% near tip, 20% on base edge, 70% on surface
        n_tip = max(1, int(n_pts * 0.1))
        n_edge = max(1, int(n_pts * 0.2))
        n_surface = n_pts - n_tip - n_edge

        # Slanted surface
        t_vals = torch.rand(n_surface, generator=self.generator)  # 0=base, 1=tip
        theta = torch.rand(n_surface, generator=self.generator) * 2 * math.pi
        r = radius * (1.0 - t_vals)
        surface = torch.stack([
            r * torch.cos(theta),
            r * torch.sin(theta),
            t_vals * height,
        ], dim=-1)

        # Tip cluster
        tip = torch.randn(n_tip, 3) * 0.01
        tip[:, 2] = tip[:, 2].abs() * 0.02 + height

        # Base edge ring
        theta_e = torch.rand(n_edge, generator=self.generator) * 2 * math.pi
        edge = torch.stack([
            radius * torch.cos(theta_e),
            radius * torch.sin(theta_e),
            torch.zeros(n_edge),
        ], dim=-1)

        pts = torch.cat([surface, tip, edge], dim=0)
        pts = pts + torch.randn_like(pts) * 0.005
        return pts

    def _random_se3(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate random SE(3) transform.

        Uses axis-angle → quaternion → rotation matrix.
        All on CPU, FP32, no Euler angles (Rule 1).

        Returns:
            R: Rotation matrix [3, 3]
            t: Translation [3]
            q: Quaternion [4] (wxyz)
        """
        # Random rotation via axis-angle
        max_angle_rad = self.rotation_range * math.pi / 180.0
        axis = torch.randn(3, generator=self.generator)
        axis = axis / axis.norm().clamp(min=1e-8)
        angle = torch.rand(1, generator=self.generator).item() * max_angle_rad

        # Axis-angle → quaternion (no Euler angles!)
        half_angle = angle / 2.0
        qw = math.cos(half_angle)
        qxyz = math.sin(half_angle) * axis
        q = torch.tensor([qw, qxyz[0], qxyz[1], qxyz[2]])
        q = q / q.norm()  # Normalize quaternion

        # Quaternion → rotation matrix
        w, x, y, z = q.unbind()
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
        ])

        # Random translation
        t = torch.randn(3, generator=self.generator) * self.translation_range

        return R, t, q
