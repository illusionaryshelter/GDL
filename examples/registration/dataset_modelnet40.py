# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""ModelNet40 registration dataset — loads preprocessed .pt cache.

Requires running preprocess_modelnet40.py first:
    python examples/registration/preprocess_modelnet40.py \\
        --root data/ModelNet40 --n_points 4096 --output data/modelnet40_4096.pt

The preprocessed file contains:
    points:  [M, 4096, 3]  — uniformly sampled surface points (unit sphere)
    normals: [M, 4096, 3]  — per-point face normals
    labels:  [M]           — category labels
    splits:  [M]           — 0=train, 1=test

Critical design decisions in __getitem__:
    - **Spherical/KNN cropping** (NOT half-space): preserves topological
      continuity of the surface. Half-space can fragment symmetric objects
      (e.g. leaving only two diagonal table legs).
    - **Strict inlier threshold** for GT correspondences: after warping the
      source into the target frame, nearest-neighbor matches with distance
      > inlier_threshold are marked as outliers (correspondences=-1) and
      routed to the Sinkhorn dustbin.
    - **Symmetric category exclusion**: objects with rotational ambiguity
      (bottles, bowls, cups, cones, etc.) are excluded by default to prevent
      the correspondence loss from penalizing physically valid alternative
      alignments.

No dependency on trimesh, scipy, open3d, or h5py (Rule 7).
"""

import math
import os
from typing import Tuple, Optional, List

import torch
from torch import Tensor
from torch.utils.data import Dataset


# ┌─────────────────────────────────────────────────────────────────┐
# │ SYMMETRIC CATEGORIES TO EXCLUDE                                 │
# │ Objects with continuous rotational symmetry (azimuthal ~Z axis) │
# │ or high discrete symmetry. Include them only AFTER asymmetric   │
# │ training is stable, and then use pure CD loss (no R/t loss).    │
# └─────────────────────────────────────────────────────────────────┘
SYMMETRIC_CATEGORIES = [
    'bottle', 'bowl', 'cone', 'cup', 'flower_pot',
    'lamp', 'tent', 'vase',
]


class ModelNet40Registration(Dataset):
    """ModelNet40 registration dataset from preprocessed .pt cache.

    Each __getitem__ call:
        1. Loads a pre-sampled, pre-normalized shape (instant)
        2. Randomly subsamples from N_dense → N points (torch.randperm)
        3. Applies random SE(3) transform
        4. Spherical/KNN crop for partial overlap (topologically coherent)
        5. Computes GT correspondences with strict inlier threshold

    Args:
        cache_path: Path to preprocessed .pt file
        num_points: Points per cloud (subsampled from N_dense)
        split: 'train' or 'test'
        rotation_range: Maximum rotation angle in degrees
        translation_range: Maximum translation magnitude
        noise_std: Gaussian noise on target cloud
        partial_ratio: Source visibility fraction (1.0 = full)
        inlier_threshold: Max distance for a warp→NN match to be GT.
            Points farther away are marked as outliers (→ dustbin).
            Default 0.05 (5% of unit sphere radius).
        exclude_symmetric: If True, exclude categories with rotational
            symmetry that cause correspondence loss ambiguity.
        categories: Optional explicit category filter.
            Takes precedence over exclude_symmetric.
    """

    def __init__(
        self,
        cache_path: str = 'data/modelnet40_4096.pt',
        num_points: int = 1024,
        split: str = 'train',
        rotation_range: float = 180.0,
        translation_range: float = 0.5,
        noise_std: float = 0.01,
        partial_ratio: float = 1.0,
        inlier_threshold: float = 0.05,
        exclude_symmetric: bool = True,
        categories: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.partial_ratio = partial_ratio
        self.inlier_threshold = inlier_threshold

        if not os.path.isfile(cache_path):
            raise FileNotFoundError(
                f"Preprocessed cache not found: {cache_path}\n"
                f"Run first:\n"
                f"  python examples/registration/preprocess_modelnet40.py \\\n"
                f"      --root data/ModelNet40 --output {cache_path}"
            )

        data = torch.load(cache_path, weights_only=False)
        all_points = data['points']    # [M, N_dense, 3]
        all_normals = data['normals']  # [M, N_dense, 3]
        all_labels = data['labels']    # [M]
        all_splits = data['splits']    # [M]
        self.category_names = data['categories']  # list[str]

        # Filter by split
        split_id = 0 if split == 'train' else 1
        split_mask = all_splits == split_id

        # Category filtering: explicit > exclude_symmetric > all
        if categories is not None:
            cat_ids = [self.category_names.index(c) for c in categories]
            cat_mask = torch.zeros(len(all_labels), dtype=torch.bool)
            for cid in cat_ids:
                cat_mask |= (all_labels == cid)
            split_mask = split_mask & cat_mask
            filter_desc = f"categories={categories}"
        elif exclude_symmetric:
            # Exclude symmetric categories
            sym_ids = []
            for name in SYMMETRIC_CATEGORIES:
                if name in self.category_names:
                    sym_ids.append(self.category_names.index(name))
            if sym_ids:
                sym_mask = torch.zeros(len(all_labels), dtype=torch.bool)
                for sid in sym_ids:
                    sym_mask |= (all_labels == sid)
                split_mask = split_mask & ~sym_mask
            filter_desc = f"exclude={SYMMETRIC_CATEGORIES}"
        else:
            filter_desc = "all categories"

        self.points = all_points[split_mask]    # [M', N_dense, 3]
        self.normals = all_normals[split_mask]  # [M', N_dense, 3]
        self.labels = all_labels[split_mask]    # [M']
        self.n_dense = self.points.shape[1]

        n_cats = len(set(self.labels.tolist()))
        print(
            f"  ModelNet40 [{split}]: {len(self.points)} shapes, "
            f"{self.n_dense}→{num_points} pts "
            f"({n_cats} categories, {filter_desc})"
        )
        if self.partial_ratio < 1.0:
            print(
                f"    Partial overlap: ratio={partial_ratio:.1%}, "
                f"crop=spherical/KNN, "
                f"inlier_thr={inlier_threshold}"
            )

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> dict:
        """Generate a registration pair from preprocessed shape.

        Uses **bilateral spherical cropping** — both source and target
        are independently cropped to partial_ratio fraction. This follows
        the standard ModelNet40 benchmark (RPM-Net, HarSoNet) where
        effective overlap ≈ partial_ratio² (e.g. 0.7² = 49%).

        Sequence:
            1. Subsample N_dense → N (random permutation)
            2. Generate random SE(3) transform
            3. Source = spherical crop of original cloud
            4. Target = INDEPENDENT spherical crop of transformed cloud
            5. Add noise to target
            6. GT correspondences via warp + NN + strict threshold
               Source points without valid match → -1 (dustbin)

        Returns:
            dict with keys:
                'source': [N_src, 3] source point cloud
                'target': [N_tgt, 3] target point cloud
                'gt_rotation': [3, 3] rotation matrix, representation: SO(3)
                'gt_translation': [3] translation vector
                'gt_quat': [4] quaternion (wxyz)
                'correspondences': [N_src] index mapping src[i] → tgt[j],
                    -1 for outliers (routed to dustbin)
                'label': int category label
        """
        N = self.num_points

        # ─── 1. Random subsample N_dense → N ───
        perm = torch.randperm(self.n_dense)[:N]
        cloud = self.points[idx, perm]  # [N, 3]

        # ─── 2. Random SE(3) transform ───
        R, t, q = self._random_se3()

        # ─── 3. Bilateral spherical crop (BOTH source and target) ───
        # This follows the standard ModelNet40 benchmark protocol
        # (RPM-Net, HarSoNet): each side independently sees partial_ratio
        # fraction. Overlap ≈ partial_ratio², so 0.7² = 49% real overlap.
        # Source points outside target's crop have NO match → dustbin.
        if self.partial_ratio < 1.0:
            n_visible = max(1, int(N * self.partial_ratio))

            # Source = spherical crop of cloud (in original frame)
            source, src_idx = self._spherical_crop(cloud, n_visible)

            # Target = INDEPENDENT spherical crop of transformed cloud
            cloud_transformed = (R @ cloud.T).T + t.unsqueeze(0)  # [N, 3]
            target, tgt_idx = self._spherical_crop(
                cloud_transformed, n_visible,
            )
        else:
            source = cloud
            src_idx = torch.arange(N)
            target = (R @ cloud.T).T + t.unsqueeze(0)
            tgt_idx = torch.arange(N)

        # ─── 4. Add noise to target ───
        if self.noise_std > 0:
            target = target + torch.randn_like(target) * self.noise_std

        # ─── 5. GT correspondences with strict inlier threshold ───
        # Because both sides are cropped independently, many source points
        # will have NO valid match in target → correspondences = -1.
        correspondences = self._compute_correspondences(
            source, target, R, t,
        )

        return {
            'source': source,
            'target': target,
            'gt_rotation': R,
            'gt_translation': t,
            'gt_quat': q,
            'correspondences': correspondences,
            'label': int(self.labels[idx].item()),
        }

    def _spherical_crop(
        self,
        cloud: Tensor,
        n_visible: int,
    ) -> Tuple[Tensor, Tensor]:
        """Spherical/KNN crop: pick a random seed, keep K nearest points.

        This is critical (陷阱 A): half-space cropping can fragment
        symmetric objects into disconnected pieces (e.g. two diagonal
        table legs without the tabletop). Spherical cropping guarantees
        the resulting point cloud is a single connected surface patch.

        Args:
            cloud: Full point cloud [N, 3]
            n_visible: Number of points to keep

        Returns:
            cropped: [n_visible, 3] topologically coherent subset
            indices: [n_visible] original indices in cloud
        """
        N = cloud.shape[0]

        # Pick a random seed point on the surface
        seed_idx = torch.randint(N, (1,)).item()
        seed_point = cloud[seed_idx]  # [3]

        # Compute distances from seed to all points
        dists = (cloud - seed_point.unsqueeze(0)).norm(dim=-1)  # [N]

        # Keep the n_visible closest points (KNN crop)
        _, sorted_idx = dists.sort()
        keep_idx = sorted_idx[:n_visible]

        return cloud[keep_idx], keep_idx

    def _compute_correspondences(
        self,
        source: Tensor,
        target: Tensor,
        R: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute GT correspondences with strict inlier threshold.

        With bilateral spherical cropping (陷阱 B), source and target are
        independently cropped. Many source points will have no close match
        in target. We:
            1. Warp source into target frame: src_warped = R @ src + t
            2. Find nearest neighbor in target for each warped point
            3. Apply strict distance threshold
            4. Points exceeding threshold → correspondences = -1 → dustbin

        The inlier_threshold is crucial: too loose (e.g. 0.3) lets
        through bad matches that poison the correspondence loss. Too
        strict (e.g. 0.001) marks everything as outlier, starving the
        loss of supervision signal. Default 0.05 is ~5% of unit sphere.

        Args:
            source: Source cloud [N_src, 3] (in original frame)
            target: Target cloud [N_tgt, 3] (in transformed frame,
                may have noise)
            R: GT rotation [3, 3], representation: SO(3)
            t: GT translation [3]

        Returns:
            correspondences: [N_src] index in target, -1 for outliers
        """
        # Warp source into target frame
        src_warped = (R @ source.T).T + t.unsqueeze(0)  # [N_src, 3]

        # Find nearest neighbor in target
        # cdist: [N_src, N_tgt]
        dists = torch.cdist(src_warped, target)  # [N_src, N_tgt]
        nn_dists, nn_idx = dists.min(dim=-1)  # [N_src], [N_src]

        # Strict inlier threshold (陷阱 B)
        correspondences = nn_idx.clone()  # [N_src]
        outlier_mask = nn_dists > self.inlier_threshold
        correspondences[outlier_mask] = -1  # → route to dustbin

        return correspondences

    def _random_se3(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate random SE(3) transform.

        Uses axis-angle → quaternion → rotation matrix.
        No Euler angles (Rule 1). No scipy (Rule 7).

        Returns:
            R: Rotation matrix [3, 3], representation: SO(3)
            t: Translation [3]
            q: Quaternion [4] (wxyz)
        """
        max_angle = self.rotation_range * math.pi / 180.0
        axis = torch.randn(3)
        axis = axis / axis.norm().clamp(min=1e-8)
        angle = torch.rand(1).item() * max_angle

        half = angle / 2.0
        qw = math.cos(half)
        qxyz = math.sin(half) * axis
        q = torch.tensor([qw, qxyz[0], qxyz[1], qxyz[2]], dtype=torch.float32)
        q = q / q.norm()

        w, x, y, z = q.unbind()
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
        ])

        t = torch.randn(3) * self.translation_range
        return R, t, q
