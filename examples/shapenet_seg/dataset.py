# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Standalone ShapeNet Part Segmentation dataset loader.

No PyG/torch_geometric dependency (AGENTS.md Rule 7).

Supports two data formats:

1. **HDF5 format** (`shapenetpart_hdf5_2048/`):
   Pre-processed HDF5 files with 2048 points per shape.
   Files: train{0..5}.h5, test{0..1}.h5, val0.h5
   Each H5 contains: data[N,2048,3], normal[N,2048,3],
                      label[N,1], pid[N,2048]

2. **TXT format** (`shapenetcore_partanno_segmentation_benchmark_v0_normal/`):
   Raw text files from Stanford. Downloaded automatically.

Reference:
    Yi et al., "A Scalable Active Framework for Region Annotation
    in 3D Shape Collections", SIGGRAPH Asia 2016.
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple
import zipfile
import urllib.request

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    import h5py  # type: ignore
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# ═══════════════════════════════════════════════════════════════════
# Category → Part label mapping (global labels 0..49)
# ═══════════════════════════════════════════════════════════════════

CATEGORY_IDS: Dict[str, str] = {
    'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340',
    'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776',
    'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649',
    'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390',
    'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987',
    'Table': '04379243',
}

SEG_CLASSES: Dict[str, List[int]] = {
    'Airplane': [0, 1, 2, 3],
    'Bag': [4, 5],
    'Cap': [6, 7],
    'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15],
    'Earphone': [16, 17, 18],
    'Guitar': [19, 20, 21],
    'Knife': [22, 23],
    'Lamp': [24, 25, 26, 27],
    'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35],
    'Mug': [36, 37],
    'Pistol': [38, 39, 40],
    'Rocket': [41, 42, 43],
    'Skateboard': [44, 45, 46],
    'Table': [47, 48, 49],
}

# Inverse: synset_id → category name
SYNSET_TO_CATEGORY: Dict[str, str] = {v: k for k, v in CATEGORY_IDS.items()}

# Category name → index (0..15)
CATEGORY_TO_IDX: Dict[str, int] = {
    name: i for i, name in enumerate(CATEGORY_IDS.keys())
}

# Category index → name
IDX_TO_CATEGORY: Dict[int, str] = {
    i: name for name, i in CATEGORY_TO_IDX.items()
}

NUM_PARTS = 50
NUM_CATEGORIES = 16
NUM_POINTS = 2048

# Category mask: [16, 50] bool — which part labels are valid for each category
CATEGORY_PART_MASK = torch.zeros(NUM_CATEGORIES, NUM_PARTS, dtype=torch.bool)
for _cat_name, _parts in SEG_CLASSES.items():
    _cat_idx = CATEGORY_TO_IDX[_cat_name]
    for _p in _parts:
        CATEGORY_PART_MASK[_cat_idx, _p] = True

DATASET_URL = (
    'https://shapenet.cs.stanford.edu/media/'
    'shapenetcore_partanno_segmentation_benchmark_v0_normal.zip'
)


# ═══════════════════════════════════════════════════════════════════
# HDF5 format loader
# ═══════════════════════════════════════════════════════════════════

def _load_h5_files(
    data_dir: str,
    split: str,
    num_points: int = NUM_POINTS,
    normalize: bool = True,
) -> List[Dict[str, Tensor]]:
    """Load ShapeNet Part data from HDF5 format.

    Expected directory structure:
        shapenetpart_hdf5_2048/
            train{0..N}.h5
            test{0..N}.h5
            val0.h5
            *_id2name.json  (category name mapping)

    Each H5 file contains:
        data:   [M, 2048, 3] float32 — point positions
        normal: [M, 2048, 3] float32 — point normals
        label:  [M, 1] int64 — category label (0-15)
        pid:    [M, 2048] int64 — per-point part label (0-49)

    Args:
        data_dir: Path to shapenetpart_hdf5_2048/
        split: 'train', 'val', 'test', or 'trainval'
        num_points: Subsample to this many points
        normalize: Center and scale to unit sphere

    Returns:
        List of dicts with keys: pos, normal, label, cat_idx
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for HDF5 format. Install: pip install h5py"
        )

    # Find all H5 files for the requested split
    if split == 'trainval':
        h5_patterns = ['train', 'val']
    else:
        h5_patterns = [split]

    h5_files = []
    for pat in h5_patterns:
        files = sorted(glob.glob(osp.join(data_dir, f'{pat}*.h5')))
        h5_files.extend(files)

    if not h5_files:
        raise FileNotFoundError(
            f"No H5 files found for split='{split}' in {data_dir}"
        )

    data_list: List[Dict[str, Tensor]] = []

    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            available_keys = list(f.keys())

            # Auto-detect key names from what's available
            # Position keys: 'data', 'points', 'pos', 'xyz'
            pos_key = None
            for k in ['data', 'points', 'pos', 'xyz']:
                if k in available_keys:
                    pos_key = k
                    break
            if pos_key is None:
                raise KeyError(
                    f"No position key found in {h5_path}. "
                    f"Available keys: {available_keys}"
                )

            # Normal keys: 'normal', 'normals', 'norm'
            normal_key = None
            for k in ['normal', 'normals', 'norm']:
                if k in available_keys:
                    normal_key = k
                    break
            # Normals may not exist — we'll generate zeros

            # Category label keys: 'label', 'category', 'cls', 'cat'
            label_key = None
            for k in ['label', 'category', 'cls', 'cat']:
                if k in available_keys:
                    label_key = k
                    break
            if label_key is None:
                raise KeyError(
                    f"No category label key found in {h5_path}. "
                    f"Available keys: {available_keys}"
                )

            # Part label keys: 'pid', 'seg', 'part', 'seg_label'
            pid_key = None
            for k in ['pid', 'seg', 'part', 'seg_label']:
                if k in available_keys:
                    pid_key = k
                    break
            if pid_key is None:
                raise KeyError(
                    f"No part label key found in {h5_path}. "
                    f"Available keys: {available_keys}"
                )

            positions = np.array(f[pos_key])    # [M, 2048, 3]
            if normal_key is not None:
                normals_np = np.array(f[normal_key])  # [M, 2048, 3]
            else:
                # Generate zero normals (network can still learn from geometry)
                normals_np = np.zeros_like(positions)
            cat_labels = np.array(f[label_key])  # [M, 1] or [M]
            part_labels = np.array(f[pid_key])   # [M, 2048]

        M = positions.shape[0]
        N_raw = positions.shape[1]

        # Flatten category labels
        if cat_labels.ndim == 2:
            cat_labels = cat_labels.squeeze(-1)  # [M]

        for i in range(M):
            pos = torch.from_numpy(positions[i]).float()     # [N_raw, 3]
            normal = torch.from_numpy(normals_np[i]).float()    # [N_raw, 3]
            label = torch.from_numpy(part_labels[i]).long()  # [N_raw]
            cat_idx = int(cat_labels[i])

            # Subsample if needed
            if num_points < N_raw:
                idx = torch.randperm(N_raw)[:num_points]
                pos = pos[idx]
                normal = normal[idx]
                label = label[idx]

            # Normalize to unit sphere
            if normalize:
                center = pos.mean(dim=0)
                pos = pos - center
                scale = pos.norm(dim=-1).max().clamp(min=1e-6)
                pos = pos / scale

            # Normalize normals to unit length
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            data_list.append({
                'pos': pos,
                'normal': normal,
                'label': label,
                'cat_idx': cat_idx,
            })

    return data_list


# ═══════════════════════════════════════════════════════════════════
# TXT format loader (Stanford original)
# ═══════════════════════════════════════════════════════════════════

def _load_txt_files(
    data_dir: str,
    split: str,
    num_points: int = NUM_POINTS,
    normalize: bool = True,
    download: bool = True,
) -> List[Dict[str, Tensor]]:
    """Load ShapeNet Part data from Stanford TXT format.

    Expected directory structure:
        shapenetcore_partanno_segmentation_benchmark_v0_normal/
            02691156/<hash>.txt
            ...
            train_test_split/shuffled_{train,val,test}_file_list.json

    Each .txt file: columns = [x, y, z, nx, ny, nz, label]
    """
    inner_dir = osp.join(
        data_dir,
        'shapenetcore_partanno_segmentation_benchmark_v0_normal',
    )

    if download and not osp.isdir(inner_dir):
        os.makedirs(data_dir, exist_ok=True)
        zip_path = osp.join(data_dir, 'shapenet_part.zip')
        if not osp.exists(zip_path):
            print(f"Downloading ShapeNet Part dataset to {zip_path}...")
            urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)

    # Load file lists
    split_dir = osp.join(inner_dir, 'train_test_split')
    splits = ['train', 'val'] if split == 'trainval' else [split]

    file_list: List[Tuple[str, str]] = []
    for s in splits:
        json_path = osp.join(split_dir, f'shuffled_{s}_file_list.json')
        with open(json_path, 'r') as f:
            flist = json.load(f)
        for entry in flist:
            parts = entry.strip().split('/')
            synset_id = parts[1]
            shape_id = parts[2]
            txt_path = osp.join(inner_dir, synset_id, shape_id + '.txt')
            if osp.exists(txt_path):
                file_list.append((synset_id, txt_path))

    data_list: List[Dict[str, Tensor]] = []
    for synset_id, txt_path in file_list:
        data = np.loadtxt(txt_path).astype(np.float32)
        pos = torch.from_numpy(data[:, :3])
        normal = torch.from_numpy(data[:, 3:6])
        label = torch.from_numpy(data[:, 6]).long()

        cat_name = SYNSET_TO_CATEGORY[synset_id]
        cat_idx = CATEGORY_TO_IDX[cat_name]

        N_raw = pos.shape[0]
        if N_raw >= num_points:
            idx = torch.randperm(N_raw)[:num_points]
        else:
            idx = torch.cat([
                torch.arange(N_raw),
                torch.randint(0, N_raw, (num_points - N_raw,)),
            ])

        pos = pos[idx]
        normal = normal[idx]
        label = label[idx]

        if normalize:
            center = pos.mean(dim=0)
            pos = pos - center
            scale = pos.norm(dim=-1).max().clamp(min=1e-6)
            pos = pos / scale

        normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        data_list.append({
            'pos': pos,
            'normal': normal,
            'label': label,
            'cat_idx': cat_idx,
        })

    return data_list


# ═══════════════════════════════════════════════════════════════════
# Unified dataset class
# ═══════════════════════════════════════════════════════════════════

class ShapeNetPartDataset(Dataset):
    """ShapeNet Part Segmentation Dataset — supports H5 and TXT formats.

    Auto-detects format based on directory contents:
        - If *.h5 files found → HDF5 mode
        - If synsetoffset2category.txt or train_test_split/ found → TXT mode

    Each sample returns:
        pos: [N, 3] float32 — point positions
        normal: [N, 3] float32 — point normals (l=1 vector feature)
        label: [N] int64 — per-point part labels (0-49)
        cat_idx: int — category index (0-15)

    Args:
        root: Root directory (either shapenetpart_hdf5_2048/ or parent of
              shapenetcore_partanno_.../)
        split: 'train', 'val', 'test', or 'trainval'
        num_points: Points per shape (default 2048)
        normalize: Center and scale each shape to unit sphere
        download: Auto-download TXT format if not found
    """

    def __init__(
        self,
        root: str,
        split: str = 'trainval',
        num_points: int = NUM_POINTS,
        normalize: bool = True,
        download: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.num_points = num_points

        # Auto-detect format
        h5_files = glob.glob(osp.join(root, '*.h5'))
        txt_dir = osp.join(
            root,
            'shapenetcore_partanno_segmentation_benchmark_v0_normal',
        )

        if h5_files:
            print(f"Detected HDF5 format in {root}")
            self.data = _load_h5_files(root, split, num_points, normalize)
        elif osp.isdir(txt_dir):
            print(f"Detected TXT format in {root}")
            self.data = _load_txt_files(root, split, num_points, normalize, download=False)
        else:
            # Try TXT with download
            print(f"No existing data found, attempting TXT format download...")
            self.data = _load_txt_files(root, split, num_points, normalize, download=download)

        print(f"Loaded {len(self.data)} shapes ({split})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return self.data[idx]


def collate_fn(
    batch: List[Dict[str, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pack a batch of shapes into ptr-based packed representation.

    Args:
        batch: List of dicts from ShapeNetPartDataset.__getitem__

    Returns:
        pos: [B*N, 3] packed positions
        normals: [B*N, 3] packed normals (l=1 vector features)
        labels: [B*N] packed part labels
        ptr: [B+1] CSR batch offsets, int64
        cat_indices: [B] category indices, int64
    """
    B = len(batch)
    N = batch[0]['pos'].shape[0]  # All shapes have same N

    pos_list = []
    normal_list = []
    label_list = []
    cat_list = []

    for item in batch:
        pos_list.append(item['pos'])
        normal_list.append(item['normal'])
        label_list.append(item['label'])
        cat_list.append(item['cat_idx'])

    pos = torch.cat(pos_list, dim=0)         # [B*N, 3]
    normals = torch.cat(normal_list, dim=0)  # [B*N, 3]
    labels = torch.cat(label_list, dim=0)    # [B*N]
    cat_indices = torch.tensor(cat_list, dtype=torch.int64)  # [B]

    # ptr: [0, N, 2N, ..., B*N]
    ptr = torch.arange(0, (B + 1) * N, N, dtype=torch.int64)

    return pos, normals, labels, ptr, cat_indices
