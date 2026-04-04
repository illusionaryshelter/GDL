#!/usr/bin/env python3
"""Quick remote validation: run 3 mini-batches on real ShapeNet data.

Prints norm diagnostics without doing a full epoch.
Run from the GDL project root:

    python examples/shapenet_seg/quick_validate.py \
        --data_root ../data/shapenetpart_hdf5_2048

Expected output with degree norm fix:
    enc:[s0≈6-8, s1≈6-8, s2≈6-8]  ← balanced (was s2=15+ without fix)
    t2_inv ≈ 6.9                    ← healthy (was decaying to 3.9)
    pool_T < 1.0                    ← attention sharpening (was > 1.0)
"""

import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# GDL imports
from examples.shapenet_seg.dataset import (
    ShapeNetPartDataset, collate_fn, SEG_CLASSES, NUM_PARTS, NUM_CATEGORIES,
)
from examples.shapenet_seg.model import SE3PartSegNet


def main():
    parser = argparse.ArgumentParser(description='Quick norm validation')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_batches', type=int, default=3,
                        help='Number of train batches to run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ──
    train_dataset = ShapeNetPartDataset(
        args.data_root, split='trainval', normalize=True, train=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True,
    )

    # ── Category-part mask ──
    cat_mask = torch.zeros(NUM_CATEGORIES, NUM_PARTS, dtype=torch.bool)
    for cat_name, part_ids in SEG_CLASSES.items():
        from examples.shapenet_seg.dataset import CATEGORY_TO_IDX
        cat_idx = CATEGORY_TO_IDX[cat_name]
        for pid in part_ids:
            cat_mask[cat_idx, pid] = True

    # ── Model ──
    model = SE3PartSegNet(
        in_channels=1,
        hidden_scalar=96, hidden_vector=24, hidden_type2=8,
        num_stages=3, layers_per_stage=2, pool_ratio=0.25,
        num_parts=NUM_PARTS, num_categories=NUM_CATEGORIES,
        head_hidden=256, use_normals=True, gate_mode='norm',
        category_part_mask=cat_mask,
        use_self_tp=True, use_bottleneck_attn=True,
        normal_drop_rate=0.3,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f'\n{"="*60}')
    print(f'  Quick Validation: {args.num_batches} batches on REAL data')
    print(f'{"="*60}\n')

    for i, batch in enumerate(train_loader):
        if i >= args.num_batches:
            break

        pos = batch['pos'].to(device)
        normals_data = batch.get('normals')
        if normals_data is not None:
            normals_data = normals_data.to(device)
        labels = batch['labels'].to(device)
        cat_ids = batch['cat_id'].to(device)
        ptr = batch['ptr'].to(device)

        N = pos.shape[0]
        t0 = time.time()

        # ── Diagnostics (eval mode, no grad) ──
        model.eval()
        with torch.no_grad():
            _, diag = model.forward_with_diagnostics(pos, ptr, cat_ids, normals_data)
            pool_stats = model.get_pool_attn_stats()

        # ── Train step (to verify backward works) ──
        model.train()
        logits = model(pos, ptr, cat_ids, normals_data)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = time.time() - t0

        # ── Print diagnostics ──
        num_stages = 3
        enc_str = ','.join(f's{j}={diag[f"enc{j}_s_norm"]:.2f}' for j in range(num_stages))
        s_norms = [diag[f'enc{j}_s_norm'] for j in range(num_stages)]
        ratio = max(s_norms) / max(min(s_norms), 1e-6)
        status = '✅' if ratio < 2.0 else '🔴'

        print(f'Batch {i} | loss={loss.item():.4f} | {N} pts | {dt:.1f}s')
        print(f'  enc:[{enc_str}] ratio={ratio:.2f} {status}')
        print(f'  v_norm={diag.get("v_norm_mean", 0):.4f}±{diag.get("v_norm_std", 0):.4f}'
              f' | t2={diag.get("t2_norm_mean", 0):.4f}±{diag.get("t2_norm_std", 0):.4f}')
        print(f'  v_inv={diag.get("v_inv_norm", 0):.4f} | t2_inv={diag.get("t2_inv_norm", 0):.4f}')

        T0 = pool_stats.get('pool_T0', 'N/A')
        T1 = pool_stats.get('pool_T1', 'N/A')
        u0 = pool_stats.get('pool_u0', 'N/A')
        u1 = pool_stats.get('pool_u1', 'N/A')
        print(f'  pool: T0={T0:.3f} T1={T1:.3f} u0={u0:.2f} u1={u1:.2f}')
        print()

    print(f'{"="*60}')
    print(f'VALIDATION KEY:')
    print(f'  enc ratio < 2.0  → degree norm working')
    print(f'  t2_inv ≈ 6.9     → type-2 features alive')
    print(f'  pool_T < 1.0     → attention sharpening (not smoothing)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
