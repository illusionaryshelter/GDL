#!/usr/bin/env python3
"""Lightweight norm trace for remote GPU — real ShapeNet data.

Design philosophy:
  - Uses REAL ShapeNet data (not synthetic)
  - Uses PRODUCTION model size (96/24/8) to reproduce real issues
  - Limits batches per epoch (default=5, ~12s/epoch → 20 epochs ≈ 4min)
  - Hooks every intermediate tensor to trace where norm growth occurs

Usage (remote):
    python examples/shapenet_seg/trace_norms.py \\
        --data_root ../data/shapenetpart_hdf5_2048 \\
        --epochs 20

    # Even lighter (2 batches, ~5s/epoch → 20 epochs ≈ 100s):
    python examples/shapenet_seg/trace_norms.py \\
        --data_root ../data/shapenetpart_hdf5_2048 \\
        --epochs 20 --batches 2

    # Full epoch comparison (218 batches, ~8min/epoch):
    python examples/shapenet_seg/trace_norms.py \\
        --data_root ../data/shapenetpart_hdf5_2048 \\
        --epochs 60 --batches 0
"""

import argparse
import os
import sys
import time
from collections import defaultdict

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.shapenet_seg.dataset import (
    ShapeNetPartDataset, collate_fn,
)
from examples.shapenet_seg.model import SE3PartSegNet


# ═══════════════════════════════════════════════════════════════════
# Hook infrastructure
# ═══════════════════════════════════════════════════════════════════

class NormTracer:
    """Non-invasive hooks on SE3PartSegNet internals."""

    def __init__(self, model: SE3PartSegNet):
        self.model = model
        self.trace: dict[str, list[float]] = defaultdict(list)
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        bb = self.model.backbone
        for i in range(bb.num_stages):
            for j, blk in enumerate(bb.encoder_stages[i].blocks):
                self._hook_block(i, j, blk)
        for i in range(bb.num_stages - 1):
            self._hook_pool(i, bb.pool_layers[i])
        self._installed = True

    def reset(self) -> None:
        self.trace.clear()

    def means(self) -> dict[str, float]:
        return {k: sum(v) / len(v) if v else 0.0
                for k, v in self.trace.items()}

    # ── Block hook (Post-Norm path: Conv → Norm → Gate → Residual) ──

    def _hook_block(self, si: int, bi: int, blk) -> None:
        tr = self.trace

        def fwd(scalars, vectors, graph, type2=None):
            p = f'e{si}b{bi}'
            has_t2 = type2 is not None and blk.channels_type2 > 0

            tr[f'{p}_in'].append(
                scalars.detach().norm(dim=-1).mean().item())

            # Conv
            if has_t2:
                s, v, t2 = blk.conv(scalars, vectors, graph, type2=type2)
            else:
                s, v = blk.conv(scalars, vectors, graph)
                t2 = None
            tr[f'{p}_conv'].append(s.detach().norm(dim=-1).mean().item())

            # Norm (Post-Norm: normalize conv output)
            if t2 is not None:
                s, v, t2 = blk.norm(s, v, t2)
            else:
                s, v = blk.norm(s, v)

            # Gate
            if t2 is not None:
                s, v, t2 = blk.gate(s, v, t2)
            else:
                s, v = blk.gate(s, v)
            tr[f'{p}_gate'].append(s.detach().norm(dim=-1).mean().item())

            # Self-TP
            if blk.self_tp_proj is not None:
                tp_in = []
                if blk.channels_vector > 0:
                    tp_in.append((v * v).sum(dim=-1))
                if t2 is not None and blk.channels_type2 > 0:
                    tp_in.append((t2 * t2).sum(dim=-1))
                if tp_in:
                    s = s + blk.self_tp_proj(
                        torch.cat(tp_in, -1)) * blk.self_tp_scale

            # Learnable skip residual
            if blk.use_residual:
                s = s + scalars * blk.skip_s_scale
                if blk.skip_v_scale is not None:
                    v = v + vectors * blk.skip_v_scale.unsqueeze(-1)
                if (t2 is not None and type2 is not None
                        and blk.skip_t2_scale is not None):
                    t2 = t2 + type2 * blk.skip_t2_scale.unsqueeze(-1)
            tr[f'{p}_out'].append(s.detach().norm(dim=-1).mean().item())

            if has_t2 and t2 is not None:
                return s, v, t2
            return s, v

        blk.forward = fwd

    def _hook_pool(self, idx: int, pool) -> None:
        orig_fwd = pool.forward
        tr = self.trace

        def fwd(*a, **kw):
            tr[f'pool{idx}_in'].append(
                a[1].detach().norm(dim=-1).mean().item())
            r = orig_fwd(*a, **kw)
            tr[f'pool{idx}_out'].append(
                r[1].detach().norm(dim=-1).mean().item())
            return r
        pool.forward = fwd


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Lightweight norm trace — real ShapeNet data')
    p.add_argument('--data_root', type=str, required=True,
                   help='Path to shapenetpart_hdf5_2048')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batches', type=int, default=5,
                   help='Batches per epoch (0=full epoch)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')
    if dev.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # ── Data ──
    ds = ShapeNetPartDataset(args.data_root, split='trainval',
                             normalize=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_fn, num_workers=args.workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=(args.workers > 0))
    total_batches = len(dl)
    use_batches = total_batches if args.batches == 0 else min(
        args.batches, total_batches)
    print(f'Data: {len(ds)} shapes, {total_batches} batches/epoch, '
          f'using {use_batches}')

    # ── Model (production size) ──
    model = SE3PartSegNet(
        in_channels=1,
        hidden_scalar=96, hidden_vector=24, hidden_type2=8,
        num_stages=3, layers_per_stage=2, pool_ratio=0.25,
        head_hidden=256, use_normals=True, gate_mode='norm',
        use_self_tp=True, use_bottleneck_attn=True,
    ).to(dev)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ── Hooks ──
    tracer = NormTracer(model)
    tracer.install()

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05)

    # ── Columns ──
    cols = [
        ('e0b0_in',   'e0b0_in'),
        ('e0b0_conv', 'e0b0_conv'),
        ('e0b0_gate', 'e0b0_gate'),
        ('e0b1_out',  'e0b1_out'),
        ('pool0',     'pool0_out'),
        ('e1b1_out',  'e1b1_out'),
        ('pool1',     'pool1_out'),
        ('e2b1_out',  'e2b1_out'),
    ]
    names = [c[0] for c in cols]

    est_time = use_batches * 2.5  # ~2.5s per batch rough estimate
    print(f'\nEstimated: ~{est_time:.0f}s/epoch × {args.epochs} '
          f'= ~{est_time * args.epochs / 60:.1f}min total')
    print()
    print('=' * 120)
    hdr = f'{"ep":>3} |'
    for n in names:
        hdr += f'{n:>10}'
    hdr += ' | loss    acc   s2/s0  time'
    print(hdr)
    print('-' * len(hdr))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Eval trace (no grad, limited batches) ──
        tracer.reset()
        model.eval()
        with torch.no_grad():
            for i, (pos, normals, labels, ptr, cat_idx) in enumerate(dl):
                if i >= use_batches:
                    break
                pos = pos.to(dev)
                normals = normals.to(dev)
                ptr = ptr.to(dev)
                cat_idx = cat_idx.to(dev)
                _ = model(pos, ptr, cat_idx, normals)
        m = tracer.get_means() if hasattr(tracer, 'get_means') else tracer.means()

        # ── Train (same limited batches) ──
        model.train()
        t_loss = 0.0; t_cor = 0; t_pts = 0; nb = 0
        for i, (pos, normals, labels, ptr, cat_idx) in enumerate(dl):
            if i >= use_batches:
                break
            pos = pos.to(dev, non_blocking=True)
            normals = normals.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            ptr = ptr.to(dev, non_blocking=True)
            cat_idx = cat_idx.to(dev, non_blocking=True)

            logits = model(pos, ptr, cat_idx, normals)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item(); nb += 1
            t_cor += (logits.detach().argmax(1) == labels).sum().item()
            t_pts += labels.shape[0]

        avg_loss = t_loss / max(nb, 1)
        avg_acc = t_cor / max(t_pts, 1)
        dt = time.time() - t0

        vals = [m.get(k, 0.) for _, k in cols]
        s0 = m.get('e0b1_out', 1)
        s2 = m.get('e2b1_out', 1)
        ratio = s2 / max(s0, 1e-6)

        row = f'{epoch:3d} |'
        for v in vals:
            row += f'{v:10.3f}'
        row += f' | {avg_loss:.3f} {avg_acc:.3f} {ratio:.3f}  {dt:.0f}s'

        # ── Skip scale diagnostics ──
        skip_parts = []
        bb = model.backbone
        for si in range(bb.num_stages):
            for bi, blk in enumerate(bb.encoder_stages[si].blocks):
                d = blk.get_skip_diagnostics()
                if d:
                    tag = f'e{si}b{bi}'
                    skip_parts.append(
                        f'{tag}[s={d["skip_s_mean"]:.3f}'
                        f',v={d.get("skip_v_mean", 0):.3f}'
                        f',t2={d.get("skip_t2_mean", 0):.3f}]'
                    )
        if skip_parts:
            row += '\n  ├─ skip: ' + ' '.join(skip_parts)

        print(row)

    print()
    print('=' * 120)
    print('KEY:  s2/s0 < 1.5 = healthy.  Growing = s2 norm explosion.')
    print('      e*_conv = raw conv output after degree norm.')
    print('      e*_gate = after LayerNorm + gate (should be ~5-10).')
    print('      skip: learnable skip scale (init=0.707, smaller=less residual).')
    print('=' * 120)


if __name__ == '__main__':
    main()
