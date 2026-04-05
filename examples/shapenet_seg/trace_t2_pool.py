#!/usr/bin/env python3
"""Fast training trace: t2 pooling cancellation + augmentation effect.

Like trace_norms.py but focused on:
  1. s/v/t2 norms at every pool layer (cancellation ratio)
  2. Projection weight norms (t2_pw, v_pw) over training
  3. head_t2_ratio evolution
  4. Comparison: augment ON vs OFF (run twice)

Uses REAL ShapeNet data, PRODUCTION model size, limited batches.

Usage (remote):
    # Default: 5 batches/epoch, 20 epochs, augmentation ON
    python examples/shapenet_seg/trace_t2_pool.py \\
        --data_root ../data/shapenetpart_hdf5_2048

    # Without augmentation (compare with above):
    python examples/shapenet_seg/trace_t2_pool.py \\
        --data_root ../data/shapenetpart_hdf5_2048 --no_augment

    # Lighter run:
    python examples/shapenet_seg/trace_t2_pool.py \\
        --data_root ../data/shapenetpart_hdf5_2048 \\
        --epochs 10 --batches 2
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from examples.shapenet_seg.dataset import (
    ShapeNetPartDataset, collate_fn,
)
from examples.shapenet_seg.model import SE3PartSegNet


# ═══════════════════════════════════════════════════════════════════
# Hook infrastructure: traces s/v/t2 at blocks AND pools
# ═══════════════════════════════════════════════════════════════════

class PoolCancelTracer:
    """Hooks every block and pool to trace s/v/t2 norms during training.

    Key difference from trace_norms.py NormTracer:
      - Tracks v and t2 norms, not just scalar s
      - Records pool pre/post norms to compute cancellation %
      - Tracks projection weight norms and head contribution ratio
    """

    def __init__(self, model: SE3PartSegNet):
        self.model = model
        self.trace: Dict[str, List[float]] = defaultdict(list)
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        bb = self.model.backbone
        for si in range(bb.num_stages):
            for bi, blk in enumerate(bb.encoder_stages[si].blocks):
                self._hook_block(si, bi, blk)
        for pi in range(bb.num_stages - 1):
            self._hook_pool(pi, bb.pool_layers[pi])
        self._installed = True

    def reset(self) -> None:
        self.trace.clear()

    def means(self) -> Dict[str, float]:
        return {k: sum(v) / len(v) if v else 0.0
                for k, v in self.trace.items()}

    # ── Block hook: captures s/v/t2 after output norm ──

    def _hook_block(self, si: int, bi: int, blk) -> None:
        tr = self.trace

        def fwd(scalars, vectors, graph, type2=None):
            has_t2 = type2 is not None and blk.channels_type2 > 0
            p = f'e{si}b{bi}'

            # Conv
            if has_t2:
                s, v, t2 = blk.conv(scalars, vectors, graph, type2=type2)
            else:
                s, v = blk.conv(scalars, vectors, graph)
                t2 = None

            # Inner norm
            if t2 is not None:
                s, v, t2 = blk.norm(s, v, t2)
            else:
                s, v = blk.norm(s, v)

            # Gate
            if t2 is not None:
                s, v, t2 = blk.gate(s, v, t2)
            else:
                s, v = blk.gate(s, v)

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

            # Skip residual
            if blk.use_residual:
                s = s + scalars * blk.skip_s_scale
                if blk.skip_v_scale is not None:
                    v = v + vectors * blk.skip_v_scale.unsqueeze(-1)
                if (t2 is not None and type2 is not None
                        and blk.skip_t2_scale is not None):
                    t2 = t2 + type2 * blk.skip_t2_scale.unsqueeze(-1)

            # Output norm
            if t2 is not None:
                s, v, t2 = blk.output_norm(s, v, t2)
            else:
                s, v = blk.output_norm(s, v)

            # Record norms
            tr[f'{p}_s'].append(s.detach().norm(dim=-1).mean().item())
            tr[f'{p}_v'].append(v.detach().norm(dim=-1).mean().item())
            if t2 is not None:
                tr[f'{p}_t2'].append(
                    t2.detach().norm(dim=-1).mean().item())

            if has_t2 and t2 is not None:
                return s, v, t2
            return s, v

        blk.forward = fwd

    # ── Pool hook: captures pre/post to compute cancellation ──

    def _hook_pool(self, idx: int, pool) -> None:
        orig_fwd = pool.forward
        tr = self.trace

        def fwd(*a, **kw):
            # Pre-pool norms
            s_in, v_in = a[1], a[2]
            t2_in = kw.get('type2', None)
            tr[f'p{idx}_s_pre'].append(
                s_in.detach().norm(dim=-1).mean().item())
            tr[f'p{idx}_v_pre'].append(
                v_in.detach().norm(dim=-1).mean().item())
            if t2_in is not None:
                tr[f'p{idx}_t2_pre'].append(
                    t2_in.detach().norm(dim=-1).mean().item())

            result = orig_fwd(*a, **kw)

            # Post-pool norms
            s_out, v_out = result[1], result[2]
            tr[f'p{idx}_s_post'].append(
                s_out.detach().norm(dim=-1).mean().item())
            tr[f'p{idx}_v_post'].append(
                v_out.detach().norm(dim=-1).mean().item())
            if len(result) >= 6 and result[3] is not None:
                tr[f'p{idx}_t2_post'].append(
                    result[3].detach().norm(dim=-1).mean().item())

            return result

        pool.forward = fwd


def get_weight_diag(model: SE3PartSegNet) -> Dict[str, float]:
    """Snapshot of projection weight norms (no grad needed)."""
    d: Dict[str, float] = {}
    if hasattr(model, 't2_inv_proj'):
        d['t2_pw'] = model.t2_inv_proj.weight.norm().item()
    if hasattr(model, 'v_inv_proj'):
        d['v_pw'] = model.v_inv_proj.weight.norm().item()
    return d


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Fast training trace — t2 pooling cancellation')
    p.add_argument('--data_root', type=str, required=True,
                   help='Path to shapenetpart_hdf5_2048')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batches', type=int, default=5,
                   help='Batches per epoch (0=full epoch)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--no_augment', action='store_true',
                   help='Disable data augmentation')
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')
    if dev.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # ── Data ──
    augment = not args.no_augment
    ds = ShapeNetPartDataset(
        args.data_root, split='trainval', normalize=True,
        augment=augment,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_fn, num_workers=args.workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=(args.workers > 0))
    total_batches = len(dl)
    use_batches = total_batches if args.batches == 0 else min(
        args.batches, total_batches)
    print(f'Data: {len(ds)} shapes, {total_batches} batches/epoch, '
          f'using {use_batches}')
    print(f'Augmentation: {"ON (jitter+scale)" if augment else "OFF"}')

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
    tracer = PoolCancelTracer(model)
    tracer.install()

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Header ──
    est_time = use_batches * 2.5
    print(f'\nEstimated: ~{est_time:.0f}s/epoch × {args.epochs} '
          f'= ~{est_time * args.epochs / 60:.1f}min total')
    print()
    print('=' * 160)
    # Two-line header for readability
    h1 = (f'{"ep":>3} │ {"loss":>6} {"acc":>5} │'
          f' {"e0b1_s":>7} {"e0b1_v":>7} {"e0b1_t2":>7} │'
          f' {"p0_Δs%":>6} {"p0_Δv%":>6} {"p0_Δt2%":>7} │'
          f' {"e2b1_s":>7} {"e2b1_t2":>7} │'
          f' {"p1_Δs%":>6} {"p1_Δv%":>6} {"p1_Δt2%":>7} │'
          f' {"t2_pw":>6} {"v_pw":>6} │ {"time":>4}')
    print(h1)
    print('-' * 160)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Eval trace (no grad) ──
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
        m = tracer.means()

        # ── Train ──
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

        # ── Compute cancellation % ──
        def drop_pct(pre_key: str, post_key: str) -> str:
            pre = m.get(pre_key, 0)
            post = m.get(post_key, 0)
            if pre < 1e-6:
                return '   N/A'
            pct = (post - pre) / pre * 100
            return f'{pct:+5.1f}%'

        # Weight diagnostics
        wd = get_weight_diag(model)

        # ── Format row ──
        row = (
            f'{epoch:3d} │'
            f' {avg_loss:6.3f} {avg_acc:5.3f} │'
            # Stage 0 output
            f' {m.get("e0b1_s", 0):7.3f}'
            f' {m.get("e0b1_v", 0):7.4f}'
            f' {m.get("e0b1_t2", 0):7.4f} │'
            # Pool 0 drop
            f' {drop_pct("p0_s_pre", "p0_s_post"):>6}'
            f' {drop_pct("p0_v_pre", "p0_v_post"):>6}'
            f' {drop_pct("p0_t2_pre", "p0_t2_post"):>7} │'
            # Stage 2 output
            f' {m.get("e2b1_s", 0):7.3f}'
            f' {m.get("e2b1_t2", 0):7.4f} │'
            # Pool 1 drop
            f' {drop_pct("p1_s_pre", "p1_s_post"):>6}'
            f' {drop_pct("p1_v_pre", "p1_v_post"):>6}'
            f' {drop_pct("p1_t2_pre", "p1_t2_post"):>7} │'
            # Projection weights
            f' {wd.get("t2_pw", 0):6.3f}'
            f' {wd.get("v_pw", 0):6.3f} │'
            f' {dt:4.0f}s'
        )
        print(row)

    # ── Summary ──
    print('=' * 160)
    print()
    print('KEY:')
    print('  e*b*_s/v/t2 = block output norms (scalar/vector/type-2)')
    print('  p*_Δ*%      = pool cancellation (negative=norm lost)')
    print('  t2_pw/v_pw  = t2_inv_proj / v_inv_proj weight norms')
    print()

    # Final cancellation report
    print('FINAL POOL CANCELLATION:')
    for pi in range(2):
        s_pre = m.get(f'p{pi}_s_pre', 0)
        s_post = m.get(f'p{pi}_s_post', 0)
        v_pre = m.get(f'p{pi}_v_pre', 0)
        v_post = m.get(f'p{pi}_v_post', 0)
        t2_pre = m.get(f'p{pi}_t2_pre', 0)
        t2_post = m.get(f'p{pi}_t2_post', 0)

        def pct(a: float, b: float) -> str:
            if b < 1e-6:
                return 'N/A'
            return f'{(a - b) / b * 100:+.1f}%'

        print(f'  Pool{pi}: s {s_pre:.3f}→{s_post:.3f} ({pct(s_post, s_pre)})  '
              f'v {v_pre:.4f}→{v_post:.4f} ({pct(v_post, v_pre)})  '
              f't2 {t2_pre:.4f}→{t2_post:.4f} ({pct(t2_post, t2_pre)})')

    print()
    print('=' * 160)


if __name__ == '__main__':
    main()
