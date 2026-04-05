#!/usr/bin/env python3
"""Trace s/v/t2 norms through every layer — verify pooling cancellation.

Hooks every encoder block and pool layer to trace per-type feature norms.
Verifies the hypothesis: attention-weighted pooling causes type-2 (and
vector) norm collapse via 5D/3D component cancellation, while scalars
are preserved.

Outputs a table like:

  Layer       |   s_norm |   v_norm |  t2_norm | v_drop% | t2_drop%
  ────────────+──────────+──────────+──────────+---------+---------
  Enc0b0      |    9.71  |    2.48  |    1.08  |         |
  Enc0b1      |    8.08  |    2.11  |    1.32  |         |
  Pool0       |    7.67  |    0.56  |    0.47  |  -73.5% |  -64.4%
  ...

Also reports:
  - t2_inv_proj / v_inv_proj weight norms (weight decay diagnostic)
  - Head column norms per feature group
  - Per-channel t2 norms at each layer (identifies dead channels)

Usage:
    # From checkpoint (no data needed, uses random input):
    python examples/shapenet_seg/trace_t2_pool.py \\
        --ckpt examples/shapenet_seg/shapenet_best.pt

    # With real data:
    python examples/shapenet_seg/trace_t2_pool.py \\
        --ckpt examples/shapenet_seg/shapenet_best.pt \\
        --data_root ../data/shapenetpart_hdf5_2048 \\
        --num_batches 3

    # From scratch (random init, shows baseline cancellation):
    python examples/shapenet_seg/trace_t2_pool.py --from_scratch
"""

import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from torch import Tensor

from examples.shapenet_seg.model import SE3PartSegNet


# ═══════════════════════════════════════════════════════════════════
# Hook infrastructure: traces s/v/t2 norms through every layer
# ═══════════════════════════════════════════════════════════════════

class SVT2Tracer:
    """Trace scalar / vector / type-2 norms at every stage of the backbone.

    Hooks block.forward to capture norms AFTER each sub-operation.
    Hooks pool.forward to capture the norm drop across pooling.
    """

    def __init__(self, model: SE3PartSegNet):
        self.model = model
        self.records: List[Dict[str, float]] = []
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
        self.records.clear()

    def _record(self, name: str, s: Tensor, v: Tensor,
                t2: Optional[Tensor]) -> None:
        """Record norms for a named location in the network."""
        rec = {
            'name': name,
            's_norm': s.detach().norm(dim=-1).mean().item(),
            'v_norm': v.detach().norm(dim=-1).mean().item(),
            'N': s.shape[0],
        }
        if t2 is not None:
            # Per-channel norms: shape [N, C_t2]
            t2_ch_norms = t2.detach().norm(dim=-1)  # [N, C_t2]
            rec['t2_norm'] = t2_ch_norms.mean().item()
            rec['t2_ch_norms'] = t2_ch_norms.mean(0).tolist()  # per channel
        else:
            rec['t2_norm'] = 0.0
            rec['t2_ch_norms'] = []
        self.records.append(rec)

    def _hook_block(self, si: int, bi: int, blk) -> None:
        """Replace block.forward with a traced version."""
        tracer = self

        def traced_forward(scalars, vectors, graph, type2=None):
            has_t2 = type2 is not None and blk.channels_type2 > 0
            name = f'Enc{si}b{bi}'

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

            tracer._record(name, s, v, t2)

            if has_t2 and t2 is not None:
                return s, v, t2
            return s, v

        blk.forward = traced_forward

    def _hook_pool(self, idx: int, pool) -> None:
        """Wrap pool.forward to record pre/post norms."""
        tracer = self
        orig_fwd = pool.forward

        def traced_fwd(*a, **kw):
            # pre-pool: a = (pos, scalars, vectors, ptr, type2=...)
            s_in, v_in = a[1], a[2]
            t2_in = kw.get('type2', None)
            tracer._record(f'Pool{idx}_pre', s_in, v_in, t2_in)

            result = orig_fwd(*a, **kw)

            # result = (pos, s_out, v_out, t2_out, ptr_out, fps_idx)
            # or (pos, s_out, v_out, ptr_out, fps_idx) if no t2
            s_out, v_out = result[1], result[2]
            # Detect t2_out from result length
            if len(result) >= 6:
                t2_out = result[3]
            else:
                t2_out = None
            tracer._record(f'Pool{idx}', s_out, v_out, t2_out)

            return result

        pool.forward = traced_fwd


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def print_trace(tracer: SVT2Tracer) -> None:
    """Print the trace table with cancellation ratios."""
    recs = tracer.records

    # Header
    print()
    print('═' * 90)
    print('  FEATURE NORM TRACE: s / v / t2 through backbone')
    print('═' * 90)
    hdr = (f'  {"Layer":<14}│{"N":>6} │ {"s_norm":>8} │ '
           f'{"v_norm":>8} │ {"t2_norm":>8} │ '
           f'{"v_Δ%":>7} │ {"t2_Δ%":>7} │ '
           f'{"s_Δ%":>7}')
    print(hdr)
    print('  ' + '─' * 86)

    prev_s, prev_v, prev_t2 = None, None, None

    for rec in recs:
        name = rec['name']
        s, v, t2 = rec['s_norm'], rec['v_norm'], rec['t2_norm']
        N = rec['N']

        # Compute drops
        def pct(cur: float, prev_val: Optional[float]) -> str:
            if prev_val is None or prev_val < 1e-6:
                return '       '
            change = (cur - prev_val) / prev_val * 100
            if abs(change) < 1:
                return '       '
            sym = '▼' if change < 0 else '▲'
            return f'{change:+5.1f}%{sym}'

        # Only show drops at pool layers (where cancellation happens)
        is_pool = 'Pool' in name and '_pre' not in name
        if is_pool:
            v_drop = pct(v, prev_v)
            t2_drop = pct(t2, prev_t2)
            s_drop = pct(s, prev_s)
        else:
            v_drop = t2_drop = s_drop = '       '

        row = (f'  {name:<14}│{N:>6} │ {s:>8.3f} │ '
               f'{v:>8.4f} │ {t2:>8.4f} │ '
               f'{v_drop:>7} │ {t2_drop:>7} │ {s_drop:>7}')

        # Highlight pool lines
        if is_pool:
            row = f'\033[93m{row}\033[0m'  # yellow

        print(row)

        prev_s, prev_v, prev_t2 = s, v, t2

    print()

    # Per-channel t2 norms at key points
    print('  t2 per-channel norms (identifies dead channels):')
    for rec in recs:
        ch = rec.get('t2_ch_norms', [])
        if ch:
            name = rec['name']
            min_ch = min(ch) if ch else 0
            max_ch = max(ch) if ch else 0
            ch_str = ' '.join(f'{c:.3f}' for c in ch)
            dead = sum(1 for c in ch if c < 0.01)
            print(f'    {name:<14}: [{ch_str}]  '
                  f'range=[{min_ch:.4f}, {max_ch:.4f}]  '
                  f'dead={dead}/{len(ch)}')
    print()


def print_weight_diagnostics(model: SE3PartSegNet) -> None:
    """Print projection weight norms and head column analysis."""
    print('═' * 90)
    print('  PROJECTION WEIGHT DIAGNOSTICS')
    print('═' * 90)

    # Projection weights
    if hasattr(model, 't2_inv_proj'):
        t2_w = model.t2_inv_proj.weight
        print(f'  t2_inv_proj: shape={list(t2_w.shape)}  '
              f'||W||={t2_w.norm():.4f}  std={t2_w.std():.4f}')
    if hasattr(model, 'v_inv_proj'):
        v_w = model.v_inv_proj.weight
        print(f'  v_inv_proj:  shape={list(v_w.shape)}  '
              f'||W||={v_w.norm():.4f}  std={v_w.std():.4f}')
    if hasattr(model, 't2_inv_proj') and hasattr(model, 'v_inv_proj'):
        ratio = model.t2_inv_proj.weight.norm() / model.v_inv_proj.weight.norm()
        print(f'  Ratio t2/v: {ratio:.4f}  '
              f'{"(OK)" if ratio > 0.5 else "(⚠ t2 proj shrinking!)"}')
    print()

    # Head first-layer column norms
    if hasattr(model, 'head') and len(model.head) > 0:
        head_w = model.head[0].weight  # [out, in]
        in_dim = head_w.shape[1]
        hs = model.hidden_scalar
        hv = hs // 2 if hasattr(model, 'v_inv_proj') else 0
        ht2 = hs // 2 if hasattr(model, 't2_inv_proj') else 0

        sections = []
        offset = 0
        sections.append(('s_out', offset, offset + hs)); offset += hs
        if hv > 0:
            sections.append(('v_inv', offset, offset + hv)); offset += hv
        if ht2 > 0:
            sections.append(('t2_inv', offset, offset + ht2)); offset += ht2
        sections.append(('rest', offset, in_dim))

        print('  Head[0] column norms by feature group:')
        for name, lo, hi in sections:
            cols = head_w[:, lo:hi]
            print(f'    {name:<8} [{lo:3d}:{hi:3d}]: '
                  f'||W||={cols.norm():.4f}  '
                  f'per_col={cols.norm(dim=0).mean():.4f}')
    print()


def print_output_norm_diagnostics(model: SE3PartSegNet) -> None:
    """Print OutputNorm weights (explains s0 decline)."""
    print('═' * 90)
    print('  OUTPUT NORM SCALAR WEIGHTS (s0 decline diagnostic)')
    print('═' * 90)
    bb = model.backbone
    for si in range(bb.num_stages):
        for bi, blk in enumerate(bb.encoder_stages[si].blocks):
            sw = blk.output_norm.scalar_weight
            print(f'  Enc{si}b{bi}: scalar_weight '
                  f'mean={sw.mean():.4f}  '
                  f'range=[{sw.min():.4f}, {sw.max():.4f}]')
            if hasattr(blk.output_norm, 'type2_weight'):
                t2w = blk.output_norm.type2_weight
                print(f'          type2_weight  '
                      f'mean={t2w.mean():.4f}  '
                      f'range=[{t2w.min():.4f}, {t2w.max():.4f}]')
    print()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Trace s/v/t2 norms — verify pooling cancellation')
    p.add_argument('--ckpt', type=str, default=None,
                   help='Checkpoint path (shapenet_best.pt)')
    p.add_argument('--data_root', type=str, default=None,
                   help='ShapeNet data root (optional, uses random if absent)')
    p.add_argument('--num_batches', type=int, default=3,
                   help='Number of batches to average over')
    p.add_argument('--from_scratch', action='store_true',
                   help='Use random init (no checkpoint)')
    p.add_argument('--device', type=str, default='cpu',
                   help='Device (cpu for local, cuda for remote)')
    args = p.parse_args()

    dev = torch.device(args.device)
    print(f'Device: {dev}')

    # ── Model ──
    model_kwargs = dict(
        in_channels=1,
        hidden_scalar=96, hidden_vector=24, hidden_type2=8,
        num_stages=3, layers_per_stage=2, pool_ratio=0.25,
        head_hidden=256, use_normals=True, gate_mode='norm',
        use_self_tp=True, use_bottleneck_attn=True,
    )

    if args.ckpt and not args.from_scratch:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        # Use checkpoint's model_args if available
        if 'model_args' in ckpt:
            a = ckpt['model_args']
            for k in ['hidden_scalar', 'hidden_vector', 'hidden_type2',
                       'num_stages', 'layers_per_stage', 'pool_ratio',
                       'head_hidden', 'gate_mode', 'use_self_tp',
                       'use_bottleneck_attn']:
                if k in a:
                    model_kwargs[k] = a[k]

    model = SE3PartSegNet(**model_kwargs).to(dev)

    if args.ckpt and not args.from_scratch:
        missing, unexpected = model.load_state_dict(
            ckpt['model_state_dict'], strict=False)
        if missing:
            print(f'  Missing keys: {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')
        epoch = ckpt.get('epoch', '?')
        miou = ckpt.get('inst_miou', ckpt.get('miou', '?'))
        print(f'  Loaded checkpoint: epoch={epoch}, inst_mIoU={miou}')

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ── Install hooks ──
    tracer = SVT2Tracer(model)
    tracer.install()
    model.eval()

    # ── Data ──
    if args.data_root:
        from examples.shapenet_seg.dataset import (
            ShapeNetPartDataset, collate_fn)
        from torch.utils.data import DataLoader
        ds = ShapeNetPartDataset(args.data_root, split='trainval',
                                 normalize=True)
        dl = DataLoader(ds, batch_size=16, shuffle=True,
                        collate_fn=collate_fn, num_workers=0,
                        drop_last=True)
        print(f'Data: {len(ds)} shapes, using {args.num_batches} batches')

        with torch.no_grad():
            for i, (pos, normals, labels, ptr, cat_idx) in enumerate(dl):
                if i >= args.num_batches:
                    break
                tracer.reset()  # reset per batch, keep last
                pos = pos.to(dev)
                normals = normals.to(dev)
                ptr = ptr.to(dev)
                cat_idx = cat_idx.to(dev)
                _ = model(pos, ptr, cat_idx, normals)
    else:
        # Random input (works without data)
        print('Using random input (no data_root specified)')
        torch.manual_seed(42)
        N = 512
        pos = torch.randn(N, 3, device=dev)
        pos = pos / pos.norm(dim=-1).max()
        normals = torch.randn(N, 3, device=dev)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        ptr = torch.tensor([0, N], dtype=torch.int64, device=dev)
        cat = torch.tensor([5], dtype=torch.int64, device=dev)

        with torch.no_grad():
            tracer.reset()
            _ = model(pos, ptr, cat, normals)

    # ── Report ──
    print_trace(tracer)
    print_weight_diagnostics(model)
    print_output_norm_diagnostics(model)

    # ── Cancellation summary ──
    print('═' * 90)
    print('  CANCELLATION SUMMARY')
    print('═' * 90)
    pool_recs = [(r, tracer.records[i - 1])
                 for i, r in enumerate(tracer.records)
                 if 'Pool' in r['name'] and '_pre' in r['name']]

    for pre, _ in pool_recs:
        # Find corresponding post-pool
        pool_name = pre['name'].replace('_pre', '')
        post = next((r for r in tracer.records if r['name'] == pool_name), None)
        if post is None:
            continue

        s_drop = (post['s_norm'] - pre['s_norm']) / max(pre['s_norm'], 1e-8) * 100
        v_drop = (post['v_norm'] - pre['v_norm']) / max(pre['v_norm'], 1e-8) * 100
        t2_drop = (post['t2_norm'] - pre['t2_norm']) / max(pre['t2_norm'], 1e-8) * 100

        print(f'  {pool_name}:')
        print(f'    scalar:  {pre["s_norm"]:.3f} → {post["s_norm"]:.3f}  '
              f'({s_drop:+.1f}%)')
        print(f'    vector:  {pre["v_norm"]:.4f} → {post["v_norm"]:.4f}  '
              f'({v_drop:+.1f}%) '
              f'{"← cancellation!" if v_drop < -30 else ""}')
        print(f'    type-2:  {pre["t2_norm"]:.4f} → {post["t2_norm"]:.4f}  '
              f'({t2_drop:+.1f}%) '
              f'{"← SEVERE cancellation!" if t2_drop < -40 else ""}')
    print()

    # Theory check: scalar survives because 1D → no direction to cancel
    s_drops = []
    t2_drops = []
    for pre_r, _ in pool_recs:
        pool_name = pre_r['name'].replace('_pre', '')
        post_r = next((r for r in tracer.records if r['name'] == pool_name), None)
        if post_r:
            s_drops.append(
                (post_r['s_norm'] - pre_r['s_norm']) / max(pre_r['s_norm'], 1e-8) * 100)
            t2_drops.append(
                (post_r['t2_norm'] - pre_r['t2_norm']) / max(pre_r['t2_norm'], 1e-8) * 100)

    if s_drops and t2_drops:
        avg_s = sum(s_drops) / len(s_drops)
        avg_t2 = sum(t2_drops) / len(t2_drops)
        print(f'  Average pooling loss:')
        print(f'    scalar: {avg_s:+.1f}%')
        print(f'    type-2: {avg_t2:+.1f}%')
        if avg_t2 < -30 and avg_s > -15:
            print()
            print('  ╔═══════════════════════════════════════════════════════╗')
            print('  ║ HYPOTHESIS CONFIRMED: Pooling selectively destroys   ║')
            print('  ║ higher-order features (t2/v) via component           ║')
            print('  ║ cancellation, while preserving scalars.              ║')
            print('  ╚═══════════════════════════════════════════════════════╝')
        elif abs(avg_t2) < 15:
            print()
            print('  Result: minimal cancellation — hypothesis NOT confirmed')
        print()

    print('═' * 90)


if __name__ == '__main__':
    main()
