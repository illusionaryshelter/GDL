#!/usr/bin/env python3
"""Remote attention autopsy — runs on the training machine with REAL ShapeNet data.

Usage:
    cd /path/to/GDL
    PYTHONPATH=. python3 examples/shapenet_seg/diagnose_pool_attn.py \
        --checkpoint examples/shapenet_seg/shapenet_best.pt \
        --data_root data/ShapeNetPart \
        --n_samples 50

Outputs a precise decomposition of pool attention into content_score
and geo_bias, using real test data. This validates or refutes the
findings from the synthetic-data local diagnosis.

Key questions this script answers:
  1. Is |Q|/K_std really 4x in pool0 on real data? (data-dependent)
  2. Is v_seed really a dead feature? (structural — should be yes)
  3. What is the actual effective_logit_range? (data-dependent)
  4. Temperature learned value? (data-independent, just read param)
"""
import argparse
import math
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geoembodied.nn.models.part_segmentation import SE3PartSegNet
from examples.shapenet_seg.dataset import ShapeNetPartDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Pool attention autopsy on real data")
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_root', type=str, default='data/ShapeNetPart')
    p.add_argument('--n_samples', type=int, default=50,
                   help='Number of test shapes to analyze')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


class PoolProbe:
    """Hooks into a single EquivariantPool layer to capture internal tensors."""
    
    def __init__(self, pool_layer, name: str):
        self.name = name
        self.pool = pool_layer
        self.captures = {}
        self._hooks = []
    
    def install(self):
        def on_q(mod, inp, out):
            self.captures['Q'] = out.detach()
        def on_k(mod, inp, out):
            self.captures['K'] = out.detach()
        def on_norm(mod, inp, out):
            self.captures['inv_raw'] = inp[0].detach()
            self.captures['inv_normed'] = out.detach()
        def on_content(mod, inp, out):
            self.captures['content_raw'] = out.detach()
        def on_geo(mod, inp, out):
            self.captures['geo_raw'] = out.detach()
            
        self._hooks.append(self.pool.q_proj.register_forward_hook(on_q))
        self._hooks.append(self.pool.k_proj.register_forward_hook(on_k))
        self._hooks.append(self.pool.attn_feat_norm.register_forward_hook(on_norm))
        self._hooks.append(self.pool.attn_vec.register_forward_hook(on_content))
        self._hooks.append(self.pool.geo_mlp[-1].register_forward_hook(on_geo))
    
    def remove(self):
        for h in self._hooks:
            h.remove()
    
    def analyze(self):
        """Return a dict of scalar metrics from captured tensors."""
        pool = self.pool
        K = pool.k_neighbors
        
        Q = self.captures['Q']                 # [N_out, d_attn]
        K_feat = self.captures['K']            # [N_out, K, d_attn]
        content_raw = self.captures['content_raw']
        geo_raw = self.captures['geo_raw']
        inv_raw = self.captures['inv_raw']     # [N_out, K, F]
        inv_normed = self.captures['inv_normed']
        
        N_out = Q.shape[0]
        
        # Reshape to [N_out, K]
        content = content_raw.reshape(N_out, K) if content_raw.numel() == N_out * K else content_raw.squeeze(-1)
        geo = geo_raw.reshape(N_out, K) if geo_raw.numel() == N_out * K else geo_raw.squeeze(-1)
        if content.dim() == 1:
            content = content.reshape(N_out, K)
        if geo.dim() == 1:
            geo = geo.reshape(N_out, K)
        
        temp = pool.log_temperature.exp().clamp(min=0.01).item()
        total = (content + geo) / temp
        attn_w = pool._last_attn_weights  # [N_out, K]
        
        def rng_std_mag(x):
            return {
                'range': (x.max(1).values - x.min(1).values).mean().item(),
                'std': x.std(1).mean().item(),
                'mag': x.abs().mean().item(),
            }
        
        # Q/K analysis
        Q_norm = Q.norm(dim=-1).mean().item()
        K_std = K_feat.std(dim=1).norm(dim=-1).mean().item()
        
        # SiLU I/O range
        silu_in = Q.unsqueeze(1) + K_feat
        silu_out = torch.nn.functional.silu(silu_in)
        silu_in_rng = (silu_in.max(1).values - silu_in.min(1).values).mean().item()
        silu_out_rng = (silu_out.max(1).values - silu_out.min(1).values).mean().item()
        
        # Attention quality
        entropy = -(attn_w * attn_w.clamp(min=1e-8).log()).sum(-1).mean().item()
        uniform = entropy / math.log(K)
        max_w = attn_w.max(1).values.mean().item()
        
        # Per-feature invariant range
        n_feat = inv_raw.shape[-1]
        feat_raw_ranges = []
        feat_normed_ranges = []
        for fi in range(n_feat):
            feat_raw_ranges.append(
                (inv_raw[:, :, fi].max(1).values - inv_raw[:, :, fi].min(1).values).mean().item()
            )
            feat_normed_ranges.append(
                (inv_normed[:, :, fi].max(1).values - inv_normed[:, :, fi].min(1).values).mean().item()
            )
        
        c = rng_std_mag(content)
        g = rng_std_mag(geo)
        t = rng_std_mag(total)
        
        return {
            'N_out': N_out,
            'content_range': c['range'], 'content_std': c['std'], 'content_mag': c['mag'],
            'geo_range': g['range'], 'geo_std': g['std'], 'geo_mag': g['mag'],
            'total_range': t['range'], 'total_std': t['std'],
            'temp': temp,
            'Q_norm': Q_norm, 'K_std': K_std, 'QK_ratio': Q_norm / (K_std + 1e-8),
            'silu_in_range': silu_in_rng, 'silu_out_range': silu_out_rng,
            'a_norm': pool.attn_vec.weight.data.norm().item(),
            'entropy': entropy, 'uniform': uniform, 'max_w': max_w,
            'feat_raw_ranges': feat_raw_ranges,
            'feat_normed_ranges': feat_normed_ranges,
        }


def main():
    args = parse_args()
    device = args.device
    
    # Load model
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    a = ckpt.get('model_args', {})
    model = SE3PartSegNet(
        num_categories=a.get('num_categories', 16),
        num_parts=a.get('num_parts', 50),
        hidden_scalar=a.get('hidden_scalar', 96),
        hidden_vector=a.get('hidden_vector', 24),
        hidden_type2=a.get('hidden_type2', 8),
        num_stages=a.get('num_stages', 3),
        layers_per_stage=a.get('layers_per_stage', 2),
        pool_ratio=a.get('pool_ratio', 0.25),
        head_hidden=a.get('head_hidden', 256),
        gate_mode=a.get('gate_mode', 'scalar'),
        use_self_tp=a.get('use_self_tp', False),
        use_bottleneck_attn=a.get('use_bottleneck_attn', False),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params) on {device}")
    
    # Load real data
    ds = ShapeNetPartDataset(args.data_root, split='test', num_points=2048, normalize=True)
    print(f"Test set: {len(ds)} shapes")
    
    # Install probes
    probes = []
    for pi, pool in enumerate(model.backbone.pool_layers):
        p = PoolProbe(pool, f"pool{pi}")
        p.install()
        probes.append(p)
    
    # Run
    n_pools = len(probes)
    accumulators = [{} for _ in range(n_pools)]
    
    indices = list(range(0, len(ds), max(1, len(ds) // args.n_samples)))[:args.n_samples]
    
    for i, idx in enumerate(indices):
        item = ds[idx]
        pos = item['pos'].to(device)
        normals = item['normal'].to(device)
        cat_idx = item['cat_idx']
        ptr = torch.tensor([0, pos.shape[0]], dtype=torch.int64, device=device)
        cat_indices = torch.tensor([cat_idx], dtype=torch.int64, device=device)
        
        with torch.no_grad():
            model(pos, ptr, cat_indices, normals=normals)
        
        for pi, probe in enumerate(probes):
            metrics = probe.analyze()
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    accumulators[pi].setdefault(k, []).append(v)
                elif isinstance(v, list):
                    accumulators[pi].setdefault(k, []).append(v)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(indices)} shapes...")
    
    for probe in probes:
        probe.remove()
    
    # ─── Print Report ───
    feat_names = ['dist', 's_ratio', 'cos_vv', 'cos_dv_j', 'cos_dv_i', 'v_nbr', 'v_seed', 't2_nbr']
    
    print("\n" + "=" * 90)
    print("ATTENTION AUTOPSY — REAL ShapeNet Test Data")
    print("=" * 90)
    
    for pi in range(n_pools):
        acc = accumulators[pi]
        n = len(acc['uniform'])
        
        def avg(key):
            return sum(acc[key]) / n
        
        K = probes[pi].pool.k_neighbors
        
        print(f"\n{'━' * 70}")
        print(f"  POOL{pi} (K={K}, N_out≈{int(avg('N_out'))}, {n} shapes)")
        print(f"{'━' * 70}")
        
        print(f"\n  ATTENTION:  entropy={avg('entropy'):.3f}  uniform={avg('uniform'):.3f}  max_w={avg('max_w'):.4f}")
        print(f"  TEMPERATURE: {avg('temp'):.4f}")
        
        print(f"\n  LOGIT DECOMPOSITION (per-seed intra-K):")
        print(f"    {'':>16s} {'RANGE':>7s} {'STD':>7s} {'|MEAN|':>7s}")
        print(f"    {'total_logits':>16s} {avg('total_range'):7.3f} {avg('total_std'):7.3f}")
        print(f"    {'content_score':>16s} {avg('content_range'):7.3f} {avg('content_std'):7.3f} {avg('content_mag'):7.3f}")
        print(f"    {'geo_bias':>16s} {avg('geo_range'):7.3f} {avg('geo_std'):7.3f} {avg('geo_mag'):7.3f}")
        
        c_pct = avg('content_range') / (avg('content_range') + avg('geo_range') + 1e-8) * 100
        eff = avg('total_range') / avg('temp')
        print(f"    content%={c_pct:.1f}%  geo%={100-c_pct:.1f}%")
        print(f"    effective_range = {avg('total_range'):.2f}/{avg('temp'):.4f} = {eff:.2f} {'✅' if eff >= 5.5 else '❌'} (need≥5.5)")
        
        print(f"\n  GATv2 Q/K ANALYSIS:")
        print(f"    |Q|={avg('Q_norm'):.3f}  K_std={avg('K_std'):.3f}  |Q|/K_std={avg('QK_ratio'):.1f}x {'⚠️' if avg('QK_ratio') > 3 else '✅'}")
        print(f"    SiLU_in_range={avg('silu_in_range'):.3f}  SiLU_out_range={avg('silu_out_range'):.3f}  |a_vec|={avg('a_norm'):.3f}")
        
        print(f"\n  INVARIANT FEATURES (per-seed range):")
        n_feat = len(acc['feat_raw_ranges'][0])
        print(f"    {'Feature':>10s} {'raw_range':>10s} {'normed_range':>12s}")
        for fi in range(n_feat):
            fn = feat_names[fi] if fi < len(feat_names) else f'f{fi}'
            raw = sum(r[fi] for r in acc['feat_raw_ranges']) / n
            normed = sum(r[fi] for r in acc['feat_normed_ranges']) / n
            flag = " ← DEAD" if raw < 0.001 else ""
            print(f"    {fn:>10s} {raw:10.4f} {normed:12.4f}{flag}")
    
    # ─── Verdict ───
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    
    for pi in range(n_pools):
        acc = accumulators[pi]
        n = len(acc['uniform'])
        def avg(key):
            return sum(acc[key]) / n
        
        eff = avg('total_range') / avg('temp')
        qk = avg('QK_ratio')
        uni = avg('uniform')
        n_feat = len(acc['feat_raw_ranges'][0])
        
        dead_feats = []
        for fi in range(n_feat):
            raw = sum(r[fi] for r in acc['feat_raw_ranges']) / n
            if raw < 0.001:
                fn = feat_names[fi] if fi < len(feat_names) else f'f{fi}'
                dead_feats.append(fn)
        
        print(f"\n  pool{pi}: uniform={uni:.3f}, eff_range={eff:.2f}, |Q|/K_std={qk:.1f}x")
        
        if eff < 5.5:
            print(f"    ❌ effective_range insufficient ({eff:.2f} < 5.5)")
        else:
            print(f"    ✅ effective_range sufficient ({eff:.2f} ≥ 5.5)")
        
        if qk > 3.0:
            print(f"    ❌ Q dominates K ({qk:.1f}x) → content has no neighbor discrimination")
        else:
            print(f"    ✅ Q/K balanced ({qk:.1f}x)")
        
        if dead_feats:
            print(f"    ❌ Dead features: {', '.join(dead_feats)} (zero intra-seed variance)")
        
        print()


if __name__ == '__main__':
    main()
