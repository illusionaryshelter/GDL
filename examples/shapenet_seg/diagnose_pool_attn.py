#!/usr/bin/env python3
"""Remote attention autopsy — runs on training machine with REAL ShapeNet data.

Usage:
    cd /path/to/GDL
    PYTHONPATH=. python3 examples/shapenet_seg/diagnose_pool_attn.py \
        --checkpoint examples/shapenet_seg/shapenet_best.pt \
        --data_root data/ShapeNetPart \
        --n_samples 50

Answers: Is |Q|/K_std really 4x on real data? Do we need QK-Norm?
"""
import argparse
import math
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geoembodied.nn.models.part_segmentation import SE3PartSegNet
from examples.shapenet_seg.dataset import ShapeNetPartDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_root', type=str, default='data/ShapeNetPart')
    p.add_argument('--n_samples', type=int, default=50)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


class PoolProbe:
    def __init__(self, pool, name):
        self.name = name
        self.pool = pool
        self.caps = {}
        self._hooks = []

    def install(self):
        self._hooks.append(self.pool.q_proj.register_forward_hook(
            lambda m, i, o: self.caps.update({'Q': o.detach()})))
        self._hooks.append(self.pool.k_proj.register_forward_hook(
            lambda m, i, o: self.caps.update({'K': o.detach()})))
        self._hooks.append(self.pool.attn_feat_norm.register_forward_hook(
            lambda m, i, o: self.caps.update({'inv_raw': i[0].detach(), 'inv_normed': o.detach()})))
        self._hooks.append(self.pool.attn_vec.register_forward_hook(
            lambda m, i, o: self.caps.update({'content': o.detach()})))
        self._hooks.append(self.pool.geo_mlp[-1].register_forward_hook(
            lambda m, i, o: self.caps.update({'geo': o.detach()})))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def analyze(self):
        pool = self.pool
        K = pool.k_neighbors
        Q = self.caps['Q']
        K_feat = self.caps['K']
        N_out = Q.shape[0]
        content = self.caps['content'].reshape(N_out, K)
        geo = self.caps['geo'].reshape(N_out, K)
        inv_raw = self.caps['inv_raw']
        temp = pool.log_temperature.exp().clamp(min=0.01).item()
        total = (content + geo) / temp
        attn_w = pool._last_attn_weights

        def rng(x): return (x.max(1).values - x.min(1).values).mean().item()
        def std(x): return x.std(1).mean().item()

        entropy = -(attn_w * attn_w.clamp(min=1e-8).log()).sum(-1).mean().item()
        Q_norm = Q.norm(dim=-1).mean().item()
        K_std = K_feat.std(dim=1).norm(dim=-1).mean().item()

        n_feat = inv_raw.shape[-1]
        feat_ranges = [(inv_raw[:,:,f].max(1).values - inv_raw[:,:,f].min(1).values).mean().item()
                       for f in range(n_feat)]

        return {
            'N_out': N_out, 'temp': temp,
            'c_rng': rng(content), 'c_std': std(content), 'c_mag': content.abs().mean().item(),
            'g_rng': rng(geo), 'g_std': std(geo), 'g_mag': geo.abs().mean().item(),
            't_rng': rng(total), 't_std': std(total),
            'Q_norm': Q_norm, 'K_std': K_std, 'QK_ratio': Q_norm / (K_std + 1e-8),
            'entropy': entropy, 'uniform': entropy / math.log(K),
            'max_w': attn_w.max(1).values.mean().item(),
            'a_norm': pool.attn_vec.weight.data.norm().item(),
            'feat_ranges': feat_ranges,
        }


def main():
    args = parse_args()
    device = args.device

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

    # Adapt pool layers to match checkpoint's invariant feature count
    ckpt_sd = ckpt['model_state_dict']
    norm_key = 'backbone.pool_layers.0.attn_feat_norm.weight'
    if norm_key in ckpt_sd:
        ckpt_n = ckpt_sd[norm_key].shape[0]
        from geoembodied.nn.modules.equivariant_pool import NeighborNorm
        import torch.nn as nn
        for pool in model.backbone.pool_layers:
            cur_n = pool.attn_feat_norm.n_features
            if cur_n != ckpt_n:
                pool.attn_feat_norm = NeighborNorm(ckpt_n).to(device)
                h = pool.geo_mlp[0].out_features
                pool.geo_mlp = nn.Sequential(
                    nn.Linear(ckpt_n, h), nn.SiLU(), nn.Linear(h, 1),
                ).to(device)

    model.load_state_dict(ckpt_sd, strict=False)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params) on {device}")

    ds = ShapeNetPartDataset(args.data_root, split='test', num_points=2048, normalize=True)
    print(f"Test set: {len(ds)} shapes")

    probes = []
    for pi, pool in enumerate(model.backbone.pool_layers):
        p = PoolProbe(pool, f"pool{pi}")
        p.install()
        probes.append(p)

    n_pools = len(probes)
    accs = [{} for _ in range(n_pools)]
    indices = list(range(0, len(ds), max(1, len(ds) // args.n_samples)))[:args.n_samples]

    for i, idx in enumerate(indices):
        item = ds[idx]
        pos = item['pos'].to(device)
        normals = item['normal'].to(device)
        ptr = torch.tensor([0, pos.shape[0]], dtype=torch.int64, device=device)
        cat = torch.tensor([item['cat_idx']], dtype=torch.int64, device=device)
        with torch.no_grad():
            model(pos, ptr, cat, normals=normals)
        for pi, probe in enumerate(probes):
            m = probe.analyze()
            for k, v in m.items():
                if isinstance(v, list):
                    accs[pi].setdefault(k, []).append(v)
                else:
                    accs[pi].setdefault(k, []).append(v)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(indices)}...")

    for p in probes:
        p.remove()

    # ─── Report ───
    feat_names = ['dist', 's_ratio', 'cos_vv', 'cos_dv_j', 'cos_dv_i', 'v_nbr', 'v_seed', 't2_nbr']

    print("\n" + "=" * 80)
    print("ATTENTION AUTOPSY — REAL DATA")
    print("=" * 80)

    for pi in range(n_pools):
        acc = accs[pi]
        n = len(acc['uniform'])
        def avg(k): return sum(acc[k]) / n
        K = probes[pi].pool.k_neighbors

        print(f"\n{'━'*60}")
        print(f"  POOL{pi} (K={K}, N≈{int(avg('N_out'))}, {n} shapes)")
        print(f"{'━'*60}")
        print(f"  uniform={avg('uniform'):.3f}  max_w={avg('max_w'):.4f}  temp={avg('temp'):.4f}")
        eff = avg('t_rng') / avg('temp')
        c_pct = avg('c_rng') / (avg('c_rng') + avg('g_rng') + 1e-8) * 100
        print(f"  total_rng={avg('t_rng'):.3f}  content_rng={avg('c_rng'):.3f}({c_pct:.0f}%)  geo_rng={avg('g_rng'):.3f}({100-c_pct:.0f}%)")
        print(f"  effective_range={eff:.2f} {'✅' if eff >= 5.5 else '❌'} (need≥5.5)")
        print(f"  |Q|={avg('Q_norm'):.3f}  K_std={avg('K_std'):.3f}  |Q|/K_std={avg('QK_ratio'):.1f}x {'⚠️' if avg('QK_ratio') > 3 else '✅'}")
        print(f"  |a_vec|={avg('a_norm'):.3f}")

        n_feat = len(acc['feat_ranges'][0])
        print(f"  Invariant features (raw intra-seed range):")
        for fi in range(n_feat):
            fn = feat_names[fi] if fi < len(feat_names) else f'f{fi}'
            r = sum(x[fi] for x in acc['feat_ranges']) / n
            print(f"    {fn:>12s}: {r:.4f}{' ← DEAD' if r < 0.001 else ''}")

    print("\n" + "=" * 80)
    print("VERDICT: Need QK-Norm?")
    print("=" * 80)
    for pi in range(n_pools):
        acc = accs[pi]
        n = len(acc['uniform'])
        def avg(k): return sum(acc[k]) / n
        qk = avg('QK_ratio')
        eff = avg('t_rng') / avg('temp')
        print(f"  pool{pi}: |Q|/K_std={qk:.1f}x, eff_range={eff:.2f}")
        if qk > 3.0:
            print(f"    → YES: Q dominates K → content attention has no discrimination")
        elif qk > 2.0:
            print(f"    → BORDERLINE: monitor but likely OK")
        else:
            print(f"    → NO: Q/K balanced")


if __name__ == '__main__':
    main()
