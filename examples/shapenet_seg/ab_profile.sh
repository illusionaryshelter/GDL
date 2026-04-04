#!/bin/bash
# A/B profiling: 43ed1c3 (old) vs HEAD (new)
# Run on remote GPU to compare epoch time precisely.
#
# Usage:  bash examples/shapenet_seg/ab_profile.sh
set -e

DATA_ROOT="../data/shapenetpart_hdf5_2048"
COMMON_ARGS="--hidden_scalar 96 --hidden_vector 24 --hidden_type2 8 \
  --num_stages 3 --layers_per_stage 2 \
  --gate_mode norm --use_self_tp --use_bottleneck_attn \
  --head_hidden 256 --batch_size 64 --lr 1e-3 --weight_decay 5e-4 \
  --no_amp --compile --profile \
  --data_root=${DATA_ROOT}"

echo "============================================"
echo "  A/B Profile Test"
echo "============================================"
echo ""

# Current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CURRENT_SHA=$(git rev-parse --short HEAD)

# ── GROUP A: old commit (43ed1c3) ──
echo ">>> [A] Checking out 43ed1c3 (old baseline)..."
git stash -q 2>/dev/null || true
git checkout 43ed1c3 -q

echo ">>> [A] Running profile..."
PYTHONPATH=.:$PYTHONPATH python examples/shapenet_seg/train.py \
  ${COMMON_ARGS} --normal_drop_rate 0.0 2>&1 | tee /tmp/profile_A.txt

echo ""
echo ">>> [A] Done."
echo ""

# ── GROUP B: current HEAD ──
echo ">>> [B] Checking out ${CURRENT_BRANCH} (${CURRENT_SHA})..."
git checkout ${CURRENT_BRANCH} -q
git stash pop -q 2>/dev/null || true

echo ">>> [B] Running profile..."
PYTHONPATH=.:$PYTHONPATH python examples/shapenet_seg/train.py \
  ${COMMON_ARGS} 2>&1 | tee /tmp/profile_B.txt

echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"
echo ""
echo "--- [A] 43ed1c3 (old) ---"
grep "Est\. epoch\|TOTAL\|Throughput" /tmp/profile_A.txt
echo ""
echo "--- [B] ${CURRENT_SHA} (new) ---"
grep "Est\. epoch\|TOTAL\|Throughput" /tmp/profile_B.txt
