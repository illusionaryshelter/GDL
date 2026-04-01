# GeoEmbodied 🌐⚙️

**Geometric Deep Learning for Robotics & Autonomous Driving**

A high-performance core library built on strict Lie group theory and differential geometry, providing physically-constrained tensor types, manifold-aware optimizers, and the foundation for equivariant neural networks.

[![Tests](https://img.shields.io/badge/tests-38%2F38%20passing-brightgreen)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)]()
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

---

## Why GeoEmbodied?

Traditional deep learning frameworks treat rotations and rigid transforms as *flat vectors* — ignoring their manifold structure. This leads to:

- 🔴 **Gimbal lock** from Euler angles
- 🔴 **Gradient explosions** from unconstrained optimization on $SE(3)$
- 🔴 **Data augmentation gymnastics** to learn trivial symmetries

**GeoEmbodied locks physics into the code.** By enforcing Lie group constraints at the tensor level, your models are *geometrically correct by construction*.

```python
from geoembodied.lietensor import SO3, SE3
from geoembodied.optim import ManifoldAdam

# Create rotation and transform — no Euler angles, ever
R = SO3.exp(torch.randn(3))          # Axis-angle → unit quaternion
T = SE3.exp(torch.randn(6))          # Twist → rigid transform (7-dim compact)

# Group operations via @ operator
composed = T1 @ T2                     # SE(3) composition
transformed_pts = T @ point_cloud      # Rigid body transform

# R + R is BLOCKED — manifold violation!
# R + R  ← TypeError: "Use Lie algebra operations instead"

# Manifold-aware optimization
pose = SE3.identity().parameter()
optimizer = ManifoldAdam([pose], lr=0.01)
loss.backward()
optimizer.step()       # Updates via exp map retraction, not additive!
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 2: Equivariant Neural Networks  (WIP)    │
│  SE3Conv, LieGroupAttention, EquivariantBN      │
├─────────────────────────────────────────────────┤
│  Layer 1: Geometry Wrappers  ✅                 │
│  LieTensor · SO3 · SE3 · ManifoldAdam/SGD       │
│  PyTree protocol · __torch_function__ guard     │
├─────────────────────────────────────────────────┤
│  Layer 0: Pure Functional Core  ✅              │
│  so3_ops · se3_ops · quaternion_ops             │
│  numeric_safe · distance                         │
│  Zero state · Zero nn.Module · Pure math         │
└─────────────────────────────────────────────────┘
```

## Key Features

### 🔒 Physically Constrained Tensors
- `SO3`: Unit quaternion (wxyz), 4-dim storage
- `SE3`: Compact representation `[q|t]`, **7-dim** (56% less memory than 4×4 matrices)
- Illegal operations (`+`, `-`) are **blocked at runtime** with clear error messages
- Periodic manifold projection corrects floating-point drift

### 🧮 Numerically Safe Math Core
- All `acos`, `sqrt`, `sinc` operations use **Taylor expansion** near singularities
- Zero NaN gradients — tested at θ = 0, θ = π, and extreme values
- **Chordal distance** for attention mechanisms (zero trig, zero singularities)

### 📐 Manifold-Aware Optimization
- `ManifoldAdam` / `ManifoldSGD`: proper Riemannian gradient + projection retraction
- Mixed parameter groups (LieTensor + standard Tensor) in one optimizer
- Auto-stabilization every N steps (AGENTS.md Rule 5)

### ⚡ torch.compile Ready
- Full **PyTree protocol** (`__tensor_flatten__` / `__tensor_unflatten__`)
- No graph breaks in `torch.compile` — Dynamo can trace through LieTensor

## Installation

```bash
# From source (development)
git clone https://github.com/your-username/GeoEmbodied.git
cd GeoEmbodied
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.1
- NumPy ≥ 1.24

## Quick Examples

### Rotation Regression
```python
import torch
from geoembodied.lietensor import SO3
from geoembodied.optim import ManifoldAdam
from geoembodied.functional.distance import so3_chordal_distance

# Target and initial rotation
target = SO3.exp(torch.tensor([0.5, -0.3, 0.8]))
current = SO3.identity().parameter()

optimizer = ManifoldAdam([current], lr=0.05)

for step in range(50):
    optimizer.zero_grad()
    loss = so3_chordal_distance(current, target)
    loss.backward()
    optimizer.step()

print(f"Final distance: {loss.item():.6f}")  # → ~0.0
```

### SE(3) Point Cloud Alignment
```python
import torch
from geoembodied.lietensor import SE3
from geoembodied.optim import ManifoldAdam

source_points = torch.randn(1000, 3)
T_gt = SE3.exp(torch.tensor([0.1, -0.2, 0.3, 1.0, 0.5, -0.5]))
target_points = (T_gt @ source_points).detach()

T_est = SE3.identity().parameter()
optimizer = ManifoldAdam([T_est], lr=0.02)

for step in range(100):
    optimizer.zero_grad()
    loss = ((T_est @ source_points - target_points) ** 2).sum()
    loss.backward()
    optimizer.step()
```

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No Euler angles** | All rotations use quaternions or SO(3) matrices |
| **Manifold-safe updates** | `T + delta` → TypeError; use `exp(δξ) ∘ T` |
| **Numerical safety** | Taylor expansion at all singularities |
| **Memory efficiency** | SE(3) in 7 floats, not 16 |
| **Ecosystem compatibility** | Works with DataLoader, DDP, torch.compile |
| **Pure functional core** | Layer 0 has zero state — portable to JAX/Triton |

## Roadmap

- [x] **Phase 0**: Core math + LieTensor types + Manifold optimizers
- [ ] **Phase 1**: Triton kernels (spherical harmonics, tensor products)
- [ ] **Phase 1**: Equivariant neural network layers (SE3Conv, LieGroupAttention)
- [ ] **Phase 2**: Differentiable SLAM / point cloud registration
- [ ] **Phase 2**: VLA (Vision-Language-Action) integration

## License

Apache License 2.0. See [LICENSE](LICENSE).
