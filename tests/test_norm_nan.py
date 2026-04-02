import torch
import torch.nn.functional as F

x = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
n = x.norm()
n.backward()
print(f"Gradient of norm at exactly 0: {x.grad}")

x = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
n = (x + 1e-8).norm()
n.backward()
print(f"Gradient of (x + 1e-8).norm() at exactly 0: {x.grad}")

x = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
n = x.norm().clamp(min=1e-8)
n.backward()
print(f"Gradient of x.norm().clamp(min=1e-8) at exactly 0: {x.grad}")
