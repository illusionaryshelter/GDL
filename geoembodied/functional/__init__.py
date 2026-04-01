# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Layer 0: Pure functional math operations.

This module contains stateless, pure functions that operate on raw tensors.
No LieTensor wrappers, no nn.Module state — just math.

This design enables:
    1. Easy 1:1 porting to JAX (replace torch → jax.numpy)
    2. Direct use in Triton kernels
    3. Clear separation from geometry wrapper logic
"""

from geoembodied.functional.numeric_safe import (
    safe_acos,
    safe_sqrt,
    taylor_sinc,
    taylor_cos_over_theta,
    taylor_theta_minus_sin_over_theta3,
    taylor_V_inv_coeff,
    taylor_half_theta_cot_half_theta,
)
from geoembodied.functional.quaternion_ops import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_apply,
    quaternion_to_matrix,
    quaternion_from_matrix,
    quaternion_slerp,
)
from geoembodied.functional.so3_ops import (
    so3_exp,
    so3_log,
    so3_hat,
    so3_vee,
    so3_multiply,
    so3_inverse,
    so3_act,
    so3_adjoint,
)
from geoembodied.functional.se3_ops import (
    se3_exp,
    se3_log,
    se3_multiply,
    se3_inverse,
    se3_act,
    se3_adjoint,
)
from geoembodied.functional.distance import (
    so3_chordal_distance,
    so3_geodesic_distance,
    se3_chordal_distance,
)
from geoembodied.functional.radius_graph import (
    radius_graph,
    compute_edge_vectors,
)
from geoembodied.functional.tensor_product import (
    tensor_product,
    tensor_product_weighted,
    get_cg_matrix,
    dot_product_l1,
    cross_product_l1,
)
from geoembodied.functional.fps import (
    farthest_point_sampling,
)
from geoembodied.functional.chamfer import (
    chamfer_distance,
)
from geoembodied.functional.knn import (
    knn,
    knn_self,
)
