# Example demonstrating 2D vector operations with different backends
import numpy as np
try:
    import awkward as ak
    HAS_AWKWARD = True
except ImportError:
    HAS_AWKWARD = False

from lvec import Vector2D

# NumPy backend example
print("=== 2D Vector Operations with NumPy Backend ===\n")
x_np = np.array([1.0, 2.0, 3.0])
y_np = np.array([4.0, 5.0, 6.0])
vec2d_np = Vector2D(x_np, y_np)

print(f"Vector components (x, y): ({vec2d_np.x}, {vec2d_np.y})")
print(f"Magnitude: {vec2d_np.r}")
print(f"Azimuthal angle (phi): {vec2d_np.phi}")

# Vector operations with NumPy
v1_np = Vector2D(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
v2_np = Vector2D(np.array([5.0, 6.0]), np.array([7.0, 8.0]))

print("\nVector Operations:")
print(f"v1 + v2: ({(v1_np + v2_np).x}, {(v1_np + v2_np).y})")
print(f"v1 · v2 (dot product): {v1_np.dot(v2_np)}")

# Awkward Array backend example (if available)
if HAS_AWKWARD:
    print("\n=== 2D Vector Operations with Awkward Backend ===\n")
    x_ak = ak.Array([1.0, 2.0, 3.0])
    y_ak = ak.Array([4.0, 5.0, 6.0])
    vec2d_ak = Vector2D(x_ak, y_ak)
    
    print(f"Vector components (x, y): ({vec2d_ak.x}, {vec2d_ak.y})")
    print(f"Magnitude: {vec2d_ak.r}")
    print(f"Azimuthal angle (phi): {vec2d_ak.phi}")
    
    # Vector operations with Awkward
    v1_ak = Vector2D(ak.Array([1.0, 2.0]), ak.Array([3.0, 4.0]))
    v2_ak = Vector2D(ak.Array([5.0, 6.0]), ak.Array([7.0, 8.0]))
    
    print("\nVector Operations:")
    print(f"v1 + v2: ({(v1_ak + v2_ak).x}, {(v1_ak + v2_ak).y})")
    print(f"v1 · v2 (dot product): {v1_ak.dot(v2_ak)}")