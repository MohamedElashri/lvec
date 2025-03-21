# Example demonstrating 3D vector operations with different backends
import numpy as np
try:
    import awkward as ak
    HAS_AWKWARD = True
except ImportError:
    HAS_AWKWARD = False

from lvec import Vector3D

# NumPy backend example
print("=== 3D Vector Operations with NumPy Backend ===\n")
x_np = np.array([1.0, 2.0, 3.0])
y_np = np.array([4.0, 5.0, 6.0])
z_np = np.array([7.0, 8.0, 9.0])
vec3d_np = Vector3D(x_np, y_np, z_np)

print(f"Vector components (x, y, z): ({vec3d_np.x}, {vec3d_np.y}, {vec3d_np.z})")
print(f"Magnitude: {vec3d_np.r}")
print(f"Cylindrical radius (rho): {vec3d_np.rho}")
print(f"Azimuthal angle (phi): {vec3d_np.phi}")
print(f"Polar angle (theta): {vec3d_np.theta}")

# Vector operations with NumPy
v1_np = Vector3D(np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]))
v2_np = Vector3D(np.array([7.0, 8.0]), np.array([9.0, 10.0]), np.array([11.0, 12.0]))

print("\nVector Operations:")
print(f"v1 + v2: ({(v1_np + v2_np).x}, {(v1_np + v2_np).y}, {(v1_np + v2_np).z})")
print(f"v1 · v2 (dot product): {v1_np.dot(v2_np)}")
print(f"v1 × v2 (cross product): ({v1_np.cross(v2_np).x}, {v1_np.cross(v2_np).y}, {v1_np.cross(v2_np).z})")

# Awkward Array backend example (if available)
if HAS_AWKWARD:
    print("\n=== 3D Vector Operations with Awkward Backend ===\n")
    x_ak = ak.Array([1.0, 2.0, 3.0])
    y_ak = ak.Array([4.0, 5.0, 6.0])
    z_ak = ak.Array([7.0, 8.0, 9.0])
    vec3d_ak = Vector3D(x_ak, y_ak, z_ak)
    
    print(f"Vector components (x, y, z): ({vec3d_ak.x}, {vec3d_ak.y}, {vec3d_ak.z})")
    print(f"Magnitude: {vec3d_ak.r}")
    print(f"Cylindrical radius (rho): {vec3d_ak.rho}")
    print(f"Azimuthal angle (phi): {vec3d_ak.phi}")
    print(f"Polar angle (theta): {vec3d_ak.theta}")
    
    # Vector operations with Awkward
    v1_ak = Vector3D(ak.Array([1.0, 2.0]), ak.Array([3.0, 4.0]), ak.Array([5.0, 6.0]))
    v2_ak = Vector3D(ak.Array([7.0, 8.0]), ak.Array([9.0, 10.0]), ak.Array([11.0, 12.0]))
    
    print("\nVector Operations:")
    print(f"v1 + v2: ({(v1_ak + v2_ak).x}, {(v1_ak + v2_ak).y}, {(v1_ak + v2_ak).z})")
    print(f"v1 · v2 (dot product): {v1_ak.dot(v2_ak)}")
    print(f"v1 × v2 (cross product): ({v1_ak.cross(v2_ak).x}, {v1_ak.cross(v2_ak).y}, {v1_ak.cross(v2_ak).z})")