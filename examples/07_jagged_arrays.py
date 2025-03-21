# Example demonstrating vector operations with jagged Awkward arrays
import numpy as np
try:
    import awkward as ak
    HAS_AWKWARD = True
except ImportError:
    HAS_AWKWARD = False
    print("This example requires the awkward package. Please install it with 'pip install awkward'")
    exit(1)

from lvec import Vector2D, Vector3D

print("=== Jagged Arrays with Vector2D ===\n")

# Create jagged arrays for 2D vectors
x_jagged = ak.Array([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]])
y_jagged = ak.Array([[2.0, 3.0], [4.0, 5.0, 6.0], [7.0]])

# Create 2D vectors with jagged arrays
vec2d_jagged = Vector2D(x_jagged, y_jagged)

print("Vector components:")
print(f"x: {vec2d_jagged.x}")
print(f"y: {vec2d_jagged.y}")

# Calculate properties
print("\nVector properties:")
print(f"Magnitude (r): {vec2d_jagged.r}")
print(f"Azimuthal angle (phi): {vec2d_jagged.phi}")

# Vector operations with jagged arrays
v2d_doubled = Vector2D(2 * x_jagged, 2 * y_jagged)

print("\nVector operations:")
print(f"Original vector: ({vec2d_jagged.x}, {vec2d_jagged.y})")
print(f"Doubled vector: ({v2d_doubled.x}, {v2d_doubled.y})")
print(f"Sum: ({(vec2d_jagged + v2d_doubled).x}, {(vec2d_jagged + v2d_doubled).y})")

# Dot product with jagged arrays
dot_product = vec2d_jagged.dot(v2d_doubled)
print(f"Dot product: {dot_product}")

# Indexing into jagged arrays
print("\nIndexing into first element of each subarray:")
first_elements = vec2d_jagged[0]
print(f"First elements: ({first_elements.x}, {first_elements.y})")

print("\n=== Jagged Arrays with Vector3D ===\n")

# Create jagged arrays for 3D vectors
x_jagged = ak.Array([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]])
y_jagged = ak.Array([[2.0, 3.0], [4.0, 5.0, 6.0], [7.0]])
z_jagged = ak.Array([[3.0, 4.0], [5.0, 6.0, 7.0], [8.0]])

# Create 3D vectors with jagged arrays
vec3d_jagged = Vector3D(x_jagged, y_jagged, z_jagged)

print("Vector components:")
print(f"x: {vec3d_jagged.x}")
print(f"y: {vec3d_jagged.y}")
print(f"z: {vec3d_jagged.z}")

# Calculate properties
print("\nVector properties:")
print(f"Magnitude (r): {vec3d_jagged.r}")
print(f"Cylindrical radius (rho): {vec3d_jagged.rho}")
print(f"Azimuthal angle (phi): {vec3d_jagged.phi}")
print(f"Polar angle (theta): {vec3d_jagged.theta}")

# Vector operations with jagged arrays
v3d_doubled = Vector3D(2 * x_jagged, 2 * y_jagged, 2 * z_jagged)

print("\nVector operations:")
print(f"Original vector: ({vec3d_jagged.x}, {vec3d_jagged.y}, {vec3d_jagged.z})")
print(f"Doubled vector: ({v3d_doubled.x}, {v3d_doubled.y}, {v3d_doubled.z})")
print(f"Sum: ({(vec3d_jagged + v3d_doubled).x}, {(vec3d_jagged + v3d_doubled).y}, {(vec3d_jagged + v3d_doubled).z})")

# Dot and cross products with jagged arrays
dot_product = vec3d_jagged.dot(v3d_doubled)
cross_product = vec3d_jagged.cross(v3d_doubled)

print(f"Dot product: {dot_product}")
print(f"Cross product: ({cross_product.x}, {cross_product.y}, {cross_product.z})")

# Indexing into jagged arrays
print("\nIndexing into first element of each subarray:")
first_elements = vec3d_jagged[0]
print(f"First elements: ({first_elements.x}, {first_elements.y}, {first_elements.z})")

print("\n=== Practical Example: Particle Physics Data ===\n")

# Simulate an event with varying number of particles per event
nevents = 3

# Create jagged arrays for particle momenta in multiple events
px = ak.Array([[10.0, 20.0, 30.0], [40.0, 50.0], [60.0, 70.0, 80.0, 90.0]])
py = ak.Array([[15.0, 25.0, 35.0], [45.0, 55.0], [65.0, 75.0, 85.0, 95.0]])
pz = ak.Array([[5.0, 10.0, 15.0], [20.0, 25.0], [30.0, 35.0, 40.0, 45.0]])

# Create 3D momentum vectors
momenta = Vector3D(px, py, pz)

print(f"Number of events: {len(momenta.x)}")
print(f"Particles per event: {ak.num(momenta.x)}")

# Calculate transverse momentum (pt) for all particles
pt = momenta.rho
print(f"\nTransverse momentum (pt): {pt}")

# Calculate total momentum per event
total_px = ak.sum(momenta.x, axis=1)
total_py = ak.sum(momenta.y, axis=1)
total_pz = ak.sum(momenta.z, axis=1)

print("\nTotal momentum per event:")
print(f"px: {total_px}")
print(f"py: {total_py}")
print(f"pz: {total_pz}")

# Create total momentum vector
total_momentum = Vector3D(total_px, total_py, total_pz)
print(f"\nTotal momentum magnitude: {total_momentum.r}")

# Find particles with pt > 50
high_pt_mask = pt > 50
high_pt_particles = momenta[high_pt_mask]

print(f"\nHigh-pt particles (pt > 50): {ak.num(high_pt_particles.x, axis=0)}")
print(f"High-pt px values: {high_pt_particles.x}")