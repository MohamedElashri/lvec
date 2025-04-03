#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example demonstrating the use of reference frames in lvec.

This example shows how to:
1. Create different reference frames
2. Transform vectors between lab and center-of-mass frames
3. Work with collision systems
4. Verify momentum conservation in reference frame transformations
"""

import numpy as np
from lvec import LVec, Frame

print("LVEC REFERENCE FRAMES EXAMPLE")
print("=============================\n")

# Create a simple lab frame (at rest)
lab_frame = Frame.rest(name="lab")
print(f"Created lab frame: {lab_frame}")

# Create two colliding particles in the lab frame
# Particle 1: moving along +z axis
p1_lab = LVec(px=0.0, py=0.0, pz=20.0, E=25.0)
# Particle 2: moving along -z axis
p2_lab = LVec(px=0.0, py=0.0, pz=-15.0, E=20.0)

print("\n=== Particles in lab frame ===")
print(f"Particle 1: px={p1_lab.px:.3f}, py={p1_lab.py:.3f}, pz={p1_lab.pz:.3f}, E={p1_lab.E:.3f}, mass={p1_lab.mass:.3f}")
print(f"Particle 2: px={p2_lab.px:.3f}, py={p2_lab.py:.3f}, pz={p2_lab.pz:.3f}, E={p2_lab.E:.3f}, mass={p2_lab.mass:.3f}")

# Calculate the center-of-mass energy
total_lab = p1_lab + p2_lab
print(f"\nTotal momentum in lab: px={total_lab.px:.3f}, py={total_lab.py:.3f}, pz={total_lab.pz:.3f}")
print(f"Total energy in lab: E={total_lab.E:.3f}")
print(f"Center-of-mass energy (invariant mass): √s={total_lab.mass:.3f}")

# Create a center-of-mass frame for this collision system
cm_frame = Frame.from_lvec(total_lab, name="center-of-mass")
print(f"\nCreated CM frame: {cm_frame}")

# Extract beta values, converting to Python scalars if they're arrays
beta_x = cm_frame.beta_x.item() if hasattr(cm_frame.beta_x, 'item') else cm_frame.beta_x
beta_y = cm_frame.beta_y.item() if hasattr(cm_frame.beta_y, 'item') else cm_frame.beta_y
beta_z = cm_frame.beta_z.item() if hasattr(cm_frame.beta_z, 'item') else cm_frame.beta_z
gamma = cm_frame.gamma.item() if hasattr(cm_frame.gamma, 'item') else cm_frame.gamma

print(f"CM frame velocity (β): ({beta_x:.6f}, {beta_y:.6f}, {beta_z:.6f})")
print(f"CM frame gamma (γ): {gamma:.6f}")
print(f"Velocity magnitude |β|: {np.sqrt(beta_x**2 + beta_y**2 + beta_z**2):.6f}")

# Manually calculate velocity to verify
manual_beta_z = -total_lab.pz / total_lab.E
print(f"Manual β calculation: {manual_beta_z:.6f}")

# Transform the particles to the center-of-mass frame
print("\n=== Testing transform_frame method ===")
p1_cm = p1_lab.transform_frame(lab_frame, cm_frame)
p2_cm = p2_lab.transform_frame(lab_frame, cm_frame)

print("\n=== Particles in center-of-mass frame ===")
print(f"Particle 1: px={p1_cm.px:.6f}, py={p1_cm.py:.6f}, pz={p1_cm.pz:.6f}, E={p1_cm.E:.6f}, mass={p1_cm.mass:.6f}")
print(f"Particle 2: px={p2_cm.px:.6f}, py={p2_cm.py:.6f}, pz={p2_cm.pz:.6f}, E={p2_cm.E:.6f}, mass={p2_cm.mass:.6f}")

# Verify momentum conservation in CM frame
total_cm = p1_cm + p2_cm
print(f"\nTotal momentum in CM: px={total_cm.px:.6f}, py={total_cm.py:.6f}, pz={total_cm.pz:.6f}")
print(f"Total energy in CM: E={total_cm.E:.6f}")
print(f"Invariant mass in CM: √s={total_cm.mass:.6f}")

# Compare with direct boost method
print("\n=== Testing direct boost method ===")
p1_cm_direct = p1_lab.boost(-beta_x, -beta_y, -beta_z)
p2_cm_direct = p2_lab.boost(-beta_x, -beta_y, -beta_z)
total_cm_direct = p1_cm_direct + p2_cm_direct

print(f"Direct boost - p1: px={p1_cm_direct.px:.6f}, py={p1_cm_direct.py:.6f}, pz={p1_cm_direct.pz:.6f}")
print(f"Direct boost - p2: px={p2_cm_direct.px:.6f}, py={p2_cm_direct.py:.6f}, pz={p2_cm_direct.pz:.6f}")
print(f"Direct boost - Total momentum: px={total_cm_direct.px:.6f}, py={total_cm_direct.py:.6f}, pz={total_cm_direct.pz:.6f}")

# Testing to_frame method
print("\n=== Testing to_frame method ===")
p1_cm2 = p1_lab.to_frame(cm_frame)
p2_cm2 = p2_lab.to_frame(cm_frame)
total_cm2 = p1_cm2 + p2_cm2

print(f"to_frame - p1: px={p1_cm2.px:.6f}, py={p1_cm2.py:.6f}, pz={p1_cm2.pz:.6f}")
print(f"to_frame - p2: px={p2_cm2.px:.6f}, py={p2_cm2.py:.6f}, pz={p2_cm2.pz:.6f}")
print(f"to_frame - Total momentum: px={total_cm2.px:.6f}, py={total_cm2.py:.6f}, pz={total_cm2.pz:.6f}")

# Verify mass invariance
print("\n=== Mass invariance verification ===")
print(f"Particle 1 mass - lab: {p1_lab.mass:.6f}, CM: {p1_cm.mass:.6f}, difference: {abs(p1_lab.mass - p1_cm.mass):.9f}")
print(f"Particle 2 mass - lab: {p2_lab.mass:.6f}, CM: {p2_cm.mass:.6f}, difference: {abs(p2_lab.mass - p2_cm.mass):.9f}")
print(f"Total mass - lab: {total_lab.mass:.6f}, CM: {total_cm.mass:.6f}, difference: {abs(total_lab.mass - total_cm.mass):.9f}")

# Let's create a decay product in the CM frame
decay_angle = np.pi/4  # 45 degrees
decay_p = 10.0  # momentum magnitude
decay_px = decay_p * np.sin(decay_angle)
decay_py = 0.0
decay_pz = decay_p * np.cos(decay_angle)
decay_mass = 0.5
decay_E = np.sqrt(decay_px**2 + decay_py**2 + decay_pz**2 + decay_mass**2)

# Create the decay product in the CM frame
decay_cm = LVec(px=decay_px, py=decay_py, pz=decay_pz, E=decay_E)
print("\n=== Decay product in center-of-mass frame ===")
print(f"Decay: px={decay_cm.px:.3f}, py={decay_cm.py:.3f}, pz={decay_cm.pz:.3f}, E={decay_cm.E:.3f}")
print(f"Decay angle in CM: {np.arctan2(decay_cm.px, decay_cm.pz) * 180/np.pi:.1f} degrees")

# Transform the decay product back to the lab frame
decay_lab = decay_cm.transform_frame(cm_frame, lab_frame)
print("\n=== Decay product in lab frame ===")
print(f"Decay: px={decay_lab.px:.3f}, py={decay_lab.py:.3f}, pz={decay_lab.pz:.3f}, E={decay_lab.E:.3f}")
print(f"Decay angle in lab: {np.arctan2(decay_lab.px, decay_lab.pz) * 180/np.pi:.1f} degrees")
print(f"Decay mass invariance: CM={decay_cm.mass:.6f}, lab={decay_lab.mass:.6f}, difference: {abs(decay_cm.mass - decay_lab.mass):.9f}")

# Example using the alternative center_of_mass method
print("\n=== Frame.center_of_mass method ===")
cm_frame2 = Frame.center_of_mass([p1_lab, p2_lab], name="cm-alternative")
print(f"CM frame from particle list: {cm_frame2}")
beta_matches = (
    abs(cm_frame.beta_x - cm_frame2.beta_x) < 1e-10 and 
    abs(cm_frame.beta_y - cm_frame2.beta_y) < 1e-10 and 
    abs(cm_frame.beta_z - cm_frame2.beta_z) < 1e-10
)
print(f"Velocities match original CM frame: {beta_matches}")
