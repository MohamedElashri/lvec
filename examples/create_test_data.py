import numpy as np
import uproot

# Generate the particle decay data
n_events = 1000

# Mother particle properties
m_px = np.zeros(n_events, dtype=np.float32)
m_py = np.zeros(n_events, dtype=np.float32)
m_pz = np.zeros(n_events, dtype=np.float32)
m_E = np.zeros(n_events, dtype=np.float32)

# Daughter particle 1 properties
d1_px = np.zeros(n_events, dtype=np.float32)
d1_py = np.zeros(n_events, dtype=np.float32)
d1_pz = np.zeros(n_events, dtype=np.float32)
d1_E = np.zeros(n_events, dtype=np.float32)

# Daughter particle 2 properties
d2_px = np.zeros(n_events, dtype=np.float32)
d2_py = np.zeros(n_events, dtype=np.float32)
d2_pz = np.zeros(n_events, dtype=np.float32)
d2_E = np.zeros(n_events, dtype=np.float32)

# Generate random data
mass = 91.2  # Z mass
for i in range(n_events):
    # Generate random momenta for the mother particle (Z boson)
    pt = np.random.uniform(20, 100)
    eta = np.random.uniform(-2.5, 2.5)
    phi = np.random.uniform(-np.pi, np.pi)
    
    # Calculate components for mother particle
    m_px[i] = pt * np.cos(phi)
    m_py[i] = pt * np.sin(phi)
    m_pz[i] = pt * np.sinh(eta)
    m_E[i] = np.sqrt(pt**2 * np.cosh(eta)**2 + mass**2)
    
    # Generate daughter particles with momentum conservation
    fraction = np.random.uniform(0.3, 0.7)
    d1_px[i] = fraction * m_px[i]
    d1_py[i] = fraction * m_py[i]
    d1_pz[i] = fraction * m_pz[i]
    d1_E[i] = np.sqrt(d1_px[i]**2 + d1_py[i]**2 + d1_pz[i]**2 + 0.105**2)  # muon mass
    
    d2_px[i] = (1 - fraction) * m_px[i]
    d2_py[i] = (1 - fraction) * m_py[i]
    d2_pz[i] = (1 - fraction) * m_pz[i]
    d2_E[i] = np.sqrt(d2_px[i]**2 + d2_py[i]**2 + d2_pz[i]**2 + 0.105**2)

# Save the data to a ROOT file using uproot
with uproot.recreate("samples/physics_data.root") as f:
    f["DecayTree"] = {
        "m_px": m_px,
        "m_py": m_py,
        "m_pz": m_pz,
        "m_E": m_E,
        "d1_px": d1_px,
        "d1_py": d1_py,
        "d1_pz": d1_pz,
        "d1_E": d1_E,
        "d2_px": d2_px,
        "d2_py": d2_py,
        "d2_pz": d2_pz,
        "d2_E": d2_E,
    }

print("ROOT file created successfully with uproot.")
