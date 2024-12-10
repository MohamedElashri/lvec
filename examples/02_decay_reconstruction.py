import uproot
import numpy as np
from lvec import LVec

def main():
    """Example of reconstructing decay kinematics."""
    # Open ROOT file
    file = uproot.open("samples/physics_data.root")
    tree = file["DecayTree"]
    
    # Read all particles - list branches explicitly
    branches = ["m_px", "m_py", "m_pz", "m_E",
               "d1_px", "d1_py", "d1_pz", "d1_E",
               "d2_px", "d2_py", "d2_pz", "d2_E"]
    data = tree.arrays(branches, library="np")
    
    # Create LVec objects
    mother = LVec(
        data["m_px"], data["m_py"],
        data["m_pz"], data["m_E"]
    )
    
    daughter1 = LVec(
        data["d1_px"], data["d1_py"],
        data["d1_pz"], data["d1_E"]
    )
    
    daughter2 = LVec(
        data["d2_px"], data["d2_py"],
        data["d2_pz"], data["d2_E"]
    )
    
    # Reconstruct and verify
    reconstructed = daughter1 + daughter2
    
    print("Decay reconstruction validation:")
    print(f"Original mass: {np.mean(mother.mass):.2f} GeV")
    print(f"Reconstructed mass: {np.mean(reconstructed.mass):.2f} GeV")
    print(f"Mass resolution: {np.std(reconstructed.mass):.3f} GeV")
    
    # Calculate some physics quantities
    dR = np.sqrt((daughter1.eta - daughter2.eta)**2 + 
                 (daughter1.phi - daughter2.phi)**2)
    
    print(f"\nAverage ΔR between daughters: {np.mean(dR):.3f}")

if __name__ == "__main__":
    main()