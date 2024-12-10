import uproot
import numpy as np
from lvec import LVec

def main():
    """Example of boosting particles to different reference frames."""
    # Open ROOT file
    file = uproot.open("samples/physics_data.root")
    tree = file["DecayTree"]
    
    # Read mother and daughters - list branches explicitly
    branches = ["m_px", "m_py", "m_pz", "m_E",
               "d1_px", "d1_py", "d1_pz", "d1_E",
               "d2_px", "d2_py", "d2_pz", "d2_E"]
    data = tree.arrays(branches, library="np")
    
    # Create LVec objects
    Z = LVec(
        data["m_px"], data["m_py"],
        data["m_pz"], data["m_E"]
    )
    
    muon1 = LVec(
        data["d1_px"], data["d1_py"],
        data["d1_pz"], data["d1_E"]
    )
    
    muon2 = LVec(
        data["d2_px"], data["d2_py"],
        data["d2_pz"], data["d2_E"]
    )
    
    # Boost to Z rest frame
    # Calculate boost vector components
    beta_x = -Z.px/Z.E
    beta_y = -Z.py/Z.E
    beta_z = -Z.pz/Z.E
    
    # Boost muons
    muon1_rest = muon1.boost(beta_x, beta_y, beta_z)
    muon2_rest = muon2.boost(beta_x, beta_y, beta_z)
    
    # Verify we're in rest frame
    Z_rest = muon1_rest + muon2_rest
    
    print("Rest frame validation:")
    print(f"Original Z pT: {np.mean(Z.pt):.2f} GeV")
    print(f"Boosted Z pT: {np.mean(Z_rest.pt):.2f} GeV")
    print(f"Original Z mass: {np.mean(Z.mass):.2f} GeV")
    print(f"Boosted Z mass: {np.mean(Z_rest.mass):.2f} GeV")
    
    # Calculate angular distributions in rest frame
    cos_theta = muon1_rest.pz/muon1_rest.p
    
    print(f"\nMean cos(theta) in rest frame: {np.mean(cos_theta):.3f}")

if __name__ == "__main__":
    main()