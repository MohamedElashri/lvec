import uproot
import numpy as np
from lvec import LVec

def main():
    """Example of making physics selections using LVec."""
    # Open ROOT file
    file = uproot.open("samples/physics_data.root")
    tree = file["DecayTree"]
    
    # Read data - list branches explicitly
    branches = ["d1_px", "d1_py", "d1_pz", "d1_E",
               "d2_px", "d2_py", "d2_pz", "d2_E"]
    data = tree.arrays(branches, library="np")
    
    # Create LVec objects for daughters
    muon1 = LVec(
        data["d1_px"], data["d1_py"],
        data["d1_pz"], data["d1_E"]
    )
    
    muon2 = LVec(
        data["d2_px"], data["d2_py"],
        data["d2_pz"], data["d2_E"]
    )
    
    # Make physics selections
    # Both muons should have pt > 20 GeV and |eta| < 2.4
    mask = (muon1.pt > 20) & (muon2.pt > 20) & \
           (np.abs(muon1.eta) < 2.4) & (np.abs(muon2.eta) < 2.4)
    
    # Apply selections
    muon1_selected = muon1[mask]
    muon2_selected = muon2[mask]
    
    # Reconstruct Z boson from selected muons
    Z = muon1_selected + muon2_selected
    
    print("Selection results:")
    print(f"Total events: {len(mask)}")
    print(f"Selected events: {np.sum(mask)}")
    print(f"\nSelected Z properties:")
    print(f"Mass mean: {np.mean(Z.mass):.2f} ± {np.std(Z.mass):.2f} GeV")
    print(f"pT mean: {np.mean(Z.pt):.2f} ± {np.std(Z.pt):.2f} GeV")

if __name__ == "__main__":
    main()