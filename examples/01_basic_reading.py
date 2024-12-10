import uproot
import numpy as np
from lvec import LVec

def main():
    """Basic example of reading ROOT file and creating LVec objects."""
    # Open ROOT file
    file = uproot.open("samples/physics_data.root")
    tree = file["DecayTree"]
    
    # Read mother particle data
    mother_data = tree.arrays(["m_px", "m_py", "m_pz", "m_E"], library="np")
    
    # Create LVec object
    mother = LVec(
        mother_data["m_px"],
        mother_data["m_py"],
        mother_data["m_pz"],
        mother_data["m_E"]
    )
    
    # Calculate and print some basic properties
    print("Mother particle properties:")
    print(f"Average pt: {np.mean(mother.pt):.2f} GeV")
    print(f"Average mass: {np.mean(mother.mass):.2f} GeV")
    print(f"Average eta: {np.mean(mother.eta):.2f}")

if __name__ == "__main__":
    main()