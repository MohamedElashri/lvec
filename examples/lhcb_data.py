import os
import sys
import urllib.request
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from lvec import LVec 
from lvec.backends import to_np  

# Set LHCb style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (12, 8)

def add_lhcb_label(ax, x=0.85, y=0.85):
    """Add the LHCb label to the plot"""
    ax.text(x, y, "LHCb", fontname="Times New Roman",
            fontsize=16, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Download the ROOT file if not present
def download_data():
    url = "https://opendata.cern.ch/record/4900/files/B2HHH_MagnetDown.root"
    filename = "B2HHH_MagnetDown.root"
    
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    return filename

def analyze_data(filename):
    # Open the ROOT file
    file = uproot.open(filename)
    tree = file["DecayTree"]
    
    # Get the momentum components
    data = tree.arrays(["H1_PX", "H1_PY", "H1_PZ",
                       "H2_PX", "H2_PY", "H2_PZ",
                       "H3_PX", "H3_PY", "H3_PZ"])
    
    # Convert awkward arrays to numpy arrays
    h1_px = to_np(data["H1_PX"])
    h1_py = to_np(data["H1_PY"])
    h1_pz = to_np(data["H1_PZ"])
    
    h2_px = to_np(data["H2_PX"])
    h2_py = to_np(data["H2_PY"])
    h2_pz = to_np(data["H2_PZ"])
    
    h3_px = to_np(data["H3_PX"])
    h3_py = to_np(data["H3_PY"])
    h3_pz = to_np(data["H3_PZ"])
    
    # Calculate energies assuming pion mass (139.57 MeV)
    pion_mass = 0.13957  # GeV
    
    def calculate_energy(px, py, pz):
        p2 = px**2 + py**2 + pz**2
        return np.sqrt(p2 + pion_mass**2)
    
    # Create Lorentz vectors for each particle using LVEC
    h1 = LVec(h1_px, h1_py, h1_pz, 
              calculate_energy(h1_px, h1_py, h1_pz))
    h2 = LVec(h2_px, h2_py, h2_pz,
              calculate_energy(h2_px, h2_py, h2_pz))
    h3 = LVec(h3_px, h3_py, h3_pz,
              calculate_energy(h3_px, h3_py, h3_pz))
    
    # Calculate two-body invariant masses
    m12 = (h1 + h2).mass
    m23 = (h2 + h3).mass
    m13 = (h1 + h3).mass
    
    return m12, m23, m13

def plot_mass_distributions(m12, m23, m13):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common histogram parameters
    bins = np.linspace(0, 5.5, 100)
    hist_kwargs = dict(histtype='step', linewidth=2, color='blue')
    
    # Plot m12 distribution
    ax1.hist(m12, bins=bins, label='m(h1,h2)', **hist_kwargs)
    ax1.set_xlabel('m(h1,h2) [GeV]')
    ax1.set_ylabel('Candidates / 55 MeV')
    add_lhcb_label(ax1)
    ax1.legend()
    
    # Plot m23 distribution
    ax2.hist(m23, bins=bins, label='m(h2,h3)', **hist_kwargs)
    ax2.set_xlabel('m(h2,h3) [GeV]')
    ax2.set_ylabel('Candidates / 55 MeV')
    add_lhcb_label(ax2)
    ax2.legend()
    
    # Plot m13 distribution
    ax3.hist(m13, bins=bins, label='m(h1,h3)', **hist_kwargs)
    ax3.set_xlabel('m(h1,h3) [GeV]')
    ax3.set_ylabel('Candidates / 55 MeV')
    add_lhcb_label(ax3)
    ax3.legend()
    
    # Add common title
    fig.suptitle('Two-Body Invariant Mass Distributions', y=1.02)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('mass_distributions.pdf', bbox_inches='tight')
    plt.close()

def main():
    # Download and analyze data
    filename = download_data()
    m12, m23, m13 = analyze_data(filename)
    
    # Print some basic statistics
    print("\nTwo-body invariant mass statistics:")
    print(f"m12 mean: {np.mean(m12):.2f} GeV")
    print(f"m23 mean: {np.mean(m23):.2f} GeV")
    print(f"m13 mean: {np.mean(m13):.2f} GeV")
    
    # Create mass distribution plots
    plot_mass_distributions(m12, m23, m13)
    print("\nMass distribution plots have been saved as 'mass_distributions.pdf'")

if __name__ == "__main__":
    main()