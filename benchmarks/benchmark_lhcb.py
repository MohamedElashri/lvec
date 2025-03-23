#!/usr/bin/env python
"""
LHCb Vector Performance Benchmark
--------------------------------
This benchmark compares the performance of the Scikit-HEP vector package
with LVec operations in a realistic LHCb physics analysis scenario.

The benchmark performs common operations in HEP analysis using B → hhh decay data:
1. Loading and preprocessing the data
2. Vector operations (addition, dot product, etc.)
3. Four-vector operations (invariant mass, transverse momentum, etc.)
4. Physics analysis with selection criteria

Both implementations are tested with identical operations to ensure a fair comparison.
"""

import os
import sys
import time
import urllib.request
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

# Import LVec package
from lvec import LVec, Vector3D
from lvec.backends import to_np

# Import competing vector package
import vector

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set LHCb style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (12, 8)

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, name=""):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"{self.name} took {self.interval:.6f} seconds")

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

def load_data(filename):
    """Load data from ROOT file and return as NumPy arrays."""
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
    
    return {
        "h1": (h1_px, h1_py, h1_pz),
        "h2": (h2_px, h2_py, h2_pz),
        "h3": (h3_px, h3_py, h3_pz)
    }

# ======== Scikit-HEP Vector Implementation ========

def vector_analysis(data, pion_mass=0.13957):
    """Perform analysis using Scikit-HEP vector package."""
    h1_px, h1_py, h1_pz = data["h1"]
    h2_px, h2_py, h2_pz = data["h2"]
    h3_px, h3_py, h3_pz = data["h3"]
    
    # Create Lorentz vectors using Scikit-HEP vector package
    # Use vector.array to create arrays of four-vectors
    h1 = vector.array({
        "px": h1_px, 
        "py": h1_py, 
        "pz": h1_pz, 
        "mass": np.full_like(h1_px, pion_mass)
    })
    
    h2 = vector.array({
        "px": h2_px, 
        "py": h2_py, 
        "pz": h2_pz, 
        "mass": np.full_like(h2_px, pion_mass)
    })
    
    h3 = vector.array({
        "px": h3_px, 
        "py": h3_py, 
        "pz": h3_pz, 
        "mass": np.full_like(h3_px, pion_mass)
    })
    
    # Basic vector operations
    total_p = h1 + h2 + h3  # Vector addition
    
    # Lorentz vector operations
    h12 = h1 + h2  # Four-vector addition
    h23 = h2 + h3
    h13 = h1 + h3
    
    # Calculate two-body invariant masses
    m12 = h12.mass
    m23 = h23.mass
    m13 = h13.mass
    
    # Selection based on kinematics
    three_body = h1 + h2 + h3
    three_body_mass = three_body.mass
    high_pt_mask = (h1.pt > 1.0) & (h2.pt > 1.0) & (h3.pt > 1.0)
    b_candidates = three_body_mass[(three_body_mass > 5.0) & (three_body_mass < 5.5)]
    
    return {
        "m12": m12, 
        "m23": m23, 
        "m13": m13,
        "three_body_mass": three_body_mass,
        "b_candidates": b_candidates,
        "high_pt_mask": high_pt_mask
    }

# ======== LVec Implementation ========

def lvec_analysis(data, pion_mass=0.13957):
    """Perform analysis using LVec package."""
    h1_px, h1_py, h1_pz = data["h1"]
    h2_px, h2_py, h2_pz = data["h2"]
    h3_px, h3_py, h3_pz = data["h3"]
    
    # Create 3-vectors
    h1_p3 = Vector3D(h1_px, h1_py, h1_pz)
    h2_p3 = Vector3D(h2_px, h2_py, h2_pz)
    h3_p3 = Vector3D(h3_px, h3_py, h3_pz)
    
    # Define energy calculation function
    def calculate_energy(p3, mass):
        return np.sqrt(p3.r**2 + mass**2)
    
    # Create Lorentz vectors (assuming pion mass)
    h1 = LVec(h1_px, h1_py, h1_pz, calculate_energy(h1_p3, pion_mass))
    h2 = LVec(h2_px, h2_py, h2_pz, calculate_energy(h2_p3, pion_mass))
    h3 = LVec(h3_px, h3_py, h3_pz, calculate_energy(h3_p3, pion_mass))
    
    # Basic vector operations
    total_p3 = h1_p3 + h2_p3 + h3_p3  # Vector addition
    mag_p3 = total_p3.r  # Vector magnitude
    
    # Basic angle calculations
    dot_12 = h1_p3.dot(h2_p3)  # Dot product
    cross_12 = h1_p3.cross(h2_p3)  # Cross product
    
    # Lorentz vector operations
    h12 = h1 + h2  # Four-vector addition
    h23 = h2 + h3
    h13 = h1 + h3
    
    # Calculate two-body invariant masses
    m12 = h12.mass
    m23 = h23.mass
    m13 = h13.mass
    
    # Selection based on kinematics
    three_body = h1 + h2 + h3
    three_body_mass = three_body.mass
    high_pt_mask = (h1.pt > 1.0) & (h2.pt > 1.0) & (h3.pt > 1.0)
    b_candidates = three_body_mass[(three_body_mass > 5.0) & (three_body_mass < 5.5)]
    
    return {
        "m12": m12, 
        "m23": m23, 
        "m13": m13,
        "three_body_mass": three_body_mass,
        "b_candidates": b_candidates,
        "high_pt_mask": high_pt_mask
    }

def plot_performance_comparison(vector_time, lvec_time, memory_vector, memory_lvec, iterations):
    """Plot performance comparison between Vector and LVec implementations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time comparison
    methods = ['Scikit-HEP Vector', 'LVec']
    times = [vector_time, lvec_time]
    colors = ['#1f77b4', '#2ca02c']
    
    ax1.bar(methods, times, color=colors)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title(f'Execution Time ({iterations} iterations)')
    for i, v in enumerate(times):
        ax1.text(i, v + 0.01, f"{v:.3f}s", ha='center')
    
    # Speedup calculation
    speedup = vector_time / lvec_time if lvec_time > 0 else float('inf')
    speedup_text = f"Speedup: {speedup:.2f}x" if speedup >= 1 else f"Slowdown: {1/speedup:.2f}x"
    ax1.text(0.5, 0.9, speedup_text, 
             transform=ax1.transAxes, ha='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Memory comparison
    memory_usage = [memory_vector, memory_lvec]
    ax2.bar(methods, memory_usage, color=colors)
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Peak Memory Usage')
    for i, v in enumerate(memory_usage):
        ax2.text(i, v + 0.5, f"{v:.1f} MB", ha='center')
    
    # Memory savings
    if memory_vector > memory_lvec:
        mem_reduction = (memory_vector - memory_lvec) / memory_vector * 100
        mem_text = f"Memory reduction: {mem_reduction:.1f}%"
    else:
        mem_increase = (memory_lvec - memory_vector) / memory_vector * 100
        mem_text = f"Memory increase: {mem_increase:.1f}%"
    
    ax2.text(0.5, 0.9, mem_text, 
             transform=ax2.transAxes, ha='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save plot to plots directory
    plot_path = os.path.join(PLOTS_DIR, "lvec_benchmark_results.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Performance comparison plot saved as '{plot_path}'")

def plot_mass_comparison(vector_results, lvec_results):
    """Plot mass distribution comparison to verify both methods produce the same physics results."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common histogram parameters
    bins = np.linspace(0, 5.5, 100)
    
    # Plot m12 distribution
    ax1.hist(vector_results["m12"], bins=bins, label='Vector', histtype='step', 
             linewidth=2, color='blue', alpha=0.7)
    ax1.hist(lvec_results["m12"], bins=bins, label='LVec', histtype='step', 
             linewidth=2, color='green', alpha=0.7)
    ax1.set_xlabel('m(h1,h2) [GeV]')
    ax1.set_ylabel('Candidates / 55 MeV')
    add_lhcb_label(ax1)
    ax1.legend()
    
    # Plot m23 distribution
    ax2.hist(vector_results["m23"], bins=bins, label='Vector', histtype='step', 
             linewidth=2, color='blue', alpha=0.7)
    ax2.hist(lvec_results["m23"], bins=bins, label='LVec', histtype='step', 
             linewidth=2, color='green', alpha=0.7)
    ax2.set_xlabel('m(h2,h3) [GeV]')
    ax2.set_ylabel('Candidates / 55 MeV')
    add_lhcb_label(ax2)
    ax2.legend()
    
    # Plot m13 distribution
    ax3.hist(vector_results["m13"], bins=bins, label='Vector', histtype='step', 
             linewidth=2, color='blue', alpha=0.7)
    ax3.hist(lvec_results["m13"], bins=bins, label='LVec', histtype='step', 
             linewidth=2, color='green', alpha=0.7)
    ax3.set_xlabel('m(h1,h3) [GeV]')
    ax3.set_ylabel('Candidates / 55 MeV')
    add_lhcb_label(ax3)
    ax3.legend()
    
    # Add common title
    fig.suptitle('Physics Results Comparison: Vector vs LVec', y=1.02)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot to plots directory
    plot_path = os.path.join(PLOTS_DIR, "physics_comparison.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Physics comparison plot saved as '{plot_path}'")

def run_benchmark(iterations=10):
    """Run the full benchmark comparing Vector and LVec implementations."""
    print(f"Running LHCb vector benchmark with {iterations} iterations per implementation...")
    print("\n1. Downloading and loading data...")
    filename = download_data()
    data = load_data(filename)
    
    print("\n2. Running Scikit-HEP Vector implementation benchmark...")
    vector_times = []
    def run_vector():
        for _ in range(iterations):
            with Timer() as t:
                results_vector = vector_analysis(data)
            vector_times.append(t.interval)
        return results_vector
    
    # Measure memory usage for Vector implementation
    memory_vector = np.max(memory_usage((run_vector, ()), interval=0.1))
    avg_time_vector = np.mean(vector_times)
    
    print(f"\nVector implementation average time: {avg_time_vector:.6f} seconds")
    print(f"Vector implementation peak memory: {memory_vector:.2f} MB")
    
    print("\n3. Running LVec implementation benchmark...")
    lvec_times = []
    def run_lvec():
        for _ in range(iterations):
            with Timer() as t:
                results_lvec = lvec_analysis(data)
            lvec_times.append(t.interval)
        return results_lvec
    
    # Measure memory usage for LVec implementation
    memory_lvec = np.max(memory_usage((run_lvec, ()), interval=0.1))
    avg_time_lvec = np.mean(lvec_times)
    
    print(f"\nLVec implementation average time: {avg_time_lvec:.6f} seconds")
    print(f"LVec implementation peak memory: {memory_lvec:.2f} MB")
    
    # Run once more to get the results for plotting
    print("\n4. Generating final results for comparison...")
    results_vector = vector_analysis(data)
    results_lvec = lvec_analysis(data)
    
    # Create visualizations
    print("\n5. Creating performance comparison plots...")
    plot_performance_comparison(avg_time_vector, avg_time_lvec, memory_vector, memory_lvec, iterations)
    plot_mass_comparison(results_vector, results_lvec)
    
    # Print summary
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Vector implementation: {avg_time_vector:.6f} seconds, {memory_vector:.2f} MB")
    print(f"LVec implementation:  {avg_time_lvec:.6f} seconds, {memory_lvec:.2f} MB")
    
    if avg_time_vector > avg_time_lvec:
        speedup = avg_time_vector / avg_time_lvec
        print(f"LVec Speedup: {speedup:.2f}x faster than Vector")
    else:
        slowdown = avg_time_lvec / avg_time_vector
        print(f"LVec Slowdown: {slowdown:.2f}x slower than Vector")
    
    if memory_vector > memory_lvec:
        mem_reduction = (memory_vector - memory_lvec) / memory_vector * 100
        print(f"Memory reduction with LVec: {mem_reduction:.1f}%")
    else:
        mem_increase = (memory_lvec - memory_vector) / memory_vector * 100
        print(f"Memory increase with LVec: {mem_increase:.1f}%")
    
    # Verify physics results match
    print("\nPhysics validation:")
    vector_mass = np.mean(results_vector["three_body_mass"])
    lvec_mass = np.mean(results_lvec["three_body_mass"])
    mass_diff_pct = abs(vector_mass - lvec_mass) / vector_mass * 100
    print(f"Mean three-body mass (Vector): {vector_mass:.4f} GeV")
    print(f"Mean three-body mass (LVec):  {lvec_mass:.4f} GeV")
    print(f"Difference: {mass_diff_pct:.6f}% (should be very close to zero)")

if __name__ == "__main__":
    run_benchmark()
