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

# Set modern scientific plotting style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 7),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'axes.grid': False,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

# Define professional color palette
COLORS = {
    'vector': '#4472C4',  # Microsoft blue
    'lvec': '#70AD47',    # Microsoft green
    'highlight': '#ED7D31', # Microsoft orange
    'accent1': '#5B9BD5',  # Light blue
    'accent2': '#FFC000',  # Gold
    'gray': '#7F7F7F',     # Gray
    'background': '#F2F2F2' # Light gray background
}

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
    # Calculate speedup and memory reduction
    speedup = vector_time / lvec_time if lvec_time > 0 else float('inf')
    mem_reduction = (memory_vector - memory_lvec) / memory_vector * 100 if memory_vector > memory_lvec else 0
    
    # Create figure with a light gray background
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Create a 2x2 grid for the plots
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1], 
                         hspace=0.3, wspace=0.3)
    
    # Time comparison - horizontal bar chart (more impactful)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.patch.set_facecolor('white')
    
    # Create horizontal bar chart
    methods = ['Scikit-HEP\nVector', 'LVec']
    times = [vector_time, lvec_time]
    colors = [COLORS['vector'], COLORS['lvec']]
    
    y_pos = np.arange(len(methods))
    ax1.barh(y_pos, times, color=colors, height=0.5, 
            edgecolor='white', linewidth=0.5)
    
    # Add value labels inside bars
    for i, v in enumerate(times):
        ax1.text(v/2, i, f"{v:.3f}s", 
                ha='center', va='center', color='white', fontweight='bold')
    
    # Customize time plot
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontweight='bold')
    ax1.set_xlabel('Execution Time (seconds)', fontweight='bold')
    ax1.set_title('Execution Time Comparison', fontweight='bold', pad=15)
    ax1.invert_yaxis()  # Puts Vector at the top
    
    # Memory comparison - horizontal bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.patch.set_facecolor('white')
    
    memory_usage = [memory_vector, memory_lvec]
    ax2.barh(y_pos, memory_usage, color=colors, height=0.5,
            edgecolor='white', linewidth=0.5)
    
    # Add value labels inside bars
    for i, v in enumerate(memory_usage):
        ax2.text(v/2, i, f"{v:.1f} MB", 
                ha='center', va='center', color='white', fontweight='bold')
    
    # Customize memory plot
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontweight='bold')
    ax2.set_xlabel('Peak Memory Usage (MB)', fontweight='bold')
    ax2.set_title('Memory Usage Comparison', fontweight='bold', pad=15)
    ax2.invert_yaxis()  # Puts Vector at the top
    
    # Speedup visualization - gauge chart
    ax3 = fig.add_subplot(gs[0, 1], polar=True)
    ax3.patch.set_facecolor('white')
    
    # Create gauge chart for speedup
    if speedup >= 1:
        # Normalize speedup to a 0-100 scale for gauge
        # Cap at 3x for visualization purposes
        norm_speedup = min(speedup, 3) / 3 * 100
        gauge_color = COLORS['lvec']
        speedup_text = f"{speedup:.2f}×\nfaster"
    else:
        norm_speedup = min(1/speedup, 3) / 3 * 100
        gauge_color = COLORS['vector']
        speedup_text = f"{1/speedup:.2f}×\nslower"
    
    # Background ring (gray)
    ax3.barh(0, 100, left=0, height=0.6, color=COLORS['gray'], alpha=0.3)
    # Foreground ring (colored)
    ax3.barh(0, norm_speedup, left=0, height=0.6, color=gauge_color)
    
    # Remove ticks and labels
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)  # Clockwise
    
    # Set limits for semi-circle
    ax3.set_thetamin(0)
    ax3.set_thetamax(180)
    
    # Add text in the middle
    ax3.text(0, -0.2, speedup_text, ha='center', va='center', 
             fontsize=14, fontweight='bold', color='black')
    
    # Add title
    ax3.set_title('Speed Comparison', fontweight='bold', pad=15)
    
    # Memory savings visualization - gauge chart
    ax4 = fig.add_subplot(gs[1, 1], polar=True)
    ax4.patch.set_facecolor('white')
    
    # Create gauge chart for memory savings
    if mem_reduction > 0:
        # Normalize memory reduction to a 0-100 scale for gauge
        # Cap at 50% for visualization purposes
        norm_mem = min(mem_reduction, 50) / 50 * 100
        mem_color = COLORS['lvec']
        mem_text = f"{mem_reduction:.1f}%\nless memory"
    else:
        mem_increase = (memory_lvec - memory_vector) / memory_vector * 100
        norm_mem = min(mem_increase, 50) / 50 * 100
        mem_color = COLORS['vector']
        mem_text = f"{mem_increase:.1f}%\nmore memory"
    
    # Background ring (gray)
    ax4.barh(0, 100, left=0, height=0.6, color=COLORS['gray'], alpha=0.3)
    # Foreground ring (colored)
    ax4.barh(0, norm_mem, left=0, height=0.6, color=mem_color)
    
    # Remove ticks and labels
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)  # Clockwise
    
    # Set limits for semi-circle
    ax4.set_thetamin(0)
    ax4.set_thetamax(180)
    
    # Add text in the middle
    ax4.text(0, -0.2, mem_text, ha='center', va='center', 
             fontsize=14, fontweight='bold', color='black')
    
    # Add title
    ax4.set_title('Memory Comparison', fontweight='bold', pad=15)
    
    # Add main title
    fig.suptitle(f'LVec Performance Benchmark ({iterations} iterations)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add benchmark details as a footer
    plt.figtext(0.5, 0.01, 
                f"Benchmark: B→hhh decay analysis | Date: {time.strftime('%Y-%m-%d')}",
                ha="center", fontsize=9, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Save plot in PDF format
    plot_path = os.path.join(PLOTS_DIR, "lvec_benchmark_results.pdf")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    
    print(f"Performance comparison plot saved as '{plot_path}'")

def plot_physics_results(vector_results, lvec_results):
    """Plot comprehensive physics results comparison."""
    # Create figure with a light gray background
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Create a 2x2 grid for the plots
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                         hspace=0.3, wspace=0.3)
    
    # Function to create a physics plot with inset ratio
    def create_physics_plot(ax, vector_data, lvec_data, title, xlabel, bins=100, range_min=0, range_max=5.5):
        ax.patch.set_facecolor('white')
        
        # Create bins
        bins = np.linspace(range_min, range_max, bins)
        bin_width = bins[1] - bins[0]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate histograms
        vector_hist, _ = np.histogram(vector_data, bins=bins)
        lvec_hist, _ = np.histogram(lvec_data, bins=bins)
        
        # Plot histograms
        ax.hist(vector_data, bins=bins, label='Vector', histtype='step', 
               linewidth=2, color=COLORS['vector'], alpha=0.9)
        ax.hist(lvec_data, bins=bins, label='LVec', histtype='step', 
               linewidth=2, color=COLORS['lvec'], alpha=0.9)
        
        # Add shaded area under curves
        ax.hist(vector_data, bins=bins, histtype='stepfilled', 
               linewidth=0, color=COLORS['vector'], alpha=0.1)
        ax.hist(lvec_data, bins=bins, histtype='stepfilled', 
               linewidth=0, color=COLORS['lvec'], alpha=0.1)
        
        # Calculate ratio for inset ratio plot
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(lvec_hist, vector_hist)
            ratio[~np.isfinite(ratio)] = 1.0  # Replace inf/NaN with 1.0
        
        # Create inset axis for ratio
        inset_ax = ax.inset_axes([0.6, 0.02, 0.38, 0.25])
        inset_ax.patch.set_facecolor('white')
        inset_ax.plot(bin_centers, ratio, '-', linewidth=1.5, color=COLORS['highlight'])
        inset_ax.axhline(y=1.0, color=COLORS['gray'], linestyle='--', alpha=0.7, linewidth=1)
        
        # Set y-range for ratio plot
        inset_ax.set_ylim(0.95, 1.05)
        inset_ax.set_ylabel('LVec/Vector', fontsize=8)
        inset_ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Remove spines
        for spine in ['top', 'right']:
            inset_ax.spines[spine].set_visible(False)
        
        # Formatting main plot
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(f'Candidates / {bin_width:.3f} GeV/c²', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.legend(frameon=True, fancybox=True, framealpha=0.7, loc='upper right')
        
        # Add main title
        fig.suptitle('Physics Results Comparison: Vector vs LVec', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    # Plot m12 distribution
    ax1 = fig.add_subplot(gs[0, 0])
    create_physics_plot(
        ax1, 
        vector_results["m12"], lvec_results["m12"],
        'Two-Body Mass m(h1,h2)', 'm(h1,h2) [GeV/c²]'
    )
    
    # Plot m23 distribution
    ax2 = fig.add_subplot(gs[0, 1])
    create_physics_plot(
        ax2, 
        vector_results["m23"], lvec_results["m23"],
        'Two-Body Mass m(h2,h3)', 'm(h2,h3) [GeV/c²]'
    )
    
    # Plot m13 distribution
    ax3 = fig.add_subplot(gs[1, 0])
    create_physics_plot(
        ax3, 
        vector_results["m13"], lvec_results["m13"],
        'Two-Body Mass m(h1,h3)', 'm(h1,h3) [GeV/c²]'
    )
    
    # Plot three-body mass distribution
    ax4 = fig.add_subplot(gs[1, 1])
    create_physics_plot(
        ax4, 
        vector_results["three_body_mass"], lvec_results["three_body_mass"],
        'Three-Body Mass m(h1,h2,h3)', 'm(h1,h2,h3) [GeV/c²]',
        bins=80, range_min=4.5, range_max=6.0
    )
    
    # Highlight B meson mass region in three-body plot
    b_mass = 5.279  # GeV/c²
    b_width = 0.1   # Approximate width to highlight
    ax4.axvspan(b_mass - b_width, b_mass + b_width, alpha=0.2, color=COLORS['highlight'])
    ax4.axvline(b_mass, color=COLORS['highlight'], linestyle='--', alpha=0.7)
    ax4.text(b_mass, ax4.get_ylim()[1]*0.95, 'B⁰', ha='center', va='top', 
             color=COLORS['highlight'], fontweight='bold')
    
    # Add benchmark details as a footer
    plt.figtext(0.5, 0.01, 
                f"B→hhh decay analysis | Date: {time.strftime('%Y-%m-%d')}",
                ha="center", fontsize=9, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Save plot in PDF format
    plot_path = os.path.join(PLOTS_DIR, "physics_comparison.pdf")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    
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
    plot_physics_results(results_vector, results_lvec)
    
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
