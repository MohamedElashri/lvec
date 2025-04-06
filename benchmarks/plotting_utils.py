"""
Standardized plotting utilities for lvec benchmarks.

This module provides consistent plotting styles and functions for all benchmark
scripts to ensure professional journal publication quality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Create plots directory if it doesn't exist
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define professional color palette
COLORS = {
    'lvec': '#3366CC',      # Blue
    'vector': '#DC3912',    # Red
    'lvec_cached': '#109618',  # Green
    'lvec_uncached': '#FF9900',  # Orange
    'vector2d': '#990099',  # Purple
    'vector3d': '#0099C6',  # Cyan
    'gray': '#7F7F7F',      # Gray
    'background': '#F8F8F8' # Light gray background
}

def set_publication_style():
    """Set the matplotlib style for publication-quality plots."""
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
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2
    })

def format_log_axes(ax, x_is_log=True, y_is_log=True):
    """Apply consistent formatting to log-scale axes."""
    if x_is_log:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    
    if y_is_log:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
    
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

def plot_performance_comparison(sizes, lvec_data, vector_data, title, filename, y_label="Time (ms)"):
    """
    Create a standardized performance comparison plot.
    
    Parameters:
    -----------
    sizes : array-like
        Array sizes used in the benchmark
    lvec_data : tuple
        Tuple containing (times, errors) for lvec
    vector_data : tuple
        Tuple containing (times, errors) for vector
    title : str
        Plot title
    filename : str
        Filename to save the plot (without path)
    y_label : str
        Label for y-axis
    """
    set_publication_style()
    
    lvec_times, lvec_errors = lvec_data
    vector_times, vector_errors = vector_data
    
    # Convert to milliseconds if needed
    if np.mean(lvec_times) < 0.1:
        lvec_times *= 1000
        lvec_errors *= 1000
        vector_times *= 1000
        vector_errors *= 1000
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot with error bars
    ax.errorbar(sizes, lvec_times, yerr=lvec_errors, fmt='o-', 
                label='lvec', color=COLORS['lvec'], 
                linewidth=2, markersize=6, capsize=3)
    ax.errorbar(sizes, vector_times, yerr=vector_errors, fmt='o-', 
                label='vector', color=COLORS['vector'], 
                linewidth=2, markersize=6, capsize=3)
    
    format_log_axes(ax)
    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_memory_comparison(sizes, lvec_memory, vector_memory, title, filename):
    """
    Create a standardized memory usage comparison plot.
    
    Parameters:
    -----------
    sizes : array-like
        Array sizes used in the benchmark
    lvec_memory : array-like
        Memory usage for lvec
    vector_memory : array-like
        Memory usage for vector
    title : str
        Plot title
    filename : str
        Filename to save the plot (without path)
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(sizes, lvec_memory, 'o-', label='lvec', 
            color=COLORS['lvec'], linewidth=2, markersize=6)
    ax.plot(sizes, vector_memory, 'o-', label='vector', 
            color=COLORS['vector'], linewidth=2, markersize=6)
    
    format_log_axes(ax)
    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_combined_performance(sizes, lvec_times, vector_times, lvec_memory, vector_memory, title, filename):
    """
    Create a standardized plot with both timing and memory usage.
    
    Parameters:
    -----------
    sizes : array-like
        Array sizes used in the benchmark
    lvec_times : array-like
        Execution times for lvec
    vector_times : array-like
        Execution times for vector
    lvec_memory : array-like
        Memory usage for lvec
    vector_memory : array-like
        Memory usage for vector
    title : str
        Plot title
    filename : str
        Filename to save the plot (without path)
    """
    set_publication_style()
    
    # Convert to milliseconds if needed
    if np.mean(lvec_times) < 0.1:
        lvec_times = lvec_times * 1000
        vector_times = vector_times * 1000
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Upper plot: timing comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sizes, lvec_times, 'o-', label='lvec', color=COLORS['lvec'], linewidth=2, markersize=6)
    ax1.plot(sizes, vector_times, 'o-', label='vector', color=COLORS['vector'], linewidth=2, markersize=6)
    format_log_axes(ax1)
    ax1.set_ylabel('Time per operation (ms)', fontsize=12)
    ax1.set_title(title, fontsize=14, pad=15)
    ax1.legend(fontsize=10)
    
    # Bottom plot: memory usage
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(sizes, lvec_memory, 'o-', label='lvec', color=COLORS['lvec'], linewidth=2, markersize=6)
    ax2.plot(sizes, vector_memory, 'o-', label='vector', color=COLORS['vector'], linewidth=2, markersize=6)
    format_log_axes(ax2)
    ax2.set_xlabel('Array Size', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_operations_grid(sizes, all_results, operations, title, filename):
    """
    Create a grid of plots for multiple operations.
    
    Parameters:
    -----------
    sizes : array-like
        Array sizes used in the benchmark
    all_results : dict
        Dictionary with operation names as keys and results as values
    operations : list
        List of operation names to plot
    title : str
        Plot title
    filename : str
        Filename to save the plot (without path)
    """
    set_publication_style()
    
    n_ops = len(operations)
    n_cols = min(3, n_ops)
    n_rows = (n_ops + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(15, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    
    for idx, operation in enumerate(operations):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        results = all_results[operation]
        
        # Extract data
        lvec_times = np.array([r['lvec']['time'] for r in results]) * 1000  # to ms
        vector_times = np.array([r['vector']['time'] for r in results]) * 1000
        
        # Timing plot
        ax.plot(sizes, lvec_times, 'o-', label='lvec', color=COLORS['lvec'], 
                linewidth=2, markersize=6)
        ax.plot(sizes, vector_times, 'o-', label='vector', color=COLORS['vector'], 
                linewidth=2, markersize=6)
        
        format_log_axes(ax)
        ax.set_xlabel('Array Size', fontsize=10)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title(operation.replace('_', ' ').title(), fontsize=12)
        ax.legend(fontsize=10)
    
    # Remove any empty subplots
    for idx in range(len(operations), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(fig.add_subplot(gs[row, col]))
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_vector_types_comparison(sizes, results, vector_types, operations, title, filename):
    """
    Create a comparison plot for different vector types.
    
    Parameters:
    -----------
    sizes : array-like
        Array sizes used in the benchmark
    results : dict
        Nested dictionary with vector types, operations, and results
    vector_types : list
        List of vector types to plot
    operations : list
        List of operations to plot
    title : str
        Plot title
    filename : str
        Filename to save the plot (without path)
    """
    set_publication_style()
    
    n_ops = len(operations)
    n_cols = min(3, n_ops)
    n_rows = (n_ops + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    
    # Assign colors to vector types
    vector_colors = {
        'LVec': COLORS['lvec'],
        'Vector2D': COLORS['vector2d'],
        'Vector3D': COLORS['vector3d'],
        'Scikit Vector': COLORS['vector']
    }
    
    for op_idx, operation in enumerate(operations):
        row = op_idx // n_cols
        col = op_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        for vtype in vector_types:
            if vtype in results and operation in results[vtype]:
                times = np.array([results[vtype][size][operation]["time"] for size in sizes]) * 1000
                ax.plot(sizes, times, 'o-', label=vtype, 
                        color=vector_colors.get(vtype, COLORS['gray']),
                        linewidth=2, markersize=6)
        
        format_log_axes(ax)
        ax.set_xlabel('Array Size', fontsize=10)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title(operation.replace('_', ' ').title(), fontsize=12)
        ax.legend(fontsize=10)
    
    # Remove any empty subplots
    for idx in range(len(operations), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(fig.add_subplot(gs[row, col]))
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_physics_results(data1, data2, labels, title, filename):
    """
    Create a standardized plot for physics results comparison.
    
    Parameters:
    -----------
    data1, data2 : dict
        Dictionaries containing physics results
    labels : tuple
        Tuple of (label1, label2) for the legend
    title : str
        Plot title
    filename : str
        Filename to save the plot (without path)
    """
    set_publication_style()
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    # Define histogram properties
    hist_props = {
        'm12': {'bins': 50, 'range': (0, 5), 'xlabel': r'$m_{12}$ (GeV/$c^2$)'},
        'm23': {'bins': 50, 'range': (0, 5), 'xlabel': r'$m_{23}$ (GeV/$c^2$)'},
        'm13': {'bins': 50, 'range': (0, 5), 'xlabel': r'$m_{13}$ (GeV/$c^2$)'},
        'three_body_mass': {'bins': 50, 'range': (4.5, 6), 'xlabel': r'$m_{123}$ (GeV/$c^2$)'}
    }
    
    # Plot histograms
    for i, (key, props) in enumerate(hist_props.items()):
        if key in data1 and key in data2:
            axs[i].hist(data1[key], bins=props['bins'], range=props['range'], 
                      alpha=0.5, label=labels[0], color=COLORS['lvec'])
            axs[i].hist(data2[key], bins=props['bins'], range=props['range'], 
                      alpha=0.5, label=labels[1], color=COLORS['vector'])
            axs[i].set_xlabel(props['xlabel'], fontsize=12)
            axs[i].set_ylabel('Counts', fontsize=12)
            axs[i].legend(fontsize=10)
            axs[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()
