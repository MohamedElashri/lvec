#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark for Lorentz boost operations in lvec.
This benchmark compares the performance of axis-specific boosts vs general boosts
between lvec and the vector package, with focus on backend optimizations.
"""

import numpy as np
import timeit
import matplotlib.pyplot as plt
import tracemalloc
import gc
import time
from functools import partial
import os
from lvec import LVec
import vector  # Comparison library
from plotting_utils import set_publication_style, COLORS

def measure_memory_usage(operation, n_repeats=5):
    """Measure memory usage for an operation."""
    memory_usages = []
    for _ in range(n_repeats):
        gc.collect()
        tracemalloc.start()
        result = operation()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak / 1024 / 1024  # Convert to MB
        memory_usages.append(memory_usage)
        del result
    return np.mean(memory_usages)

def generate_test_data(size):
    """Generate random 4-vectors for testing."""
    px = np.random.normal(0, 10, size)
    py = np.random.normal(0, 10, size)
    pz = np.random.normal(0, 10, size)
    E = np.sqrt(px**2 + py**2 + pz**2 + (0.105)**2)  # mass of muon
    return px, py, pz, E

def measure_single_timing(operation, n_repeats=10):
    """Measure timing multiple times and return mean and std."""
    times = []
    for _ in range(n_repeats):
        time = timeit.timeit(operation, number=100) / 100
        times.append(time)
    return np.mean(times), np.std(times)

def benchmark_lorentz_boost(size, n_repeats=10):
    """Benchmark Lorentz boost operations."""
    results = {}
    
    # Generate test data
    px, py, pz, E = generate_test_data(size)
    
    # Create lvec and vector objects
    lvec = LVec(px, py, pz, E)
    vec = vector.arr({"px": px, "py": py, "pz": pz, "E": E})
    
    # Define boost parameters
    beta_x, beta_y, beta_z = 0.5, 0.3, 0.6
    
    # Create a Vector3D object for general boost in vector package
    boost_vec3d = vector.obj(x=0.2, y=0.2, z=0.2)
    
    # Dictionary of operations to benchmark
    operations = {
        # X-axis boost operations
        'boostx': (
            lambda: lvec.boost(beta_x, 0.0, 0.0),  # lvec X-axis boost using general method
            lambda: vec.boostX(beta_x)             # vector X-axis boost using specialized method
        ),
        
        # Y-axis boost operations
        'boosty': (
            lambda: lvec.boost(0.0, beta_y, 0.0),  # lvec Y-axis boost using general method
            lambda: vec.boostY(beta_y)             # vector Y-axis boost using specialized method
        ),
        
        # Z-axis boost operations
        'boostz': (
            lambda: lvec.boostz(beta_z),           # lvec Z-axis boost using specialized method
            lambda: vec.boostZ(beta_z)             # vector Z-axis boost using specialized method
        ),
        
        # General 3D boost operations
        'boost_3d': (
            lambda: lvec.boost(0.2, 0.2, 0.2),     # lvec general 3D boost
            lambda: vec.boost(boost_vec3d)         # vector general 3D boost with Vector3D object
        ),
        
        # Z-axis boost using general method vs specialized method (lvec only)
        'lvec_boostz_comparison': (
            lambda: lvec.boostz(0.4),              # lvec specialized Z-axis boost
            lambda: lvec.boost(0.0, 0.0, 0.4)      # lvec general boost method for Z-axis
        ),
    }
    
    # Run benchmarks for each operation
    for op_name, (lvec_op, vector_op) in operations.items():
        # Measure timing
        lvec_time, lvec_error = measure_single_timing(lvec_op, n_repeats)
        vector_time, vector_error = measure_single_timing(vector_op, n_repeats)
        
        # Measure memory usage
        lvec_memory = measure_memory_usage(lvec_op)
        vector_memory = measure_memory_usage(vector_op)
        
        # Store results
        results[op_name] = {
            'lvec': {
                'time': lvec_time, 
                'error': lvec_error, 
                'memory': lvec_memory
            },
            'vector': {
                'time': vector_time, 
                'error': vector_error, 
                'memory': vector_memory
            }
        }
    
    return results

def plot_boost_time_comparison(sizes, all_results, operations, save_path=None):
    """Plot timing comparison for boost operations."""
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, op in enumerate(operations[:4]):  # First 4 operations for package comparison
        ax = axes[i]
        
        # Extract timing data (convert to milliseconds)
        lvec_times = np.array([r[op]['lvec']['time'] for r in all_results]) * 1000
        vector_times = np.array([r[op]['vector']['time'] for r in all_results]) * 1000
        
        # Extract error bars
        lvec_errors = np.array([r[op]['lvec']['error'] for r in all_results]) * 1000
        vector_errors = np.array([r[op]['vector']['error'] for r in all_results]) * 1000
        
        # Plot time comparison
        ax.errorbar(sizes, lvec_times, yerr=lvec_errors, fmt='o-', label='lvec', 
                   color=COLORS['lvec'], linewidth=2, markersize=8, capsize=4)
        ax.errorbar(sizes, vector_times, yerr=vector_errors, fmt='o-', label='vector', 
                   color=COLORS['vector'], linewidth=2, markersize=8, capsize=4)
        
        # Calculate speedup ratio
        speedup = vector_times / lvec_times
        
        # Add speedup text for largest size
        ax.text(0.7, 0.05, f'Speedup: {speedup[-1]:.2f}x', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Customize plot
        ax.set_title(f'{op.upper()} Operation')
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Time (ms)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Add overall title
    fig.suptitle('Lorentz Boost Performance Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_lvec_z_boost_comparison(sizes, all_results, save_path=None):
    """Plot comparison between lvec's specialized boostz and general boost methods."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract timing data (convert to milliseconds)
    specialized_times = np.array([r['lvec_boostz_comparison']['lvec']['time'] for r in all_results]) * 1000
    general_times = np.array([r['lvec_boostz_comparison']['vector']['time'] for r in all_results]) * 1000
    
    # Extract error bars
    specialized_errors = np.array([r['lvec_boostz_comparison']['lvec']['error'] for r in all_results]) * 1000
    general_errors = np.array([r['lvec_boostz_comparison']['vector']['error'] for r in all_results]) * 1000
    
    # Plot time comparison
    ax.errorbar(sizes, specialized_times, yerr=specialized_errors, fmt='o-', 
               label='Specialized boostz()', color=COLORS['lvec'], 
               linewidth=2, markersize=8, capsize=4)
    ax.errorbar(sizes, general_times, yerr=general_errors, fmt='o-', 
               label='General boost(0,0,β)', color=COLORS['vector'], 
               linewidth=2, markersize=8, capsize=4)
    
    # Calculate speedup ratio
    speedup = general_times / specialized_times
    
    # Add speedup text
    ax.text(0.7, 0.05, f'Speedup: {speedup[-1]:.2f}x', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Customize plot
    ax.set_title('Comparison of Z-Boost Methods in lvec')
    ax.set_xlabel('Array Size')
    ax.set_ylabel('Time (ms)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_memory_usage(sizes, all_results, operations, save_path=None):
    """Plot memory usage comparison for boost operations."""
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, op in enumerate(operations[:4]):  # First 4 operations for package comparison
        ax = axes[i]
        
        # Extract memory data
        lvec_memory = np.array([r[op]['lvec']['memory'] for r in all_results])
        vector_memory = np.array([r[op]['vector']['memory'] for r in all_results])
        
        # Plot memory comparison
        ax.plot(sizes, lvec_memory, 'o-', label='lvec', 
               color=COLORS['lvec'], linewidth=2, markersize=8)
        ax.plot(sizes, vector_memory, 'o-', label='vector', 
               color=COLORS['vector'], linewidth=2, markersize=8)
        
        # Calculate memory ratio
        memory_ratio = vector_memory / lvec_memory
        
        # Add ratio text for largest size
        ax.text(0.7, 0.05, f'Memory Ratio: {memory_ratio[-1]:.2f}x', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Customize plot
        ax.set_title(f'{op.upper()} Operation')
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Add overall title
    fig.suptitle('Memory Usage Comparison for Lorentz Boost Operations', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def run_benchmarks():
    """Run all benchmarks and plot results."""
    # Create plots directory if it doesn't exist
    os.makedirs("benchmarks/plots", exist_ok=True)
    
    # Define sizes to benchmark
    sizes = [1000, 10000, 100000, 1000000]
    operations = ['boostx', 'boosty', 'boostz', 'boost_3d', 'lvec_boostz_comparison']
    
    # Run benchmarks for all sizes
    all_results = []
    for size in sizes:
        print(f"\nBenchmarking with {size:,} vectors")
        results = benchmark_lorentz_boost(size)
        all_results.append(results)
        
        # Print results
        for op in operations:
            lvec_time = results[op]['lvec']['time'] * 1000  # Convert to ms
            vector_time = results[op]['vector']['time'] * 1000
            lvec_mem = results[op]['lvec']['memory']
            vector_mem = results[op]['vector']['memory']
            
            print(f"  {op.upper()} Operation:")
            print(f"    lvec:   {lvec_time:.3f} ms, {lvec_mem:.2f} MB")
            print(f"    vector: {vector_time:.3f} ms, {vector_mem:.2f} MB")
            print(f"    Speed Ratio:  {vector_time/lvec_time:.2f}x faster with lvec")
            print(f"    Memory Ratio: {vector_mem/lvec_mem:.2f}x more memory efficient with lvec")
    
    # Plot results
    plot_boost_time_comparison(sizes, all_results, operations, 
                              save_path=os.path.join("benchmarks/plots", "lorentz_boost_time_comparison.pdf"))
    
    plot_lvec_z_boost_comparison(sizes, all_results, 
                                save_path=os.path.join("benchmarks/plots", "lvec_z_boost_comparison.pdf"))
    
    plot_memory_usage(sizes, all_results, operations, 
                     save_path=os.path.join("benchmarks/plots", "lorentz_boost_memory_usage.pdf"))
    
    print("\nBenchmarks completed. Results saved to PDF files.")

if __name__ == "__main__":
    # Run all benchmarks
    run_benchmarks()
