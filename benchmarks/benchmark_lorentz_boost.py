#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark for Lorentz boost operations in LVec.
This benchmark compares the performance of axis-specific boosts vs general boosts
between LVec and the vector package, with focus on backend optimizations.
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
    
    # Create LVec and vector objects
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
            lambda: lvec.boost(beta_x, 0.0, 0.0),  # LVec X-axis boost using general method
            lambda: vec.boostX(beta_x)             # vector X-axis boost using specialized method
        ),
        
        # Y-axis boost operations
        'boosty': (
            lambda: lvec.boost(0.0, beta_y, 0.0),  # LVec Y-axis boost using general method
            lambda: vec.boostY(beta_y)             # vector Y-axis boost using specialized method
        ),
        
        # Z-axis boost operations
        'boostz': (
            lambda: lvec.boostz(beta_z),           # LVec Z-axis boost using specialized method
            lambda: vec.boostZ(beta_z)             # vector Z-axis boost using specialized method
        ),
        
        # General 3D boost operations
        'boost_3d': (
            lambda: lvec.boost(0.2, 0.2, 0.2),     # LVec general 3D boost
            lambda: vec.boost(boost_vec3d)         # vector general 3D boost with Vector3D object
        ),
        
        # Z-axis boost using general method vs specialized method (LVec only)
        'lvec_boostz_comparison': (
            lambda: lvec.boostz(0.4),              # LVec specialized Z-axis boost
            lambda: lvec.boost(0.0, 0.0, 0.4)      # LVec general boost method for Z-axis
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
    plt.style.use('default')
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
        ax.errorbar(sizes, lvec_times, yerr=lvec_errors, fmt='o-', label='LVec', 
                   color='#3498db', linewidth=2, markersize=8, capsize=4)
        ax.errorbar(sizes, vector_times, yerr=vector_errors, fmt='o-', label='vector', 
                   color='#e74c3c', linewidth=2, markersize=8, capsize=4)
        
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_lvec_z_boost_comparison(sizes, all_results, save_path=None):
    """Plot comparison between LVec's specialized boostz and general boost methods."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract timing data (convert to milliseconds)
    specialized_times = np.array([r['lvec_boostz_comparison']['lvec']['time'] for r in all_results]) * 1000
    general_times = np.array([r['lvec_boostz_comparison']['vector']['time'] for r in all_results]) * 1000
    
    # Extract error bars
    specialized_errors = np.array([r['lvec_boostz_comparison']['lvec']['error'] for r in all_results]) * 1000
    general_errors = np.array([r['lvec_boostz_comparison']['vector']['error'] for r in all_results]) * 1000
    
    # Plot time comparison
    ax.errorbar(sizes, specialized_times, yerr=specialized_errors, fmt='o-', 
               label='Specialized boostz()', color='#3498db', 
               linewidth=2, markersize=8, capsize=4)
    ax.errorbar(sizes, general_times, yerr=general_errors, fmt='o-', 
               label='General boost(0,0,β)', color='#9b59b6', 
               linewidth=2, markersize=8, capsize=4)
    
    # Calculate speedup ratio
    speedup = general_times / specialized_times
    
    # Add speedup text
    ax.text(0.7, 0.05, f'Speedup: {speedup[-1]:.2f}x', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Customize plot
    ax.set_title('LVec: Specialized vs General Z-Boost Methods')
    ax.set_xlabel('Array Size')
    ax.set_ylabel('Time (ms)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_memory_usage(sizes, all_results, operations, save_path=None):
    """Plot memory usage comparison for boost operations."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, op in enumerate(operations[:4]):  # First 4 operations for package comparison
        # Extract memory data (in MB)
        lvec_memory = np.array([r[op]['lvec']['memory'] for r in all_results])
        vector_memory = np.array([r[op]['vector']['memory'] for r in all_results])
        
        # Calculate memory ratio
        memory_ratio = lvec_memory / vector_memory
        
        # Plot memory ratio (values < 1 mean LVec uses less memory)
        ax.plot(sizes, memory_ratio, 'o-', label=op, 
                color=colors[i], linewidth=2, markersize=8)
    
    # Add horizontal line at ratio = 1 (equal memory usage)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    # Customize plot
    ax.set_title('Memory Usage Ratio (LVec / vector)')
    ax.set_xlabel('Array Size')
    ax.set_ylabel('Memory Ratio')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add explanation text
    ax.text(0.02, 0.05, 'Values < 1: LVec uses less memory\nValues > 1: vector uses less memory', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def run_benchmarks():
    """Run all benchmarks and plot results."""
    # Array sizes to benchmark
    sizes = [10, 100, 1000, 10000, 100000]
    operations = ['boostx', 'boosty', 'boostz', 'boost_3d', 'lvec_boostz_comparison']
    
    # Store results for each operation and size
    all_results = []
    
    # Run benchmarks
    for size in sizes:
        print(f"\nBenchmarking Lorentz boost operations with array size: {size:,}")
        results = benchmark_lorentz_boost(size)
        all_results.append(results)
        
        # Print some summary statistics
        for op in operations:
            if op in results:
                lvec_time = results[op]['lvec']['time'] * 1000  # ms
                vector_time = results[op]['vector']['time'] * 1000  # ms
                speedup = vector_time / lvec_time
                print(f"  {op:20s} - LVec: {lvec_time:.3f} ms, vector: {vector_time:.3f} ms, Speedup: {speedup:.2f}x")
    
    # Create plots directory if it doesn't exist
    os.makedirs("benchmarks/plots", exist_ok=True)
    
    # Plot results
    plot_boost_time_comparison(sizes, all_results, operations, 
                             save_path="benchmarks/plots/lorentz_boost_time_comparison.pdf")
    plot_lvec_z_boost_comparison(sizes, all_results, 
                               save_path="benchmarks/plots/lvec_z_boost_methods_comparison.pdf")
    plot_memory_usage(sizes, all_results, operations, 
                    save_path="benchmarks/plots/lorentz_boost_memory_comparison.pdf")
    
    print("\nBenchmarks completed. Plots saved to:")
    print("  - benchmarks/plots/lorentz_boost_time_comparison.pdf")
    print("  - benchmarks/plots/lvec_z_boost_methods_comparison.pdf")
    print("  - benchmarks/plots/lorentz_boost_memory_comparison.pdf")

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs("benchmarks/plots", exist_ok=True)
    
    run_benchmarks()
