#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark for arithmetic operations and derived properties in LVec.
This benchmark focuses on the computational speed of vector algebra and 
the effectiveness of caching in LVec compared to other vector libraries.
"""

import numpy as np
import timeit
import matplotlib.pyplot as plt
import tracemalloc
import gc
import time
from functools import partial
from lvec import LVec, Vector2D, Vector3D
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

def generate_2d_test_data(size):
    """Generate random 2D vectors for testing."""
    x = np.random.normal(0, 10, size)
    y = np.random.normal(0, 10, size)
    return x, y

def generate_3d_test_data(size):
    """Generate random 3D vectors for testing."""
    x = np.random.normal(0, 10, size)
    y = np.random.normal(0, 10, size)
    z = np.random.normal(0, 10, size)
    return x, y, z

def measure_single_timing(operation, n_repeats=10):
    """Measure timing multiple times and return mean and std."""
    times = []
    for _ in range(n_repeats):
        time = timeit.timeit(operation, number=100) / 100
        times.append(time)
    return np.mean(times), np.std(times)

def benchmark_arithmetic(size, vector_type, n_repeats=10):
    """Benchmark arithmetic operations for a given vector type."""
    results = {}
    
    if vector_type == "LVec":
        px, py, pz, E = generate_test_data(size)
        v1 = LVec(px, py, pz, E)
        v2 = LVec(px, py, pz, E)
        
        # Addition
        add_op = lambda: v1 + v2
        add_time, add_std = measure_single_timing(add_op, n_repeats)
        results["addition"] = {"time": add_time, "std": add_std}
        
        # Subtraction
        sub_op = lambda: v1 - v2
        sub_time, sub_std = measure_single_timing(sub_op, n_repeats)
        results["subtraction"] = {"time": sub_time, "std": sub_std}
        
        # Scalar multiplication
        mul_op = lambda: v1 * 2.0
        mul_time, mul_std = measure_single_timing(mul_op, n_repeats)
        results["scalar_mul"] = {"time": mul_time, "std": mul_std}
        
    elif vector_type == "Vector2D":
        x, y = generate_2d_test_data(size)
        v1 = Vector2D(x, y)
        v2 = Vector2D(x, y)
        
        # Addition
        add_op = lambda: v1 + v2
        add_time, add_std = measure_single_timing(add_op, n_repeats)
        results["addition"] = {"time": add_time, "std": add_std}
        
        # Subtraction
        sub_op = lambda: v1 - v2
        sub_time, sub_std = measure_single_timing(sub_op, n_repeats)
        results["subtraction"] = {"time": sub_time, "std": sub_std}
        
        # Scalar multiplication
        mul_op = lambda: v1 * 2.0
        mul_time, mul_std = measure_single_timing(mul_op, n_repeats)
        results["scalar_mul"] = {"time": mul_time, "std": mul_std}
        
        # Dot product
        dot_op = lambda: v1.dot(v2)
        dot_time, dot_std = measure_single_timing(dot_op, n_repeats)
        results["dot_product"] = {"time": dot_time, "std": dot_std}
        
    elif vector_type == "Vector3D":
        x, y, z = generate_3d_test_data(size)
        v1 = Vector3D(x, y, z)
        v2 = Vector3D(x, y, z)
        
        # Addition
        add_op = lambda: v1 + v2
        add_time, add_std = measure_single_timing(add_op, n_repeats)
        results["addition"] = {"time": add_time, "std": add_std}
        
        # Subtraction
        sub_op = lambda: v1 - v2
        sub_time, sub_std = measure_single_timing(sub_op, n_repeats)
        results["subtraction"] = {"time": sub_time, "std": sub_std}
        
        # Scalar multiplication
        mul_op = lambda: v1 * 2.0
        mul_time, mul_std = measure_single_timing(mul_op, n_repeats)
        results["scalar_mul"] = {"time": mul_time, "std": mul_std}
        
        # Dot product
        dot_op = lambda: v1.dot(v2)
        dot_time, dot_std = measure_single_timing(dot_op, n_repeats)
        results["dot_product"] = {"time": dot_time, "std": dot_std}
        
        # Cross product
        cross_op = lambda: v1.cross(v2)
        cross_time, cross_std = measure_single_timing(cross_op, n_repeats)
        results["cross_product"] = {"time": cross_time, "std": cross_std}
        
    elif vector_type == "Vector":  # vector package for comparison
        px, py, pz, E = generate_test_data(size)
        v1 = vector.arr({"px": px, "py": py, "pz": pz, "E": E})
        v2 = vector.arr({"px": px, "py": py, "pz": pz, "E": E})
        
        # Addition
        add_op = lambda: v1 + v2
        add_time, add_std = measure_single_timing(add_op, n_repeats)
        results["addition"] = {"time": add_time, "std": add_std}
        
        # Subtraction
        sub_op = lambda: v1 - v2
        sub_time, sub_std = measure_single_timing(sub_op, n_repeats)
        results["subtraction"] = {"time": sub_time, "std": sub_std}
        
        # Scalar multiplication
        mul_op = lambda: v1 * 2.0
        mul_time, mul_std = measure_single_timing(mul_op, n_repeats)
        results["scalar_mul"] = {"time": mul_time, "std": mul_std}
    
    return results

def benchmark_derived_properties(size, vector_type, n_repeats=10):
    """Benchmark derived properties for a given vector type."""
    results = {}
    
    if vector_type == "LVec":
        px, py, pz, E = generate_test_data(size)
        vec = LVec(px, py, pz, E)
        
        # Mass
        mass_op = lambda: vec.mass
        mass_time, mass_std = measure_single_timing(mass_op, n_repeats)
        results["mass"] = {"time": mass_time, "std": mass_std}
        
        # Transverse momentum
        pt_op = lambda: vec.pt
        pt_time, pt_std = measure_single_timing(pt_op, n_repeats)
        results["pt"] = {"time": pt_time, "std": pt_std}
        
        # Pseudorapidity
        eta_op = lambda: vec.eta
        eta_time, eta_std = measure_single_timing(eta_op, n_repeats)
        results["eta"] = {"time": eta_time, "std": eta_std}
        
        # Phi
        phi_op = lambda: vec.phi
        phi_time, phi_std = measure_single_timing(phi_op, n_repeats)
        results["phi"] = {"time": phi_time, "std": phi_std}
        
    elif vector_type == "Vector2D":
        x, y = generate_2d_test_data(size)
        vec = Vector2D(x, y)
        
        # Magnitude
        r_op = lambda: vec.r
        r_time, r_std = measure_single_timing(r_op, n_repeats)
        results["magnitude"] = {"time": r_time, "std": r_std}
        
        # Phi
        phi_op = lambda: vec.phi
        phi_time, phi_std = measure_single_timing(phi_op, n_repeats)
        results["phi"] = {"time": phi_time, "std": phi_std}
        
    elif vector_type == "Vector3D":
        x, y, z = generate_3d_test_data(size)
        vec = Vector3D(x, y, z)
        
        # Magnitude
        r_op = lambda: vec.r
        r_time, r_std = measure_single_timing(r_op, n_repeats)
        results["magnitude"] = {"time": r_time, "std": r_std}
        
        # Phi
        phi_op = lambda: vec.phi
        phi_time, phi_std = measure_single_timing(phi_op, n_repeats)
        results["phi"] = {"time": phi_time, "std": phi_std}
        
        # Theta
        theta_op = lambda: vec.theta
        theta_time, theta_std = measure_single_timing(theta_op, n_repeats)
        results["theta"] = {"time": theta_time, "std": theta_std}
        
        # Rho (cylindrical radius)
        rho_op = lambda: vec.rho
        rho_time, rho_std = measure_single_timing(rho_op, n_repeats)
        results["rho"] = {"time": rho_time, "std": rho_std}
        
    elif vector_type == "Vector":  # vector package for comparison
        px, py, pz, E = generate_test_data(size)
        vec = vector.arr({"px": px, "py": py, "pz": pz, "E": E})
        
        # Mass
        mass_op = lambda: vec.mass
        mass_time, mass_std = measure_single_timing(mass_op, n_repeats)
        results["mass"] = {"time": mass_time, "std": mass_std}
        
        # Transverse momentum
        pt_op = lambda: vec.pt
        pt_time, pt_std = measure_single_timing(pt_op, n_repeats)
        results["pt"] = {"time": pt_time, "std": pt_std}
        
        # Pseudorapidity
        eta_op = lambda: vec.eta
        eta_time, eta_std = measure_single_timing(eta_op, n_repeats)
        results["eta"] = {"time": eta_time, "std": eta_std}
        
        # Phi
        phi_op = lambda: vec.phi
        phi_time, phi_std = measure_single_timing(phi_op, n_repeats)
        results["phi"] = {"time": phi_time, "std": phi_std}
        
    return results

def benchmark_caching_effectiveness(size, n_repeats=10):
    """
    Benchmark the effectiveness of caching by measuring the time
    for repeated property access with and without cache invalidation.
    """
    px, py, pz, E = generate_test_data(size)
    vec = LVec(px, py, pz, E)
    
    # Benchmark with caching (accessing property multiple times)
    def cached_access():
        for _ in range(5):  # Access property 5 times
            _ = vec.mass
            _ = vec.pt
            _ = vec.eta
    
    # Benchmark without caching (invalidating cache between accesses)
    def uncached_access():
        for _ in range(5):  # Access property 5 times
            _ = vec.mass
            vec.touch()  # Invalidate cache
            _ = vec.pt
            vec.touch()  # Invalidate cache
            _ = vec.eta
            vec.touch()  # Invalidate cache
    
    cached_time, cached_std = measure_single_timing(cached_access, n_repeats)
    uncached_time, uncached_std = measure_single_timing(uncached_access, n_repeats)
    
    return {
        "cached": {"time": cached_time, "std": cached_std},
        "uncached": {"time": uncached_time, "std": uncached_std}
    }

def plot_arithmetic_results(sizes, results, vector_types, operations, save_path="benchmark_arithmetic.pdf"):
    """Plot arithmetic operation benchmark results."""
    plt.style.use('default')
    n_ops = len(operations)
    n_cols = 2
    n_rows = (n_ops + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    
    colors = {'LVec': '#3498db', 'Vector2D': '#2ecc71', 'Vector3D': '#9b59b6', 'Vector': '#e74c3c'}
    
    for op_idx, operation in enumerate(operations):
        row = op_idx // n_cols
        col = op_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        for vtype in vector_types:
            # Check if this vector type has this operation
            if vtype in results and operation in results[vtype][0]:
                times = [results[vtype][i].get(operation, {}).get("time", np.nan) * 1000 for i in range(len(sizes))]  # ms
                ax.plot(sizes, times, 'o-', label=vtype, color=colors[vtype], linewidth=2, markersize=6)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Array Size', fontsize=10)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title(operation.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=8)
    
    # Remove any empty subplots
    for idx in range(len(operations), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if idx < n_rows * n_cols:  # Ensure we're not out of bounds
            fig.delaxes(fig.add_subplot(gs[row, col]))
    
    plt.suptitle('Performance Comparison of Arithmetic Operations', fontsize=14, y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_derived_results(sizes, results, vector_types, properties, save_path="benchmark_derived.pdf"):
    """Plot derived properties benchmark results."""
    plt.style.use('default')
    n_props = len(properties)
    n_cols = 2
    n_rows = (n_props + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    
    colors = {'LVec': '#3498db', 'Vector2D': '#2ecc71', 'Vector3D': '#9b59b6', 'Vector': '#e74c3c'}
    
    for prop_idx, prop in enumerate(properties):
        row = prop_idx // n_cols
        col = prop_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        for vtype in vector_types:
            # Check if this vector type has this property
            if vtype in results and prop in results[vtype][0]:
                times = [results[vtype][i].get(prop, {}).get("time", np.nan) * 1000 for i in range(len(sizes))]  # ms
                ax.plot(sizes, times, 'o-', label=vtype, color=colors[vtype], linewidth=2, markersize=6)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Array Size', fontsize=10)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title(prop.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=8)
    
    # Remove any empty subplots
    for idx in range(len(properties), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if idx < n_rows * n_cols:  # Ensure we're not out of bounds
            fig.delaxes(fig.add_subplot(gs[row, col]))
    
    plt.suptitle('Performance Comparison of Derived Properties', fontsize=14, y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_caching_results(sizes, cache_results, save_path="benchmark_caching.pdf"):
    """Plot caching effectiveness benchmark results."""
    plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract times in milliseconds
    cached_times = [res["cached"]["time"] * 1000 for res in cache_results]
    uncached_times = [res["uncached"]["time"] * 1000 for res in cache_results]
    
    # Calculate speedup
    speedup = [uncached / cached for uncached, cached in zip(uncached_times, cached_times)]
    
    # Create primary plot for times
    ax.plot(sizes, cached_times, 'o-', label='With Caching', color='#2ecc71', linewidth=2, markersize=6)
    ax.plot(sizes, uncached_times, 'o-', label='Without Caching', color='#e74c3c', linewidth=2, markersize=6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='upper left')
    
    # Create secondary y-axis for speedup
    ax2 = ax.twinx()
    ax2.plot(sizes, speedup, 'o--', label='Speedup Factor', color='#3498db', linewidth=1.5, markersize=5)
    ax2.set_ylabel('Speedup Factor (Uncached/Cached)', fontsize=12, color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2.legend(fontsize=10, loc='upper right')
    
    plt.title('Caching Effectiveness in LVec', fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_benchmarks():
    """Run all benchmarks and plot results."""
    sizes = [10, 100, 1000, 10000, 100000, 1000000]
    vector_types = ["LVec", "Vector2D", "Vector3D", "Vector"]
    
    # Arithmetic operations
    arith_results = {vtype: [] for vtype in vector_types}
    
    for size in sizes:
        print(f"\nBenchmarking arithmetic operations with array size: {size:,}")
        for vtype in vector_types:
            print(f"  Vector type: {vtype}")
            res = benchmark_arithmetic(size, vtype)
            arith_results[vtype].append(res)
            
    # Derived properties
    derived_results = {vtype: [] for vtype in vector_types}
    
    for size in sizes:
        print(f"\nBenchmarking derived properties with array size: {size:,}")
        for vtype in vector_types:
            print(f"  Vector type: {vtype}")
            res = benchmark_derived_properties(size, vtype)
            derived_results[vtype].append(res)
    
    # Caching effectiveness
    cache_results = []
    
    for size in sizes:
        print(f"\nBenchmarking caching effectiveness with array size: {size:,}")
        res = benchmark_caching_effectiveness(size)
        cache_results.append(res)
        print(f"  With caching:    {res['cached']['time']*1000:.3f} ms")
        print(f"  Without caching: {res['uncached']['time']*1000:.3f} ms")
        print(f"  Speedup:         {res['uncached']['time']/res['cached']['time']:.2f}x")
    
    # Plot results
    arith_ops = ["addition", "subtraction", "scalar_mul", "dot_product", "cross_product"]
    derived_props = ["mass", "pt", "eta", "phi", "magnitude", "theta", "rho"]
    
    plot_arithmetic_results(sizes, arith_results, vector_types, arith_ops, "benchmarks/plots/benchmark_arithmetic.pdf")
    plot_derived_results(sizes, derived_results, vector_types, derived_props, "benchmarks/plots/benchmark_derived.pdf")
    plot_caching_results(sizes, cache_results, "benchmarks/plots/benchmark_caching.pdf")
    
    print("\nBenchmarks completed. Plots saved to:")
    print("  - benchmarks/plots/benchmark_arithmetic.pdf")
    print("  - benchmarks/plots/benchmark_derived.pdf")
    print("  - benchmarks/plots/benchmark_caching.pdf")

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    import os
    os.makedirs("benchmarks/plots", exist_ok=True)
    
    run_benchmarks()
