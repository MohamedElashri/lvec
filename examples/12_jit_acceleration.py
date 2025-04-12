#!/usr/bin/env python3
"""
JIT Acceleration Demo
====================

This example demonstrates how to use the JIT (Just-In-Time) compilation 
feature in the lvec package to accelerate performance-critical calculations.

It shows:
1. How to check if JIT compilation is available
2. How to enable/disable JIT compilation globally
3. Performance comparison between JIT-enabled and JIT-disabled code
4. How JIT acceleration works with the caching system
5. Best practices for JIT acceleration

Note: This example requires the numba package to demonstrate JIT acceleration.
If numba is not installed, the example will show how lvec gracefully falls back
to non-JIT implementations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from lvec import LVec, Vector3D, is_jit_available, enable_jit

def check_jit_status():
    """Check if JIT compilation is available and enabled."""
    print("\n=== JIT Acceleration Status ===")
    
    if is_jit_available():
        print(" JIT compilation is available")
        print("   Using numba for accelerated computations")
    else:
        print(" JIT compilation is not available")
        print("   To enable JIT, install numba: pip install numba")
    
    # Try enabling JIT (will only work if numba is installed)
    original_status = is_jit_available()
    enable_jit(True)
    new_status = is_jit_available()
    
    if original_status != new_status:
        print(f"JIT status changed: {original_status} -> {new_status}")
    else:
        print(f"JIT status unchanged: {new_status}")

def benchmark_jit_performance():
    """Benchmark the performance difference between JIT-enabled and disabled code."""
    print("\n=== JIT Performance Benchmark ===")
    
    # Create a large dataset for benchmarking
    n_particles = 10_000_000
    print(f"Creating test dataset with {n_particles:,} particles...")
    
    # Generate random particle data
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    px = rng.normal(0, 10, n_particles)
    py = rng.normal(0, 10, n_particles)
    pz = rng.normal(0, 10, n_particles)
    E = np.sqrt(px**2 + py**2 + pz**2 + 0.14**2)  # pion mass ~ 0.14 GeV
    
    # Define operations to benchmark
    def benchmark_operations(vectors):
        _ = vectors.pt      # Transverse momentum 
        _ = vectors.eta     # Pseudorapidity
        _ = vectors.phi     # Azimuthal angle
        _ = vectors.mass    # Invariant mass
        _ = vectors.p       # Total momentum
        
        # Additional operations that use multiple properties
        _ = vectors.pt * np.cosh(vectors.eta)  # pz calculation
        _ = vectors.pt**2 + vectors.pz**2      # p^2 calculation

    # Run benchmark with JIT enabled vs disabled
    results = {}
    
    for jit_enabled in [True, False]:
        # Set JIT status
        enable_jit(jit_enabled)
        status = "enabled" if jit_enabled else "disabled"
        print(f"\nRunning benchmark with JIT {status}...")
        
        # Create vectors (JIT setting affects vector creation too)
        start_time = time.time()
        vectors = LVec(px, py, pz, E)
        create_time = time.time() - start_time
        print(f"  Vector creation: {create_time:.4f} seconds")
        
        # First run (cold start)
        start_time = time.time()
        benchmark_operations(vectors)
        cold_time = time.time() - start_time
        print(f"  First run (cold): {cold_time:.4f} seconds")
        
        # Clear cache to ensure fair comparison
        vectors.clear_cache()
        
        # Second run (cold again due to cache clear)
        start_time = time.time()
        benchmark_operations(vectors)
        second_time = time.time() - start_time
        print(f"  Second run (cold): {second_time:.4f} seconds")
        
        # Third run (should be warm with cached values)
        start_time = time.time()
        benchmark_operations(vectors)
        warm_time = time.time() - start_time
        print(f"  Third run (warm): {warm_time:.4f} seconds")
        
        results[jit_enabled] = {
            "create": create_time,
            "cold": cold_time,
            "second": second_time,
            "warm": warm_time
        }
    
    return results

def benchmark_jit_with_caching():
    """Show how JIT acceleration works together with the caching system."""
    print("\n=== JIT with Caching System ===")
    
    # Enable JIT for this test
    enable_jit(True)
    
    # Create test data
    n_particles = 100_000
    rng = np.random.RandomState(42)
    px = rng.normal(0, 10, n_particles)
    py = rng.normal(0, 10, n_particles)
    pz = rng.normal(0, 10, n_particles)
    E = np.sqrt(px**2 + py**2 + pz**2 + 0.14**2)
    
    vectors = LVec(px, py, pz, E)
    
    # Reset cache counters
    vectors._cache.reset_counters()
    
    print("\n1. First access (JIT-accelerated calculation + cache store)")
    start_time = time.time()
    _ = vectors.mass  # This should use JIT
    first_time = time.time() - start_time
    print(f"   Time: {first_time:.6f} seconds")
    
    print("\n2. Second access (cache retrieval, no calculation)")
    start_time = time.time()
    _ = vectors.mass  # This should use cache
    cached_time = time.time() - start_time
    print(f"   Time: {cached_time:.6f} seconds")
    
    # Print cache stats
    stats = vectors._cache.get_stats()
    print(f"\nCache hit ratio for mass: {stats['properties'].get('mass', {}).get('hit_ratio', 0):.2%}")
    
    # Show the speedup from caching vs JIT vs both
    if first_time > 0:
        cache_speedup = first_time / cached_time
        print(f"\nSpeedup from caching: {cache_speedup:.1f}x")
    
    print("\n3. Clearing cache and disabling JIT")
    vectors._cache.clear_cache()
    enable_jit(False)
    
    print("\n4. Access with JIT disabled (non-JIT calculation + cache store)")
    start_time = time.time()
    _ = vectors.mass  # This should use non-JIT implementation
    nojit_time = time.time() - start_time
    print(f"   Time: {nojit_time:.6f} seconds")
    
    # Calculate JIT speedup if available
    if nojit_time > 0 and first_time > 0:
        jit_speedup = nojit_time / first_time
        print(f"\nSpeedup from JIT: {jit_speedup:.1f}x")
    
    return first_time, cached_time, nojit_time

def benchmark_vector_operations():
    """Benchmark JIT performance for different vector operations."""
    print("\n=== JIT Performance Across Different Operations ===")
    
    # Create test data
    n_particles = 500_000
    rng = np.random.RandomState(42)
    px = rng.normal(0, 10, n_particles)
    py = rng.normal(0, 10, n_particles)
    pz = rng.normal(0, 10, n_particles)
    E = np.sqrt(px**2 + py**2 + pz**2 + 0.14**2)
    
    # Define operations to benchmark
    operations = {
        "pt": lambda v: v.pt,
        "eta": lambda v: v.eta,
        "phi": lambda v: v.phi,
        "mass": lambda v: v.mass,
        "p": lambda v: v.p,
    }
    
    # Run benchmarks
    results = {}
    
    for jit_enabled in [True, False]:
        enable_jit(jit_enabled)
        status = "enabled" if jit_enabled else "disabled"
        print(f"\nRunning operation benchmarks with JIT {status}...")
        
        vectors = LVec(px, py, pz, E)
        operation_times = {}
        
        for name, operation in operations.items():
            # Clear cache for fair comparison
            vectors.clear_cache()
            
            # Time the operation
            start_time = time.time()
            result = operation(vectors)
            end_time = time.time()
            
            operation_times[name] = end_time - start_time
            print(f"  {name:5s}: {operation_times[name]:.6f} seconds")
        
        results[jit_enabled] = operation_times
    
    return results

def plot_benchmark_results(jit_results, operation_results):
    """Plot benchmark results for visualization."""
    print("\n=== Visualizing JIT Performance ===")
    
    # Plot 1: JIT vs No JIT for different run types
    plt.figure(figsize=(12, 6))
    
    # Data preparation
    categories = ["Vector Creation", "Cold Run", "Second Run", "Warm Run"]
    jit_times = [jit_results[True]["create"], jit_results[True]["cold"],
                jit_results[True]["second"], jit_results[True]["warm"]]
    nojit_times = [jit_results[False]["create"], jit_results[False]["cold"],
                  jit_results[False]["second"], jit_results[False]["warm"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, nojit_times, width, label='JIT Disabled', color='#e74c3c')
    rects2 = ax.bar(x + width/2, jit_times, width, label='JIT Enabled', color='#2ecc71')
    
    # Add labels and formatting
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Performance Comparison: JIT vs No JIT', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add speedup values on top of bars
    for i, (nojit, jit) in enumerate(zip(nojit_times, jit_times)):
        if jit > 0:
            speedup = nojit / jit
            ax.text(i, max(nojit, jit) + 0.01, f"{speedup:.1f}x", 
                   ha='center', fontsize=10, color='black')
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("jit_performance_comparison.pdf")
    print("Saved figure as 'jit_performance_comparison.pdf'")
    
    # Plot 2: Operation-specific comparison
    plt.figure(figsize=(12, 6))
    
    # Data preparation
    operations = list(operation_results[True].keys())
    jit_op_times = [operation_results[True][op] for op in operations]
    nojit_op_times = [operation_results[False][op] for op in operations]
    
    x = np.arange(len(operations))
    width = 0.35
    
    # Create bars
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, nojit_op_times, width, label='JIT Disabled', color='#e74c3c')
    rects2 = ax.bar(x + width/2, jit_op_times, width, label='JIT Enabled', color='#2ecc71')
    
    # Add labels and formatting
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('JIT Performance by Operation Type', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend()
    
    # Add speedup values on top of bars
    for i, (nojit, jit) in enumerate(zip(nojit_op_times, jit_op_times)):
        if jit > 0:
            speedup = nojit / jit
            ax.text(i, max(nojit, jit) + 0.001, f"{speedup:.1f}x", 
                   ha='center', fontsize=10, color='black')
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("jit_operation_comparison.pdf")
    print("Saved figure as 'jit_operation_comparison.pdf'")

def main():
    """Run the JIT acceleration demonstration."""
    print("LVEC JIT Acceleration Demo")
    print("=========================\n")
    
    # Check if JIT is available
    check_jit_status()
    
    # Only continue with benchmarks if JIT is available
    if not is_jit_available():
        print("\n JIT is not available, install numba to see performance benefits")
        print("   Continuing demonstration with JIT disabled...")
    
    # Run the benchmarks
    jit_results = benchmark_jit_performance()
    cache_results = benchmark_jit_with_caching()
    operation_results = benchmark_vector_operations()
    
    # Plot the results if matplotlib is available
    try:
        plot_benchmark_results(jit_results, operation_results)
    except Exception as e:
        print(f"\nCould not create plots: {e}")
    
    
    print("\nDemo completed! ")

if __name__ == "__main__":
    main()
