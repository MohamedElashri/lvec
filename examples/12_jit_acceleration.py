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
6. Pre-compilation to eliminate first-run overhead
7. Adaptive batch processing thresholds

Note: This example requires the numba package to demonstrate JIT acceleration.
If numba is not installed, the example will show how lvec gracefully falls back
to non-JIT implementations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from lvec import (LVec, Vector3D, is_jit_available, enable_jit, 
                 precompile_jit_functions, get_jit_batch_threshold)

def check_jit_status():
    """Check if JIT compilation is available and enabled."""
    print("\n=== JIT Acceleration Status ===")
    
    if is_jit_available():
        print(" JIT compilation is available")
        print("   Using numba for accelerated computations")
        print(f"   Current batch threshold: {get_jit_batch_threshold():,}")
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

def demonstrate_precompilation():
    """Demonstrate the benefits of pre-compilation."""
    print("\n=== JIT Pre-compilation Demo ===")
    
    if not is_jit_available():
        print("JIT is not available, skipping pre-compilation demo")
        return
    
    # Create test data
    n_particles = 1_000_000
    rng = np.random.RandomState(42)
    px = rng.normal(0, 10, n_particles)
    py = rng.normal(0, 10, n_particles)
    pz = rng.normal(0, 10, n_particles)
    E = np.sqrt(px**2 + py**2 + pz**2 + 0.14**2)
    
    # First, disable JIT to reset any compiled functions
    enable_jit(False)
    enable_jit(True)
    
    print("\n1. Without pre-compilation (first run includes compilation overhead)")
    # Create vectors and time the first access to properties
    vectors = LVec(px, py, pz, E)
    
    start_time = time.time()
    _ = vectors.pt
    pt_time = time.time() - start_time
    print(f"   First pt access: {pt_time:.6f} seconds")
    
    start_time = time.time()
    _ = vectors.mass
    mass_time = time.time() - start_time
    print(f"   First mass access: {mass_time:.6f} seconds")
    
    # Now reset and use pre-compilation
    enable_jit(False)
    enable_jit(True)
    
    print("\n2. With pre-compilation (compilation done ahead of time)")
    # Pre-compile JIT functions
    start_time = time.time()
    precompile_jit_functions()
    precompile_time = time.time() - start_time
    print(f"   Pre-compilation time: {precompile_time:.6f} seconds")
    
    # Create vectors and time the first access to properties
    vectors = LVec(px, py, pz, E)
    
    start_time = time.time()
    _ = vectors.pt
    pt_time_precompiled = time.time() - start_time
    print(f"   First pt access: {pt_time_precompiled:.6f} seconds")
    
    start_time = time.time()
    _ = vectors.mass
    mass_time_precompiled = time.time() - start_time
    print(f"   First mass access: {mass_time_precompiled:.6f} seconds")
    
    # Calculate speedup
    if pt_time > 0 and pt_time_precompiled > 0:
        speedup = pt_time / pt_time_precompiled
        print(f"\nSpeedup for first pt access: {speedup:.1f}x")
    
    if mass_time > 0 and mass_time_precompiled > 0:
        speedup = mass_time / mass_time_precompiled
        print(f"Speedup for first mass access: {speedup:.1f}x")
    
    return {
        "without_precompile": {"pt": pt_time, "mass": mass_time},
        "with_precompile": {"pt": pt_time_precompiled, "mass": mass_time_precompiled},
        "precompile_time": precompile_time
    }

def benchmark_jit_performance():
    """Benchmark the performance difference between JIT-enabled and disabled code."""
    print("\n=== JIT Performance Benchmark ===")
    
    # Create a large dataset for benchmarking
    n_particles = 20_000_000  # Increased from 10M to 20M for better demonstration
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
    
    for jit_enabled in [False, True]:  # Test non-JIT first, then JIT
        # Set JIT status
        enable_jit(jit_enabled)
        status = "enabled" if jit_enabled else "disabled"
        print(f"\nRunning benchmark with JIT {status}...")
        
        if jit_enabled:
            # Pre-compile JIT functions to avoid compilation overhead
            precompile_jit_functions()
            print(f"  Pre-compiled JIT functions")
            print(f"  Current batch threshold: {get_jit_batch_threshold():,}")
        
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
    
    # Calculate and display speedup
    if False in results and True in results:
        print("\n=== JIT Speedup Summary ===")
        for phase in ["cold", "second"]:
            speedup = results[False][phase] / results[True][phase]
            print(f"  JIT speedup for {phase} run: {speedup:.2f}x")
    
    return results

def benchmark_jit_with_caching():
    """Show how JIT acceleration works together with the caching system."""
    print("\n=== JIT with Caching System ===")
    
    # Enable JIT for this test
    enable_jit(True)
    precompile_jit_functions()
    
    # Create test data
    n_particles = 1_000_000  # Increased from 100K to 1M
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
    n_particles = 5_000_000  # Increased from 500K to 5M
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
    
    for jit_enabled in [False, True]:  # Test non-JIT first, then JIT
        enable_jit(jit_enabled)
        status = "enabled" if jit_enabled else "disabled"
        print(f"\nRunning with JIT {status}...")
        
        if jit_enabled:
            # Pre-compile JIT functions to avoid compilation overhead
            precompile_jit_functions()
        
        # Create vectors
        vectors = LVec(px, py, pz, E)
        
        # Benchmark each operation
        op_results = {}
        for name, op in operations.items():
            # Clear cache to ensure cold start
            vectors.clear_cache()
            
            # Time the operation
            start_time = time.time()
            _ = op(vectors)
            op_time = time.time() - start_time
            
            print(f"  {name}: {op_time:.6f} seconds")
            op_results[name] = op_time
        
        results[jit_enabled] = op_results
    
    # Calculate speedups
    if False in results and True in results:
        print("\n=== Operation-specific JIT Speedups ===")
        for op in operations:
            speedup = results[False][op] / results[True][op]
            print(f"  {op}: {speedup:.2f}x faster with JIT")
    
    return results

def benchmark_array_sizes():
    """Benchmark JIT performance across different array sizes to show adaptive thresholds."""
    print("\n=== JIT Performance Across Array Sizes ===")
    
    # Test different array sizes
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    operations = ["pt", "mass"]
    
    results = {"sizes": sizes, "operations": {}}
    
    # Enable JIT and pre-compile
    enable_jit(True)
    precompile_jit_functions()
    
    for op in operations:
        print(f"\nBenchmarking {op} calculation across array sizes:")
        op_results = {"jit": [], "nojit": [], "speedup": []}
        
        for size in sizes:
            print(f"  Array size: {size:,}")
            
            # Generate data
            rng = np.random.RandomState(42)
            px = rng.normal(0, 10, size)
            py = rng.normal(0, 10, size)
            pz = rng.normal(0, 10, size)
            E = np.sqrt(px**2 + py**2 + pz**2 + 0.14**2)
            
            # Test with JIT enabled
            enable_jit(True)
            vectors_jit = LVec(px, py, pz, E)
            vectors_jit.clear_cache()
            
            start_time = time.time()
            if op == "pt":
                _ = vectors_jit.pt
            elif op == "mass":
                _ = vectors_jit.mass
            jit_time = time.time() - start_time
            
            # Test with JIT disabled
            enable_jit(False)
            vectors_nojit = LVec(px, py, pz, E)
            vectors_nojit.clear_cache()
            
            start_time = time.time()
            if op == "pt":
                _ = vectors_nojit.pt
            elif op == "mass":
                _ = vectors_nojit.mass
            nojit_time = time.time() - start_time
            
            # Calculate speedup
            speedup = nojit_time / jit_time if jit_time > 0 else 0
            
            print(f"    JIT: {jit_time:.6f}s, Non-JIT: {nojit_time:.6f}s, Speedup: {speedup:.2f}x")
            
            op_results["jit"].append(jit_time)
            op_results["nojit"].append(nojit_time)
            op_results["speedup"].append(speedup)
        
        results["operations"][op] = op_results
    
    return results

def plot_benchmark_results(jit_results, operation_results, precompile_results=None, array_size_results=None):
    """Plot benchmark results for visualization."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: JIT vs non-JIT for different phases
    plt.subplot(2, 2, 1)
    phases = ['cold', 'second', 'warm']
    jit_times = [jit_results[True][p] for p in phases]
    nojit_times = [jit_results[False][p] for p in phases]
    
    x = np.arange(len(phases))
    width = 0.35
    
    plt.bar(x - width/2, nojit_times, width, label='JIT Disabled')
    plt.bar(x + width/2, jit_times, width, label='JIT Enabled')
    
    plt.xlabel('Benchmark Phase')
    plt.ylabel('Time (seconds)')
    plt.title('JIT vs non-JIT Performance')
    plt.xticks(x, phases)
    plt.legend()
    
    # Add speedup annotations
    for i, (nojit, jit) in enumerate(zip(nojit_times, jit_times)):
        speedup = nojit / jit if jit > 0 else 0
        plt.text(i, max(nojit, jit) + 0.05, f'{speedup:.1f}x', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Operation-specific performance
    plt.subplot(2, 2, 2)
    ops = list(operation_results[True].keys())
    jit_op_times = [operation_results[True][op] for op in ops]
    nojit_op_times = [operation_results[False][op] for op in ops]
    
    x = np.arange(len(ops))
    
    plt.bar(x - width/2, nojit_op_times, width, label='JIT Disabled')
    plt.bar(x + width/2, jit_op_times, width, label='JIT Enabled')
    
    plt.xlabel('Operation')
    plt.ylabel('Time (seconds)')
    plt.title('Operation-specific JIT Performance')
    plt.xticks(x, ops)
    plt.legend()
    
    # Add speedup annotations
    for i, (nojit, jit) in enumerate(zip(nojit_op_times, jit_op_times)):
        speedup = nojit / jit if jit > 0 else 0
        plt.text(i, max(nojit, jit) + 0.01, f'{speedup:.1f}x', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Precompilation benefits or speedup factors
    plt.subplot(2, 2, 3)
    
    if precompile_results:
        # Plot precompilation benefits
        operations = ["pt", "mass"]
        without_precompile = [precompile_results["without_precompile"][op] for op in operations]
        with_precompile = [precompile_results["with_precompile"][op] for op in operations]
        
        x = np.arange(len(operations))
        
        plt.bar(x - width/2, without_precompile, width, label='Without Precompilation')
        plt.bar(x + width/2, with_precompile, width, label='With Precompilation')
        
        plt.xlabel('Operation')
        plt.ylabel('Time (seconds)')
        plt.title('Precompilation Benefits')
        plt.xticks(x, operations)
        plt.legend()
        
        # Add speedup annotations
        for i, (wo, w) in enumerate(zip(without_precompile, with_precompile)):
            speedup = wo / w if w > 0 else 0
            plt.text(i, max(wo, w) + 0.001, f'{speedup:.1f}x', 
                    ha='center', va='bottom', fontweight='bold')
    else:
        # Calculate speedups
        phase_speedups = [nojit_times[i]/jit_times[i] if jit_times[i] > 0 else 0 
                         for i in range(len(phases))]
        op_speedups = [nojit_op_times[i]/jit_op_times[i] if jit_op_times[i] > 0 else 0 
                      for i in range(len(ops))]
        
        categories = phases + ops
        speedups = phase_speedups + op_speedups
        colors = ['blue']*len(phases) + ['green']*len(ops)
        
        plt.bar(categories, speedups, color=colors)
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
        
        plt.xlabel('Category')
        plt.ylabel('Speedup Factor (higher is better)')
        plt.title('JIT Speedup Factors')
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(speedups):
            plt.text(i, v + 0.1, f'{v:.1f}x', ha='center', va='bottom')
    
    # Plot 4: Array size performance or cache vs JIT
    plt.subplot(2, 2, 4)
    
    if array_size_results:
        # Plot array size performance
        sizes = array_size_results["sizes"]
        op = "mass"  # Choose one operation to display
        
        if op in array_size_results["operations"]:
            speedups = array_size_results["operations"][op]["speedup"]
            
            plt.semilogx(sizes, speedups, 'o-', linewidth=2, markersize=8)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            
            plt.xlabel('Array Size')
            plt.ylabel('Speedup Factor (higher is better)')
            plt.title(f'JIT Speedup vs Array Size for {op}')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (size, speedup) in enumerate(zip(sizes, speedups)):
                plt.text(size, speedup + 0.1, f'{speedup:.1f}x', 
                        ha='center', va='bottom')
    else:
        # This assumes benchmark_jit_with_caching has been run
        # and returned the times
        try:
            first_time, cached_time, nojit_time = benchmark_jit_with_caching()
            
            times = [nojit_time, first_time, cached_time]
            labels = ['non-JIT', 'JIT', 'Cached']
            
            plt.bar(labels, times)
            plt.yscale('log')  # Log scale to show the dramatic difference with caching
            
            plt.xlabel('Access Method')
            plt.ylabel('Time (seconds, log scale)')
            plt.title('JIT vs Caching Performance')
            
            # Add speedup annotations
            jit_speedup = nojit_time / first_time if first_time > 0 else 0
            cache_speedup = first_time / cached_time if cached_time > 0 else 0
            total_speedup = nojit_time / cached_time if cached_time > 0 else 0
            
            plt.text(0, nojit_time, f'{nojit_time:.6f}s', ha='center', va='bottom')
            plt.text(1, first_time, f'{first_time:.6f}s\n({jit_speedup:.1f}x faster)', ha='center', va='bottom')
            plt.text(2, cached_time, f'{cached_time:.6f}s\n({total_speedup:.1f}x faster)', ha='center', va='bottom')
            
        except Exception as e:
            plt.text(0.5, 0.5, f"Error generating cache vs JIT plot: {e}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('jit_benchmark_results.pdf')
    print("\nBenchmark plot saved as 'jit_benchmark_results.pdf'")
    plt.show()

def main():
    """Run the JIT acceleration demonstration."""
    print("=" * 80)
    print("LVec JIT Acceleration Demonstration")
    print("=" * 80)
    
    # Check if JIT is available
    check_jit_status()
    
    # Demonstrate precompilation benefits
    precompile_results = demonstrate_precompilation()
    
    # Run the benchmarks
    jit_results = benchmark_jit_performance()
    operation_results = benchmark_vector_operations()
    
    # Benchmark different array sizes to show adaptive thresholds
    array_size_results = benchmark_array_sizes()
    
    # Plot benchmark results for visualization
    plot_benchmark_results(jit_results, operation_results, 
                          precompile_results, array_size_results)
    
    print("\n" + "=" * 80)
    print("JIT Acceleration Best Practices:")
    print("=" * 80)
    print("1. JIT works best with large NumPy arrays (>100K elements)")
    print("2. Use precompile_jit_functions() to eliminate first-run compilation overhead")
    print("3. For maximum performance, combine JIT with the caching system")
    print("4. JIT only works with NumPy arrays, not with Awkward arrays")
    print("5. Enable JIT globally with: from lvec import enable_jit; enable_jit(True)")
    print("6. The adaptive threshold automatically optimizes batch processing for your hardware")
    print("=" * 80)

if __name__ == "__main__":
    main()
