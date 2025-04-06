import numpy as np
import timeit
import matplotlib.pyplot as plt
import tracemalloc
import gc
import time
import os
from functools import partial
from lvec import LVec, Vector2D, Vector3D
import vector  # Comparison library
from plotting_utils import plot_vector_types_comparison, set_publication_style

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
        
    elif vector_type == "Scikit Vector":  # Changed from "Vector" to "Scikit Vector"
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
        
    elif vector_type == "Scikit Vector":  # Changed from "Vector" to "Scikit Vector"
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
    plot_vector_types_comparison(
        sizes, 
        results, 
        vector_types, 
        operations, 
        title='Arithmetic Operations Performance', 
        filename=save_path
    )

def plot_derived_results(sizes, results, vector_types, properties, save_path="benchmark_derived.pdf"):
    """Plot derived properties benchmark results."""
    plot_vector_types_comparison(
        sizes, 
        results, 
        vector_types, 
        properties, 
        title='Derived Properties Performance', 
        filename=save_path
    )

def plot_caching_results(sizes, cache_results, save_path="benchmark_caching.pdf"):
    """Plot caching effectiveness benchmark results."""
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to milliseconds
    cached_times = np.array([res["cached"] for res in cache_results]) * 1000
    uncached_times = np.array([res["uncached"] for res in cache_results]) * 1000
    
    ax.plot(sizes, cached_times, 'o-', label='With Caching', color='#109618', linewidth=2, markersize=6)
    ax.plot(sizes, uncached_times, 'o-', label='Without Caching', color='#FF9900', linewidth=2, markersize=6)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Caching Effectiveness in lvec', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join('benchmarks/plots', save_path), bbox_inches='tight')
    plt.close()

def run_benchmarks():
    """Run all benchmarks and plot results."""
    sizes = [10, 100, 1000, 10000, 100000, 1000000]
    vector_types = ["LVec", "Vector2D", "Vector3D", "Scikit Vector"]
    
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
    # Remove operations that don't have Scikit Vector equivalents
    arith_ops = ["addition", "subtraction", "scalar_mul"]  # Removed dot_product and cross_product
    derived_props = ["mass", "pt", "eta", "phi"]  # Removed magnitude, theta, rho
    
    plot_arithmetic_results(sizes, arith_results, vector_types, arith_ops, "benchmarks/plots/benchmark_arithmetic.pdf")
    plot_derived_results(sizes, derived_results, vector_types, derived_props, "benchmarks/plots/benchmark_derived.pdf")
    plot_caching_results(sizes, cache_results, "benchmarks/plots/benchmark_caching.pdf")
    
    print("\nBenchmarks completed. Plots saved to:")
    print("  - benchmarks/plots/benchmark_arithmetic.pdf")
    print("  - benchmarks/plots/benchmark_derived.pdf")
    print("  - benchmarks/plots/benchmark_caching.pdf")

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs("benchmarks/plots", exist_ok=True)
    run_benchmarks()