import numpy as np
import timeit
import matplotlib.pyplot as plt
from lvec import LVec
import vector
import tracemalloc
import gc
import time

def measure_memory_usage(operation, n_repeats=5):
    """Measure memory usage for an operation."""
    memory_usages = []
    for _ in range(n_repeats):
        gc.collect()  # Clean up before measurement
        tracemalloc.start()
        result = operation()  # Execute operation
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak / 1024 / 1024  # Convert to MB
        memory_usages.append(memory_usage)
        del result  # Clean up
    return np.mean(memory_usages)

def generate_test_data(size):
    """Generate random 4-vectors for testing."""
    px = np.random.normal(0, 10, size)
    py = np.random.normal(0, 10, size)
    pz = np.random.normal(0, 10, size)
    E = np.sqrt(px**2 + py**2 + pz**2 + (0.105)**2)  # mass of muon
    return px, py, pz, E

def measure_initialization_time(init_function, n_repeats=5, number=10):
    """Measure initialization time with multiple repeats."""
    times = []
    for _ in range(n_repeats):
        time_taken = timeit.timeit(init_function, number=number) / number
        times.append(time_taken)
    return np.mean(times), np.std(times)

def benchmark_initialization_overhead(sizes, n_repeats=5):
    """
    Benchmark initialization overhead between LVec and vector package.
    
    Parameters:
    -----------
    sizes : list
        List of array sizes to benchmark
    n_repeats : int
        Number of repetitions for each measurement
    
    Returns:
    --------
    Tuple containing timing and memory data for both libraries
    """
    lvec_times = []
    lvec_errors = []
    vector_times = []
    vector_errors = []
    lvec_memory = []
    vector_memory = []
    
    for size in sizes:
        print(f"\nBenchmarking initialization with {size:,} vectors:")
        px, py, pz, E = generate_test_data(size)
        
        # Benchmark LVec initialization
        def lvec_init():
            return LVec(px, py, pz, E)
            
        # Determine number of iterations based on size (fewer for larger arrays)
        number = max(1, min(100, int(1e6 / size)))
        
        print(f"  Running {number} iterations per measurement")
        lvec_mean, lvec_std = measure_initialization_time(lvec_init, n_repeats, number)
        lvec_times.append(lvec_mean)
        lvec_errors.append(lvec_std)
        lvec_mem = measure_memory_usage(lvec_init, n_repeats)
        lvec_memory.append(lvec_mem)
        
        # Benchmark vector package initialization
        def vector_init():
            return vector.arr({'px': px, 'py': py, 'pz': pz, 'E': E})
            
        vector_mean, vector_std = measure_initialization_time(vector_init, n_repeats, number)
        vector_times.append(vector_mean)
        vector_errors.append(vector_std)
        vector_mem = measure_memory_usage(vector_init, n_repeats)
        vector_memory.append(vector_mem)
        
        print(f"  Results for {size:,} vectors:")
        print(f"    LVec:   {lvec_mean*1000:.3f} ± {lvec_std*1000:.3f} ms, {lvec_mem:.2f} MB")
        print(f"    Vector: {vector_mean*1000:.3f} ± {vector_std*1000:.3f} ms, {vector_mem:.2f} MB")
        print(f"    Speed Ratio:  {vector_mean/lvec_mean:.2f}x faster with LVec")
        print(f"    Memory Ratio: {vector_mem/lvec_mem:.2f}x more memory efficient with LVec")
    
    return (np.array(lvec_times), np.array(lvec_errors), 
            np.array(vector_times), np.array(vector_errors),
            np.array(lvec_memory), np.array(vector_memory))

def benchmark_batch_sizes():
    """Additional benchmark to test how initialization scales with batch size."""
    sizes = [10, 100, 1000, 10000, 100000, 1000000, 5000000]
    print("\n=== Benchmarking Initialization Overhead with Increasing Batch Sizes ===")
    
    results = benchmark_initialization_overhead(sizes)
    lvec_times, lvec_errors, vector_times, vector_errors, lvec_memory, vector_memory = results
    
    plot_results(sizes, 
                (lvec_times, lvec_errors, lvec_memory), 
                (vector_times, vector_errors, vector_memory),
                "Initialization Overhead vs Batch Size")
    
    return results

def benchmark_cached_initialization():
    """
    Benchmark to test how caching affects repeated initialization.
    This measures the overhead when repeatedly converting the same data.
    """
    size = 1000000  # Use 1M vectors for this test
    repeats = 10
    print("\n=== Benchmarking Cached Initialization Behavior ===")
    print(f"Testing with {size:,} vectors, measuring {repeats} consecutive initializations")
    
    px, py, pz, E = generate_test_data(size)
    
    # Measure LVec repeated initialization
    lvec_times = []
    print("  Measuring LVec repeated initialization...")
    for i in range(repeats):
        start = time.time()
        vec = LVec(px, py, pz, E)
        end = time.time()
        lvec_times.append((end - start) * 1000)  # convert to ms
        print(f"    Iteration {i+1}: {lvec_times[-1]:.3f} ms")
    
    # Measure vector repeated initialization
    vector_times = []
    print("  Measuring vector repeated initialization...")
    for i in range(repeats):
        start = time.time()
        vec = vector.arr({'px': px, 'py': py, 'pz': pz, 'E': E})
        end = time.time()
        vector_times.append((end - start) * 1000)  # convert to ms
        print(f"    Iteration {i+1}: {vector_times[-1]:.3f} ms")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, repeats+1), lvec_times, 'o-', label='LVec', color='#3498db', linewidth=2)
    plt.plot(range(1, repeats+1), vector_times, 'o-', label='vector', color='#9b59b6', linewidth=2)
    plt.xlabel('Initialization Iteration', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Repeated Initialization Performance (Caching Effects)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('benchmarks/plots/cached_initialization_benchmark.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return lvec_times, vector_times

def plot_results(sizes, lvec_data, vector_data, title="LVec vs vector Initialization Overhead"):
    """Plot benchmark results."""
    lvec_times, lvec_errors, lvec_memory = lvec_data
    vector_times, vector_errors, vector_memory = vector_data
    
    # Convert to milliseconds
    lvec_times *= 1000
    vector_times *= 1000
    
    # Create figure with two subplots
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    # Upper plot: timing comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sizes, lvec_times, 'o-', label='LVec', color='#3498db', linewidth=2, markersize=8)
    ax1.plot(sizes, vector_times, 'o-', label='vector', color='#9b59b6', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Initialization Time (ms)', fontsize=12)
    ax1.set_title(title, fontsize=14, pad=15)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=10)
    
    # Middle plot: memory usage
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(sizes, lvec_memory, 'o-', label='LVec', color='#2ecc71', linewidth=2, markersize=8)
    ax2.plot(sizes, vector_memory, 'o-', label='vector', color='#e74c3c', linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    ax2.tick_params(labelsize=10)
    
    # Bottom plot: performance ratio
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(sizes, vector_times / lvec_times, 'o-', label='Time Ratio (vector/LVec)', 
             color='#f39c12', linewidth=2, markersize=8)
    ax3.plot(sizes, vector_memory / lvec_memory, 'o-', label='Memory Ratio (vector/LVec)', 
             color='#16a085', linewidth=2, markersize=8)
    ax3.set_xscale('log')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Number of Vectors', fontsize=12)
    ax3.set_ylabel('Ratio (vector/LVec)', fontsize=12)
    ax3.grid(True, which='both', linestyle='--', alpha=0.7)
    ax3.legend(fontsize=12)
    ax3.tick_params(labelsize=10)
    
    # Add minor gridlines
    ax1.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax3.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    plt.savefig('benchmarks/plots/initialization_benchmark_results.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("=== LVec vs vector Initialization Overhead Benchmark ===")
    
    # Run main benchmarks
    batch_results = benchmark_batch_sizes()
    
    # Run the cached initialization benchmark
    cache_results = benchmark_cached_initialization()
    
    print("\nBenchmark completed. Results saved to PDF files.")
