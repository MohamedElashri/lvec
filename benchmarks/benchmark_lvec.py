import numpy as np
import timeit
import matplotlib.pyplot as plt
from lvec import LVec
import vector
import tracemalloc
import gc

def get_process_memory():
    """Get memory usage in MB for the current process."""
    # Removed this function as it is no longer used

def measure_memory_usage(operation, size, n_repeats=10):
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

def measure_single_timing(operation, n_repeats=10):
    """Measure timing multiple times and return mean and std."""
    times = []
    for _ in range(n_repeats):
        time = timeit.timeit(operation, number=100) / 100
        times.append(time)
    return np.mean(times), np.std(times)

def benchmark_lvec_vs_vector(sizes, n_repeats=10):
    """Compare performance between LVec and vector package operations."""
    lvec_times = []
    lvec_errors = []
    vector_times = []
    vector_errors = []
    lvec_memory = []
    vector_memory = []
    
    for size in sizes:
        px, py, pz, E = generate_test_data(size)
        
        # Benchmark LVec
        def lvec_operation():
            vec = LVec(px, py, pz, E)
            return vec.mass
            
        lvec_mean, lvec_std = measure_single_timing(lvec_operation, n_repeats)
        lvec_times.append(lvec_mean)
        lvec_errors.append(lvec_std)
        lvec_mem = measure_memory_usage(lvec_operation, size, n_repeats)
        lvec_memory.append(lvec_mem)
        
        # Benchmark vector package
        def vector_operation():
            vec = vector.arr({'px': px, 'py': py, 'pz': pz, 'E': E})
            return vec.mass
            
        vector_mean, vector_std = measure_single_timing(vector_operation, n_repeats)
        vector_times.append(vector_mean)
        vector_errors.append(vector_std)
        vector_mem = measure_memory_usage(vector_operation, size, n_repeats)
        vector_memory.append(vector_mem)
        
        print(f"Size {size:,}:")
        print(f"  LVec:   {lvec_mean*1000:.3f} ± {lvec_std*1000:.3f} ms, {lvec_mem:.1f} MB")
        print(f"  Vector: {vector_mean*1000:.3f} ± {vector_std*1000:.3f} ms, {vector_mem:.1f} MB")
        print(f"  Ratio:  {vector_mean/lvec_mean:.2f}x\n")
    
    return (np.array(lvec_times), np.array(lvec_errors), 
            np.array(vector_times), np.array(vector_errors),
            np.array(lvec_memory), np.array(vector_memory))

def plot_results(sizes, lvec_data, vector_data):
    """Plot benchmark results with error bars and ratio plot."""
    lvec_times, lvec_errors, lvec_memory = lvec_data
    vector_times, vector_errors, vector_memory = vector_data
    
    # Convert to milliseconds
    lvec_times *= 1000
    lvec_errors *= 1000
    vector_times *= 1000
    vector_errors *= 1000
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 2], hspace=0.1)
    
    # Upper plot: timing comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(sizes, lvec_times, yerr=lvec_errors, 
                fmt='o-', label='LVec', capsize=3)
    ax1.errorbar(sizes, vector_times, yerr=vector_errors, 
                fmt='o-', label='vector', capsize=3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Time per operation (ms)')
    ax1.set_title('Performance Comparison: LVec vs vector package')
    ax1.grid(True)
    ax1.legend()
    
    # Middle plot: ratio
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ratio = vector_times / lvec_times
    ratio_error = ratio * np.sqrt(
        (vector_errors/vector_times)**2 + 
        (lvec_errors/lvec_times)**2
    )
    
    ax2.errorbar(sizes, ratio, yerr=ratio_error, fmt='o-', color='black',
                capsize=3)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Vector/LVec')
    ax2.grid(True)
    
    # Set reasonable ratio plot limits
    median_ratio = np.median(ratio)
    ax2.set_ylim(0.5 * median_ratio, 1.5 * median_ratio)
    
    # Bottom plot: memory usage
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(sizes, lvec_memory, 'o-', label='LVec')
    ax3.plot(sizes, vector_memory, 'o-', label='vector')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Array Size')
    ax3.set_ylabel('Memory Usage (MB)')
    ax3.grid(True)
    ax3.legend()
    
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Test with different array sizes
    sizes = [10, 100, 1000, 10000, 100000, 1000000]
    lvec_times, lvec_errors, vector_times, vector_errors, lvec_memory, vector_memory = benchmark_lvec_vs_vector(sizes)
    plot_results(sizes, (lvec_times, lvec_errors, lvec_memory), 
                (vector_times, vector_errors, vector_memory))
