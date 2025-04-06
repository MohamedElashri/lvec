import numpy as np
import timeit
import matplotlib.pyplot as plt
import tracemalloc
import gc
import os
from lvec import LVec
import vector
from plotting_utils import plot_combined_performance, set_publication_style, COLORS

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
    """Compare performance between lvec and vector package operations."""
    lvec_times = []
    lvec_errors = []
    vector_times = []
    vector_errors = []
    lvec_memory = []
    vector_memory = []
    
    for size in sizes:
        px, py, pz, E = generate_test_data(size)
        
        # Benchmark lvec
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
        print(f"  lvec:   {lvec_mean*1000:.3f} ± {lvec_std*1000:.3f} ms, {lvec_mem:.1f} MB")
        print(f"  vector: {vector_mean*1000:.3f} ± {vector_std*1000:.3f} ms, {vector_mem:.1f} MB")
        print(f"  Ratio:  {vector_mean/lvec_mean:.2f}x\n")
    
    return (np.array(lvec_times), np.array(lvec_errors), 
            np.array(vector_times), np.array(vector_errors),
            np.array(lvec_memory), np.array(vector_memory))

def plot_results(sizes, lvec_data, vector_data):
    """Plot benchmark results using standardized plotting utilities."""
    lvec_times, lvec_errors, lvec_memory = lvec_data
    vector_times, vector_errors, vector_memory = vector_data
    
    # Use the standardized plotting utility
    plot_combined_performance(
        sizes,
        lvec_times,
        vector_times,
        lvec_memory,
        vector_memory,
        title="lvec vs vector Performance Comparison",
        filename="lvec_vs_vector_benchmark.pdf"
    )

if __name__ == '__main__':
    # Create plots directory if it doesn't exist
    os.makedirs("benchmarks/plots", exist_ok=True)
    
    # Test with different array sizes
    sizes = [10, 100, 1000, 10000, 100000, 1000000]
    results = benchmark_lvec_vs_vector(sizes)
    
    # Plot results
    plot_results(
        sizes, 
        (results[0], results[1], results[4]), 
        (results[2], results[3], results[5])
    )
    
    print("Benchmark completed. Results saved to PDF file.")
