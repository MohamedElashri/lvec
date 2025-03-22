import numpy as np
import timeit
import matplotlib.pyplot as plt
import tracemalloc
import gc
from lvec import LVec
import vector

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

def benchmark_operation(operation_name, size, n_repeats=10):
    """Benchmark a specific operation for both LVec and vector packages."""
    px, py, pz, E = generate_test_data(size)
    
    # Create vectors once to avoid initialization overhead in timing
    lvec = LVec(px, py, pz, E)
    vec = vector.arr({'px': px, 'py': py, 'pz': pz, 'E': E})
    
    operations = {
        'mass': (
            lambda: lvec.mass,
            lambda: vec.mass
        ),
        'pt': (
            lambda: lvec.pt,
            lambda: vec.pt
        ),
        'eta': (
            lambda: lvec.eta,
            lambda: vec.eta
        ),
        'phi': (
            lambda: lvec.phi,
            lambda: vec.phi
        ),
        'E': (
            lambda: lvec.E,
            lambda: vec.E
        ),
        'add': (
            lambda: lvec + lvec,
            lambda: vec + vec
        ),
        'boost': (
            lambda: lvec.boost(0.5, 0.0, 0.0),  # Using x=0.5, y=0, z=0 for x-direction boost
            lambda: vec.boostX(0.5)
        )
    }
    
    if operation_name not in operations:
        raise ValueError(f"Unknown operation: {operation_name}")
        
    lvec_op, vector_op = operations[operation_name]
    
    # Measure timing
    lvec_time, lvec_error = measure_single_timing(lvec_op, n_repeats)
    vector_time, vector_error = measure_single_timing(vector_op, n_repeats)
    
    # Measure memory
    lvec_memory = measure_memory_usage(lvec_op)
    vector_memory = measure_memory_usage(vector_op)
    
    return {
        'lvec': {'time': lvec_time, 'error': lvec_error, 'memory': lvec_memory},
        'vector': {'time': vector_time, 'error': vector_error, 'memory': vector_memory}
    }

def plot_all_operations(sizes, all_results, operations):
    """Plot all operation comparisons in subplots."""
    plt.style.use('default')
    n_ops = len(operations)
    n_cols = 3
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
        ax.plot(sizes, lvec_times, 'o-', label='lvec', color='#3498db', 
                linewidth=2, markersize=6)
        ax.plot(sizes, vector_times, 'o-', label='vector', color='#9b59b6', 
                linewidth=2, markersize=6)
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
        fig.delaxes(fig.add_subplot(gs[row, col]))
    
    plt.suptitle('Performance Comparison of Operations', fontsize=14, y=1.02)
    plt.savefig('benchmarks/plots/benchmark_all_operations.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Test with different array sizes
    sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    operations = ['mass', 'pt', 'eta', 'phi', 'E', 'add', 'boost']  
    
    # Store results for all operations
    all_results = {}
    
    for operation in operations:
        print(f"\nBenchmarking {operation}...")
        results = []
        for size in sizes:
            print(f"  Size {size:,}")
            result = benchmark_operation(operation, size)
            results.append(result)
            
            # Print current results
            lvec_time = result['lvec']['time'] * 1000
            vector_time = result['vector']['time'] * 1000
            ratio = vector_time / lvec_time
            print(f"    LVec:   {lvec_time:.3f} ms")
            print(f"    Vector: {vector_time:.3f} ms")
            print(f"    Ratio:  {ratio:.2f}x")
        
        all_results[operation] = results
    
    # Plot all operations in one figure
    plot_all_operations(sizes, all_results, operations)
    print("\nPlot saved as benchmarks/plots/benchmark_all_operations.pdf")
