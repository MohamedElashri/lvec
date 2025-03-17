import numpy as np
import timeit
import matplotlib.pyplot as plt
import vector
import tracemalloc
import gc
from lvec import LVec
from lvec.lvec_opt import LVecOpt

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
    pt = np.random.uniform(1, 100, size)
    eta = np.random.uniform(-5, 5, size)
    phi = np.random.uniform(-np.pi, np.pi, size)
    mass = np.full(size, 0.105)  # muon mass
    
    # Pre-compute px, py, pz, E for vector and normal LVec
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    sinh_eta = np.sinh(eta)
    
    px = pt * cos_phi
    py = pt * sin_phi
    pz = pt * sinh_eta
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    
    return {
        'ptepm': (pt, eta, phi, mass),
        'pxpypzE': (px, py, pz, E)
    }

def measure_single_timing(operation, n_repeats=10):
    """Measure timing multiple times and return mean and std."""
    times = []
    for _ in range(n_repeats):
        time = timeit.timeit(operation, number=100) / 100
        times.append(time)
    return np.mean(times), np.std(times)

def benchmark_all_implementations(sizes, n_repeats=10):
    """Compare performance between LVec, LVecOpt and vector package operations."""
    results = {
        'lvec': {'times': [], 'errors': [], 'memory': []},
        'lvec_opt': {'times': [], 'errors': [], 'memory': []},
        'vector': {'times': [], 'errors': [], 'memory': []}
    }
    
    operations = {
        'creation': lambda v: v,
        'pt': lambda v: v.pt,
        'mass': lambda v: v.mass,
        'eta': lambda v: v.eta,
        'phi': lambda v: v.phi
    }
    
    for size in sizes:
        print(f"\nBenchmarking size {size:,}")
        data = generate_test_data(size)
        pt, eta, phi, mass = data['ptepm']
        px, py, pz, E = data['pxpypzE']
        
        # Test each implementation
        for op_name, op_func in operations.items():
            print(f"\nOperation: {op_name}")
            
            # LVec
            def lvec_operation():
                vec = LVec(px, py, pz, E)
                return op_func(vec)
            
            lvec_mean, lvec_std = measure_single_timing(lvec_operation, n_repeats)
            lvec_mem = measure_memory_usage(lvec_operation)
            
            # LVecOpt
            def lvec_opt_operation():
                vec = LVecOpt(px, py, pz, E)
                return op_func(vec)
            
            lvec_opt_mean, lvec_opt_std = measure_single_timing(lvec_opt_operation, n_repeats)
            lvec_opt_mem = measure_memory_usage(lvec_opt_operation)
            
            # Vector package
            def vector_operation():
                vec = vector.arr({'px': px, 'py': py, 'pz': pz, 'E': E})
                return op_func(vec)
            
            vector_mean, vector_std = measure_single_timing(vector_operation, n_repeats)
            vector_mem = measure_memory_usage(vector_operation)
            
            # Store results
            if op_name == 'creation':
                results['lvec']['times'].append(lvec_mean)
                results['lvec']['errors'].append(lvec_std)
                results['lvec']['memory'].append(lvec_mem)
                
                results['lvec_opt']['times'].append(lvec_opt_mean)
                results['lvec_opt']['errors'].append(lvec_opt_std)
                results['lvec_opt']['memory'].append(lvec_opt_mem)
                
                results['vector']['times'].append(vector_mean)
                results['vector']['errors'].append(vector_std)
                results['vector']['memory'].append(vector_mem)
            
            # Print current results
            print(f"  LVec:     {lvec_mean*1000:.3f} ± {lvec_std*1000:.3f} ms, {lvec_mem:.1f} MB")
            print(f"  LVecOpt:  {lvec_opt_mean*1000:.3f} ± {lvec_opt_std*1000:.3f} ms, {lvec_opt_mem:.1f} MB")
            print(f"  Vector:   {vector_mean*1000:.3f} ± {vector_std*1000:.3f} ms, {vector_mem:.1f} MB")
            print(f"  Speedup:  LVecOpt is {lvec_mean/lvec_opt_mean:.2f}x faster than LVec")
    
    return results

def plot_benchmark_results(sizes, results):
    """Plot benchmark results comparing all implementations."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Performance plot
    ax1 = fig.add_subplot(gs[0])
    
    # Convert to milliseconds
    for impl in results:
        results[impl]['times'] = np.array(results[impl]['times']) * 1000
        results[impl]['errors'] = np.array(results[impl]['errors']) * 1000
    
    # Plot timing results
    ax1.errorbar(sizes, results['lvec']['times'], yerr=results['lvec']['errors'],
                fmt='o-', label='LVec', capsize=3)
    ax1.errorbar(sizes, results['lvec_opt']['times'], yerr=results['lvec_opt']['errors'],
                fmt='o-', label='LVecOpt', capsize=3)
    ax1.errorbar(sizes, results['vector']['times'], yerr=results['vector']['errors'],
                fmt='o-', label='vector', capsize=3)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Time per operation (ms)')
    ax1.set_title('Performance Comparison')
    ax1.grid(True)
    ax1.legend()
    
    # Memory usage plot
    ax2 = fig.add_subplot(gs[1])
    
    ax2.plot(sizes, results['lvec']['memory'], 'o-', label='LVec')
    ax2.plot(sizes, results['lvec_opt']['memory'], 'o-', label='LVecOpt')
    ax2.plot(sizes, results['vector']['memory'], 'o-', label='vector')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Array Size')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.grid(True)
    ax2.legend()
    
    plt.savefig('benchmark_comparison_jit.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Test with different array sizes
    sizes = [10, 100, 1000, 10000, 100000, 1000000]
    results = benchmark_all_implementations(sizes)
    plot_benchmark_results(sizes, results)
