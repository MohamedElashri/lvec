#!/usr/bin/env python3
"""
Cache Performance Analysis
=========================

This example demonstrates how to use the cache instrumentation
to measure the performance of the caching system in LVEC.

It shows:
1. How to enable instrumentation
2. How to track hit/miss ratios
3. How to analyze cache performance for optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from lvec import LVec, Vector2D, Vector3D

def analyze_lvec_caching():
    """Analyze cache hit ratios for Lorentz vectors."""
    print("\n=== Lorentz Vector Cache Performance ===")
    
    # Create a vector with random components
    n_particles = 10000
    px = np.random.normal(0, 10, n_particles)
    py = np.random.normal(0, 10, n_particles)
    pz = np.random.normal(0, 10, n_particles)
    E = np.sqrt(px**2 + py**2 + pz**2 + 0.14**2)  # pion mass ~ 0.14 GeV
    
    vectors = LVec(px, py, pz, E)
    
    # Reset counters to start fresh
    vectors._cache.reset_counters()
    
    # First access to properties calculates and caches
    print("\nInitial property calculation (cache misses)...")
    _ = vectors.pt   # Transverse momentum
    _ = vectors.eta  # Pseudorapidity
    _ = vectors.phi  # Azimuthal angle
    _ = vectors.mass # Invariant mass
    
    # Print statistics after first access
    stats = vectors._cache.get_stats()
    print(f"Overall hit ratio: {stats['overall']['hit_ratio']:.2%}")
    print("\nProperty hit ratios after first access:")
    for prop, data in sorted(stats['properties'].items()):
        if data['total'] > 0 and not prop.endswith('_squared'):
            print(f"  {prop:5s}: {data['hit_ratio']:.2%} ({data['hits']} hits, {data['misses']} misses)")
    
    # Second access should use cached values
    print("\nSecond access (should be cache hits)...")
    _ = vectors.pt
    _ = vectors.eta
    _ = vectors.phi
    _ = vectors.mass
    
    # Print statistics after second access
    stats = vectors._cache.get_stats()
    print(f"Overall hit ratio: {stats['overall']['hit_ratio']:.2%}")
    print("\nProperty hit ratios after second access:")
    for prop, data in sorted(stats['properties'].items()):
        if data['total'] > 0 and not prop.endswith('_squared'):
            print(f"  {prop:5s}: {data['hit_ratio']:.2%} ({data['hits']} hits, {data['misses']} misses)")
    
    # Invalidate cache and access again
    print("\nInvalidating cache (touch) and accessing again...")
    vectors.touch()
    _ = vectors.pt
    _ = vectors.eta
    _ = vectors.phi
    _ = vectors.mass
    
    # Print final statistics
    stats = vectors._cache.get_stats()
    print(f"Overall hit ratio: {stats['overall']['hit_ratio']:.2%}")
    print("\nProperty hit ratios after cache invalidation and reaccess:")
    for prop, data in sorted(stats['properties'].items()):
        if data['total'] > 0 and not prop.endswith('_squared'):
            print(f"  {prop:5s}: {data['hit_ratio']:.2%} ({data['hits']} hits, {data['misses']} misses)")
    
    return stats

def analyze_vec2d_vec3d_caching():
    """Analyze cache hit ratios for 2D and 3D vectors."""
    print("\n=== Vector2D and Vector3D Cache Performance ===")
    
    # Create 2D and 3D vectors
    n_points = 10000
    x = np.random.normal(0, 10, n_points)
    y = np.random.normal(0, 10, n_points)
    z = np.random.normal(0, 10, n_points)
    
    vec2d = Vector2D(x, y)
    vec3d = Vector3D(x, y, z)
    
    # Reset counters
    vec2d._cache.reset_counters()
    vec3d._cache.reset_counters()
    
    # Analyze Vector2D
    print("\nVector2D Cache Performance:")
    _ = vec2d.r    # Magnitude
    _ = vec2d.phi  # Azimuthal angle
    stats_2d_first = vec2d._cache.get_stats()
    
    # Access again
    _ = vec2d.r
    _ = vec2d.phi
    stats_2d = vec2d._cache.get_stats()
    
    print(f"Overall hit ratio: {stats_2d['overall']['hit_ratio']:.2%}")
    for prop, data in sorted(stats_2d['properties'].items()):
        if data['total'] > 0 and not prop.endswith('_squared'):
            print(f"  {prop:5s}: {data['hit_ratio']:.2%} ({data['hits']} hits, {data['misses']} misses)")
    
    # Analyze Vector3D
    print("\nVector3D Cache Performance:")
    _ = vec3d.r     # Magnitude
    _ = vec3d.rho   # Cylindrical radius
    _ = vec3d.phi   # Azimuthal angle
    _ = vec3d.theta # Polar angle
    stats_3d_first = vec3d._cache.get_stats()
    
    # Access again
    _ = vec3d.r
    _ = vec3d.rho
    _ = vec3d.phi
    _ = vec3d.theta
    stats_3d = vec3d._cache.get_stats()
    
    print(f"Overall hit ratio: {stats_3d['overall']['hit_ratio']:.2%}")
    for prop, data in sorted(stats_3d['properties'].items()):
        if data['total'] > 0 and not prop.endswith('_squared'):
            print(f"  {prop:5s}: {data['hit_ratio']:.2%} ({data['hits']} hits, {data['misses']} misses)")
    
    return stats_2d, stats_3d

def plot_hit_ratios(lvec_stats, vec2d_stats, vec3d_stats, filename="cache_hit_ratios.pdf"):
    """Create a visualization of cache hit ratios."""
    plt.figure(figsize=(12, 8))
    
    # Filter out intermediate calculations and focus on main properties
    lvec_props = {k: v for k, v in lvec_stats['properties'].items() 
                 if not k.endswith('_squared') and v['total'] > 0}
    vec2d_props = {k: v for k, v in vec2d_stats['properties'].items() 
                  if not k.endswith('_squared') and v['total'] > 0}
    vec3d_props = {k: v for k, v in vec3d_stats['properties'].items() 
                  if not k.endswith('_squared') and v['total'] > 0}
    
    # Set up the plot
    ax = plt.subplot(111)
    
    # Set the positions and width for the bars
    ind = np.arange(3)
    width = 0.2
    
    # Colors for different vector types
    colors = {
        'LVec': '#3498db',    # Blue
        'Vector2D': '#2ecc71', # Green
        'Vector3D': '#e74c3c'  # Red
    }
    
    # Plot overall hit ratios
    overall_ratios = [
        lvec_stats['overall']['hit_ratio'],
        vec2d_stats['overall']['hit_ratio'],
        vec3d_stats['overall']['hit_ratio']
    ]
    ax.bar(ind, overall_ratios, width, color=[colors['LVec'], colors['Vector2D'], colors['Vector3D']], 
          alpha=0.8, label=['LVec', 'Vector2D', 'Vector3D'])
    
    # Styling
    ax.set_ylabel('Hit Ratio', fontsize=14)
    ax.set_title('Cache Hit Ratios by Vector Type', fontsize=16)
    ax.set_xticks(ind)
    ax.set_xticklabels(['LVec', 'Vector2D', 'Vector3D'])
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(overall_ratios):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center', fontsize=12)
    
    # Create a second plot for property-specific hit ratios
    plt.figure(figsize=(14, 10))
    
    # Collect all unique properties
    all_props = set(lvec_props.keys()) | set(vec2d_props.keys()) | set(vec3d_props.keys())
    
    # Sort properties for consistent ordering
    all_props = sorted(all_props)
    
    # Number of properties
    n_props = len(all_props)
    
    # Set up the plot
    ind = np.arange(n_props)
    
    # Create bars for each vector type
    bars1 = plt.bar(ind - width, 
                   [lvec_props.get(prop, {'hit_ratio': 0})['hit_ratio'] if prop in lvec_props else 0 
                    for prop in all_props], 
                   width, color=colors['LVec'], label='LVec')
    
    bars2 = plt.bar(ind, 
                   [vec2d_props.get(prop, {'hit_ratio': 0})['hit_ratio'] if prop in vec2d_props else 0 
                    for prop in all_props], 
                   width, color=colors['Vector2D'], label='Vector2D')
    
    bars3 = plt.bar(ind + width, 
                   [vec3d_props.get(prop, {'hit_ratio': 0})['hit_ratio'] if prop in vec3d_props else 0 
                    for prop in all_props], 
                   width, color=colors['Vector3D'], label='Vector3D')
    
    # Styling
    plt.ylabel('Hit Ratio', fontsize=14)
    plt.title('Cache Hit Ratios by Property and Vector Type', fontsize=16)
    plt.xticks(ind, all_props, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save both figures to a multi-page PDF
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(filename) as pdf:
        # Save the first figure
        pdf.savefig(1)
        # Save the second figure
        pdf.savefig(2)
    
    plt.close('all')
    print(f"\nSaved cache hit ratio visualizations to: {filename}")

def main():
    # Run cache performance analysis
    lvec_stats = analyze_lvec_caching()
    vec2d_stats, vec3d_stats = analyze_vec2d_vec3d_caching()
    
    # Create visualizations
    plot_hit_ratios(lvec_stats, vec2d_stats, vec3d_stats, "examples/plots/cache_hit_ratios.pdf")
    
    print("\nCache Performance Summary:")
    print("-------------------------")
    print(f"LVec overall hit ratio:     {lvec_stats['overall']['hit_ratio']:.2%}")
    print(f"Vector2D overall hit ratio: {vec2d_stats['overall']['hit_ratio']:.2%}")
    print(f"Vector3D overall hit ratio: {vec3d_stats['overall']['hit_ratio']:.2%}")

if __name__ == '__main__':
    main()
