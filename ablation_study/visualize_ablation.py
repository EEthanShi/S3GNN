"""
Visualization script for S³GNN and Stable-ChebNet ablation study results.

Creates two subplots: S³GNN (left) and Stable-ChebNet (right).

Usage:
    python visualize_ablation.py
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication-quality settings
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.weight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'legend.fontsize': 20,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,  # No grid
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
})

# Color scheme - very distinct colors
COLORS = {
    'id': '#E74C3C',         # Red - Identity (No Constraint)
    'as': '#2980B9',         # Blue - AntiSymmetric  
    'ortho_cay': '#27AE60',  # Green - Orthogonal Cayley
}

# Markers - very distinct
MARKERS = {
    'id': 'o',         # Circle
    'as': 's',         # Square
    'ortho_cay': '^',  # Triangle
}

WEIGHT_NAMES = {
    'id': 'No Constraint',
    'as': 'AntiSymmetric',
    'ortho_cay': 'Orthogonal',
}


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_weight_type(model_key):
    """Extract weight parameterization type from model key."""
    if '_id' in model_key:
        return 'id'
    elif '_as' in model_key:
        return 'as'
    elif '_ortho_cay' in model_key:
        return 'ortho_cay'
    elif '_ortho' in model_key:
        # Map old 'ortho' to 'ortho_cay' for backwards compatibility
        return 'ortho_cay'
    return None


def downsample(data, num_points=20):
    """Downsample data to approximately num_points."""
    if len(data) <= num_points:
        return list(range(len(data))), data
    
    step = len(data) // num_points
    indices = list(range(0, len(data), step))
    # Always include the last point
    if indices[-1] != len(data) - 1:
        indices.append(len(data) - 1)
    return indices, [data[i] for i in indices]


def plot_model_results(ax, results, model_name, num_points=20, show_ylabel=True):
    """Plot loss curves for one model type (S³GNN or Stable-ChebNet)."""
    
    weight_order = ['id', 'as', 'ortho_cay']
    
    for wtype in weight_order:
        # Find the key for this weight type
        key = None
        for k in results.keys():
            if extract_weight_type(k) == wtype:
                key = k
                break
        
        if key is None:
            continue
        
        res = results[key]
        train_losses = res.get('train_losses', [])
        if not train_losses:
            continue
        
        # Downsample
        indices, sampled_losses = downsample(train_losses, num_points)
        
        test_mse_display = res['test_mse']
        
        # Plot with distinct style - colored lines with markers
        ax.plot(
            indices, sampled_losses,
            color=COLORS[wtype],
            marker=MARKERS[wtype],
            label=f"{WEIGHT_NAMES[wtype]} (MSE={test_mse_display:.4f})",
            linewidth=2.5,
            markersize=8,
            markerfacecolor='white',
            markeredgewidth=2,
            alpha=0.9,
        )
    
    ax.set_xlabel('Epoch', fontsize=20, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Training Loss (MSE)', fontsize=20, fontweight='bold')
    ax.set_title(model_name, fontsize=26, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=14, framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    # Make tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


def print_summary(s3gnn_results, stable_results):
    """Print a summary of all results."""
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY: Weight Parameterization on Barbell Graph")
    print("="*70)
    
    all_results = []
    
    if s3gnn_results:
        print("\n📊 S³GNN Results:")
        print("-" * 50)
        for wtype in ['id', 'as', 'ortho_cay']:
            for k, v in s3gnn_results.items():
                if extract_weight_type(k) == wtype:
                    name = f"S³GNN ({WEIGHT_NAMES[wtype]})"
                    print(f"  {name:30} | Test MSE: {v['test_mse']:.6f}")
                    all_results.append((name, v['test_mse']))
    
    if stable_results:
        print("\n📊 Stable-ChebNet Results:")
        print("-" * 50)
        for wtype in ['id', 'as', 'ortho_cay']:
            for k, v in stable_results.items():
                if extract_weight_type(k) == wtype:
                    name = f"Stable-ChebNet ({WEIGHT_NAMES[wtype]})"
                    print(f"  {name:30} | Test MSE: {v['test_mse']:.6f}")
                    all_results.append((name, v['test_mse']))
    
    # Overall ranking
    print("\n🏆 Overall Ranking (Best to Worst):")
    print("-" * 50)
    for i, (name, mse) in enumerate(sorted(all_results, key=lambda x: x[1]), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {medal} {i}. {name}: {mse:.6f}")
    
    print("\n" + "="*70)


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    parser = argparse.ArgumentParser(description='Visualize S³GNN ablation study results')
    parser.add_argument('--s3gnn_file', type=str, 
                        default=str(script_dir / 'ablation_results_20260119_044459.json'),
                        help='JSON file with S³GNN results')
    parser.add_argument('--stable_file', type=str, 
                        default=str(script_dir / 'ablation_results_20260119_033902.json'),
                        help='JSON file with Stable-ChebNet results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: show plot)')
    parser.add_argument('--num_points', type=int, default=20,
                        help='Number of points to sample (default: 20)')
    args = parser.parse_args()
    
    # Load results
    s3gnn_results = None
    stable_results = None
    
    if Path(args.s3gnn_file).exists():
        s3gnn_results = load_results(args.s3gnn_file)
        print(f"✓ Loaded S³GNN results from {Path(args.s3gnn_file).name}")
    else:
        print(f"⚠ S³GNN file not found: {args.s3gnn_file}")
    
    if Path(args.stable_file).exists():
        stable_results = load_results(args.stable_file)
        print(f"✓ Loaded Stable-ChebNet results from {Path(args.stable_file).name}")
    else:
        print(f"⚠ Stable-ChebNet file not found: {args.stable_file}")
    
    if not s3gnn_results and not stable_results:
        print("❌ No results to visualize!")
        return
    
    # Print summary
    print_summary(s3gnn_results, stable_results)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot S³GNN (left) - with y-axis label
    if s3gnn_results:
        plot_model_results(axes[0], s3gnn_results, 'S³GNN', args.num_points, show_ylabel=True)
    else:
        axes[0].text(0.5, 0.5, 'No S³GNN data', ha='center', va='center', fontsize=14)
        axes[0].set_title('S³GNN')
    
    # Plot Stable-ChebNet (right) - no y-axis label
    if stable_results:
        plot_model_results(axes[1], stable_results, 'Stable-ChebNet', args.num_points, show_ylabel=False)
    else:
        axes[1].text(0.5, 0.5, 'No Stable-ChebNet data', ha='center', va='center', fontsize=14)
        axes[1].set_title('Stable-ChebNet')
    
    plt.tight_layout()
    
    # Always save to default location
    default_save_path = script_dir / 'ablation_loss_curves.png'
    save_path = args.output if args.output else default_save_path
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to {save_path}")
    
    # Also show the plot
    print("📈 Displaying plot...")
    plt.show()
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
