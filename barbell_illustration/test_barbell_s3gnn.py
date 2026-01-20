"""
Test Script: S³GNN on Barbell Task

S³GNN Dynamic:
    H(ℓ+1) = H(ℓ) + ε * (Spatial + α * P₁) * H(ℓ)
    
    where P₁ = u₁ u₁ᵀ is the Fiedler projection (λ₁ eigenspace)

Paper Reference Results (Table 2, N=50, K=9):
    - ChebNet:        0.32 ± 0.39
    - Stable-ChebNet: 0.17 ± 0.11
"""
import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch.nn import Linear, ReLU
from functools import partialmethod
from tqdm import tqdm
import scipy.sparse.linalg as spla

# Disable tqdm progress bars
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Add barbell_data to path (contains data, nn, utils from Stable-ChebNet)
BARBELL_DATA_PATH = os.path.join(os.path.dirname(__file__), 'barbell_data')
sys.path.insert(0, BARBELL_DATA_PATH)

from data.data_loading import load_dataset
from utils.train import Evaluator

# Import layers from original code
from nn.layers import Euler_ChebConv
from torch_geometric.nn.conv import ChebConv

# Import S3GNN (in same directory)
from s3gnn_layer import S3GNNModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Fiedler Vector Computation
# ============================================================================

def compute_fiedler_vector(data):
    """
    Compute the Fiedler vector (eigenvector of λ₁) for a graph.
    
    Args:
        data: PyG Data object with edge_index and num_nodes
        
    Returns:
        fiedler_vec: Tensor [N] - the Fiedler vector (L2 normalized)
    """
    N = data.num_nodes
    
    # Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    edge_index, edge_weight = get_laplacian(
        data.edge_index, 
        normalization='sym', 
        num_nodes=N
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, N)
    
    # Compute the 2 smallest eigenvalues/vectors (λ₀ ≈ 0, λ₁ = Fiedler)
    # Using 'SM' = smallest magnitude
    try:
        eigenvalues, eigenvectors = spla.eigsh(L.tocsc(), k=2, which='SM', tol=1e-6)
        # eigenvectors[:, 1] is the Fiedler vector (corresponding to λ₁)
        fiedler_vec = torch.from_numpy(eigenvectors[:, 1]).float()
    except:
        # Fallback: use random vector if eigendecomposition fails
        fiedler_vec = torch.randn(N)
        fiedler_vec = fiedler_vec / fiedler_vec.norm()
    
    return fiedler_vec


def add_fiedler_to_dataset(dataset):
    """Add Fiedler vector to each graph in the dataset."""
    for data in dataset:
        data.fiedler_vec = compute_fiedler_vector(data)
    return dataset


# ============================================================================
# Argument Parser - All hyperparameters in one place
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description='S³GNN on Barbell Task')
    
    # Data
    parser.add_argument('--num_nodes', type=int, default=50,
                        help='Number of nodes per clique in barbell graph (default: 50)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of graph samples (default: 100)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='s3gnn',
                        choices=['s3gnn', 'chebnet', 'stable_chebnet', 'all'],
                        help='Model to run: s3gnn, chebnet, stable_chebnet, or all (default: all)')
    
    # Model architecture
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers (default: 4, same as paper)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--K', type=int, default=9,
                        help='Chebyshev polynomial order K (default: 9, same as paper)')
    parser.add_argument('--step_size', type=float, default=0.2,
                        help='Euler integration step size ε (default: 0.2, same as paper)')
    parser.add_argument('--alpha_init', type=float, default=1.0,
                        help='Initial value for α (spectral filter on λ=0) (default: 1.0)')
    parser.add_argument('--dissipation', type=float, default=0.05,
                        help='Dissipation force g for antisymmetric weights (default: 0.05)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (NOT USED - kept for API compatibility) (default: 0.0)')
    parser.add_argument('--use_fiedler', action='store_true',
                        help='Use P₁ (Fiedler vector, λ₁) in addition to P₀ (default: False, P₀ only)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of training epochs (default: 600, same as paper)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='LR scheduler patience (default: 10)')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='LR scheduler decay rate (default: 0.5)')
    
    # Experiment
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of random seeds (default: 5)')
    parser.add_argument('--start_seed', type=int, default=289469,
                        help='Starting seed (default: 289469, same as Stable-ChebNet paper)')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Print every N epochs (default: 10)')
    
    return parser.parse_args()


# ============================================================================
# Fixed ModelNode (same as original but with correct parameter passing)
# ============================================================================

class ModelNodeFixed(nn.Module):
    """
    Fixed version of ModelNode that correctly passes parameters to different layer types.
    This is identical to the original ModelNode except for the parameter handling.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer, layer_type,
                 normalize='sym', bias=True, act='relu', k=1, step_size=0.2,
                 **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layer_type = layer_type
        self.num_layers = num_layer

        ACT = {'relu': ReLU}
        self.act = ACT[act]()

        self.enc = Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()

        for _ in range(num_layer):
            if layer_type == 'Cheb':
                # ChebConv only accepts: in_channels, out_channels, K, bias
                conv = ChebConv(hidden_dim, hidden_dim, K=k, bias=bias)
            elif layer_type == 'EulerCheb':
                # Euler_ChebConv accepts: in_channels, out_channels, K, step_size, bias
                conv = Euler_ChebConv(hidden_dim, hidden_dim, K=k, step_size=step_size, bias=bias)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            self.convs.append(conv)

        self.dec = Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None, **kwargs):
        h = self.enc(x)
        for conv in self.convs:
            h = conv(h, edge_index=edge_index)
            h = self.act(h)
            # Same as original: create BatchNorm on the fly (this is how the original code works)
            h = nn.BatchNorm1d(h.size(1)).to(device)(h)
        h = self.dec(h)
        return h


# ============================================================================
# Configuration (for data loading compatibility)
# ============================================================================

def create_config(args):
    """Create configuration object from args (for data loading compatibility)"""
    class Config:
        class ExpConfig:
            seed = args.start_seed
            wandb = False
            train_eval_period = args.print_every
            
        class DataConfig:
            dataset = 'barbell'
            samples = args.samples
            task_type = 'mse_regression'
            eval_metric = 'mse'
            minimize = True
            num_nodes = args.num_nodes
            
        class ModelConfig:
            input_dim = 1
            hidden_dim = args.hidden_dim
            out_dim = 1
            num_layers = args.num_layers
            k = 9  # For baseline models
            step = args.step_size
            
        class OptimConfig:
            lr = args.lr
            epochs = args.epochs
            batch_size = args.batch_size
            lr_scheduler_decay_rate = args.lr_decay
            lr_scheduler_patience = args.lr_patience
            weight_decay = args.weight_decay
            
        exp = ExpConfig()
        data = DataConfig()
        model = ModelConfig()
        optim = OptimConfig()
        
    return Config()


# ============================================================================
# Custom Training and Evaluation (with Fiedler vector support)
# ============================================================================

def train_with_fiedler(model, device, loader, optimizer):
    """Custom training loop that passes Fiedler vector to the model."""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get Fiedler vector for this batch
        fiedler_vec = batch.fiedler_vec if hasattr(batch, 'fiedler_vec') else None
        
        # Forward pass with Fiedler vector
        out = model(batch.x, batch.edge_index, batch=batch.batch, fiedler_vec=fiedler_vec)
        
        # MSE loss
        loss = F.mse_loss(out.view(-1), batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)


def eval_with_fiedler(model, device, loader, evaluator):
    """Custom evaluation loop that passes Fiedler vector to the model."""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Get Fiedler vector for this batch
            fiedler_vec = batch.fiedler_vec if hasattr(batch, 'fiedler_vec') else None
            
            # Forward pass with Fiedler vector
            out = model(batch.x, batch.edge_index, batch=batch.batch, fiedler_vec=fiedler_vec)
            
            y_true.append(batch.y.view(-1))
            y_pred.append(out.view(-1))
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    
    # Compute MSE
    mse = ((y_true - y_pred) ** 2).mean()
    return mse, {'y_true': y_true, 'y_pred': y_pred}


# ============================================================================
# Standard Training and Evaluation (for baseline models)
# ============================================================================

def train_standard(model, device, loader, optimizer):
    """Standard training loop for baseline models (ChebNet, Stable-ChebNet)."""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.mse_loss(out.view(-1), batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def eval_standard(model, device, loader):
    """Standard evaluation loop for baseline models."""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            y_true.append(batch.y.view(-1))
            y_pred.append(out.view(-1))
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    mse = ((y_true - y_pred) ** 2).mean()
    return mse


# ============================================================================
# Run Experiment for a Specific Model
# ============================================================================

def run_single_model(args, model_type='s3gnn'):
    """
    Run experiment for a single model type.
    
    Args:
        args: Command line arguments
        model_type: One of 's3gnn', 'chebnet', 'stable_chebnet'
    
    Returns:
        Dictionary with results
    """
    model_names = {
        's3gnn': 'S³GNN',
        'chebnet': 'ChebNet',
        'stable_chebnet': 'Stable-ChebNet (EulerCheb)'
    }
    
    print(f"\n{'='*70}")
    print(f"Running {model_names[model_type]}")
    print(f"{'='*70}")
    print(f"  Graph: N={args.num_nodes} (barbell)")
    print(f"  Model: {args.num_layers} layers, hidden_dim={args.hidden_dim}")
    if model_type in ['chebnet', 'stable_chebnet']:
        print(f"  Chebyshev K={args.K}, step_size={args.step_size}")
    elif model_type == 's3gnn':
        print(f"  step_size={args.step_size}, alpha_init={args.alpha_init}")
        if args.use_fiedler:
            print(f"  Spectral: Using P₀ (λ=0) + P₁ (Fiedler) with α = [α₀, α₁]")
        else:
            print(f"  Spectral: Using P₀ (λ=0) only with α = [α₀]")
    print(f"  Training: {args.epochs} epochs, lr={args.lr}")
    print(f"  Seeds: {args.num_seeds} (starting from {args.start_seed})")
    print(f"{'='*70}\n")
    
    print(f"Using device: {device}")
    
    config = create_config(args)
    results = []
    
    for seed_idx in range(args.num_seeds):
        seed = args.start_seed + seed_idx
        print(f"\n--- Seed {seed} ---")
        
        # Set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        config.exp.seed = seed
        
        # Load data (datasets folder in barbell_data)
        root = os.path.join(BARBELL_DATA_PATH, 'datasets')
        train_data, val_data, test_data = load_dataset(
            config.data.dataset, 
            root=root, 
            config=config
        )
        
        # Precompute Fiedler vectors if needed
        if model_type == 's3gnn' and args.use_fiedler:
            print("Precomputing Fiedler vectors...")
            train_data = add_fiedler_to_dataset(train_data)
            val_data = add_fiedler_to_dataset(val_data)
            test_data = add_fiedler_to_dataset(test_data)
            print("Done.")
        
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        if model_type == 's3gnn':
            model = S3GNNModel(
                in_dim=1,
                out_dim=1,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                step_size=args.step_size,
                dissipation_force=args.dissipation,
                alpha_init=args.alpha_init,
                dropout=args.dropout,
                use_fiedler=args.use_fiedler,
            ).to(device)
        elif model_type == 'chebnet':
            model = ModelNodeFixed(
                in_dim=1,
                out_dim=1,
                hidden_dim=args.hidden_dim,
                num_layer=args.num_layers,
                layer_type='Cheb',
                k=args.K,
            ).to(device)
        elif model_type == 'stable_chebnet':
            model = ModelNodeFixed(
                in_dim=1,
                out_dim=1,
                hidden_dim=args.hidden_dim,
                num_layer=args.num_layers,
                layer_type='EulerCheb',
                k=args.K,
                step_size=args.step_size,
            ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params}")
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience
        )
        
        best_val_perf = float('inf')
        best_test_perf = float('inf')
        valid_curve = []
        test_curve = []
        train_curve = []
        
        start_time = time.time()
        
        for epoch in range(1, args.epochs + 1):
            # Train
            if model_type == 's3gnn':
                train_loss = train_with_fiedler(model, device, train_loader, optimizer)
            else:
                train_loss = train_standard(model, device, train_loader, optimizer)
            
            # Evaluate
            if model_type == 's3gnn':
                train_perf, _ = eval_with_fiedler(model, device, train_loader, None)
                valid_perf, _ = eval_with_fiedler(model, device, val_loader, None)
                test_perf, _ = eval_with_fiedler(model, device, test_loader, None)
            else:
                train_perf = eval_standard(model, device, train_loader)
                valid_perf = eval_standard(model, device, val_loader)
                test_perf = eval_standard(model, device, test_loader)
            
            train_curve.append(train_perf)
            valid_curve.append(valid_perf)
            test_curve.append(test_perf)
            
            if valid_perf < best_val_perf:
                best_val_perf = valid_perf
                best_test_perf = test_perf
            
            scheduler.step(valid_perf)
            
            # Print progress
            if epoch % args.print_every == 0 or epoch == 1:
                elapsed = time.time() - start_time
                extra_info = ""
                if model_type == 's3gnn':
                    alphas = model.get_alpha_values()
                    if args.use_fiedler:
                        extra_info = " | α: " + ", ".join([f"L{i}:[{a[0]:.2f},{a[1]:.2f}]" for i, a in enumerate(alphas)])
                    else:
                        extra_info = " | α₀: " + ", ".join([f"L{i}:{a[0]:.2f}" for i, a in enumerate(alphas)])
                
                print(f"Epoch {epoch:3d} | Train: {train_perf:.4f} | "
                      f"Val: {valid_perf:.4f} | Test: {test_perf:.4f} | "
                      f"Best: {best_test_perf:.4f}{extra_info} | Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_epoch_test = test_curve[best_val_epoch]
        
        results.append({
            'seed': seed,
            'best_val': valid_curve[best_val_epoch],
            'best_test': best_epoch_test,
            'best_epoch': best_val_epoch,
            'training_time': total_time
        })
        
        print(f"\nSeed {seed} Results: Best Test = {best_epoch_test:.6f}")
    
    # Aggregate
    best_test_losses = [r['best_test'] for r in results]
    mean_test = np.mean(best_test_losses)
    std_test = np.std(best_test_losses, ddof=1) if len(best_test_losses) > 1 else 0.0
    mean_time = np.mean([r['training_time'] for r in results])
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {model_names[model_type]}")
    print(f"{'='*70}")
    print(f"  Test Loss: {mean_test:.4f} ± {std_test:.4f}")
    print(f"  Min: {np.min(best_test_losses):.6f}, Max: {np.max(best_test_losses):.6f}")
    print(f"  Avg Training Time: {mean_time:.2f}s")
    print(f"{'='*70}\n")
    
    return {
        'model': model_type,
        'mean': mean_test,
        'std': std_test,
        'min': np.min(best_test_losses),
        'max': np.max(best_test_losses),
        'mean_time': mean_time,
        'all_results': results
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = get_args()
    
    print("\n" + "="*70)
    print("GNN Comparison on Barbell Task")
    print("="*70)
    print(f"\nSettings:")
    print(f"  N = {args.num_nodes}, K = {args.K}, layers = {args.num_layers}")
    print(f"  epochs = {args.epochs}, seeds = {args.num_seeds}")
    print(f"  Model(s) to run: {args.model}")
    
    # Reference results from paper
    print("\n" + "-"*70)
    print("REFERENCE RESULTS (Stable-ChebNet paper, Table 2, N=50, K=9)")
    print("-"*70)
    print("  ChebNet:        0.32 ± 0.39")
    print("  Stable-ChebNet: 0.17 ± 0.11")
    print("-"*70)
    
    all_results = {}
    
    # Run selected models
    if args.model == 'all':
        models_to_run = ['chebnet', 'stable_chebnet', 's3gnn']
    else:
        models_to_run = [args.model]
    
    for model_type in models_to_run:
        results = run_single_model(args, model_type)
        all_results[model_type] = results
    
    # Final comparison
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("FINAL COMPARISON (Same Seeds, Same Data)")
        print("="*70)
        print(f"{'Model':<20} {'Test MSE':<20} {'Time':<10}")
        print("-"*50)
        for model_type, res in all_results.items():
            name = {'s3gnn': 'S³GNN', 'chebnet': 'ChebNet', 'stable_chebnet': 'Stable-ChebNet'}[model_type]
            print(f"{name:<20} {res['mean']:.4f} ± {res['std']:.4f}      {res['mean_time']:.1f}s")
        print("="*70)
        
        # Find best
        best_model = min(all_results.keys(), key=lambda k: all_results[k]['mean'])
        best_name = {'s3gnn': 'S³GNN', 'chebnet': 'ChebNet', 'stable_chebnet': 'Stable-ChebNet'}[best_model]
        print(f"\n🏆 Best Model: {best_name} with MSE = {all_results[best_model]['mean']:.4f}")
    
    return all_results


if __name__ == "__main__":
    main()

