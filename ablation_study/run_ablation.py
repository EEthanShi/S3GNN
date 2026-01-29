"""
S³GNN Ablation Study on Barbell Graph

This script runs ablation experiments to compare different models and weight parameterizations:

Models (7 total):
- 'chebnet': ChebNet (baseline, no Euler residual, no weight constraint)

Stable-ChebNet variants (Chebyshev K-hop + Euler residual):
- 'stable_chebnet_id': Stable-ChebNet with Identity weights (no constraint)
- 'stable_chebnet_as': Stable-ChebNet with AntiSymmetric weights (original paper)
- 'stable_chebnet_ortho': Stable-ChebNet with Orthogonal weights

S³GNN variants (1-hop + P₀ global projection + Euler residual):
- 's3gnn_id': S³GNN with Identity weights (no constraint)
- 's3gnn_as': S³GNN with AntiSymmetric weights (S³GNN default)
- 's3gnn_ortho': S³GNN with Orthogonal weights

Uses the ORIGINAL Barbell graph from Stable-ChebNet:
- Two complete cliques of size N, connected by a single edge (node 0 ↔ node N)
- Task: predict the mean of the OPPOSITE clique for each node
- Total nodes: 2N

Usage:
    # Run single model
    python run_ablation.py --model s3gnn_as --N 50 --samples 100
    
    # Run all models comparison
    python run_ablation.py --run_all --N 50 --samples 100
    
    # Run only Stable-ChebNet variants
    python run_ablation.py --run_all --models "stable_chebnet_id,stable_chebnet_as,stable_chebnet_ortho"
    
    # Compare results
    python run_ablation.py --compare
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import ChebConv
from torch.nn import Linear
import sys
import os
import json
from datetime import datetime

from s3gnn import S3GNNModel

# ============================================================================
# Import Euler_ChebConv from local barbell_data (Stable-ChebNet code)
# ============================================================================

BARBELL_DATA_PATH = os.path.join(os.path.dirname(__file__), 'barbell_data')
sys.path.insert(0, BARBELL_DATA_PATH)

HAS_EULER_CHEB = False
try:
    from nn.layers import Euler_ChebConv
    HAS_EULER_CHEB = True
    print("✓ Euler_ChebConv loaded from barbell_data", flush=True)
except ImportError as e:
    print(f"⚠ Euler_ChebConv not found: {e}", flush=True)
    print("  Stable-ChebNet will be skipped", flush=True)


# ============================================================================
# Weight Parametrizations (for FlexibleEulerChebConv)
# ============================================================================

class AntiSymmetric(nn.Module):
    """AntiSymmetric weight parametrization: W_eff = W - W^T - g*I"""
    def __init__(self, dissipative_force=0.05):
        super().__init__()
        self.g = dissipative_force

    def forward(self, W):
        return W.triu(diagonal=1) - W.triu(diagonal=1).T - self.g * torch.eye(W.shape[0], device=W.device)

    def right_inverse(self, W):
        return W.triu(diagonal=1)


class Orthogonal(nn.Module):
    """Orthogonal weight parametrization via QR decomposition (numerically stable)."""
    def forward(self, W):
        # QR decomposition is more stable than Newton-Schulz iteration
        Q, _ = torch.linalg.qr(W)
        return Q

    def right_inverse(self, W):
        return W


class OrthogonalSTE(nn.Module):
    """Orthogonal weight parametrization via QR with Straight-Through Estimator.
    
    Forward: uses orthogonal Q from QR decomposition
    Backward: gradient flows directly to W, bypassing QR gradient computation
    
    This treats orthogonalization as a "projection" rather than a learnable transform.
    Benefits:
    - More numerically stable (no QR gradient computation)
    - Simpler optimization landscape
    - Useful for ablation: compare with full Orthogonal to see if QR gradients help
    """
    def forward(self, W):
        Q, _ = torch.linalg.qr(W)
        # Straight-through estimator: forward uses Q, backward uses W
        # (Q - W).detach() has zero gradient, so gradient flows to W directly
        return W + (Q - W).detach()

    def right_inverse(self, W):
        return W


class Identity(nn.Module):
    """Identity parametrization (no constraint)."""
    def forward(self, W):
        return W

    def right_inverse(self, W):
        return W


class DoublyStochastic(nn.Module):
    """
    Doubly Stochastic Parametrization via Sinkhorn-Knopp iteration.
    
    A weight matrix W is parametrized as:
        W_ds = Sinkhorn(exp(W))
    
    where Sinkhorn iteratively normalizes rows and columns to sum to 1.
    
    Properties:
        - All entries are non-negative
        - Each row sums to 1
        - Each column sums to 1
        - Eigenvalues satisfy |λ| ≤ 1 (spectral radius ≤ 1)
    """
    def __init__(self, num_iters: int = 10, eps: float = 1e-8):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
    
    def forward(self, W):
        # Apply exp to ensure non-negativity
        P = torch.exp(W)
        
        # Sinkhorn-Knopp iteration
        for _ in range(self.num_iters):
            # Normalize rows
            P = P / (P.sum(dim=1, keepdim=True) + self.eps)
            # Normalize columns
            P = P / (P.sum(dim=0, keepdim=True) + self.eps)
        
        return P
    
    def right_inverse(self, W):
        # For initialization, return log of the matrix (clamped for safety)
        return torch.log(W.clamp(min=1e-8))


class OrthogonalCayley(nn.Module):
    """
    Orthogonal Parametrization via Cayley Transform with Neumann Series Approximation.
    
    Constructs an orthogonal matrix Q from a skew-symmetric matrix A:
        Q = (I - A)(I + A)^{-1}
    
    Uses Neumann series to approximate the inverse for efficiency.
    
    Note: This is a linear layer, not a parametrization. It directly computes x @ Q^T.
    """
    def __init__(self, size, use_cayley_neumann=True, num_cayley_neumann_terms=5):
        super().__init__()
        self.size = size
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.a_params = nn.Parameter(torch.zeros((size * (size - 1)) // 2))
        self.register_buffer('indices_lower', torch.tril_indices(size, size, -1))
    
    def forward(self, input):
        A = torch.zeros((self.size, self.size), device=self.a_params.device)
        A[self.indices_lower[0], self.indices_lower[1]] = self.a_params
        A = A - A.t()
        
        I = torch.eye(self.size, device=self.a_params.device)
        
        if self.use_cayley_neumann:
            t = self.num_cayley_neumann_terms
            if t <= 0:
                Q = I - A
            else:
                negA = -A
                R = I.clone()
                P = negA
                for _ in range(t):
                    R.add_(P, alpha=2.0)
                    P = P @ negA
                R.add_(P)
                Q = R
        else:
            Q = torch.linalg.solve(I + A, I - A, left=False)
        
        return torch.matmul(input, Q.t())
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.a_params)


# ============================================================================
# Flexible Euler ChebConv (supports different weight parametrizations)
# ============================================================================

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.nn.inits import zeros
from torch.nn import Parameter
from torch.nn.utils.parametrize import register_parametrization


class FlexibleEulerChebConv(MessagePassing):
    """
    Flexible Euler ChebConv that supports different weight parametrizations.
    
    Same structure as Stable-ChebNet's Euler_ChebConv:
    - Chebyshev polynomial expansion (K terms)
    - Euler residual: out = x + ε * conv(x)
    
    But with configurable weight constraint:
    - 'id': Identity (no constraint)
    - 'as': AntiSymmetric (W - W^T - g*I)
    - 'ortho': Orthogonal (Newton-Schulz)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        step_size: float = 0.2,
        dissipation_force: float = 0.05,
        weight_param: str = 'as',  # 'id', 'as', 'ortho'
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert in_channels == out_channels, "FlexibleEulerChebConv requires in_channels == out_channels for weight constraints"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = 'sym'
        self.e = step_size
        self.g = dissipation_force
        self.weight_param = weight_param
        self.K = K

        self.lins = nn.ModuleList()
        for _ in range(K):
            if weight_param == 'ortho_cay':
                # OrthogonalCayley requires square matrices
                assert in_channels == out_channels, \
                    f"OrthogonalCayley requires in_channels == out_channels, got {in_channels} != {out_channels}"
                lin = OrthogonalCayley(in_channels)
            else:
                lin = PyGLinear(in_channels, out_channels, bias=False, weight_initializer='glorot')
                # Apply weight parametrization based on type
                if weight_param == 'as':
                    register_parametrization(lin, 'weight', AntiSymmetric(dissipative_force=self.g))
                elif weight_param == 'ortho':
                    register_parametrization(lin, 'weight', Orthogonal())
                elif weight_param == 'ortho_ste':
                    register_parametrization(lin, 'weight', OrthogonalSTE())
                elif weight_param == 'ds':
                    register_parametrization(lin, 'weight', DoublyStochastic(num_iters=10))
                # 'id' means no parametrization (identity)
            self.lins.append(lin)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def __norm__(self, edge_index, num_nodes, edge_weight, normalization, lambda_max, dtype, batch):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization, dtype, num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None, batch=None, lambda_max=None):
        edge_index, norm = self.__norm__(
            edge_index, x.size(self.node_dim), edge_weight,
            self.normalization, lambda_max, dtype=x.dtype, batch=batch,
        )

        Tx_0 = x
        Tx_1 = x  # Dummy
        out = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        # Euler residual
        out = x + self.e * out

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# ============================================================================
# ChebNet / Stable-ChebNet Model Wrapper
# ============================================================================

class ModelNodeFixed(nn.Module):
    """
    Wrapper for ChebNet and Stable-ChebNet variants.
    
    Supports:
    - 'Cheb': Standard ChebConv (no Euler residual, no weight constraint)
    - 'EulerCheb': Original Stable-ChebNet (Euler + AntiSymmetric)
    - 'FlexibleEulerCheb': Euler + configurable weight constraint (id/as/ortho)
    
    Note: BatchNorm is created dynamically in forward() to match original Stable-ChebNet behavior.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer, layer_type,
                 bias=True, k=9, step_size=0.2, weight_param='as', 
                 dissipation_force=0.05, **kwargs):
        super().__init__()
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.enc = Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.act = nn.ReLU()

        for _ in range(num_layer):
            if layer_type == 'Cheb':
                conv = ChebConv(hidden_dim, hidden_dim, K=k, bias=bias)
            elif layer_type == 'EulerCheb':
                if not HAS_EULER_CHEB:
                    raise ImportError("Euler_ChebConv not available")
                conv = Euler_ChebConv(hidden_dim, hidden_dim, K=k, step_size=step_size, bias=bias)
            elif layer_type == 'FlexibleEulerCheb':
                conv = FlexibleEulerChebConv(
                    hidden_dim, hidden_dim, K=k, 
                    step_size=step_size, 
                    dissipation_force=dissipation_force,
                    weight_param=weight_param, 
                    bias=bias
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            self.convs.append(conv)

        self.dec = Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None, **kwargs):
        device = x.device
        h = self.enc(x)
        for conv in self.convs:
            h = conv(h, edge_index=edge_index)
            h = self.act(h)
            # Dynamic BatchNorm (matches original Stable-ChebNet behavior)
            h = nn.BatchNorm1d(h.size(1)).to(device)(h)
        h = self.dec(h)
        return h


# ============================================================================
# Original Barbell Graph Generation (from Stable-ChebNet)
# ============================================================================

def gen_barbell(n):
    """
    Generate a barbell graph of size 2n.
    
    Structure:
        [Clique A: nodes 0..n-1] ---(single edge: 0↔n)--- [Clique B: nodes n..2n-1]
    
    Both cliques are COMPLETE graphs (all pairs connected).
    
    Args:
        n: Number of nodes per clique
    
    Returns:
        x: Node features [2n, 1]
        edge_index: Edge indices [2, E]
        y: Target labels [2n, 1] (mean of opposite clique)
    """
    edge_index1, edge_index2 = [], []
    
    # Complete graph for clique A (nodes 0 to n-1)
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_index1 += [[i, j]]
    
    # Complete graph for clique B (nodes n to 2n-1)
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_index2 += [[i + n, j + n]]
    
    # Bridge: single edge connecting node 0 (clique A) to node n (clique B)
    edge_index = edge_index1 + edge_index2 + [[0, n], [n, 0]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

    # Generate signal
    x, y = gen_barbell_signal(n)

    return x, edge_index, y


def gen_barbell_signal(n):
    """
    Generate barbell signal: predict mean of OPPOSITE clique.
    
    - Clique A (nodes 0..n-1): random features from U(μ1 - √3σ, μ1 + √3σ)
    - Clique B (nodes n..2n-1): random features from U(μ2 - √3σ, μ2 + √3σ)
    - Target for clique A: mean of clique B
    - Target for clique B: mean of clique A
    
    This is the ORIGINAL task from Stable-ChebNet.
    """
    mu1, mu2 = -np.sqrt(3 * n), np.sqrt(3 * n)
    std1, std2 = np.sqrt(n), np.sqrt(n)
    
    # Random features for each clique
    x1 = torch.empty([n, 1]).uniform_(mu1 - np.sqrt(3) * std1, mu1 + np.sqrt(3) * std1)
    x2 = torch.empty([n, 1]).uniform_(mu2 - np.sqrt(3) * std2, mu2 + np.sqrt(3) * std2)
    
    # Target: mean of opposite clique
    y1 = x2.mean(dim=0).repeat(n, 1)  # Clique A should predict mean of clique B
    y2 = x1.mean(dim=0).repeat(n, 1)  # Clique B should predict mean of clique A
    
    x = torch.cat((x1, x2), dim=0)
    y = torch.cat((y1, y2), dim=0)
    
    return x, y


def gen_many_barbells(num_samples, n):
    """Generate multiple barbell graphs with different random signals."""
    # Graph structure is fixed, only signals change.
    _, edge_index, _ = gen_barbell(n)
    data_ls = []
    for _ in range(num_samples):
        x, y = gen_barbell_signal(n)
        data_ls.append(Data(x=x, edge_index=edge_index, y=y))
    return data_ls


# ============================================================================
# Training and Evaluation
# ============================================================================

def evaluate_loader(model, loader, criterion, device):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = criterion(out, batch.y).item()
            total_loss += loss
            num_batches += 1
    return total_loss / max(num_batches, 1)


def train_model(model, train_loader, val_loader, test_loader, epochs=500, lr=0.01, weight_decay=0.0,
                batch_size=64, print_every=10):
    """
    Train the model on barbell regression task.
    
    Args:
        model: S3GNNModel or ModelNodeFixed
        train_loader: PyG DataLoader (batched)
        val_loader: PyG DataLoader (batched)
        test_loader: PyG DataLoader (batched) - for tracking best test
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: AdamW weight decay
        batch_size: DataLoader batch size (for logging only)
        print_every: Print every N epochs (always on; Notebook-friendly)
    
    Returns:
        dict with train_losses, val_losses, test_losses, best_val_epoch, best_test_mse
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    test_losses = []
    
    best_val_loss = float('inf')
    best_val_epoch = 0
    best_test_mse = float('inf')
    
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        # Training (batched)
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = evaluate_loader(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Test (for tracking)
        test_loss = evaluate_loader(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        
        # Track best validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_test_mse = test_loss
        
        if (epoch == 0) or ((epoch + 1) % max(int(print_every), 1) == 0):
            print(
                f"Epoch {epoch+1:4d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Test: {test_loss:.6f} | Best: {best_test_mse:.6f}",
                flush=True,
            )
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'best_val_epoch': best_val_epoch,
        'best_val_loss': best_val_loss,
        'best_test_mse': best_test_mse,
    }


# ============================================================================
# Model Creation
# ============================================================================

# Model configurations
MODEL_CONFIGS = {
    # ChebNet (baseline, no Euler residual, no weight constraint)
    'chebnet': {
        'name': 'ChebNet',
        'type': 'cheb',
        'color': 'gray',
    },
    # Stable-ChebNet variants (Euler residual + different weight constraints)
    'stable_chebnet_id': {
        'name': 'Stable-ChebNet (Identity)',
        'type': 'flexible_euler_cheb',
        'weight_param': 'id',
        'color': 'pink',
    },
    'stable_chebnet_as': {
        'name': 'Stable-ChebNet (AntiSym)',
        'type': 'flexible_euler_cheb',
        'weight_param': 'as',
        'color': 'orange',
    },
    'stable_chebnet_ortho': {
        'name': 'Stable-ChebNet (Ortho)',
        'type': 'flexible_euler_cheb',
        'weight_param': 'ortho',
        'color': 'brown',
    },
    'stable_chebnet_ortho_ste': {
        'name': 'Stable-ChebNet (Ortho-STE)',
        'type': 'flexible_euler_cheb',
        'weight_param': 'ortho_ste',
        'color': 'darkgoldenrod',
    },
    # S³GNN variants (1-hop + P₀ global + different weight constraints)
    's3gnn_id': {
        'name': 'S³GNN (Identity)',
        'type': 's3gnn',
        'weight_param': 'id',
        'color': 'red',
    },
    's3gnn_as': {
        'name': 'S³GNN (AntiSym)',
        'type': 's3gnn',
        'weight_param': 'as',
        'color': 'blue',
    },
    's3gnn_ortho': {
        'name': 'S³GNN (Ortho)',
        'type': 's3gnn',
        'weight_param': 'ortho',
        'color': 'green',
    },
    's3gnn_ortho_ste': {
        'name': 'S³GNN (Ortho-STE)',
        'type': 's3gnn',
        'weight_param': 'ortho_ste',
        'color': 'darkgreen',
    },
    # Doubly Stochastic variants
    'stable_chebnet_ds': {
        'name': 'Stable-ChebNet (DoublyStoch)',
        'type': 'flexible_euler_cheb',
        'weight_param': 'ds',
        'color': 'purple',
    },
    's3gnn_ds': {
        'name': 'S³GNN (DoublyStoch)',
        'type': 's3gnn',
        'weight_param': 'ds',
        'color': 'magenta',
    },
    # Orthogonal Cayley variants
    'stable_chebnet_ortho_cay': {
        'name': 'Stable-ChebNet (OrthoCay)',
        'type': 'flexible_euler_cheb',
        'weight_param': 'ortho_cay',
        'color': 'cyan',
    },
    's3gnn_ortho_cay': {
        'name': 'S³GNN (OrthoCay)',
        'type': 's3gnn',
        'weight_param': 'ortho_cay',
        'color': 'teal',
    },
}


def create_model(model_key, args, device):
    """Create a model based on the model key."""
    config = MODEL_CONFIGS[model_key]
    
    if config['type'] == 'cheb':
        # Standard ChebNet (no Euler residual)
        model = ModelNodeFixed(
            in_dim=1, out_dim=1,
            hidden_dim=args.hidden_dim,
            num_layer=args.num_layers,
            layer_type='Cheb',
            k=args.cheb_K,
        )
    elif config['type'] == 'flexible_euler_cheb':
        # Stable-ChebNet with flexible weight parametrization
        model = ModelNodeFixed(
            in_dim=1, out_dim=1,
            hidden_dim=args.hidden_dim,
            num_layer=args.num_layers,
            layer_type='FlexibleEulerCheb',
            k=args.cheb_K,
            step_size=args.step_size,
            weight_param=config['weight_param'],
            dissipation_force=args.dissipation_force,
        )
    elif config['type'] == 's3gnn':
        # S³GNN (1-hop + P₀ global projection)
        model = S3GNNModel(
            in_dim=1, out_dim=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            step_size=args.step_size,
            dissipation_force=args.dissipation_force,
            alpha_init=args.alpha_init,
            use_fiedler=False,
            weight_param=config['weight_param'],
        )
    else:
        raise ValueError(f"Unknown model type: {config['type']}")
    
    return model.to(device)


# ============================================================================
# Ablation Experiments
# ============================================================================

def run_single_experiment(args, device, model_key):
    """Run a single experiment for a specific model."""
    config = MODEL_CONFIGS[model_key]
    model_name = config['name']
    
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate data (original barbell)
    n = args.N  # nodes per clique
    samples = args.samples  # samples per split
    
    train_data = gen_many_barbells(num_samples=samples, n=n)
    val_data = gen_many_barbells(num_samples=samples, n=n)
    test_data = gen_many_barbells(num_samples=samples, n=n)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    num_nodes = 2 * n
    num_edges = train_data[0].edge_index.size(1)
    
    print(f"Graph: N={n} per clique, {num_nodes} total nodes, {num_edges} edges", flush=True)
    print(f"Data: {samples} train, {samples} val, {samples} test samples", flush=True)
    print(f"Task: Predict mean of opposite clique (regression)", flush=True)
    
    # Create model
    model = create_model(model_key, args, device)
    if model is None:
        print(f"⚠ Skipping {model_name}: dependencies not available", flush=True)
        return None
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.num_layers} layers, {args.hidden_dim} hidden dim, {num_params} params", flush=True)
    if config['type'] in ['cheb', 'flexible_euler_cheb']:
        print(f"Chebyshev K={args.cheb_K}", flush=True)
    print(f"", flush=True)
    
    # Train (now also tracks test loss and returns best validation results)
    train_result = train_model(
        model, train_loader, val_loader, test_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, print_every=args.print_every
    )
    
    # Get learned alpha values (only for S³GNN)
    alphas = None
    if hasattr(model, 'get_alpha_values'):
        alphas = model.get_alpha_values()
        alpha0_per_layer = [a[0] for a in alphas]
        print(f"  Learned α₀ (per-layer): {[f'{a:.3f}' for a in alpha0_per_layer]}", flush=True)
        print(f"  Learned α₀ (layer avg): {np.mean(alpha0_per_layer):.4f}", flush=True)
    
    print(f"\n{'='*40}", flush=True)
    print(f"Results for {model_name}:", flush=True)
    print(f"  Best Val Epoch:   {train_result['best_val_epoch'] + 1}", flush=True)
    print(f"  Best Val Loss:    {train_result['best_val_loss']:.6f}", flush=True)
    print(f"  Best Test MSE:    {train_result['best_test_mse']:.6f}", flush=True)
    print(f"  (Final Train:     {train_result['train_losses'][-1]:.6f})", flush=True)
    print(f"{'='*40}", flush=True)
    
    results = {
        'model_key': model_key,
        'model_name': model_name,
        'train_losses': train_result['train_losses'],
        'val_losses': train_result['val_losses'],
        'test_losses': train_result['test_losses'],
        'test_mse': train_result['best_test_mse'],  # Use best validation's test MSE
        'best_val_epoch': train_result['best_val_epoch'],
        'best_val_loss': train_result['best_val_loss'],
        'alphas': alphas,
        'N': n,
        'samples': samples,
        'num_layers': args.num_layers,
        'hidden_dim': args.hidden_dim,
    }
    
    return results


def run_all_experiments(args, device):
    """Run all model experiments."""
    all_results = {}
    
    # Determine which models to run
    if args.models == 'all':
        model_keys = list(MODEL_CONFIGS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(',')]
    
    for model_key in model_keys:
        if model_key not in MODEL_CONFIGS:
            print(f"⚠ Unknown model: {model_key}, skipping", flush=True)
            continue
        
        results = run_single_experiment(args, device, model_key)
        if results is not None:
            all_results[model_key] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"ablation_results_{timestamp}.json"
    
    # Convert to serializable format
    save_data = {}
    for key, res in all_results.items():
        save_data[key] = {
            'model_name': res['model_name'],
            'test_mse': res['test_mse'],  # This is now best_test_mse
            'best_val_epoch': res.get('best_val_epoch', -1),
            'best_val_loss': res.get('best_val_loss', res['val_losses'][-1]),
            'final_train_loss': res['train_losses'][-1],
            'final_val_loss': res['val_losses'][-1],
            'alphas': res['alphas'],
            'N': res['N'],
            'samples': res['samples'],
            'num_layers': res['num_layers'],
            'hidden_dim': res['hidden_dim'],
            'train_losses': res['train_losses'],
            'val_losses': res['val_losses'],
            'test_losses': res.get('test_losses', []),
        }
    
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {save_path}", flush=True)
    
    return all_results


def plot_comparison(all_results, save_path='ablation_comparison.png'):
    """Plot comparison of different models."""
    plt.rcParams.update({'font.size': 12})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get colors and labels
    colors = {}
    labels = {}
    for key, res in all_results.items():
        if key in MODEL_CONFIGS:
            colors[key] = MODEL_CONFIGS[key]['color']
            labels[key] = MODEL_CONFIGS[key]['name']
        else:
            colors[key] = 'black'
            labels[key] = res.get('model_name', key)
    
    # Plot 1: Training Loss
    ax = axes[0]
    for key, res in all_results.items():
        ax.plot(res['train_losses'], color=colors[key], label=labels[key], alpha=0.8, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[1]
    for key, res in all_results.items():
        ax.plot(res['val_losses'], color=colors[key], label=labels[key], alpha=0.8, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val Loss (MSE)', fontsize=12)
    ax.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Test MSE Bar Chart
    ax = axes[2]
    keys = list(all_results.keys())
    test_mses = [all_results[k]['test_mse'] for k in keys]
    bar_colors = [colors[k] for k in keys]
    bar_labels = [labels[k] for k in keys]
    
    bars = ax.bar(range(len(keys)), test_mses, color=bar_colors, alpha=0.8)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Test MSE (lower is better)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(bar_labels, fontsize=9, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mse in zip(bars, test_mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(test_mses),
                f'{mse:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}", flush=True)
    plt.show()
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<25} {'Test MSE':<15} {'Train Loss':<15} {'Val Loss':<15}")
    print("-"*70)
    for key in keys:
        res = all_results[key]
        print(f"{labels[key]:<25} {res['test_mse']:<15.6f} {res['train_losses'][-1]:<15.6f} {res['val_losses'][-1]:<15.6f}")
    print("="*70)
    
    # Find best
    best_key = min(keys, key=lambda k: all_results[k]['test_mse'])
    print(f"\n🏆 Best Model: {labels[best_key]} with Test MSE = {all_results[best_key]['test_mse']:.6f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='S³GNN Ablation Study on Barbell Graph')
    
    # Experiment mode
    parser.add_argument('--run_all', action='store_true', help='Run all model experiments')
    parser.add_argument('--compare', action='store_true', help='Load and plot saved results')
    
    # Model selection
    parser.add_argument('--model', type=str, default='s3gnn_as',
                        help='Single model to run (chebnet, stable_chebnet, s3gnn_id, s3gnn_as, s3gnn_ortho)')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated list of models to run, or "all"')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--step_size', type=float, default=0.2, help='Euler integration step size')
    parser.add_argument('--dissipation_force', type=float, default=0.05, help='Dissipation force g (S³GNN)')
    parser.add_argument('--alpha_init', type=float, default=1.0, help='Initial alpha value (S³GNN)')
    parser.add_argument('--cheb_K', type=int, default=9, help='Chebyshev polynomial order K')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (Adam)')
    
    # Graph parameters (ORIGINAL Barbell)
    parser.add_argument('--N', type=int, default=50, help='Nodes per clique (total nodes = 2N)')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples per split (train/val/test)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--print_every', type=int, default=10, help='Print every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    
    # Print available models
    print("\nAvailable models:", flush=True)
    for key, config in MODEL_CONFIGS.items():
        available = "✓" if (key != 'stable_chebnet' or HAS_EULER_CHEB) else "✗ (missing Euler_ChebConv)"
        print(f"  {key}: {config['name']} {available}", flush=True)
    
    if args.run_all:
        all_results = run_all_experiments(args, device)
        if all_results:
            plot_comparison(all_results)
    elif args.compare:
        # Load latest results and plot
        import glob
        result_files = glob.glob('ablation_results_*.json')
        if result_files:
            latest = sorted(result_files)[-1]
            print(f"\nLoading {latest}", flush=True)
            with open(latest, 'r') as f:
                data = json.load(f)
            
            # Print summary
            print("\n" + "="*60)
            print("Ablation Results Summary")
            print("="*60)
            for key, res in data.items():
                print(f"\n{res.get('model_name', key)}:")
                print(f"  Test MSE: {res['test_mse']:.6f}")
                print(f"  Final Train Loss: {res['final_train_loss']:.6f}")
                print(f"  Final Val Loss: {res['final_val_loss']:.6f}")
            
            # Reconstruct for plotting
            all_results = {}
            for key, res in data.items():
                all_results[key] = {
                    'model_name': res.get('model_name', key),
                    'train_losses': res['train_losses'],
                    'val_losses': res['val_losses'],
                    'test_mse': res['test_mse'],
                }
            plot_comparison(all_results)
        else:
            print("No result files found. Run --run_all first.")
    else:
        # Run single model
        run_single_experiment(args, device, args.model)


if __name__ == '__main__':
    main()

