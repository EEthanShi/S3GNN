"""
S³GNN Layer Implementation

Dynamic:
    H(ℓ+1) = H(ℓ) + ε * (Spatial + α₀ · P₀ · H_spec + α₁ · P₁ · H_spec)

where:
    - Ã = D^{-1/2} A D^{-1/2} is the symmetric normalized adjacency
    - Ŵ_in_spa, Ŵ_out_spa are antisymmetric (W - W^T - g*I) for stability (spatial path)
    - Ŵ_in_spec, Ŵ_out_spec are antisymmetric (W - W^T - g*I) for stability (spectral path)
    - P₀ = (1/N) · 1 · 1ᵀ (projection onto λ=0 eigenspace, global mean)
    - P₁ = u₁ u₁ᵀ (projection onto λ₁ eigenspace, Fiedler vector)
    - α = [α₀, α₁] are learnable scalars
    - ε is the step size (Euler integration)
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter, Module
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch.nn.utils.parametrize import register_parametrization
from torch_geometric.utils import add_self_loops, degree


# ============================================================================
# Pure PyTorch implementations of scatter operations (no torch_scatter needed)
# ============================================================================

def scatter_mean_pytorch(src: Tensor, index: Tensor, dim: int = 0, dim_size: int = None) -> Tensor:
    """
    Compute mean of src elements grouped by index.
    Pure PyTorch implementation, no torch_scatter needed.
    
    Args:
        src: Source tensor [N, d]
        index: Index tensor [N] - which group each element belongs to
        dim: Dimension to scatter along (default: 0)
        dim_size: Number of groups (optional, inferred from index if not provided)
    
    Returns:
        Tensor [num_groups, d] with mean values per group
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    
    # Sum per group
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
    
    # Count per group
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    ones = torch.ones(index.size(0), device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index, ones)
    
    # Avoid division by zero
    count = count.clamp(min=1).unsqueeze(1)
    
    return out / count


def scatter_add_pytorch(src: Tensor, index: Tensor, dim: int = 0, dim_size: int = None) -> Tensor:
    """
    Compute sum of src elements grouped by index.
    Pure PyTorch implementation, no torch_scatter needed.
    
    Args:
        src: Source tensor [N, d]
        index: Index tensor [N] - which group each element belongs to
        dim: Dimension to scatter along (default: 0)
        dim_size: Number of groups (optional, inferred from index if not provided)
    
    Returns:
        Tensor [num_groups, d] with sum values per group
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
    
    return out


class AntiSymmetric(Module):
    """
    Anti-Symmetric Parametrization
    
    A weight matrix W is parametrized as:
        W_antisym = W_upper - W_upper^T - g * I
    
    This ensures eigenvalues are purely imaginary (+ small negative real part from g),
    guaranteeing numerical stability.
    """
    def __init__(self, dissipative_force: float = 0.0):
        super().__init__()
        self.g = dissipative_force

    def forward(self, W: Tensor) -> Tensor:
        return W.triu(diagonal=1) - W.triu(diagonal=1).T - self.g * torch.eye(W.shape[0], device=W.device)

    def right_inverse(self, W: Tensor) -> Tensor:
        return W.triu(diagonal=1)


class S3GNNConv(MessagePassing):
    """
    S³GNN Convolution Layer (MLP-in, Dynamic, MLP-out architecture)
    
    Implements the dynamic:
        H(ℓ+1) = H(ℓ) + ε * (Spatial + α₀ · P₀ · H_spec [+ α₁ · P₁ · H_spec])
    
    where:
        - Ã = D^{-1/2} A D^{-1/2} is the symmetric normalized adjacency
        - P₀ = (1/N) · 1 · 1ᵀ (global mean, λ=0)
        - P₁ = u₁ u₁ᵀ (Fiedler projection, λ₁) - optional
        - Ŵ_in_spa, Ŵ_out_spa are antisymmetric for stability (spatial path)
        - Ŵ_in_spec, Ŵ_out_spec are antisymmetric for stability (spectral path)
        - α = [α₀] or [α₀, α₁] are learnable scalars
        - ε is the step size for Euler integration
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        step_size: Euler integration step size (ε)
        dissipation_force: Dissipation term for antisymmetric weights (g)
        bias: Whether to use bias
        alpha_init: Initial value for α
        add_self_loops_flag: Whether to add self-loops for normalization
        use_fiedler: Whether to use P₁ (Fiedler vector) in addition to P₀
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        step_size: float = 0.2,
        dissipation_force: float = 0.05,
        bias: bool = True,
        alpha_init: float = 1.0,
        add_self_loops_flag: bool = True,
        use_fiedler: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.step_size = step_size
        self.g = dissipation_force
        self.add_self_loops_flag = add_self_loops_flag
        self.use_fiedler = use_fiedler

        # Learnable α for spectral filtering
        # α₀: weight for P₀ (global mean, λ=0)
        # α₁: weight for P₁ (Fiedler, λ₁) - only if use_fiedler=True
        if use_fiedler:
            self.alpha = Parameter(torch.tensor([alpha_init, alpha_init], dtype=torch.float))
        else:
            self.alpha = Parameter(torch.tensor([alpha_init], dtype=torch.float))

        # ====================================================================
        # Spatial Path: Ŵ_in_spa, Ŵ_out_spa (both antisymmetric for stability)
        # ====================================================================
        self.lin_in_spatial = PyGLinear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        register_parametrization(self.lin_in_spatial, 'weight', AntiSymmetric(dissipative_force=self.g))
        
        self.lin_out_spatial = PyGLinear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        register_parametrization(self.lin_out_spatial, 'weight', AntiSymmetric(dissipative_force=self.g))

        # ====================================================================
        # Spectral Path: Ŵ_in_spec, Ŵ_out_spec (both antisymmetric)
        # ====================================================================
        self.lin_in_spectral = PyGLinear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        register_parametrization(self.lin_in_spectral, 'weight', AntiSymmetric(dissipative_force=self.g))
        
        self.lin_out_spectral = PyGLinear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        register_parametrization(self.lin_out_spectral, 'weight', AntiSymmetric(dissipative_force=self.g))

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_in_spatial.reset_parameters()
        self.lin_out_spatial.reset_parameters()
        self.lin_in_spectral.reset_parameters()
        self.lin_out_spectral.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def _compute_norm(self, edge_index: Tensor, num_nodes: int) -> tuple:
        """
        Compute symmetric normalization: D^{-1/2} A D^{-1/2}
        """
        if self.add_self_loops_flag:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return edge_index, norm

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: OptTensor = None,
        fiedler_vec: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """
        Forward pass implementing:
            H(ℓ+1) = H(ℓ) + ε * (Spatial + α₀ · P₀ · H_spec + α₁ · P₁ · H_spec)
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            batch: Batch assignment [N] - which graph each node belongs to
            fiedler_vec: Fiedler vector u_1 [N] (eigenvector of λ₁, per-graph)
            edge_weight: Optional edge weights [E] (ignored)
        
        Returns:
            Updated node features [N, out_channels]
        """
        N = x.size(0)

        # Handle batch (single graph case)
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=x.device)

        # Compute symmetric normalization
        edge_index_norm, norm = self._compute_norm(edge_index, N)

        # ================================================================
        # Spatial Path: Ŵ_out_spa · Ã · Ŵ_in_spa · H (both antisymmetric)
        # ================================================================
        # Step 1: Ŵ_in_spa · H
        h_in_spa = self.lin_in_spatial(x)  # [N, out_channels]
        # Step 2: Ã · (Ŵ_in_spa · H)
        h_agg_spa = self.propagate(edge_index_norm, x=h_in_spa, norm=norm)  # [N, out_channels]
        # Step 3: Ŵ_out_spa · (Ã · Ŵ_in_spa · H)
        spatial_out = self.lin_out_spatial(h_agg_spa)  # [N, out_channels]

        # ================================================================
        # Spectral Path: Ŵ_out_spec · (α₀·P₀ + α₁·P₁) · Ŵ_in_spec · H
        # P₀ = (1/N_g) · 1_g · 1_gᵀ (per-graph global mean)
        # P₁ = u₁ u₁ᵀ (per-graph Fiedler projection)
        # ================================================================
        # Step 1: Ŵ_in_spec · H
        h_in_spec = self.lin_in_spectral(x)  # [N, out_channels]
        
        # Step 2a: P₀ · H_spec = per-graph mean (expanded to all nodes in each graph)
        # scatter_mean computes mean per graph, then we expand back to nodes
        graph_mean = scatter_mean_pytorch(h_in_spec, batch, dim=0)  # [num_graphs, d]
        h_agg_p0 = self.alpha[0] * graph_mean[batch]  # [N, d] - broadcast to nodes
        
        # Step 2b: P₁ · H_spec = u₁ (u₁ᵀ H_spec) (per-graph Fiedler projection)
        # Only compute if use_fiedler=True and fiedler_vec is provided
        if self.use_fiedler and fiedler_vec is not None:
            u1 = fiedler_vec.view(-1, 1)  # [N, 1]
            # Compute u₁ᵀ @ H per graph: sum(u1 * h) per graph
            u1_h = u1 * h_in_spec  # [N, d]
            coeff_per_graph = scatter_add_pytorch(u1_h, batch, dim=0)  # [num_graphs, d]
            # Expand back to nodes and multiply by u1
            h_agg_p1 = self.alpha[1] * u1 * coeff_per_graph[batch]  # [N, d]
            # Combined spectral aggregation: α₀·P₀·H + α₁·P₁·H
            h_agg_spec = h_agg_p0 + h_agg_p1
        else:
            # Only use P₀ (default)
            h_agg_spec = h_agg_p0  # [N, d]
        
        # Step 3: Ŵ_out_spec · (α₀·P₀ + α₁·P₁) · Ŵ_in_spec · H
        spectral_out = self.lin_out_spectral(h_agg_spec)  # [N, out_channels]

        # ================================================================
        # Combined: spatial + spectral
        # ================================================================
        out = spatial_out + spectral_out

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        # Euler integration: H(ℓ+1) = H(ℓ) + ε * out
        return x + self.step_size * out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        """Message function for normalized spatial aggregation (Ã @ H)"""
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, step_size={self.step_size})')


class S3GNNModel(nn.Module):
    """
    S³GNN Model for node-level tasks
    
    Architecture:
        Input -> Encoder -> [S3GNNConv + Act] x num_layers -> Decoder -> Output
        
    Note: 
        - No BatchNorm, No LayerNorm
        - Spatial path: Ŵ_in_spa, Ŵ_out_spa (antisymmetric for stability)
        - Spectral path: Ŵ_in_spec, Ŵ_out_spec (antisymmetric for stability)
        - All weights are antisymmetric to ensure Jacobian stability
        - Default: uses P₀ (λ=0, global mean) only
        - Optional: use_fiedler=True adds P₁ (λ₁, Fiedler vector)
        - α = [α₀] or [α₀, α₁] are learnable scalars
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        step_size: float = 0.2,
        dissipation_force: float = 0.05,
        alpha_init: float = 1.0,
        dropout: float = 0.1,  # kept for API compatibility, but not used
        act: str = 'relu',
        bias: bool = True,
        use_fiedler: bool = False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_fiedler = use_fiedler

        # Activation
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()

        # Encoder: input -> hidden
        self.encoder = nn.Linear(in_dim, hidden_dim)

        # S³GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(S3GNNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                step_size=step_size,
                dissipation_force=dissipation_force,
                alpha_init=alpha_init,
                bias=bias,
                use_fiedler=use_fiedler,
            ))

        # Decoder: hidden -> output
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor = None, 
                fiedler_vec: Tensor = None, **kwargs) -> Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [N, in_dim]
            edge_index: Edge indices [2, E]
            batch: Batch assignment [N] - which graph each node belongs to
            fiedler_vec: Fiedler vector u_1 [N] (eigenvector of λ₁, per-graph concatenated)
        
        Returns:
            Node predictions [N, out_dim]
        """
        # Encode
        h = self.encoder(x)

        # S³GNN layers (no BatchNorm, no LayerNorm)
        for conv in self.convs:
            h = conv(h, edge_index, batch=batch, fiedler_vec=fiedler_vec)
            h = self.act(h)

        # Decode
        out = self.decoder(h)
        return out

    def get_alpha_values(self):
        """Return the learned α values for each layer.
        Returns [α₀] if use_fiedler=False, [α₀, α₁] if use_fiedler=True.
        """
        return [conv.alpha.detach().cpu().tolist() for conv in self.convs]
