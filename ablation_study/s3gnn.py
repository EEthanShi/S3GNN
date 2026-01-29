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



def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    G_dtype = G.dtype
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(dtype=G_dtype)


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



class Orthogonal(Module):
    """
    Orthogonal Parametrization via QR decomposition (numerically stable).

    A weight matrix W is parametrized as:
        W_ortho = Q from QR(W)
    """
    def __init__(self):
        super().__init__()

    def forward(self, W: Tensor) -> Tensor:
        Q, _ = torch.linalg.qr(W)
        return Q

    def right_inverse(self, W: Tensor) -> Tensor:
        return W


class OrthogonalSTE(Module):
    """
    Orthogonal Parametrization via QR with Straight-Through Estimator.
    
    Forward: uses orthogonal Q from QR decomposition
    Backward: gradient flows directly to W, bypassing QR gradient computation
    
    This treats orthogonalization as a "projection" rather than a learnable transform.
    Benefits:
    - More numerically stable (no QR gradient computation)
    - Simpler optimization landscape
    - Useful for ablation: compare with full Orthogonal to see if QR gradients help
    """
    def __init__(self):
        super().__init__()

    def forward(self, W: Tensor) -> Tensor:
        Q, _ = torch.linalg.qr(W)
        # Straight-through estimator: forward uses Q, backward uses W
        return W + (Q - W).detach()

    def right_inverse(self, W: Tensor) -> Tensor:
        return W


class Identity(Module):
    """
    Identity Parametrization

    A weight matrix W is parametrized as:
        W_antisym = W
    """
    def __init__(self):
        super().__init__()

    def forward(self, W: Tensor) -> Tensor:
        return W


class DoublyStochastic(Module):
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
    
    Args:
        num_iters: Number of Sinkhorn iterations (default: 10)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    def __init__(self, num_iters: int = 10, eps: float = 1e-8):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
    
    def forward(self, W: Tensor) -> Tensor:
        # Apply exp to ensure non-negativity
        P = torch.exp(W)
        
        # Sinkhorn-Knopp iteration
        for _ in range(self.num_iters):
            # Normalize rows
            P = P / (P.sum(dim=1, keepdim=True) + self.eps)
            # Normalize columns
            P = P / (P.sum(dim=0, keepdim=True) + self.eps)
        
        return P
    
    def right_inverse(self, W: Tensor) -> Tensor:
        # For initialization, return log of the matrix (clamped for safety)
        return torch.log(W.clamp(min=1e-8))


class OrthogonalCayley(Module):
    """
    Orthogonal Parametrization via Cayley Transform with Neumann Series Approximation.
    
    Constructs an orthogonal matrix Q from a skew-symmetric matrix A:
        Q = (I - A)(I + A)^{-1}
    
    Uses Neumann series to approximate the inverse for efficiency:
        (I + A)^{-1} ≈ I - A + A² - A³ + ...
    
    Properties:
        - Q is guaranteed to be orthogonal (Q^T Q = I)
        - More efficient than QR decomposition
        - Differentiable
    
    Args:
        size: Size of the square matrix
        use_cayley_neumann: Whether to use Neumann series approximation (default: True)
        num_cayley_neumann_terms: Number of terms in Neumann series (default: 5)
    
    Note: This is a linear layer, not a parametrization. It directly computes x @ Q^T.
    """
    def __init__(self, size: int, use_cayley_neumann: bool = True, num_cayley_neumann_terms: int = 5):
        super().__init__()
        self.size = size
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        
        # Parameters for the skew-symmetric matrix A
        # Only store the lower triangular part (size * (size - 1) / 2 parameters)
        self.a_params = Parameter(torch.zeros((size * (size - 1)) // 2))
        self.register_buffer('indices_lower', torch.tril_indices(size, size, -1))
    
    def forward(self, input: Tensor) -> Tensor:
        # Construct skew-symmetric matrix A from parameters
        A = torch.zeros((self.size, self.size), device=self.a_params.device)
        A[self.indices_lower[0], self.indices_lower[1]] = self.a_params
        A = A - A.t()  # Make it skew-symmetric: A = -A^T
        
        I = torch.eye(self.size, device=self.a_params.device)
        
        if self.use_cayley_neumann:
            # Neumann series approximation: (I + A)^{-1} ≈ I - A + A² - A³ + ...
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
            # Exact Cayley transform: Q = (I - A)(I + A)^{-1}
            Q = torch.linalg.solve(I + A, I - A, left=False)
        
        return torch.matmul(input, Q.t())
    
    def reset_parameters(self):
        # Initialize parameters to zero (identity orthogonal matrix)
        torch.nn.init.zeros_(self.a_params)


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
        weight_param: str = 'as',
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

        if weight_param == 'as':
            param_fn = AntiSymmetric(dissipative_force=self.g)
        elif weight_param == 'ortho':
            param_fn = Orthogonal()
        elif weight_param == 'ortho_ste':
            param_fn = OrthogonalSTE()
        elif weight_param == 'ortho_cay':
            param_fn = None  # Special handling below
        elif weight_param == 'id':
            param_fn = Identity()
        elif weight_param == 'ds':
            param_fn = DoublyStochastic(num_iters=20)
        else:
            raise NotImplementedError(f'Weight parameterization {weight_param} not implemented.')

        if weight_param == 'ortho_cay':
            # OrthogonalCayley requires square matrices (in_channels == out_channels)
            assert in_channels == out_channels, \
                f"OrthogonalCayley requires in_channels == out_channels, got {in_channels} != {out_channels}"
            
            # Use OrthogonalCayley directly as linear layers (not parametrization)
            self.lin_in_spatial = OrthogonalCayley(in_channels)
            self.lin_out_spatial = OrthogonalCayley(in_channels)
            self.lin_in_spectral = OrthogonalCayley(in_channels)
            self.lin_out_spectral = OrthogonalCayley(in_channels)
            
            if bias:
                self.bias = Parameter(torch.zeros(out_channels))
            else:
                self.register_parameter('bias', None)
        else:
            # ====================================================================
            # Spatial Path: Ŵ_in_spa, Ŵ_out_spa (both antisymmetric for stability)
            # ====================================================================
            self.lin_in_spatial = PyGLinear(in_channels, out_channels, bias=False, weight_initializer='glorot')
            register_parametrization(self.lin_in_spatial, 'weight', param_fn)

            self.lin_out_spatial = PyGLinear(out_channels, out_channels, bias=False, weight_initializer='glorot')
            register_parametrization(self.lin_out_spatial, 'weight', param_fn)

            # ====================================================================
            # Spectral Path: Ŵ_in_spec, Ŵ_out_spec (both antisymmetric)
            # ====================================================================
            self.lin_in_spectral = PyGLinear(in_channels, out_channels, bias=False, weight_initializer='glorot')
            register_parametrization(self.lin_in_spectral, 'weight', param_fn)

            self.lin_out_spectral = PyGLinear(out_channels, out_channels, bias=False, weight_initializer='glorot')
            register_parametrization(self.lin_out_spectral, 'weight', param_fn)

            if bias:
                self.bias = Parameter(torch.zeros(out_channels))
            else:
                self.register_parameter('bias', None)

            self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self.lin_in_spatial, 'reset_parameters'):
            self.lin_in_spatial.reset_parameters()
        if hasattr(self.lin_out_spatial, 'reset_parameters'):
            self.lin_out_spatial.reset_parameters()
        if hasattr(self.lin_in_spectral, 'reset_parameters'):
            self.lin_in_spectral.reset_parameters()
        if hasattr(self.lin_out_spectral, 'reset_parameters'):
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
        weight_param: str = 'id',  # as-antisymmetric, ortho-orthogonal, ortho_cay-cayley, id-identity, ds-doubly_stochastic
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
                weight_param=weight_param
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
