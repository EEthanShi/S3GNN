from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.functional import jacobian
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils.parametrize import register_parametrization


class AntiSymmetric(Module):
    r"""
    Anti-Symmetric Parametrization

    A weight matrix :math:`\mathbf{W}` is parametrized as
    :math:`\mathbf{W} = \mathbf{W} - \mathbf{W}^T`
    """
    def __init__(self):
        super().__init__()

    def forward(self, W: Tensor) -> Tensor:
        return W.triu(diagonal=1) - W.triu(diagonal=1).T 

    def right_inverse(self, W: Tensor) -> Tensor:
        return W.triu(diagonal=1)


class DiagLinear(Module):
    def __init__(self, in_channels, g = None):
        super().__init__()
        if g is None:
            # We learn W
            self.W = torch.nn.Parameter(torch.ones(in_channels,dtype=torch.float))
        else:
            # W is fixed
            self.W = torch.nn.Parameter((1-g) * torch.ones(in_channels,dtype=torch.float), requires_grad=False)
    
    def reset_parameters(self):
        pass

    def forward(self, x):
        return x * self.W



class NonDissip_ChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        K: int,
        dissipation_term: float = 0.0, # it pushes the eigenvalues toward 0
        eigenval_scaler: float = 1.0, # it scales the eigenvalues
        learn_additional_terms: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.normalization = 'sym'
        self.K = K
        self.lins = torch.nn.ModuleList()
        self.lins.append(DiagLinear(in_channels, g=None if learn_additional_terms else dissipation_term)) # this add dissipation to the model
        for i in range(1, K):
            self.lins.append(
                Linear(in_channels, in_channels, bias=False,weight_initializer='glorot')
            )
            register_parametrization(self.lins[i], 'weight', AntiSymmetric())
        
        if learn_additional_terms:
            self.eigenval_scaler = torch.nn.ModuleList([
                torch.nn.Sequential(
                    Linear(in_channels, in_channels), 
                    torch.nn.ELU(), 
                    Linear(in_channels,1),
                    torch.nn.Sigmoid()
                ) for _ in range(K-1)
            ])
        else:
            self.eigenval_scaler = [eigenval_scaler for _ in range(K-1)]
        self.learn_additional_terms = learn_additional_terms

        if bias:
            self.bias = Parameter(Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins[1:]:
            lin.reset_parameters()
        zeros(self.bias)


    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None, 
                batch: OptTensor = None, lambda_max: OptTensor = None, eig: bool = False) -> Tensor:

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        self.edge_index, self.norm = edge_index, norm
        self.n1, self.n2 = x.shape[0], x.shape[1]

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            eigenval_scaler = self.eigenval_scaler[0](Tx_1) if self.learn_additional_terms else self.eigenval_scaler[0] # eigenval scaler has k-1 elements
            out = out + eigenval_scaler * self.lins[1](Tx_1)
            
        for lin, eigenval_scaler in zip(self.lins[2:], self.eigenval_scaler[1:]):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            eigenval_scaler = eigenval_scaler(Tx_2) if self.learn_additional_terms else eigenval_scaler
            out = out + eigenval_scaler * lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        if eig:
          eigs_r = []
          eigs_im = []
          J = jacobian(self.conv_jac, (x.view(-1)), create_graph=True)

          print('Computing Eigenvalues')

          eigs = torch.linalg.eigvals(J).detach().cpu().numpy().tolist()
          for element in eigs:
                  eigs_r.append(element.real)
                  eigs_im.append(element.imag)

          # make two subplots: the first is the distribution of the eigenvalues and the second is the plot of the eigenvalues in the complex plane
          fig, axs = plt.subplots(1,2)
          axs[0].hist(eigs_r, bins=100, label='Real', alpha=0.5)
          axs[0].hist(eigs_im, bins=100, label='Imaginary', alpha=0.5)
          axs[0].set_xlabel('Eigenvalue')
          axs[0].set_ylabel('Frequency')
          axs[0].set_title(f'Distribution of Eigenvalues')
          axs[0].legend()
          lin = np.linspace(0, 2*np.pi, 1000)
          axs[1].plot(np.cos(lin), np.sin(lin), linewidth=1, color='k')
          axs[1].scatter(eigs_r, eigs_im, marker='x', label='Eigs, After Training', linewidth=2, rasterized=True)
          axs[1].set_xlabel('Real')
          axs[1].set_ylabel('Imaginary')
          axs[1].set_title(f'Eigenvalues')
          plt.show()

        return out

    def conv_jac(self, x: Tensor) -> Tensor:
        x = x.reshape(self.n1, self.n2)
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(self.edge_index, x=x, norm=self.norm)
            eigenval_scaler = self.eigenval_scaler(Tx_1) if self.learn_additional_terms else self.eigenval_scaler
            out = out + eigenval_scaler * self.lins[1](Tx_1)
            
        for lin in self.lins[2:]:
            Tx_2 = self.propagate(self.edge_index, x=Tx_1, norm=self.norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            eigenval_scaler = self.eigenval_scaler(Tx_2) if self.learn_additional_terms else self.eigenval_scaler
            out = out + eigenval_scaler * lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias
        return out.view(-1)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class NonDissipChebNet(Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 1,
                 K: int = 1,
                 dissipation_term: float = 0.0, # it pushes the eigenvalues toward 0
                 eigenval_scaler: float = 1.0, # it scales the eigenvalues
                 learn_additional_terms: bool = False,
                 activ_fun: str = 'tanh',
                 node_level_task: bool = False) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activ_fun = getattr(torch, activ_fun)

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        self.conv =torch.nn. ModuleList()
        for _ in range(num_layers):
            self.conv.append(NonDissip_ChebConv(
                in_channels = inp,
                K = K,
                dissipation_term = dissipation_term,
                eigenval_scaler = eigenval_scaler,
                learn_additional_terms = learn_additional_terms)
            )

        self.node_level_task = node_level_task 
        
        # Original code from Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023
        # https://github.com/gravins/Anti-SymmetricDGN/blob/main/graph_prop_pred/models/dgn_GraphProp.py
        if self.node_level_task:
            self.readout = torch.nn.Sequential(
                Linear(inp, inp // 2),
                torch.nn.LeakyReLU(),
                Linear(inp // 2, self.output_dim),
                torch.nn.LeakyReLU()
            )
        else:
            self.readout = torch.nn.Sequential(
                Linear(inp * 3, (inp * 3) // 2),
                torch.nn.LeakyReLU(),
                Linear((inp * 3) // 2, self.output_dim),
                torch.nn.LeakyReLU()
            )


    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.emb(x) if self.emb else x

        for conv in self.conv:
            x = self.activ_fun(conv(x, edge_index, batch=batch))

        if not self.node_level_task:
            # Original code from Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023
            # https://github.com/gravins/Anti-SymmetricDGN/blob/main/graph_prop_pred/models/dgn_GraphProp.py
            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.readout(x)

        return x