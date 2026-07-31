"""
Graph Convolutional Network (GCN) module for processing 3D human pose data

Supports adjacency matrix masking for handling cases where some nodes are unknown
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphConv(nn.Module):
    """
    Basic graph convolution layer

    Implements the formula: H' = σ(Â * H * W)
    where:
    - Â: Normalized adjacency matrix
    - H: Node features
    - W: Learnable weights
    - σ: Activation function
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool = True,
                 activation: str = 'relu'):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            bias: Whether to use bias
            activation: Activation function type ('relu', 'leaky_relu', 'gelu', 'none')
        """
        super(GraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self,
                x: torch.Tensor,
                adj: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features, shape (B, N, in_channels)
            adj: Adjacency matrix, shape (N, N) or (B, N, N)
            mask: Optional mask, shape (B, N,), marks which nodes are valid
                 1.0 means valid node, 0.0 means unknown/invalid node

        Returns:
            Updated node features, shape (B, N, out_channels)
        """
        B, N, _ = x.shape

        # Ensure adj has the correct dimensions
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)

        # Normalize the adjacency matrix: D^(-1/2) * A * D^(-1/2)
        # Compute the degree matrix
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        degree_inv_sqrt = degree.pow(-0.5)
        norm_adj = adj * degree_inv_sqrt * degree_inv_sqrt.transpose(-1, -2)

        # Add self-loops
        identity = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        norm_adj = norm_adj + identity

        # Graph convolution: Â * H * W
        # First apply linear transformation: H * W
        x = torch.matmul(x, self.weight)

        # Then apply the adjacency matrix for aggregation: Â * (H * W)
        x = torch.matmul(norm_adj, x)

        # Add bias
        if self.bias is not None:
            x = x + self.bias

        # Apply activation function
        x = self.activation(x)

        # Apply mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, N, 1)
            x = x * mask

        return x


class ResGraphConv(nn.Module):
    """
    Graph convolution block with residual connection

    Structure: GraphConv -> BatchNorm -> Activation -> Dropout -> Residual
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 use_residual: bool = True):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            kernel_size: Convolution kernel size (kept for future extension)
            dropout: Dropout probability
            activation: Activation function type
            use_residual: Whether to use residual connection
        """
        super(ResGraphConv, self).__init__()

        self.conv = GraphConv(in_channels, out_channels,
                             bias=True, activation=activation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

        # If input and output dimensions differ, projection is needed
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None

    def forward(self,
                x: torch.Tensor,
                adj: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features, shape (B, N, C)
            adj: Adjacency matrix, shape (N, N) or (B, N, N)
            mask: Optional mask, shape (B, N,)

        Returns:
            Updated node features, shape (B, N, C)
        """
        identity = x

        out = self.conv(x, adj, mask)

        # BatchNorm on the (N, C) dimensions
        B, N, C = out.shape
        out = out.transpose(1, 2)  # (B, C, N)
        out = self.bn(out)
        out = out.transpose(1, 2)  # (B, N, C)

        out = self.dropout(out)

        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity

        return out


class GraphConvEncoder(nn.Module):
    """
    Multi-layer graph convolution encoder

    Used to extract high-dimensional representations from input pose features
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output feature dimension
            num_layers: Number of GCN layers
            dropout: Dropout probability
            activation: Activation function type
        """
        super(GraphConvEncoder, self).__init__()

        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        # First layer: input -> hidden
        self.layers.append(
            ResGraphConv(in_channels, hidden_channels,
                        dropout=dropout, activation=activation)
        )

        # Middle layers: hidden -> hidden
        for _ in range(num_layers - 2):
            self.layers.append(
                ResGraphConv(hidden_channels, hidden_channels,
                            dropout=dropout, activation=activation)
            )

        # Last layer: hidden -> out
        if num_layers > 1:
            self.layers.append(
                ResGraphConv(hidden_channels, out_channels,
                            dropout=dropout, activation='none')  # No activation on the last layer
            )

    def forward(self,
                x: torch.Tensor,
                adj: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features, shape (B, N, C_in)
            adj: Adjacency matrix, shape (N, N) or (B, N, N)
            mask: Optional mask, shape (B, N,)

        Returns:
            Encoded node features, shape (B, N, C_out)
        """
        for layer in self.layers:
            x = layer(x, adj, mask)

        return x


if __name__ == "__main__":
    # Test code
    B, N, C_in, C_out = 4, 24, 3, 128

    # Create random inputs
    x = torch.randn(B, N, C_in)
    adj = torch.rand(N, N)
    adj = (adj + adj.t()) / 2  # Symmetrize
    adj = (adj > 0.5).float()  # Binarize

    # Create mask (some nodes unknown)
    mask = torch.ones(B, N)
    mask[:, :10] = 0.0  # Mark the first 10 nodes as unknown

    # Test GraphConv
    conv = GraphConv(C_in, C_out)
    out = conv(x, adj, mask)
    print(f"GraphConv output shape: {out.shape}")

    # Test ResGraphConv
    res_conv = ResGraphConv(C_out, C_out)
    out = res_conv(out, adj, mask)
    print(f"ResGraphConv output shape: {out.shape}")

    # Test GraphConvEncoder
    encoder = GraphConvEncoder(C_in, 64, C_out, num_layers=3)
    out = encoder(x, adj, mask)
    print(f"GraphConvEncoder output shape: {out.shape}")

    print("\nAll tests passed!")
