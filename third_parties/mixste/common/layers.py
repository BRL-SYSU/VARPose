import torch
import torch.nn as nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_ema=True, decay=0.99):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._use_ema = use_ema
        self._decay = decay
        self.eps = 1e-5

        # --- Codebook initialization ---
        # Use nn.Embedding; it is easy to understand and required for gradient-based updates.
        self.codebook = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings)

        # --- EMA-related state (as buffers) ---
        if self._use_ema:
            # register_buffer ensures these tensors move with the model (CPU/GPU)
            # and are not treated as model parameters.
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            # ema_dw accumulates encoder output vectors assigned to each codebook entry.
            self.register_buffer("ema_dw", self.codebook.weight.data.clone())

        # --- State required for codebook reset (as a buffer) ---
        self.register_buffer('usage_count', torch.zeros(num_embeddings, dtype=torch.long))

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # --- Distance computation and index lookup ---
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight.data ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebook.weight.data.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).view(input_shape)

        # --- Update logic during training ---
        if self.training:
            # 1. Always update usage counts to prepare for codebook reset.
            self.usage_count.index_add_(0, encoding_indices, torch.ones_like(encoding_indices, dtype=torch.long))

            # 2. If EMA is enabled, perform EMA updates.
            if self._use_ema:
                with torch.no_grad():
                    # Count how often each codebook entry is selected (one-hot form).
                    embed_onehot = F.one_hot(encoding_indices, self._num_embeddings).type(flat_input.dtype)
                    
                    # Update ema_cluster_size (smoothed usage counts).
                    self.ema_cluster_size.mul_(self._decay).add_(
                        embed_onehot.sum(0), alpha=1 - self._decay
                    )
                    
                    # Update ema_dw (smoothed vector sums).
                    dw = torch.matmul(flat_input.t(), embed_onehot)
                    self.ema_dw.mul_(self._decay).add_(dw.t(), alpha=1 - self._decay)
                    
                    # Update codebook weights from EMA state with numerical stabilization.
                    n = self.ema_cluster_size.sum()
                    cluster_size_stable = (
                        (self.ema_cluster_size + self.eps) / (n + self._num_embeddings * self.eps) * n
                    )
                    embed_normalized = self.ema_dw / cluster_size_stable.unsqueeze(1)
                    self.codebook.weight.data.copy_(embed_normalized)

        # --- Loss computation ---
        if self._use_ema:
            # With EMA, the VQ loss only includes the commitment loss.
            commitment_loss = F.mse_loss(inputs, quantized.detach())
            vq_loss = self._commitment_cost * commitment_loss
        else:
            # With gradient-based updates, the loss has two components.
            codebook_loss = F.mse_loss(quantized, inputs.detach())
            commitment_loss = F.mse_loss(inputs, quantized.detach())
            vq_loss = codebook_loss + self._commitment_cost * commitment_loss

        # --- Straight-through gradient estimator ---
        quantized = inputs + (quantized - inputs).detach()

        return quantized, vq_loss, encoding_indices.view(input_shape[0], input_shape[1])

    @torch.no_grad()
    def reset_dead_codebooks(self, inputs: torch.Tensor):
        flat_input = inputs.reshape(-1, self._embedding_dim)
        dead_indices = torch.where(self.usage_count == 0)[0]
        num_dead = len(dead_indices)
        
        if num_dead == 0:
            self.usage_count.zero_()
            return 0

        # Randomly sample vectors from the input to replace dead codebook entries.
        num_to_sample = min(num_dead, len(flat_input))
        if num_to_sample == 0:
            self.usage_count.zero_()
            return 0
            
        sample_indices = torch.randint(0, len(flat_input), (num_to_sample,))
        replacement_vectors = flat_input[sample_indices]
        
        # Replace the corresponding codebook weights.
        self.codebook.weight.data[dead_indices[:num_to_sample]] = replacement_vectors
        
        # If EMA is used, reset its state too to avoid conflicts between old and new state.
        if self._use_ema:
            self.ema_dw.data[dead_indices[:num_to_sample]] = replacement_vectors
            # Set ema_cluster_size to a small initial value to give new entries a warm start.
            self.ema_cluster_size.data[dead_indices[:num_to_sample]] = 1.0

        self.usage_count.zero_()
        return num_dead
    
    def getQuantized(self, encoding_indices:torch.Tensor)-> torch.Tensor:
        """
        Args:
            encoding_indices (Tensor): Encoding indices, shape: (B, N)
        Returns:
            Tensor: Quantized embeddings, shape: (B, N, D)
        """
        B,N = encoding_indices.shape
        # Reshape dimensions
        encoding_indices = encoding_indices.reshape(-1)
        quantized = self.codebook(encoding_indices).view(B,N,self._embedding_dim)
        return quantized


class MlpMixerBlock(nn.Module):
    """
    An MLP-Mixer block for mixing information along token and channel dimensions.
    """

    def __init__(self, num_tokens, embed_dim, hidden_dim_multiplier=2, dropout_rate=0.1):
        super().__init__()
        mlp_embed_hidden = embed_dim * hidden_dim_multiplier
        mlp_tokens_hidden = num_tokens * hidden_dim_multiplier
        self.norm_tokens = nn.LayerNorm(embed_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_tokens, mlp_tokens_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_tokens_hidden, num_tokens)
        )
        self.norm_channels = nn.LayerNorm(embed_dim)
        self.channel_mixing = nn.Sequential(
            nn.Linear(embed_dim, mlp_embed_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_embed_hidden, embed_dim)
        )

    def forward(self, x):
        # x shape: (B, N, D), where N is the number of tokens.
        # Token mixing
        y = self.norm_tokens(x)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y

        # Channel mixing
        y = self.norm_channels(x)
        x = x + self.channel_mixing(y)
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, seq_dim, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.seq_dim = seq_dim
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0, "Embedding dimension must be even for positional encoding."
        # Position embedding
        position = torch.arange(self.seq_dim, dtype=torch.float32).unsqueeze(1)  # (input_joints, 1)
        # Correct formula: 1 / (10000 ** (2 * j / embed_dim))
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32)/self.embed_dim * (-torch.log(torch.tensor(10000.0))))
        
        ret = torch.zeros(self.seq_dim, self.embed_dim)
        ret[:, 0::2] = torch.sin(position * div_term)  # Even columns
        ret[:, 1::2] = torch.cos(position * div_term)  # Odd columns
        
        # Register as a buffer so it automatically moves to the correct device.
        self.register_buffer('pos_embedding', ret)

    def forward(self, x):
        assert x.size(1) <= self.seq_dim, "Input sequence length is too long"
        x = x + self.pos_embedding[:x.size(1), :].to(x.device)
        return x

class ResiLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ResiLinear, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(in_features, out_features, bias),
                                        nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(out_features, out_features, bias),
                                        nn.ReLU())
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.linear1(x)
        x = self.norm(x)
        x = x + self.linear2(x)
        return x

class TokenExpansion(nn.Module):
    def __init__(self, tokens_per_joint=6):
        super().__init__()
        self.tokens_per_joint = tokens_per_joint
        if self.tokens_per_joint % 2 == 1:
            self.tokens_per_joint += 1

    def forward(self, x:torch.Tensor, adj_matrix:torch.Tensor|None):  # x: (B, J, D), adj_matrix: (J, J)
        B, J, D = x.shape

        if adj_matrix is None:
            adj_matrix = torch.ones(x.shape[1], x.shape[1], device=x.device)
            adj_matrix.fill_diagonal_(0)
        
        # Position-information tokens (3)
        pos_tokens = x.unsqueeze(2).expand(-1, -1, self.tokens_per_joint//2, -1)  # (B, J, 3, D)
        
        # Adjacency-information tokens (3)
        adj_matrix = adj_matrix.unsqueeze(0) # (1, J, J)
        adj_tokens = torch.matmul(adj_matrix, x)  # (B, J, D)
        adj_tokens = adj_tokens.unsqueeze(2).expand(-1, -1, self.tokens_per_joint//2, -1)
        
        # Merge all tokens
        tokens = torch.cat([pos_tokens, adj_tokens], dim=2)  # (B, J, 6, D)
        return tokens.reshape(B, J * self.tokens_per_joint, D)  # (B, 6J, D)


class TokenAggregation(nn.Module):
    def __init__(self, tokens_per_joint=6):
        super().__init__()
        self.tokens_per_joint = tokens_per_joint
        if self.tokens_per_joint % 2 == 1:
            self.tokens_per_joint += 1
        
    def forward(self, x:torch.Tensor):  # (B, 6J, D)
        B, nJ, D = x.shape
        x = x.reshape(B, -1, self.tokens_per_joint, D)  # (B, J, 6, D)
        
        # Process position-information and adjacency-information tokens separately.
        pos_tokens = x[:, :, :self.tokens_per_joint//2, :]  # (B, J, 3, D)
        adj_tokens = x[:, :, self.tokens_per_joint//2:, :]  # (B, J, 3, D)
        
        # Aggregate position information
        pos = pos_tokens.mean(dim=2)  # (B, J, D)
        # Aggregate adjacency information
        adj = adj_tokens.mean(dim=2)  # (B, J, D)
        
        # Merge information
        return pos + adj  # (B, J, D)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, offset: int = 0):
        # x shape: (batch, seq_len, embed_dim)
        seq_len = x.shape[1]

        if seq_len == self.seq_len_cached and offset == 0:
             return self.cos_cached, self.sin_cached
        self.seq_len_cached = seq_len if offset == 0 else None # Cache only in standard mode
        # Compute the correct position range from offset and seq_len.
        t = torch.arange(offset, offset + seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        if offset == 0:
            self.cos_cached = cos
            self.sin_cached = sin
            
        return cos, sin


def rotate_half(x):
    # Split the last dimension into two halves.
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shape: (batch, seq_len, num_heads, head_dim)
    # cos, sin shape: (seq_len, head_dim)
    # unsqueeze(1) -> (seq_len, 1, head_dim) to broadcast over num_heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed