"""
H36M 17-joint -> SMPL 24-joint 3D pose conversion model

Uses a GCN + Transformer architecture, where the Transformer uses the adjacency matrix as the attention mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.gcn_layers import GraphConvEncoder
from models.layers import PositionEmbedding
from data.common_variables import smpl_24_symmetry_augmented


def _build_adj_matrix(num_joints: int, adj_tuples: tuple) -> torch.Tensor:
    """Build a symmetric adjacency matrix from adjacency tuples"""
    adj = torch.zeros(num_joints, num_joints)
    for i, j in adj_tuples:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


# H36M 17 -> SMPL 24 semantic correspondence (16 H36M joints can be mapped to SMPL)
H36M_TO_SMPL = [
    (0,  0),   # pelvis
    (4,  1),   # LHip
    (5,  4),   # LKnee
    (6,  7),   # LAnkle
    (1,  2),   # RHip
    (2,  5),   # RKnee
    (3,  8),   # RAnkle
    (7,  3),   # Spine -> Spine1
    (8,  12),  # Thorax -> Neck
    (9,  15),  # Neck -> Head
    (11, 16),  # LShoulder
    (12, 18),  # LElbow
    (13, 20),  # LWrist
    (14, 17),  # RShoulder
    (15, 19),  # RElbow
    (16, 21),  # RWrist
]

SMPL_INDICES_FROM_H36M = [s for _, s in H36M_TO_SMPL]


class GraphAwareTransformerLayer(nn.Module):
    """Transformer layer that supports adjacency matrix masking"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            adj_mask: (N, N) bool, True = masked out (not allowed to attend)
        """
        attn_output, _ = self.self_attn(x, x, x, attn_mask=adj_mask, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


class H36MToSMPLConverter(nn.Module):
    """
    3D pose conversion model from H36M 17-joint to SMPL 24-joint

    Pipeline:
    1. Input H36M 17 joints -> initially expanded to 24 joints (14 known points mapped, 10 unknown points zero-filled)
    2. GCN encoding (using the SMPL 24 adjacency matrix, without masking unknown points)
    3. Transformer encoding (using the adjacency matrix as the attention mask)
    4. Output SMPL 24 joints

    Losses:
    - mse_loss: all 24 joints vs SMPL GT
    - common_loss: 14 mapped points in the output vs values directly mapped from input H36M
    """

    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_gcn_layers: int = 3,
                 num_transformer_layers: int = 4,
                 gcn_hidden_dim: int = 128,
                 dropout: float = 0.1,
                 common_loss_weight=0.0):
        super().__init__()

        self.num_output_joints = 24
        self.common_loss_weight = common_loss_weight

        # Build adjacency matrix from common_variables
        smpl_24_adj = _build_adj_matrix(24, smpl_24_symmetry_augmented)
        self.register_buffer('smpl_adj', smpl_24_adj)

        # Transformer attention mask: True = forbid attend
        # Positions where the adjacency matrix is 0 = not connected = forbidden
        self.register_buffer('adj_mask', smpl_24_adj == 0)

        self.input_projection = nn.Linear(3, embed_dim)
        self.pos_embedding = PositionEmbedding(24, embed_dim)

        self.gcn_encoder = GraphConvEncoder(
            in_channels=embed_dim,
            hidden_channels=gcn_hidden_dim,
            out_channels=embed_dim,
            num_layers=num_gcn_layers,
            dropout=dropout,
            activation='gelu'
        )

        self.transformer_layers = nn.ModuleList([
            GraphAwareTransformerLayer(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 3)
        )

        self._init_weights()

    def _init_weights(self):
        def _init_module(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_init_module)

    def initial_expansion(self, h36m_joints: torch.Tensor) -> torch.Tensor:
        """Expand H36M 17 joints to SMPL 24 joints, unknown points zero-filled"""
        B = h36m_joints.shape[0]
        device = h36m_joints.device
        smpl_joints = torch.zeros(B, 24, 3, device=device)
        for h36m_idx, smpl_idx in H36M_TO_SMPL:
            smpl_joints[:, smpl_idx, :] = h36m_joints[:, h36m_idx, :]
        return smpl_joints

    def forward(self,
                h36m_joints: torch.Tensor,
                smpl_joints_gt: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            h36m_joints: (B, 17, 3) root-aligned, in meters
            smpl_joints_gt: (B, 24, 3) provided during training
        """
        # 17 -> 24 initial expansion
        smpl_init = self.initial_expansion(h36m_joints)

        x = self.input_projection(smpl_init)
        x = self.pos_embedding(x)
        x = self.gcn_encoder(x, self.smpl_adj)

        for layer in self.transformer_layers:
            x = layer(x, self.adj_mask)

        x = self.final_norm(x)
        prediction = self.output_projection(x)

        output = {'prediction': prediction}

        if smpl_joints_gt is not None:
            output.update(self.compute_loss(prediction, smpl_joints_gt, h36m_joints))

        return output

    def compute_loss(self,
                     prediction: torch.Tensor,
                     target: torch.Tensor,
                     h36m_joints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Loss = 24-joint MSE + α * 14 mapped-joint MSE

        - mse_loss: predicted 24 joints vs SMPL GT 24 joints
        - common_loss: 14 mapped joints in the prediction vs values from input H36M directly mapped to SMPL
        """
        mse_loss = F.mse_loss(prediction, target)

        smpl_from_h36m = self.initial_expansion(h36m_joints)
        common_pred = prediction[:, SMPL_INDICES_FROM_H36M, :]
        common_target = smpl_from_h36m[:, SMPL_INDICES_FROM_H36M, :]
        common_loss = F.mse_loss(common_pred, common_target)

        total_loss = mse_loss + self.common_loss_weight * common_loss

        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'common_loss': common_loss,
        }

    @torch.no_grad()
    def inference(self, h36m_joints: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(h36m_joints)['prediction']


# ============================================================
# API: H36M 17 joints -> SMPL (theta, beta)
# ============================================================

_api_model: Optional[H36MToSMPLConverter] = None
_api_device: Optional[str] = None


def init_h36m_to_smpl_api(checkpoint_path: str,
                           device: str = 'cuda',
                           embed_dim: int = 256,
                           num_heads: int = 8,
                           num_gcn_layers: int = 3,
                           num_transformer_layers: int = 4,
                           gcn_hidden_dim: int = 128,
                           dropout: float = 0.1) -> None:
    """
    Initialize the H36M 17-joint -> SMPL parameter conversion API.

    Must be called once before h36m_to_smpl_params().

    Args:
        checkpoint_path: Path to trained H36MToSMPLConverter checkpoint (.pth)
        device: 'cuda' or 'cpu'
        embed_dim ... dropout: Model hyperparameters (must match checkpoint)
    """
    global _api_model, _api_device
    _api_device = device

    model = H36MToSMPLConverter(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_gcn_layers=num_gcn_layers,
        num_transformer_layers=num_transformer_layers,
        gcn_hidden_dim=gcn_hidden_dim,
        dropout=dropout,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    _api_model = model
    print(f"[H36M->SMPL API] Loaded: {checkpoint_path}")


@torch.no_grad()
def h36m_to_smpl_params(joints_17: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert H36M 17 joints to SMPL parameters (theta, beta) in camera space.

    Pipeline:
        1. H36MToSMPLConverter: 17 joints -> 24 joints
        2. SMPL-IKS (SI + PR-C + MixIK): 24 joints -> SMPL (theta, beta)

    Args:
        joints_17: (B, 17, 3) or (T, 17, 3)
                   - 3D skeleton in camera coordinates
                   - Units: meters
                   - Root-aligned (pelvis at origin) recommended
                   - H36M 17-joint order

    Returns:
        theta: (B, 72) SMPL pose parameters (24 joints x 3 axis-angle)
        beta:  (B, 10) SMPL shape parameters (PCA coefficients)

    Example:
        >>> from models.h36m_to_smpl import init_h36m_to_smpl_api, h36m_to_smpl_params
        >>>
        >>> init_h36m_to_smpl_api('checkpoints/h36m2smpl/best.pth', device='cuda')
        >>>
        >>> joints = torch.randn(4, 17, 3).cuda()  # B=4, meters
        >>> theta, beta = h36m_to_smpl_params(joints)
        >>> print(theta.shape)  # (4, 72)
        >>> print(beta.shape)   # (4, 10)

    Note:
        - Call init_h36m_to_smpl_api() before first use
        - Uses NEUTRAL SMPL model internally
        - First call to smpliks initializes its model (~1 second)
    """
    global _api_model, _api_device
    assert _api_model is not None, "Call init_h36m_to_smpl_api() first!"

    joints_17 = joints_17.to(_api_device)

    # Step 1: 17 -> 24 joints
    smpl_24 = _api_model.inference(joints_17)  # (B, 24, 3)

    # Step 2: 24 joints -> SMPL params via SMPL-IKS
    from third_parties.api.smpliks import smpliks
    theta, beta = smpliks(smpl_24)

    return theta, beta
