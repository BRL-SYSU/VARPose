"""
SMPL-IKS API: 3D Skeleton to SMPL Parameters

Interface:
    smpliks(pos_24: torch.Tensor) -> (theta: torch.Tensor, beta: torch.Tensor)
    
    Args:
        pos_24: 3D skeleton joints in camera coordinates [B, 24, 3], units=meters
                SMPL 24-joint topology, centered at root (joint 0)

    Returns:
        theta: SMPL pose parameters [B, 72] (24 joints x 3 axis-angle)
        beta:  SMPL shape parameters [B, 10]

Method: SI (Shape Inverse) + PR-C (Pose Refinement) + MixIK (Mixed Inverse Kinematics)
"""

import torch
import numpy as np
from typing import Tuple
import sys
from pathlib import Path

# Add smpliks to sys.path for its internal relative imports (from lib.xxx)
_smpliks_path = Path(__file__).parent.parent / 'smpliks'
if str(_smpliks_path) not in sys.path:
    sys.path.insert(0, str(_smpliks_path))

# Import SMPL-IKS components
from lib.models.smpl.smpl import SMPLLayer
from lib.models.smpl.model import SMPL_MixIK
from lib.si.SI import SMPL_SI_LR
from lib.pr.smpl.AP import SMPL_AP_V1
from lib.ik.smpl.AnalyIK import SMPL_AnalyIK_V3
from lib.ik.smpl.MixIK import get_body_part_func


class _SMPLIKSModel:
    """Singleton wrapper for SMPL-IKS model"""

    def __init__(self, device='cuda'):
        self.device = device
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of SMPL-IKS components"""
        if self._initialized:
            return

        # SMPL topology
        self.parent = torch.tensor([
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
        ]).to(self.device)

        self.children = torch.tensor([
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21, 22, 23, -1, -1
        ]).to(self.device)

        # Load SMPL layer
        SMPL_MODEL_PATH = 'third_parties/smpliks/IKS/data/smpl/smpliks_data/SMPL_NEUTRAL.pkl'
        SMPL_KID_MODEL_PATH = 'third_parties/smpliks/IKS/data/smpl/smpliks_data/smpl_kid_template.npy'

        self.smpl = SMPLLayer(
            model_path=SMPL_MODEL_PATH,
            kid_template_path=SMPL_KID_MODEL_PATH,
            dtype=torch.float32,
            age='adult',
        ).to(self.device)

        # Load Shape Inverse regression matrix
        SMPL_SI_DATA_PATH = 'third_parties/smpliks/IKS/data/smpl/smpliks_data/skeleton_2_beta_smpl.npz'
        data = np.load(SMPL_SI_DATA_PATH, allow_pickle=True)['lr'].item()
        A1_data = data['A1'][:, :10]
        if isinstance(A1_data, np.ndarray):
            self.A1 = torch.from_numpy(A1_data).float().to(self.device)
        else:
            self.A1 = A1_data.float().to(self.device)

        # Load MixIK network
        PRETRAINED_MODEL_PATH = 'third_parties/smpliks/IKS/data/smpl/pretrained_model/model_best.pth.tar'
        self.generator = SMPL_MixIK(num_hidden=256).to(self.device)
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, weights_only=False, map_location=self.device)
        self.generator.load_state_dict(checkpoint['gen_state_dict'])
        self.generator.eval()

        # Initialize MixIK refine functions
        self.arm_part_func = get_body_part_func('ARM')
        self.spine_part_func = get_body_part_func('SPINE')
        self.leg_part_func = get_body_part_func('LEG')

        # Twist joint indices
        self.arm_twist_index = [13, 16, 18, 20, 14, 17, 19, 21]
        self.spine_twist_index = [3, 6, 9, 12]
        self.leg_twist_index = [1, 4, 7, 2, 5, 8]

        self._initialized = True

    @torch.no_grad()
    def forward(self, p_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert 3D skeleton to SMPL parameters

        Args:
            p_pos: 3D skeleton [B, 24, 3], centered at root

        Returns:
            theta: SMPL pose [B, 72]
            beta:  SMPL shape [B, 10]
        """
        self._initialize()

        p_pos = p_pos.to(self.device)
        batch_size = p_pos.shape[0]

        # Ensure input is centered at root
        p_pos = p_pos - p_pos[:, 0:1]

        # ============ Step 1: Shape Inverse (SI) ============
        # Generate T-pose skeleton (beta=0, theta=0)
        beta_0 = torch.zeros(batch_size, 10, dtype=torch.float32).to(self.device)
        theta_0 = torch.zeros(batch_size, 24, 3, dtype=torch.float32).to(self.device)
        smpl_out = self.smpl(theta_0, beta_0)
        t0_pos = smpl_out.joints_t.contiguous()
        t0_pos = t0_pos - t0_pos[:, 0:1]

        # Predict beta from skeleton bone lengths
        pred_beta = SMPL_SI_LR(t0_pos, p_pos, self.parent, self.A1)

        # Generate T-pose with predicted shape
        smpl_out = self.smpl(theta_0, pred_beta)
        t_pos = smpl_out.joints_t.contiguous()
        t_pos = t_pos - t_pos[:, 0:1]

        # ============ Step 2: Pose Refinement (PR-C) ============
        # Adjust skeleton directions while preserving template bone lengths
        q_pos = SMPL_AP_V1(t_pos, p_pos, self.parent, self.children)

        # ============ Step 3: AnalyIK ============
        # Analytical inverse kinematics (base solution without twist)
        analyik_theta = SMPL_AnalyIK_V3(t_pos, q_pos, self.parent, self.children)

        # ============ Step 4: MixIK (neural network predicts twist) ============
        # Generate pose for MixIK input (remove root rotation)
        theta_input = analyik_theta.clone()
        theta_input[:, 0] = 0.
        smpl_out = self.smpl(theta_input, pred_beta)
        pp_pos = smpl_out.joints.contiguous()
        pp_pos = pp_pos - pp_pos[:, 0:1]

        # Predict twist angles (cos, sin form)
        arm_pred_phi, spine_pred_phi, leg_pred_phi = self.generator(pp_pos, pred_beta)

        # Normalize phi vectors
        arm_pred_phi = arm_pred_phi / (torch.norm(arm_pred_phi, dim=2, keepdim=True) + 1e-8)
        spine_pred_phi = spine_pred_phi / (torch.norm(spine_pred_phi, dim=2, keepdim=True) + 1e-8)
        leg_pred_phi = leg_pred_phi / (torch.norm(leg_pred_phi, dim=2, keepdim=True) + 1e-8)

        # Refine pose with predicted twist
        arm_pred_theta, _ = self.arm_part_func(t_pos, q_pos, analyik_theta, arm_pred_phi)
        spine_pred_theta, _ = self.spine_part_func(t_pos, q_pos, analyik_theta, spine_pred_phi)
        leg_pred_theta, _ = self.leg_part_func(t_pos, q_pos, analyik_theta, leg_pred_phi)

        # Combine final theta
        final_theta = analyik_theta.clone()
        final_theta = final_theta.view(batch_size, 24, 3)
        final_theta[:, self.arm_twist_index] = arm_pred_theta[:, self.arm_twist_index]
        final_theta[:, self.spine_twist_index] = spine_pred_theta[:, self.spine_twist_index]
        final_theta[:, self.leg_twist_index] = leg_pred_theta[:, self.leg_twist_index]

        # Reshape to [B, 72]
        final_theta = final_theta.contiguous().view(batch_size, 72)

        return final_theta, pred_beta


# Global singleton instance
_model = None


def smpliks(pos_24: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 3D skeleton to SMPL parameters using SMPL-IKS

    Method: SI + PR-C + MixIK
    - Shape Inverse: Linear regression from bone lengths to beta
    - Pose Refinement: Analytical skeleton direction adjustment
    - MixIK: Neural network predicts twist angles for optimal pose

    Performance on 3DPW:
    - MPJPE: 0.2mm (joint error)
    - MPVPE:  10.8mm (vertex error)

    Args:
        pos_24: 3D skeleton joints in camera coordinates [B, 24, 3]
                - SMPL 24-joint topology
                - Units: meters
                - Can be uncentered (will be centered automatically)
                - Joint order: root, l_hip, r_hip, spine1, l_knee, r_knee, spine2,
                              l_ankle, r_ankle, spine3, l_foot, r_foot, neck,
                              l_collar, r_collar, head, l_shoulder, r_shoulder,
                              l_elbow, r_elbow, l_wrist, r_wrist, l_hand, r_hand

    Returns:
        theta: SMPL pose parameters [B, 72]
               - Axis-angle representation: 24 joints x 3 channels
               - Root orientation (first 3 values) is valid

        beta:  SMPL shape parameters [B, 10]
               - PCA coefficients for body shape

    Example:
        >>> import torch
        >>> from third_parties.api.smpliks import smpliks
        >>>
        >>> # Load or predict 3D skeleton [B, 24, 3]
        >>> skeleton = torch.randn(1, 24, 3).cuda()  # B=1
        >>>
        >>> # Convert to SMPL parameters
        >>> theta, beta = smpliks(skeleton)
        >>>
        >>> print(theta.shape)  # torch.Size([1, 72])
        >>> print(beta.shape)   # torch.Size([1, 10])

    Note:
        - First call initializes the model (takes ~1 second)
        - Subsequent calls reuse the initialized model
        - Model runs in eval mode (no gradients)
        - GPU memory: ~100MB for model + data
    """
    global _model

    # Initialize model on first call
    if _model is None:
        device = 'cuda' if pos_24.is_cuda else 'cpu'
        _model = _SMPLIKSModel(device=device)

    # Run inference
    theta, beta = _model.forward(pos_24)

    return theta, beta


if __name__ == '__main__':
    """Test script"""
    print("Testing SMPL-IKS API...")

    # Create dummy input
    batch_size = 2
    dummy_skeleton = torch.randn(batch_size, 24, 3).cuda()

    print(f"Input shape: {dummy_skeleton.shape}")

    # Run inference
    theta, beta = smpliks(dummy_skeleton)

    print(f"Output theta shape: {theta.shape}")  # [B, 72]
    print(f"Output beta shape: {beta.shape}")    # [B, 10]
    print(f"Theta device: {theta.device}")
    print(f"Beta device: {beta.device}")

    print("\n✓ SMPL-IKS API test passed!")
