"""
SMPL Mesh Evaluation Task

Evaluates lifting predictions by converting 17-joint predictions to SMPL mesh
and comparing with ground-truth SMPL parameters.

Input NPZ format (from lifting save_prediction, extended):
    prediction:  (T, 17, 3)  H36M 3D joints, camera space, meters
    smpl_param:  dict-like with keys:
        theta:  (T, 72)   GT SMPL pose parameters
        beta:   (T, 10)   GT SMPL shape parameters
        gender: (T,) str  gender per frame  (currently uses NEUTRAL for all)

Metrics:
    MPJPE-24   Mean Per Joint Position Error on 24 SMPL joints (mm)
    PMPJPE-24  PA-MPJPE on 24 SMPL joints (mm)
    MPVPE       Mean Per Vertex Error on 6890 mesh vertices (mm)
    PMPVPE      PA-MPVPE on 6890 mesh vertices (mm)

Usage:
    python run.py --task smpl_mesh_eval_task \
        --npz_path path/to/predictions.npz \
        --h36m2smpl_checkpoint path/to/h36m2smpl_best.pth
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from utils.task.base_task import BaseTask


# ---------------------------------------------------------------------------
# smpliks sys.path setup  (for SMPLLayer used in SMPL forward pass)
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
_smpliks_path = _project_root / 'third_parties' / 'smpliks'
if str(_smpliks_path) not in sys.path:
    sys.path.insert(0, str(_smpliks_path))

SMPL_KID_MODEL_PATH = str(
    _project_root / 'third_parties/smpliks/IKS/data/smpl/smpliks_data/smpl_kid_template.npy'
)
SMPL_GENDER_MODELS = {
    'neutral': str(_project_root / 'third_parties/smpliks/IKS/data/smpl/smpliks_data/SMPL_NEUTRAL.pkl'),
    'male': str(_project_root / 'third_parties/smpliks/IKS/data/smpl/smpliks_data/SMPL_MALE.pkl'),
    'female': str(_project_root / 'third_parties/smpliks/IKS/data/smpl/smpliks_data/SMPL_FEMALE.pkl'),
}


# ============================================================
# Helper: SMPL forward -> vertices + joints
# ============================================================

_smpl_cache: dict = {}

def _get_smpl_layer(gender: str, device: str = 'cuda'):
    """Return a cached SMPLLayer for the given gender."""
    from lib.models.smpl.smpl import SMPLLayer

    if gender == "m":
        gender = "male"
    elif gender == 'f':
        gender = "female"

    key = (gender, device)
    if key not in _smpl_cache:
        model_path = SMPL_GENDER_MODELS.get(gender, SMPL_GENDER_MODELS['neutral'])
        _smpl_cache[key] = SMPLLayer(
            model_path=model_path,
            kid_template_path=SMPL_KID_MODEL_PATH,
            dtype=torch.float32,
            age='adult',
        ).to(device).eval()
    return _smpl_cache[key]


@torch.no_grad()
def _smpl_forward_batched(smpl, theta_72: torch.Tensor, beta_10: torch.Tensor,
                          batch_size: int = 64):
    """
    Run SMPL forward in chunks.  Returns (vertices, joints_24) concatenated.

    Args:
        smpl:         SMPLLayer on device
        theta_72:     (N, 72)  on device
        beta_10:      (N, 10)  on device
        batch_size:   chunk size

    Returns:
        vertices:  (N, 6890, 3)  numpy, meters
        joints24:  (N, 24, 3)    numpy, meters
    """
    N = theta_72.shape[0]
    verts_list, joints_list = [], []

    for i in range(0, N, batch_size):
        th = theta_72[i:i + batch_size].view(-1, 24, 3)
        be = beta_10[i:i + batch_size]
        out = smpl(th, be)
        verts_list.append(out.vertices.cpu().numpy())
        joints_list.append(out.joints[:, :24].cpu().numpy())

    return np.concatenate(verts_list, axis=0), np.concatenate(joints_list, axis=0)


@torch.no_grad()
def _smpl_forward_by_gender(genders: np.ndarray, theta_72: np.ndarray,
                             beta_10: np.ndarray, device: str,
                             batch_size: int = 64):
    """
    SMPL forward with per-frame gender routing.

    Args:
        genders:   (T,) str array  — 'male' / 'female' / 'neutral'
        theta_72:  (T, 72)  numpy
        beta_10:   (T, 10)  numpy
        device:    torch device
        batch_size: chunk size per gender group

    Returns:
        vertices:  (T, 6890, 3)  numpy, meters  (original order preserved)
        joints24:  (T, 24, 3)    numpy, meters  (original order preserved)
    """
    T = theta_72.shape[0]
    all_verts = np.empty((T, 6890, 3), dtype=np.float32)
    all_joints = np.empty((T, 24, 3), dtype=np.float32)

    unique_genders = np.unique(genders)
    for g in unique_genders:
        mask = np.array([genders[i] == g for i in range(T)])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        smpl = _get_smpl_layer(g, device)
        th = torch.from_numpy(theta_72[idx]).float().to(device)
        be = torch.from_numpy(beta_10[idx]).float().to(device)
        v, j = _smpl_forward_batched(smpl, th, be, batch_size=batch_size)
        all_verts[idx] = v
        all_joints[idx] = j

    return all_verts, all_joints


class SMPLMeshEvalTask(BaseTask):
    """
    Evaluate lifting predictions as SMPL mesh against GT SMPL parameters.

    Pipeline:
        1. Load NPZ  (prediction + smpl_param)
        2. prediction (T,17,3) -> h36m_to_smpl_params -> (theta, beta)
        3. SMPL forward  ->  vertices (6890) + joints (24)
        4. Root-align using pelvis (joint 0)
        5. Compute MPJPE-24 / PMPJPE-24 / MPVPE / PMPVPE
    """

    @staticmethod
    def add_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        g = parser.add_argument_group('SMPL Mesh Eval Configuration')

        g.add_argument('--npz_path', type=str, required=True,
                       help='Path to NPZ with prediction + smpl_param')
        g.add_argument('--h36m2smpl_checkpoint', type=str, required=True,
                       help='H36MToSMPLConverter checkpoint (.pth)')
        g.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for SMPL processing')

        g.add_argument('--embed_dim', type=int, default=256)
        g.add_argument('--num_heads', type=int, default=8)
        g.add_argument('--num_gcn_layers', type=int, default=6)
        g.add_argument('--num_transformer_layers', type=int, default=6)
        g.add_argument('--gcn_hidden_dim', type=int, default=128)
        g.add_argument('--dropout', type=float, default=0.1)
        g.add_argument('--save_prediction', type=str, default=None,
                       help='Save inferred SMPL params (pred theta/beta) to npz')
        return parser

    # ------------------------------------------------------------------
    def run(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"Task: {self.task_name}")
        print(f"{'=' * 80}\n")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bs = self.args.batch_size

        # ── 1. Load NPZ ────────────────────────────────────────────────
        print("[1/5] Loading NPZ file ...")
        data = np.load(self.args.npz_path, allow_pickle=True)

        _pred = data['prediction']
        predictions = _pred.item() if _pred.ndim == 0 else _pred          # (T, 17, 3)
        predictions -= predictions[..., 0:1, :]
        smpl_param = data['smpl_param'].item()

        gt_theta = smpl_param['theta']                            # (T, 72)
        gt_beta = smpl_param['beta']                              # (T, 10)
        gt_genders = smpl_param['gender']                         # (T,) str
        gt_trans = smpl_param['trans']                            # (T, 1, 3)
        gt_root_rot = gt_theta[:, :3].copy()                      # save original root rotation
        gt_theta[..., :3] = 0

        T = predictions.shape[0]
        print(f"  Frames: {T}")
        print(f"  prediction : {predictions.shape}")
        print(f"  gt_theta   : {gt_theta.shape}")
        print(f"  gt_beta    : {gt_beta.shape}")

        # ── 2. Init API  (17 -> SMPL params) ──────────────────────────
        print("[2/5] Initializing H36M->SMPL API ...")
        from models.h36m_to_smpl import init_h36m_to_smpl_api, h36m_to_smpl_params
        init_h36m_to_smpl_api(
            checkpoint_path=self.args.h36m2smpl_checkpoint,
            device=device,
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_heads,
            num_gcn_layers=self.args.num_gcn_layers,
            num_transformer_layers=self.args.num_transformer_layers,
            gcn_hidden_dim=self.args.gcn_hidden_dim,
            dropout=self.args.dropout,
        )

        # ── 3. Convert prediction -> SMPL params ──────────────────────
        print("[3/5] Converting predictions to SMPL params ...")
        pred_theta_parts, pred_beta_parts = [], []
        for i in range(0, T, bs):
            chunk = torch.from_numpy(predictions[i:i + bs]).float().to(device)
            th, be = h36m_to_smpl_params(chunk)
            pred_theta_parts.append(th.cpu())
            pred_beta_parts.append(be.cpu())

        pred_theta = torch.cat(pred_theta_parts, dim=0)            # (T, 72)
        pred_beta = torch.cat(pred_beta_parts, dim=0)              # (T, 10)
        print(f"  pred_theta: {pred_theta.shape}, pred_beta: {pred_beta.shape}")
        pred_theta[..., :3]=0

        # ── 4. SMPL forward (pred + GT) ───────────────────────────────
        print("[4/5] Running SMPL forward pass ...")
        # Prediction mesh (neutral SMPL — h36m_to_smpl pipeline uses neutral)
        smpl_neutral = _get_smpl_layer('neutral', device)
        pred_theta_dev = pred_theta.to(device)
        pred_beta_dev = pred_beta.to(device)
        pred_verts, pred_joints = _smpl_forward_batched(
            smpl_neutral, pred_theta_dev, pred_beta_dev, batch_size=bs,
        )

        # GT mesh (gender-specific SMPL)
        gt_verts, gt_joints = _smpl_forward_by_gender(
            gt_genders, gt_theta, gt_beta, device, batch_size=bs,
        )

        print(f"  pred_verts : {pred_verts.shape}")
        print(f"  gt_verts   : {gt_verts.shape}")

        # ── 5. Metrics (GPU batched) ─────────────────────────────────
        print("[5/5] Computing metrics (GPU batched) ...")

        # Root-align: subtract pelvis (joint 0) from both verts and joints
        pred_verts = pred_verts - pred_joints[:, 0:1]
        gt_verts = gt_verts - gt_joints[:, 0:1]
        pred_joints = pred_joints - pred_joints[:, 0:1]
        gt_joints = gt_joints - gt_joints[:, 0:1]

        # Convert m -> mm  and move to torch
        pred_joints_t = torch.from_numpy(pred_joints * 1000).float().to(device)
        gt_joints_t = torch.from_numpy(gt_joints * 1000).float().to(device)
        pred_verts_t = torch.from_numpy(pred_verts * 1000).float().to(device)
        gt_verts_t = torch.from_numpy(gt_verts * 1000).float().to(device)

        @torch.no_grad()
        def _batched_mpjpe(pred, gt, batch_size):
            errs = []
            for i in range(0, pred.shape[0], batch_size):
                d = pred[i:i+batch_size] - gt[i:i+batch_size]
                errs.append(d.norm(dim=-1).mean(dim=-1))
            return torch.cat(errs)

        @torch.no_grad()
        def _batched_procrustes(pred, gt, batch_size):
            errs = []
            for i in range(0, pred.shape[0], batch_size):
                p = pred[i:i+batch_size]
                g = gt[i:i+batch_size]
                mu_p = p.mean(dim=1, keepdim=True)
                mu_g = g.mean(dim=1, keepdim=True)
                P = p - mu_p
                G = g - mu_g
                normP = (P ** 2).sum(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-8)
                normG = (G ** 2).sum(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-8)
                Pn = P / normP
                Gn = G / normG
                H = torch.bmm(Gn.transpose(1, 2), Pn)
                U, S, Vt = torch.linalg.svd(H)
                V = Vt.transpose(1, 2)
                R = torch.bmm(V, U.transpose(1, 2))
                sign_detR = torch.sign(torch.linalg.det(R).unsqueeze(-1))
                V[:, :, -1] = V[:, :, -1] * sign_detR
                S[:, -1] = S[:, -1] * sign_detR.squeeze(-1)
                R = torch.bmm(V, U.transpose(1, 2))
                tr = S.sum(dim=1, keepdim=True).unsqueeze(-1)
                a = tr * normG / normP
                t = mu_g - a * torch.bmm(mu_p, R)
                aligned = a * torch.bmm(p, R) + t
                errs.append((aligned - g).norm(dim=-1).mean(dim=-1))
            return torch.cat(errs)

        metric_bs = 256
        mpjpe_24 = _batched_mpjpe(pred_joints_t, gt_joints_t, metric_bs)
        print(f"  MPJPE-24 done")
        pmpjpe_24 = _batched_procrustes(pred_joints_t, gt_joints_t, metric_bs)
        print(f"  PMPJPE-24 done")
        mpvpe = _batched_mpjpe(pred_verts_t, gt_verts_t, metric_bs)
        print(f"  MPVPE done")
        pmpvpe = _batched_procrustes(pred_verts_t, gt_verts_t, metric_bs)
        print(f"  PMPVPE done")

        del pred_verts_t, gt_verts_t, pred_joints_t, gt_joints_t
        torch.cuda.empty_cache()

        # ── Report ────────────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  Results  ({T} frames)")
        print(f"{'=' * 60}")
        print(f"  MPJPE-24 :  {mpjpe_24.mean().item():.2f} mm")
        print(f"  PMPJPE-24:  {pmpjpe_24.mean().item():.2f} mm")
        print(f"  MPVPE     :  {mpvpe.mean().item():.2f} mm")
        print(f"  PMPVPE    :  {pmpvpe.mean().item():.2f} mm")
        print(f"{'=' * 60}\n")

        # ── Save prediction ────────────────────────────────────────────
        if self.args.save_prediction:
            os.makedirs(os.path.dirname(self.args.save_prediction), exist_ok=True)
            gt_theta[:, :3] = gt_root_rot
            pred_theta_np = pred_theta.numpy()
            pred_theta_np[:, :3] = gt_root_rot
            save_dict = {
                'prediction': {'theta': pred_theta_np, 'beta': pred_beta.numpy()},
                'gt': {'theta': gt_theta, 'beta': gt_beta, 'gender':gt_genders, "trans": gt_trans},
            }
            for key in data.files:
                if key not in ('prediction', 'smpl_param'):
                    save_dict[key] = data[key]
            np.savez(self.args.save_prediction, **save_dict)
            print(f"Predictions saved to {self.args.save_prediction}")
