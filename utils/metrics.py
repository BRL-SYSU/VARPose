import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd

def root_aligned(predicted, target,root_id=0, is_remove_pseudo=True):
    """
    Root aligns the skeleton.
    # Params
    - **predicted**: TxJxC ndarray
    - **target**: TxJxC ndarray
    - **root_id**: int
    """
    J = predicted.shape[1]

    root_predicted = predicted[:,root_id:root_id+1,:]
    root_target = target[:,root_id:root_id+1,:]

    if is_remove_pseudo:
        if J==17:
            pseudo_points =[]
        elif J==48:
            pseudo_points = [23, 35, 39]
        elif J==96:
            pseudo_points = [3, 7, 11, 15, 19, 23, 27, 46, 47, 70, 71, 78, 79]
        elif J==192:
            pseudo_points = [6, 7, 14, 15, 19, 22, 23, 30, 31, 38, 39, 43, 46, 47, 54, 55, 59, 67, 75, 79, 91, 92, 93, 94, 95, 107, 123, 139, 140, 141, 142, 143, 147, 156, 157, 158, 159]
        elif J==384:
            pseudo_points = [3, 12, 13, 14, 15, 19, 28, 29, 30, 31, 35, 38, 39, 44, 45, 46, 47, 60, 61, 62, 63, 76, 77, 78, 79, 86, 87, 92, 93, 94, 95, 99, 108, 109, 110, 111, 115, 118, 119, 131, 134, 135, 139, 150, 151, 158, 159, 163, 167, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 195, 203, 214, 215, 219, 227, 246, 247, 259, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 294, 295, 307, 312, 313, 314, 315, 316, 317, 318, 319, 323]
        elif J==768:
            pseudo_points = [3, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 43, 51, 56, 57, 58, 59, 60, 61, 62, 63, 67, 70, 71, 76, 77, 78, 79, 83, 88, 89, 90, 91, 92, 93, 94, 95, 99, 107, 120, 121, 122, 123, 124, 125, 126, 127, 131, 152, 153, 154, 155, 156, 157, 158, 159, 163, 172, 173, 174, 175, 184, 185, 186, 187, 188, 189, 190, 191, 198, 199, 216, 217, 218, 219, 220, 221, 222, 223, 230, 231, 236, 237, 238, 239, 243, 251, 259, 262, 263, 268, 269, 270, 271, 275, 278, 279, 283, 300, 301, 302, 303, 307, 316, 317, 318, 319, 323, 326, 327, 334, 335, 339, 347, 355, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 387, 390, 391, 395, 406, 407, 428, 429, 430, 431, 435, 438, 439, 443, 447, 451, 454, 455, 459, 467, 492, 493, 494, 495, 499, 515, 518, 519, 523, 531, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 588, 589, 590, 591, 595, 603, 614, 615, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 646, 647, 675, 679, 707]
        else:
            pseudo_points = []
        indices = np.bincount(pseudo_points, minlength=J)
        indices = indices==0
        predicted = predicted[:, indices]
        target = target[:, indices]
        
    predicted = predicted - root_predicted
    target = target - root_target
    return predicted, target

def compute_jpe(predicted, target,root_id=0)->np.ndarray:
    """
    per-joint position error
    # Params
    - **predicted**: TxJxC ndarray
    - **target**: TxJxC ndarray
    - **root_id**: int
    """
    assert predicted.shape == target.shape
    predicted, target = root_aligned(predicted, target,root_id)
    return np.linalg.norm(predicted - target, axis=len(target.shape) - 1)

def compute_mpjpe(predicted, target,root_id=0)->np.ndarray:
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    # Params
    - **predicted**: TxJxC ndarray
    - **target**: TxJxC ndarray
    - **root_id**: int
    """
    assert predicted.shape == target.shape
    predicted, target = root_aligned(predicted, target,root_id)
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1), axis=1)

def compute_acc_error(predicted, target,root_id=0):
    """
    Calculates acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    
    # Params
    - **predicted**: TxJxC ndarray
    - **target**: TxJxC ndarray
    - **root_id**: int
    """
    predicted, target = root_aligned(predicted, target,root_id)
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = predicted[:-2] - 2 * predicted[1:-1] + predicted[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    return np.mean(normed, axis=1)

def compute_p_mpjpe(predicted, target,root_id=0):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    # Params
    - **predicted**: TxJxC ndarray
    - **target**: TxJxC ndarray
    - **root_id**: int
    """
    assert predicted.shape == target.shape
    predicted, target = root_aligned(predicted, target,root_id)
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    try:
        U, s, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        print("Unable to calculate the SVD - return default 0")
        return 0 * np.ones((predicted.shape[0]))
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)

def compute_n_mpjpe(predicted, target,root_id=0):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    # Params
    - **predicted**: TxJxC ndarray
    - **target**: TxJxC ndarray
    - **root_id**: int
    """
    assert predicted.shape == target.shape
    predicted, target = root_aligned(predicted, target,root_id)
    norm_predicted = np.mean(np.sum(predicted ** 2, axis=2, keepdims=True), axis=1, keepdims=True)
    norm_target = np.mean(np.sum(target * predicted, axis=2, keepdims=True), axis=1, keepdims=True)
    scale = norm_target / norm_predicted
    return np.mean(np.linalg.norm(scale * predicted - target, axis=len(target.shape) - 1))

def compute_3d_pck(predicted, target, pck_thresh=150,root_id=14):
    """
    Simplified 3D PCK evaluation (no joint grouping)
    
    # Params:
    - predicted: np.ndarray [T,J,C] Predicted 3D joint coordinates (T frames, J joints, 3 coordinates)
    - target: np.ndarray [T,J,C] Ground truth 3D joint coordinates
    - pck_thresh: float PCK threshold (mm)
    - output_path: str Optional result output path
    - **root_id**: int
    
    # Returns:
    (pck,auc)
    - pck: float Overall PCK value (percentage)
    - auc: float Overall AUC value (percentage)
    """
    # Compute per-joint per-frame error (Euclidean distance)
    predicted, target = root_aligned(predicted, target,root_id)
    errors:np.ndarray = np.linalg.norm(predicted - target, axis=2)  # [T,J]
    total_points = errors.size  # Total number of data points (T*J)
    
    # Compute PCK curve (0-150mm)
    thresholds = np.arange(0, 151, 5)  # 0-150mm, 5mm interval
    pck_curve = np.array([np.sum(errors < t) / total_points for t in thresholds])
    
    # Compute AUC (trapezoidal method)
    auc = np.trapz(pck_curve, thresholds) / thresholds[-1]
    
    # Compute PCK at the specific threshold
    pck = np.sum(errors < pck_thresh) / total_points

    
    return pck * 100, auc * 100  # Convert to percentage

def compute_mpvpe(pred_verts: np.ndarray, gt_verts: np.ndarray) -> np.ndarray:
    """
    Mean Per Vertex Error (per-frame).
    # Params
    - **pred_verts**: TxVx3 ndarray (e.g. V=6890 for SMPL)
    - **gt_verts**:   TxVx3 ndarray
    # Returns
    - (T,) per-frame MPVPE
    """
    assert pred_verts.shape == gt_verts.shape
    return np.mean(np.linalg.norm(pred_verts - gt_verts, axis=2), axis=1)


def compute_p_mpvpe(pred_verts: np.ndarray, gt_verts: np.ndarray) -> np.ndarray:
    """
    Procrustes-aligned Mean Per Vertex Error (per-frame).
    Same SVD rigid alignment as compute_p_mpjpe, applied to vertices.
    # Params
    - **pred_verts**: TxVx3 ndarray (e.g. V=6890 for SMPL)
    - **gt_verts**:   TxVx3 ndarray
    # Returns
    - (T,) per-frame PA-MPVPE
    """
    assert pred_verts.shape == gt_verts.shape
    T = pred_verts.shape[0]

    # Centre
    mu_p = np.mean(pred_verts, axis=1, keepdims=True)
    mu_g = np.mean(gt_verts, axis=1, keepdims=True)

    P = pred_verts - mu_p
    G = gt_verts - mu_g

    # Frobenius normalise
    normP = np.sqrt(np.sum(P ** 2, axis=(1, 2), keepdims=True))
    normG = np.sqrt(np.sum(G ** 2, axis=(1, 2), keepdims=True))

    P = P / normP
    G = G / normG

    # SVD  (H: T x 3 x 3 — cheap regardless of vertex count)
    H = np.matmul(G.transpose(0, 2, 1), P)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections)
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normG / normP  # Scale
    t = mu_g - a * np.matmul(mu_p, R)  # Translation

    aligned = a * np.matmul(pred_verts, R) + t
    return np.mean(np.linalg.norm(aligned - gt_verts, axis=2), axis=1)


def compute_mpjpe_ranks(mpjpe_per_joints:np.ndarray, baseline:np.ndarray) -> tuple[np.ndarray, np.ndarray] :
    """
    Args:
        joints_errors (np.ndarray): 1xJ
        baseline (np.ndarray): 1xJ
    Returns:
        (mpjpe_ranks, important_joints)
        mpjpe_ranks (np.ndarray): Jx2
        important_joints (np.ndarray): Jx2
    """
    mpjpe_per_joints_indices = np.argsort(mpjpe_per_joints, axis=-1)[0:, ::-1] # (1, J)
    ranks_mpjpe = np.concatenate([mpjpe_per_joints_indices, mpjpe_per_joints[0:, mpjpe_per_joints_indices[0,:]]], axis=0).T

    improvement = baseline - mpjpe_per_joints
    improvement_indices = np.argsort(improvement, axis=-1)[0:, ::-1]
    ranks_improvement = np.concatenate([improvement_indices, improvement[0:, improvement_indices[0,:]]], axis=0).T

    return ranks_mpjpe, ranks_improvement
