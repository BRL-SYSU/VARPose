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
        else:
            raise ValueError('Joint number error, please check utils/root_aligned')
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
    Simplified 3D PCK evaluation without joint grouping.
    
    # Params:
    - predicted: np.ndarray [T,J,C] predicted 3D joint coordinates
      (T frames, J joints, 3 coordinates)
    - target: np.ndarray [T,J,C] ground-truth 3D joint coordinates
    - pck_thresh: float PCK threshold (mm)
    - output_path: str optional result output path
    - **root_id**: int
    
    # Returns:
    (pck,auc)
    - pck: float overall PCK value (percentage)
    - auc: float overall AUC value (percentage)
    """
    # Compute per-joint per-frame errors (Euclidean distance).
    predicted, target = root_aligned(predicted, target,root_id)
    errors:np.ndarray = np.linalg.norm(predicted - target, axis=2)  # [T,J]
    total_points = errors.size  # Total number of data points (T*J)
    
    # Compute the PCK curve (0-150 mm).
    thresholds = np.arange(0, 151, 5)  # 0-150 mm, 5 mm interval
    pck_curve = np.array([np.sum(errors < t) / total_points for t in thresholds])
    
    # Compute AUC using the trapezoidal rule.
    auc = np.trapezoid(pck_curve, thresholds) / thresholds[-1]
    
    # Compute PCK at the specified threshold.
    pck = np.sum(errors < pck_thresh) / total_points

    
    return pck * 100, auc * 100  # Convert to percentage

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
