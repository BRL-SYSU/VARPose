import torch
import numpy as np
import hashlib

# H36M standard camera parameters
H36M_AVG_FOCAL_LENGTH = 1146.79
H36M_AVG_CENTER_X = 514.04
H36M_AVG_CENTER_Y = 506.70
H36M_CANONICAL_SIZE = 1000.0

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    # model.state_dict(model_dict).requires_grad = False
    return model


def fetch_3dpw_data(data: dict, action:str|list[str]|None=None, min_num_frames: int = 81, scale=1.85, generalization_mode='bbox'):
    """
    Extract data for specified actions from the 3DPW dataset.
    
    Args:
        data: Dictionary from the loaded npz file's 'data' key.
        action: Action name or list of action names, obtained from the first
            element of each key in the data dictionary.
        min_num_frames: Minimum frame threshold; sequences shorter than this
            are filtered out (default: 81).
        scale: Scale factor used to compensate for viewpoint differences.
        generalization_mode: Generalization mode, 'bbox' (default) or 'camera'.
            - 'bbox': use bbox normalization.
            - 'camera': use camera-projection generalization.
        
    Returns:
        cameras_act: list[ndarray] - camera intrinsics list, using the first
            frame of each subsequence.
        poses_act: list[ndarray] - 3D pose list (TxJx3).
        poses_2d_act: list[ndarray] - 2D pose list (TxJx2).
        out_metadata: list[tuple] - metadata list; each tuple is
            (seq_name, subject_id, subseq_id).
    """
    data_dict = data
    
    # Normalize action parameter to list
    if action is None or action == '*':
        # Get all unique action names
        action_names = list(set([key[0] for key in data_dict.keys()]))
    elif isinstance(action, str):
        action_names = [action]
    else:
        action_names = action
    
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    out_metadata = []
    
    # Filter by action name and collect data
    for key in data_dict.keys():
        action_name = key[0]  # First element is action name
        
        # Check if this action matches filter
        if action_name not in action_names:
            continue
        
        # Get entry for this sequence
        entry = data_dict[key]
        
        # Check minimum frame count
        num_frames = entry['poses_3d'].shape[0]
        if num_frames < min_num_frames:
            continue
        
        # Collect data
        poses_3d = entry['poses_3d'][:, :17, :]
        poses_2d = entry['poses_2d']
        cam_intrinsics = entry['cam_intrinsics']
        bboxes = entry['bboxes']

        # Select different transformation methods according to the generalization mode.
        if generalization_mode == 'bbox':
            # Mode 1: use bbox normalization.
            poses_2d, cam_intrinsics = bbox_poses_to_h36m(poses_2d, cam_intrinsics, bboxes, scale)
        elif generalization_mode == 'camera':
            # Mode 2: use camera-projection generalization.
            poses_2d, cam_intrinsics = project_to_h36m(poses_2d, cam_intrinsics, scale)

        poses_3d = poses_3d[:, :17, :]
        poses_3d[:, 1:, :] = poses_3d[:, 1:, :] - poses_3d[:, 0:1, :]
        
        # Add to output lists
        out_poses_3d.append(poses_3d)
        out_poses_2d.append(poses_2d)
        out_metadata.append(key)  # Save original key: (seq_name, subject_id, subseq_id)
        
        # Convert 3x3 camera matrix to 9-dim parameter vector
        # Camera matrix format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        # Output format: [fx, fy, cx, cy, k1, k2, k3, p1, p2]
        cam_matrix = cam_intrinsics[0]
        fx = cam_matrix[0, 0]
        fy = cam_matrix[1, 1]
        cx = cam_matrix[0, 2]
        cy = cam_matrix[1, 2]
        
        # Radial and tangential distortion set to 0
        k1, k2, k3, p1, p2 = 0.0, 0.0, 0.0, 0.0, 0.0
        
        camera_params = np.array([fx, fy, cx, cy, k1, k2, k3, p1, p2], dtype='float32')
        out_camera_params.append(camera_params)
    
    return out_camera_params, out_poses_3d, out_poses_2d, out_metadata

def bbox_poses_to_h36m(poses_2d:np.ndarray, cam_intrinsics:np.ndarray, bboxes:np.ndarray, scale=1.85):
    """
    Convert normalized 2D poses to the format used by the H36M dataset.
    
    Args:
        poses_2d: Normalized 2D poses (N, J, 2).
        cam_intrinsics: Normalized camera intrinsics (N, 3, 3).
        bboxes: Original bounding boxes (N, 4) [x, y, w, h].
        scale: Scale factor.
    Returns:
        Standardized poses_2d and cam_intrinsics.
    """
    bx = bboxes[..., 0]
    by = bboxes[..., 1]
    bw = bboxes[..., 2]
    bh = bboxes[..., 3]
    
    # Compute center.
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    center = np.stack([cx, cy], axis=-1).reshape(-1, 1, 2)
    size = np.maximum(bw, bh).reshape(-1, 1, 1) * scale

    cam_intrinsics = np.reshape(cam_intrinsics, (1, 3, 3)) + np.zeros((poses_2d.shape[0], 3, 3))

    poses_2d = poses_2d - center
    cam_intrinsics[..., :2, -1:] = cam_intrinsics[..., :2, -1:] - center.reshape(-1, 2, 1)

    poses_2d = poses_2d / size * 2
    cam_intrinsics[..., :2, :] = cam_intrinsics[..., :2, :] / size * 2

    return poses_2d, cam_intrinsics

def bbox_poses_from_h36m(poses_2d: np.ndarray, cam_intrinsics: np.ndarray, bboxes: np.ndarray, scale=1.85):
    """
    Inverse of bbox_poses_to_h36m: restore normalized coordinates to the
    original space.
    
    Args:
        poses_2d: Normalized 2D poses (N, J, 2).
        cam_intrinsics: Normalized camera intrinsics (N, 3, 3).
        bboxes: Original bounding boxes (N, 4) [x, y, w, h].
        scale: Scale factor.
    Returns:
        Restored poses_2d and cam_intrinsics.
    """
    # Extract center and size from bbox.
    bx = bboxes[..., 0]
    by = bboxes[..., 1]
    bw = bboxes[..., 2]
    bh = bboxes[..., 3]
    
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    center = np.stack([cx, cy], axis=-1).reshape(-1, 1, 2)
    size = np.maximum(bw, bh).reshape(-1, 1, 1) * scale
    
    # Inverse normalization: multiply by size and divide by 2.
    poses_2d = poses_2d * size / 2
    cam_intrinsics[..., :2, :] = cam_intrinsics[..., :2, :] * size / 2
    
    # Inverse centering: add center.
    poses_2d = poses_2d + center
    cam_intrinsics[..., :2, -1:] = cam_intrinsics[..., :2, -1:] + center.reshape(-1, 2, 1)
    
    return poses_2d, cam_intrinsics


def project_to_h36m(poses_2d: np.ndarray, cam_intrinsics_3dpw: np.ndarray, scale=1.0):
    """
    Generalize 3DPW data to H36M space using camera projection.
    
    Args:
        poses_2d: 3DPW 2D poses (N, J, 2), not standardized.
        cam_intrinsics_3dpw: 3DPW camera intrinsics (N, 3, 3).
        scale: Scale factor used to compensate for viewpoint differences
            (default: 1.0).
            - Translation is fixed at the optical center and is unaffected by scale.
            - Scaling uses scale * H36M_CANONICAL_SIZE.
    
    Returns:
        poses_2d_norm: 2D poses standardized to H36M space (N, J, 2).
        cam_intrinsics_h36m: H36M standard camera intrinsics (N, 3, 3).
    """
    N = poses_2d.shape[0]
    
    # Extract 3DPW camera parameters.
    f_3dpw = np.concatenate([cam_intrinsics_3dpw[..., 0:1, 0:1], cam_intrinsics_3dpw[..., 1:2, 1:2]], axis=-1)  # (N, 1, 2)
    c_3dpw = cam_intrinsics_3dpw[..., 0:2, 2:].swapaxes(-1, -2)

    # X/Z = (u - cx) / fx
    # Y/Z = (v - cy) / fy
    poses_2d = (poses_2d - c_3dpw) / f_3dpw

    fx_h36m = H36M_AVG_FOCAL_LENGTH
    fy_h36m = H36M_AVG_FOCAL_LENGTH
    focal = np.stack([fx_h36m, fy_h36m], axis=-1).reshape(1, 1, 2)
    cx_h36m = H36M_AVG_CENTER_X
    cy_h36m = H36M_AVG_CENTER_Y
    center = np.stack([cx_h36m, cy_h36m], axis=-1).reshape(1, 1, 2)
    poses_2d = poses_2d * focal + center

    # Compute the center point (optical center) using the original size.
    w_base = h_base = H36M_CANONICAL_SIZE
    # Normalize using the scaled size.
    w = h = H36M_CANONICAL_SIZE * scale
    center_base = np.array([w_base/2, h_base/2], dtype='float32').reshape(1, 1, 2)
    
    # Translation: subtract the optical center (center of the original size).
    poses_2d = poses_2d - center_base
    # Scaling: normalize to the [-1, 1] range using the scaled size.
    poses_2d = poses_2d / w * 2
    
    # Step 6: construct the H36M standard camera intrinsic matrix.
    cam_intrinsics_h36m = np.zeros((N, 3, 3), dtype='float32')
    cam_intrinsics_h36m[:, 0, 0] = fx_h36m
    cam_intrinsics_h36m[:, 1, 1] = fy_h36m
    cam_intrinsics_h36m[:, 0, 2] = cx_h36m
    cam_intrinsics_h36m[:, 1, 2] = cy_h36m
    cam_intrinsics_h36m[:, 2, 2] = 1.0

    cam_intrinsics_h36m[..., :2, -1:] = cam_intrinsics_h36m[..., :2, -1:] - np.array([w_base, h_base]).reshape(1, 2, 1)
    cam_intrinsics_h36m[..., :2, :] = cam_intrinsics_h36m[..., :2, :] / w * 2

    return poses_2d, cam_intrinsics_h36m


def project_from_h36m(poses_2d_norm: np.ndarray, cam_intrinsics_3dpw: np.ndarray, scale=1.0):
    """
    Restore data from H36M standardized space to the original 3DPW space
    (inverse of project_to_h36m).
    
    Args:
        poses_2d_norm: 2D poses in H36M standardized space (N, J, 2).
        cam_intrinsics_3dpw: 3DPW camera intrinsics (N, 3, 3).
        scale: Scale factor used to compensate for viewpoint differences
            (default: 1.0).
            - Translation is fixed at the optical center and is unaffected by scale.
            - Scaling uses scale * H36M_CANONICAL_SIZE.
    
    Returns:
        poses_2d: 2D poses restored to the original 3DPW space (N, J, 2).
    """
    N = poses_2d_norm.shape[0]
    
    # H36M standard camera parameters.
    fx_h36m = H36M_AVG_FOCAL_LENGTH
    fy_h36m = H36M_AVG_FOCAL_LENGTH
    cx_h36m = H36M_AVG_CENTER_X
    cy_h36m = H36M_AVG_CENTER_Y
    
    # Compute the center point (optical center) using the original size.
    w_base = h_base = H36M_CANONICAL_SIZE
    # Inverse-normalize using the scaled size.
    w = h = H36M_CANONICAL_SIZE * scale
    center_base = np.array([w_base/2, h_base/2], dtype='float32').reshape(1, 1, 2)
    
    # Step 1: inverse normalization - inverse scaling and inverse translation.
    # Inverse scaling: multiply by w/2 using the scaled size.
    poses_2d = poses_2d_norm * w / 2
    # Inverse translation: add the optical center (center of the original size).
    poses_2d = poses_2d + center_base
    
    # Step 2: back-project to X/Z and Y/Z space using H36M standard camera parameters.
    focal = np.stack([fx_h36m, fy_h36m], axis=-1).reshape(1, 1, 2)
    center = np.stack([cx_h36m, cy_h36m], axis=-1).reshape(1, 1, 2)
    poses_2d = (poses_2d - center) / focal
    
    # Step 3: reproject back to 2D using 3DPW camera parameters.
    f_3dpw = np.concatenate([cam_intrinsics_3dpw[..., 0:1, 0:1], cam_intrinsics_3dpw[..., 1:2, 1:2]], axis=-1)  # (N, 1, 2)
    c_3dpw = cam_intrinsics_3dpw[..., 0:2, 2:].swapaxes(-1, -2)
    poses_2d = poses_2d * f_3dpw + c_3dpw
    
    return poses_2d
