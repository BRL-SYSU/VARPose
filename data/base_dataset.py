"""
Dataset base class - provides a unified data processing interface for pose estimation projects
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union


class BaseDataset(Dataset, ABC):
    """Dataset base class, defines a unified data processing interface
    
    All datasets should inherit from this base class and implement the abstract methods.
    Provides standard data preprocessing, normalization, and access interfaces.
    """
    
    # Standard data output format definition
    STANDARD_OUTPUT_FORMAT = {
        'poses_2d': Dict[int, torch.Tensor],        # Multi-scale 2D pose dictionary
        'poses_3d': Dict[int, torch.Tensor],        # Multi-scale 3D pose dictionary
        'action_label': torch.Tensor,               # Action label
        'bbox': torch.Tensor,                   # Image dimensions [height, width]
        'normalized_root': torch.Tensor,            # [..., 5] dimension 2 + dimension 3
        'campose_valid': torch.Tensor,
    }
    
    def __init__(self, data_path: str, mode: str = 'train', **kwargs):
        """Initialize the dataset
        
        Args:
            data_path: Dataset path
            mode: Data mode ('train', 'test', 'val', 'warmup')
            **kwargs: Mode-specific parameters
        """
        self.data_path = data_path
        self.mode = mode
        self.mode_config = kwargs
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Preprocess data based on mode
        self.processed_data = self.prepare_data(raw_data)
        del raw_data
        
    
    @abstractmethod
    def get_adj_tuples(self) -> Dict[int, Tuple[Tuple[int, int]]]:
        """Get adjacency tuples, more readable than adjacency matrix
        
        Returns:
            Dict[int, Tuple[Tuple[int, int]]]: Adjacency tuples, each tuple represents the connection between two joints
        """
        pass
    
    @abstractmethod
    def get_root_ids(self) -> Dict[int, int]:
        """Get root joint IDs for each level
        
        Returns:
            Dict[int, int]: Root joint IDs for each level
        """
        pass
    
    @abstractmethod
    def get_pseudo_points_dict(self) -> Dict[int, List[int]]:
        """Get pseudo point dictionary, defines pseudo point positions for each level
        
        Returns:
            Dict[int, List[int]]: Keys are the number of joints per level, values are lists of pseudo point indices
        """
        pass
    
    @abstractmethod
    def load_raw_data(self) -> Any:
        """Load raw data
        
        Returns:
            Any: Raw data, format defined by subclass
        """
        pass
    
    @abstractmethod
    def prepare_data(self, raw_data) -> List[Dict]:
        """Prepare data based on mode
        
        Returns:
            List[Any]: List of processed data
        """
        pass

    def remove_pseudo_points(self, pose:torch.Tensor)->torch.Tensor:
        """Remove pseudo points
        
        Removes pseudo nodes from the pose tensor.
        
        Args:
            pose: Pose tensor [..., num_joints, D]
            
        Returns:
            torch.Tensor: Pose tensor with pseudo points zeroed out
        """
        pseudo_points_dict = self.get_pseudo_points_dict()
        num_joints = pose.shape[-2]
        pseudo_idx = pseudo_points_dict.get(num_joints, [])
        mask = torch.ones(num_joints, dtype=torch.bool, device=pose.device)
        if pseudo_idx:
            mask[pseudo_idx]=False
        pose = pose[..., mask, :]
        return pose
        
    
    def get_adj_tuples_symmetry_augmented(self)-> Dict[int, Tuple[Tuple[int, int]]]:
        """Get symmetry-augmented adjacency tuples
        
        Returns:
            out: Dict[int,Tuple[Tuple[int, int]]] Symmetry-augmented adjacency tuples for each level
        """
        return self.get_adj_tuples()

    def get_dataset_stats(self) -> Dict[str, float]:
        """Get dataset statistics (mean, standard deviation, etc.)
        
        Returns:
            Dict[str, float]: Contains statistics such as mean and standard deviation, returns an empty dict by default
        """
        return {}
    
    def get_alignment_info(self) -> Optional[Tuple]:
        """Get alignment info (optional)
        
        Returns:
            Optional[Tuple]: Alignment info, returns None if not needed
        """
        return None
    
    def get_supported_modes(self) -> List[str]:
        """Get supported data modes
        
        Returns:
            List[str]: List of supported modes ['train', 'test', 'val', 'warmup']
        """
        return ['train', 'test', 'val', 'warmup']
    
    def action_label_to_str(self, action_label:int) -> str:
        return ""
    
    
    @staticmethod
    def normalize_pose(pose: torch.Tensor, bbox:torch.Tensor) -> torch.Tensor:
        """Pose normalization
        
        Args:
            pose: Raw pose data (..., N, D)
            bbox: Bounding box (..., 4) [x, y, width, height]
            
        Returns:
            torch.Tensor: Normalized pose data
        """
        bx = bbox[..., 0]
        by = bbox[..., 1]
        bw = bbox[..., 2]
        bh = bbox[..., 3]
        
        # Compute center
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        center = torch.stack([cx, cy], dim=-1)
        size = torch.max(torch.stack([bw, bh], dim=-1), dim=-1)[0]
        size = size.unsqueeze(dim=-1)
        size[size<1e-5] = 1.0 # Prevent division by zero

        D = pose.shape[-1]

        if D==2:
            # 2D normalization: X,Y -> X,Y / a * 2 - [1, 1] as square side length
            pose_norm = (pose - center) / size * 2.0
            
        elif D==3:
            # 3D normalization: convert units
            pose_norm = pose / 1000.0
        else:
            raise ValueError(f"Unsupported dimesion: {D}")
        
        return pose_norm
    
    @staticmethod
    def denormalize_pose(pose: torch.Tensor, normalized_root: torch.Tensor, bbox:torch.Tensor) -> torch.Tensor:
        """Pose denormalization
        
        Args:
            pose: Normalized pose data (..., N, D)
            normalized_root: Normalized root joint data (..., 1, 5)
            bbox: Bounding box (..., 4) [x, y, width, height]
            
        Returns:
            torch.Tensor: Denormalized pose data
        """
        bx = bbox[..., 0]
        by = bbox[..., 1]
        bw = bbox[..., 2]
        bh = bbox[..., 3]
        
        # Compute center
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        center = torch.stack([cx, cy], dim=-1)
        size = torch.max(torch.stack([bw, bh], dim=-1), dim=-1)[0]
        size = size.unsqueeze(dim=-1)
        size[size<1e-5] = 1.0 # Prevent division by zero

        D = pose.shape[-1]
        
        if D == 2:
            pose = pose + normalized_root[..., :2]
            pose = pose * size / 2.0 + center
        elif D == 3:
            pose = pose + normalized_root[..., 2:]
            pose = pose * 1000.0
        else:
            raise ValueError(f"Invalid dimension: {D}")
        
        return pose

    @staticmethod
    def merge_dataset(inp_path:str, out_path:str, var_save_dir:str, var_result_relative_path=["train_best/phats_denormed.pkl", "test_best/phats_denormed.pkl"], preserve_sparse_joints=True):
        pass
    
    def clear_root_and_pseudo(self, phat: torch.Tensor) -> torch.Tensor:
        """Clear the predictions of the root joint and pseudo nodes
        
        Call this function before evaluation to mask the noisy results of the root joint and pseudo nodes,
        so they do not affect the computation of evaluation metrics such as MPJPE.
        
        Args:
            phat: Predicted pose tensor [..., num_joints, D]
            
        Returns:
            torch.Tensor: Pose tensor with root joint and pseudo nodes cleared
        """
        num_joints = phat.shape[-2]
        root_id = self.get_root_ids()[num_joints]
        phat[..., root_id:root_id+1, :] = 0.0
        
        pseudo_idx = self.get_pseudo_points_dict()[num_joints]
        phat[..., pseudo_idx, :] = 0.0
        return phat
    
    @staticmethod
    def get_bbox(joint_img):
        """Compute bounding box, supports vectorized operations
        
        Args:
            joint_img: Joint image coordinates [N, 2] or [T, N, 2]
            
        Returns:
            torch.Tensor: Bounding box [4] or [T, 4], format is [xmin, ymin, width, height]
        """
        x_img, y_img = joint_img[..., 0], joint_img[..., 1]
        
        # Supports vectorization: handles [T, N, 2] or [N, 2]
        xmin = torch.min(x_img, dim=-1).values
        ymin = torch.min(y_img, dim=-1).values
        xmax = torch.max(x_img, dim=-1).values
        ymax = torch.max(y_img, dim=-1).values

        x_center, width = (xmin+xmax)/2., xmax-xmin
        xmin = x_center - 0.5*width
        xmax = x_center + 0.5*width
        
        y_center, height = (ymin+ymax)/2., ymax-ymin
        ymin = y_center - 0.5*height
        ymax = y_center + 0.5*height

        bbox = torch.stack([xmin, ymin, xmax - xmin, ymax - ymin], dim=-1)
        return bbox
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dict[str, torch.Tensor]: Processed sample data
        """
        return self.processed_data[idx]


class PoseDataForVector:
    def __init__(self, pose_2d:torch.Tensor, pose_3d:torch.Tensor, action_label:torch.Tensor, bbox:torch.Tensor, normalized_root:torch.Tensor, cumulative_joints:torch.Tensor, campose_valid:torch.Tensor|None=None, cumulative_joints_3d:torch.Tensor|None=None, level_keys:List[str]|None=None):
        self.pose_2d = pose_2d # [TxNx2]
        self.pose_3d = pose_3d # [TxNx3]
        self.action_label = action_label # [Tx1]
        self.bbox = bbox # [Tx1x4]
        self.normalized_root = normalized_root # [Tx1x5]
        assert pose_2d.shape[0] == pose_3d.shape[0] and pose_3d.shape[0] == action_label.shape[0] and action_label.shape[0] == bbox.shape[0]
        self.campose_valid = campose_valid  if campose_valid is not None else torch.ones((self.pose_2d.shape[0], 1), dtype=torch.bool)

        self.cumulative_joints = cumulative_joints # [N+1]
        self.cumulative_joints_3d = cumulative_joints_3d # [N+1]
        self.level_keys = level_keys
    
    def __getitem__(self, idx:int):
        poses_2d = {}
        poses_3d = {}
        for i, (start, end) in enumerate(zip(self.cumulative_joints[:-1], self.cumulative_joints[1:])):
            int_level = (end - start).item()
            val_2d = self.pose_2d[idx, start:end, :]
            if self.level_keys is not None and i < len(self.level_keys) and self.level_keys[i]!=str(int_level):
                poses_2d[self.level_keys[i]] = val_2d
            else:
                poses_2d[int_level] = val_2d
            if self.cumulative_joints_3d is None:
                val_3d = self.pose_3d[idx, start:end, :]
                if self.level_keys is not None and i < len(self.level_keys) and self.level_keys[i]!=str(int_level):
                    poses_3d[self.level_keys[i]] = val_3d
                else:
                    poses_3d[int_level] = val_3d

        if self.cumulative_joints_3d is not None:
            for start, end in zip(self.cumulative_joints_3d[:-1], self.cumulative_joints_3d[1:]):
                level = (end - start).item()
                poses_3d[level] = self.pose_3d[idx, start:end, :]

        sample = {
            'poses_2d': poses_2d,      # Multi-scale 2D pose dictionary
            'poses_3d': poses_3d,      # Multi-scale 3D pose dictionary
            'action_label': self.action_label[idx],    # Action label
            'bbox': self.bbox[idx],       # Image dimensions [height, width]
            'normalized_root': self.normalized_root[idx], # [..., 5] dimension 2 + dimension 3
            'campose_valid': self.campose_valid[idx],
        }
        return sample
    
    def __len__(self):
        return self.pose_2d.shape[0]