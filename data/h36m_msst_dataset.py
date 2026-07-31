if __name__ == "__main__":
    import sys, os
    import time
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
    print(sys.path)

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import os
from typing import Dict, List, Tuple, Any, Optional

from data.base_dataset import BaseDataset, PoseDataForVector


class H36M_MSST_Dataset(BaseDataset):

    def __init__(self, data_path: str, mode: str = 'train', **kwargs):
        self.stride = kwargs.get('stride', 1)
        super().__init__(data_path, mode, **kwargs)

    def load_raw_data(self) -> Dict[str, Any]:
        """Load raw data
        
        Returns:
            Any: Raw data, contains training and test data
        """
        pkl_path = self.data_path
        var_save_dir = self.mode_config.get('var_save_dir', '')
        var_result_relative_path = self.mode_config.get('var_result_relative_path', ["train_best/phats_denormed.pkl", "test_best/phats_denormed.pkl"])
        
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        
        # Try to load estimated data
        try:
            if self.mode in ['train', 'warmup']:
                mode_key = 'train'
                relative_path_idx = 0
            else:  # test, val
                mode_key = 'test'
                relative_path_idx = 1
            
            with open(os.path.join(var_save_dir, var_result_relative_path[relative_path_idx]), 'rb') as f:
                tmp_data: List[np.ndarray] = pickle.load(f)  # [Tx17x2, Tx48x2, Tx96x2]
                tmp_data.reverse()
            
            original_data = all_data[mode_key]['coords']
            if len(original_data) == tmp_data[0].shape[0]:
                for i in range(len(original_data)):
                    original_data[i]["2d"] = [temp[i, :, :] for temp in tmp_data]
            else:
                raise ValueError(f"Invalid data length")
        except Exception as e:
            original_data = all_data[mode_key]['coords']
        
        del all_data
        
        if self.mode == "warmup":
            csv_path = self.mode_config.get('csv_path', "./data/mpjpe_per_joints.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['sample_idx'] = df['sample_idx'].astype(int)
                
                sorted_df = df.sort_values('overall_mpjpe')
                sorted_indices = sorted_df['sample_idx'].values
                
                original_data = [original_data[i] for i in sorted_indices]
                del df

        return original_data[::self.stride]
    
    def prepare_data(self, raw_data: Dict[str, Any]) -> List[Dict]:
        """Prepare data based on mode
        
        Args:
            raw_data: Raw data
            
        Returns:
            List[Dict]: List of processed data
        """
        bboxes = []
        poses_2d = []
        poses_3d = []
        action_labels = []
        keys = []
        level_keys = []
        for i, sample in enumerate(raw_data):
            metadata = sample['metadata']
            if 'img_hw' in metadata:
                h, w = metadata['img_hw']
            else:
                cam_idx = metadata.get('cam_idx', 1)
                if cam_idx in [1, 4]:
                    h, w = 1002, 1000
                elif cam_idx in [2, 3]:
                    h, w = 1000, 1000
                else:
                    h, w = 1000, 1000 
            bbox = torch.tensor([0, 0, w, h], dtype=torch.float32).unsqueeze(dim=0)
            original_action_label = metadata.get('action_idx', 2)  
            corrected_action_label = original_action_label - 2
            action_label = torch.clamp(torch.tensor(corrected_action_label, dtype=torch.long), 0, 14)

            poses_2d_dict:dict[int, torch.Tensor] = {}
            poses_3d_dict:dict[int, torch.Tensor] = {}
            str_key_entries:dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            has_3d = '3d' in sample
            if has_3d:
                for x,y in zip(sample['2d'], sample['3d']):
                    poses_2d_dict[x.shape[0]] = torch.from_numpy(x).float()
                    poses_3d_dict[y.shape[0]] = torch.from_numpy(y).float()
            else:
                for x in sample['2d']:
                    poses_2d_dict[x.shape[0]] = torch.from_numpy(x).float()
                    poses_3d_dict[x.shape[0]] = torch.zeros(x.shape[0], 3, dtype=torch.float32)

            if "other_2d" in sample:
                for level in sample["other_2d"].keys():
                    if level == 17:
                        coco_2d = torch.from_numpy(sample["other_2d"][level]).float()
                        coco_3d = torch.zeros(coco_2d.shape[0], 3, dtype=torch.float32)
                        if "other_3d" in sample and level in sample["other_3d"]:
                            coco_3d = torch.from_numpy(sample["other_3d"][level]).float()
                        str_key_entries['coco_17'] = (coco_2d, coco_3d)
                        continue
                    poses_2d_dict[level] = torch.from_numpy(sample["other_2d"][level]).float()
                    if "other_3d" in sample and level in sample["other_3d"]:
                        poses_3d_dict[level] = torch.from_numpy(sample["other_3d"][level]).float()
                    else:
                        poses_3d_dict[level] = torch.zeros(sample["other_2d"][level].shape[0], 3, dtype=torch.float32)
            
            if not keys:
                keys = list(sorted(poses_2d_dict.keys()))
                for sk in sorted(str_key_entries.keys()):
                    keys.append(sk)
                level_keys = [str(k) if isinstance(k, int) else k for k in keys]

            all_2d = [poses_2d_dict[k] for k in keys if isinstance(k, int)]
            all_3d = [poses_3d_dict[k] for k in keys if isinstance(k, int)]
            for sk in [k for k in keys if not isinstance(k, int)]:
                all_2d.append(str_key_entries[sk][0])
                all_3d.append(str_key_entries[sk][1])
            poses_2d.append(torch.concatenate(all_2d, dim=0))
            poses_3d.append(torch.concatenate(all_3d, dim=0))
            action_labels.append(action_label)
            bboxes.append(bbox)
        poses_2d = torch.stack(poses_2d, dim=0)
        poses_3d = torch.stack(poses_3d, dim=0)
        action_label = torch.stack(action_labels, dim=0)
        bbox = torch.stack(bboxes, dim=0)
        int_sizes = [k if isinstance(k, int) else int(k.split('_')[-1]) for k in keys]
        cumulative_joints = torch.cumsum(torch.tensor([0] + int_sizes), dim=0)

        # Get root joint
        root_2d = poses_2d[:, :1, :].clone(); root_3d = poses_3d[:, :1, :].clone()
        normed_root_2d = self.normalize_pose(root_2d, bbox); normed_root_3d = self.normalize_pose(root_3d, bbox)
        normed_root = torch.concatenate([normed_root_2d, normed_root_3d], dim=-1)
        poses_2d = self.normalize_pose(poses_2d, bbox) - normed_root[..., :2]
        poses_3d = self.normalize_pose(poses_3d, bbox) - normed_root[..., 2:]
        pseudo_dict = self.get_pseudo_points_dict()
        for i, (start, end) in enumerate(zip(cumulative_joints[:-1], cumulative_joints[1:])):
            int_level = (end - start).item()
            if int_level in pseudo_dict:
                poses_2d[:, start:end, :][:, pseudo_dict[keys[i]], :] = 0.0
                poses_3d[:, start:end, :][:, pseudo_dict[keys[i]], :] = 0.0

        data = PoseDataForVector(poses_2d, poses_3d, action_label, bbox, normed_root, cumulative_joints, level_keys=level_keys)

        return data
    
    @staticmethod
    def get_adj_tuples() -> Dict[int, Tuple[Tuple[int, int]]]:
        """Get adjacency tuples, more readable than adjacency matrix
        
        Returns:
            Dict[int, Tuple[Tuple[int, int]]]: Adjacency tuples, each tuple represents the connection between two joints
        """
        from data.common_variables import smpl_adj_tuples
        connection = smpl_adj_tuples.copy()
        return connection
    
    @staticmethod
    def get_adj_tuples_symmetry_augmented():
        connection = H36M_MSST_Dataset.get_adj_tuples().copy()
        connection[17] = (((0, 0), (0, 1), (0, 4), (0, 7), (1, 0), (1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 3), (2, 5), (3, 2), (3, 3), (3, 6), (4, 0), (4, 1), (4, 4), (4, 5), (5, 2), (5, 4), (5, 5), (5, 6), (6, 3), (6, 5), (6, 6), (7, 0), (7, 7), (7, 8), (8, 7), (8, 8), (8, 9), (8, 11), (8, 14), (9, 8), (9, 9), (9, 10), (10, 9), (10, 10), (11, 8), (11, 11), (11, 12), (11, 14), (12, 11), (12, 12), (12, 13), (12, 15), (13, 12), (13, 13), (13, 16), (14, 8), (14, 11), (14, 14), (14, 15), (15, 12), (15, 14), (15, 15), (15, 16), (16, 13), (16, 15), (16, 16)))
        return connection
    
    @staticmethod
    def get_root_ids() -> Dict[int, int]:
        """Get root joint IDs for each level
        
        Returns:
            Dict[int, int]: Root joint IDs for each level
        """
        return {
            17: 0,  # Root joint ID for 17 joints
            21: 0,
            25: 0,
            24: 0,
            48: 23,  # Root joint ID for 48 joints
            96: 23,  # Root joint ID for 96 joints
            192:23,
            384:28,
            768:28,
            'coco_17': 0,
        }
    
    @staticmethod
    def get_pseudo_points_dict() -> Dict[int, List[int]]:
        """Get pseudo point dictionary, defines pseudo point positions for each level
        
        Returns:
            Dict[int, List[int]]: Keys are the number of joints per level, values are lists of pseudo point indices
        """
        from data.common_variables import smpl_pseudo_dict
        pseudo_points_dict = smpl_pseudo_dict.copy()
        return pseudo_points_dict
    
    @staticmethod
    def get_alignment_info() -> Optional[Tuple]:
        """Get alignment info (optional)
        
        Returns:
            Optional[Tuple]: Alignment info
        """
        _alignment_48_to_96 = tuple([(2*i, 2*i+1) for i in range(48)])
        alignment_info = (
            (
                (34, 4, 3),
                (4, 5, 34),
                (32, 20, 33),
                (15, 14, 32),
                (34, 3, 31),
                (13, 12, 22),
                (11, 10, 22),
                (47, 2, 42),
                (46, 27, 21),
                (44, 41, 45),
                (40, 45, 8),
                (30, 46, 21),
                (24, 25, 29),
                (29, 28, 19),
                (43, 27, 47),
                (17, 42, 16),
                (16, 37, 36)
            ),
            _alignment_48_to_96
        )
        return alignment_info

    @staticmethod
    def action_label_to_str(action_label:int):
        ACTION_NAMES = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
        'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
        'WalkTogether'
        ]
        return ACTION_NAMES[action_label]

    @staticmethod
    def merge_dataset(inp_path:str, out_path:str, var_save_dir:str, var_result_relative_path=["train_best/phats_denormed.pkl", "test_best/phats_denormed.pkl"], preserve_sparse_joints=True):
        print(f"Loading data from {inp_path}...")
        with open(inp_path, 'rb') as f:
            all_data = pickle.load(f)

        for idx, mode in enumerate(["train", "test"]):
            try:
                with open(os.path.join(var_save_dir, var_result_relative_path[idx]), 'rb') as f:
                    tmp_data:list[np.ndarray] = pickle.load(f) # [Tx17x2, Tx48x2, Tx96x2]
                    tmp_data.reverse()
                if len(all_data[mode]['coords']) == tmp_data[0].shape[0]:
                    for i in range(len(all_data[mode]['coords'])):
                        joints_17 = all_data[mode]['coords'][i]["2d"][-1]
                        all_data[mode]['coords'][i]["2d"] = [temp[i, :, :] for temp in tmp_data]
                        if preserve_sparse_joints:
                            all_data[mode]['coords'][i]["2d"][-1] = joints_17
                    print(f"Using Estimation Data for {mode} mode.")
                else:
                    raise ValueError(f"Invalid original_train_data length")
            except Exception as e:
                print(f"Failed to load VARSR result for {mode} mode: {e}")
        
        print(f"Saving data to {out_path}...")
        with open(out_path, 'wb') as f:
            pickle.dump(all_data, f)
        

if __name__ == "__main__":
    import time
    for mode in ["train", "warmup", "test"]:
        print("Starting to load dataset...")
        start_time = time.time()
        dataset = H36M_MSST_Dataset("/data/human3.6m/processedByYangtt/msst_data_h36m_vp3d_filtered.pkl", mode)
        end_time = time.time()
        import_time = end_time - start_time
        print(f"Dataset {mode} loading completed, time: {import_time:.2f} s")
        print(f"Dataset size: {len(dataset)}")

        print("Starting to iterate over data...")
        start_time = time.time()
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        for batch in dataloader:
            pass
        end_time = time.time()
        import_time = end_time - start_time
        print(f"Dataset {mode} iteration completed, time: {import_time:.2f} s")
        print(f"Number of batches: {len(dataloader)}")
