"""
H36M 17-point -> SMPL 24-point conversion dataset

Data pkl format:
{
    'train': [{'joint_17_3d': (17,3), 'joint_24_3d': (24,3), 'metadata': dict}, ...],
    'test': [...]
}
Units: mm, root joint not aligned

Outputs standard format, root joint aligned, units in m
"""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import Dict, Any


class H36MToSMPLDataset(Dataset):

    def __init__(self, data_path: str, mode: str = 'train'):
        self.data_path = data_path
        self.mode = mode
        self.samples = []
        self._load_data()

    def _load_data(self):
        with open(self.data_path, 'rb') as f:
            all_data = pickle.load(f)
        mode_key = 'train' if self.mode in ('train', 'warmup') else 'test'
        self.samples = all_data[mode_key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        j17 = torch.from_numpy(np.asarray(sample['joint_17_3d'], dtype=np.float32))  # (17, 3) mm
        j24 = torch.from_numpy(np.asarray(sample['joint_24_3d'], dtype=np.float32))  # (24, 3) mm
        normalize_root = torch.zeros((1,5), dtype=torch.float32)
        normalize_root[:, 2:] = j24[0:1].clone()
        # Root joint alignment
        j17 = j17 - j17[0:1]
        j24 = j24 - j24[0:1]

        # mm -> m
        j17 = j17 / 1000.0
        j24 = j24 / 1000.0

        return {
            'poses_2d': {},
            'poses_3d': {17: j17, 24: j24},
            'action_label': torch.zeros(1, dtype=torch.long),
            'bbox': torch.tensor([[0, 0, 1000, 1000]], dtype=torch.float32),
            'normalized_root': normalize_root,
            'campose_valid': torch.ones(1, dtype=torch.float32),
        }
