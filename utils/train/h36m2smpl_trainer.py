"""
H36M to SMPL conversion model trainer.
Inherits BaseTrainer, directly implements all abstract methods.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, List
import numpy as np
from thop import profile
import copy

from models.h36m_to_smpl import H36MToSMPLConverter
from utils.train.base_trainer import BaseTrainer
from utils.metrics import compute_mpjpe, compute_p_mpjpe
from data.h36m_to_smpl_dataset import H36MToSMPLDataset


class H36MToSMPLTrainer(BaseTrainer):

    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_parser_args(parser):
        parser = BaseTrainer.add_parser_args(parser)

        model_group = parser.add_argument_group('H36M2SMPL Model')
        model_group.add_argument('--embed_dim', type=int, default=256)
        model_group.add_argument('--num_heads', type=int, default=8)
        model_group.add_argument('--num_gcn_layers', type=int, default=3)
        model_group.add_argument('--num_transformer_layers', type=int, default=4)
        model_group.add_argument('--gcn_hidden_dim', type=int, default=128)
        model_group.add_argument('--dropout', type=float, default=0.1)
        return parser

    def make_dataset(self, mode: str) -> Dataset:
        return H36MToSMPLDataset(self.args.data_path, mode=mode)

    def make_model(self) -> nn.Module:
        return H36MToSMPLConverter(
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_heads,
            num_gcn_layers=self.args.num_gcn_layers,
            num_transformer_layers=self.args.num_transformer_layers,
            gcn_hidden_dim=self.args.gcn_hidden_dim,
            dropout=self.args.dropout,
        )
    
    def _get_model_inference_macs(self):
        self.model.eval()
        inp = torch.rand((1, 17, 3), device=self.device)
        with torch.no_grad():
            macs, _ = profile(copy.deepcopy(self.model), (inp,), verbose=False)
        return macs

    def make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)

    def make_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.num_epochs * len(self.train_loader),
            eta_min=1e-6
        )

    def forward_pass(self, model: nn.Module, data: Dict, mode: str) -> Dict:
        poses_3d = data['poses_3d']
        h36m_joints = poses_3d[17]

        if mode == "train":
            smpl_joints_gt = poses_3d[24]
            return model(h36m_joints, smpl_joints_gt)
        return model(h36m_joints)

    def eval_inference_out(self, inference_out: List[Dict]) -> Dict[str, float]:
        if not inference_out:
            return {'loss': float('inf')}

        predictions = []
        targets = []
        for batch_out in inference_out:
            pred = batch_out['outputs']['prediction'].cpu().numpy()*1000
            gt = batch_out['data']['poses_3d'][24].cpu().numpy()*1000
            predictions.append(pred)
            targets.append(gt)

        all_pred = np.concatenate(predictions, axis=0)
        all_gt = np.concatenate(targets, axis=0)

        metrics = {
            'mpjpe': float(np.mean(compute_mpjpe(all_pred, all_gt, root_id=0))),
            'p_mpjpe': float(np.mean(compute_p_mpjpe(all_pred, all_gt, root_id=0))),
        }

        loss_keys = [k for k in inference_out[0]['outputs'].keys() if k.endswith('loss')]
        for key in loss_keys:
            values = [batch['outputs'][key].item() for batch in inference_out if key in batch['outputs']]
            if values:
                metrics[key] = sum(values) / len(values)

        return metrics

    def save_prediction(self, inference_out: List[Dict]) -> None:
        if not inference_out:
            return

        predictions = []
        targets = []

        for result in inference_out:
            if 'outputs' not in result:
                continue

            out = result['outputs']
            data = result['data']
            normalized_root = data['normalized_root'][..., 2:].cpu().numpy()

            if 'prediction' in out:
                predictions.append(out['prediction'].cpu().numpy()+normalized_root)

            if 'poses_3d' in data and 24 in data['poses_3d']:
                targets.append(data['poses_3d'][24].cpu().numpy()+normalized_root)

        if predictions:
            predictions = np.concatenate(predictions, axis=0)
            save_dict = {'prediction': predictions}

            if targets:
                targets = np.concatenate(targets, axis=0)
                save_dict['gt'] = targets

            os.makedirs(os.path.dirname(self.args.save_prediction), exist_ok=True)
            np.savez(self.args.save_prediction, **save_dict)
            print(f"Predictions saved to {self.args.save_prediction}")
