"""
DensePose trainer base class and concrete implementations.
Supports training and inference for VQVAE, VAR, and Decoder.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from utils.metrics import compute_mpjpe, compute_p_mpjpe
from data.base_dataset import BaseDataset
from utils.train.base_trainer import BaseTrainer

class DensePoseBaseTrainer(BaseTrainer, ABC):
    """DensePose trainer base class, implements common data processing and training logic"""
    
    def __init__(self, args):
        super().__init__(args)
        # Common initialization logic can be added here
    
    # Abstract methods - subclasses must implement
    @abstractmethod
    def make_model(self) -> nn.Module:
        """Create model"""
        pass
        
    @staticmethod
    def add_parser_args(parser):
        """Add DensePose-related command line arguments"""
        parser = BaseTrainer.add_parser_args(parser)
        
        # Dataset-related parameter group
        dataset_group = parser.add_argument_group('Dataset Arguments')
        dataset_group.add_argument('--dataset_class', type=str, default='H36M_MSST', 
                          choices=['H36M_MSST', 'VAR_3DPW', 'H36M_VP'],
                          help='Dataset class to use for training')
        dataset_group.add_argument('--stride', type=int, default=1,
                          help='Stride for frame sampling (1 = use all frames, 2 = use every 2nd frame, etc.)')
        
        # Common model parameter group
        model_group = parser.add_argument_group('Model Arguments')
        model_group.add_argument('--feature_dim', type=int, default=2,
                          help='Feature dimension')
        model_group.add_argument('--gt_patch_nums', type=int, nargs=3, default=[17, 48, 96],
                          help='Ground truth patch numbers as three integers')
        model_group.add_argument('--tokens_per_joint', type=int, default=6,
                          help='Tokens per joint')
        model_group.add_argument('--expansion_strategy', type=str, default='balanced',
                          choices=['balanced', 'position_only'],
                          help="Token expansion strategy used during training ('balanced' or 'position_only'). MUST match the trained model.")
        
        # Optimizer-related parameter group
        optimizer_group = parser.add_argument_group('Optimizer Arguments')
        optimizer_group.add_argument('--weight_decay', type=float, default=1e-4,
                          help='Weight decay for optimizer')
        optimizer_group.add_argument('--warmup_steps', type=int, default=1000,
                          help='Number of warmup steps for learning rate scheduler')
        
        return parser
    
    def make_dataset(self, mode: str) -> Dataset:
        """
        Common implementation for creating a dataset.
        
        Args:
            mode: Data mode ('train', 'val', 'test', 'warmup')
            
        Returns:
            Dataset: Dataset instance for the corresponding mode
        """
        # 1. Dynamically import dataset class
        DATASET_CLASSES = {
            'H36M_MSST': 'data.h36m_msst_dataset.H36M_MSST_Dataset',
        }
        
        # 2. Select dataset class based on arguments
        dataset_path = DATASET_CLASSES[self.args.dataset_class]
        module_name, class_name = dataset_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[module_name.split('.')[0]])
        dataset_class = getattr(module, class_name)
        
        # 3. Create dataset instance
        dataset_kwargs = {'stride': self.args.stride}
        return dataset_class(self.args.data_path, mode=mode, **dataset_kwargs)
    
    def make_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer.
        
        Returns:
            torch.optim.Optimizer: Configured AdamW optimizer
        """
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.lr,
            weight_decay=getattr(self.args, 'weight_decay', 1e-4)
        )
    
    def make_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Create cosine annealing learning rate scheduler.
        
        Returns:
            torch.optim.lr_scheduler.LRScheduler: Configured scheduler
        """
        # 1. Compute total training steps
        total_steps = self.args.num_epochs * len(self.train_loader)
        
        # 2. Cosine annealing scheduler configuration
        warmup_steps = getattr(self.args, 'warmup_steps', 1000)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 1e-6 + 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def eval_inference_out(self, inference_out: List[Dict]) -> Dict[str, float]:
        """Evaluate inference results"""
        if not inference_out:
            return {'loss': float('inf')}
        
        metrics = {}
        gt_patch_nums = self.args.gt_patch_nums
        root_ids = self.train_loader.dataset.get_root_ids()
        
        # ========== Collect loss metrics ==========
        metrics.update(self._collect_loss_metrics(inference_out))
        
        # ========== Use helper method to collect and process data ==========
        collected = self._collect_inference_data(
            inference_out,
            output_keys=['multi_phats', 'pseudo_phats'],
            gt_patch_nums=gt_patch_nums,
            is_multi_scale=True
        )
        
        predictions = collected['predictions']
        targets = collected['targets']
        
        # ========== Compute coordinate metrics ==========
        for num_joints in gt_patch_nums:
            if num_joints in targets and 'multi_phats' in predictions and num_joints in predictions['multi_phats']:
                pred_denorm = predictions['multi_phats'][num_joints]
                target_denorm = targets[num_joints]
                
                # Compute MPJPE
                mpjpe = compute_mpjpe(pred_denorm, target_denorm, root_id=root_ids[num_joints])
                metrics[f'mpjpe_{num_joints}'] = float(np.mean(mpjpe))
                
                # Compute PA-MPJPE
                pa_mpjpe = compute_p_mpjpe(pred_denorm, target_denorm, root_id=root_ids[num_joints])
                metrics[f'pa_mpjpe_{num_joints}'] = float(np.mean(pa_mpjpe))
                
                # Compute metrics for pseudo predictions
                if 'pseudo_phats' in predictions and num_joints in predictions['pseudo_phats']:
                    pseudo_pred_denorm = predictions['pseudo_phats'][num_joints]
                    
                    pseudo_mpjpe = compute_mpjpe(pseudo_pred_denorm, target_denorm, root_id=root_ids[num_joints])
                    metrics[f'pseudo_mpjpe_{num_joints}'] = float(np.mean(pseudo_mpjpe))
                    
                    pseudo_pa_mpjpe = compute_p_mpjpe(pseudo_pred_denorm, target_denorm, root_id=root_ids[num_joints])
                    metrics[f'pseudo_pa_mpjpe_{num_joints}'] = float(np.mean(pseudo_pa_mpjpe))
        
        # ========== Compute mean MPJPE ==========
        mpjpe_values = [metrics[f'mpjpe_{num}'] for num in gt_patch_nums if f'mpjpe_{num}' in metrics]
        if mpjpe_values:
            metrics['mpjpe'] = float(np.mean(mpjpe_values))
        
        return metrics
    
    def save_prediction(self, inference_out: List[Dict]) -> None:
        """Save prediction results to npz file, supports cases without GT data"""
        if not inference_out:
            return
        
        gt_patch_nums = self.args.gt_patch_nums
        
        # Use helper method to collect and process data
        collected = self._collect_inference_data(
            inference_out,
            output_keys=['multi_phats', 'pseudo_phats'],
            gt_patch_nums=gt_patch_nums,
            is_multi_scale=True
        )
        
        predictions = collected['predictions']
        targets = collected['targets']
        
        # Prepare dictionary to save - only save predictions
        save_dict = {
            'prediction': predictions['multi_phats']
        }
        
        # Only save gt when GT data is available
        if targets:
            save_dict['gt'] = targets
        
        # Add pseudo predictions (if exist)
        if 'pseudo_phats' in predictions and predictions['pseudo_phats']:
            save_dict['pseudo'] = predictions['pseudo_phats']
        
        # Save to npz file
        np.savez(self.args.save_prediction, **save_dict)
        print(f"Predictions saved to {self.args.save_prediction}")

    def _collect_loss_metrics(self, inference_out: List[Dict]) -> Dict[str, float]:
        """
        Common method to collect loss metrics from inference_out.
        
        Args:
            inference_out: Inference output list
            
        Returns:
            Dictionary containing all loss metrics
        """
        loss_data = {}
        for batch_item in inference_out:
            outputs = batch_item['outputs']
            for key, value in outputs.items():
                if key.endswith('loss'):
                    if key not in loss_data:
                        loss_data[key] = []
                    loss_data[key].append(value)
        
        metrics = {}
        for key, tensors in loss_data.items():
            if tensors:
                metrics[key] = torch.stack(tensors).mean().item()
        
        return metrics
    
    def _collect_inference_data(
        self,
        inference_out: List[Dict],
        output_keys: List[str],
        gt_patch_nums: List[int],
        is_multi_scale: bool = False,
        skip_clear_root: bool = False
    ) -> Dict:
        """
        Common method to collect data from inference_out and perform denormalization, supporting inference scenarios without GT data.
        
        Args:
            inference_out: Inference output list
            output_keys: List of output keys to extract, e.g. ['vae_phat', 'var_phat'] or ['multi_phats', 'pseudo_phats']
            gt_patch_nums: List of GT patch numbers
            is_multi_scale: Whether the data is multi-scale
            skip_clear_root: Whether to skip clear_root_and_pseudo (should be True for skeletons like COCO that have no standard root definition)
            
        Returns:
            Dictionary of denormalized data, in the format:
            {
                'predictions': {key: {num: ndarray, ...}, ...},
                'targets': {num: ndarray, ...}
            }
        """
        dataset: BaseDataset = self.train_loader.dataset
        
        # Initialize data containers
        predictions = {key: {num: [] for num in gt_patch_nums} for key in output_keys}
        targets = {num: [] for num in gt_patch_nums}
        bbox = {num: [] for num in gt_patch_nums}
        normalized_root = {num: [] for num in gt_patch_nums}
        
        # Collect data
        for batch_item in inference_out:
            outputs = batch_item['outputs']
            data = batch_item['data']
            
            # Collect bbox and normalized_root (shared across all scales)
            batch_bbox = data.get('bbox', None)
            batch_root = data.get('normalized_root', None)
            poses_2d = data.get('poses_2d', {})
            
            for num_joints in gt_patch_nums:
                if batch_bbox is not None:
                    bbox[num_joints].append(batch_bbox)
                if batch_root is not None:
                    normalized_root[num_joints].append(batch_root)
                if num_joints in poses_2d:
                    targets[num_joints].append(poses_2d[num_joints])
            if is_multi_scale:
                # Multi-scale: multi_phats[i] corresponds to gt_patch_nums[i]
                for output_key in output_keys:
                    output_data = outputs.get(output_key, [])
                    for i, num_joints in enumerate(gt_patch_nums):
                        if i < len(output_data) and output_data[i] is not None:
                            predictions[output_key][num_joints].append(output_data[i])
            else:
                # Single-scale: directly corresponds to the first gt_patch_num
                target_num = gt_patch_nums[0]
                for output_key in output_keys:
                    output_data = outputs.get(output_key, None)
                    if output_data is not None:
                        predictions[output_key][target_num].append(output_data)
        
        # Process and return denormalized data
        processed_predictions = {key: {} for key in output_keys}
        processed_targets = {}
        
        for num_joints in gt_patch_nums:
            # Check if there are predictions or targets
            has_predictions = any(
                len(predictions[key][num_joints]) > 0 
                for key in output_keys
            )
            has_targets = len(targets[num_joints]) > 0
            
            if not has_predictions and not has_targets:
                continue
            bbox_all = torch.cat(bbox[num_joints], dim=0)
            root_all = torch.cat(normalized_root[num_joints], dim=0)
            # Denormalize predictions (if any)
            if has_predictions:
                for output_key in output_keys:
                    if len(predictions[output_key][num_joints]) > 0:
                        pred_all = torch.cat(predictions[output_key][num_joints], dim=0)
                        if not skip_clear_root:
                            pred_all = dataset.clear_root_and_pseudo(pred_all)
                        pred_denorm = dataset.denormalize_pose(
                            pred_all, root_all, bbox_all
                        ).numpy()
                        processed_predictions[output_key][num_joints] = pred_denorm
            
            # Denormalize targets (if any)
            if has_targets:
                target_all = torch.cat(targets[num_joints], dim=0)
                
                target_denorm = dataset.denormalize_pose(
                    target_all, root_all, bbox_all
                ).numpy()
                processed_targets[num_joints] = target_denorm
        
        return {
            'predictions': processed_predictions,
            'targets': processed_targets
        }

