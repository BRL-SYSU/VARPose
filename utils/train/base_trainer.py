"""
Base Trainer class for unified training and evaluation of different models.
Supports DDP, mixed precision, checkpointing, and other common training features.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import shutil
import argparse
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import time
from thop import clever_format, profile

from utils.ddp_utils import setup_ddp, cleanup_ddp, is_main_process, gather_distributed_results, is_dist_initialized, barrier, recursive_to_device
from utils.setup_seed import setup_seed
from utils.simple_logger import Logger

class BaseTrainer(ABC):
    """
    Base trainer class that provides common training functionality.
    
    Subclasses need to implement:
    - make_dataset(): Create datasets and dataloaders
    - make_model(): Create the model
    - eval(): Evaluate the model
    """
    
    def __init__(self, args):
        """
        Initialize trainer with arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.device = None
        self.model:nn.Module = None
        self.optimizer:torch.optim.Optimizer = None
        self.scheduler:torch.optim.lr_scheduler.LRScheduler = None
        self.scaler = None
        self.writer = None
        self.stdout_logger = None
        self.log_file_path = None
        
        # Training state
        self.global_step = 0
        self.best_metric = float('inf')  # For model selection (lower is better)
        self.patience_counter = 0
        self.start_time_str: str = time.strftime('%Y%m%d_%H%M%S')
        
        # DDP state
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        # dataloader
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.warmup_loader: DataLoader = None
    
    @staticmethod
    def add_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add common training arguments to the argument parser.
        
        This method adds standard arguments that are commonly used across
        different training scripts. Subclasses can call this method and
        then add their own specific arguments.
        
        Args:
            parser: ArgumentParser object to add arguments to
            
        Returns:
            The same ArgumentParser object with added arguments
        """
        # Basic training arguments
        train_group = parser.add_argument_group('Training Configuration')
        train_group.add_argument('--batch_size', type=int, default=64, 
                                help='Batch size for training and validation')
        train_group.add_argument('--num_workers', type=int, default=4, 
                                 help='Number of workers for data loading')
        train_group.add_argument('--num_epochs', type=int, default=100, 
                                help='Number of training epochs')
        train_group.add_argument('--num_warmup_epochs', type=int, default=0, 
                                help='Number of warmup epochs (optional)')
        train_group.add_argument('--lr', type=float, default=1e-3, 
                                help='Initial learning rate')
        train_group.add_argument('--patience', type=int, default=100, 
                                help='Early stopping patience')
        train_group.add_argument('--seed', type=int, default=-1, 
                                help='Random seed. Negative values use random seed strategy')
        train_group.add_argument('--resume', action='store_true', 
                                 help='Resume training from checkpoint setting optimizer')
        train_group.add_argument('--train', action='store_true', help='Enable training')
        train_group.add_argument('--test', action='store_true', help='Enable testing')
        train_group.add_argument('--test_all', action='store_true', help='Test all datasets')
        train_group.add_argument("--val_main_metric", type=str, default="mpjpe", 
                                 help="The metric to use for selecting the best model and smaller better")
        train_group.add_argument("--save_prediction", type=str, default="", help='Save prediction to path when testing')
        
        
        # Path arguments
        path_group = parser.add_argument_group('Path Configuration')
        path_group.add_argument('--data_path', type=str, required=True,
                               help='Path to the dataset')
        path_group.add_argument('--save_dir', type=str, required=True,
                               help='Directory to save model checkpoints')
        path_group.add_argument('--log_dir', type=str, required=True,
                               help='Directory to save training logs')
        path_group.add_argument('--checkpoint_path', type=str, default=None,
                               help='Path to checkpoint')
        
        # DDP and distributed training arguments
        ddp_group = parser.add_argument_group('Distributed Training')
        ddp_group.add_argument('--distributed', action='store_true',
                               help='Enable distributed training (DDP)')
        
        # Mixed precision training
        precision_group = parser.add_argument_group('Mixed Precision')
        precision_group.add_argument('--use_amp', action='store_true',
                                     help='Enable automatic mixed precision training')
        
        # Saving and logging arguments
        save_group = parser.add_argument_group('Saving and Logging')
        save_group.add_argument('--save_per_epoch', action='store_true',
                               help='Save model checkpoint after each epoch')
        
        # Scheduler configuration
        scheduler_group = parser.add_argument_group('Scheduler Configuration')
        scheduler_group.add_argument('--scheduler_step_mode', type=str, default='batch',
                                     choices=['batch', 'epoch'],
                                     help='When to step scheduler: after each batch or epoch')
        
        return parser
    
    def run(self):
        if (self.args.train and self.args.test) or (not self.args.train and not self.args.test):
            raise ValueError(f"Invalid train test arguments: {self.args.train} {self.args.test}")
        
        self.setup_ddp_and_device()
        self.setup_logging()
        self.setup_seed()
        self._save_config()
        start_time = time.time()
        self.setup_dataloader()
        end_time = time.time()
        if is_main_process():
            print(f"import dataset time: {end_time - start_time:.2f}s")
        self.setup_mixed_precision()

        self.model = self.make_model().to(self.device)
        self._print_model_info()
        self.setup_model_ddp()
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        if self.args.checkpoint_path is not None:
            start_epoch = self.load_checkpoint(self.args.checkpoint_path)
        else:
            start_epoch = 0

        if self.args.train:
            final_epoch = self.train(start_epoch=start_epoch)
            self._save_train_summary(final_epoch=final_epoch)
        else:
            self.eval()
        cleanup_ddp()
        
    @abstractmethod
    def make_dataset(self, mode:str) -> Dataset:
        """
        Create datasets and dataloaders.
        Args:
            mode: Mode of the dataset
        Returns:
            Dataset
        """
        pass
    
    @abstractmethod
    def make_model(self) -> nn.Module:
        """
        Create and return the model.
        
        Returns:
            The model to be trained
        """
        pass

    @abstractmethod
    def make_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer.
        
        Returns:
            optimizer
        """
        pass

    @abstractmethod
    def make_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Create learning rate scheduler.
        
        Returns:
            scheduler
        """
        pass
    
    @abstractmethod
    def eval_inference_out(self, inference_out: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the model on given inference_out.
        
        Args:
            inference_out: Inference output
            
        Returns:
            Dictionary of evaluation metrics whose keys must have "loss".
        """
        pass

    @abstractmethod
    def save_prediction(self, inference_out: List[Dict]) -> None:
        """
        Saving prediction when testing.
        
        Args:
            inference_out: Inference output

        Returns:
            None
        """
        pass
    
    def _get_mode_suffix_path(self, base_path: str, mode: str) -> str:
        """
        Add mode suffix to file path while preserving directory structure.
        
        Args:
            base_path: Original path from args.save_prediction
            mode: Dataset mode ("train", "val", "test")
            
        Returns:
            Path with mode suffix added to filename
            
        Examples:
            _get_mode_suffix_path("output/predictions.npz", "train")
            -> "output/predictions_train.npz"
            
            _get_mode_suffix_path("results/eval/predictions.npz", "val")
            -> "results/eval/predictions_val.npz"
        """
        directory = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{mode}{ext}"
        
        if directory:
            return os.path.join(directory, new_filename)
        return new_filename
    
    def make_loader(self, dataset:Dataset, mode: str)->DataLoader:
        """
        Create DataLoader for the given dataset.
        
        Args:
            dataset: Dataset to create dataloader for
            mode: Mode of the dataloader ("train", "val", "test", "warmup")
            
        Returns:
            Configured DataLoader
        """
        sampler = None
        shuffle = False
        if is_dist_initialized():
            sampler = DistributedSampler(
                dataset, 
                shuffle=(mode == "train")
            )
            shuffle = False  # Don't use shuffle with DistributedSampler
        else:
            shuffle = (mode == "train")
        dataloader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.num_workers, 
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        return dataloader
    
    def forward_pass(self, model: nn.Module, data: Any, mode:str) -> Any:
        """
        Forward pass - to be implemented by subclasses for specific model logic.
        
        Args:
            model: The model
            data: Input data
            mode: "train" or "inference"

        Returns:
            Model outputs
        """
        # Default implementation - assume data can be passed directly to model
        return model(data)

    def setup_ddp_and_device(self) -> None:
        """Setup DDP environment and device."""
        if self.args.distributed:
            try:
                self.local_rank, self.global_rank, self.world_size = setup_ddp()
                print(f"Distributed training initialized: rank={self.global_rank}/{self.world_size}")
            except Exception as e:
                print(f"Failed to initialize distributed training: {e}")
                print("Falling back to single GPU training")
                self.args.distributed = False
                self.local_rank, self.global_rank, self.world_size = 0, 0, 1
        else:
            self.local_rank, self.global_rank, self.world_size = 0, 0, 1
            print("Running in single GPU mode")
        
        # Set device
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    def setup_seed(self) -> None:
        """Setup random seed for reproducibility."""
        if is_main_process():
            final_seed = setup_seed(self.args.seed)
            seed_tensor = torch.tensor(final_seed, device=self.device, dtype=torch.long)
        else:
            seed_tensor = torch.empty(1, device=self.device, dtype=torch.long)
        
        if is_dist_initialized():
            dist.broadcast(seed_tensor, src=0)
            barrier()
        
        final_seed = seed_tensor.item()
        setup_seed(final_seed)
        
        if is_main_process():
            print(f"Using synced seed: {final_seed} across all processes.")
    
    def setup_mixed_precision(self) -> None:
        """Setup mixed precision training."""
        if self.args.use_amp:
            self.scaler = GradScaler("cuda")
            print("Using mixed precision training")
        else:
            self.scaler = None  # Don't use mixed precision
            print("Using full precision training")
    
    def setup_logging(self) -> None:
        """Setup TensorBoard logging and directories."""
        if is_main_process():
            # Create directories
            # if not self.args.resume and not self.args.test:
            #     if os.path.exists(self.args.log_dir):
            #         shutil.rmtree(self.args.log_dir)
            #     if os.path.exists(self.args.save_dir):
            #         shutil.rmtree(self.args.save_dir)
            
            os.makedirs(self.args.log_dir, exist_ok=True)
            os.makedirs(self.args.save_dir, exist_ok=True)
            
            # Setup TensorBoard
            if self.args.train:
                self.writer = SummaryWriter(log_dir=self.args.log_dir)
            else:
                test_writer_path = os.path.join(self.args.log_dir, f'test_tensorboard_{self.start_time_str}')
                os.makedirs(test_writer_path, exist_ok=True)
                self.writer = SummaryWriter(log_dir=test_writer_path)
            
            # Setup file logging
            time_str = self.start_time_str
            if self.args.test:
                # In test mode, write to test_summary_... with a timestamp to prevent multiple tests from overwriting each other
                log_file_name = f'test_log_{time_str}.txt'
            else:
                # In training mode, keep the original logic
                log_file_name = f'train_log.txt'

            self.log_file_path = os.path.join(self.args.log_dir, log_file_name)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            self.stdout_logger = Logger(self.log_file_path, original_stdout)
            sys.stdout = self.stdout_logger
            sys.stderr = self.stdout_logger
            
            print(f"Logging to {self.args.log_dir}")
            print(f"Checkpoints will be saved to {self.args.save_dir}")
            print(f"Current log file: {log_file_name}")
    
    def setup_model_ddp(self) -> None:
        """Wrap model with DDP if distributed training."""
        if is_dist_initialized():
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank], 
                output_device=self.local_rank,
                find_unused_parameters=True
            )
    
    def setup_dataloader(self) -> None:
        train_dataset = self.make_dataset(mode="train")
        if self.args.num_warmup_epochs > 0:
            warmup_dataset = self.make_dataset(mode="warmup")
        else:
            warmup_dataset = None
        val_dataset, test_dataset = self.make_dataset(mode="val"), self.make_dataset(mode="test")

        self.train_loader = self.make_loader(train_dataset, "train")
        if warmup_dataset is not None:
            self.warmup_loader = self.make_loader(warmup_dataset, "warmup")
        else:
            self.warmup_loader = None
        self.val_loader, self.test_loader = self.make_loader(val_dataset, "val"), self.make_loader(test_dataset, "test")

    def log_epoch_steps(self, step_metrics_list: List[Dict[str, float]]) -> None:
        """
        Log all step metrics from an epoch to TensorBoard in batch.
        
        This method is called after epoch completion to avoid IO overhead
        during training loop.
        
        Args:
            step_metrics_list: List of step metrics dictionaries from an epoch
        """
        if not is_main_process() or not self.writer:
            return
        
        for step_metrics in step_metrics_list:
            global_step = step_metrics.get('global_step', 0)
            
            # Log every 100 steps as before, but batch the IO operations
            if global_step % 100 == 0:
                for key, value in step_metrics.items():
                    if key.endswith('loss'):
                        self.writer.add_scalar(f'Step_loss/train_{key}', value, global_step)
                    elif key == 'LR':
                        self.writer.add_scalar('Step_Learning_Rate', value, global_step)
    
    def get_train_metrics(self, step_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Compute epoch-level average metrics from step-level metrics.
        
        Args:
            step_metrics_list: List of step metrics dictionaries
            
        Returns:
            Dictionary with averaged metrics containing:
            - All keys ending with 'loss' (averaged)
            - 'LR': Learning rate from the last step
        """
        if not step_metrics_list:
            return {'loss': 0.0, 'LR': 0.0}
        
        metrics = {}
        loss_keys = [k for k in step_metrics_list[0].keys() if k.endswith('loss')]
        
        for key in loss_keys:
            total = sum(step.get(key, 0.0) for step in step_metrics_list)
            metrics[key] = total / len(step_metrics_list)
        
        # Use LR from the last step
        metrics['LR'] = step_metrics_list[-1].get('LR', 0.0)
        
        return metrics
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> List[Dict[str, float]]:
        """
        Train for one epoch and collect step-level metrics.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            
        Returns:
            List of step metrics, each containing:
            - All values with keys ending in 'loss'
            - 'LR': Current learning rate
            - 'global_step': Global step number
        """
        self.model.train()
        
        step_metrics_list = []
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}', ascii=True) if is_main_process() else dataloader
        
        for batch_idx, data in enumerate(progress_bar):
            self.optimizer.zero_grad()

            # Move data to device recursively
            data = recursive_to_device(data, self.device)
            
            # Forward pass with optional mixed precision
            if self.scaler:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.forward_pass(self.model, data, "train")
                    loss = outputs['loss']
            else:
                outputs = self.forward_pass(self.model, data, "train")
                loss = outputs['loss']
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            if self.scheduler and self.args.scheduler_step_mode == 'batch':
                self.scheduler.step()
            
            # Collect step metrics: extract all losses
            step_metrics = {}
            for key, value in outputs.items():
                if key.endswith('loss') and hasattr(value, 'item'):
                    step_metrics[key] = value.item()
            
            step_metrics['LR'] = self.optimizer.param_groups[0]['lr']
            step_metrics['global_step'] = self.global_step
            
            step_metrics_list.append(step_metrics)
            
            # Update progress bar with primary loss and LR (no IO operation)
            if is_main_process() and self.global_step % 100 == 0:
                primary_loss = step_metrics.get('loss', step_metrics.get('loss', 0.0))
                progress_bar.set_postfix({
                    'Loss': f'{primary_loss:.6f}',
                    'LR': f'{step_metrics["LR"]:.6f}'
                })

            self.global_step += 1
        
        return step_metrics_list
    
    def inference(self, dataloader: DataLoader) -> List[Any]:
        """
        Run inference on the given dataloader.
        
        In distributed training mode, this method will gather results from all processes
        and return the complete result list only on the main process. Non-main processes
        will return an empty list.
        
        Args:
            dataloader: Dataloader for inference
            
        Returns:
            List of model outputs for each batch (complete list only on main process)
        """
        self.model.eval()
        local_results = []

        progress_bar = tqdm(dataloader, desc='Inference', ascii=True) if is_main_process() else dataloader
        
        with torch.no_grad():
            for batch_idx, data in enumerate(progress_bar):
                # Move data to device recursively
                data = recursive_to_device(data, self.device)
                
                # Forward pass with optional mixed precision
                if self.scaler:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.forward_pass(self.model, data, "inference")
                else:
                    outputs = self.forward_pass(self.model, data, "inference")
                
                # Save outputs and original data for metrics computation
                # Move all tensors to CPU before gather to avoid cross-GPU memory allocation
                local_results.append({
                    'outputs': recursive_to_device(outputs, "cpu"),
                    'data': recursive_to_device(data, "cpu")
                })
                
                if is_main_process():
                    progress_bar.set_postfix({'Batch': batch_idx + 1})
        
        # Gather results from all processes if in distributed mode
        if is_dist_initialized():
            # Synchronize all processes before gathering
            barrier()
            combined_results = gather_distributed_results(local_results)
            
            if is_main_process():
                print(f"Inference completed. Gathered {len(local_results)} local batches from this process "
                      f"and combined with results from {dist.get_world_size()-1} other processes. "
                      f"Total batches: {len(combined_results)}")
            
            return combined_results
        else:
            return local_results
        
    def eval_dataloader(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on given dataloader.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Dictionary of evaluation metrics
        """
        inference_results = self.inference(dataloader)
        if is_main_process():
            if self.args.test and self.args.save_prediction:
                self.save_prediction(inference_results)
                return {}
            metrics = self.eval_inference_out(inference_results)
        else:
            metrics = {}
        if is_dist_initialized():
            dist.broadcast_object_list([metrics], src=0)
        return metrics
    
    def train(self, start_epoch: int = 0) -> int:
        """
        Main training loop.
        
        Args:
            start_epoch: Starting epoch number
            
        Returns:
            Final epoch number completed
        """
        print("Starting training...")
        final_epoch = start_epoch
        
        # Training loop
        for epoch in range(start_epoch, self.args.num_epochs):
            final_epoch = epoch
            
            # Use warmup loader for early epochs
            if ( 
                epoch < self.args.num_warmup_epochs and 
                self.warmup_loader is not None):
                current_loader = self.warmup_loader
                if is_main_process():
                    print(f"Warmup epoch {epoch+1}/{self.args.num_warmup_epochs}")
            else:
                current_loader = self.train_loader
            
            # Set epoch for distributed sampling
            if is_dist_initialized() and hasattr(self.train_loader, 'sampler'):
                if hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)
                if hasattr(self.val_loader, 'sampler') and hasattr(self.val_loader.sampler, 'set_epoch'):
                    self.val_loader.sampler.set_epoch(epoch)
            
            # Train epoch
            step_metrics_list = self.train_epoch(current_loader, epoch)
            self.log_epoch_steps(step_metrics_list)
            train_metrics = self.get_train_metrics(step_metrics_list)
            
            # Validate
            val_metrics:Dict[str, float] = self.eval_dataloader(self.val_loader)
            
            # Step scheduler after epoch if configured
            if self.scheduler and self.args.scheduler_step_mode == 'epoch':
                self.scheduler.step()
            
            self.print_progress(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            main_metric = val_metrics.get(self.args.val_main_metric, float('inf'))  # Default to loss
            is_best = main_metric < self.best_metric
            self.save_checkpoint(epoch, main_metric, is_best)
            
            # Check early stopping
            if self.should_early_stop(main_metric):
                break
        
        return final_epoch
    
    def eval(self):
        original_save_path = self.args.save_prediction
        
        if self.args.save_prediction:
            self.args.save_prediction = self._get_mode_suffix_path(original_save_path, "test")
        metrics = self.eval_dataloader(self.test_loader)
        self.print_metrics(metrics, "test")
        self._save_test_summary(metrics, "test")
        
        if self.args.test_all:
            for loader, mode in zip([self.train_loader, self.val_loader], ["train", "val"]):
                if self.args.save_prediction:
                    self.args.save_prediction = self._get_mode_suffix_path(original_save_path, mode)
                metrics = self.eval_dataloader(loader)
                self.print_metrics(metrics, mode)
                self._save_test_summary(metrics, mode)
        
        self.args.save_prediction = original_save_path
    
    def save_checkpoint(self, epoch: int, metric:float,  is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        if not is_main_process():
            return
        
        # Get state dict
        state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') and is_dist_initialized() else self.model.state_dict()
        out = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step, 
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter
        }
        
        # Save regular checkpoint
        if self.args.save_per_epoch:
            checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch+1}_metric_{metric:.2f}.pth')
            torch.save(out, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'best_model.pth')
            torch.save(out, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[int]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Starting epoch if available, None otherwise
        """
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
            
            if checkpoint.get('state_dict') is None:
                self.model.load_state_dict(checkpoint, strict=False)
                start_epoch = 0
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                if self.args.resume:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    start_epoch = checkpoint['epoch']
                    self.global_step = checkpoint.get('global_step', 0)
                    self.best_metric = checkpoint.get('best_metric', float('inf'))
                    self.patience_counter = checkpoint.get('patience_counter', 0)
                    print(f"Resuming training from epoch {start_epoch}, global step {self.global_step}, Best Metric: {self.best_metric}")
                else:
                    start_epoch = 0
            
            print(f"Checkpoint loaded successfully")
            return start_epoch
        else:
            print(f"No checkpoint found at: {checkpoint_path}")
            return None
        
    def print_progress(self, epoch: int, train_metrics: dict, val_metrics: dict):
        if is_main_process():
            print(f'Epoch [{epoch+1}/{self.args.num_epochs}]')
            for key, value in train_metrics.items():
                print(f'Train {key}: {value:.6f}')
            for key, value in val_metrics.items():
                print(f'Val {key}: {value:.6f}')
            
            # Log to TensorBoard
            if self.writer:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'Epoch_Train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Epoch_Val/{key}', value, epoch)
    def print_metrics(self, metrics: dict, labels:str):
        if is_main_process():
            print("\n" + "="*50)
            print(f"{labels.upper()} RESULTS")
            print("="*50)

            for key, value in metrics.items():
                print(f"{key}: {value:.6f}")
            if self.writer:
                for key, value in metrics.items():
                    self.writer.add_text(f"{labels}/{key}", f"{value:.6f}", 0)
            print("="*50 + "\n")
    
    def should_early_stop(self, current_metric: float) -> bool:
        """
        Check if training should stop early based on patience.
        
        Args:
            current_metric: Current validation metric
            
        Returns:
            True if should stop early
        """
        patience = self.args.patience
        
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                print(f'Early stopping at epoch (patience: {self.patience_counter}/{patience})')
                return True
        return False
       
    def _print_model_info(self) -> None:
        """Print model parameter information."""
        if is_main_process():
            model = self.model.module if hasattr(self.model, 'module') and is_dist_initialized() else self.model
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            
            total_params_str, trainable_params_str, non_trainable_params_str = clever_format(
                [total_params, trainable_params, non_trainable_params], "%.2f"
            )
            
            print(f"Model parameter statistics:")
            print(f"Total parameters: {total_params_str}")
            print(f"Trainable parameters: {trainable_params_str}")
            print(f"Non-trainable parameters: {non_trainable_params_str}")

            macs = self._get_model_inference_macs()
            if macs is not None:
                macs_str, _ = clever_format([macs, macs], "%.2f")
                print(f"Inference MACs: {macs_str}")
        if is_dist_initialized():
            dist.barrier()
    
    def _get_model_inference_macs(self) -> None|int:
        return None
    
    def _save_config(self) -> None:
        """Save training configuration or log test configuration."""
        if is_main_process():
            time_str = self.start_time_str
            config_path = os.path.join(self.args.log_dir, 'train_config.txt') if self.args.train else os.path.join(self.args.log_dir, 'test_config.txt')
            mode = 'w' if self.args.train else 'a'
            with open(config_path, mode) as f:
                f.write(f"Configuration ({time_str})\n")
                f.write(f"{'=' * 50}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}\n")
                f.write(f"Python Path: {sys.executable}\n")
                f.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}\n")
                f.write(f"{'=' * 50}\n")
                for arg, value in vars(self.args).items():
                    f.write(f"--{arg} {value} \\ \n")
                f.write(f"{'=' * 100}\n")
    
    def _save_train_summary(self, final_epoch: int) -> None:
        """Save train summary."""
        if is_main_process():
            summary_path = os.path.join(self.args.log_dir, 'train_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("Training Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total epochs: {final_epoch + 1}\n")
                f.write(f"Best metric: {self.best_metric:.6f}\n")
                f.write(f"Total steps: {self.global_step}\n")
                f.write(f"Mixed precision: {'Enabled' if self.scaler is not None else 'Disabled'}\n")
                if self.optimizer:
                    f.write(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}\n")
    
    def _save_test_summary(self, metrics: dict, labels:str) -> None:
        """Save test summary."""
        if is_main_process():
            summary_path = os.path.join(self.args.log_dir, 'test_summary.txt')
            with open(summary_path, 'a') as f:
                f.write(f"start time: {self.start_time_str}\n")
                f.write(f"{labels.upper()} RESULTS\n")
                f.write(f"{'='*50}\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.6f}\n")
                f.write(f"{'='*50}\n")