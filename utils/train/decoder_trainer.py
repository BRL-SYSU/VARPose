import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np
from thop import profile
import copy

from data.base_dataset import BaseDataset
from models.var_skeleton import VARForSkeleton
from models.vqvae_hierarchical_skeleton import HierarchicalVQVAE, HierarchicalDecoder
from utils.train.dense_pose_base_trainer import DensePoseBaseTrainer
from utils.metrics import compute_mpjpe, compute_p_mpjpe

class DensePoseDecoderTrainer(DensePoseBaseTrainer):
    """Decoder model trainer (independently trains the decoder)"""
    
    def _gt_patch_data_key(self):
        """Parse the --gt_patch argument into the dataset dictionary key.
        "192" -> 192 (int), "coco_17" -> "coco_17" (str)
        """
        try:
            return int(self.args.gt_patch)
        except ValueError:
            return self.args.gt_patch
    
    def _gt_patch_int(self):
        """Parse the --gt_patch argument into an integer (joint count) for model construction and arithmetic.
        "192" -> 192, "coco_17" -> 17
        """
        try:
            return int(self.args.gt_patch)
        except ValueError:
            return int(self.args.gt_patch.split('_')[-1])
    
    @staticmethod
    def add_parser_args(parser):
        """Add Decoder-specific command line arguments"""
        parser = DensePoseBaseTrainer.add_parser_args(parser)

        # Decoder parameter group
        decoder_group = parser.add_argument_group('Decoder Configuration')
        decoder_group.add_argument("--gt_patch", type=str, default='192', 
                                 help='GT patch number (int like "192" or string key like "coco_17")')
        decoder_group.add_argument("--eval_root_id", type=int, default=None,
                                 help='Override root joint id for decoder MPJPE evaluation.')
        
        # VQVAE model parameter group
        vqvae_group = parser.add_argument_group('VQVAE Configuration')
        vqvae_group.add_argument('--vqvae_model_path', type=str, required=True,
                               help='Path to pretrained VQVAE model')
        vqvae_group.add_argument('--vqvae_model', type=str, default='HierarchicalVQVAE',
                               choices=['HierarchicalVQVAE'],
                               help='VQVAE model name for VAR')
        vqvae_group.add_argument('--vocab_size', type=int, default=4096,
                               help='The size of codebook')
        vqvae_group.add_argument('--embedding_dim', type=int, default=128,
                               help='The size of codebook embedding dimension')
        vqvae_group.add_argument('--beta', type=float, default=0.25,
                               help='Beta value for VQVAE')
        vqvae_group.add_argument('--using_znorm', type=bool, default=False,
                               help='Whether to use z-normalization')
        vqvae_group.add_argument('--quant_resi', type=float, default=0.5,
                               help='Quantization residual')
        vqvae_group.add_argument('--v_patch_nums', type=int, nargs='+', 
                               default=[48, 102, 192, 288, 432, 576],
                               help='V patch numbers')
        vqvae_group.add_argument('--use_ema', type=bool, default=True,
                               help='Whether to use exponential moving average for the nearest neighbors')
        vqvae_group.add_argument('--use_reset', type=bool, default=False,
                               help='Whether to use reset for the nearest neighbors')
        vqvae_group.add_argument('--add_codebook_loss', type=bool, default=True,
                               help='Whether to add codebook loss')
        vqvae_group.add_argument('--recon_loss_weight', type=float, default=200.0,
                               help='Weight for reconstruction loss')
        vqvae_group.add_argument('--mlp_mixer_blocks', type=int, default=4,
                               help='Number of MlpMixerBlocks in Encoder/Decoder')
        vqvae_group.add_argument('--dropout_rate', type=float, default=0.0,
                               help='Dropout rate for regularization')
        
        # VAR model parameter group (optional)
        var_group = parser.add_argument_group('VAR Configuration (Optional)')
        var_group.add_argument('--var_model_path', type=str, default="",
                             help='Path to pretrained VAR model')
        var_group.add_argument('--depth', type=int, default=4,
                             help='Number of transformer layers')
        var_group.add_argument('--embed_dim', type=int, default=256,
                             help='The size of embedding dimension')
        var_group.add_argument('--num_heads', type=int, default=16,
                             help='Number of attention heads')
        var_group.add_argument('--mlp_ratio', type=float, default=4.0,
                             help='MLP ratio')
        var_group.add_argument('--drop_rate', type=float, default=0.0,
                             help='Dropout rate')
        var_group.add_argument('--inp_seq_len', type=int, default=17,
                             help='Input sequence length')
        var_group.add_argument('--inp_feature_dim', type=int, default=2,
                             help='Input feature dimension')
        var_group.add_argument('--sos_method', type=str, default='linear',
                             choices=['linear', 'quantizer'],
                             help='SOS method')
        var_group.add_argument("--top-k", type=int, default=1, 
                             help='Top-k for inference')
        var_group.add_argument("--top-p", type=float, default=0.96, 
                             help='Top-p for inference')
        
        return parser
    
    def make_model(self) -> nn.Module:
        """Create Decoder model."""
        # Get dataset info
        train_dataset = self.train_loader.dataset
        adj_tuples = train_dataset.get_adj_tuples_symmetry_augmented()
        adj_tuples_for_model = [adj_tuples[gt] for gt in tuple(self.args.gt_patch_nums)]
        
        # Build VQVAE parameter dictionary
        vae_params = {
            'vocab_size': self.args.vocab_size,
            'embedding_dim': self.args.embedding_dim,
            'feature_dim': self.args.feature_dim,
            'beta': self.args.beta,
            'using_znorm': self.args.using_znorm,
            'quant_resi': self.args.quant_resi,
            'v_patch_nums': tuple(self.args.v_patch_nums),
            'gt_patch_nums': tuple(self.args.gt_patch_nums),
            'adj_matrices': adj_tuples_for_model,
            'use_reset': self.args.use_reset,
            'use_ema': self.args.use_ema,
            'add_codebook_loss': self.args.add_codebook_loss,
            'recon_loss_weight': self.args.recon_loss_weight,
            'mlp_mixer_blocks': self.args.mlp_mixer_blocks,
            'dropout_rate': self.args.dropout_rate,
            'expansion_strategy': self.args.expansion_strategy,
            'tokens_per_joint': self.args.tokens_per_joint
        }
        
        vae_model = HierarchicalVQVAE(**vae_params)
        vae_model.load_state_dict(torch.load(self.args.vqvae_model_path, map_location='cpu'), strict=False)
        vae_model.eval()
        self.vae_model = vae_model.to(self.device)
        
        if self.args.var_model_path:  
            var_model = VARForSkeleton(
                vae_local=vae_model,
                depth=self.args.depth,
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                mlp_ratio=self.args.mlp_ratio,
                drop_rate=self.args.drop_rate,
                patch_nums=tuple(self.args.v_patch_nums),
                inp_seq_len=self.args.inp_seq_len,
                inp_feature_dim=self.args.inp_feature_dim,
                sos_method=self.args.sos_method
            )
            var_model.load_state_dict(torch.load(self.args.var_model_path, map_location='cpu'), strict=False)
            var_model.vae_proxy[0].eval()
            var_model.vae_quant_proxy[0].eval()
            self.var_model = var_model.to(self.device)
        else:
            self.var_model = None
        
        return HierarchicalDecoder(
            feature_dim=self.args.feature_dim,
            embedding_dim=self.args.embedding_dim,
            gt_patch=self._gt_patch_int(),
            tokens_per_joint=self.args.tokens_per_joint,
            MlpMixerBlocks=self.args.mlp_mixer_blocks,
            dropout_rate=self.args.dropout_rate
        )
    
    def _get_model_inference_macs(self):
        self.model.eval()
        inp = torch.rand((1, self.model.gt_patch*self.model.tokens_per_joint, self.model.embedding_dim), device=self.device)
        with torch.no_grad():
            macs, _ = profile(copy.deepcopy(self.model), (inp,), verbose=False)
        return macs
    
    def forward_pass(self, model, data, mode):
        gt_patch_nums = self.vae_model.gt_patch_nums
        inp = []
        for gt_patch in gt_patch_nums:
            inp.append(data["poses_2d"][gt_patch])  # Already moved to device in base_trainer
        if mode == "train":
            gt_pose = data["poses_2d"][self._gt_patch_data_key()]  # Already moved to device in base_trainer
        with torch.amp.autocast("cuda", enabled=False):
            with torch.no_grad():
                vae_out = self.vae_model(inp)
                prev_embedding = vae_out["prev_embedding"]
                fhat = self.vae_model.quantizer.prev_embedding_to_any_fhat(prev_embedding, model.gt_patch*model.tokens_per_joint)
        phat = model(fhat)   
        out = {}
        if mode == "train":
            recon_loss = torch.nn.functional.mse_loss(phat, gt_pose)
            out["loss"] = recon_loss
            out['recon_loss'] = recon_loss
        if mode == "inference":
            out['vae_phat'] = phat
            if self.args.var_model_path:
                with torch.no_grad():
                    var_out = self.var_model.inference(lr_inp=inp[0], top_k=self.args.top_k, top_p=self.args.top_p)
                    prev_embedding = var_out["prev_embedding"]
                    with torch.amp.autocast("cuda", enabled=False):
                        fhat = self.vae_model.quantizer.prev_embedding_to_any_fhat(prev_embedding, model.gt_patch*model.tokens_per_joint)
                var_phat = model(fhat)
                out["var_phat"] = var_phat
            
        return out
    
    def eval_inference_out(self, inference_out: List[Dict]) -> Dict[str, float]:
        if not inference_out:
            return {'loss': float('inf')}
        
        gt_patch = self._gt_patch_data_key()
        metrics = {}
        
        # ========== Collect loss metrics ==========
        metrics.update(self._collect_loss_metrics(inference_out))
        
        # ========== Use helper method to collect and process data ==========
        # For non-integer gt_patch (e.g. "coco_17"), skip clear_root_and_pseudo,
        # because COCO skeleton has no standard root/pseudo definition
        skip_clear_root = isinstance(gt_patch, str)
        collected = self._collect_inference_data(
            inference_out,
            output_keys=['vae_phat', 'var_phat'],
            gt_patch_nums=[gt_patch],
            is_multi_scale=False,
            skip_clear_root=skip_clear_root
        )
        
        predictions = collected['predictions']
        targets = collected['targets']
        
        # Get root_id
        root_ids = self.train_loader.dataset.get_root_ids()
        root_id = self.args.eval_root_id if self.args.eval_root_id is not None else root_ids.get(gt_patch, 0)
        
        # ========== Compute coordinate metrics ==========
        if gt_patch in targets and 'vae_phat' in predictions:
            vae_pred_denorm = predictions['vae_phat'][gt_patch]
            target_denorm = targets[gt_patch]
            
            # VAE prediction evaluation
            vae_mpjpe = compute_mpjpe(vae_pred_denorm, target_denorm, root_id=root_id)
            metrics['vae_mpjpe'] = float(np.mean(vae_mpjpe))
            
            vae_pa_mpjpe = compute_p_mpjpe(vae_pred_denorm, target_denorm, root_id=root_id)
            metrics['vae_pa_mpjpe'] = float(np.mean(vae_pa_mpjpe))
            
            # VAR prediction evaluation (if exists)
            if gt_patch in predictions.get('var_phat', {}):
                var_pred_denorm = predictions['var_phat'][gt_patch]
                
                var_mpjpe = compute_mpjpe(var_pred_denorm, target_denorm, root_id=root_id)
                metrics['var_mpjpe'] = float(np.mean(var_mpjpe))
                
                var_pa_mpjpe = compute_p_mpjpe(var_pred_denorm, target_denorm, root_id=root_id)
                metrics['var_pa_mpjpe'] = float(np.mean(var_pa_mpjpe))
        
        # ========== Set mean MPJPE (using VAE MPJPE) ==========
        if 'vae_mpjpe' in metrics:
            metrics['mpjpe'] = metrics['vae_mpjpe']
        
        return metrics
    
    def save_prediction(self, inference_out: List[Dict]) -> None:
        """Save prediction results to npz file, supports cases without GT data"""
        if not inference_out:
            return
        
        gt_patch = self._gt_patch_data_key()
        
        # Use helper method to collect and process data
        skip_clear_root = isinstance(gt_patch, str)
        collected = self._collect_inference_data(
            inference_out,
            output_keys=['vae_phat', 'var_phat'],
            gt_patch_nums=[gt_patch],
            is_multi_scale=False,
            skip_clear_root=skip_clear_root
        )
        
        predictions = collected['predictions']
        targets = collected['targets']
        
        # Prepare dictionary to save - only save predictions
        save_dict = {}
        if gt_patch in predictions.get('var_phat', {}):
            save_dict['prediction'] = predictions["var_phat"][gt_patch]
        if gt_patch in predictions.get('vae_phat', {}):
            save_dict['vae_prediction'] = predictions["vae_phat"][gt_patch]
        
        # Only save gt when GT data is available
        if gt_patch in targets:
            save_dict['gt'] = targets
        
        # Save to npz file
        np.savez(self.args.save_prediction, **save_dict)
        print(f"Predictions saved to {self.args.save_prediction}")
