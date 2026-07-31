import torch
import torch.nn as nn
from thop import profile
import copy

from utils.train.dense_pose_base_trainer import DensePoseBaseTrainer

class DensePoseVQVAETrainer(DensePoseBaseTrainer):
    """VQVAE model trainer."""
    
    @staticmethod
    def add_parser_args(parser):
        """Add VQVAE-specific command line arguments."""
        parser = DensePoseBaseTrainer.add_parser_args(parser)
        
        # VQVAE model parameter group
        vqvae_group = parser.add_argument_group('VQVAE Model Configuration')
        vqvae_group.add_argument('--vqvae_model', type=str, default='HierarchicalVQVAE',
                               choices=['HierarchicalVQVAE', 'TestHierarchicalEncoderDecoder', 'HierarchicalAE'],
                               help='VQVAE model name')
        vqvae_group.add_argument('--vocab_size', type=int, default=4096,
                               help='The size of codebook')
        vqvae_group.add_argument('--embedding_dim', type=int, default=128,
                               help='The size of codebook embedding dimension')
        vqvae_group.add_argument('--beta', type=float, default=0.25,
                               help='Beta value for VQVAE')
        vqvae_group.add_argument('--using_znorm', type=bool, default=False,
                               help='Whether to use z-normalization')
        vqvae_group.add_argument('--quant_resi', type=float, default=0.5,
                               help='Quantization residual ratio (phi mixing: 0.5 = balanced)')
        vqvae_group.add_argument('--interpolate_mode', type=str, default='area',
                               choices=['area', 'linear', 'nearest'],
                               help='Interpolation mode for multi-scale token resampling (area|linear|nearest)')
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
        vqvae_group.add_argument("--use_text_guidance", action="store_true", 
                               help="use text guidance")
        vqvae_group.add_argument('--train_joint_num', type=int, default=-1,
                                 help='Target joint number to train exclusively (e.g., 17, 48, 96). -1 means train all levels together.')
        
        # Ablation flags
        vqvae_group.add_argument('--no_residual_quant', action='store_true',
                                 help='[Ablation] Disable residual quantization in multi-scale VQ')
        vqvae_group.add_argument('--no_phi', action='store_true',
                                 help='[Ablation] Disable learned phi (MLP ResNet) in quantizer')
        
        return parser
    
    def make_model(self) -> nn.Module:
        """Create VQVAE model."""
        # Dynamically import model class
        if self.args.vqvae_model == 'HierarchicalVQVAE':
            from models.vqvae_hierarchical_skeleton import HierarchicalVQVAE
            model_class = HierarchicalVQVAE
        elif self.args.vqvae_model == 'TestHierarchicalEncoderDecoder':
            from models.vqvae_hierarchical_skeleton import TestHierarchicalEncoderDecoder
            model_class = TestHierarchicalEncoderDecoder
        elif self.args.vqvae_model == 'HierarchicalAE':
            from models.vqvae_hierarchical_skeleton import HierarchicalAE
            model_class = HierarchicalAE
        else:
            raise ValueError(f"Unsupported VQVAE model: {self.args.vqvae_model}")
        
        def freeze_other_levels(model, level):
            for i in range(len(model.encoder)):
                trainable = (i == level)
                
                for p in model.encoder[i].parameters():
                    p.requires_grad = trainable
                for p in model.decoder[i].parameters():
                    p.requires_grad = trainable
        
        # Get dataset info
        train_dataset = self.train_loader.dataset
        adj_tuples = train_dataset.get_adj_tuples_symmetry_augmented()
        adj_tuples_for_model = [adj_tuples[gt] for gt in tuple(self.args.gt_patch_nums)]
        
        # Build parameter dictionary
        if self.args.vqvae_model == 'HierarchicalVQVAE':
            # Parameters for full VQVAE model
            params = {
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
                'tokens_per_joint': self.args.tokens_per_joint,
                'use_residual_quant': not self.args.no_residual_quant,
                'use_phi': not self.args.no_phi,
                'interpolate_mode': self.args.interpolate_mode,
            }
            model = model_class(**params)
        else:  # TestHierarchicalEncoderDecoder and HierarchicalAE
            # Test model only needs partial parameters
            params = {
                'embedding_dim': self.args.embedding_dim,
                'feature_dim': self.args.feature_dim,
                'gt_patch_nums': tuple(self.args.gt_patch_nums),
                'adj_matrices': adj_tuples_for_model,
                'mlp_mixer_blocks': self.args.mlp_mixer_blocks,
                'dropout_rate': self.args.dropout_rate,
                'expansion_strategy': self.args.expansion_strategy,
                'tokens_per_joint': self.args.tokens_per_joint
            }
            model = model_class(**params)

        # Set training levels based on joint count
        if hasattr(model, 'set_progressive_level'):
            if self.args.train_joint_num != -1:
                if self.args.train_joint_num in self.args.gt_patch_nums:
                    level_idx = self.args.gt_patch_nums.index(self.args.train_joint_num)
                    model.set_progressive_level(level_idx)
                    freeze_other_levels(model, level_idx)
                    print(f"[*] Exclusively training level {level_idx} (Joints: {self.args.train_joint_num})")
                else:
                    raise ValueError(f"train_joint_num {self.args.train_joint_num} not found in gt_patch_nums {self.args.gt_patch_nums}")
            else:
                model.set_progressive_level(-1) # Restore default multi-level joint training

        return model

    def _get_model_inference_macs(self):
        self.model.eval()
        gt_patch_nums = self.args.gt_patch_nums
        inp = []
        for gt_patch in gt_patch_nums:
            inp.append(torch.rand((1, gt_patch, 2), device=self.device))
        with torch.no_grad():
            macs, _ = profile(copy.deepcopy(self.model), (inp,), verbose=False)
        return macs
    
    def forward_pass(self, model, data, mode):
        gt_patch_nums = self.args.gt_patch_nums
        # split data (data already moved to device in base_trainer)
        inp = []
        for gt_patch in gt_patch_nums:
            inp.append(data["poses_2d"][gt_patch])
        model_out:dict = model(inp)
        out = {k:v for k,v in model_out.items() if "loss" in k}
        if mode == "inference":
            out["multi_phats"] = model_out["multi_phats"]
        return out