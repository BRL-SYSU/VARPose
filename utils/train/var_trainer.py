import torch
import torch.nn as nn
from thop import profile

from utils.ddp_utils import is_dist_initialized
from utils.train.dense_pose_base_trainer import DensePoseBaseTrainer
from models.var_skeleton import VARForSkeleton, ContinuousARForSkeleton 
from models.base_class import *

class DensePoseVARTrainer(DensePoseBaseTrainer):
    """VAR model trainer."""
    
    @staticmethod
    def add_parser_args(parser):
        """Add VAR-specific command line arguments."""
        parser = DensePoseBaseTrainer.add_parser_args(parser)
        
        # VAR model parameter group
        var_group = parser.add_argument_group('VAR Model Configuration')
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
        var_group.add_argument('--ar_mode', type=str, default='scale',
                               choices=['scale', 'token'],
                               help='AR mode: scale=predict-by-scale, token=token-by-token')
        var_group.add_argument('--upsampler', type=str, default='vae',
                               choices=['vae', 'linear'],
                               help='Upsampler: vae=get_next_autoregressive_input, linear=nn.Linear+right-shift (ablation)')
        
        # VAR inference parameter group
        var_inference_group = parser.add_argument_group('VAR Inference Configuration')
        var_inference_group.add_argument("--top-k", type=int, default=1, 
                                       help='Top-k for inference')
        var_inference_group.add_argument("--top-p", type=float, default=0.96, 
                                       help='Top-p for inference')
        var_inference_group.add_argument("--has_pseudo_mpjpe", action="store_true", 
                                       help='Whether to use pseudo MPJPE for inference')
        
        # VQVAE model parameter group
        vqvae_group = parser.add_argument_group('VQVAE Configuration')
        vqvae_group.add_argument('--vqvae_model_path', type=str, required=True,
                               help='Path to pretrained VQVAE model')
        vqvae_group.add_argument('--vqvae_model', type=str, default='HierarchicalVQVAE',
                               choices=['HierarchicalVQVAE', 'HierarchicalAE'],
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
        vqvae_group.add_argument("--use_text_guidance", action="store_true", 
                               help="use text guidance")
        
        return parser
    
    def make_model(self) -> nn.Module:
        """Create VAR model."""
        # First create the VQVAE model
        if self.args.vqvae_model == 'HierarchicalVQVAE':
            from models.vqvae_hierarchical_skeleton import HierarchicalVQVAE
            vae_model_class = HierarchicalVQVAE
        elif self.args.vqvae_model == 'HierarchicalAE':
            from models.vqvae_hierarchical_skeleton import HierarchicalAE
            vae_model_class = HierarchicalAE
        else:
            raise ValueError(f"Unsupported VQVAE model: {self.args.vqvae_model}")
        
        # Get dataset info
        train_dataset = self.train_loader.dataset
        adj_tuples = train_dataset.get_adj_tuples_symmetry_augmented()
        adj_tuples_for_model = [adj_tuples[gt] for gt in tuple(self.args.gt_patch_nums)]

        if self.args.vqvae_model == 'HierarchicalAE':
            # Continuous AE model does not need quantization parameters
            vqvae_params = {
                'embedding_dim': self.args.embedding_dim,
                'feature_dim': self.args.feature_dim,
                'gt_patch_nums': tuple(self.args.gt_patch_nums),
                'tokens_per_joint': self.args.tokens_per_joint,
                'expansion_strategy': self.args.expansion_strategy,
                'adj_matrices': adj_tuples_for_model,
                'recon_loss_weight': self.args.recon_loss_weight,
                'mlp_mixer_blocks': self.args.mlp_mixer_blocks,
                'dropout_rate': self.args.dropout_rate
            }
        else:
            # Build VQVAE common parameter dictionary
            vqvae_params = {
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
        
        # Instantiate AE model
        vae_model = vae_model_class(**vqvae_params)
        # Load pretrained weights
        ckpt = torch.load(self.args.vqvae_model_path, map_location='cpu')
        if 'state_dict' in ckpt:
            vae_model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            vae_model.load_state_dict(ckpt, strict=False)

        vae_model.to(self.device)
        vae_model.eval()
        
        # Create VAR model
        if self.args.vqvae_model == 'HierarchicalAE':
            model = ContinuousARForSkeleton(
                ae_local=vae_model,
                depth=self.args.depth,
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                mlp_ratio=self.args.mlp_ratio,
                drop_rate=self.args.drop_rate,
                inp_seq_len=self.args.inp_seq_len,
                inp_feature_dim=self.args.inp_feature_dim,
                sos_method=self.args.sos_method
            )
            # No quantizer, only freeze AE
            model.ae_proxy[0].eval()
        else:
            model = VARForSkeleton(
                vae_local=vae_model,
                depth=self.args.depth,
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                mlp_ratio=self.args.mlp_ratio,
                drop_rate=self.args.drop_rate,
                patch_nums=tuple(self.args.v_patch_nums),
                inp_seq_len=self.args.inp_seq_len,
                inp_feature_dim=self.args.inp_feature_dim,
                sos_method=self.args.sos_method,
                ar_mode=self.args.ar_mode,
                upsampler=self.args.upsampler,
            )
            model.vae_proxy[0].eval()
            model.vae_quant_proxy[0].eval()

        return model

    def _get_model_inference_macs(self):
        import copy
        raw_model = self.model.module if hasattr(self.model, 'module') and is_dist_initialized() else self.model
        test_model = ModelWrapper(copy.deepcopy(raw_model)).to(self.device)
        test_model.eval()
        test_input = torch.rand((1,17,2), device=self.device)
        with torch.no_grad():
            macs, _ = profile(test_model, (test_input,), verbose=False)
        del test_model
        return macs

    def forward_pass(self, model, data, mode):
        raw_model = model.module if hasattr(model, 'module') and is_dist_initialized() else model
        # Compatible with both continuous model (ae_proxy) and quantized model (vae_proxy)
        is_continuous = hasattr(raw_model, 'ae_proxy')
        if is_continuous:
            ae:MultiScaleAEBase = raw_model.ae_proxy[0] 
        else:
            ae:VQVAEBase = raw_model.vae_proxy[0]

        gt_patch_nums = ae.gt_patch_nums
        inp = []
        for gt_patch in gt_patch_nums:
            inp.append(data["poses_2d"][gt_patch])  # Already moved to device in base_trainer

        if is_continuous:
            if mode == "train":
                with torch.no_grad():
                    ae_out = ae(inp)
                    gt_embeddings = ae_out["prev_embedding"]

                out = model(gt_embeddings=gt_embeddings, lr_inp=inp[0])
                out = {k:v for k,v in out.items() if 'loss' in k}
            elif mode == "inference":
                out = {}
                inference_out = raw_model.inference(lr_inp=inp[0])
                pred_embeddings = inference_out["pred_embeddings"]

                # Directly pass through Decoder to convert continuous Embedding back to pose coordinates
                phats = ae.embeddings_to_multi_pose(pred_embeddings)
                out["multi_phats"] = phats

                # Pseudo MPJPE: Use Teacher Forcing generation as comparison baseline
                if self.args.has_pseudo_mpjpe:
                    with torch.no_grad():
                        ae_out = ae(inp)
                        gt_embeddings = ae_out["prev_embedding"]

                        pseudo_out = model(gt_embeddings=gt_embeddings, lr_inp=inp[0])
                        pseudo_phats = ae.embeddings_to_multi_pose(pseudo_out["pred_embeddings"])
                        out["pseudo_phats"] = pseudo_phats
            return out
        # Discrete quantization branch
        else:
            if mode == "train":
                # split data
                with torch.amp.autocast("cuda", enabled=False):
                    with torch.no_grad():
                        vae_out = raw_model.vae_proxy[0](inp)
                        idx_Bl_gt_list = vae_out['idx_Bl']
                        x_BLC_indices = torch.cat(idx_Bl_gt_list, dim=1)
                        x_BLC = raw_model.vae_quant_proxy[0].embedding(x_BLC_indices)
                out = model(
                    x_BLC=x_BLC,
                    lr_inp=inp[0],
                    idx_Bl_gt=idx_Bl_gt_list
                )
                out = {k:v for k,v in out.items() if 'loss' in k}
            elif mode == "inference":
                out = {}
                if self.args.has_pseudo_mpjpe:
                    with torch.amp.autocast("cuda", enabled=False):
                        with torch.no_grad():
                            vae_out = raw_model.vae_proxy[0](inp)
                            idx_Bl_gt_list = vae_out['idx_Bl']
                            x_BLC_indices = torch.cat(idx_Bl_gt_list, dim=1)
                            x_BLC = raw_model.vae_quant_proxy[0].embedding(x_BLC_indices)
                    out = model(
                        x_BLC=x_BLC,
                        lr_inp=inp[0],
                        idx_Bl_gt=idx_Bl_gt_list
                    )
                    out["pseudo_phats"] = raw_model.vae_proxy[0].idxbl_to_multi_pose(out["idx_Bl_hat"])
                inference_out = raw_model.inference(lr_inp=inp[0], top_k=self.args.top_k, top_p=self.args.top_p)
                phats = raw_model.vae_proxy[0].idxbl_to_multi_pose(inference_out["idx_Bl"])
                out["multi_phats"]= phats
            return out

class  ModelWrapper(nn.Module):
    def __init__(self, model:VARBase):
        super().__init__()
        self.model:VARBase = model
    def forward(self, input):
        return self.model.inference(input)
