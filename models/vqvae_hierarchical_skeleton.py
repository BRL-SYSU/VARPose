import torch
import torch.nn as nn
import torch.nn.functional as F

from models.quant import VectorQuantizerForSkeleton
from models.layers import MlpMixerBlock, ResiLinear, TokenExpansion, TokenAggregation
from models.base_class import *

class HierarchicalEncoder(nn.Module):
    def __init__(self,
        feature_dim = 2,
        embedding_dim = 128,
        gt_patch = 17,
        MlpMixerBlocks = 4,
        tokens_per_joint = 6,
        expansion_strategy='balanced', 
        dropout_rate=0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.gt_patch = gt_patch
        self.MlpMixerBlocks = MlpMixerBlocks
        self.tokens_per_joint = tokens_per_joint
        self.expansion_strategy = expansion_strategy
        
        self.token_expansion = TokenExpansion(tokens_per_joint=tokens_per_joint, strategy=expansion_strategy)

        self.steps = int(torch.log2(embedding_dim//feature_dim * torch.ones(1).long()))+1
        self.resi_linears = nn.Sequential(*[ResiLinear(feature_dim*2**(i), feature_dim*2**(i+1)) for i in range(self.steps)])
        self.linear = nn.Linear(feature_dim*2**(self.steps), embedding_dim)

        self.mlpmixers = nn.Sequential(*[MlpMixerBlock(self.gt_patch*self.tokens_per_joint, self.embedding_dim, dropout_rate=dropout_rate) for _ in range(self.MlpMixerBlocks)])
        self.out_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x:torch.Tensor, adj_matrices)->torch.Tensor:
        """
        Args:
            x: [B, J, F]
        Returns:
            [B, r*J, D]
        """
        x = self.token_expansion(x, adj_matrices)

        x = self.resi_linears(x)
        x = self.linear(x)
        x = self.mlpmixers(x)

        x = self.out_norm(x)
        return x

class HierarchicalDecoder(nn.Module):
    def __init__(self,
        feature_dim = 2,
        embedding_dim = 128,
        gt_patch = 17,
        MlpMixerBlocks = 4,
        tokens_per_joint = 6,
        expansion_strategy='balanced',
        dropout_rate=0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.gt_patch = gt_patch
        self.MlpMixerBlocks = MlpMixerBlocks
        self.tokens_per_joint = tokens_per_joint

        self.inp_norm = nn.LayerNorm(embedding_dim)
        self.mlpmixers = nn.Sequential(*[MlpMixerBlock(self.gt_patch*self.tokens_per_joint, self.embedding_dim, dropout_rate=dropout_rate) for _ in range(self.MlpMixerBlocks)])

        self.steps = int(torch.log2(embedding_dim//feature_dim * torch.ones(1).long()))-1
        self.resi_linears = nn.Sequential(*[ResiLinear(embedding_dim//(2**(i)), embedding_dim//(2**(i+1))) for i in range(self.steps)])
        self.linear = nn.Linear(embedding_dim//(2**(self.steps)), feature_dim)

        self.token_aggregation = TokenAggregation(tokens_per_joint=tokens_per_joint, strategy=expansion_strategy)

        self.tanh = nn.Tanh()
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Args:
            x: [B, r*J, D]
        Returns:
            [B, J, F]
        """
        x = self.inp_norm(x)
        x = self.mlpmixers(x)

        x = self.resi_linears(x)
        x = self.linear(x)

        x = self.token_aggregation(x)
        x = self.tanh(x)

        return x

class HierarchicalVQVAE(VQVAEBase):
    def __init__(
        self,
        vocab_size=4096,
        embedding_dim=128, 
        feature_dim=2,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        v_patch_nums=None, # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        gt_patch_nums=(17, 48, 96),
        tokens_per_joint = 6,
        expansion_strategy='balanced',
        adj_matrices:Tuple[Tuple[int,int], ...] = None,
        use_ema=True,           # whether to use exponential moving average for the nearest neighbors
        use_reset=True,
        add_codebook_loss=True,
        recon_loss_weight=200,
        mlp_mixer_blocks=4,
        dropout_rate = 0.1,
        use_residual_quant=True,
        use_phi=True,
        interpolate_mode='area',
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )
        self.feature_dim = feature_dim
        self.beta = beta
        self.quant_resi = quant_resi
        self.using_znorm = using_znorm
        self.v_patch_nums = v_patch_nums
        self.gt_patch_nums = gt_patch_nums
        self.tokens_per_joint = tokens_per_joint
        self.expansion_strategy = expansion_strategy
        self.adj_matrices = adj_matrices
        self.use_ema = use_ema
        self.use_reset = use_reset
        self.add_codebook_loss = add_codebook_loss
        self.recon_loss_weight = recon_loss_weight
        self.mlp_mixer_blocks=mlp_mixer_blocks
        self.dropout_rate = dropout_rate
        self.current_level = -1  # Added: -1 means training all levels simultaneously, 0,1,2 means training the corresponding level
        self._build_adj_matrix()
        if self.v_patch_nums is None:
            self.v_patch_nums = [sn*tokens_per_joint for sn in gt_patch_nums]
            v_patch_nums = self.v_patch_nums
        assert all(vpn*tokens_per_joint in v_patch_nums for vpn in gt_patch_nums), f'v_patch_nums must contain gt_patch_nums, but got {v_patch_nums} and {gt_patch_nums}'
        self._split_v_patch_nums()
        
        self.encoder = nn.ModuleList([HierarchicalEncoder(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            gt_patch=gt_patch,
            MlpMixerBlocks=mlp_mixer_blocks,
            tokens_per_joint=tokens_per_joint,
            expansion_strategy=self.expansion_strategy, 
            dropout_rate=dropout_rate
        ) for gt_patch in gt_patch_nums])

        self.decoder = nn.ModuleList([HierarchicalDecoder(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            gt_patch=gt_patch,
            MlpMixerBlocks=mlp_mixer_blocks,
            tokens_per_joint=tokens_per_joint,
            expansion_strategy=self.expansion_strategy, 
            dropout_rate=dropout_rate
        ) for gt_patch in gt_patch_nums])

        self.quantizer = VectorQuantizerForSkeleton(
            vocab_size=self.vocab_size,
            Cvae=self.embedding_dim,
            beta=self.beta,
            using_znorm=self.using_znorm,
            v_patch_nums=self.v_patch_nums,
            quant_resi=self.quant_resi,
            use_ema=self.use_ema,
            use_reset=self.use_reset,
            add_codebook_loss=self.add_codebook_loss,
            begin_sis=self.begin_sis,
            end_sis=self.end_sis,
            use_residual_quant=use_residual_quant,
            use_phi=use_phi,
            interpolate_mode=interpolate_mode,
        )

    def forward(self, p_BJDs: list[torch.Tensor], hasLoss=True):
        assert len(p_BJDs) == len(self.gt_patch_nums)
        out_dict = {
            'multi_phats': []
        }
        # Build adjacency matrices for each level
        level_adj_matrices = [adj.to(p_BJDs[0].device) for adj in self.adj_matrices]

        if hasLoss:
            vq_loss = torch.zeros((1,), dtype=torch.float, device=p_BJDs[-1].device)

        for i, p_BJD in enumerate(p_BJDs):
            embeddings = self.encoder[i](p_BJD, level_adj_matrices[i])
            
            if self.current_level < 0 or i<=self.current_level:
                out_quantizer = self.quantizer(embeddings, gt_idx=i, ema_gt_idx=self.current_level, hasLoss=hasLoss)
                fhat = out_quantizer['multi_fhats'][-1]
                phat = self.decoder[i](fhat)
                out_dict["multi_phats"].append(phat)
                if hasLoss:
                    if self.current_level >= 0:
                        if i == self.current_level:
                            vq_loss += out_quantizer["loss"]
                    else:
                        vq_loss += out_quantizer["loss"] * (self.end_sis[i] - self.begin_sis[i] + 1)
                if i == len(self.gt_patch_nums) - 1:
                    out_dict["idx_Bl"] = out_quantizer["idx_Bl"]
                    out_dict["prev_embedding"]= out_quantizer["prev_embedding"]
                    out_dict["multi_fhats"]=out_quantizer["multi_fhats"]
            else:
                # Maintain stable memory usage
                phat = self.decoder[i](embeddings)

        if hasLoss:
            # Progressive training: only compute the reconstruction loss for the current level
            if self.current_level >= 0:
                recon_loss = torch.zeros((1,), dtype=torch.float, device=p_BJDs[-1].device)
                recon_loss += F.mse_loss(p_BJDs[self.current_level], out_dict["multi_phats"][self.current_level])
            else:
                # Original multi-level loss computation logic
                recon_loss = torch.zeros((1,), dtype=torch.float, device=p_BJDs[-1].device)
                vq_loss /= len(self.v_patch_nums)
                for si in range(len(self.gt_patch_nums)):
                    recon_loss += F.mse_loss(p_BJDs[si], out_dict["multi_phats"][si])
                recon_loss = recon_loss / len(self.gt_patch_nums)
            
            alignment_loss = torch.zeros((1,), dtype=torch.float, device=p_BJDs[-1].device)
            out_dict["vq_loss"] = vq_loss
            out_dict["recon_loss"] = recon_loss
            out_dict["alignment_loss"] = alignment_loss
            out_dict["loss"] = self.recon_loss_weight*recon_loss + out_dict["vq_loss"]
        
        return out_dict
    
    def _build_adj_matrix(self):
        """Build adjacency matrices from adj_matrices"""
        level_adj_matrices = []
        for k, gt_patch_num in enumerate(self.gt_patch_nums):
            if self.adj_matrices is None:
                # If no adjacency matrix is provided, use full connectivity
                adj_matrix = torch.ones(gt_patch_num, gt_patch_num)
                adj_matrix.fill_diagonal_(0)
            else:
                adj_matrix = torch.zeros(gt_patch_num, gt_patch_num)
                for i, j in self.adj_matrices[k]:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
            level_adj_matrices.append(adj_matrix)
        self.adj_matrices = level_adj_matrices
    
    def _split_v_patch_nums(self):
        begin_sis = [0]
        end_sis = []
        gi = 0
        for k, v_patch_num in enumerate(self.v_patch_nums):
            if v_patch_num%self.tokens_per_joint==0 and  v_patch_num//self.tokens_per_joint == self.gt_patch_nums[gi]:
                end_sis.append(k)
                begin_sis.append(k+1)
                gi += 1
                if gi >= len(self.gt_patch_nums)-1:
                    end_sis.append(len(self.v_patch_nums)-1)
                    break
        self.begin_sis = tuple(begin_sis)
        self.end_sis = tuple(end_sis)
    
    def idxbl_to_multi_pose(self, idx_bl:list[torch.Tensor])->torch.Tensor:
        phats = []
        for gt_idx, gt_patch in enumerate(self.gt_patch_nums):
            fhat =self.quantizer.idxBl_to_multi_fhat(idx_bl, gt_idx)[-1]
            phats.append(self.decoder[gt_idx](fhat))
        return phats
    
    def prev_embedding_to_multi_pose(self, prev_embedding:list[torch.Tensor])->torch.Tensor:
        phats = []
        for gt_idx, gt_patch in enumerate(self.gt_patch_nums):
            fhat =self.quantizer.prev_embedding_to_multi_fhat(prev_embedding, gt_idx)[-1]
            phats.append(self.decoder[gt_idx](fhat))
        return phats

    def reset_dead_codebook_after_one_epochs(self, p_BJDs:list[torch.Tensor]):
        # Build adjacency matrices for each level
        level_adj_matrices = [adj.to(p_BJDs[0].device) for adj in self.adj_matrices]
        cnt = 0
        for i in range(len(level_adj_matrices)):
            embeddings = self.encoder[i](p_BJDs[i], level_adj_matrices[i])
            cnt += self.quantizer(embeddings, gt_idx=i, hasLoss=False, reset_dead=True)
        return cnt
    
    def first_pose_quantize(self, first_pose:torch.Tensor)->dict[str, torch.Tensor]|None:
        first_adj = self.adj_matrices[0].to(first_pose.device)
        embeddings = self.encoder[0](first_pose, first_adj)
            
        out_quantizer = self.quantizer(embeddings, gt_idx=0, ema_gt_idx=0)
        out = {
            "prev_embedding": out_quantizer["prev_embedding"],
            "idx_Bl": out_quantizer["idx_Bl"],
        }
        return out
        
    
    def set_progressive_level(self, level):
        """Set the current level for progressive training
        
        Args:
            level (int): Level index, -1 means training all levels simultaneously, 0,1,2 means training the corresponding level
        """
        if level < -1 or level >= len(self.gt_patch_nums):
            raise ValueError(f"Level must be between -1 and {len(self.gt_patch_nums)-1}, got {level}")
        
        self.current_level = level
        print(f"Set progressive training level to {level}")

class TestHierarchicalEncoderDecoder(VQVAEBase):
    def __init__(
        self,
        embedding_dim=128, 
        feature_dim=2,
        gt_patch_nums=(17, 48, 96),
        tokens_per_joint = 6,
        adj_matrices:Tuple[Tuple[int,int], ...] = None,
        mlp_mixer_blocks=4,
        dropout_rate = 0.1
    ):
        super().__init__(
            vocab_size=0,
            embedding_dim = embedding_dim
        )
        self.feature_dim = feature_dim
        self.gt_patch_nums = gt_patch_nums
        self.tokens_per_joint = tokens_per_joint
        self.adj_matrices = adj_matrices
        self._build_adj_matrix()
        
        self.encoder = nn.ModuleList([HierarchicalEncoder(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            gt_patch=gt_patch,
            MlpMixerBlocks=mlp_mixer_blocks,
            tokens_per_joint=tokens_per_joint,
            dropout_rate=dropout_rate
        ) for gt_patch in gt_patch_nums])

        self.decoder = nn.ModuleList([HierarchicalDecoder(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            gt_patch=gt_patch,
            MlpMixerBlocks=mlp_mixer_blocks,
            tokens_per_joint=tokens_per_joint,
            dropout_rate=dropout_rate
        ) for gt_patch in gt_patch_nums])

    def forward(self, p_BJDs: list[torch.Tensor], hasLoss=True):
        assert len(p_BJDs) == len(self.gt_patch_nums)
        out_dict = {
            'multi_phats': []
        }
        # Build adjacency matrices for each level
        level_adj_matrices = [adj.to(p_BJDs[0].device) for adj in self.adj_matrices]
        for i, p_BJD in enumerate(p_BJDs):
            embeddings = self.encoder[i](p_BJD, level_adj_matrices[i])
            multi_phats = self.decoder[i](embeddings)
            out_dict["multi_phats"].append(multi_phats)

        if hasLoss:
            recon_loss = torch.zeros((1,), dtype=torch.float, device=p_BJDs[-1].device)
            vq_loss = torch.zeros((1,), dtype=torch.float, device=p_BJDs[-1].device)
            alignment_loss = torch.zeros((1,), dtype=torch.float, device=p_BJDs[-1].device)
            for si in range(len(self.gt_patch_nums)):
                recon_loss += F.mse_loss(p_BJDs[si], out_dict["multi_phats"][si])
            recon_loss = recon_loss / len(self.gt_patch_nums)
            out_dict["vq_loss"] = vq_loss
            out_dict["recon_loss"] = recon_loss
            out_dict["alignment_loss"] = alignment_loss
            out_dict["loss"] = recon_loss + out_dict["vq_loss"]
        
        return out_dict
    
    def _build_adj_matrix(self):
        """Build adjacency matrices from adj_matrices"""
        level_adj_matrices = []
        for k, gt_patch_num in enumerate(self.gt_patch_nums):
            if self.adj_matrices is None:
                # If no adjacency matrix is provided, use full connectivity
                adj_matrix = torch.ones(gt_patch_num, gt_patch_num)
                adj_matrix.fill_diagonal_(0)
            else:
                adj_matrix = torch.zeros(gt_patch_num, gt_patch_num)
                for i, j in self.adj_matrices[k]:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
            level_adj_matrices.append(adj_matrix)
        self.adj_matrices = level_adj_matrices
    
    def idxbl_to_multi_pose(self, idx_bl:list[torch.Tensor])->torch.Tensor:
        pass
    
    def prev_embedding_to_multi_pose(self, prev_embedding:list[torch.Tensor])->torch.Tensor:
        pass

    def reset_dead_codebook_after_one_epochs(self, p_BJDs:list[torch.Tensor]):
        return 0

class HierarchicalAE(MultiScaleAEBase):
    """
    Pure continuous hierarchical autoencoder without quantization
    Without adding multimodal for now, only used to test reconstruction quality without a Codebook.
    """
    def __init__(
        self,
        embedding_dim=128,
        feature_dim=2,
        gt_patch_nums=(17, 48, 96),
        tokens_per_joint=6,
        expansion_strategy='balanced',
        adj_matrices:Tuple[Tuple[int,int], ...] = None,
        recon_loss_weight=200,
        mlp_mixer_blocks=4,
        dropout_rate=0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.gt_patch_nums = gt_patch_nums
        self.tokens_per_joint = tokens_per_joint
        self.expansion_strategy = expansion_strategy
        self.adj_matrices = adj_matrices
        self.recon_loss_weight = recon_loss_weight
        self.mlp_mixer_blocks = mlp_mixer_blocks
        self.dropout_rate = dropout_rate
        self.current_level = -1
        self._build_adj_matrix()
        
        self.encoder = nn.ModuleList([HierarchicalEncoder(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            gt_patch=gt_patch,
            MlpMixerBlocks=mlp_mixer_blocks,
            tokens_per_joint=tokens_per_joint,
            expansion_strategy=self.expansion_strategy, 
            dropout_rate=dropout_rate
        ) for gt_patch in gt_patch_nums])

        self.decoder = nn.ModuleList([HierarchicalDecoder(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            gt_patch=gt_patch,
            MlpMixerBlocks=mlp_mixer_blocks,
            tokens_per_joint=tokens_per_joint,
            expansion_strategy=self.expansion_strategy, 
            dropout_rate=dropout_rate
        ) for gt_patch in gt_patch_nums])

    def forward(self, p_BJDs: list[torch.Tensor], hasLoss=True):
        assert len(p_BJDs) == len(self.gt_patch_nums)
        out_dict = {
            'multi_phats': [],
            'prev_embedding': [],
            'prev_emb_single': None,
            'idx_Bl': None,
            'multi_fhats': None
        }

        # Ensure adj_matrices are on the correct device
        device = p_BJDs[0].device
        adj_matrices = [adj.to(device) for adj in self.adj_matrices]

        for i, p_BJD in enumerate(p_BJDs):
            # Directly obtain continuous features (quantization step removed)
            # If a specific training level is specified and the current one does not match, skip directly
            if self.current_level >= 0 and i != self.current_level:
                out_dict["prev_embedding"].append(None)
                out_dict["multi_phats"].append(None)
                continue
            embeddings = self.encoder[i](p_BJD, adj_matrices[i])
            out_dict["prev_embedding"].append(embeddings)
            # Send directly to the decoder for decoding
            phat = self.decoder[i](embeddings)
            out_dict["multi_phats"].append(phat)

            # Record the features of the current training level (or the last level)
            if self.current_level >= 0:
                if i == self.current_level:
                    out_dict["prev_emb_single"] = embeddings
            elif i == len(self.gt_patch_nums) - 1:
                # Record the finest-grained features and placeholder the remaining outputs
                out_dict["prev_emb_single"] = embeddings


        if hasLoss:
            # Continuous AE only needs to compute the MSE reconstruction loss
            recon_loss = torch.zeros((1,), dtype=torch.float, device=p_BJDs[-1].device)

            if self.current_level >= 0:
                # Single-level training: only compute the loss for the selected level
                recon_loss += F.mse_loss(p_BJDs[self.current_level], out_dict["multi_phats"][self.current_level])
            else:
                # Multi-level joint training
                level_weights = [4.0, 2.0, 1.0]
                total_weight = sum(level_weights[:len(self.gt_patch_nums)])

                for si in range(len(self.gt_patch_nums)):
                    curr_loss = F.mse_loss(p_BJDs[si], out_dict["multi_phats"][si])
                    recon_loss += level_weights[si] * curr_loss
                recon_loss = recon_loss / total_weight
            
            out_dict["recon_loss"] = recon_loss
            out_dict["loss"] = recon_loss

        return out_dict


    def _build_adj_matrix(self):
        """Build adjacency matrices from adj_matrices"""
        level_adj_matrices = []
        for k, gt_patch_num in enumerate(self.gt_patch_nums):
            if self.adj_matrices is None:
                # If no adjacency matrix is provided, use full connectivity
                adj_matrix = torch.ones(gt_patch_num, gt_patch_num)
            else:
                adj_matrix = torch.zeros(gt_patch_num, gt_patch_num)
                for i, j in self.adj_matrices[k]:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                adj_matrix.fill_diagonal_(1)
            level_adj_matrices.append(adj_matrix)

        # Store as a plain instance variable (list of tensors)
        self.adj_matrices = level_adj_matrices


    def embeddings_to_multi_pose(self, embeddings_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode continuous features directly into multi-scale poses"""
        phats = []
        for gt_idx, emb in enumerate(embeddings_list):
            phats.append(self.decoder[gt_idx](emb))
        return phats
