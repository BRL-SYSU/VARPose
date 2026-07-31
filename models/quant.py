from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F
from models.base_class import *
from models.layers import MlpMixerBlock
from utils.ddp_utils import *


# # this file only provides the VectorQuantizer2 used in VQVAE
# __all__ = ['VectorQuantizer2',]


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)
    
class Phi_mlpmixer(nn.Module):
    def __init__(self, embed_dim, num_joints, quant_resi):
        super().__init__()
        self.resi_ratio = abs(quant_resi)
        self.mlp = MlpMixerBlock(num_joints, embed_dim ,4)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + self.mlp(h_BChw).mul_(self.resi_ratio)

class Phi_mlp(nn.Module):
    def __init__(self, embed_dim, quant_resi):
        super().__init__()
        self.resi_ratio = abs(quant_resi)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
        )
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + self.mlp(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class VectorQuantizerForSkeleton(VectorQuantizerBase):
    def __init__(self, vocab_size: int, Cvae: int, beta: float, using_znorm, v_patch_nums=None, quant_resi=0.5, use_ema=True, use_reset=True, add_codebook_loss=True, begin_sis=(0,), end_sis=(-1,), use_residual_quant=True, use_phi=True, interpolate_mode='area'):
        super().__init__(vocab_size, Cvae, beta, using_znorm)
        assert len(begin_sis)==len(end_sis)
        self.v_patch_nums:Tuple[int] = v_patch_nums
        self.quant_resi_ratio:float = quant_resi
        self.use_residual_quant = use_residual_quant
        self.use_phi = use_phi
        # Ablation: interpolation mode for multi-scale token resampling along joint dimension.
        # 'area' (default): area-weighted average, preserves feature energy, anti-aliased.
        # 'linear': linear interpolation, supports align_corners.
        # 'nearest': nearest-neighbor, no smoothing.
        assert interpolate_mode in ('area', 'linear', 'nearest'), f"Unsupported interpolate_mode: {interpolate_mode}"
        self.interpolate_mode = interpolate_mode

        if len(begin_sis)==1:
            self.begin_sis, self.end_sis = (0,), (len(self.v_patch_nums)-1,)
            self.quant_resi: nn.ModuleList = nn.ModuleList([Phi_mlpmixer(self.Cvae, self.v_patch_nums[-1], self.quant_resi_ratio) for _ in range(len(self.v_patch_nums))])
        else:
            self.begin_sis, self.end_sis = begin_sis, end_sis
            self.quant_resi: nn.ModuleList = nn.ModuleList([Phi_mlp(self.Cvae, self.quant_resi_ratio) for _ in range(len(self.v_patch_nums))])

        self._use_ema = use_ema
        self._use_reset = use_reset
        self._add_codebook_loss = add_codebook_loss
        self.eps = 1e-5
        self.decay = 0.99
        self.register_buffer('ema_cluster_size', torch.zeros(self.vocab_size))
         # Fix the initialization of ema_dw, should be consistent with the embedding weights
        self.register_buffer('ema_dw', self.embedding.weight.data.clone())
        # If using znorm, normalize the initial weights
        if self.using_znorm:
            self.ema_dw = F.normalize(self.ema_dw, dim=-1)     
        self.register_buffer('usage_count', torch.zeros(self.vocab_size, dtype=torch.long))
        
        # Added: pre-allocate buffers for EMA updates
        # Initialize as empty tensors, resized on first use based on batch_size
        self.register_buffer('prealloc_idx_N', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('prealloc_rest_NC', torch.empty(0, self.Cvae), persistent=False)
        self.register_buffer('prealloc_scale_idx', torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer('prealloc_prev_embedding', torch.empty(0, self.Cvae), persistent=False)
        self.register_buffer('prealloc_offset', torch.tensor(0, dtype=torch.long), persistent=False)
        
    @torch.no_grad()    
    def _calculate_prealloc_size(self, batch_size, ema_gt_idx=-1):
        """Compute the size of the pre-allocated tensors"""
        ema_si = self._normalize_idx(self.end_sis[self._normalize_idx(ema_gt_idx, len(self.end_sis))], len(self.v_patch_nums))
        scale_sizes = [batch_size * pn for pn in self.v_patch_nums[:ema_si+1]]
        return sum(scale_sizes)
    
    @torch.no_grad()
    def _ensure_prealloc_size(self, batch_size, ema_gt_idx=-1):
        """Ensure pre-allocated tensors are large enough, re-allocate if not (supports multi-GPU)"""
        device = self.prealloc_idx_N.device
        
        # Compute the size required for a single card
        single_card_size = self._calculate_prealloc_size(batch_size, ema_gt_idx)
        
        # If multi-GPU training, compute the total size across all cards
        if tdist.is_initialized():
            world_size = tdist.get_world_size()
            total_size = single_card_size * world_size
        else:
            total_size = single_card_size

        if self.prealloc_idx_N.size(0) < total_size:
            if is_main_process():
                print(f"Re-allocating pre-allocated tensors (multi-GPU), current size: {self.prealloc_idx_N.size(0)}, target size: {total_size}")
            del self.prealloc_prev_embedding
            del self.prealloc_idx_N
            self.register_buffer('prealloc_idx_N', torch.zeros(total_size, dtype=torch.long, device=device), persistent=False)
            self.register_buffer('prealloc_prev_embedding', torch.empty(total_size, self.Cvae, device=device), persistent=False)
            self.prealloc_scale_idx.zero_()
            self.prealloc_offset.zero_()
            if self.training and self._use_ema: 
                del self.prealloc_rest_NC
                self.register_buffer('prealloc_rest_NC', torch.zeros(total_size, self.Cvae, device=device), persistent=False)

    def _normalize_idx(self, idx:int, total_size:int):
        if idx>=0:
            return idx%total_size
        else:
            h = (-idx)//total_size+1
            return (total_size*h + idx)%total_size

    @torch.no_grad()
    def _gather_multi_card_data(self):
        """Gather multi-GPU data to the main card"""
        if not tdist.is_initialized():
            return
        
        world_size = tdist.get_world_size()
        if world_size <= 1:
            return
        
        # Get the effective data size of the current card
        current_offset = self.prealloc_offset.item()
        
        # Create lists for gathering data
        gathered_offsets = [torch.zeros_like(self.prealloc_offset) for _ in range(world_size)]
        gathered_idx_N = [torch.zeros_like(self.prealloc_idx_N[:current_offset]) for _ in range(world_size)]
        gathered_prev_embedding = [torch.zeros_like(self.prealloc_prev_embedding[:current_offset]) for _ in range(world_size)]
        
        # Gather offsets from all cards
        tdist.all_gather(gathered_offsets, self.prealloc_offset)
        
        # Gather idx_N data from all cards
        tdist.all_gather(gathered_idx_N, self.prealloc_idx_N[:current_offset])
        
        # Gather prev_embedding data from all cards
        tdist.all_gather(gathered_prev_embedding, self.prealloc_prev_embedding[:current_offset])
        
        # Only concatenate data on the main card
        if tdist.get_rank() == 0:
            # Compute total data volume
            total_offset = sum(offset.item() for offset in gathered_offsets)
            
            # Re-allocate a sufficiently large buffer
            if self.prealloc_idx_N.size(0) < total_offset:
                del self.prealloc_prev_embedding
                del self.prealloc_idx_N
                device = self.prealloc_offset.device
                self.register_buffer('prealloc_idx_N', torch.zeros(total_offset, dtype=torch.long, device=device), persistent=False)
                self.register_buffer('prealloc_prev_embedding', torch.empty(total_offset, self.Cvae, device=device), persistent=False)
            
            # Concatenate data from all cards
            offset = 0
            for i in range(world_size):
                card_offset = gathered_offsets[i].item()
                if card_offset > 0:
                    self.prealloc_idx_N[offset:offset + card_offset] = gathered_idx_N[i]
                    self.prealloc_prev_embedding[offset:offset + card_offset] = gathered_prev_embedding[i]
                    offset += card_offset
            
            self.prealloc_offset.fill_(offset)

    def forward(self, f_BJD: torch.Tensor, hasLoss=True, reset_dead = False, gt_idx=-1, ema_gt_idx=-1) -> dict[str, torch.Tensor]:
        if reset_dead and not self._use_reset:
            self.usage_count.zero_()
            return 0
        if reset_dead and not self.training:
            return 0
        
        SN = len(self.v_patch_nums)
        gt_idx = self._normalize_idx(gt_idx, len(self.begin_sis))
        ema_gt_idx = self._normalize_idx(ema_gt_idx, len(self.begin_sis))
        begin_si, end_si = self.begin_sis[gt_idx], self.end_sis[gt_idx]
        begin_si = self._normalize_idx(begin_si, SN)
        end_si = self._normalize_idx(end_si, SN)
        ema_si = self._normalize_idx(self.end_sis[ema_gt_idx], SN)

        dtype = f_BJD.dtype
        if dtype != torch.float32: f_BJD = f_BJD.float()
        B, _, D = f_BJD.shape
        f_no_grad = f_BJD.detach()
        
        f_rest:torch.Tensor = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        idx_Bl:list[torch.Tensor] = []
        prev_embedding:list[torch.Tensor] = []
        multi_fhats:list[torch.Tensor] = []
        
        with torch.amp.autocast("cuda", enabled=False):
            mean_vq_loss: torch.Tensor = 0.0

            with torch.no_grad():
                # Simplified conditional: no need to check if the attribute exists
                if not reset_dead:
                    self._ensure_prealloc_size(B, ema_gt_idx=ema_gt_idx)  # Ensure pre-allocated space is sufficient
                if self.prealloc_scale_idx != begin_si:
                    raise ValueError(f"self.prealloc_scale_idx != begin_si")

                if self.training and reset_dead:
                    reset_source = []
                previous_size =0
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                if si >= begin_si and si <= end_si:
                    # find the nearest embedding
                    if self.using_znorm:
                        rest_NC = F.interpolate(f_rest.permute(0, 2, 1), size=pn, mode=self.interpolate_mode).permute(0, 2, 1).reshape(-1, D) if (si != end_si) else f_rest.reshape(-1, D)
                        rest_NC = F.normalize(rest_NC, dim=-1)
                        idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                    else:
                        rest_NC = F.interpolate(f_rest.permute(0, 2, 1), size=pn, mode=self.interpolate_mode).permute(0, 2, 1).reshape(-1, D) if (si != end_si) else f_rest.reshape(-1, D)
                        d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                        d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                        idx_N = torch.argmin(d_no_grad, dim=1)
                    
                    # calc loss
                    idx_BJ = idx_N.view(B, pn)
                    idx_Bl.append(idx_BJ)
                    h_BJD = self.embedding(idx_BJ)
                    prev_embedding.append(h_BJD)
                    
                    with torch.no_grad():
                        self.usage_count.index_add_(0, idx_N, torch.ones_like(idx_N, dtype=torch.long))
                        current_size = idx_N.size(0)
                        offset = self.prealloc_offset.item()
                        self.prealloc_prev_embedding[offset:offset + current_size] = h_BJD.reshape(current_size, self.Cvae)
                        self.prealloc_idx_N[offset:offset + current_size] = idx_N
                        self.prealloc_scale_idx.add_(1)

                        if self.training and reset_dead:
                            reset_source.append(rest_NC)

                        if self.training and self._use_ema and not reset_dead:
                            # Simplified fill logic: fill directly into the pre-allocated buffer
                            self.prealloc_rest_NC[offset:offset + current_size] = rest_NC

                        self.prealloc_offset.add_(current_size)

                    h_BJD = self.prev_embedding_to_post(h_BJD, si, end_si)

                    if self.use_residual_quant:
                        f_hat = f_hat + h_BJD
                        f_rest = f_rest - h_BJD
                    else:
                        f_hat = h_BJD

                    f_out = (f_hat.data - f_no_grad) + f_BJD
                    multi_fhats.append(f_out)
                    
                    if hasLoss:
                        if self._add_codebook_loss:
                            mean_vq_loss += F.mse_loss(f_hat.data, f_BJD).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
                        else:
                            mean_vq_loss += F.mse_loss(f_hat.data, f_BJD).mul_(self.beta)
                elif si < begin_si:
                    with torch.no_grad():
                        h_BJD = self.prealloc_prev_embedding[previous_size : previous_size+B*self.v_patch_nums[si]].reshape(B, self.v_patch_nums[si], self.Cvae)
                        idx_BJ = self.prealloc_idx_N[previous_size : previous_size+B*self.v_patch_nums[si]].reshape(B, self.v_patch_nums[si])
                        previous_size += B*self.v_patch_nums[si]
                        prev_embedding.append(h_BJD)
                        idx_Bl.append(idx_BJ)
                        h_BJD = self.prev_embedding_to_post(h_BJD, si, end_si)
                        if self.use_residual_quant:
                            f_hat = f_hat + h_BJD
                            f_rest = f_rest - h_BJD
                        else:
                            f_hat = h_BJD
                        multi_fhats.append(f_hat.detach())
                else:
                    break
                
            out = {
                'multi_fhats':multi_fhats,
                'idx_Bl': idx_Bl,
                'prev_embedding': prev_embedding
            }
            if hasLoss:
                mean_vq_loss *= 1/(end_si - begin_si +1)
                out['loss'] = mean_vq_loss

            if end_si == ema_si:
                with torch.no_grad():
                    if self.training and self._use_ema and not reset_dead:
                        # Gather multi-GPU data
                        self._gather_multi_card_data()
                        if is_main_process():
                            self.update_ema_from_buffer()
                        # Sync embedding weights to all cards
                        if tdist.is_initialized():
                            tdist.broadcast(self.embedding.weight.data, 0)
                    self.prealloc_offset.zero_()
                    self.prealloc_scale_idx.zero_()

                    if self.training and reset_dead:
                        # Only perform reset on the main card
                        if is_main_process():
                            reset_source = torch.concatenate(reset_source, dim=0)
                            num_replaced = self.reset_dead_codebooks(reset_source)
                        else:
                            num_replaced = 0
                        
                        return num_replaced
        if reset_dead:
            return 0        
        return out
    
    @torch.no_grad()
    def update_ema_from_buffer(self):
        """Use the pre-allocated buffer for EMA updates (supports multi-GPU)"""
        # Simplified conditional
        if self.prealloc_offset.item() == 0:
            return
        
        # Directly use the valid portion of the pre-allocated buffer
        offset = self.prealloc_offset.item()
        idx_N = self.prealloc_idx_N[:offset]
        rest_NC = self.prealloc_rest_NC[:offset]
        
        # Optimized EMA update logic
        cluster_counts = idx_N.bincount(minlength=self.vocab_size).float()
        self.ema_cluster_size.mul_(self.decay).add_(cluster_counts, alpha=1 - self.decay)
        
        dw = torch.zeros(self.vocab_size, self.Cvae, device=rest_NC.device, dtype=rest_NC.dtype)
        dw.index_add_(0, idx_N, rest_NC)
        self.ema_dw.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        # Normalized update
        n = self.ema_cluster_size.sum()
        cluster_size_stable = (
            (self.ema_cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n
        )
        embed_normalized = self.ema_dw / cluster_size_stable.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

    @torch.no_grad()
    def reset_dead_codebooks(self, inputs: torch.Tensor):
        # Check whether usage_count is all zeros
        if torch.all(self.usage_count == 0):
            print("Error: usage_count is all zero, this indicates a problem!")
            print("This means no codebooks were used in the current epoch, which should not happen.")
            return 0
        
        flat_input = inputs.reshape(-1, self.Cvae)
        dead_indices = torch.where(self.usage_count == 0)[0]
        num_dead = len(dead_indices)
        
        if num_dead == 0:
            self.usage_count.zero_()
            return 0

        # Randomly select vectors from inputs to replace dead codebook entries
        num_to_sample = min(num_dead, len(flat_input))
        if num_to_sample == 0:
            self.usage_count.zero_()
            return 0
            
        sample_indices = torch.randint(0, len(flat_input), (num_to_sample,))
        replacement_vectors = flat_input[sample_indices]

        # Normalize the new vectors
        if self.using_znorm:
            replacement_vectors = F.normalize(replacement_vectors, dim=-1)
        
        # Replace the corresponding weights in the codebook
        self.embedding.weight.data[dead_indices[:num_to_sample]] = replacement_vectors
        
        # If EMA is used, also reset the EMA state to avoid conflict between old and new states
        if self._use_ema:
            self.ema_dw.data[dead_indices[:num_to_sample]] = replacement_vectors
            # Set ema_cluster_size to a small initial value to give new codebook entries a "startup" chance
            self.ema_cluster_size.data[dead_indices[:num_to_sample]] = self.eps

        self.usage_count.zero_()
        return num_dead
    
    def prev_embedding_to_post(self, prev_embedding: torch.Tensor, si:int, end_si:int=-1)->torch.Tensor:
        """
        A transition function to convert previous embeddings to post-transition embeddings.

        Args:
            prev_embedding (BxJxD): Embeddings before a transition
            si: Current scale index
            end_si: Target scale index
        Returns:
            Embeddings after a transition
        """
        end_si = self._normalize_idx(end_si, len(self.v_patch_nums))
        J = self.v_patch_nums[end_si]
        h_BJD = F.interpolate(prev_embedding.permute(0, 2, 1), size=J, mode=self.interpolate_mode).permute(0, 2, 1).contiguous() if (si != end_si) else prev_embedding.contiguous()
        if self.use_phi:
            h_BJD = self.quant_resi[si](h_BJD)
        return h_BJD

    def idxBl_to_prev_embedding(self, idx_Bl: List[torch.Tensor]) -> list[torch.Tensor]:
        """
        Convert multi-scale indices to embeddings before a transition.
        
        Args:
            idx_Bl [BxJ]xl: Ground truth multi-scale indices
            
        Returns:
            embeddings before a transition tensor
        """
        prev_embeddings =[]
        for si, idx in enumerate(idx_Bl):
            h_BJD = self.embedding(idx)
            prev_embeddings.append(h_BJD)
        return prev_embeddings

    def prev_embedding_to_multi_fhat(self, h_BJD:list[torch.Tensor], gt_idx=-1) -> list[torch.Tensor]:
        """
        Convert embeddings to multi-sclae feature representation.
        
        Args:
            h_BJD: Multi-scale embeddings
            gt_idx: gt patch index
        Returns:
            multi-sclae feature representation
        """
        gt_idx = self._normalize_idx(gt_idx, len(self.end_sis))
        fhats = []
        f_hat = torch.zeros_like(h_BJD[self.end_sis[gt_idx]])
        h_BJDs = h_BJD
        for si in range(self.end_sis[gt_idx]+1): # from small to large
            h_BJD = h_BJDs[si]
            h_BJD = self.prev_embedding_to_post(h_BJD, si, self.end_sis[gt_idx])
            f_hat.add_(h_BJD)
            fhats.append(f_hat.clone())
        return fhats
    
    def prev_embedding_to_any_fhat(self, h_BJDs:list[torch.Tensor], scale:int) -> torch.Tensor:
        if scale in self.v_patch_nums:
            si_max = self.v_patch_nums.index(scale)
            is_in_patch = True
        elif scale > self.v_patch_nums[-1]:
            si_max = len(self.v_patch_nums)-1
            is_in_patch = False
        else:
            for i in range(len(self.v_patch_nums)):
                if self.v_patch_nums[i] >= scale:
                    si_max = i
                    is_in_patch = False
                    break
        
        f_hat = torch.zeros_like(h_BJDs[si_max])
        for si in range(si_max+1): # from small to large
            h_BJD = h_BJDs[si]
            h_BJD = self.prev_embedding_to_post(h_BJD, si, si_max)
            f_hat.add_(h_BJD)

        if is_in_patch:
            return f_hat
        else:
            return  F.interpolate(f_hat.permute(0, 2, 1), size=scale, mode=self.interpolate_mode).permute(0, 2, 1).contiguous()


    
    def idxBl_to_multi_fhat(self, idx_Bl: List[torch.Tensor], gt_idx=-1) -> list[torch.Tensor]:
        """
        Convert multi-scale indices to multi-scale reconstructed features.
        
        Args:
            idx_Bl: Ground truth multi-scale indices
            v_patch_nums: Patch numbers for different scales
            gt_idx: gt patch index
        Returns:
            multi-sclae feature representation
        """
        fhats = self.prev_embedding_to_multi_fhat(self.idxBl_to_prev_embedding(idx_Bl), gt_idx)
        return fhats
    
    def idxBl_to_any_fhat(self, idx_Bl: List[torch.Tensor], scale:int) -> torch.Tensor:
        return self.prev_embedding_to_any_fhat(self.idxBl_to_prev_embedding(idx_Bl), scale)
    
    @torch.no_grad()
    def get_next_autoregressive_input(self, si: int, multi_fhats: list[torch.Tensor], h_BJD: torch.Tensor, gt_idx=-1) -> Tuple[Optional[torch.Tensor], torch.Tensor]: 
        gt_idx = self._normalize_idx(gt_idx, len(self.end_sis))
        if len(multi_fhats)>0:
            f_hat = multi_fhats[-1].clone()
            h = self.prev_embedding_to_post(h_BJD, si, self.end_sis[gt_idx]) 
            f_hat.add_(h)
        else:
            f_hat = self.prev_embedding_to_post(h_BJD, si, self.end_sis[gt_idx]) 
        if si != self.end_sis[gt_idx]:
            return f_hat, F.interpolate(f_hat.permute(0, 2, 1), size=self.v_patch_nums[si+1], mode=self.interpolate_mode).permute(0, 2, 1).contiguous()
        else:
            return f_hat, f_hat
    
    @ torch.no_grad()
    def training_cache_reset(self):
        device = self.embedding.weight.device
        self.register_buffer('ema_cluster_size', torch.zeros(self.vocab_size, device=device))
        self.register_buffer('ema_dw', self.embedding.weight.data.clone())
        self.register_buffer('usage_count', torch.zeros(self.vocab_size, dtype=torch.long, device=device))
        self.register_buffer('prealloc_idx_N', torch.empty(0, dtype=torch.long, device=device), persistent=False)
        self.register_buffer('prealloc_rest_NC', torch.empty(0, self.Cvae, device=device), persistent=False)
        self.register_buffer('prealloc_scale_idx', torch.tensor(0, dtype=torch.long, device=device), persistent=False)
        self.register_buffer('prealloc_prev_embedding', torch.empty(0, self.Cvae, device=device), persistent=False)
        self.register_buffer('prealloc_offset', torch.tensor(0, dtype=torch.long, device=device), persistent=False)
