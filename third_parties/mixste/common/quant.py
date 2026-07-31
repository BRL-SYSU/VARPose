from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F
from common.base_class import *
from common.layers import MlpMixerBlock
from common.ddp_utils import *


# # this file only provides the VectorQuantizer2 used in VQVAE
# __all__ = ['VectorQuantizer2',]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        self.vocab_size:int = vocab_size
        self.Cvae:int = Cvae
        self.beta: float = beta
        self.using_znorm:bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                
                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
                
                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
                
                if self.training and dist.initialized():
                    handler.wait()
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        
        if tdist.is_initialized():
            margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        else:
            margin = (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: usages = None
        return f_hat, usages, mean_vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        idx_N_list: List[torch.Tensor] = []
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_gt = torch.argmin(d_no_grad, dim=1)
            
            scale = -40.
            idx_N = idx_gt

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            if si != SN - 1:
                # consistency
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            if si == SN - 1:
                f_rest_wo_last_discrete = f_rest.clone()
            f_rest.sub_(h_BChw)
            idx_N_list.append(idx_N.reshape(B, ph * pw))
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_gt.reshape(B, ph*pw))
    
        f_hat_or_idx_Bl.append(
            f_rest_wo_last_discrete.clone()
            .permute(0, 2, 3, 1)
            .reshape(-1, ph * pw, self.Cvae)
        )
        
        return f_hat_or_idx_Bl, idx_N_list
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor], v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        if v_patch_nums is None:
            v_patch_nums = self.v_patch_nums
        H = W = v_patch_nums[-1]
        SN = len(v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = v_patch_nums[0]
        for si in range(SN-1):
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32

    def idxBl_to_input(self, gt_ms_idx_Bl: List[torch.Tensor], v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        if v_patch_nums is None:
            v_patch_nums = self.v_patch_nums
        H = W = v_patch_nums[-1]
        SN = len(v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = v_patch_nums[0]
        for si in range(SN):
            pn_next = v_patch_nums[si]
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            #h = self.quant_resi[si/(SN-1)](h_BChw)
            h = h_BChw
            f_hat.add_(h)
            return f_hat, f_hat


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
    def __init__(self, vocab_size: int, Cvae: int, beta: float, using_znorm, v_patch_nums=None, quant_resi=0.5, use_ema=True, use_reset=True, add_codebook_loss=True, begin_sis=(0,), end_sis=(-1,)):
        super().__init__(vocab_size, Cvae, beta, using_znorm)
        assert len(begin_sis)==len(end_sis)
        self.v_patch_nums:Tuple[int] = v_patch_nums
        self.quant_resi_ratio:float = quant_resi

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
        # Fix ema_dw initialization so it matches the embedding weights.
        self.register_buffer('ema_dw', self.embedding.weight.data.clone())
        # If znorm is used, normalize the initial weights.
        if self.using_znorm:
            self.ema_dw = F.normalize(self.ema_dw, dim=-1)     
        self.register_buffer('usage_count', torch.zeros(self.vocab_size, dtype=torch.long))
        
        # Preallocated buffers for EMA updates.
        # Initialize as empty tensors and resize according to batch_size on first use.
        self.register_buffer('prealloc_idx_N', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('prealloc_rest_NC', torch.empty(0, self.Cvae), persistent=False)
        self.register_buffer('prealloc_scale_idx', torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer('prealloc_prev_embedding', torch.empty(0, self.Cvae), persistent=False)
        self.register_buffer('prealloc_offset', torch.tensor(0, dtype=torch.long), persistent=False)
        
    @torch.no_grad()    
    def _calculate_prealloc_size(self, batch_size, ema_gt_idx=-1):
        """Compute the size of the preallocated tensors."""
        ema_si = self._normalize_idx(self.end_sis[self._normalize_idx(ema_gt_idx, len(self.end_sis))], len(self.v_patch_nums))
        scale_sizes = [batch_size * pn for pn in self.v_patch_nums[:ema_si+1]]
        return sum(scale_sizes)
    
    @torch.no_grad()
    def _ensure_prealloc_size(self, batch_size, ema_gt_idx=-1):
        """Ensure preallocated tensors are large enough; reallocate if needed (multi-GPU aware)."""
        device = self.prealloc_idx_N.device
        
        # Compute the size required for a single GPU.
        single_card_size = self._calculate_prealloc_size(batch_size, ema_gt_idx)
        
        # If this is multi-GPU training, compute the total size across all GPUs.
        if tdist.is_initialized():
            world_size = tdist.get_world_size()
            total_size = single_card_size * world_size
        else:
            total_size = single_card_size

        if self.prealloc_idx_N.size(0) < total_size:
            # if is_main_process():
            #     print(f"Reallocating preallocated tensors (multi-GPU), current size: {self.prealloc_idx_N.size(0)}, target size: {total_size}")
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
        """Gather multi-GPU data onto the main GPU."""
        if not tdist.is_initialized():
            return
        
        world_size = tdist.get_world_size()
        if world_size <= 1:
            return
        
        # Get the valid data size on the current GPU.
        current_offset = self.prealloc_offset.item()
        
        # Create lists used for gathering data.
        gathered_offsets = [torch.zeros_like(self.prealloc_offset) for _ in range(world_size)]
        gathered_idx_N = [torch.zeros_like(self.prealloc_idx_N[:current_offset]) for _ in range(world_size)]
        gathered_prev_embedding = [torch.zeros_like(self.prealloc_prev_embedding[:current_offset]) for _ in range(world_size)]
        
        # Gather offsets from all GPUs.
        tdist.all_gather(gathered_offsets, self.prealloc_offset)
        
        # Gather idx_N data from all GPUs.
        tdist.all_gather(gathered_idx_N, self.prealloc_idx_N[:current_offset])
        
        # Gather prev_embedding data from all GPUs.
        tdist.all_gather(gathered_prev_embedding, self.prealloc_prev_embedding[:current_offset])
        
        # Concatenate data only on the main GPU.
        if tdist.get_rank() == 0:
            # Compute the total amount of data.
            total_offset = sum(offset.item() for offset in gathered_offsets)
            
            # Reallocate a large enough buffer.
            if self.prealloc_idx_N.size(0) < total_offset:
                del self.prealloc_prev_embedding
                del self.prealloc_idx_N
                device = self.prealloc_offset.device
                self.register_buffer('prealloc_idx_N', torch.zeros(total_offset, dtype=torch.long, device=device), persistent=False)
                self.register_buffer('prealloc_prev_embedding', torch.empty(total_offset, self.Cvae, device=device), persistent=False)
            
            # Concatenate data from all GPUs.
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
                # Simplified condition: no need to check whether attributes exist.
                if not reset_dead:
                    self._ensure_prealloc_size(B, ema_gt_idx=ema_gt_idx)  # Ensure enough preallocated space
                if self.prealloc_scale_idx != begin_si:
                    raise ValueError(f"self.prealloc_scale_idx != begin_si")

                if self.training and reset_dead:
                    reset_source = []
                previous_size =0
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                if si >= begin_si and si <= end_si:
                    # find the nearest embedding
                    if self.using_znorm:
                        rest_NC = F.interpolate(f_rest.permute(0, 2, 1), size=pn, mode='area').permute(0, 2, 1).reshape(-1, D) if (si != end_si) else f_rest.reshape(-1, D)
                        rest_NC = F.normalize(rest_NC, dim=-1)
                        idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                    else:
                        rest_NC = F.interpolate(f_rest.permute(0, 2, 1), size=pn, mode='area').permute(0, 2, 1).reshape(-1, D) if (si != end_si) else f_rest.reshape(-1, D)
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
                            # Simplified fill logic: write directly into the preallocated buffer.
                            self.prealloc_rest_NC[offset:offset + current_size] = rest_NC

                        self.prealloc_offset.add_(current_size)

                    h_BJD = self.prev_embedding_to_post(h_BJD, si, end_si)

                    f_hat = f_hat + h_BJD
                    f_rest = f_rest - h_BJD

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
                        f_hat = f_hat + h_BJD
                        f_rest = f_rest - h_BJD
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
                        # Gather multi-GPU data.
                        self._gather_multi_card_data()
                        if is_main_process():
                            self.update_ema_from_buffer()
                        # Synchronize embedding weights to all GPUs.
                        if tdist.is_initialized():
                            tdist.broadcast(self.embedding.weight.data, 0)
                    self.prealloc_offset.zero_()
                    self.prealloc_scale_idx.zero_()

                    if self.training and reset_dead:
                        # Run the reset operation only on the main GPU.
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
        """Update EMA from the preallocated buffer (multi-GPU aware)."""
        # Simplified condition
        if self.prealloc_offset.item() == 0:
            return
        
        # Use the valid part of the preallocated buffer directly.
        offset = self.prealloc_offset.item()
        idx_N = self.prealloc_idx_N[:offset]
        rest_NC = self.prealloc_rest_NC[:offset]
        
        # Optimized EMA update logic.
        cluster_counts = idx_N.bincount(minlength=self.vocab_size).float()
        self.ema_cluster_size.mul_(self.decay).add_(cluster_counts, alpha=1 - self.decay)
        
        dw = torch.zeros(self.vocab_size, self.Cvae, device=rest_NC.device, dtype=rest_NC.dtype)
        dw.index_add_(0, idx_N, rest_NC)
        self.ema_dw.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        # Normalized update.
        n = self.ema_cluster_size.sum()
        cluster_size_stable = (
            (self.ema_cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n
        )
        embed_normalized = self.ema_dw / cluster_size_stable.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

    @torch.no_grad()
    def reset_dead_codebooks(self, inputs: torch.Tensor):
        # Check whether usage_count is all zeros.
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

        # Randomly sample vectors from the input to replace dead codebook entries.
        num_to_sample = min(num_dead, len(flat_input))
        if num_to_sample == 0:
            self.usage_count.zero_()
            return 0
            
        sample_indices = torch.randint(0, len(flat_input), (num_to_sample,))
        replacement_vectors = flat_input[sample_indices]

        # Normalize new vectors.
        if self.using_znorm:
            replacement_vectors = F.normalize(replacement_vectors, dim=-1)
        
        # Replace the corresponding codebook weights.
        self.embedding.weight.data[dead_indices[:num_to_sample]] = replacement_vectors
        
        # If EMA is used, reset its state too to avoid conflicts between old and new state.
        if self._use_ema:
            self.ema_dw.data[dead_indices[:num_to_sample]] = replacement_vectors
            # Set ema_cluster_size to a small initial value to give new entries a warm start.
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
        h_BJD = F.interpolate(prev_embedding.permute(0, 2, 1), size=J, mode='area').permute(0, 2, 1).contiguous() if (si != end_si) else prev_embedding.contiguous()
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
            return  F.interpolate(f_hat.permute(0, 2, 1), size=scale, mode='area').permute(0, 2, 1).contiguous()


    
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
            return f_hat, F.interpolate(f_hat.permute(0, 2, 1), size=self.v_patch_nums[si+1], mode='area').permute(0, 2, 1).contiguous()
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
