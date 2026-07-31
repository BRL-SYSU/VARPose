import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import ddp_utils

from models.helpers import sample_with_top_k_top_p_
from models.base_class import *
from models.layers import *
from models.layers import rotate_half


class SimpleSelfAttention(nn.Module):
    """SelfAttention with integrated RoPE"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

        self.k_cache = None
        self.v_cache = None
        
    def forward(self, x, attn_mask=None, use_cache=False, position_offset: int = 0):
        B, L, C = x.shape # Batch, SeqLen, EmbedDim

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k_new = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v_new = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        if use_cache:
            if self.k_cache is not None:
                # Concatenate the new k, v with the cached k, v
                k = torch.cat([self.k_cache, k_new], dim=1)
                v = torch.cat([self.v_cache, v_new], dim=1)
            else:
                k = k_new
                v = v_new
            
            # Update the cache for the next iteration
            self.k_cache = k
            self.v_cache = v
        else:
            k = k_new
            v = v_new

        # Apply rotation to q, its position has an offset
        cos_q, sin_q = self.rotary_emb(q, offset=position_offset)
        cos_q, sin_q = cos_q.unsqueeze(1), sin_q.unsqueeze(1)
        q = (q * cos_q) + (rotate_half(q) * sin_q)

        # Apply rotation to the full k sequence, its position starts from 0
        cos_k, sin_k = self.rotary_emb(k, offset=0)
        cos_k, sin_k = cos_k.unsqueeze(1), sin_k.unsqueeze(1)
        k = (k * cos_k) + (rotate_half(k) * sin_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(B, L, C)

        return self.out_proj(context)
    
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None


class SimpleTransformerBlock(nn.Module):
    """Simplified TransformerBlock"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SimpleSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.use_cache = False
        
    def forward(self, x, attn_mask=None, position_offset: int = 0):
        # Self-attention with causal mask
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, use_cache=self.use_cache, position_offset=position_offset)
        # Feed-forward
        x = x + self.mlp(self.norm2(x))
        return x
    
    def set_use_cache(self, use_cache:bool):
        self.use_cache = use_cache
        self.attn.reset_cache()

class ProjectorForSparse(nn.Module):
    def __init__(self, feature_dim=2, embedding_dim=1024, inp_seq_len=17, out_seq_len=17*6):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding = embedding_dim
        self.inp_seq_len = inp_seq_len
        self.out_seq_len = out_seq_len
        self.prejector_seq = nn.Sequential(
            nn.Linear(self.inp_seq_len, self.out_seq_len//2),
            nn.GELU(),
            nn.Linear(self.out_seq_len//2, self.out_seq_len)
        )
        self.projector_dim = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim//2),
            nn.GELU(),
            nn.Linear(embedding_dim//2, embedding_dim)
        )
    
    def forward(self, x):
        x = self.prejector_seq(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x = self.projector_dim(x)
        return x

class VARForSkeleton(VARBase):
    def __init__(
        self, 
        vae_local: VQVAEBase, 
        depth=16, 
        embed_dim=1024, 
        num_heads=16, 
        mlp_ratio=4., 
        drop_rate=0.,  
        patch_nums=(102, 288, 576), 
        inp_seq_len=17,
        inp_feature_dim=2,
        sos_method="linear", # linear or quantizer
        inp_patch_cnt = 2,
        ar_mode="scale",  # "scale" for predict-by-scale, "token" for token-by-token AR
        upsampler="vae",  # "vae" for get_next_autoregressive_input, "linear" for ablation
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        assert upsampler in ("vae", "linear"), f"upsampler must be 'vae' or 'linear', got {upsampler!r}"
        self.Cvae, self.V = vae_local.embedding_dim, vae_local.vocab_size
        self.depth, self.C, self.num_heads = depth, embed_dim,  num_heads
        self.sos_method = sos_method
        self.inp_patch_cnt = inp_patch_cnt
        assert ar_mode in ("scale", "token"), f"ar_mode must be 'scale' or 'token', got {ar_mode!r}"
        self.ar_mode = ar_mode
        self.upsampler = upsampler
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn for pn in self.patch_nums)
        self.first_l = self.patch_nums[0]  # Use the patch number of the first level
        self.begin_ends = []
        context_token = self.first_l
        self.context_token = context_token
        self.begin_ends.append((0, context_token))
        cur = context_token
        self.L = sum(pn for pn in self.patch_nums)
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur + pn))
            cur += pn

        self.last_level_pns = self.patch_nums[-1]
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        device = torch.device(f'cuda:{ddp_utils.get_local_rank()}' if torch.cuda.is_available() else 'cpu')
        self.rng = torch.Generator(device=device)
        self.drop_rate = drop_rate

        
        # 1. input (word) embedding
        quant: VectorQuantizerBase = vae_local.quantizer
        self.vae_proxy: Tuple[VQVAEBase] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizerBase] = (quant,)

        self.con_embedding = ProjectorForSparse(feature_dim=inp_feature_dim, embedding_dim=embed_dim, inp_seq_len=inp_seq_len, out_seq_len=self.first_l)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        if self.upsampler == "linear":
            self.linear_upsampler = nn.Linear(self.C, self.C)

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        init_std = math.sqrt(1 / self.C / 3)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        # Use the simplified TransformerBlock
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(
                embed_dim=self.C, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=self.drop_rate
            ) 
            for block_idx in range(depth)
        ])
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([
                torch.full((pn,), i )
                for i, pn in enumerate(self.patch_nums)
            ]
        ).view(1, self.L, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)

        if self.ar_mode == "scale":
            can_attend_mask = (d >= dT).reshape(1, 1, self.L, self.L)
            attn_mask = torch.zeros(1, 1, self.L, self.L, dtype=torch.float32)
            attn_mask.masked_fill_(~can_attend_mask, float('-inf'))
        else:
            attn_mask = torch.triu(
                torch.full((1, 1, self.L, self.L), float('-inf'), dtype=torch.float32),
                diagonal=1,
            )
        self.register_buffer("attn_mask", attn_mask.contiguous())

        print(f"[{self.ar_mode}] attn_mask.shape: {attn_mask.shape}")
        
        # 6. classifier head
        # Modified
        self.head_nm = ResiLinear(self.C, 2*self.C)
        self.head = nn.Linear(2*self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: torch.Tensor):
        h = h_or_h_and_residual
        return self.head(self.head_nm(h.float()))

    # @torch.no_grad()
    def inference(
        self, lr_inp, top_k=1, top_p=0.96, enable_timing=False
    ) -> dict[str, list[torch.Tensor]]:
        """
        only used for inference, on autoregressive model
        Args:
            lr_inp: BxJsxF
            top_k: top-k sampling
            top_p: top-p sampling
            enable_timing: whether to enable per-stage latency timing
        Returns:
            list[torch.Tensor]: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        B = lr_inp.shape[0]

        stage_start_events = []
        stage_end_events = []

        # Initialization - process the sos token, consistent with the forward logic
        with torch.amp.autocast("cuda", enabled=False):
            with torch.no_grad():
                out_quantize_first = self.vae_proxy[0].first_pose_quantize(lr_inp)

            if out_quantize_first is not None and self.sos_method=="quantizer":
                sos = self.word_embed(out_quantize_first["prev_embedding"][0])
            else:
                sos = self.con_embedding(lr_inp)

            sos = sos.expand(B, self.first_l, -1)
            sos = sos + self.lvl_embed(self.lvl_1L[:, :self.first_l].expand(B, -1))

        # Initialize intermediate state
        multi_fhats:list[torch.Tensor] = []
        idx_Bls = []
        logits_BLV = []
        prev_embeddings = []
        next_token_map = sos

        for b in self.blocks:
            b.set_use_cache(True)

        current_seq_len = 0
        prev_linear_token = None
        
        # Multi-stage processing
        for si, pn in enumerate(self.patch_nums):

            if self.ar_mode == "token" and self.upsampler == "linear":
                if enable_timing:
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    stage_start_events.append(start_event)

                is_prefill = si < self.inp_patch_cnt and out_quantize_first is not None
                stage_logits_parts = []
                stage_indices = []
                stage_embeddings = []

                for t in range(pn):
                    if current_seq_len < self.first_l:
                        token_input = sos[:, current_seq_len:current_seq_len+1, :]
                    else:
                        token_input = prev_linear_token

                    position_offset = current_seq_len
                    for b in self.blocks:
                        token_input = b(x=token_input, attn_mask=None, position_offset=position_offset)
                    current_seq_len += 1

                    if not is_prefill:
                        logit_t = self.get_logits(token_input.float())
                        stage_logits_parts.append(logit_t)

                        idx_t = sample_with_top_k_top_p_(
                            logit_t.clone(), rng=self.rng, top_k=int(top_k), top_p=top_p, num_samples=1,
                        )[:, :, 0]
                        stage_indices.append(idx_t)

                        h_t: torch.Tensor = self.vae_quant_proxy[0].embedding(idx_t)
                        stage_embeddings.append(h_t)
                    else:
                        h_t = out_quantize_first["prev_embedding"][si][:, t:t+1, :]

                    prev_linear_token = self.linear_upsampler(self.word_embed(h_t))
                    if current_seq_len < self.L:
                        prev_linear_token = prev_linear_token + self.lvl_embed(
                            self.lvl_1L[:, current_seq_len:current_seq_len+1].expand(B, -1)
                        )

                if is_prefill:
                    h_BJD: torch.Tensor = out_quantize_first["prev_embedding"][si]
                    idx_Bls.append(out_quantize_first["idx_Bl"][si])
                else:
                    logits_BLV.append(torch.cat(stage_logits_parts, dim=1))
                    h_BJD = torch.cat(stage_embeddings, dim=1).reshape(B, pn, self.Cvae)
                    idx_Bls.append(torch.cat(stage_indices, dim=1))
                prev_embeddings.append(h_BJD)

                if enable_timing:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    stage_end_events.append(end_event)
            elif self.ar_mode == "token" and self.upsampler == "vae":
                if enable_timing:
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    stage_start_events.append(start_event)

                is_prefill = si < self.inp_patch_cnt and out_quantize_first is not None
                stage_logits_parts = []
                stage_indices = []
                stage_embeddings = []

                for t in range(pn):
                    token_input = next_token_map[:, t:t+1, :]
                    position_offset = current_seq_len
                    for b in self.blocks:
                        token_input = b(x=token_input, attn_mask=None, position_offset=position_offset)
                    current_seq_len += 1

                    if not is_prefill:
                        logit_t = self.get_logits(token_input.float())
                        stage_logits_parts.append(logit_t)

                        idx_t = sample_with_top_k_top_p_(
                            logit_t.clone(), rng=self.rng, top_k=int(top_k), top_p=top_p, num_samples=1,
                        )[:, :, 0]
                        stage_indices.append(idx_t)

                        h_t: torch.Tensor = self.vae_quant_proxy[0].embedding(idx_t)
                        stage_embeddings.append(h_t)

                if is_prefill:
                    h_BJD: torch.Tensor = out_quantize_first["prev_embedding"][si]
                    idx_Bls.append(out_quantize_first["idx_Bl"][si])
                else:
                    logits_BLV.append(torch.cat(stage_logits_parts, dim=1))
                    h_BJD = torch.cat(stage_embeddings, dim=1).reshape(B, pn, self.Cvae)
                    idx_Bls.append(torch.cat(stage_indices, dim=1))
                prev_embeddings.append(h_BJD)

                f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, multi_fhats, h_BJD
                )
                multi_fhats.append(f_hat)
                if si < len(self.patch_nums) - 1:
                    next_token_map = self.word_embed(next_token_map)
                    pos_start_idx = sum(self.patch_nums[:si+1])
                    next_token_map = next_token_map + self.lvl_embed(
                        self.lvl_1L[:, pos_start_idx:pos_start_idx+self.patch_nums[si+1]].expand(B, -1)
                    )

                if enable_timing:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    stage_end_events.append(end_event)
            elif self.ar_mode == "scale" and self.upsampler == "linear":
                if enable_timing:
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    stage_start_events.append(start_event)

                x_BLC = next_token_map
                position_offset = current_seq_len
                for b in self.blocks:
                    x_BLC = b(x=x_BLC, attn_mask=None, position_offset=position_offset)
                current_seq_len += x_BLC.shape[1]

                if si < self.inp_patch_cnt and out_quantize_first is not None:
                    h_BJD: torch.Tensor = out_quantize_first["prev_embedding"][si]
                    idx_Bls.append(out_quantize_first["idx_Bl"][si])
                else:
                    stage_logits = self.get_logits(x_BLC.float())
                    logits_BLV.append(stage_logits)

                    idx_Bl = sample_with_top_k_top_p_(
                        stage_logits.clone(), rng=self.rng, top_k=int(top_k), top_p=top_p, num_samples=1,
                    )[:, :, 0]
                    h_BJD: torch.Tensor = self.vae_quant_proxy[0].embedding(idx_Bl)
                    h_BJD = h_BJD.reshape(B, pn, self.Cvae)
                    idx_Bls.append(idx_Bl)
                prev_embeddings.append(h_BJD)

                if si < len(self.patch_nums) - 1:
                    next_token_map = F.interpolate(
                        self.linear_upsampler(self.word_embed(h_BJD)).transpose(1, 2),
                        size=self.patch_nums[si + 1], mode='linear', align_corners=True,
                    ).transpose(1, 2)
                    pos_start_idx = sum(self.patch_nums[:si+1])
                    next_token_map = next_token_map + self.lvl_embed(
                        self.lvl_1L[:, pos_start_idx:pos_start_idx+self.patch_nums[si+1]].expand(B, -1)
                    )

                if enable_timing:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    stage_end_events.append(end_event)
            elif self.ar_mode == "scale" and self.upsampler == "vae":
                if enable_timing:
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    stage_start_events.append(start_event)

                x_BLC = next_token_map
                position_offset = current_seq_len
                for b in self.blocks:
                    x_BLC = b(x=x_BLC, attn_mask=None, position_offset=position_offset)
                current_seq_len += x_BLC.shape[1]

                if si < self.inp_patch_cnt and out_quantize_first is not None:
                    h_BJD: torch.Tensor = out_quantize_first["prev_embedding"][si]
                    idx_Bls.append(out_quantize_first["idx_Bl"][si])
                else:
                    stage_logits = self.get_logits(x_BLC.float())
                    logits_BLV.append(stage_logits)

                    idx_Bl = sample_with_top_k_top_p_(
                        stage_logits.clone(), rng=self.rng, top_k=int(top_k), top_p=top_p, num_samples=1,
                    )[:, :, 0]
                    h_BJD: torch.Tensor = self.vae_quant_proxy[0].embedding(idx_Bl)
                    h_BJD = h_BJD.reshape(B, pn, self.Cvae)
                    idx_Bls.append(idx_Bl)
                prev_embeddings.append(h_BJD)

                f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, multi_fhats, h_BJD
                )
                multi_fhats.append(f_hat)
                if si < len(self.patch_nums) - 1:
                    next_token_map = self.word_embed(next_token_map)
                    pos_start_idx = sum(self.patch_nums[:si+1])
                    next_token_map = next_token_map + self.lvl_embed(
                        self.lvl_1L[:, pos_start_idx:pos_start_idx+self.patch_nums[si+1]].expand(B, -1)
                    )

                if enable_timing:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    stage_end_events.append(end_event)
            else:
                raise ValueError(f"Invalid ar_mode: {self.ar_mode} and upsampler: {self.upsampler}")

        for b in self.blocks:
            b.set_use_cache(False)

        stage_latencies_ms = None
        if enable_timing:
            torch.cuda.synchronize()
            stage_latencies_ms = [stage_start_events[i].elapsed_time(stage_end_events[i]) for i in range(len(stage_start_events))]

        logits_BLV = torch.concat(logits_BLV, dim=1)

        out = {
            "multi_fhats": multi_fhats,
            "idx_Bl": idx_Bls,
            "prev_embedding": prev_embeddings,
            "logits_BLV": logits_BLV,
            "stage_latencies_ms": stage_latencies_ms,
        }

        return out

    def forward(self, 
        x_BLC: torch.Tensor|None, 
        lr_inp: torch.Tensor,
        idx_Bl_gt:list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        ATTENTION: This forward method is designed for TRAINING ONLY.
        It uses Teacher Forcing and an attention mask for parallel computation.
        For inference, please use the .inference() method.
        """
        B = lr_inp.shape[0]

        with torch.no_grad():
            if self.upsampler != "linear":
                input_list = torch.split(x_BLC, self.patch_nums, dim=1)
                multi_fhats = []
                input_list = input_list[:-1]
                new_input_list = []
                for si, h_BJD in enumerate(input_list):
                    f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                        si, multi_fhats, h_BJD
                    )
                    multi_fhats.append(f_hat)
                    new_input_list.append(next_token_map)
                input_list = new_input_list
                input_list = torch.cat(input_list, dim=1)

        with torch.amp.autocast("cuda", enabled=False):
            # SOS token
            with torch.no_grad():
                out_quantize_first = self.vae_proxy[0].first_pose_quantize(lr_inp)

            if out_quantize_first is not None and self.sos_method == "quantizer":
                sos = self.word_embed(out_quantize_first["prev_embedding"][0])
            else:
                sos = self.con_embedding(lr_inp)

            sos = sos.expand(B, self.first_l, -1)

            if x_BLC is None:
                x_BLC_with_sos = sos
            elif self.upsampler == "linear":
                gt_word = self.word_embed(x_BLC)
                right_shifted = self.linear_upsampler(gt_word[:, :-1, :])
                x_BLC_with_sos = torch.cat([sos, right_shifted[:, self.first_l - 1:, :]], dim=1)
            else:
                x_BLC_with_sos = torch.cat((sos, self.word_embed(input_list)), dim=1)
            
            # Add level embeddings to the entire sequence
            seq_len = x_BLC_with_sos.shape[1]
            x_BLC_with_sos += self.lvl_embed(self.lvl_1L[:, :seq_len].expand(B, -1))

        for b in self.blocks:
            b.set_use_cache(False)

        # Get the attention mask applicable to the current sequence length
        attn_mask = self.attn_mask[0, 0, :seq_len, :seq_len]

        h = x_BLC_with_sos
        for i, b in enumerate(self.blocks):
            h = b(x=h, attn_mask=attn_mask)

        logits_BLV = self.get_logits(h.float())

        # Concatenate GT tokens from all levels into a long sequence
        idx_Bl_gt_full = torch.cat(idx_Bl_gt, dim=1)

        cross_entropy_loss = F.cross_entropy(
            logits_BLV.reshape(-1, self.vae_proxy[0].vocab_size), 
            idx_Bl_gt_full.reshape(-1)
        )

        loss = cross_entropy_loss

        # idx_Bl_hat should now be based on the model's actual sampling, not the interpolated sequence
        with torch.no_grad():
            idx_BL_from_model = sample_with_top_k_top_p_(
                logits_BLV.clone().detach(),
                rng=self.rng, top_k=1, top_p=0.96, num_samples=1,
            )[:, :, 0]

        idx_Bl_hat = list(torch.split(idx_BL_from_model, self.patch_nums, dim=1))
        if out_quantize_first is not None:
            idx_Bl_hat[:self.inp_patch_cnt] = idx_Bl_gt[:self.inp_patch_cnt]

        out = {
            "loss": loss,
            "cross_entropy_loss": cross_entropy_loss,
            "idx_Bl_hat": idx_Bl_hat
        }

        return out


class ContinuousARForSkeleton(VARBase):
    def __init__(
        self, 
        ae_local, # Pass in HierarchicalAE
        depth=16, 
        embed_dim=1024, 
        num_heads=16, 
        mlp_ratio=4., 
        drop_rate=0.,  
        inp_seq_len=17,
        inp_feature_dim=2,
        sos_method="linear"
    ):
        super().__init__()
        self.ae_proxy = (ae_local,)
        self.sos_method = sos_method
        
        # 1. Automatically compute the sequence length of continuous features per level
        # The AE Encoder output length is: gt_patch_num * tokens_per_joint
        self.patch_nums = tuple(gt * ae_local.tokens_per_joint for gt in ae_local.gt_patch_nums)
        self.L = sum(self.patch_nums)
        self.first_l = self.patch_nums[0]
        
        self.Cvae = ae_local.embedding_dim
        self.C, self.num_heads = embed_dim, num_heads
        
        # 2. Conditional feature (SOS token) embedding
        self.con_embedding = ProjectorForSparse(
            feature_dim=inp_feature_dim, embedding_dim=embed_dim, 
            inp_seq_len=inp_seq_len, out_seq_len=self.first_l
        )
        
        # 3. Dimension projection and position-level embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        
        # 4. Sequence-length upsampler for continuous sequences (expands level i features to level i+1 input)
        self.stage_upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.patch_nums[i], self.patch_nums[i+1]),
                nn.GELU(),
                nn.Linear(self.patch_nums[i+1], self.patch_nums[i+1])
            ) for i in range(len(self.patch_nums)-1)
        ])
        
        # 5. Backbone Transformer Blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(
                embed_dim=self.C, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=drop_rate
            ) for _ in range(depth)
        ])
        
        # 6. Attention mask (Causal Mask ensures autoregression)
        d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        self.register_buffer("lvl_1L", dT[:, 0].contiguous())

        can_attend_mask = (d >= dT).reshape(1, 1, self.L, self.L)
        attn_mask = torch.zeros(1, 1, self.L, self.L, dtype=torch.float32)
        attn_mask.masked_fill_(~can_attend_mask, float('-inf'))
        self.register_buffer("attn_mask", attn_mask.contiguous())
        
        # 7. Prediction head (continuous output, replacing the original vocabulary classification)
        self.head_nm = ResiLinear(self.C, 2*self.C)
        # The output dimension is the autoencoder's continuous feature dimension self.Cvae
        self.head = nn.Linear(2*self.C, self.Cvae) 
        
    def forward(self, gt_embeddings: list[torch.Tensor], lr_inp: torch.Tensor):
        """Training forward pass, uses MSE loss to directly regress continuous features"""
        B = lr_inp.shape[0]

        # 1. Extract conditional SOS
        if self.sos_method == "quantizer":
            sos = self.word_embed(gt_embeddings[0])
        else:
            sos = self.con_embedding(lr_inp).expand(B, self.first_l, -1)
        
        # 2. Prepare the input sequence for parallel training
        upsampled_inputs =[]
        for i in range(len(self.patch_nums) - 1):
            h_i = self.word_embed(gt_embeddings[i])  # Convert to Transformer dimension: [B, pn_i, C]
            h_i_t = h_i.transpose(1, 2)              # [B, C, pn_i]
            h_next_t = self.stage_upsamplers[i](h_i_t) # Sequence-length upsampling
            h_next = h_next_t.transpose(1, 2)        # [B, pn_{i+1}, C]
            upsampled_inputs.append(h_next)
            
        # Concatenate SOS and upsampled features to build the full sequence, and add level embeddings
        x_BLC = torch.cat([sos] + upsampled_inputs, dim=1) # [B, L, C]
        seq_len = x_BLC.shape[1]
        x_BLC += self.lvl_embed(self.lvl_1L[:, :seq_len].expand(B, -1))
        
        # 3. Transformer forward
        h = x_BLC
        attn_mask = self.attn_mask[0, 0, :seq_len, :seq_len]
        for b in self.blocks:
            b.set_use_cache(False)
            h = b(x=h, attn_mask=attn_mask)
            
        # 4. Regression prediction
        pred_embeddings_flat = self.head(self.head_nm(h.float())) # [B, L, Cvae]
        gt_embeddings_flat = torch.cat(gt_embeddings, dim=1)      # [B, L, Cvae]
        
        # 5. Loss computation
        mse_loss = F.mse_loss(pred_embeddings_flat, gt_embeddings_flat)
        pred_embeddings = list(torch.split(pred_embeddings_flat, self.patch_nums, dim=1))
        
        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
            "pred_embeddings": pred_embeddings
        }

    @torch.no_grad()
    def inference(self, lr_inp: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Inference phase adopts incremental autoregressive generation"""
        B = lr_inp.shape[0]

        ae = self.ae_proxy[0]
        level_adj_matrices = [adj.to(lr_inp.device) for adj in ae.adj_matrices]
        gt_0 = ae.encoder[0](lr_inp, level_adj_matrices[0])

        if self.sos_method == "quantizer":
            sos = self.word_embed(gt_0)
        else:
            sos = self.con_embedding(lr_inp).expand(B, self.first_l, -1)

        sos = sos + self.lvl_embed(self.lvl_1L[:, :self.first_l].expand(B, -1))
        
        for b in self.blocks:
            b.set_use_cache(True)
            
        next_token_map = sos
        current_seq_len = 0
        pred_embeddings =[]
        
        # Autoregressive generation layer by layer
        for si, pn in enumerate(self.patch_nums):
            x_BLC = next_token_map
            position_offset = current_seq_len
            
            for b in self.blocks:
                x_BLC = b(x=x_BLC, attn_mask=None, position_offset=position_offset)
            current_seq_len += x_BLC.shape[1]
            
            # Predicted features for the current level
            stage_pred = self.head(self.head_nm(x_BLC.float())) #[B, pn, Cvae]

            if si == 0:
                stage_pred = gt_0
            pred_embeddings.append(stage_pred)
            
            # If not the last level, project and upsample it as input for the next level
            if si < len(self.patch_nums) - 1:
                h_i = self.word_embed(stage_pred)
                h_next_t = self.stage_upsamplers[si](h_i.transpose(1, 2))
                next_token_map = h_next_t.transpose(1, 2)
                
                pos_start_idx = sum(self.patch_nums[:si+1])
                next_token_map += self.lvl_embed(self.lvl_1L[:, pos_start_idx:pos_start_idx+self.patch_nums[si+1]].expand(B, -1))
                
        for b in self.blocks:
            b.set_use_cache(False)
        
        return {
            "pred_embeddings": pred_embeddings
        }