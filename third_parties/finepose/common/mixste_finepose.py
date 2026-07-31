## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import clip
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: [B, N, C] C=2 for (x,y)
        position = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        sin_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        # emb: [N, C]
        x_rotated = torch.cat((-x[..., 1:], x[..., :1]), dim=-1)
        return x * emb.cos() + x_rotated * emb.sin()

class DensePoseEncoder(nn.Module):
    """Encode 144 dense keypoints into fixed-width pose embeddings."""
    def __init__(self, input_dim=2, embed_dim=512, num_joints=144):
        super().__init__()
        self.rope = RotaryEmbedding(dim=input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim), 
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: [B, 144, 2]
        x_with_rope = self.rope(x)
        dense_embeddings = self.encoder(x_with_rope) # -> [B, 144, 512]

        return dense_embeddings



def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        emb_out = self.emb_layers(emb)
        
        # (B, 2*D) -> (B, 1, 2*D)
        emb_out = emb_out.unsqueeze(1)

        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        query = self.query(self.norm(x)).unsqueeze(2)
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        key = key.repeat(int(B/key.shape[0]), 1, 1, 1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)

        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).unsqueeze(1)
        value = value.repeat(int(B/value.shape[0]), 1, 1, 1)
        value = value.view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class  MixSTE2(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True, finetune=False): 
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3
        self.is_train=is_train
        self.finetune = finetune
        
        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans + 3, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        if self.finetune:
            self.dense_encoder = DensePoseEncoder(input_dim=2, embed_dim=embed_dim_ratio, num_joints=144)
            self.fusion_cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim_ratio, 
                num_heads=num_heads, 
                dropout=attn_drop_rate,
                batch_first=True
            )
            self.fusion_norm = norm_layer(embed_dim_ratio)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
            nn.GELU(),
            nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)


        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

        self.temporal_cross_attn = TemporalCrossAttention(512, 512, num_heads, drop_rate, 512)
        self.text_pre_proj = nn.Identity()
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=drop_rate,
            activation="gelu")
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=4)
        self.text_ln = nn.LayerNorm(512)
        self.text_proj = nn.Sequential(
            nn.Linear(512, 512)
        )

        self.clip_text, _ = clip.load('ViT-B/32', "cpu")
        set_requires_grad(self.clip_text, False)

        self.remain_len = 4

        ctx_vectors_subject = torch.empty((7-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_subject, std=0.02)
        self.ctx_subject = nn.Parameter(ctx_vectors_subject)

        ctx_vectors_verb = torch.empty((12-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_verb, std=0.02)
        self.ctx_verb = nn.Parameter(ctx_vectors_verb)

        ctx_vectors_speed = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_speed, std=0.02)
        self.ctx_speed = nn.Parameter(ctx_vectors_speed)

        ctx_vectors_head = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_head, std=0.02)
        self.ctx_head = nn.Parameter(ctx_vectors_head)
        
        ctx_vectors_body = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_body, std=0.02)
        self.ctx_body = nn.Parameter(ctx_vectors_body)
        
        ctx_vectors_arm = torch.empty((14-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_arm, std=0.02)
        self.ctx_arm = nn.Parameter(ctx_vectors_arm)

        ctx_vectors_leg = torch.empty((14-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_leg, std=0.02)
        self.ctx_leg = nn.Parameter(ctx_vectors_leg)


    def STE_forward(self, x_2d, x_3d, t, xf_proj, x_2d_dense=None):
        b_prime, f, n, _ = x_2d.shape
        x = torch.cat((x_2d, x_3d), dim=-1)
        x = rearrange(x, 'b f n c -> (b f) n c') # -> (b_prime*f, 17, 5)
        x_embedded = self.Spatial_patch_to_embedding(x) # -> (b_prime*f, 17, 512)

        # Fuse dense pose context into the base-joint embeddings.
        if self.finetune and x_2d_dense is not None:
            b_d, f_d, n_d, c_d = x_2d_dense.shape
            x_2d_dense_reshaped = x_2d_dense.view(b_d * f_d, n_d, c_d) # -> (b_prime*f, 144, 2)

            dense_embeddings = self.dense_encoder(x_2d_dense_reshaped) 

            query = x_embedded
            key = dense_embeddings
            value = dense_embeddings
            
            # Cross-attention
            attn_output, _ = self.fusion_cross_attention(query, key, value)
            x_embedded = self.fusion_norm(x_embedded + attn_output)

        x_embedded += self.Spatial_pos_embed

        time_embed = self.time_mlp(t) # (b_prime, D_emb)
        time_text_embed = time_embed + xf_proj # (b_prime, D_emb)
        
        time_text_embed_expanded = time_text_embed.repeat_interleave(f, dim=0).unsqueeze(1)
        x_embedded += time_text_embed_expanded

        x = self.pos_drop(x_embedded)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b_prime f) n cw -> (b_prime n) f cw', b_prime=b_prime, f=f)
        
        return x, time_embed

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        return x
    
    def encode_text(self, text, pre_text_tensor):
        with torch.no_grad():
            x = self.clip_text.token_embedding(text).type(self.clip_text.dtype)
            pre_text_tensor = self.clip_text.token_embedding(pre_text_tensor).type(self.clip_text.dtype)

        learnable_prompt_subject = self.ctx_subject
        learnable_prompt_subject = learnable_prompt_subject.view(1, self.ctx_subject.shape[0], self.ctx_subject.shape[1])
        learnable_prompt_subject = learnable_prompt_subject.repeat(x.shape[0], 1, 1)
        learnable_prompt_subject = torch.cat((learnable_prompt_subject, pre_text_tensor[:, 0, :self.remain_len, :]), dim=1)

        learnable_prompt_verb = self.ctx_verb
        learnable_prompt_verb = learnable_prompt_verb.view(1, self.ctx_verb.shape[0], self.ctx_verb.shape[1])
        learnable_prompt_verb = learnable_prompt_verb.repeat(x.shape[0], 1, 1)
        learnable_prompt_verb = torch.cat((learnable_prompt_verb, x[:, :self.remain_len, :]), dim=1)

        learnable_prompt_speed = self.ctx_speed
        learnable_prompt_speed = learnable_prompt_speed.view(1, self.ctx_speed.shape[0], self.ctx_speed.shape[1])
        learnable_prompt_speed = learnable_prompt_speed.repeat(x.shape[0], 1, 1)
        learnable_prompt_speed = torch.cat((learnable_prompt_speed, pre_text_tensor[:, 1, :self.remain_len, :]), dim=1)

        learnable_prompt_head = self.ctx_head
        learnable_prompt_head = learnable_prompt_head.view(1, self.ctx_head.shape[0], self.ctx_head.shape[1])
        learnable_prompt_head = learnable_prompt_head.repeat(x.shape[0], 1, 1)
        learnable_prompt_head = torch.cat((learnable_prompt_head, pre_text_tensor[:, 2, :self.remain_len, :]), dim=1)

        learnable_prompt_body = self.ctx_body
        learnable_prompt_body = learnable_prompt_body.view(1, self.ctx_body.shape[0], self.ctx_body.shape[1])
        learnable_prompt_body = learnable_prompt_body.repeat(x.shape[0], 1, 1)
        learnable_prompt_body = torch.cat((learnable_prompt_body, pre_text_tensor[:, 3, :self.remain_len, :]), dim=1)

        learnable_prompt_arm = self.ctx_arm
        learnable_prompt_arm = learnable_prompt_arm.view(1, self.ctx_arm.shape[0], self.ctx_arm.shape[1])
        learnable_prompt_arm = learnable_prompt_arm.repeat(x.shape[0], 1, 1)
        learnable_prompt_arm = torch.cat((learnable_prompt_arm, pre_text_tensor[:, 4, :self.remain_len, :]), dim=1)

        learnable_prompt_leg = self.ctx_leg
        learnable_prompt_leg = learnable_prompt_leg.view(1, self.ctx_leg.shape[0], self.ctx_leg.shape[1])
        learnable_prompt_leg = learnable_prompt_leg.repeat(x.shape[0], 1, 1)
        learnable_prompt_leg = torch.cat((learnable_prompt_leg, pre_text_tensor[:, 5, :self.remain_len, :]), dim=1)

        x = torch.cat((learnable_prompt_subject, learnable_prompt_verb, learnable_prompt_speed, learnable_prompt_head, learnable_prompt_body, learnable_prompt_arm, learnable_prompt_leg), dim=1)

        with torch.no_grad():
            x = x + self.clip_text.positional_embedding.type(self.clip_text.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_text.transformer(x)
            x = self.clip_text.ln_final(x).type(self.clip_text.dtype)

        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def forward(self, x_2d, x_3d, t, text, pre_text_tensor, x_2d_dense=None):
        if self.is_train:
            b, f, n, _ = x_2d.shape
            num_joints = n
            b_prime = b
        else:
            b, h, f, n, _ = x_3d.shape
            num_joints = n
            b_prime = b * h

            # Keep proposal expansion in the same B-major order as x_3d.view(B*H, ...).
            x_2d = x_2d.repeat_interleave(h, dim=0)
            x_3d = x_3d.view(b_prime, f, n, -1)
            t = t.repeat_interleave(h)
            text = text.repeat_interleave(h, dim=0)
            pre_text_tensor = pre_text_tensor.repeat_interleave(h, dim=0)
            if x_2d_dense is not None:
                x_2d_dense = x_2d_dense.repeat_interleave(h, dim=0)

        xf_proj, xf_out = self.encode_text(text, pre_text_tensor)

        x, time_embed = self.STE_forward(x_2d, x_3d, t, xf_proj, x_2d_dense=x_2d_dense)

        # (b_prime, D) -> (b_prime, 1, D)
        time_embed_unsqueezed = time_embed.unsqueeze(1)
        # (b_prime, 1, D) -> (b_prime, n, D)
        time_embed_repeated = time_embed_unsqueezed.repeat(1, num_joints, 1)
        # (b_prime, n, D) -> (b_prime * n, D)
        time_embed_expanded_for_attn = time_embed_repeated.view(b_prime * num_joints, -1)

        x = self.temporal_cross_attn(x, xf_out, time_embed_expanded_for_attn)
        
        # TTE
        x = self.TTE_foward(x)
        x = rearrange(x, '(b_prime n) f cw -> b_prime f n cw', b_prime=b_prime, n=num_joints)

        # ST Blocks
        x = self.ST_foward(x)
        
        x = self.head(x)

        if not self.is_train:
            x = x.view(b, h, f, num_joints, -1)

        return x