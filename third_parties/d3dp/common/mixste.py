## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
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
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
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

class SpatialCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Query from sparse features, Key/Value from dense features
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_sparse, x_dense):
        # x_sparse: (B, N_sparse, C) -> Query
        # x_dense: (B, N_dense, C) -> Key, Value
        B_s, N_s, C = x_sparse.shape
        B_d, N_d, C = x_dense.shape
        
        # Calculate Query
        q = self.q(x_sparse).reshape(B_s, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Calculate Key, Value
        kv = self.kv(x_dense).reshape(B_d, N_d, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B_s, N_s, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class CrossAttentionSTEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_sparse = norm_layer(dim)
        self.norm_dense = norm_layer(dim)
        self.attn = SpatialCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_sparse, x_dense):
        # Apply cross-attention and update sparse features
        x_sparse_updated = self.attn(self.norm_sparse(x_sparse), self.norm_dense(x_dense))
        # Residual connection for sparse features
        x_sparse = x_sparse + self.drop_path(x_sparse_updated)
        # MLP part for sparse features
        x_sparse = x_sparse + self.drop_path(self.mlp(self.norm2(x_sparse)))
        return x_sparse

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
    """
    Processes 144 dense joints and compresses them into a single fixed-dimensional
    pose context vector.
    """    
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
        # global_context = torch.mean(dense_embeddings, dim=1) # -> [B, 512]
        
        # return global_context
        return dense_embeddings

class  MixSTE2(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, num_joints_dense=144,
                 in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, is_train=True,
                 dense_fuse_strategy='concat'):
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
            dense_fuse_strategy (str): strategy to fuse dense features choice: [concat, cross_attention]
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3
        self.is_train=is_train
        self.dense_fuse_strategy = dense_fuse_strategy
        self.num_sparse_joints = num_joints
        self.num_dense_joints = num_joints_dense

        if self.num_dense_joints > 0:
            self.d3_embedding = nn.Sequential(
                nn.Linear(out_dim, out_dim*2),
                nn.GELU(),
                nn.Linear(out_dim*2, out_dim)
            )
            self.d3_joints_dense_embed = nn.Sequential(
                    nn.Linear(num_joints, num_joints),
                    nn.GELU(),
                    nn.Linear(num_joints, num_joints+num_joints_dense),
            )
            self.time_for_d3_embded = nn.Sequential(
                SinusoidalPositionEmbeddings(out_dim*2),
                nn.Linear(out_dim*2, out_dim*2),
                nn.GELU(),
                nn.Linear(out_dim*2, out_dim),
            )

        self.Spatial_patch_to_embedding = nn.Linear(in_chans + 3, embed_dim_ratio)
        if dense_fuse_strategy == "cross_attention":
            self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        elif dense_fuse_strategy == "concat":
            self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints+num_joints_dense, embed_dim_ratio))
        else:
            raise ValueError("dense_fuse_strategy should be one of [concat, cross_attention]")

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

        if dense_fuse_strategy == 'cross_attention':
            self.dense_encoder = DensePoseEncoder(input_dim=2, embed_dim=embed_dim_ratio, num_joints=num_joints_dense)
            self.fusion_cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim_ratio, 
                num_heads=num_heads, 
                dropout=attn_drop_rate,
                batch_first=True
            )
            self.fusion_norm = norm_layer(embed_dim_ratio)

        if dense_fuse_strategy == 'concat':
            self.joints_head = nn.Sequential(
                nn.LayerNorm(num_joints+num_joints_dense),
                nn.Linear(num_joints+num_joints_dense, num_joints)
            ) if num_joints_dense > 0 else nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

    def d3_densification(self, x_3d:torch.Tensor, t):
        # x_3d: [(b h f) n c]
        if self.is_train:
            b, f, n, c = x_3d.shape
            x_3d = rearrange(x_3d, 'b f n c -> (b f) n c')
        else:
            b, h, f, n, c = x_3d.shape
            x_3d = rearrange(x_3d, 'b h f n c -> (b h f) n c')
        if self.num_dense_joints > 0:
            x_3d = x_3d + self.d3_embedding(x_3d)
            x_3d = self.d3_joints_dense_embed(x_3d.transpose(-1, -2)).transpose(-1, -2).contiguous() # [(b h f) c n]
            n = x_3d.shape[-2]
            if self.is_train:
                time_embed:torch.Tensor = self.time_for_d3_embded(t)[:, None, None, :]
                time_embed = time_embed.repeat(1, f, n, 1)
                time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            else:
                time_embed:torch.Tensor = self.time_for_d3_embded(t)[:, None, None, None, :]
                time_embed = time_embed.repeat(1, h, f, n, 1)
                time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x_3d =  x_3d + time_embed
            return x_3d
        else:
            return x_3d


    def fuse_poses(self, x_2d:torch.Tensor, x_3d:torch.Tensor, t):
        if self.is_train:
            b, f, n, c = x_2d.shape
            x_2d_flat = rearrange(x_2d, 'b f n c -> (b f) n c')
        else:
            b, h, f, n, c = x_3d.shape
            x_2d_flat = rearrange(x_2d, 'b h f n c -> (b h f) n c')
        
        x_3d_flat_dense = self.d3_densification(x_3d, t)

        if self.dense_fuse_strategy == "cross_attention":
            if self.is_train:
                x_3d_flat = rearrange(x_3d, 'b f n c -> (b f) n c')
            else:
                x_3d_flat = rearrange(x_3d, 'b h f n c -> (b h f) n c')
            x_2d_sparse_part = x_2d_flat[:, :self.num_sparse_joints, :] # Shape: (BF or BHF, 17, 2)
            x_sparse_in = torch.cat((x_2d_sparse_part, x_3d_flat), dim=-1)
            x_sparse = self.Spatial_patch_to_embedding(x_sparse_in)

            if self.num_dense_joints > 0:
                x_2d_dense_part = x_2d_flat[:, self.num_sparse_joints:self.num_sparse_joints+self.num_dense_joints, :]   # Shape: (BF or BHF, 128, 2)
                x_dense = self.dense_encoder(x_2d_dense_part)
                attn_output, _ = self.fusion_cross_attention(x_sparse, x_dense, x_dense)
                x_out = self.fusion_norm(x_sparse + attn_output)
            else:
                x_out = x_sparse

            x_out = x_out + self.Spatial_pos_embed
        else:
            fuse_pose = torch.cat([x_2d_flat[:, :self.num_sparse_joints+self.num_dense_joints, :], x_3d_flat_dense], dim=-1)
            x_out = self.Spatial_patch_to_embedding(fuse_pose)
            x_out = x_out + self.Spatial_pos_embed
    
        return x_out


    def STE_forward(self, x_2d, x_3d, t):

        if self.is_train:
            b, f, n, _ = x_2d.shape
            x = self.fuse_poses(x_2d, x_3d, t)
            _, n, _ = x.shape
            time_embed = self.time_mlp(t)[:, None, None, :]
            time_embed = time_embed.repeat(1, f, n, 1)
            time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            x += time_embed
        else:
            x_2d = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            b, h, f, n, _ = x_2d.shape
            x = self.fuse_poses(x_2d, x_3d, t)
            _, n, _ = x.shape
            time_embed = self.time_mlp(t)[:, None, None, None, :]
            time_embed = time_embed.repeat(1, h, f, n, 1)
            time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x += time_embed

        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

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

    def forward(self, x_2d, x_3d, t):
        if self.is_train:
            b, f, n, c = x_2d.shape
        else:
            b, h, f, n, c = x_3d.shape

        x = self.STE_forward(x_2d, x_3d, t)

        x = self.TTE_foward(x)

        if self.dense_fuse_strategy=='concat':
            n = self.num_sparse_joints + self.num_dense_joints
        elif self.dense_fuse_strategy=='cross_attention':
            n = self.num_sparse_joints
        else:
            raise ValueError('Invalid dense_fuse_strategy')
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)

        if self.dense_fuse_strategy == 'concat':
            x = self.joints_head(x.transpose(-1, -2)).transpose(-1, -2).contiguous()

        x = self.head(x)

        if self.is_train:
            x = x.view(b, f, self.num_sparse_joints, -1)
        else:
            x = x.view(b, h, f, self.num_sparse_joints, -1)

        return x


