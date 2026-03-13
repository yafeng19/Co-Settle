# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import numpy as np
import torch.nn.functional as F

from ST_adapter import SpatioTemporalAdapter

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .utils import trunc_normal_
EPS = 1e-6

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        norm1 = self.norm1(x)
        y, attn = checkpoint(self.attn, norm1)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        norm2 = self.norm2(x)
        x = x + self.drop_path(checkpoint(self.mlp, norm2))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        apply_stop=False,
        noise_var=0.25,
        learned_pos_emb=False,
        **kwargs
    ):
        super().__init__()
        if depth == 12:
            num_heads = 16
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --

        
        if not learned_pos_emb:
            self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                    requires_grad=False)
            predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                        int(num_patches**.5),
                                                        cls_token=False)
            self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        else:
            self.predictor_pos_embed = nn.Parameter(torch.randn(1, num_patches, predictor_embed_dim) * 0.02, requires_grad=True)
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.init_std = init_std
        self.apply_stop = apply_stop
        self.noise_var = noise_var

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_enc, masks_x, masks):
        assert masks is not None, 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        n_enc_masks = len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x_enc)
        B, N_ctxt, D = x.shape

        # -- add positional embedding to x tokens
        all_mask_keep = []
        for m in masks_x:
            all_mask_keep += [m.unsqueeze(-1).repeat(1, 1, D)]
        all_mask_keep = torch.cat(all_mask_keep, dim=0)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += torch.gather(x_pos_embed, dim=1, index=all_mask_keep)

        # -- concat mask tokens to x
        all_x = []

        for _i, m in enumerate(masks):
            _, N = m.shape
            mask_keep = m.unsqueeze(-1).repeat(n_enc_masks, 1, D)
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = torch.gather(pos_embs, dim=1, index=mask_keep)
            pred_emb = pos_embs

            if self.apply_stop:
                noise = torch.normal(mean=0., std=self.noise_var, size=(pred_emb.shape[0], pred_emb.shape[1], x_enc.shape[-1]), device=pred_emb.device)
                pred_emb += self.predictor_embed(noise)

            all_x += [torch.cat([x, pred_emb], dim=1)]
        x = torch.cat(all_x, dim=0)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

    

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.patch_num = img_size[0] // patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.adapter = SpatioTemporalAdapter(embed_dim=self.embed_dim, fusion_net='linear')
        self.critics = nn.CrossEntropyLoss(reduction="none")
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_and_crop_pos_embed(self, pos_embed, interpolate_ratio=0.5):
        bs, num_patches, dim = pos_embed.shape
        origin_size = self.patch_num    # 14 when use vitb16
        upsample_size = int((1+interpolate_ratio)*self.patch_num)   # 21
        assert num_patches == origin_size * origin_size 
        
        pos_embed_2d = pos_embed.reshape(bs, origin_size, origin_size, dim).permute(0, 3, 1, 2)  # [bs, dim, 14, 14]
        pos_embed_interp = F.interpolate(pos_embed_2d, size=(upsample_size, upsample_size), mode='bicubic', align_corners=False)    # [bs, dim, 21, 21]

        max_offset = upsample_size - origin_size
        offset_h = torch.randint(0, max_offset + 1, (1,)).item()
        offset_w = torch.randint(0, max_offset + 1, (1,)).item()
        pos_embed_crop = pos_embed_interp[:, :, offset_h:offset_h+origin_size, offset_w:offset_w+origin_size]  # [bs, dim, 14, 14]
        pos_embed_final = pos_embed_crop.permute(0, 2, 3, 1).reshape(bs, num_patches, dim)  # [bs, 196, dim]

        return pos_embed_final
    
    def stoch_mat(self, A, zero_diagonal=False, do_dropout=False, do_sinkhorn=False, temp=0.07):
        ''' Affinity -> Stochastic Matrix '''

        if zero_diagonal:
            A = self.zeroout_diag(A)
        return F.softmax(A/temp, dim=-1)

    def forward_asy_cycle_loss(self, feats, win_size=2, temp=0.03, lambda_reg=0.25):
        B, T, N, D = feats.shape
        S = self.patch_num
        assert N == S * S, "N should be square, get {}".format(N)
        feats = feats.view(B*T, S, S, D)    # torch.Size([bs*T, 14, 14, 768])
        feats = feats.permute(0, 3, 1, 2).contiguous()  # torch.Size([bs*T, 768, 14, 14])
        feats = F.avg_pool2d(feats, kernel_size=win_size, stride=win_size)  # torch.Size([bs*T, 768, 7, 7])
        feats = feats.view(B, T, D, -1) # torch.Size([bs, T, 768, 49])
        feats = feats.permute(0, 1, 3, 2)   # torch.Size([bs, T, 49, 768])

        As = self.affinity(feats[:, :-1], feats[:, 1:]) # first T-1 frame compute inner product with last T-1 frame
        forward_sim = [self.stoch_mat(As[:, i], temp=temp, do_dropout=False) for i in range(T-1)]  # 1->2, 2->3, ...  random dropout walk path
        
        forward_chain = forward_sim # lists [1->2,2->1'], length of T-1
        mat_mul_forward = forward_chain[0]
        for mat in forward_chain[1:]:
            mat_mul_forward = mat_mul_forward @ mat # matrix multiply with chain

        mat_mul = mat_mul_forward

        logits = torch.log(mat_mul+EPS).flatten(0, -2)

        I = torch.arange(mat_mul.shape[-1])[None].repeat(mat_mul.shape[0], 1)
        targets = I.view(-1).to(mat_mul.device)

        cycle_loss = self.critics(logits, targets).mean()

        acc = (torch.argmax(logits, dim=-1) == targets).float().mean()

        total_loss = cycle_loss

        # penalty for trivial solution
        identity_matrix = torch.eye(mat_mul.shape[-1], device=mat_mul.device).unsqueeze(0).repeat(B, 1, 1)    # torch.Size([bs, 49, 49]) 
        cosine_sim_forward = F.cosine_similarity(mat_mul_forward.flatten().unsqueeze(0), identity_matrix.flatten().unsqueeze(0), dim=1)

        return total_loss, acc, cosine_sim_forward, forward_sim


    def forward_adapter(self, imgs_lst, use_asy_pos_embedding=True, temp=0.03, asy_pos_ratio=3.0, lambda_reg=0.25):
        imgs_1, imgs_2 = imgs_lst[0], imgs_lst[1]
        imgs_1 = imgs_1.chunk(2, dim=1)  # list of torch.Size([bs, 1, 3, 224, 224])
        imgs_1 = [img.squeeze(1) for img in imgs_1]  # list of torch.Size([bs, 3, 224, 224])
        past_img_1, current_img_1 = imgs_1[0], imgs_1[1]
        imgs_2 = imgs_2.chunk(2, dim=1)  # list of torch.Size([bs, 1, 3, 224, 224])
        imgs_2 = [img.squeeze(1) for img in imgs_2]  # list of torch.Size([bs, 3, 224, 224])
        past_img_2, current_img_2 = imgs_2[0], imgs_2[1]

        assert use_asy_pos_embedding, NotImplementedError("only support use_asy_pos_embedding.")
        with torch.no_grad():
            latent_current_1, latent_asy_current_1 = self.forward_encoder_asy(current_img_1, asy_pos_ratio)
            latent_current_2 = self.forward_encoder(current_img_2)
            
        latent_concat_1 = self.adapter(latent_current_1)
        latent_concat_2 = self.adapter(latent_current_2)
        latent_asy_concat_1 = self.adapter(latent_asy_current_1)

        latent_list = [latent_concat_1.unsqueeze(1), latent_concat_2.unsqueeze(1), latent_asy_concat_1.unsqueeze(1)]
        latent_concat = torch.cat(latent_list, dim=1)

        latent_concat = F.normalize(latent_concat, p=2, dim=-1)
        cycle_unsup_loss, acc, sim_sum, forward_sim = self.forward_asy_cycle_loss(latent_concat, temp=temp, lambda_reg=lambda_reg)   # cycle consistent loss
        # loss = cycle_unsup_loss 

        ori_feat_lst = [latent_current_1.unsqueeze(1), latent_current_2.unsqueeze(1), latent_asy_current_1.unsqueeze(1)]
        ori_feat_concat = torch.cat(ori_feat_lst, dim=1)
        ori_feat_concat = F.normalize(ori_feat_concat, p=2, dim=-1)

        reg_loss = self.forward_kl_loss(ori_feat_concat, latent_concat)

        loss = cycle_unsup_loss + lambda_reg*reg_loss

        return loss, acc, sim_sum

    def affinity(self, x1, x2):
        ndim = x1.ndim
        if ndim < 4:  # add time dimension
            x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)

        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)

        A = torch.einsum('bctn,bctm->btnm', x1, x2)

        return A
    
    def forward_encoder(self, x):

        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)
            
        return x

    def forward_encoder_asy(self, x, asy_pos_ratio):
        x_asy = x

        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        B, N, D = x.shape
        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        x_out = x

        B, nc, w, h = x_asy.shape
        x_asy = self.patch_embed(x_asy)
        B, N, D = x_asy.shape
        # -- add positional embedding to x_asy
        pos_embed = self.interpolate_pos_encoding(x_asy, w, h)
        crop_pos_embed = self.interpolate_and_crop_pos_embed(pos_embed, asy_pos_ratio)
        x_asy = x_asy + crop_pos_embed
        for blk in self.blocks:
            x_asy = blk(x_asy)
        x_asy = self.norm(x_asy)
        x_asy_out = x_asy

        return x_out, x_asy_out


    def forward_kl_loss(self, target, pred):
        # target: The target image representation, shape [B, N, D]
        # pred: The predicted video representation, shape [B, N, D]
        return F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')

    def forward_no_adapter(self, x, return_all_tokens=None, mask=None):
        return self.forward_encoder(x)
    
    def forward_features_eval(self, imgs_lst):
        current_img = imgs_lst.unsqueeze(0)    # [1,3,480,880]
        with torch.no_grad():
            current_img = self.forward_encoder(current_img)
            latent_concat = self.adapter(current_img)  # remove [CLS] token
        return latent_concat
    
    def forward_features_eval_metric(self, imgs_lst):
        with torch.no_grad():
            imgs = self.forward_encoder(imgs_lst)
            latent_concat = self.adapter(imgs)  # remove [CLS] token
        return latent_concat

    def forward(self, x_lst, use_adapter=True, use_asy_pos_embedding=True, temp=0.03, asy_pos_ratio=3.0, lambda_reg=0.5, is_training=False, eval_metric=False, **kwargs):
        if use_adapter:
            if is_training:
                ret = self.forward_adapter(x_lst, use_asy_pos_embedding, temp, asy_pos_ratio, lambda_reg)
            else:
                if eval_metric:
                    ret = self.forward_features_eval_metric(x_lst)
                else:
                    ret = self.forward_features_eval(x_lst)
        else:
            ret = self.forward_no_adapter(x_lst, **kwargs)
        return ret

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] 
        N = self.pos_embed.shape[1] 
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def get_embeddings(self, x):
        """Get the class token embeddings."""
        # Embed patches
        B, nc, w, h = x.shape

        x = self.patch_embed(x)
        B, N, D = x.shape

        pos_embed = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embed

        cls_embeddings = []
        patch_embeddings = []

        # Apply Transformer blocks and collect embeddings
        for blk in self.blocks:
            x = blk(x)
            if self.norm is not None:
                _x = self.norm(x)
            else:
                _x = x

            cls_embeddings.append(_x.mean(dim = 1))
            patch_embeddings.append(_x)

        return cls_embeddings, patch_embeddings 

def deit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def deit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model