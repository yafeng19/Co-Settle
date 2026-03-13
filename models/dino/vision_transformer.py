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

import torch
import torch.nn as nn

from .utils import trunc_normal_
import torch.nn.functional as F

from ST_adapter import SpatioTemporalAdapter

EPS = 1e-6

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


class Mlp(nn.Module):
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
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 fusion_net='linear', reg_loss='kl',
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_num = img_size[0] // patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.fusion_net = fusion_net
        self.reg_loss = reg_loss

        self.adapter = SpatioTemporalAdapter(embed_dim=embed_dim, fusion_net=self.fusion_net)
        self.critics = nn.CrossEntropyLoss(reduction="none")

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
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
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def interpolate_and_crop_pos_embed(self, pos_embed, interpolate_ratio=0.5):
        bs, num_patches, dim = pos_embed.shape
        origin_size = self.patch_num
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


    def forward_encoder(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embeddings = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embeddings
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]
    
    
    def forward_encoder_asy(self, x, asy_pos_ratio):
        x_asy = x

        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embeddings = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embeddings
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x_out = x[:, 1:]

        x_asy = self.patch_embed(x_asy)
        x_asy = x_asy.flatten(2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_asy = torch.cat((cls_tokens, x_asy), dim=1)
        pos_embeddings = self.interpolate_pos_encoding(x_asy, w, h)
        crop_pos_embed = self.interpolate_and_crop_pos_embed(pos_embeddings[:, 1:, :], asy_pos_ratio)
        crop_pos_embed_with_cls = torch.cat((pos_embeddings[:, :1], crop_pos_embed), dim=1)
        x_asy = x_asy + crop_pos_embed_with_cls
        x_asy = self.pos_drop(x_asy)
        for blk in self.blocks:
            x_asy = blk(x_asy)
        x_asy = self.norm(x_asy)
        x_asy_out = x_asy[:, 1:]

        return x_out, x_asy_out



    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)


    def forward_no_adapter(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
    
    def stoch_mat(self, A, zero_diagonal=False, do_dropout=False, do_sinkhorn=False, temp=0.07):
        ''' Affinity -> Stochastic Matrix '''

        if zero_diagonal:
            A = self.zeroout_diag(A)
        return F.softmax(A/temp, dim=-1)
    

    def affinity(self, x1, x2):
        ndim = x1.ndim
        if ndim < 4:  # add time dimension
            x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        A = torch.einsum('bctn,bctm->btnm', x1, x2)

        return A
    
    def forward_reg_loss(self, target, pred):
        if self.reg_loss == 'mse':
            loss = self.forward_mse_loss(target, pred)
        elif self.reg_loss == 'kl':
            loss = self.forward_kl_loss(target, pred)
        return loss
    
    def forward_mse_loss(self, feat1, feat2):
        loss = F.mse_loss(feat1, feat2)
        return loss
    
    def forward_kl_loss(self, target, pred):
        # target: The target image representation, shape [B, N, D]
        # pred: The predicted video representation, shape [B, N, D]
        return F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')


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

        reg_loss = self.forward_reg_loss(ori_feat_concat, latent_concat)

        loss = cycle_unsup_loss + lambda_reg*reg_loss

        return loss, acc, sim_sum


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
            ret = self.forward_no_adapter(x_lst)
        return ret

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

