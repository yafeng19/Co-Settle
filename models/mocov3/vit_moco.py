# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
import numpy as np
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import to_2tuple
from timm.models.layers import PatchEmbed
import torch.nn.functional as F

from ST_adapter import SpatioTemporalAdapter

__all__ = [
    'vit_small', 
    'vit_base',
    'vit_conv_small',
    'vit_conv_base',
]

EPS = 1e-6
class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, fusion_net='linear', **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()
        self.dist_token = None
        
        self.fusion_net = fusion_net
        self.adapter = SpatioTemporalAdapter(embed_dim=self.embed_dim, fusion_net=self.fusion_net)
        self.critics = nn.CrossEntropyLoss(reduction="none")
        self.patch_num = self.patch_embed.grid_size[0]
        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def get_intermediate_layers(self, x, n=1):
        B, nc, w, h = x.shape
        self.patch_embed.img_size = (h,w)

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i == n:
                return self.norm(x), None

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
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

    def get_embeddings(self, x):
        """Get the class token embeddings."""
        B, nc, w, h = x.shape
        self.patch_embed.img_size = (w,h)
        
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        cls_embeddings = []
        patch_embeddings = []
        for blk in self.blocks:
            x = blk(x)
            _x = self.norm(x)
            cls_embeddings.append(_x[:, 0])
            patch_embeddings.append(_x[:, 1:])
        return cls_embeddings, patch_embeddings  

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
        self.patch_embed.img_size = (w,h)
        
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x[:, 1:]

    def forward_encoder_asy(self, x, asy_pos_ratio):
        x_asy = x

        B, nc, w, h = x.shape
        self.patch_embed.img_size = (w,h)
        
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x_out = x[:, 1:]

        B, nc, w, h = x_asy.shape
        self.patch_embed.img_size = (w,h)
        
        x_asy = self.patch_embed(x_asy)
        cls_token = self.cls_token.expand(x_asy.shape[0], -1, -1)
        
        if self.dist_token is None:
            x_asy = torch.cat((cls_token, x_asy), dim=1)
        else:
            x_asy = torch.cat((cls_token, self.dist_token.expand(x_asy.shape[0], -1, -1), x_asy), dim=1)

        pos_embeddings = self.interpolate_pos_encoding(x_asy, w, h)
        crop_pos_embed = self.interpolate_and_crop_pos_embed(pos_embeddings[:, 1:, :], asy_pos_ratio)
        crop_pos_embed_with_cls = torch.cat((pos_embeddings[:, :1], crop_pos_embed), dim=1)
        x_asy = x_asy + crop_pos_embed_with_cls

        for blk in self.blocks:
            x_asy = blk(x_asy)
        x_asy = self.norm(x_asy)
        x_asy_out = x_asy[:, 1:]

        return x_out, x_asy_out

    def forward_mse_loss(self, feat1, feat2):
        loss = F.mse_loss(feat1, feat2)
        return loss
    
    def forward_kl_loss(self, target, pred):
        # target: The target image representation, shape [B, N, D]
        # pred: The predicted video representation, shape [B, N, D]
        return F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')


    def forward_no_adapter(self, x):
        B, nc, w, h = x.shape
        self.patch_embed.img_size = (w,h)
        
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x
        

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
    

class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model