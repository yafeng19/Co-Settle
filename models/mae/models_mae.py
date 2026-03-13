# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block

from util.pos_embed import get_2d_sincos_pos_embed
from ST_adapter import SpatioTemporalAdapter

EPS = 1e-6

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
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x    # torch.Size([bs, N, dim])

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, 
                 fusion_net='linear', reg_loss='kl',
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.embed_dim = embed_dim

        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_num = img_size // patch_size

        self.fusion_net = fusion_net
        self.reg_loss = reg_loss
        
        self.adapter = SpatioTemporalAdapter(embed_dim=self.embed_dim, fusion_net=self.fusion_net)
        self.critics = nn.CrossEntropyLoss(reduction="none")
        self.critics_mse = nn.MSELoss()
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

        
    def interpolate_pos_encoding(self, x, h, w):
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
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


    def interpolate_and_crop_pos_embed(self, pos_embed, interpolate_ratio=0.5):
        origin_size = self.patch_num    # 14
        upsample_size = int((1+interpolate_ratio)*self.patch_num)   # 21
        bs, num_patches, dim = pos_embed.shape
        assert num_patches == origin_size * origin_size 
        
        pos_embed_2d = pos_embed.reshape(bs, origin_size, origin_size, dim).permute(0, 3, 1, 2)  # [bs, dim, 14, 14]
        pos_embed_interp = F.interpolate(pos_embed_2d, size=(upsample_size, upsample_size), mode='bicubic', align_corners=False)    # [bs, dim, 21, 21]

        max_offset = upsample_size - origin_size
        offset_h = torch.randint(0, max_offset + 1, (1,)).item()
        offset_w = torch.randint(0, max_offset + 1, (1,)).item()
        pos_embed_crop = pos_embed_interp[:, :, offset_h:offset_h+origin_size, offset_w:offset_w+origin_size]  # [bs, dim, 14, 14]
        pos_embed_final = pos_embed_crop.permute(0, 2, 3, 1).reshape(bs, num_patches, dim)  # [bs, 196, dim]

        return pos_embed_final


    def forward_encoder_asy(self, x, asy_pos_ratio):
        x_asy = x
        # x_asy = x.clone()

        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # embed patches
        x_asy = self.patch_embed(x_asy)
        # add pos embed w/o cls token
        crop_pos_embed = self.interpolate_and_crop_pos_embed(self.pos_embed[:, 1:, :], asy_pos_ratio)
        x_asy = x_asy + crop_pos_embed
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_asy.shape[0], -1, -1)
        x_asy = torch.cat((cls_tokens, x_asy), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x_asy = blk(x_asy)
        x_asy = self.norm(x_asy)

        return x, x_asy


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
        return A * mask


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


    def forward_asy_cycle_loss(self, feats, win_size=2, temp=0.0, lambda_reg=0.25):
        B, T, N, D = feats.shape
        S = self.img_size // self.patch_size

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
            
        latent_concat_1 = self.adapter(latent_current_1[:, 1:, :])
        latent_concat_2 = self.adapter(latent_current_2[:, 1:, :])
        latent_asy_concat_1 = self.adapter(latent_asy_current_1[:, 1:, :])

        latent_list = [latent_concat_1.unsqueeze(1), latent_concat_2.unsqueeze(1), latent_asy_concat_1.unsqueeze(1)]
        latent_concat = torch.cat(latent_list, dim=1)

        latent_concat = F.normalize(latent_concat, p=2, dim=-1)
        cycle_unsup_loss, acc, sim_sum, forward_sim = self.forward_asy_cycle_loss(latent_concat, temp=temp, lambda_reg=lambda_reg)   # cycle consistent loss
        # loss = cycle_unsup_loss 

        ori_feat_lst = [latent_current_1[:, 1:, :].unsqueeze(1), latent_current_2[:, 1:, :].unsqueeze(1), latent_asy_current_1[:, 1:, :].unsqueeze(1)]
        ori_feat_concat = torch.cat(ori_feat_lst, dim=1)
        ori_feat_concat = F.normalize(ori_feat_concat, p=2, dim=-1)

        reg_loss = self.forward_reg_loss(ori_feat_concat, latent_concat)

        loss = cycle_unsup_loss + lambda_reg*reg_loss

        return loss, acc, sim_sum
    

    def forward_encoder_eval(self, x):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, h, w)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x[:, 1:]


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

    def forward_no_adapter(self, imgs):
        with torch.no_grad():
            output = self.forward_encoder_eval(imgs)
        return output
    

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


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
