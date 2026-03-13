from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from ST_adapter import SpatioTemporalAdapter

EPS = 1e-6

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.patch_num = input_resolution // patch_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        self.adapter = SpatioTemporalAdapter(embed_dim=width, fusion_net='linear')
        self.critics = nn.CrossEntropyLoss(reduction="none")

    
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        pos_embed = self.positional_embedding.float()
        class_pos_embed = pos_embed[0]
        patch_pos_embed = pos_embed[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=True,
            **kwargs,
        )
        class_pos_embed = class_pos_embed.reshape(1, dim)
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
    

    def forward_no_adapter(self, x: torch.Tensor):
        B, nc, w, h = x.shape

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        pos_embeddings = self.interpolate_pos_encoding(x, w, h)
        
        x = x + pos_embeddings.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 1:, :])

        return x
    

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

    
    def forward_encoder(self, x):
        B, nc, w, h = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        pos_embeddings = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embeddings.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 1:, :])

        return x
    
    def forward_encoder_asy(self, x, asy_pos_ratio):
        x_asy = x

        B, nc, w, h = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        pos_embeddings = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embeddings.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_out = self.ln_post(x[:, 1:, :])


        B, nc, w, h = x_asy.shape
        x_asy = self.conv1(x_asy)  # shape = [*, width, grid, grid]
        x_asy = x_asy.reshape(x_asy.shape[0], x_asy.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_asy = x_asy.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_asy = torch.cat([self.class_embedding.to(x_asy.dtype) + torch.zeros(x_asy.shape[0], 1, x_asy.shape[-1], dtype=x_asy.dtype, device=x_asy.device), x_asy], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        pos_embeddings = self.interpolate_pos_encoding(x_asy, w, h)
        crop_pos_embed = self.interpolate_and_crop_pos_embed(pos_embeddings[1:, :].unsqueeze(0), asy_pos_ratio)
        crop_pos_embed_with_cls = torch.cat((pos_embeddings[:1].unsqueeze(0), crop_pos_embed), dim=1)
        x_asy = x_asy + crop_pos_embed_with_cls
        x_asy = self.ln_pre(x_asy)
        x_asy = x_asy.permute(1, 0, 2)  # NLD -> LND
        x_asy = self.transformer(x_asy)
        x_asy = x_asy.permute(1, 0, 2)  # LND -> NLD
        x_asy_out = self.ln_post(x_asy[:, 1:, :])

        return x_out, x_asy_out
    
    
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
    
    def forward_kl_loss(self, target, pred):
        # target: The target image representation, shape [B, N, D]
        # pred: The predicted video representation, shape [B, N, D]
        return F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')


    def forward_asy_cycle_loss(self, feats, win_size=2, temp=0.03, lambda_reg=0.25):
        B, T, N, D = feats.shape
        S = self.patch_num
        assert N == S * S, "N should be square, get {}".format(N)
        assert win_size==1 or win_size==2, "Only support win_size=1 or 2, get {}".format(win_size)
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

    def get_num_layers(self):
        return len(self.blocks)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, use_adapter=True, use_asy_pos_embedding=True, temp=0.03, asy_pos_ratio=3.0, lambda_reg=0.5, is_training=False, eval_metric=False, finetuning=False):
        # print(self.dtype)   # torch.float16
        if not use_adapter:
            return self.visual(image.type(self.dtype), use_adapter=use_adapter, is_training=is_training, eval_metric=eval_metric, finetuning=finetuning)
        else:
            if eval_metric:
                return self.visual(image.type(self.dtype),
                               use_asy_pos_embedding=use_asy_pos_embedding, temp=temp, asy_pos_ratio=asy_pos_ratio, lambda_reg=lambda_reg, is_training=is_training, eval_metric=eval_metric, finetuning=finetuning)
            else:
                return self.visual([x.type(self.dtype) for x in image], 
                               use_asy_pos_embedding=use_asy_pos_embedding, temp=temp, asy_pos_ratio=asy_pos_ratio, lambda_reg=lambda_reg, is_training=is_training, eval_metric=eval_metric, finetuning=finetuning)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load model message: {msg}")
    return model.eval()
