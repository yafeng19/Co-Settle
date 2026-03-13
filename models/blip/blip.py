import warnings
warnings.filterwarnings("ignore")

from .vit import VisionTransformer, interpolate_pos_embed
# from .med import BertConfig, BertModel, BertLMHeadModel
# from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

def create_vit(vit, image_size, num_classes=0, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    # Map parameter names: remove 'visual_encoder.' prefix and filter out text_encoder
    mapped_state_dict = {}
    for key, value in state_dict.items():
        # Skip text_encoder parameters
        if key.startswith('text_encoder'):
            continue
        # Remove 'visual_encoder.' prefix to match model parameter names
        if key.startswith('visual_encoder.'):
            new_key = key[len('visual_encoder.'):]
            mapped_state_dict[new_key] = value
        else:
            mapped_state_dict[key] = value
    
    # Handle position embedding interpolation
    if 'pos_embed' in mapped_state_dict:
        mapped_state_dict['pos_embed'] = interpolate_pos_embed(mapped_state_dict['pos_embed'], model)
    
    # Remove parameters with shape mismatch
    for key in list(mapped_state_dict.keys()):
        if key in model.state_dict().keys():
            if mapped_state_dict[key].shape != model.state_dict()[key].shape:
                del mapped_state_dict[key]
    
    msg = model.load_state_dict(mapped_state_dict, strict=False)
    print('load checkpoint from {} with msg: {}'.format(url_or_filename, msg))
    return model, msg

