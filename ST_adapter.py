
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdapterBlock(nn.Module):
    def __init__(self, embed_dim=384, fusion_net='linear'):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.fusion_net = fusion_net
        self.total_dim = self.embed_dim

        if self.fusion_net == 'linear':
            self.adapter_linear = nn.Linear(self.total_dim, embed_dim, bias=True)
            self.adapter_activation = nn.Tanh()
            self.adapter_layer_norm1 = nn.LayerNorm(embed_dim)
        elif self.fusion_net == 'mlp2':
            self.adapter_linear1 = nn.Linear(embed_dim, embed_dim, bias=True)
            self.adapter_linear2 = nn.Linear(embed_dim, embed_dim, bias=True)
            self.adapter_activation = nn.Tanh()
            self.adapter_layer_norm1 = nn.LayerNorm(embed_dim)
        elif self.fusion_net == 'mlp3':
            self.adapter_linear1 = nn.Linear(embed_dim, embed_dim, bias=True)
            self.adapter_linear2 = nn.Linear(embed_dim, embed_dim, bias=True)
            self.adapter_linear3 = nn.Linear(embed_dim, embed_dim, bias=True)
            self.adapter_activation = nn.Tanh()
            self.adapter_layer_norm1 = nn.LayerNorm(embed_dim)
        else:
            raise NotImplementedError(f"Fusion net {self.fusion_net} not implemented yet.")

    def forward(self, inp):
        if self.fusion_net == 'linear':
            feats = self.adapter_linear(inp)  # torch.Size([bs, 196, 768])
            feats = self.adapter_activation(feats) 
            output = self.adapter_layer_norm1(feats) # layer norm
        elif self.fusion_net == 'mlp2':
            feats = self.adapter_linear1(inp)  # torch.Size([bs, 196, 768])
            feats = self.adapter_activation(feats)
            feats = self.adapter_linear2(feats)  # torch.Size([bs, 196, 768])
            output = self.adapter_layer_norm1(feats) # layer norm
        elif self.fusion_net == 'mlp3':
            feats = self.adapter_linear1(inp)  # torch.Size([bs, 196, 768])
            feats = self.adapter_activation(feats)
            feats = self.adapter_linear2(feats)  # torch.Size([bs, 196, 768])
            feats = self.adapter_activation(feats)
            feats = self.adapter_linear3(feats)  # torch.Size([bs, 196, 768])
            output = self.adapter_layer_norm1(feats) # layer norm
        else:
            raise NotImplementedError(f"Fusion net {self.fusion_net} not implemented yet.")
        
        return output


class SpatioTemporalAdapter(nn.Module):
    def __init__(self, embed_dim=384, fusion_net='linear'):
        super().__init__()
        self.adapter = AdapterBlock(embed_dim, fusion_net)

    def forward(self, inp):
        output = self.adapter(inp)
        return output
