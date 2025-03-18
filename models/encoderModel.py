import torch
import numpy as np
import torch.nn as nn
import sys, os
sys.path.append('..')

import newModelBackbone

class encoderModel(nn.Module):
    def __init__(self, configs):
        super(encoderModel, self).__init__()
        self.config = configs
        self.backbone = newModelBackbone.Model(configs)
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.quantilies = configs.quantilies
        self.output_attention = configs.output_attention

        self.backbone = newModelBackbone.Model(configs)
        self.encoder = self.backbone.encoder
        self.enc_proj = self.backbone.enc_proj  
        self.backcast_proj = self.backbone.backcast_proj
        self.enc_embedding = self.backbone.patch_embedding
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, M = x_enc.shape # B: batch size, T: time steps, M: number of variables
        # Normalization from Non-stationary Transformer
        raw_x_enc = x_enc.permute(0, 2, 1) 


        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T] #B: batch size, M: number of variables, T: time steps
        enc_in, _= self.enc_embedding(x_enc) # [B * M, N, D] # value_embedding + position_embedding # N: Number of Patch, D: d_model
        # Encoder blocks
        enc_out, attns = self.encoder(enc_in) # [B * M, N, D] 

        # Backcast Block
        backcast = self.backcast_proj(enc_out) # [B * M, N, L], L: patch_len
        backcast = backcast.reshape(B, M, -1).transpose(1, 2)

        return backcast # [B, T, M]

if __name__ == '__main__':
    model = encoderModel(configs)
    for name in model.parameters().keys():
        print(name)