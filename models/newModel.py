import torch
from torch import nn
import sys
sys.path.append('..')
from models import TimerBackbone, newModelBackbone

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.task_name = configs.task_name
        # self.ckpt_path = configs.ckpt_path
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
        self.decoder = self.backbone.decoder
        self.dec_proj = self.backbone.dec_proj
        self.enc_proj = self.backbone.enc_proj
        self.backcast_proj = self.backbone.backcast_proj
        self.enc_embedding = self.backbone.patch_embedding
        self.raw_enc_embedding = self.backbone.raw_patch_embedding
        self.quantile_proj = self.backbone.quantile_proj

        '''
        if self.ckpt_path != '':
            if self.ckpt_path == 'random':
                print('loading model randomly')
            else:
                print('loading model: ', self.ckpt_path)
                if self.ckpt_path.endswith('.pth'):
                    self.backbone.load_state_dict(torch.load(self.ckpt_path))
                elif self.ckpt_path.endswith('.ckpt'):
                    sd = torch.load(self.ckpt_path, map_location="cpu")["state_dict"]
                    sd = {k[6:]: v for k, v in sd.items()}
                    self.backbone.load_state_dict(sd, strict=True)

                else:
                    raise NotImplementedError
        '''

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, M = x_enc.shape # B: batch size, T: time steps, M: number of variables
        # Normalization from Non-stationary Transformer
        raw_x_enc = x_enc.permute(0, 2, 1) 
        raw_enc_in, _ = self.raw_enc_embedding(raw_x_enc) 

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

        # Residual 
        dec_in = enc_in-enc_out
        
        # Reverse Normalization
        dec_in = dec_in * stdev + means

        dec_out = self.decoder(dec_in, raw_enc_in, x_mask=None, cross_mask=None)
        BM, N, D = dec_out.shape # BM=B*M, N=Number of Patch, D=d_model
        Q = len(self.quantilies)
        dec_out_list = []
        for i in range(N):
            token_quantile_dec_out = self.quantile_proj(dec_out[:, i, :])
            dec_out_list.append(token_quantile_dec_out)
        
        dec_out = torch.stack(dec_out_list, dim=1) # [BM, N, D * Q]
        
        dec_out = dec_out.reshape(B, N, Q, -1) # [BM, N, Q, D]
        # Reshape and Projection
        dec_out = self.dec_proj(dec_out) # [BM, N, Q, L] L: patch_len
        L = self.patch_len # 96
        enc_out = self.enc_proj(enc_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, T, Q).transpose(1, 2).transpose(2, 3)# [B, T, Q, M]
        enc_out = enc_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # 最终输出结果
        enc_out = enc_out * stdev + means
        dec_out = dec_out + enc_out.unsqueeze(-2)
        # dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns, backcast
        return dec_out, backcast # [B, T, Q, M], [B, T, M]
    
if __name__ == '__main__':
    configs = {
        'task_name': 'pretrain',
        'patch_len': 10,
        'd_model': 512,
        'd_ff': 2048,
        'e_layers': 6,
        'd_layers': 6,
        'n_heads': 8,
        'dropout': 0.1,
        'output_attention': False,
        'factor': 5,
        'devices': 'cuda'
    }
    model = Model(configs)
    # 打印模型中的所有参数及其名称
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
