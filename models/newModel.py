import torch
from torch import nn

from models import TimerBackbone

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.task_name = configs.task_name
        self.ckpt_path = configs.ckpt_path
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout

        self.output_attention = configs.output_attention
    