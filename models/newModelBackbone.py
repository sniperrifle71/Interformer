import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, SeasonalityLayer, TrendLayer

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        padding = 0

        # patching and embedding
        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, padding, self.dropout)
        self.raw_patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, padding, self.dropout)

        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            interpreter_layers=[
                SeasonalityLayer(theta_dim=100, device = configs.devices, backcast_length=configs.d_model*configs.token_len, forecast_length=configs.d_model*configs.token_len),
                TrendLayer(theta_dim=4, device = configs.devices, backcast_length=configs.d_model*configs.token_len, forecast_length=configs.d_model*configs.token_len)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),

            )
        self.dec_proj = nn.Linear(self.d_model, configs.patch_len, bias=True)
        self.backcast_proj = nn.Linear(self.d_model, configs.patch_len, bias=True)
        self.enc_proj = nn.Linear(self.d_model, configs.patch_len, bias=True)
        self.quantile_proj = nn.Linear(self.d_model, self.d_model*len(configs.quantilies), bias=True)





        