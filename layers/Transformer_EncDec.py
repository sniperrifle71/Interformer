import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))# 矩阵乘法

def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))# 矩阵乘法

def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class SeasonalityLayer(nn.Module):

    def __init__(self, theta_dim, device, backcast_length=10, forecast_length=5):
        super().__init__()
        self.device = device
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.theta_dim = theta_dim
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        self.theta_fc = nn.Linear(self.backcast_length, self.theta_dim)

    def forward(self, x):
        thetas = self.theta_fc(x)
        B, L, D = x.shape
        x = x.reshape(B * D, L, 1)
        backcast = seasonality_model(thetas, self.backcast_linspace, self.device)
        forecast = seasonality_model(thetas, self.forecast_linspace, self.device)
        backcast = backcast.reshape(B, self.backcast_length, D)
        forecast = forecast.reshape(B, self.forecast_length, D)
        x = torch.cat([backcast, forecast], dim=1)
        return x
    
class TrendLayer(nn.Module):
    def __init__(self, theta_dim, device, backcast_length=10, forecast_length=5):
        super().__init__()
        self.device = device
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.theta_dim = theta_dim
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        self.theta_fc = nn.Linear(self.backcast_length, self.theta_dim)

    def forward(self, x):
        thetas = self.theta_fc(x)
        B, L, D = x.shape
        x = x.reshape(B * D, L, 1)
        backcast = trend_model(thetas, self.backcast_linspace, self.device)
        forecast = trend_model(thetas, self.forecast_linspace, self.device)
        backcast = backcast.reshape(B, self.backcast_length, D)
        forecast = forecast.reshape(B, self.forecast_length, D)
        x = torch.cat([backcast, forecast], dim=1)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", device=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.device = device
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # new_x: [Batch, Query Length, d_model]
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # y: [Batch, Query Length, d_model]

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, interpreter_layers = None,conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.interpreter_layers = nn.ModuleList(interpreter_layers) if interpreter_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, interpreter_layer, conv_layer) in enumerate(zip(self.attn_layers, self.interpreter_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = interpreter_layer(x)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
