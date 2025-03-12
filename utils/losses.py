# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F
import torch


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result



class forecast_backcast_qt_loss(nn.Module):
    def __init__(self, quantile_list, quantile_flag = False):
        super(forecast_backcast_qt_loss, self).__init__()
        self.forecast_func = nn.MSELoss()
        self.backcast_func = nn.MSELoss()
        self.quantile_flag = quantile_flag
        if quantile_flag:
            self.quantile_list = quantile_list

    def forward(self, forecast: t.Tensor, batch_y: t.Tensor, backcast: t.Tensor = None, batch_x: t.Tensor = None) -> t.Tensor:
        if not self.quantile_flag:
            forecast = torch.squeeze(forecast, dim=2)
            forecast_loss = self.forecast_func(forecast, batch_y)
            if backcast is None or batch_x is None:
                return forecast_loss
            backcast_loss = self.backcast_func(backcast, batch_x)
            return forecast_loss + backcast_loss
        
        else:
            # quantile loss
            quantile_loss = torch.zeros_like(forecast) #(B, T, M)
            backcast_loss = self.backcast_func(backcast, batch_x)
            B, T, Q, M = forecast.shape
            for q, rho in enumerate(self.quantile_list):
                if rho == 0.5:
                    forecast_loss = self.forecast_func(forecast[:, :, q, :], batch_y)
                    continue
                ypred_rho = forecast[:, :, q].view(B, T, -1) #(B, T, M)
    #             loss += (rho*torch.max(torch.zeros_like(yf),yf-ypred_rho) + (1-rho)*torch.max(torch.zeros_like(yf),ypred_rho-yf))            
                e = batch_y - ypred_rho
                quantile_loss += torch.max(rho * e, (rho - 1) * e).unsqueeze(-2)
            quantile_loss = quantile_loss.mean()

            return quantile_loss + backcast_loss + forecast_loss
         