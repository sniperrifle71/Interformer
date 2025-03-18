import newModel
import torch
import sys, os
sys.path.append('..')
ckpt_pth = '../../data/newModel/checkpoints/forecast_etth1_sr_0.01_newModel_ETTh1_ftM_sl1440_ll0_pl96_pl96_dm256_nh8_el3_dl3_df512_fc3_ebtimeF_dtTrue_Exp25-03-17_17-15-54/checkpoint.pth'
model = newModel.Model()