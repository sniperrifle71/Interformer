#!/bin/sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/borong/data/raw_data/huggingface
model_name=newModel
#15*96
seq_len=1440
#14*96, 1*96
label_len=0
pred_len=96
output_len=1440
patch_len=96
# ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=UTSD

# for subset_rand_ratio in 0.01 0.02 0.03 0.04 0.05 0.1 0.15 0.2 0.25 0.5 0.75 1
for subset_rand_ratio in 0.01 
do
# train
# num_workers = 4
# --ckpt_path $ckpt_path \
torchrun --nnodes=1 --nproc_per_node=4 ../run/run_newModel.py \
  --task_name pretrain \
  --model $model_name \
  --is_training 1 \
  --is_finetuning 0 \
  --seed 1 \
  --root_path ../../../data/raw_data/ETT/ \
  --checkpoints ../../../data/newModel/checkpoints/ \
  --data_path $data.csv \
  --data $data \
  --model_id etth1_sr_$subset_rand_ratio \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 3 \
  --d_layers 3 \
  --factor 3 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 512 \
  --train_epochs 10 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --quantile_flag 1 \
  --quantilies 0.25 0.5 0.75 \
  --theta_dim 500 \
  --use_ims \
  --use_multi_gpu
done
