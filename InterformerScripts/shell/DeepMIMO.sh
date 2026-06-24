#!/bin/sh
model_name=Interformer

seq_len=1440
label_len=0
pred_len=96
output_len=1440
patch_len=96

data=DeepMIMO
data_path=DeepMIMO.npz

for subset_rand_ratio in 1
do
torchrun --nnodes=1 --nproc_per_node=1 ../run/run_interformer.py \
  --task_name forecast \
  --model $model_name \
  --is_training 1 \
  --is_finetuning 0 \
  --seed 1 \
  --root_path ../../../data/raw_data/DeepMIMO/ \
  --checkpoints ../../../data/Interformer/checkpoints/ \
  --data_path $data_path \
  --data $data \
  --model_id deepmimo_sr_$subset_rand_ratio \
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
  --batch_size 128 \
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
  --use_ims
done
