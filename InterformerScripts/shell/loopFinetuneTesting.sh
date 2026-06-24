#!/bin/sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/borong/data/raw_data/huggingface
model_name=Interformer
#7*96
seq_len=1440
#14*96, 1*96
label_len=0
# pred_len=24
output_len=1440
patch_len=96
ckpt_path=UTSD-1G_GENERIC
#ckpt_path=UTSD-1G_INTREPETER
#data=ECL
data_list=(N_Abrupt_0.0 N_Gradual_0.0 N_Recurrent_0.0 N_Sudden_0.0 
          N_Abrupt_0.1 N_Gradual_0.1 N_Recurrent_0.1 N_Sudden_0.1
          N_Abrupt_0.2 N_Gradual_0.2 N_Recurrent_0.2 N_Sudden_0.2
          N_Abrupt_0.3 N_Gradual_0.3 N_Recurrent_0.3 N_Sudden_0.3
          N_Abrupt_0.4 N_Gradual_0.4 N_Recurrent_0.4 N_Sudden_0.4
          N_Abrupt_0.5 N_Gradual_0.5 N_Recurrent_0.5 N_Sudden_0.5
            )
subset_rand_ratio=0.01

# for subset_rand_ratio in 0.01 0.02 0.03 0.04 0.05 0.1 0.15 0.2 0.25 0.5 0.75 1
for data in ${data_list[@]}
do
  for pred_len in 96
  do
  # finetune
  # num_workers = 4
  torchrun --nnodes=1 --nproc_per_node=4 ../run/run_interformer.py \
    --ckpt_path $ckpt_path \
    --task_name forecast \
    --model $model_name \
    --is_training 0 \
    --is_finetuning 1 \
    --seed 1 \
    --root_path ../../../data/raw_data/ \
    --checkpoints ../../../data/Interformer/checkpoints/ \
    --data_path $data.csv \
    --data $data \
    --model_id $data_sr_$subset_rand_ratio \
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
    --batch_size 4096 \
    --finetune_epochs 10 \
    --learning_rate 3e-5 \
    --num_workers 4 \
    --patch_len $patch_len \
    --train_test 0 \
    --subset_rand_ratio $subset_rand_ratio \
    --itr 1 \
    --gpu 0 \
    --quantile_flag 1 \
    --theta_dim 500 \
    --quantilies 0.25 0.5 0.75 \
    --output_interpretability \
    --use_ims \
    --use_multi_gpu 
  done
done
