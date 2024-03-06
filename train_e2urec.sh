#!/bin/bash

# e2urec
# first, train the augmented model
CUDA_VISIBLE_DEVICES=0 accelerate launch finetune_aug.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--forget_data train/forget_0.2_user_10_simple.json --lr 5e-5 --batch_size 16 \
--language_model_path checkpoint/ml-1m-base-original-0.0005 \
--checkpoint forget_models/ml-1m-base-0.2-5e-5

# evaluate aug model
CUDA_VISIBLE_DEVICES=0 accelerate launch evaluate.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--forget_data train/forget_0.2_user_10_simple.json  \
--test_data test/test_10_simple.json  \
--teacher_path checkpoint/ml-1m-base-0.2-0.0005/model.pt \
--checkpoint forget_models/ml-1m-base-0.2-5e-5/model_1200.pt

# begin unlearning
CUDA_VISIBLE_DEVICES=0 accelerate launch unlearning_e2urec.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--train_data 'train/retain_0.2_user_10_simple.json' \
--forget_data 'train/forget_0.2_user_10_simple.json' \
--language_model_path 'checkpoint/ml-1m-base-original-0.0005' \
--forget_model_path forget_models/ml-1m-base-0.2-5e-5/model_1200.pt \
--checkpoint 'unlearning_models/ml-1m-base-0.2-lr0.001-weight0.6-lora' \
--lr 0.001 --batch_size 16 --weight 0.6 --alpha 2 > ml-1m-base-0.2-lr0.001-weight0.6-lora.log

# evaluate unlearned model
CUDA_VISIBLE_DEVICES=1 accelerate launch evaluate.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--forget_data train/forget_0.2_user_10_simple.json  \
--test_data test/test_10_simple.json  \
--teacher_path checkpoint/ml-1m-base-0.2-0.0005/model.pt \
--checkpoint unlearning_models/ml-1m-base-0.2-lr0.001-weight0.6-lora/model_1000.pt

