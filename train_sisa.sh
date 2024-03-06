#!/bin/bash

# Movielens
# sisa
CUDA_VISIBLE_DEVICES=0 accelerate launch normal_train.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--train_data 'sisa/shared_0_0.2_user_10_simple.json' \
--valid_data 'valid/valid_10_simple.json' \
--test_data 'test/test_10_simple.json' \
--checkpoint 'sisa_models/ml-1m-base-0-0.2-0.0005' --lr 0.0005 > ml-1m-base-0-0.2-0.0005.log

CUDA_VISIBLE_DEVICES=1 accelerate launch normal_train.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--train_data 'sisa/shared_1_0.2_user_10_simple.json' \
--valid_data 'valid/valid_10_simple.json' \
--test_data 'test/test_10_simple.json' \
--checkpoint 'sisa_models/ml-1m-base-1-0.2-0.0005' --lr 0.0005 > ml-1m-base-1-0.2-0.0005.log

CUDA_VISIBLE_DEVICES=0 accelerate launch normal_train.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--train_data 'sisa/shared_2_0.2_user_10_simple.json' \
--valid_data 'valid/valid_10_simple.json' \
--test_data 'test/test_10_simple.json' \
--checkpoint 'sisa_models/ml-1m-base-2-0.2-0.0005' --lr 0.0005 > ml-1m-base-2-0.2-0.0005.log

CUDA_VISIBLE_DEVICES=1 accelerate launch normal_train.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--train_data 'sisa/shared_3_0.2_user_10_simple.json' \
--valid_data 'valid/valid_10_simple.json' \
--test_data 'test/test_10_simple.json' \
--checkpoint 'sisa_models/ml-1m-base-3-0.2-0.0005' --lr 0.0005 > ml-1m-base-3-0.2-0.0005.log

# evaluate
CUDA_VISIBLE_DEVICES=0 accelerate launch sisa_evaluate.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--forget_data train/forget_0.2_user_10_simple.json \
--checkpoint 'sisa_models/' \
--teacher_path 'checkpoint/ml-1m-base-0.2-0.0005/model.pt'

