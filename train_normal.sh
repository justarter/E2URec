#!/bin/bash

# Movielens
# normal train
CUDA_VISIBLE_DEVICES=1 accelerate launch normal_train.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--train_data 'train/train_10_simple.json' \
--valid_data 'valid/valid_10_simple.json' \
--test_data 'test/test_10_simple.json' \
--checkpoint 'checkpoint/ml-1m-base-original-0.0005' \
--lr 0.0005 > ml-1m-base-original-0.0005.log

# forget 0.2 and retrain
CUDA_VISIBLE_DEVICES=0 accelerate launch normal_train.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--train_data 'train/retain_0.2_user_10_simple.json' \
--valid_data 'valid/valid_10_simple.json' \
--test_data 'test/test_10_simple.json' \
--checkpoint 'checkpoint/ml-1m-base-0.2-0.0001' \
--lr 0.0001 > ml-1m-base-0.2-0.0001.log

# evaluate
CUDA_VISIBLE_DEVICES=0 accelerate launch evaluate.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--forget_data train/forget_0.2_user_10_simple.json  \
--test_data test/test_10_simple.json  \
--teacher_path checkpoint/ml-1m-base-0.2-0.0005/model.pt \
--checkpoint 'checkpoint/ml-1m-base-0.2-0.0005/model.pt'

CUDA_VISIBLE_DEVICES=0 accelerate launch evaluate.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--forget_data train/forget_0.2_user_10_simple.json  \
--test_data test/test_10_simple.json  \
--teacher_path checkpoint/ml-1m-base-0.2-0.0005/model.pt \
--checkpoint 'checkpoint/ml-1m-base-original-0.0005/model.pt'