#!/bin/bash

# ML
# train random label
CUDA_VISIBLE_DEVICES=0 accelerate launch unlearning_rl.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--train_data 'train/retain_0.2_user_10_simple.json' \
--forget_data 'train/forget_0.2_user_10_simple.json'  \
--language_model_path 'checkpoint/ml-1m-base-original-0.0005' \
--checkpoint 'label_models/ml-1m-base-0.2-label-5e-5-weight0.2' \
--lr 5e-5 --batch_size 16 --weight 0.2 > ml-1m-base-0.2-label-5e-5-weight0.2.log

# evaluate
CUDA_VISIBLE_DEVICES=0 accelerate launch evaluate.py \
--data_dir 'datasets/ml-1m/benchmark_proc_data/data/' \
--forget_data train/forget_0.2_user_10_simple.json  \
--test_data test/test_10_simple.json  \
--teacher_path checkpoint/ml-1m-base-0.2-0.0005/model.pt \
--checkpoint label_models/ml-1m-base-0.2-label-5e-5-weight0.2/model_2400.pt
