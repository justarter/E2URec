# Towards Efficient and Effective Unlearning of Large Language Models for Recommendation
## Introduction
This is the pytorch implementation of ***E2URec*** proposed in the paper [Towards Efficient and Effective Unlearning of Large Language Models for Recommendation](http://arxiv.org/abs/2403.03536). (Frontiers of Computer Science 2024)


## Requirements
~~~python
pip install -r requirments.txt
~~~

## Data preprocess
Scripts for data preprocessing are included in data_preprocess.
First, use ml-1m.ipynb to preprocess MovieLens-1M.
Then, convert data into text
~~~python
python data2json.py --K 10 --temp_type simple --set train --dataset ml-1m
python data2json.py --K 10 --temp_type simple --set valid --dataset ml-1m
python data2json.py --K 10 --temp_type simple --set test --dataset ml-1m
~~~
Finally, use split_ml-1m.ipynb to split train/valid/test, retained/forgotten data.

## How to run E2URec
Our method `E2URec` can be trained by
~~~python
sh train_e2urec.sh
~~~



## How to run baselines
We also provide shell scripts for baselines.

To run the `Retrain` baseline:
~~~python
sh train_normal.sh
~~~
To run the `SISA` baseline:
~~~python
sh train_sisa.sh
~~~
To run the `NegGrad` baseline:
~~~python
sh train_ga.sh
~~~
To run the `Bad-T` baseline:
~~~python
sh train_rl.sh
~~~


