# Towards Efficient and Effective Unlearning of Large Language Models for Recommendation
## Introduction
This is the pytorch implementation of ***E2URec*** proposed in the paper Towards Efficient and Effective Unlearning of Large Language Models for Recommendation.


## Requirements
~~~python
pip install -r requirments.txt
~~~

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


