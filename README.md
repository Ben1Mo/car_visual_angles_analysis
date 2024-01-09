# code_challenge
The material for solving the code challenge use case.

## Approach description
In order to answer the coding challenge, a simple vanilla ResNET18 architecture is implemented, trained and tested.\
The two continuous variables: `perspective_score_hood` and `perspective_score_backdoor_left` need to be predicted.\
The problem at hand can be described as a Multi-Outcome regression task, where we train our model to learn how to predict, in this case, two continuous variables.\
The model is evaluated based on a randomly selected set of images (400) and using a term including the Mean Absolute Error (MAE) to calculate an accuracy score of how much of the predictions are close enough to the actual values (targets).


## Running the code
The `deep_multi_output_regression.ipynb` includes all the needed material for the coding challenge. We first install the requirements by running the following:

```bash
pip install -r requirements.txt
```

The we can launch the jupyter notebook launching the following comment:
```bash
jupyter-notebook deep_multi_output_regression.ipynb
```

## Hardware setting:
The following hardware is used to train and test the model:
```
GPU: 			NVIDIA GeForce GTX 1050 Ti with Max-Q Design
Total Dedicated Memory: 4042 MB
NVIDIA Driver Version:  470.223

Model name:          Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
CPU(s):              12
Thread(s) per core:  2
NUMA node(s):        1
```

## Refferences:
1. https://arxiv.org/abs/1512.03385
2. https://pytorch.org/docs/stable/index.html
3. https://lightning.ai/docs/pytorch/stable/
4. https://pandas.pydata.org/docs/
5. https://seaborn.pydata.org/tutorial/introduction
