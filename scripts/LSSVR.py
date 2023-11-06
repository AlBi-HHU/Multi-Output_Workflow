import numpy as np
import pandas as pd
from functions import *

def LSSVR(K, y, C):
    n = y.shape[0]
    
    ones = np.ones(n)
    KC = K + (1/C)*np.eye(n)

    A = np.vstack((np.hstack((0, ones.T)), np.hstack((ones[:, np.newaxis], KC))))
    B = np.hstack((0, y))
    X = np.linalg.inv(A)@B
    b = X[0]
    alpha = X[1:]
    
    return alpha, b

trial = snakemake.config['trial']
imputation = eval(snakemake.wildcards['imputation'])
metric = snakemake.wildcards['metric']
x = pd.read_csv(snakemake.input[0], index_col=0).to_numpy().astype(float)
y = pd.read_csv(snakemake.input[1], index_col=0).to_numpy().astype(float)
if trial:
    y_min = np.nanmin(y, axis=0)
    y_max = np.nanmax(y, axis=0)
    y = (y - y_min)/(y_max - y_min)
y_split = np.load(snakemake.input[2])

# y_train = outer training set, y_test = test set
x_train, x_test, y_train, y_test = train_test_split(x, y, y_split, imputation)
if snakemake.rule.startswith('train_and_validate'):
    y_split = np.load(snakemake.input[3])
    # y_train = inner training set, y_test = validation set
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, y_split, imputation)
    kernel = snakemake.wildcards['kernel']
    kernel_param = snakemake.wildcards['kernel_param']
    if kernel_param != 'default':
        kernel_param = float(kernel_param)
    C = float(snakemake.wildcards['C'])
else:
    with open(snakemake.input[3], 'r') as f:
        hyperparams = json.load(f)
    kernel = hyperparams['kernel']
    kernel_param = hyperparams['kernel_param']
    if kernel_param != 'default':
        kernel_param = float(kernel_param)
    C = float(hyperparams['C'])

# kick out datapoints with nan features
mask = np.all(np.isnan(x_train), axis=1)
x_train = x_train[~mask]
y_train = y_train[~mask]

y_pred = []
for j in range(y_train.shape[1]):
    y_nan_indices = np.isnan(y_train[:, j])
    x_trainj = x_train[~y_nan_indices]
    Kj = compute_kernel(kernel, kernel_param, x_trainj)
    y_trainj = y_train[:, j][~y_nan_indices]
    alphaj, bj = LSSVR(Kj, y_trainj, C)
    K_testj = compute_kernel(kernel, kernel_param, x_test, x_trainj)
    y_pred.append(K_testj@alphaj + bj)
y_pred = np.array(y_pred).T

scores = score(y_test, y_pred, y_split, imputation, metric)

np.save(snakemake.output[0], scores)
