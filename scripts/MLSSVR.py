import numpy as np
import pandas as pd
from functions import *

def MLSSVR(K, y, C, CC):
    n, p = y.shape
    zeros = np.zeros((n, 1))
    ones = np.ones((n, 1))

    B = np.vstack((np.hstack(([[0]], ones.T)), np.hstack((ones, 1/C*np.eye(n) + (p/CC)*K))))
    Binv = np.linalg.inv(B)
    
    D = np.vstack((np.hstack(([[0]], zeros.T)), np.hstack((zeros, K))))
    G = B + p*D
    Ginv = np.linalg.inv(G)
    
    R = np.vstack((np.zeros((1, p)), y))
    R2 = np.repeat((np.sum(R, axis=1)[:, np.newaxis]), p, axis=1)
    
    X = Binv@R - 1/p*(Binv - Ginv)@R2
    
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
    CC = float(snakemake.wildcards['CC'])
else:
    with open(snakemake.input[3], 'r') as f:
        hyperparams = json.load(f)
    kernel = hyperparams['kernel']
    kernel_param = hyperparams['kernel_param']
    if kernel_param != 'default':
        kernel_param = float(kernel_param)
    C = float(hyperparams['C'])
    CC = float(hyperparams['CC'])

# kick out datapoints with nan features
mask = np.all(np.isnan(x_train), axis=1)
x_train = x_train[~mask]
y_train = y_train[~mask]

K = compute_kernel(kernel, kernel_param, x_train)

# impute NaN with column means
y_means = np.nanmean(y_train, axis=0)
nan_indices = np.where(np.isnan(y_train))
y_train[nan_indices] = np.take(y_means, nan_indices[1])

alpha, b = MLSSVR(K, y_train, C, CC)

K_test = compute_kernel(kernel, kernel_param, x_test, x_train)
_, p = y.shape
alpha2 = np.tile(alpha@np.ones((p, 1)), p)
y_pred = K_test@alpha2 + p/CC*(K_test@alpha) + b.T

scores = score(y_test, y_pred, y_split, imputation, metric)

np.save(snakemake.output[0], scores)