import numpy as np
import pandas as pd
from functions import *

def ELSSVR(K, y, C, zeta):
    n, p = y.shape
    
    S = np.tile(np.eye(p), n)
    K = np.repeat(K, p, axis=0)
    K = np.repeat(K, p, axis=1)

    z = np.full((p, p), fill_value=zeta)
    np.fill_diagonal(z, 1)
    K = K*np.tile(z, (n, n))
    KC = K + (1/C)*np.eye(n*p)

    A = np.vstack((np.hstack((np.zeros((p, p)), S)), np.hstack((S.T, KC))))
    B = np.vstack((np.zeros((p, 1)), y.flatten()[:, np.newaxis]))
    X = np.linalg.inv(A)@B
    b = X[:p]
    alpha = X[p:].reshape((n, p))
    
    return alpha, b[:, 0]

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
    zeta = float(snakemake.wildcards['zeta'])
else:
    with open(snakemake.input[3], 'r') as f:
        hyperparams = json.load(f)
    kernel = hyperparams['kernel']
    kernel_param = hyperparams['kernel_param']
    if kernel_param != 'default':
        kernel_param = float(kernel_param)
    C = float(hyperparams['C'])
    zeta = float(hyperparams['zeta'])

# kick out datapoints with nan features
mask = np.all(np.isnan(x_train), axis=1)
x_train = x_train[~mask]
y_train = y_train[~mask]

K = compute_kernel(kernel, kernel_param, x_train)

# impute NaN with column means
y_means = np.nanmean(y_train, axis=0)
nan_indices = np.where(np.isnan(y_train))
y_train[nan_indices] = np.take(y_means, nan_indices[1])

alpha, b = ELSSVR(K, y_train, C, zeta)

K_test = compute_kernel(kernel, kernel_param, x_test, x_train)
y_pred = np.zeros((x_test.shape[0], y_train.shape[1]))
for k in range(y_pred.shape[0]):
    for n in range(y_pred.shape[1]):
        for i in range(y_train.shape[0]):
            for j in range(y_train.shape[1]):
                y_pred[k, n] += alpha[i, j]*K_test[k, i]*(zeta**(n != j))
        y_pred[k, n] += b[n]

scores = score(y_test, y_pred, y_split, imputation, metric)

np.save(snakemake.output[0], scores)