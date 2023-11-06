import json
import numpy as np
import pandas as pd

def train_test_split(x, y, y_split, imputation):
    p = y.shape[1]
    if imputation:
        row_indices = y_split // p
        col_indices = y_split % p
        x_train = x.copy()
        x_test = x.copy()
        y_train = y.copy()
        y_train[row_indices, col_indices] = np.nan
        y_test = y[row_indices, col_indices]
    else:
        x_train = x.copy()
        x_train[y_split] = np.nan
        x_test = x[y_split]
        y_train = y.copy()
        y_train[y_split] = np.nan
        y_test = y[y_split]
    return x_train, x_test, y_train, y_test

def compute_kernel(kernel, kernel_param, x1, x2=None):
    if kernel == 'rbf':
        if x2 is None:
            x2 = x1
        if kernel_param == 'default':
            x_var = np.var(x2)
            if x_var != 0:
                kernel_param = 1 / (x2.shape[1] * x_var) # like in kernel function of scikit-learn
            else:
                kernel_param = 1
        K = np.exp(-kernel_param*(np.tile(np.sum(x1**2, axis=1), (x2.shape[0], 1)).T + np.tile(np.sum(x2**2, axis=1).T, (x1.shape[0], 1)) - 2*x1@x2.T))
        return K

def metric_score(y_test, y_pred, metric):
    if np.isnan(y_pred).any():
        return float('inf')
    if len(y_test) == 0:
        return np.nan
    
    if metric == 'RMSE':
        rmse = np.sqrt(np.nansum((y_test - y_pred)**2)/np.sum(~np.isnan(y_test)))
        return rmse
    if metric == 'MAE': # not mape because divisions by zero for y_test entries that are 0
        mae = np.nansum(np.abs(y_test - y_pred))/np.sum(~np.isnan(y_test))
        return mae

def score(y_test, y_pred, y_split, imputation, metric, current_j=None):
    p = y_pred.shape[1]
    scores = np.zeros(p)
    if imputation:
        row_indices = y_split // p
        col_indices = y_split % p
        y_pred = y_pred[row_indices, col_indices]
        for j in range(p):
            if current_j is not None and j != current_j: # only needed for ANNind
                scores[j] = 0
            else:
                scores[j] = metric_score(y_test[col_indices == j], y_pred[col_indices == j], metric)
    else:
        for j in range(p):
            if current_j is not None and j != current_j: # only needed for ANNind
                scores[j] = 0
            else:
                scores[j] = metric_score(y_test[:, j], y_pred[:, j], metric)
    return scores