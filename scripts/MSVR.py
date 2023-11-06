import numpy as np
import pandas as pd
import math
from functions import *

def L(y, K, alpha, b, eps):
    e = y - K@alpha - b
    u = np.linalg.norm(e, axis=1)
    Lu = u**2 - eps
    Lu[u**2 - eps < 0] = 0
    return np.sum(Lu)

def Lp(y, K, alpha, b, eps, C):
    return 0.5*np.trace(alpha.T@K@alpha) + C*L(y, K, alpha, b, eps) # C for same setup as SVR

def MSVR(K, y, C, eps2, tol):
    n, p = y.shape
    alpha = np.zeros((n, p))
    b = np.zeros(p)
    Palpha = np.full(alpha.shape, fill_value=np.inf)
    Pb = np.full(b.shape, fill_value=np.inf)
    eta = 1
    while eta > tol:
        e = y - K@alpha - b
        u = np.linalg.norm(e, axis=1)
        Lpk = Lp(y, K, alpha, b, eps2, C)
        a = np.full(u.shape, C)
        a[u**2 - eps2 < 0] = 0
        D = np.diag(a)
        if (a == 0).all(): # if all a_i are 0, we are finished
            break
        if (a == 0).any(): # D is not invertible if any a_i is 0
            l2 = n - np.sum(a == 0)
            z2 = y[a != 0, :]
            a2 = a[a != 0]
            D2 = D[a != 0, :]
            D2 = D2[:, a != 0]
            K2 = K[a != 0, :]
            K2 = K2[:, a != 0]
            A = np.hstack((np.vstack((K2 + np.linalg.inv(D2), a2@K2)), np.vstack((np.ones((l2, 1)), np.sum(a2)))))
            B = np.vstack((z2, a2@z2))
            X = np.linalg.solve(A, B)
            alphas = X[:l2]
            bs = X[l2:]
            for i in np.where(a == 0)[0]:
                alphas = np.insert(alphas, i, 0, axis=0)
        else:
            A = np.hstack((np.vstack((K + np.linalg.inv(D), a@K)), np.vstack((np.ones((n, 1)), np.sum(a)))))
            B = np.vstack((y, a@y))
            X = np.linalg.solve(A, B)
            alphas = X[:n]
            bs = X[n:]
        Palpha = alphas - alpha
        Pb = bs - b
        eta = 1
        while eta > tol:
            Lp_new = Lp(y, K, alpha + eta*Palpha, b + eta*Pb, eps2, C)
            if Lp_new < Lpk:
                break
            eta *= 0.1
        alpha = alpha + eta*Palpha
        b = b + eta*Pb
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
    eps = float(snakemake.wildcards['eps'])
else:
    with open(snakemake.input[3], 'r') as f:
        hyperparams = json.load(f)
    kernel = hyperparams['kernel']
    kernel_param = hyperparams['kernel_param']
    if kernel_param != 'default':
        kernel_param = float(kernel_param)
    C = float(hyperparams['C'])
    eps = float(hyperparams['eps'])

# kick out datapoints with nan features
mask = np.all(np.isnan(x_train), axis=1)
x_train = x_train[~mask]
y_train = y_train[~mask]

K = compute_kernel(kernel, kernel_param, x_train)

# impute nan with column means
y_means = np.nanmean(y_train, axis=0)
nan_indices = np.where(np.isnan(y_train))
y_train[nan_indices] = np.take(y_means, nan_indices[1])

tol = 1e-8 # treat values below tol as 0 for convergence

# to have the same hypervolume as SVR, we change eps to eps2 by solving (2*eps)**p = pi**(p/2)*eps2**p/gamma(p/2+1) with eps2 as the variable
p = y_train.shape[1]
try:
    eps2 = eps*2*math.pi**(-1/2)*math.gamma(p/2+1)**(1/p)
except:
    eps2 = eps

alpha, b = MSVR(K, y_train, C, eps2, tol)

K_test = compute_kernel(kernel, kernel_param, x_test, x_train)
y_pred = K_test@alpha + b

scores = score(y_test, y_pred, y_split, imputation, metric)

np.save(snakemake.output[0], scores)