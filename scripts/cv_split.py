import json
import numpy as np
import pandas as pd

x = pd.read_csv(snakemake.input[0], index_col=0).to_numpy()
y = pd.read_csv(snakemake.input[1], index_col=0).to_numpy()

n_splits_outer = snakemake.config['n_splits_outer']
n_splits_inner = snakemake.config['n_splits_inner']
imputation = eval(snakemake.wildcards['imputation'])

n, p = y.shape
np.random.seed(0)

index_length = n
if imputation:
    index_length *= p

indices = np.arange(index_length)
if imputation:
    mask = np.isnan(y.flatten())
    indices = indices[~mask]
    index_length -= np.sum(mask)

np.random.shuffle(indices)
for k in range(n_splits_outer):
    outer_indices_k = indices[(k*index_length)//n_splits_outer:((k+1)*index_length)//n_splits_outer]
    np.save(snakemake.output[k], outer_indices_k)
    indices_k = np.setdiff1d(indices, outer_indices_k, assume_unique=True)
    for k2 in range(n_splits_inner):
        index_length_k = len(indices_k)
        inner_indices_k_k2 = indices_k[(k2*index_length_k)//n_splits_inner:((k2+1)*index_length_k)//n_splits_inner]
        np.save(snakemake.output[n_splits_outer+n_splits_inner*k+k2], inner_indices_k_k2)