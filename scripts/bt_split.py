import math
import json
import numpy as np
import pandas as pd

x = pd.read_csv(snakemake.input[0], index_col=0).to_numpy()
y = pd.read_csv(snakemake.input[1], index_col=0).to_numpy()

repetitions = snakemake.config['repetitions']
test_percentage = float(snakemake.wildcards['test_percentage'])/100
validation_percentage = float(snakemake.wildcards['validation_percentage'])/100
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

test = math.ceil(index_length*test_percentage) # ceil to prevent empty test set
val = math.ceil((index_length - test)*validation_percentage) # ceil to prevent empty validation set
for k in range(repetitions):
    np.random.shuffle(indices)
    outer_indices_k = indices[:test]
    inner_indices_k = indices[test:test+val]
    np.save(snakemake.output[k], outer_indices_k)
    np.save(snakemake.output[repetitions+k], inner_indices_k)