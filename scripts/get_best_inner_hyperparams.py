import itertools
import json
import numpy as np

params = []
param_names = []
if snakemake.rule.endswith('SVR') or snakemake.rule.endswith('MSVR2'):
    params.append(snakemake.config['kernel'])
    param_names.append('kernel')
    params.append(snakemake.config['kernel_param'])
    param_names.append('kernel_param')
if snakemake.rule.endswith('_SVR'):
    params.append(snakemake.config['C'])
    param_names.append('C')
    params.append(snakemake.config['eps'])
    param_names.append('eps')
if snakemake.rule.endswith('_LSSVR'):
    params.append(snakemake.config['C'])
    param_names.append('C')
if snakemake.rule.endswith('_MSVR'):
    params.append(snakemake.config['C'])
    param_names.append('C')
    params.append(snakemake.config['eps'])
    param_names.append('eps')
if snakemake.rule.endswith('_MSVR2'):
    params.append(snakemake.config['C'])
    param_names.append('C')
    params.append(snakemake.config['eps'])
    param_names.append('eps')
if snakemake.rule.endswith('_ELSSVR'):
    params.append(snakemake.config['C'])
    param_names.append('C')
    params.append(snakemake.config['zeta'])
    param_names.append('zeta')
if snakemake.rule.endswith('_MLSSVR'):
    params.append(snakemake.config['C'])
    param_names.append('C')
    params.append(snakemake.config['CC'])
    param_names.append('CC')
if snakemake.rule.endswith('_ANN'):
    params.append(snakemake.config['num_layers'])
    param_names.append('nl')
    params.append(snakemake.config['hidden_size'])
    param_names.append('hs')
    params.append(snakemake.config['learning_rate'])
    param_names.append('lr')
if snakemake.rule.endswith('_ANNnan'):
    params.append(snakemake.config['num_layers'])
    param_names.append('nl')
    params.append(snakemake.config['hidden_size'])
    param_names.append('hs')
    params.append(snakemake.config['learning_rate'])
    param_names.append('lr')
if snakemake.rule.endswith('_ANNind'):
    params.append(snakemake.config['num_layers'])
    param_names.append('nl')
    params.append(snakemake.config['hidden_size'])
    param_names.append('hs')
    params.append(snakemake.config['learning_rate'])
    param_names.append('lr')

hyperparams = list(itertools.product(*params))
best_mean_score = float('inf')
best_hyperparams_index = 0
for i in range(len(hyperparams)):
    mean_score = np.load(snakemake.input[i])
    if mean_score < best_mean_score:
        best_mean_score = mean_score
        best_hyperparams_index = i
best_inner_hyperparams = {}
for i in range(len(param_names)):
    best_inner_hyperparams[param_names[i]] = hyperparams[best_hyperparams_index][i]
with open(snakemake.output[0], 'w') as f:
    json.dump(best_inner_hyperparams, f)