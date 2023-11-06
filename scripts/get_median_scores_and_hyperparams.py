import json
import numpy as np

# if we have a_1, ..., a_n, n is even, we take a_(n/2 + 1) as the median

def median_index(mean_score_list):
    median = np.sort(mean_score_list)[len(mean_score_list)//2]
    return np.where(mean_score_list == median)[0][0]

try:
    loop = snakemake.config['repetitions']
    test_percentage = float(snakemake.wildcards['test_percentage'])/100
    validation_percentage = float(snakemake.wildcards['validation_percentage'])/100

except:
    loop = snakemake.config['n_splits_outer']
    
mean_score_list = []
median_scores_list = []
median_hyperparams_list = []
for k in range(loop):
    scores = np.load(snakemake.input[k])
    median_scores_list.append(scores)
    mean_score = np.nanmean(scores)
    mean_score_list.append(mean_score)
    with open(snakemake.input[loop+k], 'r') as f:
        hyperparams = json.load(f)
        median_hyperparams_list.append(hyperparams)
median_index = median_index(mean_score_list)
mean_score = mean_score_list[median_index]
median_scores = median_scores_list[median_index]
median_hyperparams = median_hyperparams_list[median_index]

median_scores_and_hyperparams = median_hyperparams
median_scores_and_hyperparams['mean_score'] = mean_score
median_scores_and_hyperparams['scores'] = list(median_scores)
with open(snakemake.output[0], 'w') as f:
    json.dump(median_scores_and_hyperparams, f)