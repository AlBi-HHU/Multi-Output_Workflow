import numpy as np

mean_score = 0
c = 0
for file in snakemake.input:
    scores = np.load(file)
    mean_score += np.nansum(scores)
    c += np.count_nonzero(scores == scores)
mean_score /= c
np.save(snakemake.output[0], mean_score)