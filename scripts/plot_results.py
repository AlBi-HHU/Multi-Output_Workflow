import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io

# plot each score of each output for each split

solver = snakemake.config['solver']
separate = snakemake.config['plot_outputs_separately']
imputation = eval(snakemake.wildcards['imputation'])
metric = snakemake.wildcards['metric']

try:
    loop = snakemake.config['repetitions']
    test_percentage = float(snakemake.wildcards['test_percentage'])
    validation_percentage = float(snakemake.wildcards['validation_percentage'])
    if test_percentage % 1 == 0:
        test_percentage = int(test_percentage)
    if validation_percentage % 1 == 0:
        validation_percentage = int(validation_percentage)
    title = 'Subject-Wise'
    if imputation:
        title = 'Record-Wise'
    title += f' BT (repetitions: {loop}, test: {test_percentage} %, val: {validation_percentage} %, {snakemake.wildcards["file"]})'
except:
    loop = snakemake.config['n_splits_outer']
    n_splits_inner = snakemake.config['n_splits_inner']
    title = 'Subject-Wise Nested'
    if imputation:
        title = 'Record-Wise Nested'
    title += f' CV (k = {loop}, k\'= {n_splits_inner}, {snakemake.wildcards["file"]})'

colors = px.colors.qualitative.Plotly

score_info = {}

fig = go.Figure()
for i, s in enumerate(solver):
    scores = []
    for k in range(loop):
        scores.append(np.load(snakemake.input[i*loop+k]))
    scores = np.array(scores)
    score_info[s] = [np.nanmedian(scores), np.nanmean(scores), np.nanstd(scores)]
    if separate:
        for j in range(scores.shape[1]):
            fig.add_trace(
                go.Box(
                    name=s,
                    y=scores[:, j],
                    marker_color=colors[i],
                    offsetgroup='offset' + str(j),
                    showlegend=False,
                    boxmean='sd'
                )
            )
    else:
        fig.add_trace(
            go.Box(
                name=s,
                y=scores.flatten(),
                marker_color=colors[i],
                showlegend=False,
                boxmean='sd'
            )
        )

fig.update_layout(
    title=title,
    title_x=0.5,
    yaxis_title=metric,
    #margin = {'l':40, 'r':40, 't':50, 'b':40} # plotly default {'l':80, 'r':80, 't':100, 'b':80}
)
if separate:
    fig.update_layout(
        boxmode='group'
    )

plotly.io.write_image(fig, snakemake.output[0], format='pdf')

with open(snakemake.output[0][:-3] + 'json', 'w') as f:
    json.dump(score_info, f)