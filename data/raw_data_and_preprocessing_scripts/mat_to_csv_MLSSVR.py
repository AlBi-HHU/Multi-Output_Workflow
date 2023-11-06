import numpy as np
import pandas as pd
import scipy.io

filenames = ['Synthetic'] # ['broomcorn', 'polymer', 'CFD500', 'corn', 'enb', 'Synthetic']
for filename in filenames:
    mat = scipy.io.loadmat('MLSSVR_data/' + filename + '.mat')
    if filename == 'corn':
        x = mat['m5spec']['data'][0, 0]
        y = mat['propvals']['data'][0, 0]
    else:
        x = np.vstack((mat['trnX'], mat['tstX']))
        y = np.vstack((mat['trnY'], mat['tstY']))
    if filename == 'Synthetic':
        filename += '_MLSSVR'
    pd.DataFrame(x).to_csv('../' + filename + '.csv')
    pd.DataFrame(y).to_csv('../' + filename + '_outputs.csv')