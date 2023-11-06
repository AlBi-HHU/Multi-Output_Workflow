import numpy as np
import pandas as pd

np.random.seed(0)
n = 1000
x = np.sort(np.random.random(n))
y1 = x**2
y2 = 0.5*y1
y12 = np.vstack((y1, y2)).T
pd.DataFrame(x).to_csv('../synthetic_1_2.csv')
pd.DataFrame(y12).to_csv('../synthetic_1_2_outputs.csv')

np.random.seed(0)
y3 = np.sort(np.random.normal(0, 1, n))
y4 = np.sort(np.random.random(n))
y34 = np.vstack((y3, y4)).T
pd.DataFrame(x).to_csv('../synthetic_3_4.csv')
pd.DataFrame(y34).to_csv('../synthetic_3_4_outputs.csv')

y1_outliers = np.copy(y1)
y2_outliers = np.copy(y2)
for i in range(n):
    if i%50 == 49:
        y1_outliers[i] += 10
    if i%100 == 99:
        y2_outliers[i] += 5
y12outliers = np.vstack((y1_outliers, y2_outliers)).T
pd.DataFrame(x).to_csv('../synthetic_1_2_outliers.csv')
pd.DataFrame(y12outliers).to_csv('../synthetic_1_2_outliers_outputs.csv')