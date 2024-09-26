# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from scipy import stats
import Orange
import matplotlib.pyplot as plt
#https://github.com/EGiunchiglia/C-HMCNN/blob/master/friedman_test.py

# %%
csv_path = '/Users/dhruveshpatel/Downloads/scorenn-paper-assets/for-stat-test.csv'
multiplication_factor = -1 #-1 for rank 1-6 decreasing order
# subset_list = ['MBM-T', 'MHM-T', 'C-HMCNN']
# subset_list = ['MVM', 'MHM', 'MBM', 'C-HMCNN', 'Box-E']
subset_list = [
 'CE',
 'SPEN',
 'DVN',
 'NCE ranking',
 'SEAL static margin',
 'SEAL static regression',
 'SEAL static NCE ranking',
 'SEAL dynamic margin',
 'SEAL dynamic regression',
 'SEAL dynamic regression-s',
 'SEAL dynamic NCE ranking']

# %%
data = pd.read_csv(csv_path)
data

# %%
data = data[['Dataset']+subset_list]
data

# %%
model_names = data.columns[1:].to_list()
model_names

# %%
measurements = (data[data.columns[1:]].to_numpy().T)*multiplication_factor
measurements

# %%
rank_data = stats.rankdata(measurements, axis=0)
rank_data

# %%

avranks =  np.mean(rank_data, axis=1)
print(list(zip(model_names, avranks)))
avranks

# %%
array_ranks = [list(rank_data[i, :]) for i in range(rank_data.shape[0])]
print(*array_ranks)

# %%
stats.friedmanchisquare(*array_ranks)

# %%
avranks = list(avranks)

# %%
cd = Orange.evaluation.compute_CD(avranks, 7)
print(cd)
Orange.evaluation.graph_ranks(avranks, model_names, cd=cd, width=6, textspace=1.5, alpha=0.05)
plt.title(f"Based on results across datasets")
plt.savefig(f"nemenyi.pdf", format='pdf' ,bbox_inches='tight',pad_inches=0)

# %%
data

# %%
model_names

# %%
for model1 in ['CE', 'SEAL dynamic NCE ranking']:
    for  model2 in model_names:
        if model1 == model2:
            continue
        print(f"Wilcoxon between {model1} and {model2}")
        print(stats.wilcoxon(data[model1], data[model2], alternative="less"))

# %%
for model1 in ['SEAL static NCE ranking']:
    for  model2 in model_names:
        if model1 == model2:
            continue
        print(f"Wilcoxon between {model1} and {model2}")
        print(stats.wilcoxon(data[model1], data[model2], alternative="less"))

# %%
data['SEAL dynamic NCE ranking']

# %%

# %%

# %%
