import sys
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import len, range

def univariate_plotter(feature, input_data, target_col, bins=10):
    data = input_data.copy()

    nan_flag = 0
    if pd.isnull(data[feature]).sum() > 0:
        nan_flag = 1
        print("NANs present")
    # cuts=[0]
    cuts = []
    prev_cut = -1000000000
    if nan_flag != 1:
        for i in range(bins + 1):
            next_cut = np.percentile(data[feature], i * 100 / bins)
            if next_cut != prev_cut:
                cuts.append(next_cut)
            prev_cut = next_cut
        # print(cuts)
        cuts[0] = cuts[0] - 1
        cuts[len(cuts) - 1] = cuts[len(cuts) - 1] + 1
        # cuts=[ -1, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 2]
        cut_series = pd.cut(data[feature], cuts)
    else:
        data_feat_nonan = data[feature].copy()
        data_feat_nonan[np.isinf(data_feat_nonan)] = np.nan
        data_feat_nonan = data_feat_nonan[~np.isnan(data_feat_nonan)]
        data[feature][np.isinf(data[feature])] = np.nan
        data_nonan = data[~np.isnan(data[feature])]

        for i in range(bins + 1):
            next_cut = np.percentile(data_feat_nonan, i * 100 / bins)
            if next_cut != prev_cut:
                cuts.append(next_cut)
            prev_cut = next_cut.copy()

        cuts[0] = cuts[0] - 0.00001
        cuts[len(cuts) - 1] = cuts[len(cuts) - 1] + 0.00001
        # cuts=[ -1, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
        cut_series = pd.cut(data_nonan[feature], cuts)
    if nan_flag != 1:

        grouped = data.groupby([cut_series], as_index=True).agg({target_col: [np.size, np.mean], feature: [np.mean]})
        grouped1 = pd.DataFrame(grouped.index)
        grouped = data.groupby([cut_series], as_index=False).agg({target_col: [np.size, np.mean], feature: [np.mean]})
        grouped.columns = ['_'.join(cols).strip() for cols in grouped.columns.values]

        grouped = pd.DataFrame(grouped.to_records())
        grouped1[target_col+'_mean'] = grouped[target_col+'_mean']
        grouped1[target_col+'_sum'] = grouped[target_col+'_size']
        grouped1[feature + '_mean'] = grouped[feature + '_mean']

    else:
        grouped = data_nonan.groupby([cut_series], as_index=True).agg({target_col: [np.size, np.mean]})
        grouped1 = pd.DataFrame(grouped.index)
        grouped = data_nonan.groupby([cut_series], as_index=False).agg({target_col: [np.size, np.mean]})
        grouped.columns = ['_'.join(cols).strip() for cols in grouped.columns.values]
        grouped = pd.DataFrame(grouped.to_records())
        grouped1[target_col+'_mean'] = grouped[target_col+'_mean']
        grouped1[target_col+'_sum'] = grouped[target_col+'_sum']
        grouped1_nan = grouped1[0:1]
        grouped1_nan[feature] = (grouped1_nan[feature]).astype('str')
        grouped1_nan[feature][0] = 'Nan'
        grouped1_nan[target_col+'_mean'][0] = y_train[np.isnan(data[feature])].mean()
        grouped1_nan[target_col+'_sum'][0] = y_train[np.isnan(data[feature])].sum()
        grouped1 = pd.concat([grouped1, grouped1_nan])

    grouped1 = grouped1.reset_index(drop=True)
    a = plt.plot(grouped1[target_col+'_mean'], marker='o')
    plt.xticks(np.arange(len(grouped1)), (grouped1[feature]).astype('str'), rotation=45)
    plt.xlabel('Bins of ' + feature)
    plt.ylabel('% Incomplete Orders')
    plt.show()
    b = plt.bar(np.arange(len(grouped1)), grouped1[target_col+'_sum'], alpha=0.5)
    plt.xticks(np.arange(len(grouped1)), (grouped1[feature]).astype('str'), rotation=45)
    plt.xlabel('Bins of ' + feature)
    plt.ylabel('Bin-wise Population')
    plt.show()
    return(grouped1)
