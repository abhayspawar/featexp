import sys
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import len, range

def get_grouped_data(input_data, feature, target_col, bins, cuts=0, has_null=0, is_train=False):
    if has_null == 1:
        data_null = input_data[pd.isnull(input_data[feature])]
        input_data = input_data[~pd.isnull(input_data[feature])]
        input_data.reset_index(inplace=True, drop=True)

    if is_train:
        prev_cut = min(input_data[feature]) - 1
        cuts = [prev_cut]
        for i in range(1, bins + 1):
            next_cut = np.percentile(input_data[feature], i * 100.0 / bins)
            if next_cut != prev_cut:
                cuts.append(next_cut)
            else:
                print('Reduced the number of bins due to less variation in feature')
            prev_cut = next_cut

        cut_series = pd.cut(input_data[feature], cuts)
    else:
        cut_series = pd.cut(input_data[feature], cuts)

    grouped = input_data.groupby([cut_series], as_index=True).agg(
        {target_col: [np.size, np.mean], feature: [np.mean]})
    grouped.columns = ['_'.join(cols).strip() for cols in grouped.columns.values]
    grouped[grouped.index.name] = grouped.index
    grouped.reset_index(inplace=True, drop=True)
    grouped = grouped[[feature] + list(grouped.columns[0:3])]
    grouped = grouped.rename(index=str, columns={target_col + '_size': 'Samples_in_bin'})
    grouped = grouped.reset_index(drop=True)
    corrected_bin_name = '[' + str(min(input_data[feature])) + ', ' + str(grouped.loc[0, feature]).split(',')[1]
    grouped[feature] = grouped[feature].astype('category')
    grouped[feature] = grouped[feature].cat.add_categories(corrected_bin_name)
    grouped.loc[0, feature] = corrected_bin_name

    if has_null == 1:
        grouped_null = grouped.loc[0:0, :].copy()
        grouped_null[feature] = grouped_null[feature].astype('category')
        grouped_null[feature] = grouped_null[feature].cat.add_categories('Nulls')
        grouped_null.loc[0, feature] = 'Nulls'
        grouped_null.loc[0, 'Samples_in_bin'] = len(data_null)
        grouped_null.loc[0, target_col + '_mean'] = data_null[target_col].mean()
        grouped_null.loc[0, feature + '_mean'] = np.nan
        grouped = pd.concat([grouped_null, grouped], axis=0)
        grouped.reset_index(inplace=True, drop=True)

    return (cuts, grouped)


def draw_plots(input_data, feature, target_col):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(input_data[target_col + '_mean'], marker='o')
    plt.xticks(np.arange(len(input_data)), (input_data[feature]).astype('str'), rotation=45)
    plt.xlabel('Bins of ' + feature)
    plt.ylabel('Average of ' + feature)
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(input_data)), input_data['Samples_in_bin'], alpha=0.5)
    plt.xticks(np.arange(len(input_data)), (input_data[feature]).astype('str'), rotation=45)
    plt.xlabel('Bins of ' + feature)
    plt.ylabel('Bin-wise Population')
    plt.tight_layout()
    plt.show()


def univariate_plotter(feature, data, target_col, bins=10, data_test=0):
    has_null = pd.isnull(data[feature]).sum() > 0

    cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col, bins=bins,
                                     has_null=has_null, is_train=True)
    if type(data_test) == pd.core.frame.DataFrame:
        has_null_test = pd.isnull(data_test[feature]).sum() > 0
        cuts, grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True), feature=feature,
                                              target_col=target_col, bins=bins, has_null=has_null_test, cuts=cuts)
        draw_plots(input_data=grouped, feature=feature, target_col=target_col)
        draw_plots(input_data=grouped_test, feature=feature, target_col=target_col)
    else:
        draw_plots(input_data=grouped, feature=feature, target_col=target_col)

    return (grouped)