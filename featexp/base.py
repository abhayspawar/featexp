import sys
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt


def get_grouped_data(input_data, feature, target_col, bins, cuts=0):
    """
    Bins continuous features into equal sample size buckets and returns the target mean in each bucket. Separates out
    nulls into another bucket.
    :param input_data: dataframe containg features and target column
    :param feature: feature column name
    :param target_col: target column
    :param bins: Number bins required
    :param cuts: if buckets of certain specific cuts are required. Used on test data to use cuts from train.
    :return: If cuts are passed only grouped data is returned, else cuts and grouped data is returned
    """
    has_null = pd.isnull(input_data[feature]).sum() > 0
    if has_null == 1:
        data_null = input_data[pd.isnull(input_data[feature])]
        input_data = input_data[~pd.isnull(input_data[feature])]
        input_data.reset_index(inplace=True, drop=True)

    is_train = 0
    if cuts == 0:
        is_train = 1
        prev_cut = min(input_data[feature]) - 1
        cuts = [prev_cut]
        reduced_cuts = 0
        for i in range(1, bins + 1):
            next_cut = np.percentile(input_data[feature], i * 100 / bins)
            if next_cut != prev_cut:
                cuts.append(next_cut)
            else:
                reduced_cuts = reduced_cuts + 1
            prev_cut = next_cut

        # if reduced_cuts>0:
        #     print('Reduced the number of bins due to less variation in feature')
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
        grouped[feature] = grouped[feature].astype('str')
        grouped = pd.concat([grouped_null, grouped], axis=0)
        grouped.reset_index(inplace=True, drop=True)

    grouped[feature] = grouped[feature].astype('str').astype('category')
    if is_train == 1:
        return (cuts, grouped)
    else:
        return (grouped)


def draw_plots(input_data, feature, target_col, trend_correlation=None):
    """
    Draws univariate dependence plots for a feature
    :param input_data: grouped data contained bins of feature and target mean.
    :param feature: feature column name
    :param target_col: target column
    :param trend_correlation: correlation between train and test trends of feature wrt target
    :return:
    """
    trend_changes = get_trend_changes(grouped_data=input_data, feature=feature, target_col=target_col)
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(input_data[target_col + '_mean'], marker='o')
    ax1.set_xticks(np.arange(len(input_data)))
    ax1.set_xticklabels((input_data[feature]).astype('str'))
    plt.xticks(rotation=45)
    ax1.set_xlabel('Bins of ' + feature)
    ax1.set_ylabel('Average of ' + target_col)
    comment = "Trend changed " + str(trend_changes) + " times"
    if trend_correlation == 0:
        comment = comment + '\n' + 'Correlation with train trend: NA'
    elif trend_correlation != None:
        comment = comment + '\n' + 'Correlation with train trend: ' + str(int(trend_correlation * 100)) + '%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.05, 0.95, comment, fontsize=12, verticalalignment='top', bbox=props, transform=ax1.transAxes)
    plt.title('Average of ' + target_col + ' wrt ' + feature)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(np.arange(len(input_data)), input_data['Samples_in_bin'], alpha=0.5)
    ax2.set_xticks(np.arange(len(input_data)))
    ax2.set_xticklabels((input_data[feature]).astype('str'))
    plt.xticks(rotation=45)
    ax2.set_xlabel('Bins of ' + feature)
    ax2.set_ylabel('Bin-wise sample size')
    plt.title('Samples in bins of ' + feature)
    plt.tight_layout()
    plt.show()


def get_trend_changes(grouped_data, feature, target_col, threshold=0.03):
    """
    Calculates number of times the trend of feature wrt target changed direction.
    :param grouped_data: grouped dataset
    :param feature: feature column name
    :param target_col: target column
    :param threshold: minimum % difference required to count as trend change
    :return:
    """
    grouped_data = grouped_data.loc[grouped_data[feature] != 'Nulls', :].reset_index(drop=True)
    target_diffs = grouped_data[target_col + '_mean'].diff()
    target_diffs = target_diffs[~np.isnan(target_diffs)].reset_index(drop=True)
    max_diff = grouped_data[target_col + '_mean'].max() - grouped_data[target_col + '_mean'].min()
    target_diffs_mod = target_diffs.abs()
    low_change = target_diffs_mod < threshold * max_diff
    target_diffs_norm = target_diffs.divide(target_diffs_mod)
    target_diffs_norm[low_change] = 0
    target_diffs_norm = target_diffs_norm[target_diffs_norm != 0]
    target_diffs_lvl2 = target_diffs_norm.diff()
    changes = target_diffs_lvl2.abs() / 2
    tot_trend_changes = int(changes.sum()) if ~np.isnan(changes.sum()) else 0
    return (tot_trend_changes)


def get_trend_correlation(grouped, grouped_test, feature, target_col):
    """
    Calculates correlation between train and test trend of feature wrt target.
    :param grouped: train grouped data
    :param grouped_test: test grouped data
    :param feature: feature column name
    :param target_col: target column name
    :return:
    """
    grouped = grouped[grouped[feature] != 'Nulls'].reset_index(drop=True)
    grouped_test = grouped_test[grouped_test[feature] != 'Nulls'].reset_index(drop=True)

    if grouped_test.loc[0, feature] != grouped.loc[0, feature]:
        grouped_test[feature] = grouped_test[feature].cat.add_categories(grouped.loc[0, feature])
        grouped_test.loc[0, feature] = grouped.loc[0, feature]
    grouped_test_train = grouped.merge(grouped_test[[feature, target_col + '_mean']], on=feature, how='left',
                                       suffixes=('', '_test'))
    nan_rows = pd.isnull(grouped_test_train[target_col + '_mean']) | pd.isnull(
        grouped_test_train[target_col + '_mean_test'])
    grouped_test_train = grouped_test_train.loc[~nan_rows, :]
    trend_correlation = np.corrcoef(grouped_test_train[target_col + '_mean'],
                                    grouped_test_train[target_col + '_mean_test'])[0, 1]

    if np.isnan(trend_correlation):
        trend_correlation = 0
        print("Only one bin created for " + feature + ". Correlation can't be calculated")
    return (trend_correlation)


def univariate_plotter(feature, data, target_col, bins=10, data_test=0):
    """
    Calls the draw plot function and editing around the plots
    :param feature: feature column name
    :param data: dataframe containing features and target columns
    :param target_col: target column name
    :param bins: number of bins to be created from continuous feature
    :param data_test: test data which has to be compared with input data for correlation
    :return:
    """
    print(' {:^100} '.format('Plots for ' + feature))
    if data[feature].dtype == 'O':
        print('Categorical feature not supported')
    else:
        cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col, bins=bins)
        has_test = type(data_test) == pd.core.frame.DataFrame
        if has_test:
            grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True), feature=feature,
                                            target_col=target_col, bins=bins, cuts=cuts)
            trend_corr = get_trend_correlation(grouped, grouped_test, feature, target_col)
            print(' {:^100} '.format('Train data plots'))

            draw_plots(input_data=grouped, feature=feature, target_col=target_col)
            print(' {:^100} '.format('Test data plots'))

            draw_plots(input_data=grouped_test, feature=feature, target_col=target_col, trend_correlation=trend_corr)
        else:
            draw_plots(input_data=grouped, feature=feature, target_col=target_col)
        print(
            '--------------------------------------------------------------------------------------------------------------')
        print('\n')
        if has_test:
            return (grouped, grouped_test)
        else:
            return (grouped)


def get_univariate_plots(data, target_col, features_list=0, bins=10, data_test=0):
    """
    Creates univariate dependence plots for features in the dataset
    :param data: dataframe containing features and target columns
    :param target_col: target column name
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
    :param bins: number of bins to be created from continuous feature
    :param data_test: test data which has to be compared with input data for correlation
    :return:
    """
    if type(features_list) == int:
        features_list = list(data.columns)
        features_list.remove(target_col)

    for cols in features_list:
        if cols != target_col and data[cols].dtype == 'O':
            print(cols + ' is categorical. Categorical features not supported yet.')
        elif cols != target_col and data[cols].dtype != 'O':
            univariate_plotter(feature=cols, data=data, target_col=target_col, bins=10, data_test=data_test)


def get_trend_stats_feature(data, target_col, features_list=0, bins=10, data_test=0):
    """
    Calculates trend changes and correlation between train/test for list of features
    :param data: dataframe containing features and target columns
    :param target_col: target column name
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
    :param bins: number of bins to be created from continuous feature
    :param data_test: test data which has to be compared with input data for correlation
    :return:
    """
    if type(features_list) == int:
        features_list = list(data.columns)
        features_list.remove(target_col)

    stats_all = []
    has_test = type(data_test) == pd.core.frame.DataFrame
    ignored = []
    for feature in features_list:
        if data[feature].dtype == 'O' or feature == target_col:
            ignored.append(feature)
        else:
            cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col, bins=bins)
            trend_changes = get_trend_changes(grouped_data=grouped, feature=feature, target_col=target_col)
            if has_test:
                grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True), feature=feature,
                                                target_col=target_col, bins=bins, cuts=cuts)
                trend_corr = get_trend_correlation(grouped, grouped_test, feature, target_col)
                trend_changes_test = get_trend_changes(grouped_data=grouped_test, feature=feature,
                                                       target_col=target_col)
                stats = [feature, trend_changes, trend_changes_test, trend_corr]
            else:
                stats = [feature, trend_changes]
            stats_all.append(stats)
    stats_all_df = pd.DataFrame(stats_all)
    stats_all_df.columns = ['Feature', 'Trend_changes'] if has_test == False else ['Feature', 'Trend_changes',
                                                                                   'Trend_changes_test',
                                                                                   'Trend_correlation']
    print('Categorical features ' + str(ignored) + ' ignored. Categorical features not supported yet.')
    return (stats_all_df)
