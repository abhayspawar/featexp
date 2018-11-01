# featexp
Feature exploration for supervised learning. Helps with feature understanding, identifying noisy features, feature debugging, leakage detection and model monitoring.

featexp draws plots similar to partial dependence plots, but directly from data instead of using a trained model like current implementations of pdp do. 

```
from featexp import get_univariate_plots
get_univariate_plots(data=data_train, target_col='target', data_test=data_test, features_list=['DAYS_EMPLOYED'])

# data_test and features_list are optional. 
# Draws plots for all columns if features_list not passed
# Draws only train data plots if no test_data passed
```
![Output1](demo/sample_outputs/days_employed.png)
featexp bins a feature into equal population bins and shows mean value of dependent variable (target) in each bin. Here's how to read these plots:
  1. Trend plot on left helps you understand the relationship between target and feature.
  2. Population distribution helps you make sure the feature is correct. 
  3. Also, shows number of trend direction changes and correlation between train and test trend which can be used to identify      noisy features. High number of trend changes or low trend correlation implies high noise.

Getting trend changes and trend correlation for all features in a dataframe:
```
from featexp import get_trend_stats_feature
stats = get_trend_stats(data=data_train, target_col='target', data_test=data_test)

# data_test is optional. If nothing is passed, trend correlations aren't calculated.
```
Returns a dataframe with trend changes and trend correlation which can be used for dropping the noisy features, etc.
![Output1](demo/sample_outputs/stats_output.png)

Blog post on how to use featexp with elaborate exmaples: coming soon 
