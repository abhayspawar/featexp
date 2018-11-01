# featexp
Feature exploration for supervised learning. Helps with feature understanding, identifying noisy features, feature debugging, leakage detection and model monitoring.

featexp draws plots similar to partial dependence plots, but directly from data instead of using a trained model like current implementations of pdp do. 

```
from featexp import get_univariate_plots
get_univariate_plots(data=data_train, target_col='target', data_test=data_test, features_list=['DAYS_EMPLOYED'])
# data_test and features_list are optional. 
# Draws plots for all columns in data if features_list not passed
# Draws only train data plots if no test_data passed
```
![Output1](demo/sample_outputs/days_employed.png)


Thats all!





sdf
