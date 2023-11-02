import pandas as pd
#Loading the y-paramter from file, might be changed to y_a etc.
y_a = pd.read_parquet('../../Data/Data_and_task/A/train_targets.parquet')
y_b = pd.read_parquet('../../Data/Data_and_task/B/train_targets.parquet')
y_c = pd.read_parquet('../../Data/Data_and_task/C/train_targets.parquet')
y_b = y_b.dropna()
y_c=y_c.dropna()

#Loading estimated/forecasted training_weather from file
X_estimated_a = pd.read_parquet('../../Data/Data_and_task/A/X_train_estimated.parquet')
X_estimated_b = pd.read_parquet('../../Data/Data_and_task/B/X_train_estimated.parquet')
X_estimated_c = pd.read_parquet('../../Data/Data_and_task/C/X_train_estimated.parquet')

#Loading observed weather from file
X_observed_a = pd.read_parquet('../../Data/Data_and_task/A/X_train_observed.parquet')
X_observed_b = pd.read_parquet('../../Data/Data_and_task/B/X_train_observed.parquet')
X_observed_c = pd.read_parquet('../../Data/Data_and_task/C/X_train_observed.parquet')

#Loading estimated/forecasted test_weather from file
X_test_a = pd.read_parquet('../../Data/Data_and_task/A/X_test_estimated.parquet')
X_test_b = pd.read_parquet('../../Data/Data_and_task/B/X_test_estimated.parquet')
X_test_c = pd.read_parquet('../../Data/Data_and_task/C/X_test_estimated.parquet')




X_test_a['dif_dat_rad'] = 0.0
X_test_b['dif_dat_rad'] = 0.0
X_test_c['dif_dat_rad'] = 0.0

condition_a = X_test_a['diffuse_rad:W'] != 0
condition_b = X_test_b['diffuse_rad:W'] != 0
condition_c = X_test_c['diffuse_rad:W'] != 0

X_test_a.loc[condition_a, 'dif_dat_rad'] = X_test_a.loc[condition_a, 'direct_rad:W'] / X_test_a.loc[condition_a, 'diffuse_rad:W']
X_test_b.loc[condition_b, 'dif_dat_rad'] = X_test_b.loc[condition_b, 'direct_rad:W'] / X_test_b.loc[condition_b, 'diffuse_rad:W']
X_test_c.loc[condition_c, 'dif_dat_rad'] = X_test_c.loc[condition_c, 'direct_rad:W'] / X_test_c.loc[condition_c, 'diffuse_rad:W']
