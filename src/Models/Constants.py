import pandas as pd
from utilities import *
#Loading the y-paramter from file, might be changed to y_a etc.
y_a = pd.read_parquet('../../Data/Data_and_task/A/train_targets.parquet')
y_b = pd.read_parquet('../../Data/Data_and_task/B/train_targets.parquet')
y_c = pd.read_parquet('../../Data/Data_and_task/C/train_targets.parquet')
y_a = y_a.dropna()
y_b = y_b.dropna()
y_c = y_c.dropna()

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

# y_c = delete_ranges_of_zeros_and_interrupting_values(y_c,5,[19.6,9.8])

# y_b = delete_ranges_of_zeros_and_interrupting_values(y_b,10,[3.45,0.8625,1.725])

# y_b = remove_repeating_non_zero_numbers(y_b)