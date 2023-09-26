import numpy as np
import pandas as pd

train_a = pd.read_parquet('../Data_and_task/A/train_targets.parquet')
train_b = pd.read_parquet('../Data_and_task/B/train_targets.parquet')
train_c = pd.read_parquet('../Data_and_task/C/train_targets.parquet')

X_train_estimated_a = pd.read_parquet('../Data_and_task/A/X_train_estimated.parquet')
X_train_estimated_b = pd.read_parquet('../Data_and_task/B/X_train_estimated.parquet')
X_train_estimated_c = pd.read_parquet('../Data_and_task/C/X_train_estimated.parquet')

X_train_observed_a = pd.read_parquet('../Data_and_task/A/X_train_observed.parquet')
X_train_observed_b = pd.read_parquet('../Data_and_task/B/X_train_observed.parquet')
X_train_observed_c = pd.read_parquet('../Data_and_task/C/X_train_observed.parquet')

X_test_estimated_a = pd.read_parquet('../Data_and_task/A/X_test_estimated.parquet')
X_test_estimated_b = pd.read_parquet('../Data_and_task/B/X_test_estimated.parquet')
X_test_estimated_c = pd.read_parquet('../Data_and_task/C/X_test_estimated.parquet')

wanted_attributes = []