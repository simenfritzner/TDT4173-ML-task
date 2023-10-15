import sys
import pandas as pd

from utilities import *

#Loading the y-paramter from file, might be changed to y_a etc.
y_a = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/A/train_targets.parquet')
y_b = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/B/train_targets.parquet')
y_c = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/C/train_targets.parquet')

#Loading estimated/forecasted training_weather from file
X_train_estimated_a = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/A/X_train_estimated.parquet')
X_train_estimated_b = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/B/X_train_estimated.parquet')
X_train_estimated_c = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/C/X_train_estimated.parquet')

#Loading observed weather from file
X_train_observed_a = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/A/X_train_observed.parquet')
X_train_observed_b = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/B/X_train_observed.parquet')
X_train_observed_c = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/C/X_train_observed.parquet')

#Loading estimated/forecasted test_weather from file
X_test_estimated_a = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/A/X_test_estimated.parquet')
X_test_estimated_b = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/B/X_test_estimated.parquet')
X_test_estimated_c = pd.read_parquet('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_and_task/C/X_test_estimated.parquet')

y_b_clean = remove_repeating_non_zero_numbers(y_b.copy())
y_c_clean = delete_ranges_of_zeros_and_interrupting_values(y_c.copy(),5,[19.6,9.8])

y_a_delete = delete_repeating_sequences_with_interrupting_values(y_a.copy(),[])
y_c_delete = delete_repeating_sequences_with_interrupting_values(y_c.copy(),[19.6,9.8])
y_b_delete = delete_repeating_sequences_with_interrupting_values(y_b.copy(),[3.45,0.8625,1.725])


with pd.ExcelWriter('/Users/erik/Library/CloudStorage/OneDrive-NTNU/Maskinlæring/Oppgave 2 gruppe/Main branch/TDT4173-ML-task/src/Data/Data_exploration/Excel/y_data.xlsx') as appender:
    y_a.to_excel(appender, sheet_name='y_a', index=False)
    y_a_delete.to_excel(appender, sheet_name='y_a_delete', index=False)

    y_b.to_excel(appender, sheet_name='y_b', index=False)
    y_b_clean.to_excel(appender, sheet_name='y_b_clean', index=False)
    y_b_delete.to_excel(appender, sheet_name='y_b_delete', index=False)

    y_c.to_excel(appender, sheet_name='y_c', index=False)
    y_c_clean.to_excel(appender, sheet_name='y_c_clean', index=False)
    y_c_delete.to_excel(appender, sheet_name='y_c_delete', index=False)
