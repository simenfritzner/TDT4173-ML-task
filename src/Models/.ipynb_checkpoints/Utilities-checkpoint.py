import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import numpy as np

#scaler = StandardScaler()
#scaler = MinMaxScaler()
scaler = RobustScaler()

def date_forecast_to_time(df):
    df['month'] = df['date_forecast'].dt.month
    df['hour'] = df['date_forecast'].dt.hour
    df['day'] = df['date_forecast'].dt.day
    
    df['hour_sin'] = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2. * np.pi / 24))
    df['month_sin'] = np.sin((df['month']-1) * (2. * np.pi / 12))
    df['month_cos'] = np.cos((df['month']-1) * (2. * np.pi / 12))
    return df
            
#Scales all the feature value in a way they take a simmilar range
def scale_df(df, fit):
    if fit == True:
        df = scaler.fit_transform(df)
    else:
        df = scaler.transform(df)
    return df

#Removes all features from a df except selected_features
def clean_df(df, selected_features):
    return df[selected_features]

#Function which resizes the training data such that only the rows with the same date and time for weather is kept.
#X_train is either observed or forcasted weather and y_train is how much energy is produced. 
#y_features are a list containing the column names of y_train
#X_date_feature is the feature name which the date and time for the weather is savew. This will probably always be "date_forecast" and may be changed
def resize_training_data(X_train, y_train):
    y_features = y_train.columns.tolist()
    X_date_feature = "date_forecast"
    
    merged = pd.merge(X_train, y_train,left_on=X_date_feature, right_on='time', how='inner')
    y_train_resized = merged[y_features]
    columns_to_drop = y_features + [X_date_feature]
    X_train_resized = merged.drop(columns = columns_to_drop)
    return X_train_resized, y_train_resized
    
#Splits the training data such that it is training set is observed and some estimated, valid is some estimated and test is some estimated
def training_data_split(X_observed_clean, X_estimated_clean_mean):
    X_train_estimated = X_estimated_clean_mean[:int(X_estimated_clean_mean.shape[0] * 3 / 4)]
    X_valid = X_estimated_clean_mean[int(X_estimated_clean_mean.shape[0] * 3 / 4):int(X_estimated_clean_mean.shape[0] * 9 / 10)]
    X_test = X_estimated_clean_mean[int(X_estimated_clean_mean.shape[0] * 9 / 10):]
    
    X_train = pd.concat([X_observed_clean, X_train_estimated])
    return X_train, X_valid, X_test
    
#A function which takes the mean out of every 4th column and saves it on the time on the time of the 4th. Makes it so it is every hour.
#TODO: Should be swapped for Gustavs code!
def mean_df(df):
    # Assuming df is your DataFrame and 'date_forecast' is your date column
    # Making a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()
    
    # Step 1: Keeping every 4th row in the date column
    date_column = df_copy['date_forecast'].iloc[::4]
    
    """#Made it such that all the intresting columns having data for 1 hour average back in time is saved such for the last hour. Ex diffuse_rad_1h:j cl. 23:00 is used for the weather prediction 22:00
    selected_col = ['diffuse_rad_1h:J', 'direct_rad_1h:J']
    selected_values = df_copy[selected_col].iloc[4::4].reset_index(drop=True)
    last_row = pd.DataFrame(df_copy[selected_col].iloc[-1]).T.reset_index(drop=True)
    selected_values = pd.concat([selected_values, last_row], ignore_index=True)"""
    
    # Step 2: Creating a grouping key
    grouping_key = np.floor(np.arange(len(df_copy)) / 4)
    
    # Step 3: Group by the key and calculate the mean, excluding the date column
    averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()
    # Step 4: Reset index and merge the date column
    averaged_data.reset_index(drop=True, inplace=True)
    averaged_data['date_forecast'] = date_column.values
    #averaged_data[selected_col] = selected_values.values
    return averaged_data

#Saves the predictions in proper format, y_pred needs to contain predicitions for all 3 locatoins

def submission(filename, y_pred, path_to_src):
    test = pd.read_csv(path_to_src + '/Data/CSV/test.csv')
    submission = pd.read_csv(path_to_src + '/Data/CSV/sample_submission.csv')
    test['prediction'] = y_pred
    submission = submission[['id']].merge(test[['id', 'prediction']], on='id', how='left')
    submission.to_csv(path_to_src + "/Data/CSV/" + filename, index=False)

    def drop_repeating_sequences(df):
    indexes_to_drop = []
    prev_val = None
    consecutive_count = 0

    for i, val in enumerate(df["pv_measurement"]):
        if val != 0:
            if val == prev_val:
                consecutive_count += 1
            else:
                prev_val = val
                consecutive_count = 0

            if consecutive_count >= 1:
                indexes_to_drop.extend([i - consecutive_count, i])

    return df.drop(indexes_to_drop)

def delete_ranges_of_zeros_and_interrupting_values(df, number_of_recurring_zeros, interrupting_values = []):
    count = 0
    drop_indices = []

    df = df.dropna()

    for index, row in df.iterrows():
        if row["pv_measurement"] == 0 or row["pv_measurement"] in interrupting_values:
            count += 1
        else:
            if count > number_of_recurring_zeros:
                drop_indices.extend(df.index[index - count:index])
            count = 0

    if count > number_of_recurring_zeros:
        drop_indices.extend(df.index[index - count + 1:index + 1])

    df.drop(drop_indices, inplace=True)

    return df

def drop_long_sequences(df, x):
    indexes_to_drop = []
    zero_count = 0

    for i, val in enumerate(df['pv_measurement']):
        if val == 0:
            zero_count += 1
        else:
            if zero_count >= x:
                start_index = i - zero_count
                end_index = i - 1  # inclusive
                if start_index >= 0 and end_index < len(df):
                    indexes_to_drop.extend(list(range(start_index, end_index + 1)))
            zero_count = 0

    # In case the sequence ends with zeros, this will handle it
    if zero_count >= x:
        start_index = len(df) - zero_count
        end_index = len(df) - 1
        if start_index >= 0 and end_index < len(df):
            indexes_to_drop.extend(list(range(start_index, end_index + 1)))

    return df.drop(df.index[indexes_to_drop])