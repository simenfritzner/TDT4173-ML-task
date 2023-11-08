import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import numpy as np

#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = RobustScaler()
def augment_y_c(df_y_c):
    y_b_to_fit_1, y_c_to_predict_1= delete_ranges_of_zeros_and_interrupting_values_and_return_y_with_dropped_indices(df_y_c,5,[19.6,9.8])
    return y_b_to_fit_1, y_c_to_predict_1

def augment_y_b(df_y_b):
    y_b_to_fit, y_b_to_predict_1 = drop_repeating_sequences_and_return_y_with_droped_indixes(df_y_b)
    y_b_to_fit_2, y_b_to_predict_2 = delete_ranges_of_zeros_and_interrupting_values_and_return_y_with_dropped_indices(y_b_to_fit.copy(),200,[0.8625])
    y_b_to_fit_3, y_b_to_predict_3 = delete_ranges_of_zeros_and_interrupting_values_and_return_y_with_dropped_indices(y_b_to_fit_2.copy(),25,[0.8625])
    y_b_to_fit_4, y_b_to_predict_4 = drop_long_sequences_and_return_y_with_dropped_indices(y_b_to_fit_3.copy(),25)
    y_b_to_predict = pd.concat([y_b_to_predict_1, y_b_to_predict_2, y_b_to_predict_3, y_b_to_predict_4], axis=0)
    return y_b_to_fit_4, y_b_to_predict

def drop_repeating_sequences_and_return_y_with_droped_indixes(df):
    indexes_to_drop = set()  # Change this to a set to avoid duplicates
    prev_val = None
    consecutive_count = 0
    y_with_indexes_to_drop = df.copy()

    for i, val in enumerate(df["pv_measurement"]):
        if val != 0:
            if val == prev_val:
                consecutive_count += 1
            else:
                prev_val = val
                consecutive_count = 0

            if consecutive_count >= 1:
                indexes_to_drop.add(i - consecutive_count)  # Add to set to ensure uniqueness
                indexes_to_drop.add(i)  # Add to set to ensure uniqueness
    
    # Convert the set to a sorted list to use for indexing
    indexes_to_drop = sorted(indexes_to_drop)
    
    # Create the DataFrame without the dropped indices
    df_without_dropped = df.drop(indexes_to_drop)
    
    # Create the DataFrame with only the dropped indices
    df_with_only_dropped = df.loc[indexes_to_drop]

    return df_without_dropped, df_with_only_dropped


def delete_ranges_of_zeros_and_interrupting_values_and_return_y_with_dropped_indices(df, number_of_recurring_zeros, interrupting_values=[]):
    count = 0
    drop_indices = []

    # Store the original DataFrame to reinsert NaN values later
    original_df = df.copy()

    # Get the indices of the NaN values
    nan_indices = df[df['pv_measurement'].isna()].index.tolist()
    if len(nan_indices) > 0:
        print("penis")
    # Drop the NaN values for processing zeros and interrupting values
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
    
      # Convert the set to a sorted list to use for indexing
    indexes_to_drop = sorted(drop_indices)
    
    # Create the DataFrame without the dropped indices
    df_without_dropped = df.drop(indexes_to_drop)
    
    # Combine drop_indices with nan_indices to get all indices to be dropped
    all_drop_indices = sorted(set(drop_indices + nan_indices))

    # Create the DataFrame with only the dropped indices
    df_with_only_dropped = original_df.loc[all_drop_indices]
    
    return df_without_dropped, df_with_only_dropped  #, df_with_only_dropped (uncomment if needed)

def drop_long_sequences_and_return_y_with_dropped_indices(df, x):
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

    # Create a dataframe with only the dropped rows
    df_with_only_dropped = df.loc[df.index[indexes_to_drop]].copy()

    # Drop the rows from the original dataframe
    df_dropped = df.drop(df.index[indexes_to_drop])
    
    return df_dropped, df_with_only_dropped

#gustav sitt
def agumenting_time(df):
    df["new_time"] = pd.to_datetime(df['date_forecast'])
    df['hour'] = df['new_time'].dt.hour
    df['minute'] = df['new_time'].dt.minute
    #df["day"]  = df['new_time'].dt.day
    df["month"]  = df['new_time'].dt.month
    df['time_decimal'] = df['hour'] + df['minute'] / 60.0
    phase_adjustment = (np.pi/2) - 11 * (2 * np.pi / 24)
    df['hour_sin'] = np.sin(df['time_decimal'] * (2. * np.pi / 24) + phase_adjustment)
    df['hour_cos'] = np.cos(df['time_decimal'] * (2. * np.pi / 24) + phase_adjustment)
    df = df.drop(columns = ["new_time"])
    return df

def direct_rad_div_diffuse_rad(df):
    df['dif_dat_rad'] = 0.0
    condition = df['diffuse_rad:W'] != 0
    df.loc[condition, 'dif_dat_rad'] = df.loc[condition, 'direct_rad:W'] / df.loc[condition, 'diffuse_rad:W']
    return df

def get_hyperparameters_for_rf(x_observed, x_estimated, y, selected_features ):
    X_train = pd.concat([clean_df(x_observed, selected_features), clean_df(x_estimated, selected_features)])
    X_train, y_train = resize_training_data(X_train,y)
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Create the scorer
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Create the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scorer, cv=5)

    # Fit the grid search
    grid_search.fit(X_train, y_train["pv_measurement"])

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters b: {best_params}")
    return best_params

#gammelt under


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
    
    #Made it such that all the intresting columns having data for 1 hour average back in time is saved such for the last hour. Ex diffuse_rad_1h:j cl. 23:00 is used for the weather prediction 22:00
    selected_col = ['diffuse_rad_1h:J', 'direct_rad_1h:J', 'clear_sky_energy_1h:J']
    selected_values = df_copy[selected_col].iloc[4::4].reset_index(drop=True)
    last_row = pd.DataFrame(df_copy[selected_col].iloc[-1]).T.reset_index(drop=True)
    selected_values = pd.concat([selected_values, last_row], ignore_index=True)
    
    # Step 2: Creating a grouping key
    grouping_key = np.floor(np.arange(len(df_copy)) / 4)
    
    # Step 3: Group by the key and calculate the mean, excluding the date column
    averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()
    # Step 4: Reset index and merge the date column
    averaged_data.reset_index(drop=True, inplace=True)
    averaged_data['date_forecast'] = date_column.values
    averaged_data[selected_col] = selected_values.values
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

def clean_mean_combine(X_observed, X_estimated, selected_features):
    X_observed_clean = clean_df(X_observed, selected_features)
    X_estimated_clean = clean_df(X_estimated, selected_features)
    X_estimated_clean_mean = mean_df(X_estimated_clean)
    X_observed_clean_mean = mean_df(X_observed_clean)
    X_train = pd.concat([X_observed_clean_mean, X_estimated_clean_mean])
    return X_train

def prepare_X(X_observed, X_estimated, selected_features, wanted_months):
    X_observed = subset_months(X_observed.copy(), wanted_months)
    X_train = clean_mean_combine(X_observed, X_estimated, selected_features)
    X_train = add_lag_and_lead_features(X_train, 1, ['direct_plus_diffuse', 'direct_plus_diffuse_1h'])
    return X_train

def prepare_testdata_rf_a(X_test, selected_features):
    X_test = clean_df(X_test, selected_features)
    X_test = mean_df(X_test)
    X_test = add_lag_and_lead_features(X_test, 1, ['direct_plus_diffuse', 'direct_plus_diffuse_1h'])
    X_test = X_test.drop(columns = ["date_forecast"])
    return X_test

def add_all_features(df):
    df = direct_rad_div_diffuse_rad(df)
    df = agumenting_time(df)
    df["direct_plus_diffuse"] = df["direct_rad:W"] + df["diffuse_rad:W"]
    df["direct_plus_diffuse_1h"] = df["direct_rad_1h:J"] + df["diffuse_rad_1h:J"]
    return df

def subset_months(df, wanted_months):
    df["month"]  = df['date_forecast'].dt.month
    df_subset = df[df["month"].isin(wanted_months)]
    return df_subset

def remove_all_predicted_values_during_given_time_frame(X_test_c, x_pred, hours_to_zero_out_b):
    new_df = pd.DataFrame({
        "date_forecast": X_test_c["date_forecast"].iloc[::4].reset_index(drop=True),
        "pv_measurement": x_pred  # Assuming x_pred is the correct variable here
    })

    # Convert 'date_forecast' to datetime and extract hour and month
    new_df["new_time"] = pd.to_datetime(new_df['date_forecast'])
    new_df['hour'] = new_df['new_time'].dt.hour
    new_df["month"] = new_df['new_time'].dt.month
    
    hourly_sum = new_df.groupby('hour')['pv_measurement'].sum()
    """
    # Iterate through each hour and print the total sum of 'pv_measurements'
    print("before augmentation")
    for hour, sum_pv in hourly_sum.items():
        print(f"Hour {hour}: Total PV Measurements = {sum_pv}")
    """ 
    # Update pv_measurements to 0 based on the month-hour mapping
    for month, hours in hours_to_zero_out_b.items():
        new_df.loc[(new_df['month'] == month) & (new_df['hour'].isin(hours)), 'pv_measurement'] = 0
    
    #print_zeros(new_df)
    augmented_pv_measurements = new_df['pv_measurement']

    # Convert to array if needed
    augmented_pv_measurements_array = augmented_pv_measurements.to_numpy()
    return augmented_pv_measurements_array

def add_lag_and_lead_features(df, lag_steps=1, columns = []):
    # Create a new DataFrame to hold the lagged and lead features
    lagged_df = pd.DataFrame()

    # Make sure the 'date' column is a datetime type
    df['date_forecast'] = pd.to_datetime(df['date_forecast'])

    # Group by date to ensure continuity within each day
    grouped = df.groupby(df['date_forecast'].dt.date)

    for _, group in grouped:
        # Reset index to allow proper shifting within group
        group = group.reset_index(drop=True)

        # Copy the current group to avoid modifying the original data
        temp_group = group.copy()

        # Iterate over all columns to create lagged and lead versions
        for column in columns:
            # Skip the date column if it exists
            if column == 'date' or column == 'date_forecast':
                continue

            # Create lagged feature for previous values
            lagged_column_name = f"{column}_lag{lag_steps}"
            temp_group[lagged_column_name] = group[column].shift(lag_steps).fillna(group[column])

            # Create lead feature for future values
            lead_column_name = f"{column}_lead{lag_steps}"
            temp_group[lead_column_name] = group[column].shift(-lag_steps).fillna(group[column])

            # Create a column for the difference between the lagged value and the present value (lag -1 - present)
            diff_lag_column_name = f"{column}_diff_lag{lag_steps}"
            temp_group[diff_lag_column_name] = temp_group[lagged_column_name] - group[column]

            # Create a column for the difference between the lead value and the present value (lag +1 - present)
            #diff_lead_column_name = f"{column}_diff_lead{lag_steps}"
            #temp_group[diff_lead_column_name] = temp_group[lead_column_name] - group[column]

        # Append the processed group to the lagged_df
        lagged_df = pd.concat([lagged_df, temp_group], axis=0)

    # Reset the index of the resulting DataFrame
    lagged_df = lagged_df.reset_index(drop=True)

    return lagged_df
