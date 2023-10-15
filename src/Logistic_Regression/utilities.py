import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

#Function which resizes the training data such that only the rows with the same date and time for weather is kept.
#X_train is either observed or forcasted weather and y_train is how much energy is produced. 
#y_features are a list containing the column names of y_train
#X_date_feature is the feature name which the date and time for the weather is savew. This will probably always be "date_forecast" and may be changed

#resize_trainingdata(X_train_observed_a_clean_selected_features, train_a, "date_forecast", y_features)

def resize_trainingdata(X_train, y_train, X_date_feature, y_features):
    y_train.dropna(inplace=True)
    merged = pd.merge(X_train, y_train,left_on=X_date_feature, right_on='time', how='inner')
    y_train_resized = merged[y_features]
    columns_to_drop = y_features + [X_date_feature]
    X_train_resized = merged.drop(columns = columns_to_drop)
    return X_train_resized, y_train_resized


def submission(filename, y_pred):
    test = pd.read_csv('CSV/test.csv')
    submission = pd.read_csv('CSV/sample_submission.csv')
    test['prediction'] = y_pred
    submission = submission[['id']].merge(test[['id', 'prediction']], on='id', how='left')
    submission.to_csv(filename, index=False)

def correlation_plot(X_frame,Y_frame):

    correlation_df = pd.merge(X_frame, Y_frame,left_on="date_forecast", right_on='time', how='inner')
    
    #Regne ut korrelasjon
    correlation = correlation_df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    #Korrelasjon plot
    f, ax = plt.subplots(figsize=(40, 30))
    sns.heatmap(correlation, mask=mask, cmap='jet', vmax=.3, center=0, annot=True, fmt='.3f',
                square=True, linewidths=.5, cbar_kws={"shrink": .8});
    plt.title('Correlation analysis for all sites');

def correlation_plot_no_resizing(X_frame,Y_frame):

    correlation_df = X_frame.copy()
    correlation_df = correlation_df.assign(pv_measurement=Y_frame["pv_measurement"])
    
    #Regne ut korrelasjon
    correlation = correlation_df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    #Korrelasjon plot
    f, ax = plt.subplots(figsize=(26, 23))
    sns.heatmap(correlation, mask=mask, cmap='jet', vmax=.3, center=0, annot=True, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .8});
    plt.title('Correlation analysis for all sites');


def clean_X_data(X_frame, columns_to_drop = []):
    #Fjerner "date_calc" kolonnen
    try:
        X_frame = X_frame.drop("date_calc",axis = 1)
    except:
        pass
    X_frame = X_frame.drop(columns_to_drop,axis = 1)
    #Fjerne hver fjerde rad for å beholde kun hele timer
    X_frame = X_frame[::4].copy()
    
    return X_frame

def correlation(X_frame,Y_frame):

    correlation_df = pd.merge(X_frame, Y_frame,left_on="date_forecast", right_on='time', how='inner')
    correlation = correlation_df.corr()

    return correlation

# Function to delete sequences with repeating numbers
def delete_sequence_with_repeating_numbers(df, column_name, threshold):
    count = 1
    prev_value = df.iloc[0][column_name]
    indexes_to_drop = []
    drop_ranges = []
    for index, row in df.iterrows():
        if row[column_name] == prev_value:
            count += 1
        else:
            if count > threshold:
                drop_ranges.append((index - count, index - 1))
                indexes_to_drop.extend(range(index - count, index))
            count = 1
            prev_value = row[column_name]
    if count > threshold:
        drop_ranges.append((len(df) - count, len(df) - 1))
        indexes_to_drop.extend(range(len(df) - count, len(df)))

    df.drop(indexes_to_drop, inplace=True)

    return drop_ranges

def round_to_two_decimals(df, column_name):
    for index, row in df.iterrows():
        if isinstance(row[column_name], float):
            if len(str(row[column_name]).split('.')[-1]) > 2:
                df.at[index, column_name] = round(row[column_name], 2)


# Function to delete repeating non-zero numbers
def delete_repeating_non_zero_numbers(df, column_name, threshold):
    count = 1
    prev_value = df.iloc[0][column_name]
    drop_indices = []
    start = 0

    for index, row in df.iterrows():
        if row[column_name] == prev_value and row[column_name] != 0:
            count += 1
        else:
            if count > threshold:
                drop_indices.extend(range(start, start + count))
            count = 1
            start = index
            prev_value = row[column_name]

    if count > threshold:
        drop_indices.extend(range(start, start + count))

    df.drop(df.index[drop_indices], inplace=True)

# Function to delete ranges of more than 22 zeros
def delete_ranges_of_zeros(df, column_name, threshold, interrupting_values):
    count = 0
    drop_indices = []

    for index, row in df.iterrows():
        if row[column_name] == 0 or row[column_name] in interrupting_values:
            count += 1
        else:
            if count > threshold:
                drop_indices.extend(df.index[index - count:index])
            count = 0

    if count > threshold:
        drop_indices.extend(df.index[index - count + 1:index + 1])

    df.drop(drop_indices, inplace=True)
