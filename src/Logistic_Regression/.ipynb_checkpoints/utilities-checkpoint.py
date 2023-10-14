import pandas as pd

#Function which resizes the training data such that only the rows with the same date and time for weather is kept.
#X_train is either observed or forcasted weather and y_train is how much energy is produced. 
#y_features are a list containing the column names of y_train
#X_date_feature is the feature name which the date and time for the weather is savew. This will probably always be "date_forecast" and may be changed
def resize_trainingdata(X_train, y_train, X_date_feature, y_features):
    merged = pd.merge(X_train, y_train,left_on=X_date_feature, right_on='time', how='inner')
    y_train_resized = merged[y_features]
    columns_to_drop = y_features + [X_date_feature]
    X_train_resized = merged.drop(columns = columns_to_drop)
    return X_train_resized, y_train_resized

#removes Nan and changes it with the last leagal value. if the fist value is Nan, than it becomes the first leagal value
def remove_NaN_from_Y(y_train,y_features):
    y_train[y_features].fillna(method='bfill', inplace=True, limit=1)
    y_train[y_features].fillna(method='ffill', inplace=True)
    return y_train


def get_consecutive_nan_indices(series):
    nan_indices = []
    count = 0
    for i, val in enumerate(series):
        if pd.isna(val):
            count += 1
            if count >= 4:
                nan_indices.append(i)
        else:
            count = 0
    return nan_indices

def remove_rows_if_NAN_recurrs(y_train, y_features):
    indices_to_drop = set()
    for col in df.columns:
        indices_to_drop.update(get_consecutive_nan_indices(y_train[y_features]))
    y_train.drop(index=indices_to_drop, inplace=True)


def submission(filename, y_pred):
    test = pd.read_csv('CSV/test.csv')
    submission = pd.read_csv('CSV/sample_submission.csv')
    test['prediction'] = y_pred
    submission = submission[['id']].merge(test[['id', 'prediction']], on='id', how='left')
    submission.to_csv(filename, index=False)
    
    
def custom_mean(group):
    return group.apply(lambda x: np.nanmean(x))