import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import time
from sklearn.inspection import permutation_importance



#Function which resizes the training data such that only the rows with the same date and time for weather is kept.
#X_train is either observed or forcasted weather and y_train is how much energy is produced. 
#y_features are a list containing the column names of y_train
#X_date_feature is the feature name which the date and time for the weather is savew. This will probably always be "date_forecast" and may be changed
"""
def resize_trainingdata(X_train, y_train, X_date_feature, y_features):
    merged = pd.merge(X_train, y_train,left_on=X_date_feature, right_on='time', how='inner')
    y_train_resized = merged[y_features]
    columns_to_drop = y_features + [X_date_feature]
    X_train_resized = merged.drop(columns = columns_to_drop)
    return X_train_resized, y_train_resized
"""

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


#takes in an random forrest model and computes the Feature importance based on mean decrease in impurity¶
def feature_importances_mean_decrease_in_impurity_with_rf(rf):

    start_time = time.time()
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances = pd.Series(importances, index=feature_names)
    
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    
    
#takes in an random forrest model and computes Feature importance based on feature permutation
#X_test and y_test are from said function:
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
#thus will need to get them form the forrest
def feature_importances_mean_decrease_in_impurity_with_rf(forest, X_test , y_test ):
    start_time = time.time()
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()