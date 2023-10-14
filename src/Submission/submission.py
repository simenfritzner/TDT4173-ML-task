import numpy as np
import pandas as pd
#Saves the predictions in proper format, y_pred needs to contain predicitions for all 3 locatoins
def submission(filename, y_pred):
    test = pd.read_csv('../Data/CSV/test.csv')
    submission = pd.read_csv('../Data/CSV/sample_submission.csv')
    test['prediction'] = y_pred
    submission = submission[['id']].merge(test[['id', 'prediction']], on='id', how='left')
    submission.to_csv(filename, index=False)