import sys
import xgboost as xgb
sys.path.append('../pricingml/utils')
from time_features import *
from preprocess import *
from interpretibility import *
import numpy as np


def xgboost_predictions(df_train, df_test, fee_column):
    """
    Performs forecasting using XGBoost model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from XGBoost model.
    """
    df_train_new, y_train = create_time_features(df_train, target=fee_column)
    df_test_new, y_test = create_time_features(df_test, target=fee_column)

    df_train_new_scaled, df_test_new_scaled = scale_data(df_train_new, df_test_new)

    yhat = list()
    reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150000)
    for i in range(df_test_new_scaled.shape[0]):
        X = np.reshape(df_test_new_scaled[i:i+1], (1, df_test_new_scaled[i:i+1].size))
        reg.fit(df_train_new_scaled, y_train, verbose=False, early_stopping_rounds=50,
                eval_set=[(df_train_new_scaled, y_train), (df_test_new_scaled, y_test)])
        yhat.append(reg.predict(X))
        L = (df_train_new_scaled, df_test_new_scaled[i:i+1])
        np.vstack(L)

    # interpret_model(reg, df_train_new_scaled, 'Tree')
    return yhat
