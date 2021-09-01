import sys
sys.path.append('../pricingml/utils')
from time_features import *
from preprocess import *
from sklearn.ensemble import RandomForestRegressor
from interpretibility import *

def rforest_predictions(df_train, df_test, fee_column):
    """
    Performs forecasting using Random Forest Regression model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from Random Forest Regression model.
    """
    df_train_new, y_train = create_time_features(df_train, target=fee_column)
    df_test_new, y_test = create_time_features(df_test, target=fee_column)
    df_train_new_scaled, df_test_new_scaled = scale_data(df_train_new, df_test_new)

    reg = RandomForestRegressor(max_depth=10, random_state=0)
    reg.fit(df_train_new_scaled, y_train)

    yhat = reg.predict(df_test_new_scaled)

    return yhat


