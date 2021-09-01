import sys
sys.path.append('../pricingml/utils')
from time_features import *
from sklearn import linear_model
from preprocess import *
from interpretibility import *


def regression_predictions(df_train, df_test, fee_column):
    """
    Performs forecasting using Bayesian Ridge Regression model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from Bayesian Ridge Regression model.
    """
    df_train_new, y_train = create_time_features(df_train, target=fee_column)
    df_test_new, y_test = create_time_features(df_test, target=fee_column)

    # df_train_new_scaled, df_test_new_scaled = scale_data(df_train_new, df_test_new)
    reg = linear_model.BayesianRidge()
    reg.fit(df_train_new, y_train)
    # interpret_model(reg, df_test_new)
    yhat = reg.predict(df_test_new)

    return yhat

def lasso_predictions(df_train, df_test, fee_column):
    """
    Performs forecasting using Lasso Regression model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from Lasso Regression model.
    """

    df_train_new, y_train = create_time_features(df_train, target=fee_column)
    df_test_new, y_test = create_time_features(df_test, target=fee_column)
    df_train_new_scaled, df_test_new_scaled = scale_data(df_train_new, df_test_new)
    reg = linear_model.Lasso(alpha=0.00001)
    reg.fit(df_train_new_scaled, y_train)
    yhat = reg.predict(df_test_new_scaled)

    return yhat


