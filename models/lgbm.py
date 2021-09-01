import sys
sys.path.append('../pricingml/utils')
from time_features import *
from preprocess import *
import lightgbm as lgb
from interpretibility import *


def instantiate_LGBMregressor():
    """
    Instantiating the model

    Returns
    ----------------
    The instantiated LGBM

    """
    lightGBM = lgb.LGBMRegressor()
    #     nthread=10,
    #     max_depth=5,
    #     task='train',
    #     boosting_type='gbdt',
    #     objective='regression_l1',
    #     metric='rmse',
    #     num_leaves=64,
    #     learning_rate=0.01,
    #     feature_fraction=0.9,
    #     bagging_fraction=0.8,
    #     bagging_freq=5,
    #     verbose=0
    # )
    return lightGBM

def lgbm_predictions(df_train, df_test, fee_column):

    """
    Performs forecasting using Microsoft's Light Gradient Boosting Machine (LGBM) model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from (LGBM) model.
    """

    df_train_new, y_train = create_time_features(df_train, target=fee_column)
    df_test_new, y_test = create_time_features(df_test, target=fee_column)

    df_train_new_scaled, df_test_new_scaled = scale_data(df_train_new, df_test_new)

    reg = instantiate_LGBMregressor()

    reg.fit(df_train_new_scaled, y_train)
    yhat = reg.predict(df_test_new_scaled)

    return yhat