import prophet
import pandas as pd
import numpy as np
import logging
import datetime
import sys
sys.path.append('../pricingml/utils')
import matplotlib.pyplot as plt
from time_features import *
from interpretibility import *
from plot import *
from prophet.plot import plot_cross_validation_metric, plot_components, plot_forecast_component, plot_seasonality, \
        plot_weekly, plot_yearly, seasonality_plot_df

logging.getLogger('prophet').setLevel(logging.WARNING)

def instantiate_prophet():
    """
    Instantiating the model

    Returns
    ----------------
    The instantiated Prophet Model

    """

    model = prophet.Prophet(
        growth='linear',
        seasonality_mode='multiplicative',
        holidays_prior_scale=20,
        daily_seasonality=True,
        weekly_seasonality=True
    ).add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=10
    ).add_seasonality(
        name='daily',
        period=1,
        fourier_order=10
    ).add_seasonality(
        name='weekly',
        period=7,
        fourier_order=10
    ).add_seasonality(
        name='quarterly',
        period=365.25 /4,
        fourier_order=10
    ).add_country_holidays(country_name='US')

    return model

def prepare_data(df, df_test, fee_column):
    """
    Preparing the data

    Parameters
    ----------------
    :param df : whole dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field in the dataframe being analyzed

    Returns
    ----------------
    Modified train and test dataframes that have date column named as 'ds' and dependent variable as 'y'
    """

    df_train_prophet = df.copy()
    df_train_prophet.rename({fee_column: "y", "date": "ds"}, inplace=True, axis=1)

    df_test_prophet = df_test.copy()
    df_test_prophet.rename({fee_column: "y", "date": "ds"}, inplace=True, axis=1)

    return df_train_prophet, df_test_prophet

def get_forecast(df_train_prophet, hour):
    """
    Forecasting for next 30 days.

    Parameter
    ----------------
    :param df_train_prophet: training dataframe
    :param hour: whether hourly analysis is being performed or not

    Returns
    ----------------
    Dataframe with next 30 days forecast.
    """

    model = instantiate_prophet()
    model_fit = model.fit(df_train_prophet, iter=250)

    if hour is not None and hour.isnumeric():
        periods = int(30*24/int(hour))
        future = model_fit.make_future_dataframe(periods=periods, freq=str(hour)+'H')
    else:
        future = model_fit.make_future_dataframe(periods=30)

    print(future)
    forecast = model_fit.predict(future)
    split_date = None
    if hour:
        split_date = datetime.datetime.now()
    else:
        split_date = datetime.datetime.today().strftime('%Y-%m-%d')

    forecast = forecast[forecast["ds"] >= split_date]

    return forecast


def walk_forward(df_train_prophet, df_test_prophet, hour=None):
    """
    Performs forecasting using Prophet Model in walk-forward approach.

    Parameters
    ----------------
    :param df_train_prophet : training dataframe
    :param df_test_prophet : testing dataframe
    :param hour: whether hourly analysis is being performed or not

    Returns
    ----------------
    Predictions dataframe, instantiated model and forecast for next 30 days.
    """

    f_walk_forward = pd.DataFrame()
    end_idx = df_train_prophet.shape[0] - df_test_prophet.shape[0] - 1
    start_idx = 0

    model_prev = None

    for i in range(df_test_prophet.shape[0]):
        model = instantiate_prophet()
        for col in df_train_prophet.columns:
            if col not in ('ds', 'y'):
                model.add_regressor(col)

        model_fit = model.fit(df_train_prophet[start_idx:end_idx], iter=250)
        model_prev = model_fit

        future = model_fit.predict(df_test_prophet[i:i + 1])

        if (df_test_prophet.loc[i]['y'] < future.loc[0]["yhat_lower"]) or (df_test_prophet.loc[i]['y'] > future.loc[0]["yhat_upper"]):
            df_test_prophet.loc[i]['y'] = (future.loc[0]["yhat_lower"] + future.loc[0]["yhat_upper"])/2.0

        if i == 0:
            f_walk_forward = future
        else:
            f_walk_forward = f_walk_forward.append(future, ignore_index=True)

        end_idx = end_idx + 1
        start_idx = start_idx + 1

    forecast = get_forecast(df_train_prophet, hour)

    f_walk_forward.reset_index(inplace=True)

    return f_walk_forward, model_prev, forecast


def prophet_multi_predictions(df, df_test, fee_column, hour=None):
    """
    Performs forecasting using Multivariate Prophet model.

    Parameters
    ----------------
    :param df : whole dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed
    :param hour : if the forecast needs to calculated on hourly basis

    Returns
    ----------------
    Predictions from Multivariate Prophet model.
    """

    df_train_new = create_time_features(df, target=fee_column, multiprophet=True)
    df_test_new = create_time_features(df_test, target=fee_column, multiprophet=True)

    df_train_prophet, df_test_prophet = prepare_data(df_train_new, df_test_new, fee_column)

    f_walk_forward, model, forecast = walk_forward(df_train_prophet, df_test_prophet, hour)

    model.plot_components(forecast)

    if hour:
        model.plot_components(f_walk_forward).savefig('./utils/hourly_component_plots.png')
    else:
        model.plot_components(f_walk_forward).savefig('./utils/component_plots.png')
    # plt.show()

    return f_walk_forward["yhat"], f_walk_forward["yhat_lower"], f_walk_forward["yhat_upper"], forecast



