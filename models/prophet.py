import prophet
import pandas as pd
import logging
import datetime
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

def get_forecast(df_train_prophet):
    """
    Forecasting for next 30 days.

    Parameter
    ----------------
    :param df_train_prophet: training dataframe

    Returns
    ----------------
    Dataframe with next 30 days forecast.
    """

    model = instantiate_prophet()
    model_fit = model.fit(df_train_prophet, iter=250)
    future = model_fit.make_future_dataframe(periods=60, freq='12H')
    forecast = model_fit.predict(future)
    split_date = datetime.datetime.now()
    forecast = forecast[forecast["ds"] >= split_date]

    return forecast

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
    df_train_prophet.rename({fee_column: "y", "datetime": "ds"}, inplace=True, axis=1)

    df_test_prophet = df_test.copy()
    df_test_prophet.rename({fee_column: "y", "datetime": "ds"}, inplace=True, axis=1)

    return df_train_prophet, df_test_prophet


def prophet_predictions(df, df_test, fee_column):
    """
    Performs forecasting using Prophet model.

    Parameters
    ----------------
    :param df : whole dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from Multivariate Prophet model.
    """

    df_train_prophet, df_test_prophet = prepare_data(df, df_test, fee_column)

    idx = df_train_prophet.shape[0]

    f_walk_forward = pd.DataFrame()

    for i in range(df_test_prophet.shape[0]):
        idx = idx + i
        model = instantiate_prophet()
        model.add_regressor('total_sale_price')
        model_fit = model.fit(df_train_prophet[:idx], iter=250)
        future = model_fit.predict(df_test_prophet[i:i + 1])
        if i == 0:
            f_walk_forward = future
        else:
            f_walk_forward = f_walk_forward.append(future, ignore_index=True)

    f_walk_forward.reset_index(inplace=True)
    forecast = get_forecast(df_train_prophet)

    return f_walk_forward["yhat"], f_walk_forward["yhat_lower"], f_walk_forward["yhat_upper"], forecast



