import numpy as np


def create_time_features(df, target=None, multiprophet=False, drop_date=True):
    """
    Creates time series features from datetime index

    Parameters
    ----------------
    :param df : dataframe
    :param target : dependent variable
    :param multiprophet : if the model is Multivariate Prophet
    :param drop_date : if the date column should be dropped or not

    Returns
    ----------------
    df if model name is Multivariate Prophet else return df and y column
    """

    df['date'] = df['datetime']
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['quarter'] = df['datetime'].dt.quarter
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_day'] = np.sin(df['dayofyear'])
    df['cos_day'] = np.cos(df['dayofyear'])
    df['dayofmonth'] = df['datetime'].dt.day
    df['weekofyear'] = df['datetime'].dt.weekofyear

    if multiprophet:
        return df

    if drop_date:
        df = df.drop(['date'], axis=1)
        df = df.drop(['datetime'], axis=1)

    if target:
        y = df[target]
        df = df.drop([target], axis=1)
        return df, y

    return df
