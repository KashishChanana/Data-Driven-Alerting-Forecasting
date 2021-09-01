import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime
import numpy as np


def clean(df, columns):
    """
    Cleans data. Converts object datatype to float. Filters data by the given date.

    Parameters
    ----------------
    :param df : dataframe
    :params cols : list of columns to be cleaned

    Returns
    ----------------
    Cleaned dataframe.
    """

    df.dropna(inplace=True)

    df['datetime'] = pd.to_datetime(df['datetime'])

    for col in columns:
        df[col] = df[col].astype(float)

    # filtering by the date
    filter_date = pd.Timestamp("2020-07-01 00:00:00")
    df = df[df["datetime"] >= filter_date]

    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)

    return df


def train_test_split(df, hour=None):
    """
    Splits dataframe by the date into training and testing dataframe.

    Parameters
    ----------------
    :param df : dataframe
    :param hour : whether hourly analysis is being performed or not

    Returns
    ----------------
    Training dataframe and testing dataframe.
    """

    split_date = datetime.datetime.now() - datetime.timedelta(30)

    df_train = df.loc[df['datetime'] <= split_date]
    df_test = df.loc[df['datetime'] > split_date]
    if hour is None:
        df_test = df_test.loc[df_test['datetime'] < np.datetime64('today')]
    print(f"\n \n {len(df_train)} points of training data days \n {len(df_test)} points of testing data days \n \n")

    df_test.reset_index(inplace=True)
    df_test.drop(['index'], axis=1, inplace=True)

    return df_train, df_test

def scale_data(df_train, df_test):
    """
    Scales data, i.e,  standardizes features by removing the mean and scaling to unit variance

    Parameters
    ----------------
    :param df_train : training data
    :param df_test : testing data

    Returns
    ----------------
    Scaled training dataframe and scaled testing dataframe.
    """

    features = list(df_train.columns)
    if "datetime" in features:
        features.remove("datetime")

    if "date" in features:
        features.remove("date")

    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()

    scaler = StandardScaler()
    scaler.fit(df_train[features])
    df_train_scaled[features] = scaler.transform(df_train[features])
    df_test_scaled[features] = scaler.transform(df_test[features])

    return df_train_scaled, df_test_scaled

def aggregate(df, hour='12', WoW=False):
    """
    Aggregates/ Groups dataframe based on hours/week

    Parameters
    ----------------
    :param df : dataframe
    :param hour : aggregate dataframe based on specified number of hours, default 12
    :param WoW : whether data needs to be aggregated on a weekly basis or not

    Returns
    ----------------
    Aggregated dataframe.
    """

    df['datetime'] = df['datetime'] + pd.to_timedelta(df['hour'], unit='h')
    df.drop(columns=['hour'], inplace=True)
    df = df.set_index('datetime')

    if WoW:
        df = df.groupby(pd.Grouper(freq='W')).sum()
    else:
        df = df.groupby(pd.Grouper(freq=hour+'h', offset='3h00min')).sum()
    df = df.reset_index()

    return df
