import pmdarima as pm
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def sarimax_predictions(df_train, df_test, fee_column):
    """
    Performs forecasting using Sarimax model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from Sarimax model.
    """

    autoModel = pm.auto_arima(df_train[fee_column], trace=True, error_action='ignore',
                              suppress_warnings=True, seasonal=True, m=6, stepwise=True)
    autoModel.fit(df_train[fee_column])

    order = autoModel.order
    seasonalOrder = autoModel.seasonal_order

    history = [x for x in df_train[fee_column]]
    predictions = list()
    yhat = list()

    for time_step in tqdm(range(len(df_test))):
        model = SARIMAX(history, order=order, seasonal_order=seasonalOrder)
        model_fit = model.fit()
        yhat = model_fit.forecast()
        predictions.append(yhat[0])
        current_obs = df_test.loc[time_step][fee_column]
        history.append(current_obs)

    return predictions
