import pmdarima as pm
from tqdm import tqdm
from statsmodels.tsa.arima_model import ARIMA


def auto_arima_predictions(df_train, df_test, fee_column):

    """
    Performs forecasting using Arima model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from Auto-Arima model.
    """

    auto_model = pm.auto_arima(df_train[fee_column], trace=True,
                               error_action='ignore', suppress_warnings=True, seasonal=False)
    auto_model.fit(df_train[fee_column])

    order = auto_model.order
    history = [x for x in df_train[fee_column]]
    predictions = list()

    for time_step in tqdm(range(len(df_test))):
        model = ARIMA(history, order=(1, 1, 2))
        model_fit = model.fit()
        yhat = model_fit.forecast()
        predictions.append(yhat[0])
        current_obs = df_test.loc[time_step][fee_column]
        history.append(current_obs)

    return predictions
