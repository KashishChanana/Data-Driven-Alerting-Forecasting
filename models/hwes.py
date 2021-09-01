from tqdm import tqdm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def hwes_predictions(df_train, df_test, fee_column):

    """
    Performs forecasting using Holt Winter's Exponential Smoothing (HWES) model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from HWES model.
    """

    history = [x for x in df_train[fee_column]]
    predictions = list()

    for time_step in tqdm(range(len(df_test))):
        model = ExponentialSmoothing(history)
        model_fit = model.fit(smoothing_level=1, optimized=True)
        yhat = model_fit.forecast()
        predictions.append(yhat[0])
        current_obs = df_test.loc[time_step][fee_column]
        history.append(current_obs)

    return predictions
