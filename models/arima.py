import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import autocorrelation_plot
from pmdarima.arima.utils import ndiffs
import warnings

warnings.filterwarnings("ignore")


def arima_predictions(df_train, df_test, fee_column):
    """
    Performs forecasting using Arima model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from Arima model.
    """

    # p value
    autocorrelation_plot(df_train[fee_column])
    plt.show()

    # d value
    d = ndiffs(df_train[fee_column], test="adf")
    print("d value = ", d)

    # q value
    diff = df_train[fee_column].diff().dropna()
    plot_pacf(diff)
    plt.show()

    order = input("Enter Order : ")
    # Enter (1,1,1) as default. I

    history = [x for x in df_train[fee_column]]
    predictions = list()

    for time_step in tqdm(range(len(df_test))):
        model = ARIMA(history, order=eval(order))
        model_fit = model.fit()
        yhat = model_fit.forecast()
        predictions.append(yhat[0])
        current_obs = df_test.loc[time_step][fee_column]
        history.append(current_obs)

    return predictions
