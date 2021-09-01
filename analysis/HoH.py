import sys
sys.path.append('../')

from connection import *
from query.run_query import *
from utils.preprocess import *
from query.generate_query import *
from utils.evaluate import *
from run import *
from notify import *

def HOH_analysis(df, fee_column, listing_site_id, fee_code, model_name, hours=None):
    """
    Performs hourly analysis based on hours entered.

    Parameters
    ----------------
    :param df: whole dataframe
    :param fee_column : name of the fee field being analyzed
    :param listing_site_id: listing site id on eBay
    :param fee_code: code of fee_type being analyzed
    :param hours : number of hours aggregation is performed on
    :param model_name : name of the model

    Returns
    ----------------
    Dictionary with trends, alerts and forecasts.
    """

    df = aggregate(df, hours)
    df_train, df_test = train_test_split(df, hour=hours)

    # Get predictions from the above specified model
    predictionsDict, predictionsLower, predictionsUpper, alerts, forecast = run_model(model_name, df, df_train, df_test, fee_column, fee_code, listing_site_id, hours)

    if model_name == "Prophet-Multi":
        HOH_data = {"train_dates": df_train["datetime"], "y_train": df_train[fee_column], "test_dates": df_test["datetime"],
                   "y_test": df_test[fee_column], "predictions": predictionsDict[model_name],
                    "lower_bound": predictionsLower[model_name], "upper_bound": predictionsUpper[model_name]}

        last_30_days = [key for key in alerts.keys() if key >= datetime.datetime.now() - datetime.timedelta(30)]
        HOH_alerts = [alerts[key] for key in last_30_days]
        HOH_forecast = forecast
        # print(HOH_forecast)

        last_24_hours = [key for key in alerts.keys() if key >= datetime.datetime.now() - datetime.timedelta(hours=48)]
        HOH_emails = [alerts[key] for key in last_24_hours]
        send_email_notifications(fee_code, listing_site_id, HOH_emails)

        HOH = {"HOH_data": HOH_data,
               "HOH_alerts": HOH_alerts,
               "HOH_forecast": HOH_forecast
               }

        return HOH
    else:
        evaluationsDict = rmse(predictionsDict, df_test, fee_column)
        bar_plot(evaluationsDict)



