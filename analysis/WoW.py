import sys
sys.path.append('../')

from connection import *
from query.run_query import *
from utils.preprocess import *
from utils.plot import *
from query.generate_query import *
from utils.evaluate import *
from notify import *
from run import *

def WOW_analysis(df, fee_column, listing_site_id, fee_code, model_name):
    """"
    Performs hourly analysis based on hours entered.

    Parameters
    ----------------
    :param df: whole dataframe
    :param fee_column : name of the fee field being analyzed
    :param listing_site_id: listing site id on eBay
    :param fee_code: code of fee_type being analyzed
    :param model_name : name of the model

    Returns
    ----------------
    Dictionary with trends, alerts and forecasts.
    """

    df = aggregate(df, WoW=True)
    df_train, df_test = train_test_split(df)

    # Get predictions from the above specified model
    predictionsDict, predictionsLower, predictionsUpper, alerts, forecast = run_model(model_name, df, df_train, df_test, fee_column, fee_code, listing_site_id)

    WOW_data = {"train_dates": df_train["datetime"], "y_train": df_train[fee_column], "test_dates": df_test["datetime"],
                "y_test": df_test[fee_column], "predictions": predictionsDict[model_name]}

    last_30_days = [key for key in alerts.keys() if key >= datetime.datetime.now() - datetime.timedelta(30)]
    WOW_alerts = [alerts[key] for key in last_30_days]

    WOW_forecast = forecast
    last_24_hours = [key for key in alerts.keys() if key >= datetime.datetime.now() - datetime.timedelta(hours=48)]
    WOW_emails = [alerts[key] for key in last_24_hours]
    send_email_notifications(fee_code, listing_site_id, WOW_emails)

    WOW = {"WOW_data": WOW_data,
           "WOW_alerts": WOW_alerts,
           "WOW_forecast": WOW_forecast
           }

    return WOW


