import sys

sys.path.append('../')

from connection import *
from query.run_query import *
from utils.preprocess import *
from query.generate_query import *
from utils.plot import *
from utils.evaluate import *
from run import *
from notify import *
import datetime


def DOD_analysis(listing_site_id, fee_code, model_name):
    """
    Performs Day-on-day analysis.

    Parameters
    ----------------
    :param listing_site_id: listing site id on eBay
    :param fee_code: code of fee_type being analyzed
    :param model_name : name of the model

    Returns
    ----------------
    Dictionary with trends, alerts and forecasts
    """

    query, fee_column = generate_query_multi(listing_site_id, fee_code)
    df = run_query(query, columns=['datetime', fee_column, 'total_sale_price', 'total_promo_discount'])
    df = clean(df, columns=[fee_column, "total_sale_price", 'total_promo_discount'])

    df_train, df_test = train_test_split(df)

    # Get predictions from the above specified model
    predictionsDict, predictionsLower, predictionsUpper, alerts, forecast = run_model(model_name, df, df_train, df_test, fee_column, fee_code,
                                                  listing_site_id)

    if model_name == "Prophet-Multi":

        DOD_dict = {"train_dates": df_train["datetime"], "y_train": df_train[fee_column], "test_dates": df_test["datetime"],
                    "y_test": df_test[fee_column], "predictions": predictionsDict[model_name],
                    "lower_bound": predictionsLower[model_name], "upper_bound": predictionsUpper[model_name]}

        last_30_days = [key for key in alerts.keys() if key >= datetime.datetime.now() - datetime.timedelta(30)]
        DOD_alerts = [alerts[key] for key in last_30_days]

        DOD_forecast = forecast

        last_24_hours = [key for key in alerts.keys() if key >= datetime.datetime.now() - datetime.timedelta(hours=48)]
        DOD_emails = [alerts[key] for key in last_24_hours]
        send_email_notifications(fee_code, listing_site_id, DOD_emails)

        DOD = {"DOD_data": DOD_dict,
               "DOD_alerts": DOD_alerts,
               "DOD_forecast": DOD_forecast
               }

        return DOD
    else:
        evaluationsDict = rmse(predictionsDict, df_test, fee_column)
        bar_plot(evaluationsDict)

