import sys
sys.path.append('../')
from connection import *
from query.run_query import *
from utils.preprocess import *
from query.generate_query import *
from utils.evaluate import *
from run import *
from notify import *
import datetime


def substier_analysis(listing_site_id, fee_code, model_name):
    """
    Performs substier-wise analysis.

    Parameters
    ----------------
    :param listing_site_id: listing site id on eBay
    :param fee_code: code of fee_type being analyzed
    :param model_name : name of the model

    Returns
    ----------------
    Dictionary with substier-wise trends, alerts and forecasts
    """
    query, fee_column = generate_query_multi_substier(listing_site_id, fee_code)
    df = run_query(query,
                   columns=['datetime', fee_column, 'total_sale_price', 'total_promo_discount', 'subs_tier'])

    df = clean(df, columns=[fee_column, "total_sale_price", 'total_promo_discount', 'subs_tier'])

    substier_data = {}
    substier_alerts = {}
    substier_forecast = {}

    for uni in df['subs_tier'].unique():

        df_uni = df[df['subs_tier'] == uni]
        df_train, df_test = train_test_split(df_uni)
        substier_predictions, predictionsLower, predictionsUpper, alerts, forecast = run_model("Lasso", df_uni, df_train, df_test, fee_column, fee_code,
                                                           listing_site_id)

        substier_data[str(uni)] = {"train_dates": df_train["datetime"], "y_train": df_train[fee_column],
                         "test_dates": df_test["datetime"],
                         "y_test": df_test[fee_column], "predictions": substier_predictions["Lasso"]}

        last_30_days = [key for key in alerts.keys() if key >= datetime.datetime.now() - datetime.timedelta(30)]
        substier_alerts[str(uni)] = [alerts[key] for key in last_30_days]
        substier_forecast[str(uni)] = forecast

    substier = {"substier_data": substier_data,
                "substier_alerts": substier_alerts,
                "substier_forecast": substier_forecast
                }
    return substier
