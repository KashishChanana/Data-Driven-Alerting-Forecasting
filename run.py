import pandas as pd

from utils.plot import *
from models.arima import *
from models.sarimax import *
from models.auto_arima import *
from models.prophet import *
from models.ses import *
from models.hwes import *
from models.regression import *
from models.random_forest import *
from models.xgboost import *
from models.lgbm import *
from models.prophet_multi import *
from models.deepar import *
from models.lstm import *
from models.baseline import *
import sys

sys.path.append('../pricingml/utils')
from dashboard import *


def run_model(model_name, df, df_train, df_test, fee_column, fee_code, listing_site_id, hours=None):
    """
    Runs the desired model(s) and plots trends.

    Parameters
    ----------------
    :param model_name : name of the model
    :param df : whole dataframe
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed
    :param fee_code: code of fee_type being analyzed
    :param listing_site_id : listing site id on eBay of the site being analyzed
    :param hour:

    Returns
    ----------------
    Key-value pairs of model name and it's corresponding predictions.

    """
    predictionsDict = {}
    alerts = {}
    forecast_by_model = pd.DataFrame()
    predictionsDict["Baseline"] = baseline(df_train, df_test, fee_column)
    predictionsLower = {}
    predictionsUpper = {}
    alertsDict = {}
    forecastDict = {}

    if model_name == "All":
        predictionsDict["Arima"] = arima_predictions(df_train, df_test, fee_column)
        predictionsDict["Auto-Arima"] = auto_arima_predictions(df_train, df_test, fee_column)
        predictionsDict["Sarimax"] = sarimax_predictions(df_train, df_test, fee_column)
        predictionsDict["Prophet"], predictionsLower["Prophet"], predictionsUpper[
            "Prophet"], forecastDict["Prophet"] = prophet_predictions(df, df_test,
                                                       fee_column)
        predictionsDict["SES"] = ses_predictions(df_train, df_test, fee_column)
        predictionsDict["HWES"] = hwes_predictions(df_train, df_test, fee_column)
        predictionsDict["Regression"] = regression_predictions(df_train, df_test, fee_column)
        predictionsDict["Lasso"] = lasso_predictions(df_train, df_test, fee_column)
        predictionsDict["Random Forest"] = rforest_predictions(df_train, df_test, fee_column)
        predictionsDict["XGBoost"] = xgboost_predictions(df_train, df_test, fee_column)
        predictionsDict["LGBM"] = lgbm_predictions(df_train, df_test, fee_column)
        predictionsDict["Prophet-Multi"], predictionsLower["Prophet-Multi"], predictionsUpper[
            "Prophet-Multi"], forecastDict["Prophet-Multi"] = prophet_multi_predictions(df, df_test, fee_column, hours)
        predictionsDict["DeepAR"] = DeepAR_predictions(df_train, df_test, fee_column)

        for key in predictionsDict.keys():
            if key in predictionsLower.keys() and key in predictionsLower.keys():
                comparison_plot(df_train, df_test, predictionsDict[key], predictionsLower[key], predictionsUpper[key],
                                key,
                                fee_code,
                                listing_site_id, all_trends=True)
            else:
                comparison_plot(df_train, df_test, predictionsDict[key], None, None,
                                key,
                                fee_code,
                                listing_site_id, all_trends=True)

        return predictionsDict, predictionsLower, predictionsUpper, alerts, forecastDict

    if model_name == "SES":
        predictions = ses_predictions(df_train, df_test, fee_column)

    elif model_name == "HWES":
        predictions = hwes_predictions(df_train, df_test, fee_column)

    elif model_name == "Arima":
        predictions = arima_predictions(df_train, df_test, fee_column)

    elif model_name == "Auto-Arima":
        predictions = auto_arima_predictions(df_train, df_test, fee_column)

    elif model_name == "Sarimax":
        predictions = sarimax_predictions(df_train, df_test, fee_column)

    elif model_name == "Prophet":
        predictions, predictions_lower, predictions_upper, forecast = prophet_predictions(df, df_test, fee_column)

        for test_idx in range(len(df_test)):
            if (df_test.loc[test_idx][fee_column] < predictions_lower[test_idx]) or (
                    df_test.loc[test_idx][fee_column] > predictions_upper[test_idx]):
                content = "ML based alert for fee code = {} on listing site id = {} on {}. The observed value is {:.2f}. The lower bound of forecast is {:.2f}. The upper bound of forecast is {:.2f} .".format(
                    fee_code, listing_site_id, df_test.loc[test_idx].datetime, df_test.loc[test_idx][fee_column],
                    predictions_lower[test_idx], predictions_upper[test_idx])

                alerts[df_test.loc[test_idx].datetime] = content
        forecast_by_model = forecast

    elif model_name == "Prophet-Multi":

        predictions, predictions_lower, predictions_upper, forecast = prophet_multi_predictions(df, df_test, fee_column,
                                                                                                hours)
        for test_idx in range(len(predictions)):
            if (df_test.loc[test_idx][fee_column] < predictions_lower[test_idx]) or (
                    df_test.loc[test_idx][fee_column] > predictions_upper[test_idx]):
                content = "ML based alert for fee code = {} on listing site id = {} on {}. The observed value is {:.2f}. The lower bound of forecast is {:.2f}. The upper bound of forecast is {:.2f} .".format(
                    fee_code, listing_site_id, df_test.loc[test_idx].datetime, df_test.loc[test_idx][fee_column],
                    predictions_lower[test_idx], predictions_upper[test_idx])
                alerts[df_test.loc[test_idx].datetime] = content
        forecast_by_model = forecast
        predictionsLower[model_name] = predictions_lower
        predictionsUpper[model_name] = predictions_upper

    elif model_name == "Regression":
        predictions = regression_predictions(df_train, df_test, fee_column)

    elif model_name == "Lasso":
        predictions = lasso_predictions(df_train, df_test, fee_column)
        for test_idx in range(len(df_test)):
            if (df_test.loc[test_idx][fee_column] < predictions[test_idx] * 0.90) or (
                    df_test.loc[test_idx][fee_column] > predictions[test_idx] * 1.10):
                content = "ML based alert for fee code = {} on listing site id = {} on {}. The observed value is {:.2f}. The Lower bound of forecast is {:.2f}. The Upper bound of forecast is {:.2f} .".format(
                    fee_code, listing_site_id, df_test.loc[test_idx].datetime, df_test.loc[test_idx][fee_column],
                    predictions[test_idx] * 0.90, predictions[test_idx] * 1.10)
                alerts[df_test.loc[test_idx].datetime] = content

    elif model_name == 'Random Forest':
        predictions = rforest_predictions(df_train, df_test, fee_column)

    elif model_name == 'XGBoost':
        predictions = xgboost_predictions(df_train, df_test, fee_column)

    elif model_name == 'LGBM':
        predictions = lgbm_predictions(df_train, df_test, fee_column)

    elif model_name == 'DeepAR':
        predictions = DeepAR_predictions(df_train, df_test, fee_column)

    elif model_name == 'lstm':
        predictions = lstm_predictions(df_train, df_test, fee_column)

    else:
        print("Invalid Choice")
        return

    predictionsDict[model_name] = predictions

    if model_name in predictionsLower.keys() and model_name in predictionsLower.keys():
        comparison_plot(df_train, df_test, predictionsDict[model_name], predictionsLower[model_name], predictionsUpper[model_name],
                        model_name,
                        fee_code,
                        listing_site_id, all_trends=True)
    else:
        comparison_plot(df_train, df_test, predictionsDict[model_name], None, None,
                        model_name,
                        fee_code,
                        listing_site_id, all_trends=True)

    return predictionsDict, predictionsLower, predictionsUpper, alerts, forecast_by_model
