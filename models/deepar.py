from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
import pandas as pd
import sys
sys.path.append('../pricingml/utils')
from time_features import *
from preprocess import *
from interpretibility import *

def DeepAR_predictions(df_train, df_test, fee_column):

    """
    Performs forecasting using Amazon's DeepAR model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from DeepAR model.
    """

    df_train_new, y_train = create_time_features(df_train, target=fee_column, drop_date=False)

    df_test_new, y_test = create_time_features(df_test, target=fee_column, drop_date=False)

    df_train_new_scaled, df_test_new_scaled = scale_data(df_train_new, df_test_new)

    start_train = pd.Timestamp("2020-07-01 00:00:00")
    start_test = pd.Timestamp("2021-05-02 00:00:00")

    training_data = ListDataset(
        [{"start": start_train, "target": y_train,
          'feat_dynamic_real': [df_train_new_scaled[feature] for feature in df_train_new_scaled.columns]
          }],
        freq="d"
    )
    test_data = ListDataset(
        [{"start": start_test, "target": y_test,
          'feat_dynamic_real': [df_test_new_scaled[feature] for feature in df_test_new_scaled.columns]
          }],
        freq="d"
    )

    estimator = DeepAREstimator(freq="d",
                                prediction_length=1,
                                context_length=1,
                                cell_type='lstm',
                                num_layers=2,
                                num_cells=128,
                                trainer=Trainer(epochs=15))

    predictor = estimator.train(training_data=training_data)

    forecast_it, ts_it = make_evaluation_predictions(
        test_data, predictor=predictor, num_samples=len(df_test_new_scaled))

    forecasts = list(forecast_it)
    tss = list(ts_it)

    yhat = forecasts[0].samples.reshape(1, -1)[0]

    return yhat


