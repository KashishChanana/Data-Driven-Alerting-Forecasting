import math
from sklearn.metrics import mean_squared_error

def rmse(predictionsDict, df_test, fee_column):
    """
    Evaluates predictions on metric - Root Mean Square Error (RMSE).

    Parameters
    ----------------
    :param predictionsDict : key-value pairs of models with predictions for test set
    :param df_test : testing dataframe
    :param fee_column : name of the fee field in the dataframe being analyzed

    Returns
    ----------------
    Key-value pairs of model name and scalar RMSE value.
    """
    evaluationsDict = {}

    ytest = df_test[fee_column].values

    for key in predictionsDict.keys():
        evaluationsDict[key] = math.sqrt(mean_squared_error(ytest, predictionsDict[key]))
        print("\n RMSE of the model " + key + " is " + str(evaluationsDict[key]))

    return evaluationsDict
