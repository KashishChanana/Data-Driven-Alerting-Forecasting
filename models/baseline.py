
def baseline(df_train, df_test, fee_column):
    """
    Sets baseline performance.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Baseline predictions.
    """
    predictions =[]
    predictions.append(df_train[fee_column].iloc[-1])

    for i in range(len(df_test)-1):
        predictions.append(df_test[fee_column].iloc[-1])

    return predictions

