import tensorflow as tf
import numpy as np
import sys
sys.path.append('../pricingl/utils')
from time_features import *

BATCH_SIZE = 64
BUFFER_SIZE = 100
WINDOW_LENGTH = 24


def window_data(X, Y, window=7):
    '''
    The dataset length will be reduced to guarantee all samples have the window, so new length will be len(dataset)-window
    '''
    x = []
    y = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
        y.append(Y[i])
    return np.array(x), np.array(y)

def lstm(X_train, X_test, y_train, y_test):

    # Since we are doing sliding, we need to join the datasets again of train and test
    X_w = np.concatenate((X_train, X_test))
    y_w = np.concatenate((y_train, y_test))

    X_w, y_w = window_data(X_w, y_w, window=WINDOW_LENGTH)
    X_train_w = X_w[:-len(X_test)]
    y_train_w = y_w[:-len(X_test)]
    X_test_w = X_w[-len(X_test):]
    y_test_w = y_w[-len(X_test):]

    # Check we will have same test set as in the previous models, make sure we didnt screw up on the windowing
    print(f"Test set equal: {np.array_equal(y_test_w,y_test)}")

    train_data = tf.data.Dataset.from_tensor_slices((X_train_w, y_train_w))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((X_test_w, y_test_w))
    val_data = val_data.batch(BATCH_SIZE).repeat()
    return X_train_w.shape[-2:], train_data, val_data, X_test_w

def train(shape, train_data, val_data):
    dropout = 0.0
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(
            256, input_shape=shape, dropout=dropout),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='rmsprop', loss='mae')

    EVALUATION_INTERVAL = 200
    EPOCHS = 20

    model_history = simple_lstm_model.fit(train_data, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data, validation_steps=50)
    return simple_lstm_model, model_history

def lstm_predictions(df_train, df_test, fee_column):
    """
    Performs forecasting using Bayesian Ridge Regression model.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param fee_column : name of the fee field being analyzed

    Returns
    ----------------
    Predictions from Bayesian Ridge Regression model.
    """
    df_train_new, y_train = create_time_features(df_train, target=fee_column)
    df_test_new, y_test = create_time_features(df_test, target=fee_column)

    shape, train_data, val_data , X_test_w= lstm(df_train_new, df_test_new, y_train, y_test)

    simple_lstm_model, model_history = train(shape, train_data, val_data)
    print(model_history)

    yhat = simple_lstm_model.predict(X_test_w).reshape(1, -1)[0]
    return yhat

