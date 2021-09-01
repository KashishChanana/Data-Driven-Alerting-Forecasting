from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, \
    confusion_matrix


def nn_model(X_train, X_test, y_train, y_test):
    """
    Instantiates decision tree classifier, performs hyperparameter tuning and trains & tests the model
    :param X_train: Training Dataframe containing features
    :param X_test: Testing Dataframe containing features
    :param y_train: Training labels (target column)
    :param y_test: Testing labels (target column)
    :return: model: classifier
             acc : accuracy obtained by the classifier on test set
             f1 : f1 score obtained by the classifier on test set
             recall : recall obtained by the classifier on test set
             precision : precision obtained by the classifier on test set
    """
    model = keras.Sequential(
        [layers.Dense(128, input_dim=11, activation="sigmoid", name="layer1"),
         layers.Dense(64, activation="sigmoid", name="layer2"),
         layers.Dense(1, activation="sigmoid", name="layer3"),
         ]
    )

    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',
                                                patience=4,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0001)

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=["accuracy"]  # , tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives()]
    )

    model.fit(
        X_train, y_train,
        epochs=1,
        validation_split=0.2,
        batch_size=32,
        callbacks=[learning_rate_reduction]
    )

    pred = list((model.predict(X_test)).round())

    acc = round(accuracy_score(y_test, pred) * 100, 2)

    f1 = round(f1_score(y_test, pred) * 100, 2)

    recall = round(recall_score(y_test, pred) * 100, 2)

    precision = round(precision_score(y_test, pred) * 100, 2)

    print("#######################################################")

    cm_nn = confusion_matrix(y_test, pred)
    print("Offer {} - confusion matrix:".format(1001460))
    print(cm_nn, "\n")

    cr_nn = classification_report(y_test, pred, target_names=["0", "1"])
    print("Offer {} - classification report:".format(str("Neural Networks")))
    print(cr_nn)

    print("Offer {}:".format(1001460),
          "Accuracy: {} % | F1-score: {} % \n\
        Recall: {} % | Precision: {} %".format(acc, f1, recall, precision), "\n")

    return model, acc, f1, recall, precision
