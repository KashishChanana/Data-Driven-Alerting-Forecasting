from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, \
    confusion_matrix


def classifier(clf, offer_id, X_train, X_test, y_train, y_test):

    """
    Method for training classifier and predicting on test set.
    :param clf: classifier ith parameters, if procurable including GridSearchCV for parameter tuning
    :param offer_id: integer with offer_id according master dataframe
    :param X_train: DataFrame with feature columns of train set
    :param X_test: Series with target column of train set
    :param y_train: DataFrame with feature columns of test set
    :param y_test: Series with target column of test set
    :return: clf: classifier
             acc : accuracy obtained by the classifier on test set
             f1 : f1 score obtained by the classifier on test set
             recall : recall obtained by the classifier on test set
             precision : precision obtained by the classifier on test set
    """

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    accuracy = round(accuracy_score(y_test, pred) * 100, 2)

    f1 = round(f1_score(y_test, pred) * 100, 2)

    recall = round(recall_score(y_test, pred) * 100, 2)

    precision = round(precision_score(y_test, pred) * 100, 2)

    print("#######################################################")

    cm = confusion_matrix(y_test, pred)
    print("Offer {} - confusion matrix:".format(offer_id))
    print(cm, "\n")

    cr = classification_report(y_test, pred, target_names=["0", "1"])
    print("Offer {} - classification report:".format(str(clf)))
    print(cr)

    print("Offer {}:".format(offer_id),
          "Accuracy: {} % | F1-score: {} % \n\
        Recall: {} % | Precision: {} %".format(accuracy, f1, recall, precision), "\n")

    return clf, accuracy, f1, recall, precision
