from models.run import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def lr_model(X_train, X_test, y_train, y_test):
    """
    Instantiates decision tree classifier, performs hyperparameter tuning and trains & tests the model
    :param X_train: Training Dataframe containing features
    :param X_test: Testing Dataframe containing features
    :param y_train: Training labels (target column)
    :param y_test: Testing labels (target column)
    :return: lr: classifier
             acc : accuracy obtained by the classifier on test set
             f1 : f1 score obtained by the classifier on test set
             recall : recall obtained by the classifier on test set
             precision : precision obtained by the classifier on test set
    """
    parameters = {"penalty": ["l1", "l2"],
                  "C": [1, 10, 100, 1000]}
    LogReg = LogisticRegression(solver='liblinear', random_state=42)
    clf = GridSearchCV(LogReg, parameters, cv=3, n_jobs=-1, verbose=3)
    logreg, acc, f1, recall, precision = classifier(clf, 1001460, X_train, X_test, y_train, y_test)

    return logreg, acc, f1, recall, precision
