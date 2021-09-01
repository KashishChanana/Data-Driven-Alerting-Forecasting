from models.run import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB


def gnb_model(X_train, X_test, y_train, y_test):
    """
    Instantiates decision tree classifier, performs hyperparameter tuning and trains & tests the model
    :param X_train: Training Dataframe containing features
    :param X_test: Testing Dataframe containing features
    :param y_train: Training labels (target column)
    :param y_test: Testing labels (target column)
    :return: gnb: classifier
             acc : accuracy obtained by the classifier on test set
             f1 : f1 score obtained by the classifier on test set
             recall : recall obtained by the classifier on test set
             precision : precision obtained by the classifier on test set
    """

    parameters = {}
    GNB = StratifiedKFold(n_splits = 4)
    clf = GridSearchCV(GaussianNB(), cv=GNB, param_grid=parameters)

    gnb, acc, f1, recall, precision = classifier(clf, 1001460 , X_train, X_test, y_train, y_test)

    return gnb, acc, f1, recall, precision
