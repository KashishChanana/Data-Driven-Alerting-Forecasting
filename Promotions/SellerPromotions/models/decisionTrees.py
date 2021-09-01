from models.run import *
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def dt_model(X_train, X_test, y_train, y_test):
    """
    Instantiates decision tree classifier, performs hyperparameter tuning and trains & tests the model
    :param X_train: Training Dataframe containing features
    :param X_test: Testing Dataframe containing features
    :param y_train: Training labels (target column)
    :param y_test: Testing labels (target column)
    :return: dt: classifier
             acc : accuracy obtained by the classifier on test set
             f1 : f1 score obtained by the classifier on test set
             recall : recall obtained by the classifier on test set
             precision : precision obtained by the classifier on test set
    """

    parameters = {"criterion": ("gini", "entropy"),
                  "max_features": [0.5, 0.75, None],
                  "max_depth": [8, 10, None],
                  "min_samples_split": [100, 20, 2],
                  "min_samples_leaf": [50, 10, 1]}

    DT = DecisionTreeClassifier(random_state=0)
    clf = GridSearchCV(DT, parameters, scoring="roc_auc", cv=4, n_jobs=4, verbose=2)

    dt, acc, f1, recall, precision = classifier(clf, 1001460, X_train, X_test, y_train, y_test)

    return dt, acc, f1, recall, precision
