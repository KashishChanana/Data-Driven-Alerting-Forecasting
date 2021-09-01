from models.run import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


def rf_model(X_train, X_test, y_train, y_test):
    """
    Instantiates decision tree classifier, performs hyperparameter tuning and trains & tests the model
    :param X_train: Training Dataframe containing features
    :param X_test: Testing Dataframe containing features
    :param y_train: Training labels (target column)
    :param y_test: Testing labels (target column)
    :return: rf: classifier
             acc : accuracy obtained by the classifier on test set
             f1 : f1 score obtained by the classifier on test set
             recall : recall obtained by the classifier on test set
             precision : precision obtained by the classifier on test set
    """
    parameters = {"bootstrap": [True],
                  "max_depth": [2, 6, 10],
                  "max_features": [0.5, 1, 2],
                  "min_samples_leaf": [1, 5],
                  "min_samples_split": [2, 5],
                  "n_estimators": [10, 20, 50, 150]}
    RF = RandomForestClassifier()
    clf = GridSearchCV(RF, parameters, scoring="roc_auc", cv=4, n_jobs=4, verbose=2)

    rf, acc, f1, recall, precision = classifier(clf, 1001460, X_train, X_test, y_train, y_test)

    return rf, acc, f1, recall, precision
