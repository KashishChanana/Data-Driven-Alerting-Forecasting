from models.run import *
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def knn_model(X_train, X_test, y_train, y_test):
    """
    Instantiates decision tree classifier, performs hyperparameter tuning and trains & tests the model
    :param X_train: Training Dataframe containing features
    :param X_test: Testing Dataframe containing features
    :param y_train: Training labels (target column)
    :param y_test: Testing labels (target column)
    :return: knn: classifier
             acc : accuracy obtained by the classifier on test set
             f1 : f1 score obtained by the classifier on test set
             recall : recall obtained by the classifier on test set
             precision : precision obtained by the classifier on test set
    """
    parameters = {"n_neighbors": [50, 100, 150],
                  "p": [0.5, 1, 2],
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan"]}
    KNN = KNeighborsClassifier()
    clf = GridSearchCV(KNN, parameters, scoring='roc_auc', cv=4, n_jobs=4, verbose=2)

    knn, acc, f1, recall, precision = classifier(clf, 1001460, X_train, X_test, y_train, y_test)

    return knn, acc, f1, recall, precision
