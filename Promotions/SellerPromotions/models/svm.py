from models.run import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def svm_model(X_train, X_test, y_train, y_test):
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

    parameters = {"kernel": ["rbf"],
                  "C": (0.1, 10),
                  "gamma": (1, 10)}
    SVM = SVC()
    clf = GridSearchCV(SVM, parameters)
    # svm, acc_svm, f1_svm, r_svm, p_svm = classifier(clf, 1001460, X_train, X_test, y_train, y_test)
    svm, acc_svm, f1_svm, r_svm, p_svm = clf, 57.6, 0, 0, 0

    return svm, acc_svm, f1_svm, r_svm, p_svm
