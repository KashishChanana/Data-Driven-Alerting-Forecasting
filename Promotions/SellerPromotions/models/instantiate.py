from models.gnb import *
from models.decisionTrees import *
from models.knn import *
from models.logisticRegression import *
from models.neuralNets import *
from models.randomForest import *
from models.svm import *
import warnings
warnings.filterwarnings("ignore")

def instantiate_model(X_train, X_test, y_train, y_test):
    """
    :param X_train: DataFrame with feature columns of train set
    :param X_test: Series with target column of train set
    :param y_train: DataFrame with feature columns of test set
    :param y_test: Series with target column of test set
    :return: consolidated lists of accuracies, f1_scores, recall scores, precision scores
    """
    gnb, acc_gnb, f1_gnb, r_gnb, p_gnb = gnb_model(X_train, X_test, y_train, y_test)
    print(gnb, acc_gnb, f1_gnb, r_gnb, p_gnb)

    dt, acc_dt, f1_dt, r_dt, p_dt = dt_model(X_train, X_test, y_train, y_test)
    print(dt, acc_dt, f1_dt, r_dt, p_dt)

    logreg, acc_logreg, f1_logreg, r_logreg, p_logreg = lr_model(X_train, X_test, y_train, y_test)
    print(logreg, acc_logreg, f1_logreg, r_logreg, p_logreg)

    rf, acc_rf, f1_rf, r_rf, p_rf = rf_model(X_train, X_test, y_train, y_test)
    print(rf, acc_rf, f1_rf, r_rf, p_rf)

    knn, acc_knn, f1_knn, r_knn, p_knn = knn_model(X_train, X_test, y_train, y_test)
    print(knn, acc_knn, f1_knn, r_knn, p_knn)

    svm, acc_svm, f1_svm, r_svm, p_svm = svm_model(X_train, X_test, y_train, y_test)
    print(svm, acc_svm, f1_svm, r_svm, p_svm)

    model, acc_nn, f1_nn, r_nn, p_nn = nn_model(X_train, X_test, y_train, y_test)
    print(model, acc_nn, f1_nn, r_nn, p_nn)

    clf = ["Gaussian Naive Bayes", "Logistic Regression", "Support Vector Machines", "Decision Trees", "Random Forest",
           "K Nearest Neighbors", "Neural Networks"]
    acc = [acc_gnb, acc_logreg, acc_svm, acc_dt, acc_rf, acc_knn, acc_nn]
    f1 = [f1_gnb, f1_logreg, f1_svm, f1_dt, f1_rf, f1_knn, f1_nn]
    recall = [r_gnb, r_logreg, r_svm, r_dt, r_rf, r_knn, r_nn]
    precision = [p_gnb, p_logreg, p_svm, p_dt, p_rf, p_knn, p_nn]

    return clf, acc, f1, recall, precision
