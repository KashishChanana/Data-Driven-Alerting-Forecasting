from sklearn.model_selection import train_test_split
test_size = 0.20
random_state = 42

def split(X, y):
    """
    Split feature and target variables into train and test sets
    :param X: Features
    :param y: Target variable
    :return: train-test split of X and y in 80:20 ratio
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("Total X data \t \t", X.shape)
    print("Total Training X data \t",X_train.shape)
    print("Total Testing X data \t",X_test.shape)
    print("Total Training y labels\t",y_train.shape)
    print("Total Testing y labels\t ",y_test.shape)
    return X_train, X_test, y_train, y_test