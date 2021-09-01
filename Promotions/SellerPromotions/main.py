from utils.load import *
from utils.featureSelection import *
from utils.clustering_features import *
from utils.clustering_plots import *
from utils.clustering_elbow import *
from utils.traintestsplit import *
from utils.clustering_scaling import *
from models.instantiate import *
from models.clustering import *
import pandas as pd

df = load_data()

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = split(X, y)

instantiate_model(X_train, X_test, y_train, y_test)

clf, acc, f1, recall, precision = instantiate_model(X_train, X_test, y_train, y_test)

comparison = pd.DataFrame({
    "classifier": clf,
    "accuracy": acc,
    "f1-score": f1,
    "recall": recall,
    "precision": precision
})

comparison.set_index("classifier", inplace=True)
comparison = comparison.transpose()
comparison["high score"] = comparison[["Gaussian Naive Bayes", "Logistic Regression", "Support Vector Machines", "Decision Trees", "Random Forest", "K Nearest Neighbors", "Neural Networks"]].idxmax(axis=1)
print(comparison)


df_clusters = clustering_fs(df)
df_clusters_scaled = scale(df_clusters)
elbow_method(df_clusters_scaled)
X_HAC_0, X_HAC_1 = agglomerative_clustering(df_clusters_scaled, df)
comparison_plots(X_HAC_0, X_HAC_1)






