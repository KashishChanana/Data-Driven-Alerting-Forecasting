import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def elbow_method(df_clusters):
    """
    Plots two elbow method plots to check how many clusters inherently exist in the dataset
    :param df_clusters: dataframe with selected features
    :return: Displays charts indicating the number of clusters
    """
    distortions = []
    inertias = []
    mapping1 = []
    mapping2 = []
    K = range(1, 10)

    for k in K:

        kmeanModel = KMeans(n_clusters=k).fit(df_clusters)
        kmeanModel.fit(df_clusters)

        distortions.append(sum(np.min(cdist(df_clusters, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / df_clusters.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1.append(sum(np.min(cdist(df_clusters, kmeanModel.cluster_centers_,
                                         'euclidean'), axis=1)) / df_clusters.shape[0])
        mapping2.append(kmeanModel.inertia_)

    plt.plot(K, mapping1, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.ylabel("Sqaured Error (Cost)")
    plt.show()

    plt.plot(K, mapping2, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.ylabel("Inertia")
    plt.show()