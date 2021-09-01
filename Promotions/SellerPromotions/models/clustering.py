from sklearn.cluster import AgglomerativeClustering

def agglomerative_clustering(df_clusters, df):
    """
    Performs Hierarchical clustering on the dataframe
    :param df_clusters: dataframe with selected features
    :param df: master dataframe containing feature and target columns
    :return: Two clusters found after hierarchical clustering
    """
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(df_clusters)
    df_clusters["label"] = cluster.labels_

    X_HAC_0 = df[df_clusters["label"] == 0]
    X_HAC_1 = df[df_clusters["label"] == 1]

    return X_HAC_0 , X_HAC_1
