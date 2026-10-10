import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def run_pca(df_scaled, n_components = 2, random_state = 42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(df_scaled)
    return X_pca, pca


def train_kmeans(X_pca, n_clusters = 3, random_state = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    return labels, kmeans


def train_dbscan(X_pca, eps = 0.3, min_samples = 10):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_pca)
    return labels, dbscan


def evaluate_clustering(features, labels):
    mask = labels != -1 if -1 in labels else slice(None)
    eval_features = features[mask]
    eval_labels = labels[mask]

    if len(set(eval_labels)) <= 1:
        return {"silhouette": -1.0, "davies_bouldin": None, "calinski_harabasz": None}

    return {
        "silhouette": silhouette_score(eval_features, eval_labels),
        "davies_bouldin": davies_bouldin_score(eval_features, eval_labels),
        "calinski_harabasz": calinski_harabasz_score(eval_features, eval_labels),
    }


def map_business_segments(labels: np.ndarray) -> pd.Series:
    segment_mapping = {
        0: "Clients à Risque",
        1: "Clients Premium",
        2: "Clients à Faible Activité",
    }
    return pd.Series(labels).map(segment_mapping)