import os
import joblib
import pandas as pd

from preprocessing import load_data, clean_data, prepare_features
from clustering import run_pca, train_kmeans, evaluate_clustering, map_business_segments
from classification import split_data, tune_and_evaluate


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_raw_dir = os.path.join(base_dir, "../data/raw/dataset.csv")
    data_processed_dir = os.path.join(base_dir, "../data/processed")
    models_dir = os.path.join(base_dir, "../models")

    os.makedirs(data_processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("[1/5] Loading and cleaning raw dataset...")
    df = load_data(data_raw_dir)
    clean_df = clean_data(df)
    clean_df.to_csv(os.path.join(data_processed_dir, "df_clean.csv"), index=False)

    print("[2/5] Applying log1p transformation & scaling...")
    df_scaled, scaler, numeric_cols = prepare_features(clean_df)

    df_scaled_export = pd.DataFrame(df_scaled, columns=numeric_cols)
    df_scaled_export.to_csv(os.path.join(data_processed_dir, "df_prepare_clustering.csv"), index=False)

    print("[3/5] Running PCA (2 components) & K-Means (k=3)...")
    X_pca2, pca_model = run_pca(df_scaled, n_components=2)
    k_labels, kmeans_model = train_kmeans(X_pca2, n_clusters=3)

    metrics = evaluate_clustering(X_pca2, k_labels)
    print(f"Clustering Silhouette Score: {metrics['silhouette']:.4f}")

    clean_df["cluster_final"] = k_labels
    clean_df["target"] = map_business_segments(k_labels)
    clean_df = clean_data(clean_df)
    clean_df.to_csv(os.path.join(data_processed_dir, "df_clean.csv"), index=False)
    
    print("[4/5] Preparing train/test splits and tuning classifiers...")
    X = clean_df[numeric_cols].copy()
    y = clean_df["target"].copy()

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    best_model, best_name, best_score, _ = tune_and_evaluate(
        X_train, y_train, X_test, y_test
    )
    print(f"Best Model: {best_name} (F1-Weighted CV: {best_score:.4f})")

    print("[5/5] Saving pipeline artifacts into models/...")
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(pca_model, os.path.join(models_dir, "pca_model.joblib"))
    joblib.dump(kmeans_model, os.path.join(models_dir, "kmeans_model.joblib"))
    joblib.dump(best_model, os.path.join(models_dir, "best_overall_model.joblib"))

    print("Pipeline run completed successfully.")


if __name__ == "__main__":
    main()