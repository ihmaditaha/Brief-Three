import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path: str = "https://drive.google.com/uc?id=1_pxeBMqFGuvGzg1lSW_Kqm4VQegsMdUs") -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    if "CUST_ID" in clean_df.columns:
        clean_df = clean_df.drop(columns=["CUST_ID"])
    clean_df = clean_df.dropna()
    return clean_df


def prepare_features(clean_df: pd.DataFrame):
    df_clustering = clean_df.copy()
    numeric_cols = df_clustering.select_dtypes(include="number").columns

    df_clustering[numeric_cols] = np.log1p(df_clustering[numeric_cols])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clustering[numeric_cols])

    return df_scaled, scaler, numeric_cols