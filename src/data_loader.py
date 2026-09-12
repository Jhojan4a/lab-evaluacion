import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

DATA_PATH = os.path.join("data", "dataset.csv")

def generate_and_save_data(n_samples=1000, random_state=42):
    os.makedirs("data", exist_ok=True)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.60, 0.40],
        random_state=random_state
    )
    df = pd.DataFrame({
        "cpu_usage_pct": np.round(np.clip(50 + X[:, 0] * 15, 10, 100), 2),
        "ram_usage_gb": np.round(np.clip(16 + X[:, 1] * 8, 4, 64), 2),
        "network_throughput_mbps": np.round(np.clip(450 + X[:, 2] * 180, 50, 1000), 2),
        "temperature_celsius": np.round(np.clip(60 + X[:, 3] * 12, 35, 95), 2),
        "server_status": y
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"[+] Dataset generado en: {DATA_PATH}")
    return df

def load_data():
    if not os.path.exists(DATA_PATH):
        generate_and_save_data()
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["server_status"])
    y = df["server_status"]
    return X, y