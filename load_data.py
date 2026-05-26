# load_data.py
import os
import pandas as pd

def load_dataset(dataset_name, censorship_level, replication):
    """
    Load a dataset given its name, censorship level, and replication number.
    Looks for the file under:
        ./data/{dataset_name}/{dataset_name}_{censorship_level}_repl_{replication}.csv

    Args:
        dataset_name (str): e.g. 'pbc', 'gbmlgg', 'flchain', 'support', 'nacd', 'metabric'
        censorship_level (int): e.g. 10, 30, 50, 70, 90
        replication (int): replication number (e.g. 1–5)

    Returns:
        df (pd.DataFrame): loaded dataset
        features_col (list[str]): feature columns (numerical, excluding time/event/true_time)
    """

    base_path = f"./data/{dataset_name}/{dataset_name}_{censorship_level}_repl_{replication}.csv"

    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f" Dataset file not found at path: {base_path}\n"
            f"Make sure your data is organized as: ./data/{dataset_name}/{dataset_name}_{censorship_level}_repl_{replication}.csv"
        )

    print(f" Loading dataset from: {base_path}")
    df = pd.read_csv(base_path)
    df.columns = df.columns.str.strip()

    exclude_cols = ['time', 'event', 'true_time']

    # Keep only numeric feature columns
    features_col = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f" Loaded {len(df)} samples | {len(features_col)} features | censorship={censorship_level}% | replication={replication}")
    return df, features_col
