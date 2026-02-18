from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]


def load_customer_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the Mall Customers dataset."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at '{path}'. Place Mall_Customers.csv in the data folder."
        )

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing)
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        df = df.dropna(subset=REQUIRED_COLUMNS)

    return df
