from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_utils import load_customer_data


DEFAULT_INPUT_PATH = Path("data") / "Mall_Customers.csv"
DEFAULT_OUTPUT_DIR = Path("outputs")
FEATURE_COLUMNS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer segmentation using K-means clustering"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to Mall_Customers.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store generated outputs",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Maximum number of clusters to evaluate",
    )
    return parser.parse_args()


def compute_wcss(features_scaled: pd.DataFrame, max_k: int) -> list[float]:
    wcss = []
    for n_clusters in range(1, max_k + 1):
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        model.fit(features_scaled)
        wcss.append(model.inertia_)
    return wcss


def choose_best_k(wcss: list[float], features_scaled: pd.DataFrame, max_k: int) -> int:
    x_values = list(range(1, max_k + 1))
    knee = KneeLocator(x_values, wcss, curve="convex", direction="decreasing")
    if knee.knee is not None:
        return int(knee.knee)

    best_k = 2
    best_score = -1.0
    for n_clusters in range(2, max_k + 1):
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = n_clusters
    return best_k


def compute_silhouette_scores(
    features_scaled: pd.DataFrame, max_k: int
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for n_clusters in range(2, max_k + 1):
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(features_scaled)
        scores[n_clusters] = silhouette_score(features_scaled, labels)
    return scores


def train_kmeans(features_scaled: pd.DataFrame, n_clusters: int) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    model.fit(features_scaled)
    return model


def save_elbow_plot(wcss: list[float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(wcss) + 1), wcss, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("WCSS")
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_plot.png", dpi=200)
    plt.close()


def save_cluster_plot(df_clustered: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_clustered,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="tab10",
        s=90,
    )
    plt.title("Customer Segments")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_plot.png", dpi=200)
    plt.close()


def save_cluster_summary(df_clustered: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        df_clustered.groupby("Cluster")
        .agg(
            customer_count=("CustomerID", "count"),
            avg_age=("Age", "mean"),
            avg_income_k=("Annual Income (k$)", "mean"),
            avg_spending_score=("Spending Score (1-100)", "mean"),
        )
        .round(2)
        .sort_index()
    )
    summary.to_csv(output_dir / "cluster_summary.csv")


def save_silhouette_scores(silhouette_scores: dict[int, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame(
        {
            "K": list(silhouette_scores.keys()),
            "silhouette_score": list(silhouette_scores.values()),
        }
    )
    scores_df.to_csv(output_dir / "silhouette_scores.csv", index=False)


def run_pipeline(input_path: Path, output_dir: Path, max_k: int) -> None:
    if max_k < 3:
        raise ValueError("max_k must be at least 3.")

    df = load_customer_data(input_path)

    features = df[FEATURE_COLUMNS]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    wcss = compute_wcss(features_scaled, max_k=max_k)
    silhouette_scores = compute_silhouette_scores(features_scaled, max_k=max_k)
    best_k = choose_best_k(wcss, features_scaled, max_k=max_k)

    model = train_kmeans(features_scaled, n_clusters=best_k)
    df_clustered = df.copy()
    df_clustered["Cluster"] = model.labels_

    output_dir.mkdir(parents=True, exist_ok=True)
    df_clustered.to_csv(output_dir / "clustered_customers.csv", index=False)

    save_elbow_plot(wcss, output_dir)
    save_cluster_plot(df_clustered, output_dir)
    save_cluster_summary(df_clustered, output_dir)
    save_silhouette_scores(silhouette_scores, output_dir)

    print(f"Optimal K selected: {best_k}")
    print("Silhouette scores by K:")
    for n_clusters, score in silhouette_scores.items():
        print(f"  K={n_clusters}: {score:.4f}")
    print(f"Selected K silhouette score: {silhouette_scores[best_k]:.4f}")
    print(f"Saved outputs to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    run_pipeline(args.input, args.output_dir, args.max_k)


if __name__ == "__main__":
    main()
