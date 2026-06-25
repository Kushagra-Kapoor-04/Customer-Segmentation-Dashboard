"""
K-Means clustering, elbow method, and PCA visualisation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def find_optimal_k(X_scaled: np.ndarray,
                   k_range=range(2, 11)) -> list[float]:
    """
    Run K-Means for each k in *k_range*, record inertia, save an elbow plot.

    Returns the list of inertia values.
    """
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, marker="o", color="#6366f1",
             linewidth=2, markersize=8)
    plt.fill_between(list(k_range), inertias, alpha=0.08, color="#6366f1")
    plt.xlabel("Number of clusters (k)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method — Optimal k", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "elbow_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    return inertias


def run_kmeans(rfm_df: pd.DataFrame,
               n_clusters: int = 4) -> tuple:
    """
    Scale features, fit K-Means, project onto 2 PCA components.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Must contain columns Recency, Frequency, Monetary.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    rfm_df : pd.DataFrame
        Original df with added Cluster, PC1, PC2 columns.
    scaler : StandardScaler
    km : KMeans
    pca : PCA
    """
    features = rfm_df[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df = rfm_df.copy()
    rfm_df["Cluster"] = km.fit_predict(X_scaled)

    # PCA for 2-D visualisation
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    rfm_df["PC1"] = components[:, 0]
    rfm_df["PC2"] = components[:, 1]

    return rfm_df, scaler, km, pca


def save_pca_plot(rfm_df: pd.DataFrame) -> None:
    """Save a PCA scatter plot coloured by Cluster to reports/."""
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = ["#6366f1", "#f59e0b", "#10b981", "#ef4444",
               "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6"]

    for cluster_id in sorted(rfm_df["Cluster"].unique()):
        mask = rfm_df["Cluster"] == cluster_id
        label = rfm_df.loc[mask, "Segment"].iloc[0] if "Segment" in rfm_df.columns \
            else f"Cluster {cluster_id}"
        ax.scatter(
            rfm_df.loc[mask, "PC1"],
            rfm_df.loc[mask, "PC2"],
            label=label,
            alpha=0.55,
            s=18,
            color=palette[cluster_id % len(palette)],
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title("Customer Clusters — PCA Projection", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(REPORTS_DIR, "cluster_pca.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
