"""
CLI entry point: run the full RFM + K-Means pipeline and save outputs.
"""
import os
import sys

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.load_data import load_and_merge
from src.rfm import compute_rfm
from src.cluster import find_optimal_k, run_kmeans, save_pca_plot
from src.labels import auto_label_clusters, assign_labels
from sklearn.preprocessing import StandardScaler


def main():
    os.makedirs("reports", exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────
    print("📂  Loading data...")
    orders, payments = load_and_merge()
    print(f"    Orders : {len(orders):,} rows")
    print(f"    Payments: {len(payments):,} rows")

    # ── 2. RFM ──────────────────────────────────────────────────────────
    print("\n📊  Computing RFM scores...")
    rfm = compute_rfm(orders, payments)
    print(f"    Unique customers: {len(rfm):,}")
    print(f"    Recency  — mean {rfm['Recency'].mean():.0f} days, "
          f"median {rfm['Recency'].median():.0f} days")
    print(f"    Frequency — mean {rfm['Frequency'].mean():.2f}")
    print(f"    Monetary  — mean R${rfm['Monetary'].mean():,.2f}")

    # ── 3. Elbow plot ───────────────────────────────────────────────────
    print("\n🔍  Running elbow method (k = 2..10)...")
    features = rfm[["Recency", "Frequency", "Monetary"]]
    X = StandardScaler().fit_transform(features)
    find_optimal_k(X)
    print("    Saved → reports/elbow_curve.png")

    # ── 4. K-Means ──────────────────────────────────────────────────────
    n_clusters = 4
    print(f"\n🤖  Running K-Means (k={n_clusters})...")
    rfm, scaler, km, pca = run_kmeans(rfm, n_clusters=n_clusters)

    # ── 5. Inspect cluster means ────────────────────────────────────────
    print("\n📋  Cluster means:")
    cluster_means = (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(1)
    )
    print(cluster_means.to_string())

    # ── 6. Auto-assign labels ───────────────────────────────────────────
    print("\n🏷️   Auto-assigning segment labels...")
    label_map = auto_label_clusters(rfm)
    for cid, name in sorted(label_map.items()):
        print(f"    Cluster {cid} → {name}")
    rfm = assign_labels(rfm, label_map)

    # ── 7. Save outputs ────────────────────────────────────────────────
    save_pca_plot(rfm)
    print("\n    Saved → reports/cluster_pca.png")

    output_csv = os.path.join("reports", "rfm_clustered.csv")
    rfm.to_csv(output_csv, index=False)
    print(f"    Saved → {output_csv}")

    # ── 8. Summary ──────────────────────────────────────────────────────
    print("\n✅  Done!  Segment breakdown:")
    print(rfm["Segment"].value_counts().to_string())
    print(f"\n🚀  Run the dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()
