"""
Assign business-meaningful labels to K-Means clusters based on RFM means.

The mapping is determined *after* inspecting cluster means from a training
run — see main.py.
"""
import pandas as pd

# ── Default label map (will be overridden dynamically in main.py) ───────────
CLUSTER_LABELS: dict[int, str] = {
    0: "Champions",
    1: "At Risk",
    2: "New Customers",
    3: "Hibernating",
}

# Descriptions used in the Streamlit sidebar
SEGMENT_DESCRIPTIONS: dict[str, str] = {
    "Champions":     "💎 Low recency · High frequency · High spend",
    "Loyal":         "🏆 Moderate recency · High frequency · High spend",
    "At Risk":       "⚠️ High recency · Was frequent before",
    "New Customers": "🌱 Low recency · Low frequency · Low spend",
    "Hibernating":   "💤 High recency · Low frequency · Low spend",
    "Big Spenders":  "💰 Moderate recency · Low frequency · Very high spend",
}


def auto_label_clusters(rfm_df: pd.DataFrame) -> dict[int, str]:
    """
    Inspect cluster means and heuristically assign a business label.

    Heuristic priority
    ------------------
    1. Lowest mean Recency + highest Frequency  → Champions
    2. Highest mean Monetary + low Frequency    → Big Spenders
    3. Lowest Recency + lowest Frequency        → New Customers
    4. Highest Recency + higher Frequency       → At Risk
    5. Remaining                                → Hibernating

    Returns a dict mapping cluster_id → label string.
    """
    means = (
        rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    label_map: dict[int, str] = {}
    assigned: set[int] = set()

    def _pick(series_name: str, ascending: bool = True):
        """Return the cluster id with min/max of *series_name* not yet assigned."""
        candidates = means.drop(index=list(assigned), errors="ignore")[series_name]
        if candidates.empty:
            return None
        return candidates.sort_values(ascending=ascending).index[0]

    # Champions: lowest recency among high-frequency clusters
    freq_median = means["Frequency"].median()
    high_freq = means[means["Frequency"] >= freq_median].drop(
        index=list(assigned), errors="ignore"
    )
    if not high_freq.empty:
        champ = high_freq["Recency"].idxmin()
        label_map[champ] = "Champions"
        assigned.add(champ)

    # At Risk: highest recency among remaining high-frequency
    remaining_hf = means[
        (means["Frequency"] >= freq_median)
        & (~means.index.isin(assigned))
    ]
    if not remaining_hf.empty:
        at_risk = remaining_hf["Recency"].idxmax()
        label_map[at_risk] = "At Risk"
        assigned.add(at_risk)

    # New Customers: lowest recency among remaining
    new_cust = _pick("Recency", ascending=True)
    if new_cust is not None:
        label_map[new_cust] = "New Customers"
        assigned.add(new_cust)

    # Remaining clusters → Hibernating (or Big Spenders if monetary is high)
    for cid in means.index:
        if cid not in assigned:
            label_map[cid] = "Hibernating"
            assigned.add(cid)

    return label_map


def assign_labels(rfm_df: pd.DataFrame,
                  label_map: dict[int, str]) -> pd.DataFrame:
    """Add a 'Segment' column based on *label_map*."""
    rfm_df = rfm_df.copy()
    rfm_df["Segment"] = rfm_df["Cluster"].map(label_map)
    return rfm_df
