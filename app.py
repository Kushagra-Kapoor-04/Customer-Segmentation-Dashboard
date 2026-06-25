"""
🎯 E-Commerce Customer Segmentation Dashboard
RFM Analysis + K-Means Clustering on the Olist Dataset
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation — RFM + K-Means",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141432 0%, #1e1e4a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc !important;
    }

    /* ── Metric cards ────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.15);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* ── Headings ────────────────────────────────────────────── */
    .main h1 {
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    .main h2, .main h3 {
        color: #c7d2fe !important;
        font-weight: 600 !important;
    }

    /* ── Dataframe ───────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ── Divider ─────────────────────────────────────────────── */
    hr {
        border-color: rgba(99,102,241,0.15) !important;
    }

    /* ── Caption / small text ────────────────────────────────── */
    .stCaption, small {
        color: #64748b !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Colour palette (consistent across all charts) ───────────────────────────
SEGMENT_COLORS = {
    "Champions":     "#6366f1",
    "Loyal":         "#8b5cf6",
    "At Risk":       "#f59e0b",
    "New Customers": "#10b981",
    "Hibernating":   "#64748b",
    "Big Spenders":  "#ec4899",
}

SEGMENT_ICONS = {
    "Champions":     "💎",
    "Loyal":         "🏆",
    "At Risk":       "⚠️",
    "New Customers": "🌱",
    "Hibernating":   "💤",
    "Big Spenders":  "💰",
}


# ── Load pre-computed results ───────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "reports", "rfm_clustered.csv")
    if not os.path.exists(csv_path):
        st.error(
            "⚠️ `reports/rfm_clustered.csv` not found. "
            "Run `python main.py` first to generate clustering results."
        )
        st.stop()
    return pd.read_csv(csv_path)


rfm = load_data()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Customer Segmentation")
    st.caption("RFM Analysis · K-Means · PCA")
    st.divider()

    st.markdown("### Filters")
    all_segments = sorted(rfm["Segment"].unique())
    segments = st.multiselect(
        "Select Segments",
        all_segments,
        default=all_segments,
        help="Choose which customer segments to display.",
    )

    st.divider()

    # Segment legend
    st.markdown("### Segment Guide")
    for seg in all_segments:
        icon = SEGMENT_ICONS.get(seg, "📌")
        color = SEGMENT_COLORS.get(seg, "#94a3b8")
        st.markdown(
            f'<span style="color:{color}; font-weight:600;">'
            f'{icon} {seg}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "Built with Streamlit · Data from "
        "[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)"
    )

# ── Apply filter ────────────────────────────────────────────────────────────
if not segments:
    st.warning("Select at least one segment from the sidebar.")
    st.stop()

filtered = rfm[rfm["Segment"].isin(segments)]

# ── Header ──────────────────────────────────────────────────────────────────
st.title("E-Commerce Customer Segmentation")
st.caption(
    "RFM (Recency · Frequency · Monetary) analysis with K-Means clustering "
    "on ~93 000 Brazilian e-commerce customers from the Olist dataset."
)

# ── KPI row ─────────────────────────────────────────────────────────────────
st.markdown("")  # spacing
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{len(filtered):,}")
k2.metric("Avg Recency", f"{filtered['Recency'].mean():.0f} days")
k3.metric("Avg Frequency", f"{filtered['Frequency'].mean():.2f} orders")
k4.metric("Avg Monetary", f"R$ {filtered['Monetary'].mean():,.0f}")

st.markdown("")

# ── Row 1: PCA scatter + Segment pie ───────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("Customer Clusters — PCA 2D Projection")
    color_list = [SEGMENT_COLORS.get(s, "#94a3b8") for s in
                  sorted(filtered["Segment"].unique())]
    fig_pca = px.scatter(
        filtered,
        x="PC1", y="PC2",
        color="Segment",
        hover_data={"Recency": True, "Frequency": True,
                    "Monetary": ":.2f", "PC1": False, "PC2": False},
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.6,
    )
    fig_pca.update_traces(marker=dict(size=5, line=dict(width=0.3, color="white")))
    fig_pca.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        legend=dict(
            bgcolor="rgba(20,20,50,0.7)",
            bordercolor="rgba(99,102,241,0.2)",
            borderwidth=1,
            font=dict(size=12),
        ),
        xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zeroline=False),
        margin=dict(l=0, r=0, t=10, b=0),
        height=480,
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with col_right:
    st.subheader("Segment Distribution")
    seg_counts = filtered["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    fig_pie = px.pie(
        seg_counts, names="Segment", values="Count",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        hole=0.45,
    )
    fig_pie.update_traces(
        textposition="inside", textinfo="percent+label",
        textfont_size=12,
        marker=dict(line=dict(color="#0f0f23", width=2)),
    )
    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=480,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── Row 2: RFM bar charts per segment ──────────────────────────────────────
st.subheader("RFM Averages by Segment")

means = (
    filtered.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
    .mean()
    .round(1)
    .reset_index()
)

b1, b2, b3 = st.columns(3)

for col_obj, metric, fmt, prefix in [
    (b1, "Recency",   "{:.0f}", ""),
    (b2, "Frequency", "{:.2f}", ""),
    (b3, "Monetary",  "{:,.0f}", "R$ "),
]:
    with col_obj:
        fig_bar = px.bar(
            means.sort_values(metric, ascending=True),
            x=metric, y="Segment",
            orientation="h",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            text=metric,
        )
        fig_bar.update_traces(
            texttemplate=prefix + fmt.replace("{", "%{x").replace("}", "}"),
            textposition="outside",
            marker_line_width=0,
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#cbd5e1"),
            xaxis=dict(
                title=metric,
                gridcolor="rgba(99,102,241,0.08)",
                zeroline=False,
            ),
            yaxis=dict(title=""),
            showlegend=False,
            margin=dict(l=0, r=40, t=30, b=0),
            height=300,
            title=dict(text=metric, font=dict(size=14)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Row 3: Segment size bar + data table ────────────────────────────────────
col_chart, col_table = st.columns([1, 1], gap="large")

with col_chart:
    st.subheader("Segment Size")
    seg_sorted = seg_counts.sort_values("Count", ascending=True)
    fig_h = px.bar(
        seg_sorted, x="Count", y="Segment",
        orientation="h",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        text="Count",
    )
    fig_h.update_traces(
        texttemplate="%{x:,}",
        textposition="outside",
        marker_line_width=0,
    )
    fig_h.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        showlegend=False,
        xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zeroline=False,
                   title="Customers"),
        yaxis=dict(title=""),
        margin=dict(l=0, r=60, t=10, b=0),
        height=350,
    )
    st.plotly_chart(fig_h, use_container_width=True)

with col_table:
    st.subheader("RFM Summary Table")
    summary = (
        rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
        .agg(["mean", "median", "std"])
        .round(1)
    )
    summary.columns = [f"{m} ({s})" for m, s in summary.columns]
    summary = summary.reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

# ── Row 4: Static report images ────────────────────────────────────────────
st.subheader("📈 Training Artifacts")
img1, img2 = st.columns(2)

elbow_path = os.path.join(os.path.dirname(__file__), "reports", "elbow_curve.png")
pca_path = os.path.join(os.path.dirname(__file__), "reports", "cluster_pca.png")

with img1:
    if os.path.exists(elbow_path):
        st.image(elbow_path, caption="Elbow Curve", use_container_width=True)
    else:
        st.info("Elbow curve not generated yet.")

with img2:
    if os.path.exists(pca_path):
        st.image(pca_path, caption="PCA Cluster Plot", use_container_width=True)
    else:
        st.info("PCA plot not generated yet.")
