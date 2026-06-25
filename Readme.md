# Customer Segmentation Dashboard

An end-to-end RFM analysis and K-Means clustering pipeline on ~93,000 real Brazilian e-commerce customers, with an interactive Streamlit dashboard for exploring customer segments.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)]((https://customer-segmentation-dashboard-01.streamlit.app/))
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

---

## What this project does

Rather than clustering on raw features, this project uses **RFM analysis** — the standard industry framework for customer segmentation:

- **Recency** — days since the customer last ordered
- **Frequency** — total number of orders placed
- **Monetary** — total amount spent (BRL)

These three signals are standardised, clustered with K-Means, and automatically labelled with business-meaningful segment names (Champions, At Risk, New Customers, Hibernating) using a heuristic that inspects cluster means — no hard-coded label map.

The results are visualised in a dark-themed Streamlit dashboard with interactive filters, a PCA 2D scatter, segment distribution pie, per-segment RFM bar charts, and a summary statistics table.

---

## Project structure

```
customer-segmentation/
│
├── src/
│   ├── load_data.py    # Load + merge Olist orders, payments, customers CSVs
│   ├── rfm.py          # RFM computation (delivered orders only)
│   ├── cluster.py      # K-Means, elbow method, PCA visualisation
│   └── labels.py       # Auto-assign segment labels from cluster means
│
├── reports/
│   ├── elbow_curve.png     # Optimal k visualisation
│   ├── cluster_pca.png     # 2D PCA scatter coloured by segment
│   └── rfm_clustered.csv   # Final segmented customer data (Streamlit input)
│
├── data/
│   └── .gitkeep            # CSVs not committed — see Dataset section
│
├── app.py           # Streamlit dashboard
├── main.py          # CLI: runs full pipeline and saves outputs
└── requirements.txt
```

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/Kushagra-Kapoor-04/customer-segmentation.git
cd customer-segmentation

# Install dependencies
pip install -r requirements.txt

# Add dataset CSVs to data/ (see Dataset section below)

# Step 1 — Run the full pipeline (RFM + clustering + saves reports/)
python main.py

# Step 2 — Launch the dashboard
python -m streamlit run app.py
```

---

## Dataset

**Olist Brazilian E-Commerce Dataset** — 100k+ real orders from 2016–2018.

Download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Place these files in the `data/` folder:

```
data/
├── olist_orders_dataset.csv
├── olist_order_payments_dataset.csv
└── olist_customers_dataset.csv
```

The loader handles filename variations and gives clear error messages if files are missing. Only **delivered** orders are used for RFM computation — cancelled and processing orders are excluded to avoid skewing recency and monetary values.

| Stat | Value |
|---|---|
| Total orders | ~99,000 |
| Unique customers | ~93,000 |
| Date range | Sep 2016 – Oct 2018 |
| Payment currency | BRL (Brazilian Real) |

---

## How it works

### 1. RFM computation

For each unique customer, three signals are computed against a snapshot date (max purchase date + 1 day):

```
Recency   = days since last delivered order
Frequency = count of unique delivered orders
Monetary  = sum of all payment values (BRL)
```

### 2. Clustering

Features are standardised with `StandardScaler` before clustering — required because Recency (days) and Monetary (BRL) are on very different scales.

The elbow method runs K-Means for k=2 to k=10 and plots inertia to suggest the optimal k. k=4 is used by default.

PCA reduces the 3D feature space to 2 components for visualisation, explaining the majority of variance.

### 3. Auto-labelling

Cluster labels are assigned algorithmically by inspecting RFM means — no hard-coded mapping:

- **Champions** — lowest recency among high-frequency clusters
- **At Risk** — highest recency among high-frequency clusters  
- **New Customers** — lowest recency among remaining clusters
- **Hibernating** — everything else

This means labels adapt correctly regardless of which cluster number K-Means assigns to each group across different runs.

---

## Results

| Segment | Typical profile |
|---|---|
| Champions | Bought recently, buy often, spend most |
| At Risk | Were frequent buyers, haven't returned in a while |
| New Customers | First or second purchase, low spend so far |
| Hibernating | Haven't ordered in a long time, low engagement |

Output plots saved to `reports/` after running `main.py`:

**Elbow curve** — shows inertia drop-off to justify k=4 selection.

**PCA cluster plot** — 2D projection of all ~93k customers coloured by segment, showing clear separation between Champions and Hibernating groups.

---

## Dashboard features

- Segment filter (sidebar multiselect) — drill into any combination of segments
- 4 KPI metrics — total customers, avg recency, avg frequency, avg monetary
- Interactive PCA scatter (Plotly) — hover for individual customer RFM values
- Segment distribution donut chart
- Side-by-side RFM bar charts per segment
- RFM summary table with mean, median, and std per segment
- Static training artifacts (elbow curve, PCA plot) embedded in the dashboard

---

## Key concepts demonstrated

- **RFM analysis** — industry-standard customer segmentation framework used by real e-commerce and marketing teams
- **K-Means clustering** — unsupervised ML on real-world messy data
- **Elbow method** — data-driven selection of optimal cluster count
- **PCA** — dimensionality reduction for high-dimensional cluster visualisation
- **StandardScaler** — why feature scaling is mandatory before distance-based algorithms
- **Auto-labelling heuristic** — translating cluster numbers into business meaning programmatically
- **Streamlit** — building interactive data apps without frontend code

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## What's next

- [ ] Add RFM scoring (1–5 scale per dimension) alongside raw values
- [ ] DBSCAN comparison — density-based clustering to handle outlier high-spenders
- [ ] Export segment lists as CSV from the dashboard for marketing team use
- [ ] Add time-series view showing how segment sizes shift month over month

---

## Author

**Kushagra Kapoor**  
[GitHub](https://github.com/Kushagra-Kapoor-04) · [LinkedIn](https://linkedin.com/in/kushagra-kapoor-04)