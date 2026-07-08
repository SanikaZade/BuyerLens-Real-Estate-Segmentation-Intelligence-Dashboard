# Machine Learning-based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

**Client:** Parcl Co. Limited  
**Partner:** Unified Mentor  
**Domain:** Financial Analytics & Real Estate Market Intelligence

---

## Project Overview

This project introduces AI-driven buyer intelligence into the Parcl real estate platform.
By identifying hidden customer segments through clustering algorithms, the system reveals
investment behaviors that traditional analytics cannot detect — enabling Parcl to build
smarter marketing strategies and drive data-driven real estate investment decisions.

---

## 4 Buyer Segments (Discovered via K-Means Clustering)

| Cluster | Buyer Type | Characteristics |
|---------|-----------|----------------|
| C1 | 🔵 Global Investors | High income, investment purchases |
| C2 | 🔴 First-Time Buyers | Younger, loan dependent |
| C3 | 🟢 Corporate Buyers | Companies purchasing multiple units |
| C4 | 🟣 Luxury Investors | High satisfaction, large investments |

---

## Project Structure

```
buyer-segmentation-parcl/
├── app.py              ← Streamlit dashboard (single entry point)
├── pipeline.py         ← Full ML pipeline — Steps 1 to 6
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── data/
    ├── clients.csv     ← 2,000 client records
    └── properties.csv  ← 10,000 property listings
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| App Framework | Streamlit |
| ML — Primary Clustering | K-Means (K=4) |
| ML — Validation | Hierarchical / Agglomerative Clustering |
| Feature Scaling | StandardScaler |
| Feature Encoding | One-Hot Encoding + Label Encoding |
| Cluster Evaluation | Elbow Method + Silhouette Score |
| Charts | Plotly |
| Data Processing | Pandas, NumPy, SciPy |

---

## 6-Step Data Science Methodology (per PRD)

| Step | Task |
|------|------|
| 1 | **Data Cleaning** — handle missing attributes, normalize labels, remove duplicates |
| 2 | **Feature Encoding** — Label Encoding + One-Hot Encoding for client_type, region, acquisition_purpose, referral_channel, country |
| 3 | **Feature Scaling** — StandardScaler on age, satisfaction score, and all encoded columns |
| 4 | **Clustering** — K-Means (primary) + Hierarchical (validation) |
| 5 | **Optimal K Selection** — Elbow Method + Silhouette Score for K=2..10 |
| 6 | **Cluster Interpretation** — investment purpose, geographic distribution, loan behavior, customer demographics |

---

## Dashboard Modules (per PRD)

| Module | Description |
|--------|-------------|
| 📊 Buyer Segmentation Overview | Cluster distribution — pie chart, bar chart, age distribution |
| 📈 Investor Behavior Dashboard | Investment patterns by cluster — prices, loans, acquisition purpose |
| 🌍 Geographic Buyer Analysis | Buyer segments by country and region, referral channels |
| 🔍 Segment Insights Panel | Descriptive statistics, radar profile, raw data table |
| ⚙️ Model Evaluation | Elbow curve, silhouette scores, dendrogram, K-Means vs Hierarchical |

### Sidebar Filters (apply to all modules)
- Country (multiselect)
- Region (multiselect, scoped to selected countries)
- Acquisition Purpose — All / Investment / Home
- Client Type — All / Individual / Company

> Clustering is trained on all 2,000 clients; filters only affect the displayed view.

---

## Deliverables (per PRD)

- ✅ Streamlit dashboard (live analytics) — `app.py`
- ✅ ML pipeline — `pipeline.py`
- ✅ Research paper / EDA — see Segment Insights Panel in the dashboard
