"""
pipeline.py — Parcl Buyer Segmentation ML Pipeline
====================================================
Title  : Machine Learning-based Buyer Segmentation and Investment
         Profiling for Real Estate Market Intelligence
Client : Parcl Co. Limited  x  Unified Mentor
Domain : Financial Analytics & Real Estate Market Intelligence

Strictly follows the 6-step methodology from the PRD PDF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFERENCE_DATE = pd.Timestamp("2024-01-01")
N_CLUSTERS = 4
RANDOM_STATE = 42

# PRD-mandated segment colour palette
SEGMENT_COLORS: dict[str, str] = {
    "Global Investors":  "#636EFA",
    "First-Time Buyers": "#EF553B",
    "Corporate Buyers":  "#00CC96",
    "Luxury Investors":  "#AB63FA",
}

DOB_FORMATS = ["%m-%d-%Y", "%m/%d/%Y"]


# ---------------------------------------------------------------------------
# Step 1 — Data Cleaning  (PRD PDF Step 1)
# Tasks: handle missing client attributes, normalize categorical labels,
#        remove duplicate client entries
# ---------------------------------------------------------------------------

def _parse_dob(series: pd.Series) -> pd.Series:
    """Try multiple date formats to parse date_of_birth robustly."""
    parsed = pd.Series(pd.NaT, index=series.index)
    for fmt in DOB_FORMATS:
        mask = parsed.isna()
        if not mask.any():
            break
        parsed[mask] = pd.to_datetime(series[mask], format=fmt, errors="coerce")
    return parsed


def load_and_clean(clients_path: Path, properties_path: Path) -> pd.DataFrame:
    """
    PRD Step 1 — Data Cleaning.
    - Handle missing client attributes
    - Normalize categorical labels
    - Remove duplicate client entries
    - Derive age from date_of_birth
    - Aggregate property metrics per client and merge
    """
    # --- Load clients ---
    clients = pd.read_csv(clients_path)
    clients = clients.drop_duplicates(subset="client_id")  # remove duplicates

    # Normalize categorical labels (strip whitespace)
    for col in ["client_type", "gender", "acquisition_purpose",
                "loan_applied", "referral_channel", "country", "region"]:
        clients[col] = clients[col].astype(str).str.strip()

    # Parse date_of_birth → derive age
    clients = clients.copy()
    clients["dob_parsed"] = _parse_dob(clients["date_of_birth"])
    clients["age"] = (
        (REFERENCE_DATE - clients["dob_parsed"]).dt.days / 365.25
    ).fillna(0).astype(int)

    # Encode loan_applied: Yes→1, No→0
    clients["loan_applied"] = clients["loan_applied"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # --- Load properties ---
    props = pd.read_csv(properties_path)

    # Normalize sale_price: strip "$" and ","
    props["sale_price_clean"] = (
        props["sale_price"]
        .str.replace(r"[$,]", "", regex=True)
        .astype(float)
    )

    # Parse transaction_date
    props["transaction_date"] = pd.to_datetime(
        props["transaction_date"], format="%m-%d-%Y"
    )

    # Keep only Sold rows with a valid client reference
    sold = props[
        (props["listing_status"] == "Sold") & (props["client_ref"].notna())
    ].copy()

    # Aggregate per client
    agg = (
        sold.groupby("client_ref")
        .agg(
            num_properties=("listing_id", "count"),
            avg_sale_price=("sale_price_clean", "mean"),
            total_spent=("sale_price_clean", "sum"),
            avg_floor_area=("floor_area_sqft", "mean"),
        )
        .reset_index()
        .rename(columns={"client_ref": "client_id"})
    )

    # Merge clients ← property aggregates (left join)
    merged = clients.merge(agg, on="client_id", how="left")

    # Fill NaN for clients with no purchases
    for col in ["num_properties", "avg_sale_price", "total_spent", "avg_floor_area"]:
        merged[col] = merged[col].fillna(0)

    return merged


# ---------------------------------------------------------------------------
# Step 2 — Feature Encoding  (PRD PDF Step 2)
# Variables encoded per PRD: client_type, region, acquisition_purpose,
#                            referral_channel, country
# Methods: One-Hot Encoding + Label Encoding
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PRD Step 2 — Feature Encoding.
    Label Encoding for binary fields; One-Hot Encoding for
    client_type, region, acquisition_purpose, referral_channel, country.
    """
    df = df.copy()

    # Label Encoding — binary/ordinal fields
    df["gender_enc"]              = df["gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)
    df["loan_applied_enc"]        = df["loan_applied"]  # already 0/1

    # One-Hot Encoding — per PRD PDF (all 5 listed variables)
    ohe_cols = {
        "client_type":          "client_type",
        "acquisition_purpose":  "acq_purpose",
        "referral_channel":     "referral_channel",
        "country":              "country",
        "region":               "region",
    }
    for col, prefix in ohe_cols.items():
        dummies = pd.get_dummies(df[col], prefix=prefix, drop_first=False)
        df = pd.concat([df, dummies], axis=1)

    return df


# ---------------------------------------------------------------------------
# Step 3 — Feature Scaling  (PRD PDF Step 3)
# StandardScaler on age, satisfaction_score + all encoded columns
# ---------------------------------------------------------------------------

def _build_feature_cols(df: pd.DataFrame) -> list[str]:
    """Dynamically collect all feature columns from the encoded DataFrame."""
    base = ["age", "satisfaction_score", "loan_applied_enc",
            "num_properties", "avg_sale_price", "total_spent", "avg_floor_area",
            "gender_enc"]
    ohe_prefixes = ("client_type_", "acq_purpose_", "referral_channel_",
                    "country_", "region_")
    ohe = [c for c in df.columns if c.startswith(ohe_prefixes)]
    return base + ohe


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """
    PRD Step 3 — Feature Scaling using StandardScaler.
    Returns (scaled_array, fitted_scaler, feature_col_names).
    """
    feature_cols = _build_feature_cols(df)
    # Ensure all cols exist (OHE may miss categories on filtered subsets)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, feature_cols


# ---------------------------------------------------------------------------
# Step 4 — Clustering  (PRD PDF Step 4)
# K-Means (primary) + Hierarchical / Agglomerative (validation)
# ---------------------------------------------------------------------------

def run_kmeans(X_scaled: np.ndarray, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """PRD Step 4a — K-Means Clustering. Efficient, easy to interpret."""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    return km.fit_predict(X_scaled)


def run_agglomerative(X_scaled: np.ndarray, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """PRD Step 4b — Hierarchical (Agglomerative) Clustering, Ward linkage."""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return agg.fit_predict(X_scaled)


def compute_linkage_matrix(X_scaled: np.ndarray) -> np.ndarray:
    """Ward linkage matrix for dendrogram visualisation."""
    return linkage(X_scaled, method="ward")


# ---------------------------------------------------------------------------
# Step 5 — Optimal Cluster Selection  (PRD PDF Step 5)
# Elbow Method + Silhouette Score
# ---------------------------------------------------------------------------

def elbow_and_silhouette(
    X_scaled: np.ndarray, k_range: range = range(2, 11)
) -> pd.DataFrame:
    """
    PRD Step 5 — Compute inertia (Elbow) and Silhouette Score for K=2..10.
    Returns DataFrame with columns [k, inertia, silhouette].
    """
    from sklearn.metrics import silhouette_score
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6 — Cluster Interpretation  (PRD PDF Step 6)
# 4 buyer segments: Global Investors, First-Time Buyers,
#                   Corporate Buyers, Luxury Investors
# ---------------------------------------------------------------------------

def assign_segment_labels(df: pd.DataFrame, n_clusters: int = 4) -> dict[int, str]:
    """
    PRD Step 6 — Data-driven segment labelling based on cluster summary stats.
    Supports dynamic K-value mapping.
    """
    if "client_type_enc" not in df.columns:
        df = df.copy()
        df["client_type_enc"] = df["client_type"].map({"Individual": 1, "Company": 0}).fillna(0)

    summary = (
        df.groupby("cluster")
        .agg(
            pct_individual=("client_type_enc", "mean"),   # 1=Individual, 0=Company
            avg_sale_price=("avg_sale_price", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            pct_loan=("loan_applied", "mean"),
            avg_age=("age", "mean"),
            avg_properties=("num_properties", "mean"),
        )
        .reset_index()
    )

    assigned: dict[int, str] = {}
    remaining = set(summary["cluster"].tolist())

    if len(remaining) == 4:
        # PRD-specific naming logic
        corp_id = int(summary.loc[summary["pct_individual"].idxmin(), "cluster"])
        assigned[corp_id] = "Corporate Buyers"
        remaining.discard(corp_id)

        sub = summary[summary["cluster"].isin(remaining)]
        lux_id = int(sub.loc[sub["avg_sale_price"].idxmax(), "cluster"])
        assigned[lux_id] = "Luxury Investors"
        remaining.discard(lux_id)

        sub = summary[summary["cluster"].isin(remaining)]
        ft_id = int(sub.loc[sub["pct_loan"].idxmax(), "cluster"])
        assigned[ft_id] = "First-Time Buyers"
        remaining.discard(ft_id)

        global_id = remaining.pop()
        assigned[global_id] = "Global Investors"
    else:
        # General dynamic labeling for arbitrary K
        if remaining:
            sub = summary[summary["cluster"].isin(remaining)]
            corp_id = int(sub.loc[sub["pct_individual"].idxmin(), "cluster"])
            assigned[corp_id] = "Corporate Buyers"
            remaining.discard(corp_id)
            
        if remaining:
            sub = summary[summary["cluster"].isin(remaining)]
            lux_id = int(sub.loc[sub["avg_sale_price"].idxmax(), "cluster"])
            assigned[lux_id] = "Luxury Investors"
            remaining.discard(lux_id)
            
        if remaining:
            sub = summary[summary["cluster"].isin(remaining)]
            ft_id = int(sub.loc[sub["pct_loan"].idxmax(), "cluster"])
            assigned[ft_id] = "First-Time Buyers"
            remaining.discard(ft_id)
            
        if remaining:
            sub = summary[summary["cluster"].isin(remaining)]
            glob_id = int(sub.loc[sub["avg_age"].idxmax(), "cluster"])
            assigned[glob_id] = "Global Investors"
            remaining.discard(glob_id)

        if remaining:
            sub = summary[summary["cluster"].isin(remaining)]
            prop_id = int(sub.loc[sub["avg_properties"].idxmax(), "cluster"])
            assigned[prop_id] = "High-Volume Buyers"
            remaining.discard(prop_id)
            
        idx = 1
        for r in sorted(list(remaining)):
            assigned[r] = f"Segment {idx}"
            idx += 1

    return assigned


# ---------------------------------------------------------------------------
# Interactive PCA Dimensionality Reduction
# ---------------------------------------------------------------------------

def compute_pca_2d(X_scaled: np.ndarray) -> np.ndarray:
    """Reduce features to 2 principal components using PCA for 2D plotting."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(X_scaled)


# ---------------------------------------------------------------------------
# Master Pipeline Entry Point
# ---------------------------------------------------------------------------

def run_pipeline(
    clients_path: Path, properties_path: Path, n_clusters: int = 4
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, pd.DataFrame, list[str], KMeans]:
    """
    Execute the full 6-step PRD pipeline with dynamic K.

    Returns
    -------
    df           : enriched DataFrame with 'cluster' and 'segment' columns
    X_scaled     : scaled feature matrix (numpy array)
    scaler       : fitted StandardScaler instance
    eval_df      : elbow + silhouette evaluation table
    feature_cols : list of feature column names used for training
    km           : fitted KMeans model
    """
    # Step 1 — Clean
    df = load_and_clean(clients_path, properties_path)

    # Step 2 — Encode
    df = encode_features(df)

    # Add client_type_enc for labelling (Individual=1, Company=0)
    df["client_type_enc"] = df["client_type"].map(
        {"Individual": 1, "Company": 0}
    ).fillna(0).astype(int)

    # Step 3 — Scale
    X_scaled, scaler, feature_cols = scale_features(df)

    # Step 4 — Cluster (K-Means primary)
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    df["cluster"] = km.fit_predict(X_scaled)

    # Step 5 — Evaluate
    eval_df = elbow_and_silhouette(X_scaled)

    # Step 6 — Label segments
    label_map = assign_segment_labels(df, n_clusters=n_clusters)
    df["segment"] = df["cluster"].map(label_map)

    return df, X_scaled, scaler, eval_df, feature_cols, km


# ---------------------------------------------------------------------------
# Segment Summary Helper (used by app.py dashboard)
# ---------------------------------------------------------------------------

def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-segment descriptive statistics for the Segment Insights panel.
    Covers: investment purpose, geographic distribution, loan behavior,
            customer demographics (as per PRD Step 6 interpretation criteria).
    """
    summary = (
        df.groupby("segment")
        .agg(
            Count=("client_id", "count"),
            Avg_Age=("age", "mean"),
            Avg_Satisfaction=("satisfaction_score", "mean"),
            Pct_Loan=("loan_applied", "mean"),
            Avg_Properties=("num_properties", "mean"),
            Avg_Sale_Price=("avg_sale_price", "mean"),
            Avg_Total_Spent=("total_spent", "mean"),
        )
        .reset_index()
    )
    summary["Pct_Loan"]       = (summary["Pct_Loan"] * 100).round(1)
    summary["Avg_Age"]        = summary["Avg_Age"].round(1)
    summary["Avg_Satisfaction"]= summary["Avg_Satisfaction"].round(2)
    summary["Avg_Properties"] = summary["Avg_Properties"].round(2)
    return summary
