"""
app.py — Parcl Buyer Segmentation Streamlit Dashboard
=======================================================
Title  : Machine Learning-based Buyer Segmentation and Investment
         Profiling for Real Estate Market Intelligence
Client : Parcl Co. Limited  x  Unified Mentor
Domain : Financial Analytics & Real Estate Market Intelligence

Bespoke Design: Architectural Blueprint & Technical Drafting Schematic.
"""

from __future__ import annotations

import sys
if "pipeline" in sys.modules:
    import importlib
    importlib.reload(sys.modules["pipeline"])

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle as _pk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from pipeline import (
    SEGMENT_COLORS,
    compute_linkage_matrix,
    compute_pca_2d,
    elbow_and_silhouette,
    run_agglomerative,
    run_pipeline,
    segment_summary,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Parcl // Buyer Segmentation Schematic",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Bespoke Blueprint & Drafting Styling (Custom CSS)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&family=Inter:wght@400;500&display=swap');
    
    /* Main Layout & Drafting Grid Paper */
    .stApp {
        background-color: #0F2537 !important;
        background-image: 
            linear-gradient(rgba(0, 210, 255, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 210, 255, 0.025) 1px, transparent 1px) !important;
        background-size: 24px 24px !important;
        color: #F0F6FC !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'DM Sans', sans-serif !important;
        color: #00D2FF !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Drawing Title Block styling */
    .title-block {
        border: 2px solid #00D2FF;
        padding: 1.5rem;
        border-radius: 4px;
        background-color: rgba(11, 30, 45, 0.8);
        margin-bottom: 2rem;
        position: relative;
    }
    
    .title-block::before {
        content: "SCHEMATIC // REF-A109";
        position: absolute;
        top: -10px;
        right: 20px;
        background-color: #0F2537;
        padding: 0 10px;
        color: #00D2FF;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }
    
    /* Metric panels styled as parameter readouts */
    div[data-testid="metric-container"] {
        background-color: rgba(11, 30, 45, 0.8) !important;
        border: 1px solid rgba(0, 210, 255, 0.25) !important;
        border-radius: 4px !important;
        padding: 1rem 1.2rem !important;
        font-family: 'Space Mono', monospace !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 1.8rem !important;
        color: #FFB703 !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.75rem !important;
        color: #F0F6FC !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        border-bottom: 1px dashed rgba(0, 210, 255, 0.15) !important;
        padding-bottom: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Technical Tabs (Flat blueprint line design) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: rgba(11, 30, 45, 0.5) !important;
        border: 1px solid rgba(0, 210, 255, 0.2) !important;
        padding: 4px;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent !important;
        border-radius: 4px;
        color: rgba(240, 246, 252, 0.65) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 500;
        border: none !important;
        padding: 0 16px !important;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 210, 255, 0.08) !important;
        color: #00D2FF !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00D2FF !important;
        color: #0F2537 !important;
        font-weight: 700 !important;
    }
    
    /* Technical Drafting borders for content segments */
    .drafting-panel {
        border: 1px solid rgba(0, 210, 255, 0.2);
        padding: 1.5rem;
        background-color: rgba(11, 30, 45, 0.4);
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styled as control console */
    section[data-testid="stSidebar"] {
        background-color: #0B1E2D !important;
        border-right: 1px solid rgba(0, 210, 255, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #00D2FF !important;
    }
    
    /* Input field blueprint styling */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    .stTextInput input, .stNumberInput input {
        background-color: #0F2537 !important;
        border: 1px solid rgba(0, 210, 255, 0.25) !important;
        color: #F0F6FC !important;
        font-family: 'Space Mono', monospace !important;
        border-radius: 4px !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00D2FF !important;
        color: #0F2537 !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #FFB703 !important;
        color: #0F2537 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Sidebar — Dynamic K Setup & Filters
# ---------------------------------------------------------------------------
st.sidebar.title("📐 Parcl Schematic")
st.sidebar.caption("Architectural Real Estate Segmentation & Analysis")
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Parameter Setup")
# Dynamic K Slider (2 to 8)
n_clusters = st.sidebar.slider(
    "Set Cluster Parameter (K)",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
    help="Define the number of buyer classes calculated by the engine."
)

# ---------------------------------------------------------------------------
# Data Loading — cached so ML runs ONCE per selected K
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Updating blueprint matrices...")
def load_data_cached(k_val: int) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, pd.DataFrame, list[str], KMeans]:
    return run_pipeline(
        BASE / "data" / "clients.csv",
        BASE / "data" / "properties.csv",
        n_clusters=k_val,
    )

df_full, X_scaled, scaler, eval_df, feature_cols, km = load_data_cached(n_clusters)

# Build custom blueprint color palette dynamically based on segment names
unique_segs = sorted(df_full["segment"].unique().tolist())
# Technical blueprint line colors: Cyan, Golden Amber, Slate Blue, Ink White, Coral, Rose
color_palette = ["#00D2FF", "#FFB703", "#38BDF8", "#E2E8F0", "#F472B6", "#FB7185", "#34D399"]
segment_colors_map = {}
for i, seg in enumerate(unique_segs):
    segment_colors_map[seg] = color_palette[i % len(color_palette)]

# Cache additional evaluation models
@st.cache_data(show_spinner=False)
def get_agglo_labels(X_bytes: np.ndarray, k_val: int) -> np.ndarray:
    return run_agglomerative(X_bytes, n_clusters=k_val)

@st.cache_data(show_spinner=False)
def get_linkage(X_bytes: np.ndarray) -> np.ndarray:
    return compute_linkage_matrix(X_bytes[:300])

agglo_labels = get_agglo_labels(X_scaled, n_clusters)
linkage_matrix = get_linkage(X_scaled)

# Sidebar Filters
st.sidebar.subheader("🔎 View Filters")

all_countries = sorted(df_full["country"].unique())
sel_countries = st.sidebar.multiselect("Country", all_countries, default=all_countries)

region_pool = sorted(
    df_full.loc[df_full["country"].isin(sel_countries), "region"].unique()
)
sel_regions = st.sidebar.multiselect("Region", region_pool, default=region_pool)

sel_purpose = st.sidebar.selectbox(
    "Acquisition Purpose", ["All", "Investment", "Home"]
)
sel_client_type = st.sidebar.selectbox(
    "Client Type", ["All", "Individual", "Company"]
)

# Apply filters
df = df_full.copy()
if sel_countries:
    df = df[df["country"].isin(sel_countries)]
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
if sel_purpose != "All":
    df = df[df["acquisition_purpose"] == sel_purpose]
if sel_client_type != "All":
    df = df[df["client_type"] == sel_client_type]

# ---------------------------------------------------------------------------
# Helper Chart Styling for Blueprint Look
# ---------------------------------------------------------------------------
def apply_blueprint_style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(11, 30, 45, 0.4)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Space Mono",
        font_color="#F0F6FC",
        title_font_family="DM Sans",
        title_font_color="#00D2FF",
        legend=dict(
            bgcolor="rgba(11, 30, 45, 0.8)",
            bordercolor="rgba(0, 210, 255, 0.2)",
            borderwidth=1
        ),
        margin=dict(t=50, b=40, l=40, r=40)
    )
    fig.update_xaxes(
        gridcolor="rgba(0, 210, 255, 0.08)",
        zerolinecolor="rgba(0, 210, 255, 0.2)",
        tickfont=dict(family="Space Mono", size=10)
    )
    fig.update_yaxes(
        gridcolor="rgba(0, 210, 255, 0.08)",
        zerolinecolor="rgba(0, 210, 255, 0.2)",
        tickfont=dict(family="Space Mono", size=10)
    )
    return fig

def money(v: float) -> str:
    return f"${v:,.0f}"

# ---------------------------------------------------------------------------
# Header Title Block
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="title-block">
        <h1 style="margin: 0; font-size: 2.2rem; color: #00D2FF;">PARCL REAL ESTATE ANALYTICAL SCHEMATIC</h1>
        <p style="margin: 0.5rem 0 0 0; color: #F0F6FC; opacity: 0.8; font-size: 0.9rem;">
            A ML-based segmentation matrix of 2,000 property buyers using K-Means clustering.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Buyers Accounted", f"{len(df):,}")
k2.metric("Mean Satisfaction", f"{df['satisfaction_score'].mean():.2f}")
k3.metric("Leveraged (Loan Applied)", f"{df['loan_applied'].mean()*100:.1f}%")
k4.metric("Average Unit Price", money(df["avg_sale_price"].mean()))
k5.metric("Active Segments", df["segment"].nunique())

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 SEGMENTATION_MAP",
    "📈 INVESTOR_METRICS",
    "🌍 GEOGRAPHIC_DRAFT",
    "🔍 SEGMENT_PROFILES",
    "⚙️ CLUSTERING_EVAL",
    "🔮 SIMULATION_PLAYGROUND",
])

# ===========================================================================
# MODULE 1 — Buyer Segmentation Overview
# ===========================================================================
with tab1:
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD A-1 // Cluster Volume Distribution")
    st.caption("Distribution of calculated buyer groupings mapped across active database filters.")

    seg_counts = df.groupby("segment").size().reset_index(name="count")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_pie = px.pie(
            seg_counts,
            names="segment",
            values="count",
            color="segment",
            color_discrete_map=segment_colors_map,
            title="Cluster Share Breakdown",
            hole=0.45,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        apply_blueprint_style(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar_seg = px.bar(
            seg_counts.sort_values("count", ascending=False),
            x="segment", y="count",
            color="segment",
            color_discrete_map=segment_colors_map,
            title="Absolute Group Count",
            labels={"segment": "Segment", "count": "Buyers Count"},
            text="count",
        )
        fig_bar_seg.update_traces(textposition="outside")
        apply_blueprint_style(fig_bar_seg)
        st.plotly_chart(fig_bar_seg, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2D PCA Scatter plot
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD A-2 // Multi-Dimensional PCA Space Map")
    st.caption("Projection of full feature-set coordinates mapped down to 2 principal components.")
    
    pca_coords = compute_pca_2d(X_scaled)
    df_pca = df_full.copy()
    df_pca["PCA Component 1"] = pca_coords[:, 0]
    df_pca["PCA Component 2"] = pca_coords[:, 1]
    df_pca_filtered = df_pca.loc[df.index]
    
    fig_pca = px.scatter(
        df_pca_filtered,
        x="PCA Component 1",
        y="PCA Component 2",
        color="segment",
        color_discrete_map=segment_colors_map,
        hover_data=["client_id", "age", "satisfaction_score", "avg_sale_price", "total_spent"],
        title="2D Plot Projection Map",
        opacity=0.75,
    )
    apply_blueprint_style(fig_pca)
    st.plotly_chart(fig_pca, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Age distribution
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD A-3 // Demographics Distribution (Age Range)")
    fig_box = px.box(
        df, x="segment", y="age",
        color="segment",
        color_discrete_map=segment_colors_map,
        title="Age Spread per Group",
        labels={"segment": "Segment", "age": "Age (years)"},
        points="outliers",
    )
    apply_blueprint_style(fig_box)
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 2 — Investor Behavior Dashboard
# ===========================================================================
with tab2:
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD B-1 // Transaction Profiles")
    st.caption("Average purchase volume and financing ratio comparisons.")

    seg_agg = (
        df.groupby("segment")
        .agg(
            avg_sale_price=("avg_sale_price", "mean"),
            pct_loan=("loan_applied", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_total_spent=("total_spent", "mean"),
        )
        .reset_index()
    )
    seg_agg["pct_loan_pct"] = seg_agg["pct_loan"] * 100

    col1, col2 = st.columns(2)

    with col1:
        sorted_price = seg_agg.sort_values("avg_sale_price", ascending=False)
        fig_price = px.bar(
            sorted_price,
            x="segment", y="avg_sale_price",
            color="segment",
            color_discrete_map=segment_colors_map,
            title="Mean Property Sale Price",
            labels={"segment": "Segment", "avg_sale_price": "Avg Price"},
            text=sorted_price["avg_sale_price"].apply(money),
        )
        fig_price.update_traces(textposition="outside")
        apply_blueprint_style(fig_price)
        fig_price.update_layout(showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        sorted_loan = seg_agg.sort_values("pct_loan_pct", ascending=False)
        fig_loan = px.bar(
            sorted_loan,
            x="segment", y="pct_loan_pct",
            color="segment",
            color_discrete_map=segment_colors_map,
            title="Loan Application Percentage",
            labels={"segment": "Segment", "pct_loan_pct": "Loan Applied %"},
            text=sorted_loan["pct_loan_pct"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_loan.update_traces(textposition="outside")
        apply_blueprint_style(fig_loan)
        fig_loan.update_layout(showlegend=False)
        st.plotly_chart(fig_loan, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Scatter: satisfaction vs avg_sale_price
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD B-2 // Satisfaction vs Average Price")
    fig_scatter = px.scatter(
        df,
        x="satisfaction_score", y="avg_sale_price",
        color="segment", size="total_spent",
        color_discrete_map=segment_colors_map,
        hover_data=["client_id", "country", "acquisition_purpose"],
        title="Satisfaction Score Map (Bubble Area = Total Investment Volume)",
        labels={
            "satisfaction_score": "Satisfaction Rating",
            "avg_sale_price": "Property Price ($)",
            "total_spent": "Total Investment ($)",
        },
        opacity=0.6,
    )
    apply_blueprint_style(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Investment purpose breakdown
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD B-3 // Purchase Intent Profile")
    purpose_seg = (
        df.groupby(["segment", "acquisition_purpose"])
        .size()
        .reset_index(name="count")
    )
    fig_purpose = px.bar(
        purpose_seg,
        x="segment", y="count",
        color="acquisition_purpose",
        barmode="group",
        title="Acquisition Intent mapping (Home vs Investment)",
        labels={
            "segment": "Segment",
            "count": "Total Buyers",
            "acquisition_purpose": "Purpose",
        },
        color_discrete_sequence=["#00D2FF", "#FFB703"],
    )
    apply_blueprint_style(fig_purpose)
    st.plotly_chart(fig_purpose, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 3 — Geographic Buyer Analysis
# ===========================================================================
with tab3:
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD C-1 // Country Segment Breakdown")
    geo_seg = (
        df.groupby(["country", "segment"]).size().reset_index(name="count")
    )
    fig_geo_stacked = px.bar(
        geo_seg,
        x="country", y="count",
        color="segment",
        color_discrete_map=segment_colors_map,
        barmode="stack",
        title="Buyer Group Concentration by Country",
        labels={"country": "Country", "count": "Buyers Count", "segment": "Segment"},
    )
    apply_blueprint_style(fig_geo_stacked)
    st.plotly_chart(fig_geo_stacked, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
        st.subheader("COORD C-2 // Capital Allocation by Country")
        country_spent = (
            df.groupby("country")["total_spent"]
            .mean()
            .reset_index(name="avg_total_spent")
            .sort_values("avg_total_spent")
        )
        fig_spent = px.bar(
            country_spent,
            x="avg_total_spent", y="country",
            orientation="h",
            color="avg_total_spent",
            color_continuous_scale="Blues",
            title="Mean Outlay by Region",
            labels={"avg_total_spent": "Mean Spend", "country": "Country"},
            text=country_spent["avg_total_spent"].apply(money),
        )
        fig_spent.update_traces(textposition="outside")
        apply_blueprint_style(fig_spent)
        fig_spent.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_spent, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_g2:
        st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
        st.subheader("COORD C-3 // Customer Funnels")
        ref_country = (
            df.groupby(["country", "referral_channel"])
            .size()
            .reset_index(name="count")
        )
        fig_ref = px.bar(
            ref_country,
            x="country", y="count",
            color="referral_channel",
            barmode="group",
            title="Marketing Channels by Geography",
            labels={
                "country": "Country",
                "count": "Buyers count",
                "referral_channel": "Channel",
            },
            color_discrete_sequence=["#00D2FF", "#38BDF8", "#E2E8F0"],
        )
        apply_blueprint_style(fig_ref)
        fig_ref.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_ref, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD C-4 // Top 20 Region-level Allocations")
    region_seg = (
        df.groupby(["region", "segment"]).size().reset_index(name="count")
    )
    top_regions = (
        df.groupby("region").size().nlargest(20).index.tolist()
    )
    region_seg = region_seg[region_seg["region"].isin(top_regions)]
    fig_region = px.bar(
        region_seg,
        x="region", y="count",
        color="segment",
        color_discrete_map=segment_colors_map,
        barmode="stack",
        title="Buyer Group Concentration by Active Sub-Region",
        labels={"region": "Region", "count": "Buyers Count", "segment": "Segment"},
    )
    apply_blueprint_style(fig_region)
    fig_region.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_region, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 4 — Segment Insights Panel
# ===========================================================================
with tab4:
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD D-1 // Descriptive Analytics Panel")
    st.caption("Descriptive averages mapped per buyer grouping.")

    # Summary statistics table
    summ = segment_summary(df)
    display_summ = summ.copy()
    display_summ["Avg_Sale_Price"]  = display_summ["Avg_Sale_Price"].apply(money)
    display_summ["Avg_Total_Spent"] = display_summ["Avg_Total_Spent"].apply(money)
    display_summ.columns = [
        "Segment", "Count", "Avg Age", "Avg Satisfaction",
        "Loan Applied %", "Avg Properties", "Avg Price", "Avg Outlay",
    ]
    st.dataframe(display_summ, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Radar / Spider chart
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD D-2 // Normalised Cluster Fingerprints")
    st.caption("Metric comparisons scaled between 0.0 and 1.0 to profile cluster behaviors.")
    
    radar_cols   = ["Avg_Age", "Avg_Satisfaction", "Pct_Loan",
                    "Avg_Sale_Price", "Avg_Total_Spent", "Avg_Properties"]
    radar_labels = ["Age", "Satisfaction", "Loan %",
                    "Avg Price", "Total Outlay", "Properties count"]

    radar_data = summ[["segment"] + radar_cols].copy()
    for col in radar_cols:
        lo, hi = radar_data[col].min(), radar_data[col].max()
        rng = hi - lo if hi != lo else 1
        radar_data[col] = (radar_data[col] - lo) / rng

    fig_radar = go.Figure()
    for _, row in radar_data.iterrows():
        vals = [row[c] for c in radar_cols] + [row[radar_cols[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=row["segment"],
            line_color=segment_colors_map.get(row["segment"], "#888"),
            fillcolor=segment_colors_map.get(row["segment"], "#888"),
            opacity=0.3,
        ))
    apply_blueprint_style(fig_radar)
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0, 210, 255, 0.1)"),
            angularaxis=dict(gridcolor="rgba(0, 210, 255, 0.1)")
        )
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Gender breakdown per segment
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD D-3 // Client Demographics")
    gender_seg = (
        df.groupby(["segment", "gender"]).size().reset_index(name="count")
    )
    fig_gender = px.bar(
        gender_seg,
        x="segment", y="count",
        color="gender",
        barmode="group",
        title="Gender Allocation mapping per Group",
        labels={"segment": "Segment", "count": "Buyers Count", "gender": "Gender"},
        color_discrete_sequence=["#00D2FF", "#FFB703"],
    )
    apply_blueprint_style(fig_gender)
    st.plotly_chart(fig_gender, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Raw data table
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD D-4 // Query Database Ledger")
    seg_opts = ["All"] + sorted(df["segment"].unique().tolist())
    sel_seg  = st.selectbox("Select Segment to Filter Ledger", seg_opts, key="raw_seg_filter")

    raw = df[[
        "client_id", "client_type", "gender", "age", "country", "region",
        "acquisition_purpose", "satisfaction_score", "loan_applied",
        "referral_channel", "num_properties", "avg_sale_price",
        "total_spent", "segment",
    ]].copy()
    raw["avg_sale_price"] = raw["avg_sale_price"].apply(money)
    raw["total_spent"]    = raw["total_spent"].apply(money)

    if sel_seg != "All":
        raw = raw[raw["segment"] == sel_seg]

    st.dataframe(raw, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 5 — Model Evaluation
# ===========================================================================
with tab5:
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD E-1 // Cluster Metric Optimization")
    st.caption("Validating optimization points using the Elbow curve (Inertia) and Silhouette index mapping.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        fig_elbow = px.line(
            eval_df, x="k", y="inertia",
            markers=True,
            title="Elbow Curve Model (Inertia)",
            labels={"k": "Number of Clusters (K)", "inertia": "Inertia"},
            color_discrete_sequence=["#00D2FF"],
        )
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="#FFB703",
                            annotation_text="K=4 Selected", annotation_font_color="#FFB703")
        apply_blueprint_style(fig_elbow)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_e2:
        fig_sil = px.line(
            eval_df, x="k", y="silhouette",
            markers=True,
            title="Silhouette Index Profile",
            labels={"k": "Number of Clusters (K)", "silhouette": "Silhouette Score"},
            color_discrete_sequence=["#FFB703"],
        )
        fig_sil.add_vline(x=4, line_dash="dash", line_color="#00D2FF",
                          annotation_text="K=4 Selected", annotation_font_color="#00D2FF")
        apply_blueprint_style(fig_sil)
        st.plotly_chart(fig_sil, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # K-Means vs Hierarchical comparison
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD E-2 // Cross-Clustering Cross-Reference Map")
    df_compare = df_full.copy()
    df_compare["agglo_cluster"] = agglo_labels
    comp = (
        df_compare.groupby(["segment", "agglo_cluster"])
        .size()
        .reset_index(name="count")
    )
    comp["agglo_cluster"] = "Agglo-" + comp["agglo_cluster"].astype(str)
    fig_comp = px.bar(
        comp, x="segment", y="count",
        color="agglo_cluster",
        barmode="group",
        title="K-Means Segments vs Agglomerative Clusters Distribution",
        labels={"segment": "K-Means Segment", "count": "Buyers Count", "agglo_cluster": "Agglo Cluster"},
    )
    apply_blueprint_style(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Dendrogram
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD E-3 // Ward Linkage Tree (Agglomerative Dendrogram)")
    try:
        import plotly.figure_factory as ff
        dendro_fig = ff.create_dendrogram(
            X_scaled[:300],
            orientation="bottom",
            linkagefun=lambda x: linkage_matrix,
        )
        apply_blueprint_style(dendro_fig)
        dendro_fig.update_layout(
            title="Tree Diagram (Subsample N=300)",
            xaxis=dict(showticklabels=False),
        )
        st.plotly_chart(dendro_fig, use_container_width=True)
    except Exception as e:
        st.info(f"Dendrogram skipped: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Evaluation metrics table
    st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
    st.subheader("COORD E-4 // Index Matrix Output Ledger")
    ev = eval_df.copy()
    ev.columns = ["K-Value", "Inertia Scale", "Silhouette Score"]
    ev["Inertia Scale"]      = ev["Inertia Scale"].apply(lambda x: f"{x:,.0f}")
    ev["Silhouette Score"] = ev["Silhouette Score"].apply(
        lambda x: f"{x:.4f}" if not np.isnan(x) else "—"
    )
    st.dataframe(ev, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 6 — Predict Segment & Marketing Recommendations Engine
# ===========================================================================
with tab6:
    st.header("🔮 Buyer Profiling & Prediction Simulator")
    st.caption(
        "Inject virtual buyer parameters to compute classifications and map targeted marketing plays."
    )
    
    st.markdown("---")
    
    # Marketing Recommendations Dictionary
    def get_segment_marketing_insights(seg_name: str) -> dict:
        insights = {
            "Corporate Buyers": {
                "profile": "B2B client profiles buying multiple real estate parcels. Low relative loan reliance.",
                "strategy": "Deploy dedicated B2B institutional channels, volume acquisition discounts, and specialized portfolio reports.",
                "channels": "LinkedIn ads, corporate broker alliances, institutional investment roundtables.",
                "badge": "🏢 B2B / High-Volume Institutional"
            },
            "Luxury Investors": {
                "profile": "Ultra-high net-worth private buyers acquiring high-value prime units. High satisfaction indicators.",
                "strategy": "Launch VIP white-glove offerings, early access to pre-sales list, personal concierge allocation.",
                "channels": "Wealth management networks, premium asset events, exclusive investor circulars.",
                "badge": "💎 High-Net-Worth / Premium Focus"
            },
            "First-Time Buyers": {
                "profile": "Younger demographic acquiring entry-level units. Highly dependent on loan approvals.",
                "strategy": "Partner with local digital loan brokers, supply step-by-step buyer guidelines, offer referral incentives.",
                "channels": "Social media channels (Instagram, TikTok), educational web hubs, homebuyer seminars.",
                "badge": "🔑 Loan-Dependent / Entry-Level"
            },
            "Global Investors": {
                "profile": "Cross-border purchasers focused on international asset diversification.",
                "strategy": "Offer international transaction support, local tax advisory connections, and virtual VR viewing portals.",
                "channels": "Global real estate forums, international search engine ads, multi-lingual portal pages.",
                "badge": "🌐 Cross-Border / Portfolio Investors"
            },
            "High-Volume Buyers": {
                "profile": "Yield-driven portfolio builders buying multiple mid-level yield units.",
                "strategy": "Provide property management service packages and automated cashflow report interfaces.",
                "channels": "Investor-focused web podcasts, tax consulting panels.",
                "badge": "📊 Yield-Driven Portfolio Buyers"
            }
        }
        return insights.get(seg_name, {
            "profile": "Custom behavior grouping computed by dynamic algorithm.",
            "strategy": "Map layout and averages in COORD D-1 to calibrate segment offers.",
            "channels": "Custom digital advertising & broker networks.",
            "badge": "🎯 Niche Segment"
        })

    # Predict Function
    def predict_single_client(
        sc: StandardScaler,
        f_cols: list[str],
        kmeans_model: KMeans,
        c_df: pd.DataFrame,
        age: int,
        satisfaction: float,
        loan: int,
        props: int,
        sale_price: float,
        spent: float,
        gender: str,
        c_type: str,
        purpose: str,
        channel: str,
        country: str,
        region: str,
    ) -> int:
        # Default all feature cols to 0.0
        data = {col: 0.0 for col in f_cols}
        
        # Populate basic numerical attributes
        data["age"] = float(age)
        data["satisfaction_score"] = float(satisfaction)
        data["loan_applied_enc"] = float(loan)
        data["num_properties"] = float(props)
        data["avg_sale_price"] = float(sale_price)
        data["total_spent"] = float(spent)
        data["avg_floor_area"] = 1500.0
        data["gender_enc"] = 1.0 if gender == "M" else 0.0
        
        # Map categorical variables using OHE formatting prefix
        ohe_mappings = [
            ("client_type_", c_type),
            ("acq_purpose_", purpose),
            ("referral_channel_", channel),
            ("country_", country),
            ("region_", region),
        ]
        
        for prefix, val in ohe_mappings:
            col_name = f"{prefix}{val}"
            if col_name in data:
                data[col_name] = 1.0
                
        # Reconstruct exactly matching dataframe
        df_row = pd.DataFrame([data])[f_cols]
        X_row_scaled = sc.transform(df_row.values)
        return int(kmeans_model.predict(X_row_scaled)[0])

    # Form layout
    col_f1, col_f2 = st.columns([1, 1.2])
    
    with col_f1:
        st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
        st.subheader("📝 Virtual Buyer Parameters")
        
        # Fetch inputs dynamically
        unique_countries = sorted(df_full["country"].unique())
        unique_regions = sorted(df_full["region"].unique())
        unique_channels = sorted(df_full["referral_channel"].unique())
        
        with st.form("predict_form"):
            in_age = st.slider("Buyer Age", 18, 90, 35)
            in_gender = st.radio("Gender Profile", ["F", "M"], horizontal=True)
            in_client_type = st.selectbox("Client Type Category", ["Individual", "Company"])
            in_satisfaction = st.slider("Satisfaction Score Index", 1.0, 5.0, 4.0, step=0.5)
            in_loan = st.radio("Requires Loan Financing?", ["No", "Yes"], horizontal=True)
            
            st.markdown("**Transaction Inputs**")
            in_properties = st.slider("Properties Acquired", 0, 10, 1)
            in_sale_price = st.number_input("Average Purchase Price ($)", value=350000, step=10000)
            in_total_spent = st.number_input("Total Outlay Volume ($)", value=350000, step=10000)
            
            st.markdown("**Channel & Location Parameters**")
            in_purpose = st.selectbox("Purchase Intention", ["Investment", "Home"])
            in_channel = st.selectbox("Referral Funnel", unique_channels)
            in_country = st.selectbox("Target Country", unique_countries)
            in_region = st.selectbox("Target Sub-Region", unique_regions)
            
            submitted = st.form_submit_button("🔮 COMPUTE CLASSIFICATION")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col_f2:
        st.markdown('<div class="drafting-panel">', unsafe_allow_html=True)
        st.subheader("🎯 Classification Output Summary")
        
        if submitted:
            loan_bin = 1 if in_loan == "Yes" else 0
            
            # Run prediction
            pred_cluster = predict_single_client(
                scaler, feature_cols, km, df_full,
                in_age, in_satisfaction, loan_bin, in_properties,
                in_sale_price, in_total_spent, in_gender, in_client_type,
                in_purpose, in_channel, in_country, in_region
            )
            
            # Map cluster to segment name
            from pipeline import assign_segment_labels
            label_map = assign_segment_labels(df_full, n_clusters=n_clusters)
            predicted_segment = label_map.get(pred_cluster, f"Segment {pred_cluster}")
            
            # Retrieve Marketing Insights
            meta = get_segment_marketing_insights(predicted_segment)
            
            # UI display cards
            st.markdown(
                f"""
                <div style="background-color: rgba(0, 210, 255, 0.05); border: 2px solid #00D2FF; padding: 1.5rem; border-radius: 4px; margin-top: 1rem;">
                    <span style="font-size: 0.75rem; font-family: 'Space Mono', monospace; font-weight: 700; color: #00D2FF; text-transform: uppercase; letter-spacing: 0.05em;">Calculated Classification</span>
                    <h2 style="margin: 0.2rem 0; color: #00D2FF; font-family: 'DM Sans', sans-serif;">{predicted_segment}</h2>
                    <span style="background-color: #FFB703; color: #0F2537; padding: 0.25rem 0.6rem; border-radius: 2px; font-size: 0.75rem; font-family: 'Space Mono', monospace; font-weight: 700;">{meta['badge']}</span>
                    <hr style="border-top: 1px dashed rgba(0, 210, 255, 0.2); margin: 1rem 0;">
                    <p style="font-size: 0.9rem; color: #F0F6FC;"><strong>Segment Profile:</strong><br>{meta['profile']}</p>
                    <p style="font-size: 0.9rem; color: #F0F6FC;"><strong>Marketing Protocol:</strong><br>{meta['strategy']}</p>
                    <p style="font-size: 0.9rem; color: #F0F6FC;"><strong>Primary Outreach Channels:</strong><br>{meta['channels']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Compare stats
            st.markdown("### Profile Deviation Comparison")
            st.caption("Direct comparison of input parameters vs active segment benchmarks.")
            
            current_seg_averages = summ[summ["segment"] == predicted_segment]
            if not current_seg_averages.empty:
                avg_row = current_seg_averages.iloc[0]
                
                metrics_comparison = pd.DataFrame([
                    {"Metric Parameters": "Age", "Client Input Value": in_age, "Segment Benchmarks": avg_row["Avg_Age"]},
                    {"Metric Parameters": "Satisfaction Index", "Client Input Value": in_satisfaction, "Segment Benchmarks": avg_row["Avg_Satisfaction"]},
                    {"Metric Parameters": "Properties Acquired", "Client Input Value": in_properties, "Segment Benchmarks": avg_row["Avg_Properties"]},
                    {"Metric Parameters": "Average Price ($)", "Client Input Value": in_sale_price, "Segment Benchmarks": avg_row["Avg_Sale_Price"]},
                    {"Metric Parameters": "Total Outlay ($)", "Client Input Value": in_total_spent, "Segment Benchmarks": avg_row["Avg_Total_Spent"]}
                ])
                st.dataframe(metrics_comparison, use_container_width=True, hide_index=True)
        else:
            st.info("Form idle. Update variables and click COMPUTE CLASSIFICATION to execute matrix run.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. General Segment Recommendations Overview Cards
    st.markdown("---")
    st.subheader("💡 Strategic Marketing Playbooks")
    st.caption("Active segment playbooks catalogued by the profiling engine.")
    
    from pipeline import assign_segment_labels
    label_map = assign_segment_labels(df_full, n_clusters=n_clusters)
    active_seg_names = list(label_map.values())
    
    col_rec1, col_rec2 = st.columns(2)
    for index, seg_name in enumerate(active_seg_names):
        meta = get_segment_marketing_insights(seg_name)
        target_col = col_rec1 if index % 2 == 0 else col_rec2
        with target_col:
            st.markdown(
                f"""
                <div style="background-color: rgba(11, 30, 45, 0.8); border: 1px solid rgba(0, 210, 255, 0.25); padding: 1.2rem; border-radius: 4px; margin-bottom: 1rem;">
                    <h4 style="margin:0; color: #00D2FF; font-family: 'DM Sans', sans-serif;">{seg_name}</h4>
                    <small style="color: #FFB703; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 0.75rem;">{meta['badge']}</small>
                    <p style="margin: 0.5rem 0; font-size: 0.85rem; color: #F0F6FC; opacity: 0.8;">{meta['profile']}</p>
                    <p style="margin: 0; font-size: 0.85rem; color: #F0F6FC;"><strong>Strategic Action:</strong> {meta['strategy']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
