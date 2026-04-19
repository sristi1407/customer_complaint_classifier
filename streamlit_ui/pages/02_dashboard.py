"""Model Performance Dashboard page for the Customer Complaint Classifier UI."""

import os
import sys

import pandas as pd
import streamlit as st

_UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)

from utils.config import COLORS, MODELS
from utils.metrics_loader import load_all_metrics
from utils.visualizations import (
    accuracy_bar_chart,
    metrics_heatmap,
    radar_chart,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
def _configure_page():
    try:
        st.set_page_config(
            page_title="Performance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except st.errors.StreamlitAPIException:
        pass


# ---------------------------------------------------------------------------
# Insight generator
# ---------------------------------------------------------------------------

def _generate_insights(df: pd.DataFrame, sort_metric: str) -> list[str]:
    """Auto-generate bullet-point insights from the metrics DataFrame."""
    if df.empty:
        return ["No metrics data available."]

    insights = []
    best_row = df.loc[df[sort_metric].idxmax()]
    worst_row = df.loc[df[sort_metric].idxmin()]

    insights.append(
        f"✅ **{best_row['Model']}** achieves the highest {sort_metric}: "
        f"**{best_row[sort_metric]:.2%}**"
    )

    if "Baseline" in worst_row["Model"] or worst_row[sort_metric] < 0.5:
        insights.append(
            f"⚠️ **{worst_row['Model']}** performs poorly as expected "
            f"({worst_row[sort_metric]:.2%} {sort_metric})"
        )

    if "F1-Score" in df.columns:
        best_f1_row = df.loc[df["F1-Score"].idxmax()]
        insights.append(
            f"📊 **{best_f1_row['Model']}** leads in weighted F1-Score: "
            f"**{best_f1_row['F1-Score']:.2%}**"
        )

    # Agreement hint
    non_baseline = df[~df["Model"].str.contains("Baseline", case=False)]
    if len(non_baseline) >= 2:
        avg_acc = non_baseline["Accuracy"].mean()
        insights.append(
            f"📈 Average accuracy across non-baseline models: **{avg_acc:.2%}**"
        )

    return insights


# ---------------------------------------------------------------------------
# Main page function
# ---------------------------------------------------------------------------

def run():
    """Render the dashboard page."""
    _configure_page()

    st.title("📊 Model Performance Comparison Dashboard")
    st.markdown(
        "*Customer Complaint Classifier — 18 Financial Product Categories · "
        "366,945 real CFPB complaints*"
    )
    st.markdown("---")

    # Load metrics
    metrics_df = load_all_metrics()

    if metrics_df.empty:
        st.error(
            "⚠️ No metrics data found.  "
            "Ensure the results CSV files exist at `results/Kunal/` and `results/Dev/`."
        )
        return

    metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score"]
    available_metrics = [m for m in metric_columns if m in metrics_df.columns]
    all_model_names = metrics_df["Model"].tolist()

    # -----------------------------------------------------------------------
    # Sidebar filters
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("🔧 Filters")

        selected_models = st.multiselect(
            "Select models to compare:",
            options=all_model_names,
            default=all_model_names,
        )

        selected_metrics = st.multiselect(
            "Select metrics to display:",
            options=available_metrics,
            default=available_metrics,
        )

        sort_by = st.selectbox(
            "Sort models by:",
            options=available_metrics,
            index=0,
        )

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Filter dataframe
    filtered_df = metrics_df[metrics_df["Model"].isin(selected_models)].copy()
    if filtered_df.empty:
        st.warning("No data for selected filters.")
        return

    # -----------------------------------------------------------------------
    # Summary metric cards
    # -----------------------------------------------------------------------
    st.subheader("📌 Summary Statistics")
    num_cols = min(len(selected_metrics), 4)
    metric_cols = st.columns(num_cols) if num_cols else []

    for i, metric in enumerate(selected_metrics[:4]):
        best_row = filtered_df.loc[filtered_df[metric].idxmax()]
        with metric_cols[i]:
            st.metric(
                label=f"Best {metric}",
                value=f"{best_row[metric]:.2%}",
                delta=best_row["Model"],
            )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section A: Accuracy comparison
    # -----------------------------------------------------------------------
    st.subheader("📈 Section A — Overall Model Comparison")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**Accuracy Comparison (sorted by {sort_by})**")
        fig_bar = accuracy_bar_chart(filtered_df, sort_by)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col2:
        if len(selected_metrics) >= 2:
            st.markdown("**Multi-Metric Heatmap**")
            fig_heat = metrics_heatmap(filtered_df, selected_metrics)
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # Radar chart (full width)
    if len(selected_metrics) >= 3:
        st.markdown("**Radar Chart — Model Strengths & Weaknesses**")
        fig_radar = radar_chart(filtered_df, selected_metrics)
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section B: Detailed metrics table
    # -----------------------------------------------------------------------
    st.subheader("📋 Section B — Detailed Metrics Table")

    display_cols = ["Model"] + [m for m in selected_metrics if m in filtered_df.columns]
    display_df = filtered_df[display_cols].copy()

    # Format as percentages for readability
    for m in selected_metrics:
        if m in display_df.columns:
            display_df[m] = display_df[m].map(lambda v, _m=m: f"{v:.2%}")

    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section C: Summary & Insights
    # -----------------------------------------------------------------------
    st.subheader("💡 Section C — Automated Insights")

    insights = _generate_insights(filtered_df, sort_by)
    for insight in insights:
        st.markdown(f"- {insight}")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Model summary cards
    # -----------------------------------------------------------------------
    st.subheader("🗂️ Model Summary Cards")

    model_icon_map = {
        "baseline": "📊",
        "naive_bayes": "🔢",
        "svm": "⚡",
        "decision_tree": "🌳",
        "random_forest": "🌲",
    }

    cards = st.columns(min(len(filtered_df), 3))
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        col = cards[i % len(cards)]
        key = row.get("Model Key", "")
        icon = model_icon_map.get(key, "🤖")
        acc = row.get("Accuracy", 0)
        f1 = row.get("F1-Score", 0)
        color = "#2ca02c" if acc >= 0.75 else "#ff9900" if acc >= 0.50 else "#d62728"
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid {color}; border-radius:8px; padding:12px;
                            background:{color}11; margin-bottom:8px; text-align:center;">
                    <h4 style="margin:0;">{icon} {row['Model']}</h4>
                    <p style="margin:4px 0; font-size:1.4rem; font-weight:700; color:{color};">
                        {acc:.2%}
                    </p>
                    <p style="margin:0; color:#555; font-size:0.85rem;">Accuracy</p>
                    <hr style="margin:6px 0;">
                    <p style="margin:0; font-size:0.9rem;">F1-Score: <b>{f1:.2%}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )


run()
