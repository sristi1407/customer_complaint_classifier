"""Visualization helpers using Plotly for the Customer Complaint Classifier UI."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import COLORS, CATEGORIES


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

_MODEL_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def _bar_color(value: float) -> str:
    """Return a green/yellow/red color based on a 0–1 metric value."""
    if value >= 0.75:
        return COLORS["success"]
    if value >= 0.50:
        return COLORS["warning"]
    return COLORS["danger"]


# ---------------------------------------------------------------------------
# Dashboard Charts
# ---------------------------------------------------------------------------

def accuracy_bar_chart(df: pd.DataFrame, sort_by: str = "Accuracy") -> go.Figure:
    """Horizontal bar chart comparing model accuracy (or another metric).

    Parameters
    ----------
    df:
        DataFrame with columns ``Model``, ``Accuracy``, etc.
    sort_by:
        Column name to sort models by.

    Returns
    -------
    plotly Figure
    """
    df_sorted = df.sort_values(sort_by, ascending=True).reset_index(drop=True)
    colors = [_bar_color(v) for v in df_sorted[sort_by]]

    fig = go.Figure(
        go.Bar(
            x=df_sorted[sort_by],
            y=df_sorted["Model"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2%}" for v in df_sorted[sort_by]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>" + sort_by + ": %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Model Comparison — {sort_by}",
        xaxis_title=sort_by,
        yaxis_title="",
        xaxis=dict(range=[0, 1.1], tickformat=".0%"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(l=10, r=80, t=50, b=40),
        font=dict(size=13),
    )
    return fig


def metrics_heatmap(df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Heatmap of selected metrics across models.

    Parameters
    ----------
    df:
        DataFrame with ``Model`` column and metric columns.
    metrics:
        Which metric columns to include.

    Returns
    -------
    plotly Figure
    """
    plot_df = df[["Model"] + [m for m in metrics if m in df.columns]].set_index("Model")
    z = plot_df.values
    x_labels = list(plot_df.columns)
    y_labels = list(plot_df.index)

    text_vals = [[f"{v:.2%}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            text=text_vals,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Score", tickformat=".0%"),
        )
    )
    fig.update_layout(
        title="Multi-Metric Performance Heatmap",
        height=320,
        margin=dict(l=10, r=10, t=50, b=40),
        font=dict(size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def radar_chart(df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Radar / spider chart showing all metrics per model.

    Parameters
    ----------
    df:
        DataFrame with ``Model`` column and metric columns.
    metrics:
        Metric columns to use as radar dimensions.

    Returns
    -------
    plotly Figure
    """
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return go.Figure()

    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in available_metrics]
        # Close the polygon
        values_closed = values + [values[0]]
        metrics_closed = available_metrics + [available_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=metrics_closed,
                fill="toself",
                name=row["Model"],
                line_color=_MODEL_COLORS[i % len(_MODEL_COLORS)],
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Radar Chart — Model Strengths & Weaknesses",
        height=420,
        margin=dict(l=60, r=60, t=60, b=40),
        font=dict(size=12),
        paper_bgcolor="white",
    )
    return fig


def category_bar_chart(df: pd.DataFrame) -> go.Figure | None:
    """Grouped bar chart for per-category metrics (if data available).

    Expects ``df`` to have columns ``Category``, ``Model``, ``F1-Score``.
    Returns None if the DataFrame is empty.
    """
    if df is None or df.empty:
        return None
    fig = px.bar(
        df,
        x="Category",
        y="F1-Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=_MODEL_COLORS,
        title="Per-Category F1-Score by Model",
        labels={"F1-Score": "F1-Score", "Category": "Product Category"},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=450,
        margin=dict(l=10, r=10, t=50, b=120),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Classification-page charts
# ---------------------------------------------------------------------------

def top5_bar(categories: list[str], probabilities: list[float]) -> go.Figure:
    """Horizontal bar chart of top-5 predicted categories for a single prediction.

    Parameters
    ----------
    categories:
        List of category names (ordered by probability descending).
    probabilities:
        Corresponding probability values (0–1).

    Returns
    -------
    plotly Figure
    """
    cats = categories[:5]
    probs = probabilities[:5]

    colors = [_bar_color(p) for p in probs]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=cats,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1.15], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        height=220,
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=False,
    )
    return fig
