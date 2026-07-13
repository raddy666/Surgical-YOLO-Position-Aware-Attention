import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import CATEGORY_COLORS, CATEGORY_MAP, CONFIG_ORDER, load_class_distribution, load_fps_vs_map, load_phase1_heatmap, load_per_class_results, load_training_curves, load_confusion_matrix, compute_phase2_statistics,compute_structure_significance

st.set_page_config(page_title="Surgical-YOLO: Position-Aware Attention", layout="wide")

st.sidebar.title("Surgical-YOLO")
st.sidebar.markdown(
    "Position-aware attention placement in YOLO11n-seg for real-time spinal "
    "endoscopic video segmentation — statistically validated across 166 "
    "training runs. Hybrid-L15CA: +2.82% mAP50-95, +18'%' FPS over baseline."
)
st.sidebar.markdown("[Full repo & thesis](https://github.com/raddy666/Surgical-YOLO-Position-Aware-Attention)")

st.title("Results Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Speed vs. Accuracy", "Per-Structure Performance",
    "Phase 1: Position × Mechanism", "Training Diagnostics", "Confusion Matrix", "Phase 2 Statistics", 
    "Structure Significance & Generalizability", "Class Distribution"
])

with tab1:
    st.subheader("Accuracy vs. Speed Trade-off")
    df1 = load_fps_vs_map()
    fig1 = px.scatter(
        df1, x="fps", y="mean_map", text="config",
        hover_data={"cv_pct": ":.2f"},
        labels={"fps": "Inference Speed (FPS)", "mean_map": "mAP50-95"},
    )
    fig1.update_traces(textposition="top center", marker=dict(size=12))
    st.plotly_chart(fig1, width=True)

with tab2:
    st.subheader("Per-Structure mAP50-95")
    df2 = load_per_class_results()
    configs_available = df2["config"].unique().tolist()
    selected = st.multiselect("Configurations to compare", configs_available, default=configs_available)
    fig2 = px.bar(
        df2[df2["config"].isin(selected)],
        x="class_name", y="map50_95", color="config", barmode="group",
        labels={"class_name": "Anatomical Structure", "map50_95": "mAP50-95", "config": "Configuration"},
    )
    st.plotly_chart(fig2, width=True)

with tab3:
    st.subheader("Phase 1 Screening: mAP50-95 Improvement (Δ%) by Position × Mechanism")
    st.caption("50 epochs, 3 seeds, mean. Blank cells = combination not tested.")
    pivot = load_phase1_heatmap()
    fig3 = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="RdYlGn", zmid=0,
        text=pivot.values, texttemplate="%{text:.2f}",
        hoverongaps=False,
    ))
    fig3.update_layout(xaxis_title="Attention Mechanism", yaxis_title="Attention Position")
    st.plotly_chart(fig3, width=True)

with tab4:
    st.subheader("Training Diagnostics")
    curves_df = load_training_curves()

    metric_cols = [c for c in curves_df.columns if c not in ("epoch", "time", "config", "seed")]
    default_metric = "metrics/mAP50-95(M)"
    metric = st.selectbox(
        "Metric", metric_cols,
        index=metric_cols.index(default_metric) if default_metric in metric_cols else 0,
    )

    configs_available = sorted(curves_df["config"].unique())
    default_configs = [c for c in ["Baseline", "Hybrid-L15CA"] if c in configs_available] or configs_available[:2]
    selected_configs = st.multiselect("Configurations", configs_available, default=default_configs)

    view = st.radio("View", ["Mean across seeds", "All seeds individually"], horizontal=True)

    plot_df = curves_df[curves_df["config"].isin(selected_configs)]
    if view == "Mean across seeds":
        plot_df = plot_df.groupby(["config", "epoch"])[metric].mean().reset_index()
        fig4 = px.line(plot_df, x="epoch", y=metric, color="config")
    else:
        plot_df = plot_df.copy()
        plot_df["run"] = plot_df["config"] + " – seed" + plot_df["seed"].astype(str)
        fig4 = px.line(plot_df, x="epoch", y=metric, color="run")

    st.plotly_chart(fig4, width=True)

with tab5:
    st.subheader("Confusion Matrix (Normalized)")
    config_choice = st.selectbox("Model", ["Hybrid-L15CA", "Baseline", "Full-Triplet", "Full-CA"])
    cm_pivot = load_confusion_matrix(config=config_choice)
    fig5 = go.Figure(data=go.Heatmap(
        z=cm_pivot.values, x=cm_pivot.columns, y=cm_pivot.index,
        colorscale="Blues", text=cm_pivot.values, texttemplate="%{text:.2f}",
    ))
    fig5.update_layout(xaxis_title="Predicted", yaxis_title="True")
    st.plotly_chart(fig5, width=True)

with tab6:
    st.subheader("Phase 2 Configuration Comparison (100 Epochs, 10 Seeds Each)")
    stats_df = compute_phase2_statistics()
    stats_df["config"] = pd.Categorical(stats_df["config"], categories=CONFIG_ORDER, ordered=True)
    stats_df = stats_df.sort_values("config")
    stats_df["category"] = stats_df["config"].map(CATEGORY_MAP)

    def make_label(row):
        if row["config"] == "Baseline":
            return f"{row['mean_map']:.4f}"
        marker = "*" if (row["significant"] and row["stable"]) else "ns"
        marker += "†" if not row["stable"] else ""
        return f"{row['mean_map']:.4f}{marker}"
    stats_df["label"] = stats_df.apply(make_label, axis=1)

    fig6 = px.bar(
        stats_df, x="config", y="mean_map", color="category",
        error_y="std_map", text="label",
        color_discrete_map=CATEGORY_COLORS,
        labels={"mean_map": "Mask mAP50-95 (mean ± std, 10 seeds)", "config": ""},
    )
    fig6.update_traces(textposition="outside")
    fig6.add_hline(
        y=stats_df.loc[stats_df["config"] == "Baseline", "mean_map"].values[0],
        line_dash="dash", line_color="gray",
    )
    st.plotly_chart(fig6, width=True)
    st.caption("* p<0.05 vs baseline (and CV<1.5%) · ns = not significant · † fails CV<1.5% stability threshold")

with tab7:
    st.subheader("Effect Size & Generalizability")
    headline = ["Hybrid-L15CA", "Full-CA", "Full-Triplet", "Hybrid-MSCA"]

    stats_df = compute_phase2_statistics()
    stats_df = stats_df[stats_df["config"].isin(headline)]
    stats_df["config"] = pd.Categorical(stats_df["config"], categories=headline, ordered=True)
    stats_df = stats_df.sort_values("config")

    sig_df = compute_structure_significance(configs=headline)
    sig_df["config"] = pd.Categorical(sig_df["config"], categories=headline, ordered=True)
    sig_df = sig_df.sort_values("config")

    fig7 = make_subplots(rows=1, cols=2,
        subplot_titles=("Effect Size (Cohen's d)", "Generalizability vs. Stability"),
        specs=[[{}, {"secondary_y": True}]])

    fig7.add_trace(go.Bar(x=stats_df["config"], y=stats_df["cohens_d"], marker_color="#2E7D32",
                            text=stats_df["cohens_d"].round(3), textposition="outside", showlegend=False),
                    row=1, col=1)
    fig7.add_hline(y=0.8, line_dash="dash", line_color="red", row=1, col=1)

    fig7.add_trace(go.Bar(x=sig_df["config"], y=sig_df["n_significant_structures"],
                            marker_color="#4472C4", name="Structures p<0.05 (max 6)"),
                    row=1, col=2, secondary_y=False)
    fig7.add_trace(go.Scatter(x=stats_df["config"], y=stats_df["cv_pct"], mode="lines+markers",
                                marker_color="black", name="CV%"),
                    row=1, col=2, secondary_y=True)

    fig7.update_yaxes(title_text="Cohen's d", row=1, col=1)
    fig7.update_yaxes(title_text="Structures significantly improved", row=1, col=2, secondary_y=False, range=[0, 7])
    fig7.update_yaxes(title_text="CV%", row=1, col=2, secondary_y=True)
    st.plotly_chart(fig7, width=True)

with tab8:
    st.subheader("Dataset: Class Instance Distribution")
    dist_df = load_class_distribution()
    fig8 = px.bar(
        dist_df, x="class_name", y="instances", color="split", barmode="group",
        labels={"class_name": "Anatomical Structure", "instances": "Label Instances", "split": "Split"},
    )
    st.plotly_chart(fig8, width=True)
    st.caption(
        "Minority classes: Muscle and IntervertebralDiscHerniation have the fewest "
        "instances, which limits confidence in structure-level conclusions for those two."
    )