import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_fps_vs_map, load_phase1_heatmap, load_per_class_results

st.set_page_config(page_title="Surgical-YOLO: Position-Aware Attention", layout="wide")

st.sidebar.title("Surgical-YOLO")
st.sidebar.markdown(
    "Position-aware attention placement in YOLO11n-seg for real-time spinal "
    "endoscopic video segmentation — statistically validated across 166 "
    "training runs. Hybrid-L15CA: +2.82% mAP50-95, +18'%' FPS over baseline."
)
st.sidebar.markdown("[Full repo & thesis](https://github.com/raddy666/Surgical-YOLO-Position-Aware-Attention)")

st.title("Results Dashboard")

tab1, tab2, tab3 = st.tabs(["Speed vs. Accuracy", "Per-Structure Performance", "Phase 1: Position × Mechanism"])

with tab1:
    st.subheader("Accuracy vs. Speed Trade-off")
    df1 = load_fps_vs_map()
    fig1 = px.scatter(
        df1, x="fps", y="mean_map", text="config",
        hover_data={"cv_pct": ":.2f"},
        labels={"fps": "Inference Speed (FPS)", "mean_map": "mAP50-95"},
    )
    fig1.update_traces(textposition="top center", marker=dict(size=12))
    st.plotly_chart(fig1, use_container_width=True)

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
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Phase 1 Screening: Mean mAP50-95 by Position × Mechanism")
    pivot = load_phase1_heatmap()
    fig3 = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="RdYlGn", text=pivot.values, texttemplate="%{text:.3f}",
    ))
    st.plotly_chart(fig3, use_container_width=True)