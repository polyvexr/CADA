"""
Interactive Streamlit Dashboard for CADA Driving Telemetry Monitoring & Risk Analysis.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, MODELS_DIR
from src.data.loader import load_motion_data
from src.data.preprocessor import MotionDataPreprocessor
from src.features.kinematics import KinematicFeatureExtractor
from src.scoring.cada_scorer import CADACompositeScorer
from src.models.trainer import train_cada_models

st.set_page_config(
    page_title="CADA : Continuous Anomaly Detection Architecture",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-normal { color: #28a745; font-weight: bold; }
    .badge-moderate { color: #ffc107; font-weight: bold; }
    .badge-high { color: #fd7e14; font-weight: bold; }
    .badge-critical { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_scorer():
    bundle_path = MODELS_DIR / "cada_model_bundle.joblib"
    if bundle_path.exists():
        return CADACompositeScorer.load(bundle_path)
    return train_cada_models()


scorer = get_scorer()

st.title("CADA : Continuous Anomaly Detection Algorithm")
st.caption("Real-Time Multi-Axis IMU Driving Risk Assessment & Safety Monitoring")

# Sidebar Configuration
st.sidebar.header("Telemetry Source")
data_source = st.sidebar.radio(
    "Select Telemetry Feed:",
    ["Default Test Dataset", "Upload Custom CSV"]
)

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Motion Telemetry CSV", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a telemetry CSV or switch to default test dataset.")
        st.stop()
else:
    test_path = RAW_DATA_DIR / "test_motion_data.csv"
    df_raw = load_motion_data(test_path)

# Preprocessing & Scoring
preprocessor = MotionDataPreprocessor()
kinematics = KinematicFeatureExtractor()
df_clean = preprocessor.fit_transform(df_raw)
df_feat = kinematics.fit_transform(df_clean)
df_scored = scorer.score_batch(df_feat)

# Top Key Performance Indicators
col1, col2, col3, col4, col5 = st.columns(5)

mean_score = df_scored['CADA_Score'].mean()
max_score = df_scored['CADA_Score'].max()
critical_count = (df_scored['Risk_Tier'] == 'CRITICAL').sum()
high_count = (df_scored['Risk_Tier'] == 'HIGH').sum()
total_samples = len(df_scored)

with col1:
    st.metric("Total Samples", f"{total_samples:,}")
with col2:
    st.metric("Mean CADA Score", f"{mean_score:.1f} / 100")
with col3:
    st.metric("Max Peak Score", f"{max_score:.1f} / 100")
with col4:
    st.metric("High Risk Events", f"{high_count} ({high_count/total_samples*100:.1f}%)")
with col5:
    st.metric("Critical Events", f"{critical_count} ({critical_count/total_samples*100:.1f}%)", delta_color="inverse")

st.divider()

# Main Visualizations Tabs
tab_timeline, tab_gauge, tab_breakdown, tab_raw, tab_manual = st.tabs([
    "Risk Timeline & Sensor Traces",
    "Live Sample Inspector & Gauge",
    "Risk Component Decomposition",
    "Data Explorer",
    "Unit Data Evaluation"
])

with tab_timeline:
    st.subheader("Continuous Driving Risk vs. Multi-Axis Sensor Telemetry")
    
    # Range slider for inspection
    x_axis = df_scored['Timestamp'] if 'Timestamp' in df_scored.columns else df_scored.index
    
    # 1. CADA Score Timeline with Colored Bands
    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=x_axis, y=df_scored['CADA_Score'],
        mode='lines', name='CADA Risk Score',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Threshold background zones
    fig_score.add_hrect(y0=0, y1=25, fillcolor="green", opacity=0.1, line_width=0, annotation_text="NORMAL")
    fig_score.add_hrect(y0=25, y1=50, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="MODERATE")
    fig_score.add_hrect(y0=50, y1=75, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="HIGH")
    fig_score.add_hrect(y0=75, y1=100, fillcolor="red", opacity=0.2, line_width=0, annotation_text="CRITICAL")

    fig_score.update_layout(
        title="Composite CADA Risk Score Over Time",
        xaxis_title="Timestamp / Step",
        yaxis_title="Risk Score (0 - 100)",
        yaxis_range=[0, 100],
        height=350,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # 2. Sensor Acceleration & Gyro Traces
    col_a, col_b = st.columns(2)
    with col_a:
        fig_acc = px.line(
            df_scored, x=x_axis, y=['AccX', 'AccY', 'AccZ', 'AccMag'],
            title="Tri-Axial Acceleration & Total Magnitude",
            labels={'value': 'Acceleration (g)', 'variable': 'Axis'}
        )
        fig_acc.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=30))
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_b:
        fig_gyro = px.line(
            df_scored, x=x_axis, y=['GyroX', 'GyroY', 'GyroZ', 'GyroMag'],
            title="Tri-Axial Angular Velocity & Total Magnitude",
            labels={'value': 'Angular Velocity (rad/s)', 'variable': 'Axis'}
        )
        fig_gyro.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=30))
        st.plotly_chart(fig_gyro, use_container_width=True)

with tab_gauge:
    st.subheader("Point-in-Time Telemetry Inspector & Risk Gauge")
    sample_idx = st.slider("Select Telemetry Sample Index:", 0, len(df_scored) - 1, 0)
    sample = df_scored.iloc[sample_idx]

    col_g1, col_g2, col_g3 = st.columns([1.5, 1, 1])

    with col_g1:
        # Speedometer Gauge
        tier_label = sample['Risk_Tier']
        st.markdown(f"#### CADA Risk Score: **{sample['CADA_Score']:.1f}**")
        st.caption(f"Operational Tier: **{tier_label}**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sample['CADA_Score'],
            number={'font': {'size': 44}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#666"},
                'bar': {'color': "#111827"},
                'steps': [
                    {'range': [0, 25], 'color': '#86efac'},
                    {'range': [25, 50], 'color': '#fde047'},
                    {'range': [50, 75], 'color': '#fdba74'},
                    {'range': [75, 100], 'color': '#fca5a5'}
                ]
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        st.markdown("#### Risk Components")
        st.metric("Isolation Forest Risk", f"{sample['Iso_Risk']:.1f}")
        st.metric("Statistical Z-Deviation", f"{sample['Stat_Risk']:.1f}")
        st.metric("Temporal Jerk Risk", f"{sample['Temporal_Risk']:.1f}")

    with col_g3:
        st.markdown("#### Kinematics")
        st.metric("Acc Magnitude", f"{sample['AccMag']:.3f} g")
        st.metric("Gyro Magnitude", f"{sample['GyroMag']:.3f} rad/s")
        st.metric("Acc Jerk (ΔAcc)", f"{sample['AccMag_Change']:.3f}")

with tab_breakdown:
    st.subheader("Sub-Component Attribution & Risk Tier Summary")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        tier_counts = df_scored['Risk_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Risk_Tier', 'Count']
        fig_pie = px.pie(
            tier_counts, names='Risk_Tier', values='Count',
            title='Operational Risk Tier Distribution',
            color='Risk_Tier',
            color_discrete_map={'NORMAL': '#28a745', 'MODERATE': '#ffc107', 'HIGH': '#fd7e14', 'CRITICAL': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_d2:
        if 'Class' in df_scored.columns:
            comp_by_class = df_scored.groupby('Class')[['Iso_Risk', 'Stat_Risk', 'Temporal_Risk', 'CADA_Score']].mean().reset_index()
            fig_bar = px.bar(
                comp_by_class, x='Class', y=['Iso_Risk', 'Stat_Risk', 'Temporal_Risk'],
                barmode='group',
                title='Mean Risk Attribution by Ground Truth Behavior Class'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            fig_hist = px.histogram(df_scored, x='CADA_Score', nbins=40, title='CADA Score Frequency Distribution')
            st.plotly_chart(fig_hist, use_container_width=True)

with tab_raw:
    st.subheader("Scored Telemetry Data View")
    st.dataframe(df_scored, use_container_width=True)
    csv_bytes = df_scored.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Scored CSV",
        data=csv_bytes,
        file_name="cada_scored_telemetry.csv",
        mime="text/csv"
    )

with tab_manual:
    st.subheader("Manual Unit Telemetry Evaluation")
    st.caption("Input custom 6-DOF IMU sensor readings to evaluate instantaneous CADA risk score, sub-components, and safety tier.")

    # Preset scenarios for quick testing
    preset_choice = st.selectbox(
        "Load Preset Example (Optional):",
        [
            "Custom Manual Input",
            "Normal Driving (NORMAL)",
            "Moderate Turn / Acceleration (MODERATE)",
            "Sharp Maneuver / Swerve (HIGH)",
            "Emergency Braking / Shock (CRITICAL)"
        ]
    )

    defaults = {
        "AccX": 0.0, "AccY": 0.0, "AccZ": 0.0,
        "GyroX": 0.0, "GyroY": 0.0, "GyroZ": 0.0
    }
    if preset_choice == "Normal Driving (NORMAL)":
        defaults = {"AccX": 0.7582, "AccY": -0.2178, "AccZ": 0.4573, "GyroX": 0.0000, "GyroY": 0.0000, "GyroZ": 0.0000}
    elif preset_choice == "Moderate Turn / Acceleration (MODERATE)":
        defaults = {"AccX": 0.5709, "AccY": 0.9929, "AccZ": 0.0296, "GyroX": 0.0269, "GyroY": -0.0297, "GyroZ": 0.0359}
    elif preset_choice == "Sharp Maneuver / Swerve (HIGH)":
        defaults = {"AccX": 0.4422, "AccY": -1.6468, "AccZ": 0.0653, "GyroX": -0.0104, "GyroY": -0.0242, "GyroZ": 0.0011}
    elif preset_choice == "Emergency Braking / Shock (CRITICAL)":
        defaults = {"AccX": 2.7244, "AccY": -7.5841, "AccZ": 2.3909, "GyroX": 0.0238, "GyroY": 0.0137, "GyroZ": -0.0380}

    with st.form("manual_evaluation_form"):
        st.markdown("##### 1. Accelerometer Readings (g)")
        c_acc1, c_acc2, c_acc3 = st.columns(3)
        with c_acc1:
            in_acc_x = st.number_input("AccX (Lateral)", value=float(defaults["AccX"]), step=0.05, format="%.4f")
        with c_acc2:
            in_acc_y = st.number_input("AccY (Longitudinal)", value=float(defaults["AccY"]), step=0.05, format="%.4f")
        with c_acc3:
            in_acc_z = st.number_input("AccZ (Vertical)", value=float(defaults["AccZ"]), step=0.05, format="%.4f")

        st.markdown("##### 2. Gyroscope Readings (rad/s or deg/s)")
        c_gyr1, c_gyr2, c_gyr3 = st.columns(3)
        with c_gyr1:
            in_gyr_x = st.number_input("GyroX (Pitch Rate)", value=float(defaults["GyroX"]), step=0.01, format="%.4f")
        with c_gyr2:
            in_gyr_y = st.number_input("GyroY (Roll Rate)", value=float(defaults["GyroY"]), step=0.01, format="%.4f")
        with c_gyr3:
            in_gyr_z = st.number_input("GyroZ (Yaw Rate)", value=float(defaults["GyroZ"]), step=0.01, format="%.4f")

        submitted = st.form_submit_button("Evaluate Telemetry Point", use_container_width=True)

    # Reset streaming state so manual single-point evaluation is pure point-in-time
    scorer.reset_stream_state()
    manual_sample = {
        "AccX": in_acc_x,
        "AccY": in_acc_y,
        "AccZ": in_acc_z,
        "GyroX": in_gyr_x,
        "GyroY": in_gyr_y,
        "GyroZ": in_gyr_z
    }
    result = scorer.score_sample(manual_sample)

    st.markdown("---")
    st.markdown("### Evaluation Results")

    res_col1, res_col2 = st.columns([1.2, 1.8])

    with res_col1:
        # Gauge visualization
        score_val = result["CADA_Score"]
        tier_val = result["Risk_Tier"]
        tier_colors = {
            "NORMAL": "#28a745",
            "MODERATE": "#ffc107",
            "HIGH": "#fd7e14",
            "CRITICAL": "#dc3545"
        }
        st.markdown(f"#### Safety Tier: <span style='color:{tier_colors.get(tier_val, '#111827')}'>{tier_val}</span>", unsafe_allow_html=True)
        st.caption(f"Score: **{score_val:.2f} / 100**")
        fig_eval_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_val,
            number={'font': {'size': 44}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#666"},
                'bar': {'color': tier_colors.get(tier_val, '#111827')},
                'steps': [
                    {'range': [0, 25], 'color': '#86efac'},
                    {'range': [25, 50], 'color': '#fde047'},
                    {'range': [50, 75], 'color': '#fdba74'},
                    {'range': [75, 100], 'color': '#fca5a5'}
                ]
            }
        ))
        fig_eval_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_eval_gauge, use_container_width=True)

    with res_col2:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Composite CADA Score", f"{score_val:.2f} / 100")
            st.metric("Novelty Risk (Iso_Risk)", f"{result['Iso_Risk']:.2f}")
            st.metric("Statistical Deviation (Stat_Risk)", f"{result['Stat_Risk']:.2f}")
        with m2:
            st.metric("Assigned Safety Tier", tier_val)
            st.metric("Temporal Jerk Risk", f"{result['Temporal_Risk']:.2f}")
            st.metric("Derived Acc Magnitude", f"{result['AccMag']:.4f} g")

    # Detailed breakdown table
    st.markdown("##### Detailed Output Summary")
    df_eval_summary = pd.DataFrame([{
        "CADA_Score": result["CADA_Score"],
        "Risk_Tier": result["Risk_Tier"],
        "Iso_Risk": result["Iso_Risk"],
        "Stat_Risk": result["Stat_Risk"],
        "Temporal_Risk": result["Temporal_Risk"],
        "AccMag": result["AccMag"],
        "GyroMag": result["GyroMag"],
        "AccMag_Change": result["AccMag_Change"],
        "GyroMag_Change": result["GyroMag_Change"]
    }])
    st.dataframe(df_eval_summary, use_container_width=True)

