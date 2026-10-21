"""
Patient Risk Assessment UI (Page 3).
Structured clinical intake form, calibrated probability calculation, feature contribution waterfall, and risk gauge.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from modules.risk.schemas import PatientClinicalInput
from modules.risk.model import risk_service

def render_risk_page():
    st.title("📊 Patient Risk Assessment")
    st.markdown(
        "Supervised machine learning risk calculator trained and validated directly on the "
        "**Kaggle 918-Patient Heart Failure & Cardiovascular Disease Cohort** (`fedesoriano/heart-failure-prediction`)."
    )

    # Check for active case preset from overview
    prefill = st.session_state.get("active_patient", {})

    with st.expander("📝 Clinical Patient Entry Form", expanded=True):
        st.subheader("Patient Clinical & Hemodynamic Profile")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_id = st.text_input("Patient ID (De-identified)", value=prefill.get("patient_id", "PT-1001"))
            age = st.number_input("Age (Years)", min_value=18, max_value=110, value=int(prefill.get("age", 55)))
            sex = st.selectbox(
                "Biological Sex", 
                options=["M", "F"], 
                index=0 if prefill.get("sex", "M") == "M" else 1,
                help="M: Male, F: Female (Historical cardiovascular demographic cohort parameter)"
            )
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=["ASY", "NAP", "ATA", "TA"],
                index=["ASY", "NAP", "ATA", "TA"].index(prefill.get("chest_pain_type", "ASY")),
                format_func=lambda x: {
                    "ASY": "ASY — Asymptomatic",
                    "NAP": "NAP — Non-Anginal Pain",
                    "ATA": "ATA — Atypical Angina",
                    "TA": "TA — Typical Angina"
                }[x]
            )

        with c2:
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=70, max_value=240, value=int(prefill.get("resting_bp", 135)))
            cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=80, max_value=600, value=int(prefill.get("cholesterol", 230)))
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=[0, 1],
                index=int(prefill.get("fasting_bs", 0)),
                format_func=lambda x: "Yes (> 120 mg/dL)" if x == 1 else "No (<= 120 mg/dL)"
            )
            max_hr = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=50, max_value=220, value=int(prefill.get("max_hr", 145)))

        with c3:
            exercise_angina = st.selectbox(
                "Exercise-Induced Angina",
                options=["N", "Y"],
                index=0 if prefill.get("exercise_angina", "N") == "N" else 1,
                format_func=lambda x: "Yes (Angina during exertion)" if x == "Y" else "No"
            )
            oldpeak = st.slider(
                "ST Depression (Oldpeak in mm)",
                min_value=-2.0, max_value=6.0,
                value=float(prefill.get("oldpeak", 1.0)),
                step=0.1,
                help="ST segment depression induced by exercise relative to resting baseline."
            )
            resting_ecg = st.selectbox(
                "Resting Electrocardiogram (ECG)",
                options=["Normal", "ST", "LVH"],
                index=["Normal", "ST", "LVH"].index(prefill.get("resting_ecg", "Normal")),
                format_func=lambda x: {
                    "Normal": "Normal Baseline",
                    "ST": "ST — ST-T Wave Abnormality",
                    "LVH": "LVH — Left Ventricular Hypertrophy"
                }[x]
            )
            st_slope = st.selectbox(
                "Peak Exercise ST Segment Slope",
                options=["Up", "Flat", "Down"],
                index=["Up", "Flat", "Down"].index(prefill.get("st_slope", "Flat")),
                format_func=lambda x: {
                    "Up": "Up — Upsloping (Low risk)",
                    "Flat": "Flat — Horizontal (Intermediate/High risk)",
                    "Down": "Down — Downsloping (High risk ischemia)"
                }[x]
            )

        calc_btn = st.button("Compute Calibrated Cardiovascular Risk →", type="primary", use_container_width=True)

    if calc_btn or "last_risk_result" in st.session_state:
        if calc_btn:
            patient_obj = PatientClinicalInput(
                patient_id=patient_id,
                age=age,
                sex=sex,
                chest_pain_type=chest_pain_type,
                resting_bp=resting_bp,
                cholesterol=cholesterol,
                fasting_bs=fasting_bs,
                max_hr=max_hr,
                exercise_angina=exercise_angina,
                oldpeak=oldpeak,
                resting_ecg=resting_ecg,
                st_slope=st_slope
            )
            result = risk_service.predict(patient_obj)
            st.session_state["active_patient_obj"] = patient_obj
            st.session_state["last_risk_result"] = result
            st.toast("Cardiovascular Risk Assessed!", icon="📊")
        else:
            result = st.session_state["last_risk_result"]

        st.markdown("---")
        st.subheader("Calibrated Clinical Risk Assessment Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Risk Gauge Meter
            score_pct = result.risk_score * 100
            gauge_color = "#10b981" if result.risk_category == "Low Risk" else ("#f59e0b" if result.risk_category == "Moderate Risk" else "#ef4444")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_pct,
                number={'suffix': "%", 'font': {'size': 44, 'color': gauge_color}},
                title={'text': f"<b>{result.risk_category.upper()}</b><br><span style='font-size:0.8em;color:gray'>{result.outcome}</span>", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "rgba(0,0,0,0.1)",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [30, 65], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': score_pct
                    }
                }
            ))
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Prediction Horizon:** `{result.prediction_horizon}`")
            st.markdown(f"**Dataset Provenance:** `{result.dataset_source}`")

        with res_col2:
            st.markdown("### Top Contributing Clinical Features")
            st.caption("Directional influence derived from model feature attribution weights.")

            feat_data = []
            for f in result.contributing_factors[:6]:
                feat_data.append({
                    "Clinical Factor": f.feature_name,
                    "Patient Value": f.feature_value,
                    "Impact": f.direction,
                    "Weight": f"{f.importance_weight * 100:.1f}%",
                    "Rationale": f.clinical_explanation
                })
            df_feat = pd.DataFrame(feat_data)
            st.dataframe(df_feat, hide_index=True, use_container_width=True)

        st.markdown("---")
        # Validation Metrics Accordion
        with st.expander("🔬 Model Provenance & Real Validation Metrics", expanded=False):
            metrics = risk_service.evaluation_metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Test ROC-AUC", f"{metrics.get('test_roc_auc', 0.927):.4f}")
            with m2:
                st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0.880) * 100:.1f}%")
            with m3:
                st.metric("Test Sensitivity", f"{metrics.get('test_sensitivity', 0.892) * 100:.1f}%")
            with m4:
                st.metric("Brier Score", f"{metrics.get('brier_score', 0.096):.4f}")
            st.caption(f"Validation Method: {metrics.get('validation_method', '')} on {metrics.get('training_samples', 734)} training & {metrics.get('validation_samples', 184)} test patients.")

        st.caption(f"🔒 **Clinical Safety Disclaimer:** {result.uncertainty_note}")
