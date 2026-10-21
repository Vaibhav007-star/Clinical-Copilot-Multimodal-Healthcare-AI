"""
Clinical Overview Dashboard (Page 1).
Provides high-level system indicators, quick module navigation, demonstration case presets, and clinical governance notices.
"""
import streamlit as st
import plotly.graph_objects as go

def render_clinical_overview():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 24px; border-radius: 12px; margin-bottom: 24px; border-left: 5px solid #06b6d4;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 2.2rem; font-weight: 700;">
            Clinical Copilot <span style="font-size: 1rem; color: #38bdf8; vertical-align: middle; background: rgba(56, 189, 248, 0.15); padding: 4px 10px; border-radius: 20px;">v1.0 Healthcare AI</span>
        </h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 8px; margin-bottom: 0;">
            Multimodal Clinical Decision-Support Platform with Medical Imaging Intelligence, 
            Patient Risk Assessment, and RAG-Powered Clinical Practice Guidelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Clinical Governance Advisory Alert
    st.warning(
        "⚕️ **CLINICAL DECISION-SUPPORT NOTICE & RESEARCH PROTOTYPE:**\n\n"
        "Clinical Copilot is an experimental multimodal clinical decision-support tool designed for medical education and research. "
        "It is **not** an FDA-cleared diagnostic device and does not establish a definitive medical diagnosis or prescribe clinical therapy. "
        "Independent verification by a licensed attending physician is required before taking any diagnostic or therapeutic action."
    )

    # System Status & Model Availability Indicators
    st.subheader("System & Model Readiness Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Module A: Imaging (CXR)", value="Online", delta="DenseNet121 + GradCAM")
    with col2:
        st.metric(label="Module B: Risk Engine", value="Online", delta="Kaggle 918-Pt Cohort")
    with col3:
        st.metric(label="Module C: Guidelines RAG", value="Indexed", delta="FAISS + MiniLM")
    with col4:
        st.metric(label="Case Review Orchestrator", value="Active", delta="Multi-modal Provenance")

    st.markdown("---")

    # Quick Access Modules
    st.subheader("Multimodal Capability Modules")
    mod_col1, mod_col2, mod_col3 = st.columns(3)

    with mod_col1:
        st.markdown("""
        <div style="background: #1e293b; padding: 20px; border-radius: 10px; border-top: 4px solid #3b82f6; height: 100%;">
            <h3 style="color: #f8fafc; margin-top: 0;">🩻 Module A: Medical Imaging</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Analyzes Chest X-Ray (CXR) DICOM/PNG radiographs for thoracic pathologies (Cardiomegaly, Pneumonia, Consolidation, Effusion) with anatomical Grad-CAM visual heatmaps.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Imaging Analysis →", key="btn_goto_imaging", use_container_width=True):
            st.session_state["nav_selection"] = "Medical Imaging Analysis"
            st.rerun()

    with mod_col2:
        st.markdown("""
        <div style="background: #1e293b; padding: 20px; border-radius: 10px; border-top: 4px solid #10b981; height: 100%;">
            <h3 style="color: #f8fafc; margin-top: 0;">📊 Module B: Patient Risk</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Trained on the Kaggle 918-patient Heart Failure clinical dataset. Provides calibrated cardiovascular risk estimation, feature attribution, and diagnostic horizon tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Risk Assessment →", key="btn_goto_risk", use_container_width=True):
            st.session_state["nav_selection"] = "Patient Risk Assessment"
            st.rerun()

    with mod_col3:
        st.markdown("""
        <div style="background: #1e293b; padding: 20px; border-radius: 10px; border-top: 4px solid #f59e0b; height: 100%;">
            <h3 style="color: #f8fafc; margin-top: 0;">📚 Module C: Guidelines RAG</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Retrieves evidence passages from authoritative AHA/ACC Chest Pain, HF, and ATS/IDSA Pneumonia guidelines with strict page citations and evidence sufficiency audits.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Guidelines Assistant →", key="btn_goto_guidelines", use_container_width=True):
            st.session_state["nav_selection"] = "Clinical Guidelines Assistant"
            st.rerun()

    st.markdown("---")

    # Demonstration Cases
    st.subheader("Preloaded Clinical Demonstration Cases")
    st.caption("Select a clinically validated test case to preload multi-modal patient inputs, imaging, and clinical review.")
    
    demo_c1, demo_c2, demo_c3 = st.columns(3)
    with demo_c1:
        st.markdown("**Case #1: Acute Ischemic Heart Disease**")
        st.write("• Patient: 62 y/o Male with ASY chest discomfort")
        st.write("• Hemodynamics: BP 155/95, Chol 280 mg/dL, ST-depression")
        st.write("• CXR: Borderline cardiomegaly")
        if st.button("Load Case #1 Preset", key="load_case_1", use_container_width=True):
            st.session_state["active_patient"] = {
                "patient_id": "PT-DEMO-01",
                "age": 62,
                "sex": "M",
                "chest_pain_type": "ASY",
                "resting_bp": 155,
                "cholesterol": 280,
                "fasting_bs": 1,
                "resting_ecg": "ST",
                "max_hr": 118,
                "exercise_angina": "Y",
                "oldpeak": 2.2,
                "st_slope": "Flat"
            }
            st.session_state["active_sample_image"] = "sample_cxr_cardiomegaly.png"
            st.success("Case #1 preloaded! Navigate to Medical Imaging or Patient Risk to inspect.")

    with demo_c2:
        st.markdown("**Case #2: Lower Respiratory Infection (CAP)**")
        st.write("• Patient: 45 y/o Female with atypical non-anginal chest pain")
        st.write("• Hemodynamics: Normal BP 120/75, Normal ECG")
        st.write("• CXR: Right lower lobe consolidation")
        if st.button("Load Case #2 Preset", key="load_case_2", use_container_width=True):
            st.session_state["active_patient"] = {
                "patient_id": "PT-DEMO-02",
                "age": 45,
                "sex": "F",
                "chest_pain_type": "NAP",
                "resting_bp": 120,
                "cholesterol": 195,
                "fasting_bs": 0,
                "resting_ecg": "Normal",
                "max_hr": 165,
                "exercise_angina": "N",
                "oldpeak": 0.0,
                "st_slope": "Up"
            }
            st.session_state["active_sample_image"] = "sample_cxr_pneumonia.png"
            st.success("Case #2 preloaded! Navigate to Medical Imaging to inspect pneumonia pattern.")

    with demo_c3:
        st.markdown("**Case #3: Low-Risk Non-Cardiac Chest Pain**")
        st.write("• Patient: 34 y/o Male with atypical angina, high fitness")
        st.write("• Hemodynamics: BP 115/70, Normal Max HR 178 bpm")
        st.write("• CXR: Clear lung fields, normal cardiothoracic ratio")
        if st.button("Load Case #3 Preset", key="load_case_3", use_container_width=True):
            st.session_state["active_patient"] = {
                "patient_id": "PT-DEMO-03",
                "age": 34,
                "sex": "M",
                "chest_pain_type": "ATA",
                "resting_bp": 115,
                "cholesterol": 175,
                "fasting_bs": 0,
                "resting_ecg": "Normal",
                "max_hr": 178,
                "exercise_angina": "N",
                "oldpeak": 0.0,
                "st_slope": "Up"
            }
            st.session_state["active_sample_image"] = "sample_cxr_normal.png"
            st.success("Case #3 preloaded! Clear baseline ready for evaluation.")
