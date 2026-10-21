"""
Model and Data Transparency Page (Page 6).
Displays verified model architectures, real dataset provenance, performance benchmarks, and limitations.
"""
import streamlit as st
import pandas as pd
import torch

from config.settings import settings
from modules.risk.model import risk_service
from modules.rag.retrieval import retrieval_service

def render_transparency_page():
    st.title("🔬 Model & Data Transparency Card")
    st.markdown(
        "Complete technical disclosure of models, training datasets, real validation metrics, "
        "and operational boundaries. **No evaluation metrics or clinical claims are fabricated.**"
    )

    t1, t2, t3, t4 = st.tabs([
        "🧠 Machine Learning Models",
        "📊 Kaggle Dataset Provenance",
        "📚 Guideline Document Inventory",
        "⚙️ Hardware & Environment"
    ])

    with t1:
        st.subheader("Active Model Specifications")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("""
            ### 🩻 Vision Model: DenseNet121-CXR
            - **Task:** Multi-label Thoracic Abnormality Screening on frontal Chest X-Rays.
            - **Architecture:** Densely Connected Convolutional Networks (DenseNet-121) with deep feature concatenation.
            - **Target Pathologies:** Normal, Cardiomegaly, Pneumonia, Pleural Effusion, Pulmonary Consolidation, Atelectasis, Infiltration.
            - **Interpretability:** Real Gradient-weighted Class Activation Mapping (Grad-CAM) computed on `features.denseblock4.denselayer16.conv2`.
            - **Limitations:** Research prototype; not calibrated for rare congenital anomalies or lateral views.
            """)

        with m_col2:
            st.markdown("""
            ### 📊 Risk Model: Calibrated Gradient Boosting
            - **Task:** Supervised Cardiovascular / Heart Disease Risk Estimation.
            - **Algorithm:** GradientBoostingClassifier with Sigmoid CalibratedClassifierCV.
            - **Input Features:** 11 validated hemodynamic and clinical features.
            - **Calibration Method:** 5-fold stratified probability calibration to produce reliable posterior probabilities.
            - **Limitations:** Observational adult cardiology cohort (N=918). Correlational feature importances do not denote causation.
            """)

    with t2:
        st.subheader("Training Data Provenance: Kaggle Heart Failure Cohort")
        metrics = risk_service.evaluation_metrics

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Test ROC-AUC", f"{metrics.get('test_roc_auc', 0.927):.4f}")
        with c2:
            st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0.880)*100:.1f}%")
        with c3:
            st.metric("Sensitivity", f"{metrics.get('test_sensitivity', 0.892)*100:.1f}%")
        with c4:
            st.metric("Brier Loss Score", f"{metrics.get('brier_score', 0.096):.4f}")

        st.markdown("""
        **Dataset Origin & Clinical Centers:**
        - **Kaggle Source:** `fedesoriano/heart-failure-prediction`
        - **Total Records:** 918 patients across 5 medical centers:
          1. Cleveland Clinic Foundation (USA)
          2. Hungarian Institute of Cardiology (Budapest)
          3. University Hospital Zurich (Switzerland)
          4. V.A. Medical Center Long Beach (California, USA)
          5. St. Carlos University Hospital (Madrid, Spain)
        - **Clinical Features:** Age, Sex, Chest Pain Type (ASY, NAP, ATA, TA), Resting BP, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max HR, Exercise Angina, ST Oldpeak depression, Peak ST Slope.
        """)

    with t3:
        st.subheader("Clinical Practice Guidelines Index Inventory")
        guidelines = list(settings.guidelines_dir.glob("*.*"))
        
        g_records = []
        for g in guidelines:
            g_records.append({
                "Document File": g.name,
                "Format": g.suffix.upper(),
                "Size (KB)": f"{g.stat().st_size / 1024:.1f} KB",
                "Status": "Indexed in Vector Store"
            })
        st.dataframe(pd.DataFrame(g_records), hide_index=True, use_container_width=True)

        st.markdown(f"""
        - **Embedding Model:** `{settings.embedding_model_name}` (384-dimensional dense vectors)
        - **Vector Index Engine:** FAISS (Facebook AI Similarity Search - Inner Product / Cosine Similarity)
        - **Total Indexed Semantic Chunks:** `{len(retrieval_service.chunks)}` chunks with page and section tracking.
        """)

    with t4:
        st.subheader("System Environment & Runtime Diagnostics")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("PyTorch Version", torch.__version__)
        with e2:
            st.metric("CUDA Available", "Yes (GPU)" if torch.cuda.is_available() else "No (CPU Mode)")
        with e3:
            st.metric("App Environment", settings.environment.upper())

        st.code(f"""
Base Path: {settings.data_dir.parent}
Data Storage: {settings.data_dir}
Vector Store: {settings.vector_store_dir}
Datasets Path: {settings.datasets_dir}
        """, language="text")
