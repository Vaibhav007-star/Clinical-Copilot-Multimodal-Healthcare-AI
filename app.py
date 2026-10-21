"""
Clinical Copilot — Main Streamlit Application Entrypoint.
Multimodal Clinical Assistant with Medical Imaging, Patient Risk Assessment, and RAG-Powered Clinical Guidelines.
"""
import streamlit as st

# Configure wide layout and page icon
st.set_page_config(
    page_title="Clinical Copilot — Multimodal Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Dark Clinical Theme
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    /* Button Aesthetics */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    /* Sidebar Headers */
    .css-1d391kg {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Import Pages
from ui.dashboard import render_clinical_overview
from ui.imaging_page import render_imaging_page
from ui.risk_page import render_risk_page
from ui.guidelines_page import render_guidelines_page
from ui.case_review_page import render_case_review_page
from ui.transparency_page import render_transparency_page

# Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="color: #38bdf8; margin: 0; font-size: 1.6rem;">⚕️ Clinical Copilot</h2>
    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">Multimodal Clinical Assistant</p>
</div>
""", unsafe_allow_html=True)

PAGES = {
    "Clinical Overview": render_clinical_overview,
    "Medical Imaging Analysis": render_imaging_page,
    "Patient Risk Assessment": render_risk_page,
    "Clinical Guidelines Assistant": render_guidelines_page,
    "Unified Patient Case Review": render_case_review_page,
    "Model & Data Transparency": render_transparency_page
}

# Sync with programmatic navigation if set in session state
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Clinical Overview"

# Sidebar Page Selector
selected_page = st.sidebar.radio(
    "Navigation Menu",
    options=list(PAGES.keys()),
    index=list(PAGES.keys()).index(st.session_state["nav_selection"])
)

# Keep session state in sync
st.session_state["nav_selection"] = selected_page

st.sidebar.markdown("---")

# Active Case Tracker in Sidebar
st.sidebar.markdown("##### 👤 Current Active Case")
patient_obj = st.session_state.get("active_patient_obj", None)
if patient_obj:
    st.sidebar.markdown(f"**ID:** `{patient_obj.patient_id}` | Age: {patient_obj.age} | Sex: {patient_obj.sex}")
else:
    st.sidebar.caption("No patient intake submitted yet. Use presets on Overview or enter details on Risk page.")

# Reset Case Button
if st.sidebar.button("Clear / Reset Active Case", use_container_width=True):
    for key in ["active_patient", "active_patient_obj", "last_imaging_result", "last_risk_result", "active_sample_image", "last_guideline_answer"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["guideline_history"] = []
    st.sidebar.success("Case data reset.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Clinical Copilot v1.0 | Decision Support Prototype | Not FDA cleared")

# Render Selected View
PAGES[selected_page]()
