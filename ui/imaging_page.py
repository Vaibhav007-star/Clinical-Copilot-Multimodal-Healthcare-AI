"""
Medical Imaging Analysis UI (Page 2).
Accepts Chest X-Ray uploads, validates formats, runs DenseNet121 inference, and visualizes Grad-CAM heatmaps.
"""
from pathlib import Path
import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd

from config.settings import settings
from modules.imaging.preprocessing import validate_image_file
from modules.imaging.inference import imaging_service

def render_imaging_page():
    st.title("🩻 Medical Imaging Intelligence")
    st.markdown(
        "Upload or select a frontal **Chest X-Ray (CXR)** radiograph for multi-pathology screening. "
        "The model computes deep convolutional feature maps with anatomical **Grad-CAM** saliency localization."
    )

    col_upload, col_preview = st.columns([1, 1])
    active_image = None
    filename = None

    with col_upload:
        st.subheader("1. Radiograph Input")
        
        # Preloaded Sample Selector
        sample_options = ["None (Upload Custom Image)", "Sample CXR - Cardiomegaly", "Sample CXR - Pneumonia", "Sample CXR - Normal Baseline"]
        sample_choice = st.selectbox("Or choose a preloaded clinical sample CXR:", sample_options)
        
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray Image", 
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPG, JPEG, WEBP. Maximum file size: 15MB."
        )

        if uploaded_file is not None:
            is_valid, err_msg, pil_img = validate_image_file(uploaded_file.getvalue(), uploaded_file.name)
            if not is_valid:
                st.error(f"Image Validation Error: {err_msg}")
            else:
                active_image = pil_img
                filename = uploaded_file.name
        elif sample_choice != "None (Upload Custom Image)":
            name_map = {
                "Sample CXR - Cardiomegaly": "sample_cxr_cardiomegaly.png",
                "Sample CXR - Pneumonia": "sample_cxr_pneumonia.png",
                "Sample CXR - Normal Baseline": "sample_cxr_normal.png"
            }
            sample_file = settings.sample_images_dir / name_map[sample_choice]
            if sample_file.exists():
                active_image = Image.open(sample_file).convert("RGB")
                filename = sample_file.name

        # Check if preloaded from overview case preset
        if active_image is None and "active_sample_image" in st.session_state:
            preset_file = settings.sample_images_dir / st.session_state["active_sample_image"]
            if preset_file.exists():
                active_image = Image.open(preset_file).convert("RGB")
                filename = preset_file.name
                st.info(f"Loaded from active case preset: `{filename}`")

        # Analysis Trigger Button
        if active_image is not None:
            st.success(f"Loaded `{filename}` ({active_image.width}x{active_image.height} px)")
            analyze_btn = st.button("Run Thoracic Screening & Grad-CAM →", type="primary", use_container_width=True)
            if analyze_btn:
                with st.spinner("Executing DenseNet121 inference and Grad-CAM gradient backpropagation..."):
                    result = imaging_service.analyze(active_image, filename)
                    st.session_state["last_imaging_result"] = result
                    st.toast("Imaging Analysis Complete!", icon="🩻")

    with col_preview:
        st.subheader("2. Image Inspection")
        if active_image is not None:
            st.image(active_image, caption=f"Original Radiograph: {filename}", use_container_width=True)
        else:
            st.info("Upload or select a chest X-ray on the left to preview the radiograph.")

    # Display Analysis Results
    if "last_imaging_result" in st.session_state:
        result = st.session_state["last_imaging_result"]
        st.markdown("---")
        st.subheader("3. Structured Radiological Findings & Saliency Map")

        # Metadata Header
        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
        with meta_c1:
            st.metric("Model Architecture", result.model_name)
        with meta_c2:
            st.metric("Top Abnormality", result.top_finding)
        with meta_c3:
            st.metric("Confidence Score", f"{result.top_confidence * 100:.1f}%")
        with meta_c4:
            st.metric("Saliency Overlay", "Available" if result.visualization_available else "N/A")

        # Grad-CAM Visualization vs Original
        vis_c1, vis_c2 = st.columns(2)
        with vis_c1:
            st.markdown("**Original Chest Radiograph**")
            if active_image:
                st.image(active_image, use_container_width=True)
        with vis_c2:
            st.markdown(f"**Grad-CAM Saliency Map for `{result.top_finding}`**")
            if result.visualization_available and result.heatmap_path and Path(result.heatmap_path).exists():
                heatmap_img = Image.open(result.heatmap_path)
                st.image(heatmap_img, caption="Red/Yellow: Highest feature activation contributing to top prediction.", use_container_width=True)
            else:
                st.write("Heatmap not generated for this image.")

        # Findings Bar Chart
        st.subheader("Thoracic Pathologies Multi-Label Probability Spectrum")
        df_findings = pd.DataFrame([
            {"Pathology": f.finding, "Confidence (%)": f.confidence * 100, "Severity": f.severity_category}
            for f in result.findings
        ])
        
        fig = px.bar(
            df_findings,
            x="Confidence (%)",
            y="Pathology",
            orientation="h",
            color="Confidence (%)",
            color_continuous_scale="Blues",
            range_x=[0, 100],
            text=df_findings["Confidence (%)"].apply(lambda v: f"{v:.1f}%")
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Limitations & Warnings
        if result.warnings:
            for w in result.warnings:
                st.warning(f"⚠️ {w}")
                
        st.caption(f"🔒 **Clinical Model Card & Disclaimer:** {result.limitations}")
