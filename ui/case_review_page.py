"""
Unified Patient Case Review UI (Page 5).
Aggregates multimodal evidence across Imaging, Risk Model, and Guidelines with strict provenance tracking.
"""
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

from modules.orchestration.clinical_review import orchestrator_service
from modules.orchestration.schemas import UnifiedCaseSummary

def render_case_review_page():
    st.title("📋 Unified Patient Case Review")
    st.markdown(
        "Synthesizes multimodal findings across **Medical Imaging**, **Patient Risk Assessment**, and "
        "**Retrieved Clinical Guidelines** into a consolidated case review with **strict provenance tracking**."
    )

    # Pull active elements from session state
    patient_input = st.session_state.get("active_patient_obj", None)
    imaging_result = st.session_state.get("last_imaging_result", None)
    risk_result = st.session_state.get("last_risk_result", None)
    guideline_history = st.session_state.get("guideline_history", [])

    # Case ID / Selector
    case_id = patient_input.patient_id if patient_input else "CASE-ACTIVE-DEMO"
    
    # Build consolidated case summary
    summary: UnifiedCaseSummary = orchestrator_service.build_case_summary(
        case_id=case_id,
        patient_input=patient_input,
        imaging_result=imaging_result,
        risk_result=risk_result,
        guideline_answers=guideline_history
    )

    # Header Status Banner
    status_color = "#10b981" if summary.overall_status == "COMPLETE" else ("#f59e0b" if summary.overall_status == "PARTIAL" else "#64748b")
    st.markdown(
        f"""
        <div style="background: #1e293b; padding: 16px 24px; border-radius: 8px; border-left: 5px solid {status_color}; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #f8fafc;">Case Identifier: <code>{summary.case_id}</code></h3>
                <span style="background: {status_color}33; color: {status_color}; font-weight: 600; padding: 4px 12px; border-radius: 16px; border: 1px solid {status_color};">
                    Status: {summary.overall_status}
                </span>
            </div>
            <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 0.9rem;">Review Timestamp: {summary.timestamp}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Missing Data & Uncertainties Alerts
    if summary.missing_data_notices:
        for m in summary.missing_data_notices:
            st.info(f"ℹ️ **Incomplete Case Component:** {m}")

    if summary.conflicts_or_warnings:
        for c in summary.conflicts_or_warnings:
            st.warning(f"⚠️ **Clinical Conflict / Divergence:** {c}")

    # Tabs for structured multimodal review
    tab_provenance, tab_synthesis, tab_modalities, tab_export = st.tabs([
        "🔍 Evidence Provenance Table",
        "🧠 Clinical Synthesis & Correlations",
        "🩻 Individual Modality Panels",
        "📄 Export Case Report"
    ])

    with tab_provenance:
        st.subheader("Multimodal Data Origin & Provenance Audit Table")
        st.caption("Each finding is strictly attributed to its generation source. No finding from one module is treated as verifying another without evidence.")

        if summary.provenance_table:
            prov_records = [
                {
                    "Source Module": p.source_module,
                    "Clinical Element": p.element_name,
                    "Reported Observation": p.value,
                    "Confidence / Metric": p.confidence_or_status,
                    "Governance Boundary": p.provenance_disclaimer
                }
                for p in summary.provenance_table
            ]
            st.dataframe(pd.DataFrame(prov_records), hide_index=True, use_container_width=True)
        else:
            st.warning("No multimodal data has been submitted yet. Navigate to Imaging or Risk to enter patient data.")

    with tab_synthesis:
        if summary.differential_diagnosis:
            ddx = summary.differential_diagnosis
            prim = ddx.get("primary_diagnosis", {})
            st.markdown(f"### 🩺 Primary Working Diagnosis: {prim.get('condition_name', 'N/A')}")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.info(f"**ICD-10 Code:** `{prim.get('icd10_code', 'N/A')}` | **Likelihood:** {prim.get('probability_estimate_pct', 0)}%")
            with col_d2:
                st.caption(f"**Confirmatory Diagnostic Criteria:** {prim.get('confirmatory_criteria', 'N/A')}")
            st.markdown(f"*{prim.get('pathophysiological_summary', '')}*")
            st.markdown("---")

        st.subheader("Multimodal Clinical Correlation & Synthesis")
        if summary.clinical_synthesis_notes:
            for note in summary.clinical_synthesis_notes:
                st.markdown(f"• {note}")
        else:
            st.write("Complete both Medical Imaging and Risk Assessment to generate clinical correlation notes.")

        st.markdown("---")
        st.subheader("Documented Clinical Uncertainties & Limitations")
        if summary.unresolved_uncertainties:
            for unc in summary.unresolved_uncertainties:
                st.markdown(f"• *{unc}*")
        else:
            st.write("No unresolved uncertainties flagged.")

    with tab_modalities:
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### Module A: Radiograph & Saliency")
            if imaging_result and imaging_result.inference_status == "SUCCESS":
                st.write(f"**Top Finding:** {imaging_result.top_finding} ({imaging_result.top_confidence*100:.1f}%)")
                if imaging_result.heatmap_path and Path(imaging_result.heatmap_path).exists():
                    st.image(imaging_result.heatmap_path, caption="Grad-CAM Saliency Map", use_container_width=True)
            else:
                st.info("Medical Imaging not yet processed.")

        with col_m2:
            st.markdown("#### Module B: Patient Risk")
            if risk_result:
                st.write(f"**Outcome:** {risk_result.outcome}")
                st.write(f"**Risk Level:** **{risk_result.risk_category}** ({risk_result.risk_score*100:.1f}%)")
                st.write(f"**Cohort:** {risk_result.dataset_source}")
                top_fac = [f"{f.feature_name}: {f.feature_value} ({f.direction})" for f in risk_result.contributing_factors[:4]]
                st.write("**Top Factors:**")
                for tf in top_fac:
                    st.write(f"- {tf}")
            else:
                st.info("Risk assessment not yet calculated.")

    with tab_export:
        st.subheader("Generate Clinical Case Summary Document")
        
        # Build text export
        report_text = f"""================================================================================
CLINICAL COPILOT MULTIMODAL CASE REVIEW
Case ID: {summary.case_id}
Date: {summary.timestamp}
Status: {summary.overall_status}
================================================================================

1. PATIENT DEMOGRAPHICS & CLINICAL INTAKE
{f'Age: {patient_input.age}, Sex: {patient_input.sex}, Chest Pain: {patient_input.chest_pain_type}' if patient_input else 'Not recorded.'}
{f'Resting BP: {patient_input.resting_bp} mmHg, Cholesterol: {patient_input.cholesterol} mg/dL, Max HR: {patient_input.max_hr} bpm' if patient_input else ''}

2. MODULE A: MEDICAL IMAGING INTELLIGENCE
{f'Model: {imaging_result.model_name}' if imaging_result else 'No imaging data.'}
{f'Top Finding: {imaging_result.top_finding} (Confidence: {imaging_result.top_confidence*100:.1f}%)' if imaging_result else ''}

3. MODULE B: SUPERVISED RISK ASSESSMENT
{f'Model: {risk_result.model_name}' if risk_result else 'No risk score computed.'}
{f'Calibrated Risk: {risk_result.risk_category} ({risk_result.risk_score*100:.1f}%)' if risk_result else ''}
{f'Data Provenance: {risk_result.dataset_source}' if risk_result else ''}

4. CLINICAL SYNTHESIS & MULTIMODAL CORRELATIONS
{chr(10).join(['- ' + n for n in summary.clinical_synthesis_notes]) if summary.clinical_synthesis_notes else 'None.'}

5. UNRESOLVED UNCERTAINTIES & WARNINGS
{chr(10).join(['- ' + u for u in summary.unresolved_uncertainties]) if summary.unresolved_uncertainties else 'None.'}

================================================================================
GOVERNANCE DISCLAIMER:
{summary.governance_statement}
================================================================================
"""
        st.text_area("Case Summary Preview", value=report_text, height=260)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label="Download Structured Case Report (.txt)",
                data=report_text,
                file_name=f"clinical_copilot_case_{summary.case_id.lower()}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )
        with col_btn2:
            import json
            from modules.orchestration.fhir import fhir_service
            from modules.orchestration.ddx import ddx_engine
            ddx_obj = ddx_engine.generate_ddx(case_id=summary.case_id, patient=patient_input, imaging=imaging_result, risk=risk_result)
            fhir_bundle = fhir_service.build_fhir_r4_bundle(case_id=summary.case_id, patient=patient_input, imaging=imaging_result, risk=risk_result, ddx=ddx_obj)
            st.download_button(
                label="Download HL7 FHIR R4 Bundle (.json)",
                data=json.dumps(fhir_bundle, indent=2),
                file_name=f"case_{summary.case_id.lower()}_fhir_r4.json",
                mime="application/json",
                use_container_width=True
            )
