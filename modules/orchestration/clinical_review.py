"""
Clinical Orchestrator and Unified Case Review Engine.
Synthesizes findings across Imaging, Risk Assessment, and Guidelines while strictly preserving provenance.
"""
from datetime import datetime
from typing import List, Optional
from modules.imaging.schemas import ImagingAnalysisResult
from modules.risk.schemas import RiskAssessmentResult, PatientClinicalInput
from modules.rag.schemas import GuidelineAnswer
from modules.orchestration.schemas import UnifiedCaseSummary, ProvenanceItem

class ClinicalReviewOrchestrator:
    @staticmethod
    def build_case_summary(
        case_id: str,
        patient_input: Optional[PatientClinicalInput] = None,
        imaging_result: Optional[ImagingAnalysisResult] = None,
        risk_result: Optional[RiskAssessmentResult] = None,
        guideline_answers: Optional[List[GuidelineAnswer]] = None
    ) -> UnifiedCaseSummary:
        """
        Consolidates active multimodal case elements into a traceable case review object.
        Checks for discrepancies, missing modules, and clinical uncertainties.
        """
        provenance_table: List[ProvenanceItem] = []
        synthesis_notes: List[str] = []
        uncertainties: List[str] = []
        conflicts: List[str] = []
        missing_notices: List[str] = []

        # 1. Patient Input Provenance
        if patient_input:
            provenance_table.append(ProvenanceItem(
                source_module="Patient Input (EHR)",
                element_name="Demographics & History",
                value=f"Age {patient_input.age}, Sex {patient_input.sex}, Chest Pain: {patient_input.chest_pain_type}",
                confidence_or_status="Clinician Entered",
                provenance_disclaimer="Direct clinical input; unverified by external EHR."
            ))
            provenance_table.append(ProvenanceItem(
                source_module="Patient Input (Hemodynamics)",
                element_name="Vitals & Labs",
                value=f"BP {patient_input.resting_bp} mmHg, Chol {patient_input.cholesterol} mg/dL, HR {patient_input.max_hr} bpm",
                confidence_or_status="Clinician Entered",
                provenance_disclaimer="Point-of-care clinical measurements."
            ))
        else:
            missing_notices.append("Module B patient baseline measurements are incomplete or not yet submitted.")

        # 2. Imaging Provenance
        if imaging_result and imaging_result.inference_status == "SUCCESS":
            provenance_table.append(ProvenanceItem(
                source_module="Module A (Imaging)",
                element_name=f"Top Radiological Finding ({imaging_result.model_name})",
                value=f"{imaging_result.top_finding} (Confidence: {imaging_result.top_confidence * 100:.1f}%)",
                confidence_or_status=f"{imaging_result.top_confidence * 100:.1f}%",
                provenance_disclaimer="AI vision inference on uploaded 2D chest radiograph."
            ))
            if imaging_result.top_finding != "Normal / No Infiltrate":
                synthesis_notes.append(
                    f"Chest X-Ray demonstrates visual pattern suspicious for {imaging_result.top_finding} "
                    f"with {imaging_result.top_confidence * 100:.1f}% model confidence."
                )
            if imaging_result.warnings:
                uncertainties.extend(imaging_result.warnings)
        elif imaging_result and imaging_result.inference_status == "FAILED":
            conflicts.append("Imaging analysis failed during execution; image outputs are excluded from synthesis.")
        else:
            missing_notices.append("Module A medical image has not been processed for this case.")

        # 3. Risk Assessment Provenance
        if risk_result:
            provenance_table.append(ProvenanceItem(
                source_module="Module B (Risk Model)",
                element_name="Cardiovascular Risk Score",
                value=f"{risk_result.risk_category} ({risk_result.risk_score * 100:.1f}%)",
                confidence_or_status=f"{risk_result.risk_score * 100:.1f}%",
                provenance_disclaimer="Supervised ML prediction calibrated on Kaggle 918-patient Heart Failure cohort."
            ))
            synthesis_notes.append(
                f"Patient risk model categorizes cardiovascular risk as {risk_result.risk_category} "
                f"({risk_result.risk_score * 100:.1f}% calibrated probability)."
            )
            top_factors = [f"{f.feature_name} ({f.direction})" for f in risk_result.contributing_factors[:3]]
            synthesis_notes.append(f"Primary driving risk features: {', '.join(top_factors)}.")
            uncertainties.append(risk_result.uncertainty_note)
        else:
            missing_notices.append("Module B structured risk assessment has not been generated.")

        # 4. RAG Guidelines Provenance
        if guideline_answers:
            for g_ans in guideline_answers:
                if g_ans.citations:
                    top_c = g_ans.citations[0]
                    provenance_table.append(ProvenanceItem(
                        source_module="Module C (Guidelines RAG)",
                        element_name=f"Guideline Retrieval: '{g_ans.query[:30]}...'",
                        value=f"Matched: {top_c.document_title} (Page {top_c.page_number})",
                        confidence_or_status=f"Similarity {top_c.similarity_score:.2f}",
                        provenance_disclaimer="Retrieved from indexed clinical practice guidelines."
                    ))
                if not g_ans.evidence_sufficient:
                    uncertainties.append(f"Guideline search for '{g_ans.query}' yielded insufficient evidence.")

        # Cross-modality conflict / correlation detection
        if patient_input and imaging_result and risk_result:
            # Check if chest pain is atypical but imaging shows cardiomegaly
            if imaging_result.top_finding == "Cardiomegaly" and risk_result.risk_category in ["Moderate Risk", "High Risk"]:
                synthesis_notes.append(
                    "Clinical Correlation Alert: Patient's elevated cardiovascular risk score aligns with radiological "
                    "finding of cardiomegaly. Recommend echocardiography (TTE) per AHA/ACC HF guidelines."
                )
            if imaging_result.top_finding == "Pneumonia" and patient_input.chest_pain_type == "ASY":
                synthesis_notes.append(
                    "Diagnostic Divergence: CXR suggests pneumonia/infiltrate while patient reports asymptomatic cardiac presentation. "
                    "Evaluate for atypical lower respiratory infection."
                )

        # Generate Advanced Syntheses if sufficient data available
        multi_scores_data = None
        ddx_data = None
        safety_data = None

        if patient_input:
            try:
                from modules.risk.calculators import compute_multi_score_panel
                from modules.rag.safety import safety_guardrail_service
                multi_scores_data = compute_multi_score_panel(patient_input).model_dump()
                safety_data = safety_guardrail_service.evaluate_patient_safety(patient_input).model_dump()
            except Exception:
                pass

        if patient_input or (imaging_result and imaging_result.inference_status == "SUCCESS"):
            try:
                from modules.orchestration.ddx import ddx_engine
                from modules.risk.calculators import compute_multi_score_panel
                m_panel = compute_multi_score_panel(patient_input) if patient_input else None
                ddx_obj = ddx_engine.generate_ddx(
                    case_id=case_id,
                    patient=patient_input,
                    imaging=imaging_result,
                    risk=risk_result,
                    multi_scores=m_panel
                )
                ddx_data = ddx_obj.model_dump()
            except Exception:
                pass

        # Status determination
        if patient_input and imaging_result and risk_result:
            overall_status = "COMPLETE"
        elif patient_input or imaging_result or risk_result:
            overall_status = "PARTIAL"
        else:
            overall_status = "INCOMPLETE"

        return UnifiedCaseSummary(
            case_id=case_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            patient_input=patient_input,
            imaging_result=imaging_result,
            risk_result=risk_result,
            guideline_answers=guideline_answers or [],
            provenance_table=provenance_table,
            clinical_synthesis_notes=synthesis_notes,
            unresolved_uncertainties=uncertainties,
            conflicts_or_warnings=conflicts,
            missing_data_notices=missing_notices,
            differential_diagnosis=ddx_data,
            multi_scores=multi_scores_data,
            safety_review=safety_data,
            overall_status=overall_status
        )

# Global singleton orchestrator
orchestrator_service = ClinicalReviewOrchestrator()
