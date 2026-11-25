"""
Multimodal Triage Acuity Radar & Automated Clinical Documentation Engine.
Computes an ESI-inspired acuity level, 5-axis multimodal risk radar, and generates
structured Physician SOAP notes, ED SBAR handoffs, and plain-language Patient Summaries.
"""
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field

from modules.imaging.schemas import ImagingAnalysisResult
from modules.risk.schemas import PatientClinicalInput, RiskAssessmentResult
from modules.risk.calculators import compute_multi_score_panel, MultiScoreSummary
from modules.rag.schemas import GuidelineAnswer
from modules.rag.safety import safety_guardrail_service
from modules.orchestration.ddx import ddx_engine, DifferentialDiagnosisResult


class TriageAxisScore(BaseModel):
    axis: str
    score: float = Field(..., ge=0.0, le=100.0)
    detail: str


class MultimodalTriageSoapResult(BaseModel):
    case_id: str
    generated_at: str
    esi_level: int = Field(..., ge=1, le=5)
    esi_label: str
    acuity_score: float = Field(..., ge=0.0, le=100.0)
    disposition: str
    axes: List[TriageAxisScore]
    soap_note: str
    sbar_handoff: str
    patient_summary: str


class TriageAndSoapEngine:
    @staticmethod
    def evaluate(
        case_id: str,
        patient: Optional[PatientClinicalInput] = None,
        imaging: Optional[ImagingAnalysisResult] = None,
        risk: Optional[RiskAssessmentResult] = None,
        guideline_answers: Optional[List[GuidelineAnswer]] = None,
        multi_scores: Optional[MultiScoreSummary] = None,
        ddx: Optional[DifferentialDiagnosisResult] = None,
    ) -> MultimodalTriageSoapResult:
        """Synthesizes 5-axis multimodal acuity radar, ESI triage level, and clinical notes."""
        if patient and multi_scores is None:
            multi_scores = compute_multi_score_panel(patient)

        if ddx is None and (patient or imaging or risk):
            ddx = ddx_engine.generate_ddx(
                case_id=case_id,
                patient=patient,
                imaging=imaging,
                risk=risk,
                multi_scores=multi_scores,
            )

        safety_report = safety_guardrail_service.evaluate_patient_safety(patient) if patient else None

        # 1. Radiographic Burden Axis
        if imaging and imaging.inference_status == "SUCCESS":
            if imaging.top_finding == "Normal / No Infiltrate":
                rad_score = round(min(20.0, (1.0 - imaging.top_confidence) * 40.0), 1)
                rad_detail = f"Clear CXR ({imaging.top_confidence * 100:.0f}% Normal)"
            else:
                rad_score = round(min(100.0, imaging.top_confidence * 100.0), 1)
                rad_detail = f"{imaging.top_finding} ({imaging.top_confidence * 100:.0f}%)"
        else:
            rad_score = 0.0
            rad_detail = "No CXR analyzed"

        # 2. Ischemic / ECG Stress Axis
        isc_score = 10.0
        isc_parts: List[str] = []
        if patient:
            if patient.oldpeak >= 2.0:
                isc_score += 35.0
                isc_parts.append(f"ST dep {patient.oldpeak}mm")
            elif patient.oldpeak >= 1.0:
                isc_score += 20.0
                isc_parts.append(f"ST dep {patient.oldpeak}mm")
            if patient.exercise_angina == "Y":
                isc_score += 25.0
                isc_parts.append("Exertional angina")
            if patient.resting_ecg in ("ST", "LVH"):
                isc_score += 18.0
                isc_parts.append(f"ECG {patient.resting_ecg}")
            if patient.chest_pain_type in ("TA", "ASY"):
                isc_score += 12.0
        isc_score = round(min(100.0, isc_score), 1)
        isc_detail = ", ".join(isc_parts) if isc_parts else "No acute ischemic markers"

        # 3. Hemodynamic Load Axis
        hemo_score = round(risk.risk_score * 100.0, 1) if risk else 15.0
        if patient and (patient.resting_bp >= 160 or patient.resting_bp <= 95):
            hemo_score = round(min(100.0, max(hemo_score, 72.0)), 1)
        hemo_detail = (
            f"CVD Risk {risk.risk_score * 100:.0f}%, BP {patient.resting_bp} mmHg"
            if (risk and patient)
            else (f"BP {patient.resting_bp} mmHg" if patient else "Awaiting vitals")
        )

        # 4. Thromboembolic / PE Risk Axis
        if multi_scores:
            wells = multi_scores.wells_pe.score
            pe_score = round(min(100.0, (wells / 9.0) * 100.0 + 10.0), 1)
            pe_detail = f"Wells PE {wells:.1f} ({multi_scores.wells_pe.risk_tier})"
        else:
            pe_score = 10.0
            pe_detail = "Wells PE not computed"

        # 5. Infectious / Respiratory Severity Axis
        resp_score = 10.0
        if multi_scores:
            resp_score = max(resp_score, multi_scores.curb65.score * 20.0 + 10.0)
        if imaging and imaging.inference_status == "SUCCESS":
            for f in imaging.findings:
                if f.finding in ("Pneumonia", "Consolidation", "Pleural Effusion"):
                    resp_score = max(resp_score, f.confidence * 100.0)
        resp_score = round(min(100.0, resp_score), 1)
        resp_detail = (
            f"CURB-65 {multi_scores.curb65.score}/5"
            if multi_scores
            else "Awaiting respiratory scores"
        )

        axes = [
            TriageAxisScore(axis="Radiographic Burden", score=rad_score, detail=rad_detail),
            TriageAxisScore(axis="Ischemic / ECG Stress", score=isc_score, detail=isc_detail),
            TriageAxisScore(axis="Hemodynamic Load", score=hemo_score, detail=hemo_detail),
            TriageAxisScore(axis="Thromboembolic Risk", score=pe_score, detail=pe_detail),
            TriageAxisScore(axis="Respiratory Severity", score=resp_score, detail=resp_detail),
        ]

        acuity_score = round(
            0.25 * rad_score + 0.25 * isc_score + 0.25 * hemo_score + 0.10 * pe_score + 0.15 * resp_score,
            1,
        )
        red_flags = len(ddx.red_flag_alerts) if ddx else 0

        if acuity_score >= 65.0 or red_flags >= 2:
            esi_level = 2
            esi_label = "ESI-2 Emergent"
            disposition = "Immediate Telemetry / Continuous ECG, STAT Labs & Specialist Consult"
        elif acuity_score >= 40.0 or red_flags == 1:
            esi_level = 3
            esi_label = "ESI-3 Urgent"
            disposition = "Expedited ED / Acute Care Evaluation with Serial Biomarkers"
        elif acuity_score >= 22.0:
            esi_level = 4
            esi_label = "ESI-4 Less Urgent"
            disposition = "Same-Day Outpatient Cardiology / Primary Care Follow-Up"
        else:
            esi_level = 5
            esi_label = "ESI-5 Low Acuity"
            disposition = "Routine Preventive Care & Risk Factor Counseling"

        soap_note, sbar_handoff, patient_summary = TriageAndSoapEngine._compose_documents(
            case_id=case_id,
            esi_label=esi_label,
            acuity_score=acuity_score,
            disposition=disposition,
            patient=patient,
            imaging=imaging,
            risk=risk,
            multi_scores=multi_scores,
            ddx=ddx,
            safety_report=safety_report,
            guideline_answers=guideline_answers or [],
        )

        return MultimodalTriageSoapResult(
            case_id=case_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            esi_level=esi_level,
            esi_label=esi_label,
            acuity_score=acuity_score,
            disposition=disposition,
            axes=axes,
            soap_note=soap_note,
            sbar_handoff=sbar_handoff,
            patient_summary=patient_summary,
        )

    @staticmethod
    def _compose_documents(
        case_id: str,
        esi_label: str,
        acuity_score: float,
        disposition: str,
        patient: Optional[PatientClinicalInput],
        imaging: Optional[ImagingAnalysisResult],
        risk: Optional[RiskAssessmentResult],
        multi_scores: Optional[MultiScoreSummary],
        ddx: Optional[DifferentialDiagnosisResult],
        safety_report,
        guideline_answers: List[GuidelineAnswer],
    ) -> tuple[str, str, str]:
        pt_line = (
            f"{patient.age} y/o {patient.sex} presenting with chest pain type {patient.chest_pain_type}"
            if patient
            else "Patient clinical profile pending"
        )
        vitals_line = (
            f"BP {patient.resting_bp} mmHg | Max HR {patient.max_hr} bpm | "
            f"Chol {patient.cholesterol} mg/dL | Fasting BS >120: {'Yes' if patient.fasting_bs else 'No'} | "
            f"ECG: {patient.resting_ecg} | Oldpeak: {patient.oldpeak} mm ({patient.st_slope}) | "
            f"Exertional Angina: {patient.exercise_angina}"
            if patient
            else "Hemodynamics not recorded"
        )
        img_line = (
            f"{imaging.top_finding} ({imaging.top_confidence * 100:.1f}% confidence, {imaging.model_name})"
            if imaging and imaging.inference_status == "SUCCESS"
            else "No chest radiograph analyzed"
        )
        risk_line = (
            f"{risk.risk_category} ({risk.risk_score * 100:.1f}% calibrated probability)"
            if risk
            else "Risk score not computed"
        )
        scores_line = (
            f"ASCVD 10-Yr: {multi_scores.ascvd.ten_year_risk_pct:.1f}% | "
            f"CURB-65: {multi_scores.curb65.score}/5 | "
            f"Wells PE: {multi_scores.wells_pe.score:.1f} | "
            f"CHA2DS2-VASc: {multi_scores.cha2ds2_vasc.score}/9"
            if multi_scores
            else "Multi-score panel pending"
        )
        primary_dx = (
            f"{ddx.primary_diagnosis.condition_name} ({ddx.primary_diagnosis.icd10_code}, "
            f"Est. {ddx.primary_diagnosis.probability_estimate_pct:.0f}%)"
            if ddx
            else "Pending multimodal synthesis"
        )
        workup_lines = (
            "\n".join(
                f"  - [{w.urgency}] {w.test_name}: {w.clinical_justification}"
                for w in ddx.recommended_workup
            )
            if ddx and ddx.recommended_workup
            else "  - Complete multimodal intake to generate targeted workup orders."
        )
        safety_lines = (
            "\n".join(
                f"  - [{a.severity}] {a.drug_class}: {a.recommended_alternative_action}"
                for a in safety_report.alerts
            )
            if safety_report and safety_report.alerts
            else "  - No acute pharmacotherapy contraindications triggered."
        )
        cite_lines = (
            "\n".join(
                f"  - {c.document_title} (p. {c.page_number}): {c.section_heading or 'Guideline Excerpt'}"
                for g in guideline_answers
                for c in g.citations[:2]
            )
            if guideline_answers
            else "  - Query Guidelines RAG to attach bibliographic citations."
        )

        soap_note = (
            f"CLINICAL COPILOT — STRUCTURED SOAP NOTE ({case_id})\n"
            f"Triage Acuity: {esi_label} (Acuity Index: {acuity_score}/100)\n"
            f"------------------------------------------------------------\n"
            f"S (Subjective):\n"
            f"  {pt_line}.\n\n"
            f"O (Objective):\n"
            f"  - Vitals & ECG: {vitals_line}\n"
            f"  - Chest Radiograph: {img_line}\n"
            f"  - ML Risk Stratification: {risk_line}\n"
            f"  - Validated Scores: {scores_line}\n\n"
            f"A (Assessment):\n"
            f"  - Primary Working Diagnosis: {primary_dx}\n"
            f"  - Disposition: {disposition}\n\n"
            f"P (Plan & Safety Guardrails):\n"
            f"  Recommended Workup:\n{workup_lines}\n"
            f"  Pharmacotherapy Safety:\n{safety_lines}\n"
            f"  Guideline References:\n{cite_lines}\n\n"
            f"[CDS Disclaimer: Decision-support draft only; requires licensed clinician verification.]"
        )

        sbar_handoff = (
            f"ED / INPATIENT SBAR HANDOFF — {case_id} ({esi_label})\n"
            f"------------------------------------------------------------\n"
            f"S (Situation): {pt_line}. Acuity Index {acuity_score}/100. Primary working Dx: {primary_dx}.\n"
            f"B (Background): {vitals_line}.\n"
            f"A (Assessment): CXR shows {img_line}; Calibrated CVD Risk is {risk_line}; {scores_line}.\n"
            f"R (Recommendation): {disposition}.\n"
            f"Key Orders:\n{workup_lines}"
        )

        dx_plain = ddx.primary_diagnosis.condition_name if ddx else "cardiopulmonary evaluation"
        patient_summary = (
            f"YOUR VISIT SUMMARY ({case_id})\n"
            f"------------------------------------------------------------\n"
            f"What We Checked:\n"
            f"We reviewed your vital signs, heart tracing (ECG), and chest X-ray together.\n\n"
            f"What It Shows:\n"
            f"- Your chest X-ray screening result: {img_line}.\n"
            f"- Your overall heart risk assessment: {risk_line}.\n"
            f"- Your care team is focusing on: {dx_plain}.\n\n"
            f"Recommended Next Steps:\n"
            f"- Care Priority: {esi_label} — {disposition}.\n"
            f"- Please discuss these results directly with your doctor. Seek immediate emergency care "
            f"if you develop worsening chest pain, shortness of breath, dizziness, or fainting."
        )

        return soap_note, sbar_handoff, patient_summary


triage_soap_engine = TriageAndSoapEngine()
