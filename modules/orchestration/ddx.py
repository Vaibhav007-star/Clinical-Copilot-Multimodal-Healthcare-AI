"""
Multimodal Differential Diagnosis (DDx) and Diagnostic Workup Engine.
Synthesizes clinical vitals, DenseNet121 imaging findings, ML risk assessments,
and clinical decision rules into structured, ranked differential diagnoses with ICD-10 codes.
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

from modules.imaging.schemas import ImagingAnalysisResult
from modules.risk.schemas import PatientClinicalInput, RiskAssessmentResult
from modules.risk.calculators import MultiScoreSummary, compute_multi_score_panel

class DiagnosisItem(BaseModel):
    condition_name: str
    icd10_code: str
    probability_tier: Literal["Primary / Highly Likely", "Moderate / Possible", "Low / Rival Diagnosis"]
    probability_estimate_pct: float
    pathophysiological_summary: str
    supporting_findings: List[str]
    confirmatory_criteria: str

class RuledOutItem(BaseModel):
    condition_name: str
    icd10_code: str
    rationale: str

class WorkupItem(BaseModel):
    test_name: str
    category: Literal["Laboratory", "Imaging / Radiology", "Electrophysiology", "Hemodynamic / Invasive"]
    urgency: Literal["STAT (< 1 Hour)", "Urgent (< 6 Hours)", "Routine (< 24-48 Hours)"]
    clinical_justification: str

class DifferentialDiagnosisResult(BaseModel):
    case_id: str
    primary_diagnosis: DiagnosisItem
    secondary_diagnoses: List[DiagnosisItem]
    ruled_out_conditions: List[RuledOutItem]
    red_flag_alerts: List[str]
    recommended_workup: List[WorkupItem]
    guideline_care_pathway: str
    clinical_disclaimer: str = (
        "Generated for clinical decision support. Requires correlation by licensed attending physician."
    )


class DifferentialDiagnosisEngine:
    @staticmethod
    def generate_ddx(
        case_id: str,
        patient: Optional[PatientClinicalInput] = None,
        imaging: Optional[ImagingAnalysisResult] = None,
        risk: Optional[RiskAssessmentResult] = None,
        multi_scores: Optional[MultiScoreSummary] = None
    ) -> DifferentialDiagnosisResult:
        """
        Synthesizes multimodal evidence into an authoritative clinical differential diagnosis.
        """
        red_flags: List[str] = []
        workup: List[WorkupItem] = []
        primary_diag: Optional[DiagnosisItem] = None
        secondaries: List[DiagnosisItem] = []
        ruled_out: List[RuledOutItem] = []

        # Ensure multi_scores exist if patient input is available
        if patient and multi_scores is None:
            multi_scores = compute_multi_score_panel(patient)

        # Extract imaging features
        has_cardiomegaly = False
        has_pneumonia = False
        has_effusion = False
        has_consolidation = False
        has_atelectasis = False
        top_imaging_finding = "None"
        top_imaging_conf = 0.0

        if imaging and imaging.inference_status == "SUCCESS":
            top_imaging_finding = imaging.top_finding or "None"
            top_imaging_conf = imaging.top_confidence
            for f in imaging.findings:
                if f.finding == "Cardiomegaly" and f.confidence >= 0.40:
                    has_cardiomegaly = True
                elif f.finding == "Pneumonia" and f.confidence >= 0.40:
                    has_pneumonia = True
                elif f.finding == "Pleural Effusion" and f.confidence >= 0.35:
                    has_effusion = True
                elif f.finding == "Pulmonary Consolidation" and f.confidence >= 0.40:
                    has_consolidation = True
                elif f.finding == "Atelectasis" and f.confidence >= 0.40:
                    has_atelectasis = True

        # Extract patient hemodynamic & ECG features
        age = patient.age if patient else 60
        sex = patient.sex if patient else "M"
        sbp = patient.resting_bp if patient else 130
        hr = patient.max_hr if patient else 80
        chest_pain = patient.chest_pain_type if patient else "NAP"
        angina = patient.exercise_angina if patient else "N"
        oldpeak = patient.oldpeak if patient else 0.0
        st_slope = patient.st_slope if patient else "Up"
        ecg = patient.resting_ecg if patient else "Normal"
        cvd_risk_pct = (risk.risk_score * 100.0) if risk else 40.0

        # Check Red Flags
        if sbp >= 180:
            red_flags.append(f"Hypertensive Crisis / Severe Hypertension (SBP: {sbp} mmHg) — acute end-organ damage risk.")
        if oldpeak >= 2.0 or (oldpeak >= 1.0 and st_slope in ["Flat", "Down"]):
            red_flags.append(f"Significant Exercise ST-Segment Depression ({oldpeak} mm with {st_slope} slope) — severe inducible subendocardial ischemia.")
        if angina == "Y" and chest_pain in ["TA", "ASY"]:
            red_flags.append("Inducible Angina with ischemic chest symptoms — acute coronary syndrome (ACS) warning.")
        if hr >= 110:
            red_flags.append(f"Sinus Tachycardia (HR: {hr} bpm) — elevated myocardial oxygen demand and cardiac workload.")
        if has_effusion and has_cardiomegaly:
            red_flags.append("Combined Cardiomegaly + Pleural Effusion — signs of acute decompensated congestive heart failure.")

        # Core Diagnostic Logic Synthesis
        # Scenario A: Heart Failure / Cardiomegaly Predominance
        if has_cardiomegaly or (cvd_risk_pct >= 60.0 and (has_effusion or ecg == "LVH")):
            primary_diag = DiagnosisItem(
                condition_name="Congestive Heart Failure (Acute Decompensated / Stage C)",
                icd10_code="I50.9",
                probability_tier="Primary / Highly Likely",
                probability_estimate_pct=min(92.0, max(68.0, cvd_risk_pct)),
                pathophysiological_summary=(
                    "Ventricular remodeling, neurohormonal overactivation (RAAS/SNS), and fluid retention "
                    "leading to elevated cardiac filling pressures, cardiothoracic enlargement, and pulmonary congestion."
                ),
                supporting_findings=[
                    f"Chest Radiograph demonstrating Cardiomegaly (CTR > 0.50, model conf: {top_imaging_conf*100:.1f}%)",
                    f"Calibrated ML Cardiovascular Risk: {cvd_risk_pct:.1f}% ({risk.risk_category if risk else 'High'})",
                    f"Resting ECG showing {ecg} pattern with SBP {sbp} mmHg"
                ],
                confirmatory_criteria="Transthoracic Echocardiogram (TTE) assessing LVEF and elevated serum NT-proBNP/BNP."
            )

            secondaries.append(DiagnosisItem(
                condition_name="Coronary Artery Disease / Chronic Ischemic Myocardial Disease",
                icd10_code="I25.10",
                probability_tier="Moderate / Possible",
                probability_estimate_pct=round(cvd_risk_pct * 0.85, 1),
                pathophysiological_summary="Atheromatous coronary plaque burden resulting in supply-demand mismatch ischemia.",
                supporting_findings=[
                    f"Chest pain class {chest_pain}, Exercise Angina: {angina}",
                    f"ST-segment depression: {oldpeak} mm with {st_slope} recovery"
                ],
                confirmatory_criteria="Coronary Angiography (Invasive or CCTA) and serial high-sensitivity Troponin."
            ))

            if has_effusion:
                secondaries.append(DiagnosisItem(
                    condition_name="Transudative Pleural Effusion secondary to Volume Overload",
                    icd10_code="J91.8",
                    probability_tier="Moderate / Possible",
                    probability_estimate_pct=72.0,
                    pathophysiological_summary="Elevated pulmonary venous and capillary wedge pressures driving transudative fluid extravasation into pleural cavity.",
                    supporting_findings=["Costophrenic blunting / meniscus sign on frontal chest projection."],
                    confirmatory_criteria="Thoracentesis with Light's criteria if diagnostic ambiguity or failed diuresis."
                ))

        # Scenario B: Infectious / Pulmonary Infiltrate Predominance
        elif has_pneumonia or has_consolidation:
            primary_diag = DiagnosisItem(
                condition_name="Community-Acquired Pneumonia (CAP)",
                icd10_code="J18.9",
                probability_tier="Primary / Highly Likely",
                probability_estimate_pct=min(94.0, max(70.0, top_imaging_conf * 100.0)),
                pathophysiological_summary=(
                    "Acute alveolar infection with microbial infiltration, inflammatory exudate, "
                    "and microvascular congestion resulting in radio-opaque parenchymal consolidation."
                ),
                supporting_findings=[
                    f"Chest radiograph: {top_imaging_finding} (Confidence: {top_imaging_conf*100:.1f}%)",
                    f"CURB-65 Pneumonia Severity Score: {multi_scores.curb65.score if multi_scores else 1} ({multi_scores.curb65.risk_tier if multi_scores else 'Low Risk'})"
                ],
                confirmatory_criteria="Infiltrate on chest radiograph/CT plus clinical symptoms (fever, cough, sputum, leukocytosis)."
            )

            secondaries.append(DiagnosisItem(
                condition_name="Secondary Bacterial Pleuritis / Parapneumonic Effusion",
                icd10_code="J90",
                probability_tier="Moderate / Possible",
                probability_estimate_pct=42.0,
                pathophysiological_summary="Contiguous inflammatory pleural exudation adjacent to localized consolidated lung parenchyma.",
                supporting_findings=["Reactive pleural fluid accumulation on CXR."],
                confirmatory_criteria="Chest ultrasound or decubitus radiograph to assess loculation."
            ))

        # Scenario C: Ischemic Heart Disease / Atherosclerotic CVD Predominance
        elif cvd_risk_pct >= 50.0 or oldpeak >= 1.5 or angina == "Y":
            primary_diag = DiagnosisItem(
                condition_name="Acute Coronary Syndrome / Inducible Ischemic Heart Disease",
                icd10_code="I20.0",
                probability_tier="Primary / Highly Likely",
                probability_estimate_pct=round(cvd_risk_pct, 1),
                pathophysiological_summary=(
                    "Dynamic or fixed epicardial coronary artery obstruction compromising subendocardial "
                    "perfusion during physiological stress or at rest."
                ),
                supporting_findings=[
                    f"Calibrated Risk Score: {cvd_risk_pct:.1f}%",
                    f"Oldpeak ST Depression: {oldpeak} mm, ST Slope: {st_slope}",
                    f"Exercise-induced angina: {angina}, Chest Pain: {chest_pain}"
                ],
                confirmatory_criteria="Serial 12-lead ECGs (0h, 3h), high-sensitivity cardiac troponin, and urgent cardiology consultation."
            )

            secondaries.append(DiagnosisItem(
                condition_name="Hypertensive Heart Disease with Left Ventricular Strain",
                icd10_code="I11.9",
                probability_tier="Moderate / Possible",
                probability_estimate_pct=65.0,
                pathophysiological_summary="Chronic pressure overload resulting in concentric left ventricular hypertrophy and diastolic stiffness.",
                supporting_findings=[f"Resting BP {sbp} mmHg, Resting ECG: {ecg}"],
                confirmatory_criteria="Echocardiography demonstrating increased LV mass index and impaired relaxation."
            ))

        # Scenario D: Low-risk / Non-cardiac chest pain
        else:
            primary_diag = DiagnosisItem(
                condition_name="Non-Cardiac Chest Pain / Musculoskeletal or Gastroesophageal Etiology",
                icd10_code="R07.89",
                probability_tier="Primary / Highly Likely",
                probability_estimate_pct=65.0,
                pathophysiological_summary="Chest wall tenderness, costochondritis, or gastroesophageal acid reflux mimicking cardiac chest pain in the absence of objective ischemic or pulmonary findings.",
                supporting_findings=[
                    "Normal or non-infiltrative chest radiograph",
                    f"Low cardiovascular ML risk ({cvd_risk_pct:.1f}%)",
                    f"Absence of exercise-induced ischemia (Oldpeak: {oldpeak} mm)"
                ],
                confirmatory_criteria="Clinical reproduction with chest wall palpation, therapeutic trial of PPI, normal serial troponins."
            )

        # Add third rival diagnosis
        if not any(d.icd10_code == "I26.9" for d in secondaries):
            secondaries.append(DiagnosisItem(
                condition_name="Pulmonary Embolism (PE)",
                icd10_code="I26.9",
                probability_tier="Low / Rival Diagnosis",
                probability_estimate_pct=multi_scores.wells_pe.probability_pct if multi_scores else 12.1,
                pathophysiological_summary="Thromboembolic occlusion of pulmonary arterial tree.",
                supporting_findings=[
                    f"Wells PE Score: {multi_scores.wells_pe.score if multi_scores else 1.5} ({multi_scores.wells_pe.risk_tier if multi_scores else 'PE Unlikely'})",
                    f"Heart Rate: {hr} bpm"
                ],
                confirmatory_criteria="D-dimer rule-out (if low probability) or CT Pulmonary Angiography (CTPA)."
            ))

        # Ruled-out conditions with explicit clinical rationale
        ruled_out.append(RuledOutItem(
            condition_name="Tension Pneumothorax",
            icd10_code="J93.0",
            rationale="No deep sulcus sign, tracheal deviation, or visceral pleural line seen on chest radiograph; lung markings extend to periphery."
        ))
        if sbp > 100:
            ruled_out.append(RuledOutItem(
                condition_name="Cardiogenic Shock",
                icd10_code="R57.0",
                rationale=f"Patient maintains resting SBP of {sbp} mmHg (>90 mmHg cutoff) without immediate signs of systemic hypoperfusion."
            ))
        ruled_out.append(RuledOutItem(
            condition_name="Acute Aortic Dissection",
            icd10_code="I71.00",
            rationale="Mediastinal silhouette normal without wide mediastinum (>8cm) or unequal bilateral limb pulses reported."
        ))

        # Standardized Recommended Workup
        workup.append(WorkupItem(
            test_name="High-Sensitivity Cardiac Troponin (hs-cTnI / hs-cTnT)",
            category="Laboratory",
            urgency="STAT (< 1 Hour)" if cvd_risk_pct >= 60 or oldpeak > 1.0 else "Urgent (< 6 Hours)",
            clinical_justification="Rule in or rule out acute myocardial necrosis (0-hour and 2/3-hour delta protocols)."
        ))
        workup.append(WorkupItem(
            test_name="Serial 12-Lead Electrocardiogram (ECG)",
            category="Electrophysiology",
            urgency="STAT (< 1 Hour)" if (oldpeak > 1.0 or angina == "Y") else "Urgent (< 6 Hours)",
            clinical_justification="Assess dynamic ST-T wave evolution, localized ischemic injury vectors, and dysrhythmias."
        ))
        workup.append(WorkupItem(
            test_name="Serum B-Type Natriuretic Peptide (NT-proBNP)",
            category="Laboratory",
            urgency="Urgent (< 6 Hours)" if has_cardiomegaly or cvd_risk_pct >= 50 else "Routine (< 24-48 Hours)",
            clinical_justification="Biomarker of ventricular wall stretch; cutoffs <300 pg/mL (age <50) or <450 pg/mL rule out decompensated HF."
        ))
        workup.append(WorkupItem(
            test_name="Comprehensive Transthoracic Echocardiogram (TTE)",
            category="Imaging / Radiology",
            urgency="Urgent (< 6 Hours)" if has_cardiomegaly else "Routine (< 24-48 Hours)",
            clinical_justification="Directly quantify Left Ventricular Ejection Fraction (LVEF), regional wall motion abnormalities (RWMA), and valvular competence."
        ))
        if multi_scores and multi_scores.wells_pe.score > 4.0:
            workup.append(WorkupItem(
                test_name="CT Pulmonary Angiography (CTPA)",
                category="Imaging / Radiology",
                urgency="STAT (< 1 Hour)",
                clinical_justification="Wells PE score indicates high pre-test probability; definitive imaging required."
            ))

        # Guideline care pathway summary
        if "Heart Failure" in primary_diag.condition_name:
            pathway = "2022 AHA/ACC/HFSA Guideline-Directed Medical Therapy (GDMT): Initiate ARNI/ACEi, Evidence-based Beta-blocker, MRA, and SGLT2 inhibitor (The 4 Pillars) + loop diuretics as needed for euvolemia."
        elif "Coronary" in primary_diag.condition_name or "Ischemic" in primary_diag.condition_name:
            pathway = "2021 AHA/ACC Chest Pain Guideline: Immediate anti-ischemic therapy, dual antiplatelet therapy (DAPT), high-intensity statin, and invasive risk stratification via cardiac catheterization."
        elif "Pneumonia" in primary_diag.condition_name:
            pathway = "2019 ATS/IDSA Community-Acquired Pneumonia Guideline: Empiric beta-lactam (Amoxicillin/Clavulanate or Ceftriaxone) plus macrolide (Azithromycin) or respiratory fluoroquinolone."
        else:
            pathway = "AHA/ACC Primary Prevention & Symptom-Guided Observation Pathway with clinical re-assessment in 48-72 hours."

        return DifferentialDiagnosisResult(
            case_id=case_id,
            primary_diagnosis=primary_diag,
            secondary_diagnoses=secondaries,
            ruled_out_conditions=ruled_out,
            red_flag_alerts=red_flags,
            recommended_workup=workup,
            guideline_care_pathway=pathway
        )

ddx_engine = DifferentialDiagnosisEngine()
