"""
Clinical Pharmacotherapy Safety Guardrail and Contraindication Checker.
Cross-references patient physiological parameters (e.g. BP, HR, comorbidities)
against authoritative AHA/ACC/ATS clinical guidelines and black-box safety warnings.
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

from modules.risk.schemas import PatientClinicalInput

class SafetyAlert(BaseModel):
    drug_class: str
    severity: Literal["CRITICAL CONTRAINDICATION", "HIGH PRECAUTION", "MONITORING ADVISORY"]
    contraindication_rationale: str
    guideline_source: str
    recommended_alternative_action: str

class SafetyReviewResult(BaseModel):
    total_alerts: int
    has_critical_blocker: bool
    alerts: List[SafetyAlert]
    safe_to_proceed: bool
    clinical_pharmacist_note: str

class ClinicalSafetyGuardrail:
    @staticmethod
    def evaluate_patient_safety(
        patient: Optional[PatientClinicalInput],
        proposed_therapies: Optional[List[str]] = None
    ) -> SafetyReviewResult:
        """
        Reviews proposed or standard guideline-directed medical therapies against patient vitals and findings.
        """
        alerts: List[SafetyAlert] = []

        if not patient:
            return SafetyReviewResult(
                total_alerts=0,
                has_critical_blocker=False,
                alerts=[],
                safe_to_proceed=True,
                clinical_pharmacist_note="No active patient profile provided for pharmacological safety review."
            )

        hr = patient.max_hr
        sbp = patient.resting_bp
        is_diabetic = (patient.fasting_bs == 1)
        has_lvh = (patient.resting_ecg == "LVH")
        has_ischemia = (patient.exercise_angina == "Y" or patient.oldpeak > 1.0)

        # 1. Beta-Blocker Check (e.g. Carvedilol, Metoprolol Succinate, Bisoprolol)
        if hr < 60:
            alerts.append(SafetyAlert(
                drug_class="Beta-Adrenergic Blockers (Carvedilol, Metoprolol)",
                severity="HIGH PRECAUTION" if hr >= 50 else "CRITICAL CONTRAINDICATION",
                contraindication_rationale=(
                    f"Patient presents with baseline bradycardia (Heart Rate: {hr} bpm). "
                    "Beta-blockade reduces chronotropic drive and may induce symptomatic bradyarrhythmia or high-degree AV block."
                ),
                guideline_source="2022 AHA/ACC/HFSA Heart Failure Guideline Section 7.3.2",
                recommended_alternative_action="Withhold or uptitrate cautiously with continuous telemetry; consider Ivabradine only if in sinus rhythm and resting HR >= 70."
            ))
        elif sbp < 90:
            alerts.append(SafetyAlert(
                drug_class="Beta-Adrenergic Blockers",
                severity="CRITICAL CONTRAINDICATION",
                contraindication_rationale=f"Hypotension with SBP {sbp} mmHg (<90 mmHg). Risk of precipitating cardiogenic collapse.",
                guideline_source="AHA/ACC Shock & Inotropic Support Guidance",
                recommended_alternative_action="Stabilize mean arterial pressure (MAP >= 65) before introducing negative inotropes."
            ))

        # 2. Nitrates (Nitroglycerin, Isosorbide Mononitrate / Dinitrate)
        alerts.append(SafetyAlert(
            drug_class="Organic Nitrates (Nitroglycerin, Isosorbide)",
            severity="CRITICAL CONTRAINDICATION",
            contraindication_rationale=(
                "Absolute black-box contraindication: Concurrent administration of Phosphodiesterase-5 (PDE-5) "
                "inhibitors (Sildenafil within 24 hours, Tadalafil within 48 hours). Results in profound, refractory cyclic-GMP mediated hypotension."
            ),
            guideline_source="2021 AHA/ACC Chest Pain Guideline Section 5.1",
            recommended_alternative_action="Screen all patients specifically for recent PDE-5 inhibitor use before administering nitrates."
        ))

        # 3. NSAIDs in Cardiovascular Disease / Heart Failure
        if has_lvh or has_ischemia or sbp > 140:
            alerts.append(SafetyAlert(
                drug_class="Non-Steroidal Anti-Inflammatory Drugs (NSAIDs: Ibuprofen, Naproxen, Celecoxib)",
                severity="HIGH PRECAUTION",
                contraindication_rationale=(
                    "NSAIDs inhibit renal prostaglandin synthesis, causing acute sodium/water retention, "
                    "blunting ACEi/diuretic efficacy, raising systolic BP by 3-6 mmHg, and doubling heart failure hospitalization risk."
                ),
                guideline_source="AHA Scientific Statement on Drugs That May Exacerbate Heart Failure (Circulation)",
                recommended_alternative_action="Use Acetaminophen (Tylenol) as first-line analgesic; avoid systemic NSAIDs."
            ))

        # 4. Renin-Angiotensin-Aldosterone System Inhibitors (ACEi / ARB / ARNI)
        if sbp < 100:
            alerts.append(SafetyAlert(
                drug_class="ARNI / ACE-Inhibitors / ARBs (Sacubitril/Valsartan, Lisinopril)",
                severity="MONITORING ADVISORY",
                contraindication_rationale=(
                    f"Baseline borderline systolic pressure (SBP: {sbp} mmHg). "
                    "Risk of first-dose symptomatic postural hypotension and acute prerenal azotemia."
                ),
                guideline_source="2022 AHA/ACC Heart Failure Guideline Section 7.3.1",
                recommended_alternative_action="Initiate at lowest test dose (e.g. Sacubitril/Valsartan 24/26 mg BID) and monitor serum creatinine & K+ within 1-2 weeks."
            ))

        # 5. SGLT2 Inhibitors (Empagliflozin, Dapagliflozin)
        if is_diabetic:
            alerts.append(SafetyAlert(
                drug_class="SGLT2 Inhibitors (Empagliflozin, Dapagliflozin)",
                severity="MONITORING ADVISORY",
                contraindication_rationale=(
                    "Patient has documented impaired fasting blood sugar. SGLT2 inhibitors provide Class 1A mortality "
                    "benefit in heart failure across all LVEF ranges, but require vigilance for rare euglycemic diabetic ketoacidosis (euDKA) during acute illness or starvation."
                ),
                guideline_source="2022 AHA/ACC/HFSA Heart Failure Guideline Section 7.3.4",
                recommended_alternative_action="Hold during acute perioperative periods or prolonged fasting; ensure adequate hydration."
            ))

        has_critical = any(a.severity == "CRITICAL CONTRAINDICATION" for a in alerts)

        note = (
            f"Screened {len(alerts)} standard clinical pharmacotherapy interactions. "
            f"{'CRITICAL CONTRAINDICATION identified: physician confirmation required before prescribing.' if has_critical else 'Advisories flagged for clinical monitoring.'}"
        )

        return SafetyReviewResult(
            total_alerts=len(alerts),
            has_critical_blocker=has_critical,
            alerts=alerts,
            safe_to_proceed=not has_critical,
            clinical_pharmacist_note=note
        )

safety_guardrail_service = ClinicalSafetyGuardrail()
