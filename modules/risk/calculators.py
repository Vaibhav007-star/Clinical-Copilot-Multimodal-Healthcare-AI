"""
Evidence-Based Clinical Risk Calculators & Counterfactual Simulation Engine.
Implements:
1. Framingham / ACC-AHA 10-Year ASCVD Risk Score
2. CURB-65 Pneumonia Severity Score
3. Wells' Criteria for Pulmonary Embolism
4. CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk
5. Interactive Counterfactual "What-If" CVD Risk Simulator
"""
from typing import Dict, Any, List, Optional
import math
from pydantic import BaseModel, Field

from modules.risk.schemas import PatientClinicalInput
from modules.risk.model import risk_service

# ==========================================
# Schema Definitions
# ==========================================

class ASCVDResult(BaseModel):
    calculator_name: str = "Framingham / ACC-AHA 10-Year ASCVD Risk"
    ten_year_risk_pct: float = Field(..., description="Estimated 10-year risk of atherosclerotic CVD event")
    risk_category: str = Field(..., description="Low (<5%), Borderline (5-7.4%), Intermediate (7.5-19.9%), High (>=20%)")
    clinical_recommendation: str = Field(..., description="AHA/ACC primary prevention statin & lifestyle recommendation")
    parameters_used: Dict[str, Any] = Field(default_factory=dict)

class CURB65Result(BaseModel):
    calculator_name: str = "CURB-65 Pneumonia Severity Score"
    score: int = Field(..., ge=0, le=5, description="CURB-65 point score (0-5)")
    mortality_risk_pct: float = Field(..., description="Estimated 30-day in-hospital/30d mortality risk")
    risk_tier: str = Field(..., description="Low Risk, Intermediate Risk, High Risk")
    site_of_care_recommendation: str = Field(..., description="Outpatient vs Inpatient vs ICU recommendation")
    criteria_breakdown: Dict[str, bool] = Field(default_factory=dict)

class WellsPEResult(BaseModel):
    calculator_name: str = "Wells' Criteria for Pulmonary Embolism"
    score: float = Field(..., description="Total Wells PE score")
    risk_tier: str = Field(..., description="PE Unlikely (<=4.0) vs PE Likely (>4.0)")
    probability_pct: float = Field(..., description="Estimated clinical pre-test probability of PE")
    recommended_workup: str = Field(..., description="D-dimer vs Diagnostic CTPA recommendation")
    criteria_breakdown: Dict[str, float] = Field(default_factory=dict)

class CHA2DS2VAScResult(BaseModel):
    calculator_name: str = "CHA2DS2-VASc Atrial Fibrillation Stroke Risk"
    score: int = Field(..., ge=0, le=9, description="CHA2DS2-VASc score (0-9)")
    annual_stroke_risk_pct: float = Field(..., description="Adjusted annual stroke rate without anticoagulation")
    anticoagulation_recommendation: str = Field(..., description="ACC/AHA/ESC guideline recommendation on DOAC therapy")
    criteria_breakdown: Dict[str, int] = Field(default_factory=dict)

class MultiScoreSummary(BaseModel):
    ascvd: ASCVDResult
    curb65: CURB65Result
    wells_pe: WellsPEResult
    cha2ds2_vasc: CHA2DS2VAScResult

class CounterfactualSimulationRequest(BaseModel):
    baseline_patient: PatientClinicalInput
    target_resting_bp: Optional[int] = None
    target_cholesterol: Optional[int] = None
    target_max_hr: Optional[int] = None
    target_exercise_angina: Optional[str] = None
    target_fasting_bs: Optional[int] = None
    target_oldpeak: Optional[float] = None
    target_st_slope: Optional[str] = None

class ModifiableFactorGain(BaseModel):
    factor_name: str
    baseline_value: str
    target_value: str
    risk_reduction_pct: float
    clinical_note: str

class CounterfactualSimulationResult(BaseModel):
    baseline_risk_pct: float
    counterfactual_risk_pct: float
    absolute_risk_reduction_pct: float  # ARR
    relative_risk_reduction_pct: float  # RRR
    number_needed_to_treat: Optional[float]  # NNT
    interpretation: str
    top_interventions: List[ModifiableFactorGain]


# ==========================================
# Calculator Implementations
# ==========================================

def calculate_ascvd_risk(
    age: int,
    sex: str,
    total_chol: int,
    hdl_chol: int = 50,
    sbp: int = 130,
    treated_htn: bool = False,
    smoker: bool = False,
    diabetes: bool = False
) -> ASCVDResult:
    """
    Computes 10-Year ASCVD Risk using Framingham / ACC-AHA validated logarithmic regression equations.
    """
    is_male = (sex.upper() == "M")
    ln_age = math.log(max(20, min(79, age)))
    ln_tc = math.log(max(130, min(320, total_chol)))
    ln_hdl = math.log(max(20, min(100, hdl_chol)))
    ln_sbp = math.log(max(90, min(200, sbp)))

    if is_male:
        # Male coefficients (ACC/AHA Pooled Cohort Equations - Non-Hispanic White baseline)
        coeff_age = 12.344
        coeff_tc = 11.853
        coeff_age_tc = -2.664
        coeff_hdl = -7.990
        coeff_age_hdl = 1.769
        coeff_sbp = 1.797 if treated_htn else 1.764
        coeff_smoker = 7.837
        coeff_age_smoker = -1.795
        coeff_diab = 0.658
        mean_terms = 61.18
        baseline_survival = 0.9144

        indiv_sum = (
            coeff_age * ln_age
            + coeff_tc * ln_tc
            + coeff_age_tc * (ln_age * ln_tc)
            + coeff_hdl * ln_hdl
            + coeff_age_hdl * (ln_age * ln_hdl)
            + coeff_sbp * ln_sbp
            + (coeff_smoker * (1 if smoker else 0))
            + (coeff_age_smoker * (ln_age if smoker else 0))
            + (coeff_diab * (1 if diabetes else 0))
        )
    else:
        # Female coefficients
        coeff_age = -29.799
        coeff_age2 = 4.884
        coeff_tc = 13.540
        coeff_age_tc = -3.114
        coeff_hdl = -13.578
        coeff_age_hdl = 3.149
        coeff_sbp = 2.019 if treated_htn else 1.957
        coeff_smoker = 7.574
        coeff_age_smoker = -1.665
        coeff_diab = 0.661
        mean_terms = -29.18
        baseline_survival = 0.9665

        indiv_sum = (
            coeff_age * ln_age
            + coeff_age2 * (ln_age ** 2)
            + coeff_tc * ln_tc
            + coeff_age_tc * (ln_age * ln_tc)
            + coeff_hdl * ln_hdl
            + coeff_age_hdl * (ln_age * ln_hdl)
            + coeff_sbp * ln_sbp
            + (coeff_smoker * (1 if smoker else 0))
            + (coeff_age_smoker * (ln_age if smoker else 0))
            + (coeff_diab * (1 if diabetes else 0))
        )

    risk = 1.0 - (baseline_survival ** math.exp(indiv_sum - mean_terms))
    risk_pct = round(max(0.5, min(99.0, risk * 100.0)), 1)

    if risk_pct < 5.0:
        cat = "Low Risk (< 5%)"
        rec = "Emphasize healthy lifestyle behaviors (diet, exercise, smoking avoidance). Routine statin therapy not indicated."
    elif risk_pct < 7.5:
        cat = "Borderline Risk (5.0 - 7.4%)"
        rec = "Clinician-patient risk discussion. If risk enhancers present (family history, metabolic syndrome, CAC > 0), consider moderate-intensity statin."
    elif risk_pct < 20.0:
        cat = "Intermediate Risk (7.5 - 19.9%)"
        rec = "Initiate moderate-intensity statin therapy (e.g. Atorvastatin 20mg or Rosuvastatin 10mg) to reduce LDL-C by 30-49%."
    else:
        cat = "High Risk (>= 20.0%)"
        rec = "Initiate high-intensity statin therapy (e.g. Atorvastatin 40-80mg or Rosuvastatin 20-40mg) aiming for >=50% LDL-C reduction + strict BP control."

    return ASCVDResult(
        ten_year_risk_pct=risk_pct,
        risk_category=cat,
        clinical_recommendation=rec,
        parameters_used={
            "age": age,
            "sex": sex,
            "total_cholesterol": total_chol,
            "hdl_cholesterol": hdl_chol,
            "sbp": sbp,
            "treated_htn": treated_htn,
            "smoker": smoker,
            "diabetes": diabetes
        }
    )


def calculate_curb65(
    age: int,
    confusion: bool = False,
    bun_elevated: bool = False,
    resp_rate_high: bool = False,
    low_bp: bool = False
) -> CURB65Result:
    """
    Computes CURB-65 Pneumonia Severity Score.
    """
    age_ge_65 = (age >= 65)
    score = sum([confusion, bun_elevated, resp_rate_high, low_bp, age_ge_65])

    mortality_map = {
        0: 0.6,
        1: 2.7,
        2: 6.8,
        3: 14.0,
        4: 27.8,
        5: 35.0
    }
    mortality = mortality_map.get(score, 30.0)

    if score <= 1:
        tier = "Low Risk (Score 0-1)"
        site = "Outpatient treatment suitable in most cases. First-line oral empiric antimicrobial therapy."
    elif score == 2:
        tier = "Intermediate Risk (Score 2)"
        site = "Short inpatient hospitalization or supervised outpatient observation with 48h re-evaluation."
    else:
        tier = "High Risk (Score 3-5)"
        site = "Urgent hospital admission. Scores 4-5 strongly warrant ICU or high-dependency stepdown admission."

    return CURB65Result(
        score=score,
        mortality_risk_pct=mortality,
        risk_tier=tier,
        site_of_care_recommendation=site,
        criteria_breakdown={
            "Confusion (new mental impairment)": confusion,
            "Urea > 7 mmol/L (BUN > 19 mg/dL)": bun_elevated,
            "Respiratory rate >= 30 breaths/min": resp_rate_high,
            "Blood pressure (SBP <90 or DBP <=60)": low_bp,
            "Age >= 65 years": age_ge_65
        }
    )


def calculate_wells_pe(
    heart_rate: int,
    dvt_symptoms: bool = False,
    pe_most_likely: bool = False,
    immobilization_surgery: bool = False,
    prior_pe_dvt: bool = False,
    hemoptysis: bool = False,
    malignancy: bool = False
) -> WellsPEResult:
    """
    Computes Wells' Criteria for Pulmonary Embolism (PE).
    """
    hr_high = (heart_rate > 100)
    score = 0.0
    breakdown = {}

    if dvt_symptoms:
        score += 3.0
        breakdown["Clinical signs of DVT (leg swelling/pain)"] = 3.0
    if pe_most_likely:
        score += 3.0
        breakdown["PE is #1 or equally likely diagnosis"] = 3.0
    if hr_high:
        score += 1.5
        breakdown["Heart rate > 100 bpm"] = 1.5
    if immobilization_surgery:
        score += 1.5
        breakdown["Immobilization >=3d or surgery in past 4 weeks"] = 1.5
    if prior_pe_dvt:
        score += 1.5
        breakdown["Prior objectively confirmed DVT or PE"] = 1.5
    if hemoptysis:
        score += 1.0
        breakdown["Hemoptysis (coughing blood)"] = 1.0
    if malignancy:
        score += 1.0
        breakdown["Active malignancy / treatment within 6 months"] = 1.0

    score = round(score, 1)

    if score <= 4.0:
        tier = "PE Unlikely (Score <= 4.0)"
        prob = 12.1
        rec = "Consider High-Sensitivity D-dimer assay. If D-dimer < age-adjusted cutoff (<500 ug/L or Age x 10 in >50yo), PE is safely ruled out without imaging."
    else:
        tier = "PE Likely (Score > 4.0)"
        prob = 37.1
        rec = "Urgent diagnostic CT Pulmonary Angiography (CTPA) indicated. Consider empiric therapeutic anticoagulation while awaiting imaging if no contraindications."

    return WellsPEResult(
        score=score,
        risk_tier=tier,
        probability_pct=prob,
        recommended_workup=rec,
        criteria_breakdown=breakdown
    )


def calculate_cha2ds2_vasc(
    age: int,
    sex: str,
    chf_history: bool = False,
    hypertension: bool = False,
    diabetes: bool = False,
    stroke_tia_history: bool = False,
    vascular_disease: bool = False
) -> CHA2DS2VAScResult:
    """
    Computes CHA2DS2-VASc score for stroke risk in atrial fibrillation.
    """
    is_female = (sex.upper() == "F")
    score = 0
    breakdown = {}

    if chf_history:
        score += 1
        breakdown["Congestive Heart Failure / LVEF <=40%"] = 1
    if hypertension:
        score += 1
        breakdown["Hypertension history (BP > 140/90 or treated)"] = 1
    if age >= 75:
        score += 2
        breakdown["Age >= 75 years"] = 2
    elif 65 <= age <= 74:
        score += 1
        breakdown["Age 65-74 years"] = 1

    if diabetes:
        score += 1
        breakdown["Diabetes mellitus"] = 1
    if stroke_tia_history:
        score += 2
        breakdown["Prior Stroke, TIA, or Thromboembolism"] = 2
    if vascular_disease:
        score += 1
        breakdown["Vascular disease (prior MI, PAD, aortic plaque)"] = 1
    if is_female:
        score += 1
        breakdown["Sex category (Female)"] = 1

    stroke_rates = {
        0: 0.2, 1: 0.6, 2: 2.2, 3: 3.2, 4: 4.8, 5: 7.2, 6: 9.7, 7: 11.2, 8: 12.5, 9: 15.2
    }
    annual_rate = stroke_rates.get(score, 15.0)

    # Thresholds: >=2 in males, >=3 in females -> anticoagulation strongly recommended
    threshold = 3 if is_female else 2
    if score == (1 if is_female else 0):
        rec = "Low thromboembolic risk. Oral anticoagulation generally not recommended."
    elif score == (2 if is_female else 1):
        rec = "Intermediate thromboembolic risk. Oral anticoagulation with Direct Oral Anticoagulant (DOAC) should be considered after patient discussion."
    else:
        rec = f"High thromboembolic risk (Score >= {threshold}). Oral anticoagulation with DOAC (Apixaban, Rivaroxaban, Dabigatran) strongly recommended unless contraindicated."

    return CHA2DS2VAScResult(
        score=score,
        annual_stroke_risk_pct=annual_rate,
        anticoagulation_recommendation=rec,
        criteria_breakdown=breakdown
    )


def compute_multi_score_panel(patient: PatientClinicalInput) -> MultiScoreSummary:
    """Generates all 4 evidence-based clinical scores for a given patient."""
    is_htn = (patient.resting_bp >= 140)
    is_diabetic = (patient.fasting_bs == 1)
    is_tachycardic = (patient.max_hr > 100)
    has_lvh = (patient.resting_ecg == "LVH")
    has_ischemia = (patient.exercise_angina == "Y" or patient.oldpeak > 1.0)

    ascvd = calculate_ascvd_risk(
        age=patient.age,
        sex=patient.sex,
        total_chol=patient.cholesterol,
        hdl_chol=50,
        sbp=patient.resting_bp,
        treated_htn=is_htn,
        smoker=False,
        diabetes=is_diabetic
    )

    curb = calculate_curb65(
        age=patient.age,
        confusion=False,
        bun_elevated=False,
        resp_rate_high=False,
        low_bp=(patient.resting_bp < 90)
    )

    wells = calculate_wells_pe(
        heart_rate=patient.max_hr,
        dvt_symptoms=False,
        pe_most_likely=False,
        immobilization_surgery=False,
        prior_pe_dvt=False,
        hemoptysis=False,
        malignancy=False
    )

    cha = calculate_cha2ds2_vasc(
        age=patient.age,
        sex=patient.sex,
        chf_history=has_lvh,
        hypertension=is_htn,
        diabetes=is_diabetic,
        stroke_tia_history=False,
        vascular_disease=has_ischemia
    )

    return MultiScoreSummary(
        ascvd=ascvd,
        curb65=curb,
        wells_pe=wells,
        cha2ds2_vasc=cha
    )


def run_counterfactual_simulation(
    req: CounterfactualSimulationRequest
) -> CounterfactualSimulationResult:
    """
    Simulates counterfactual clinical adjustments against the ML Risk Assessment model.
    Calculates absolute and relative risk reductions and identifies high-yield interventions.
    """
    baseline_pt = req.baseline_patient
    baseline_result = risk_service.predict_risk(baseline_pt)
    baseline_risk = baseline_result.risk_score * 100.0

    # Build modified patient
    simulated_dict = baseline_pt.model_dump()
    if req.target_resting_bp is not None:
        simulated_dict['resting_bp'] = req.target_resting_bp
    if req.target_cholesterol is not None:
        simulated_dict['cholesterol'] = req.target_cholesterol
    if req.target_max_hr is not None:
        simulated_dict['max_hr'] = req.target_max_hr
    if req.target_exercise_angina is not None:
        simulated_dict['exercise_angina'] = req.target_exercise_angina
    if req.target_fasting_bs is not None:
        simulated_dict['fasting_bs'] = req.target_fasting_bs
    if req.target_oldpeak is not None:
        simulated_dict['oldpeak'] = req.target_oldpeak
    if req.target_st_slope is not None:
        simulated_dict['st_slope'] = req.target_st_slope

    simulated_pt = PatientClinicalInput(**simulated_dict)
    simulated_result = risk_service.predict_risk(simulated_pt)
    counterfactual_risk = simulated_result.risk_score * 100.0

    arr = round(baseline_risk - counterfactual_risk, 2)
    rrr = round((arr / baseline_risk * 100.0) if baseline_risk > 0 else 0.0, 1)
    nnt = round(100.0 / arr, 1) if arr >= 1.0 else None

    # Identify individual factor gains
    interventions: List[ModifiableFactorGain] = []

    # 1. BP intervention check
    if baseline_pt.resting_bp > 130:
        bp_test = baseline_pt.model_dump()
        bp_test['resting_bp'] = 120
        res = risk_service.predict_risk(PatientClinicalInput(**bp_test))
        gain = round(baseline_risk - (res.risk_score * 100.0), 2)
        interventions.append(ModifiableFactorGain(
            factor_name="Systolic Blood Pressure Optimization",
            baseline_value=f"{baseline_pt.resting_bp} mmHg",
            target_value="120 mmHg (Guideline Target)",
            risk_reduction_pct=max(0.0, gain),
            clinical_note="Lowering resting SBP to <130 mmHg reduces left ventricular hypertrophy progression and vascular shear stress."
        ))

    # 2. Cholesterol intervention check
    if baseline_pt.cholesterol > 200:
        chol_test = baseline_pt.model_dump()
        chol_test['cholesterol'] = 180
        res = risk_service.predict_risk(PatientClinicalInput(**chol_test))
        gain = round(baseline_risk - (res.risk_score * 100.0), 2)
        interventions.append(ModifiableFactorGain(
            factor_name="Lipid-Lowering Therapy (Statin / Ezetimibe)",
            baseline_value=f"{baseline_pt.cholesterol} mg/dL",
            target_value="180 mg/dL (or LDL-C < 70 mg/dL)",
            risk_reduction_pct=max(0.0, gain),
            clinical_note="Statin therapy stabilizes coronary atheromatous plaques and reduces major adverse cardiovascular events (MACE)."
        ))

    # 3. Glycemic control
    if baseline_pt.fasting_bs == 1:
        bs_test = baseline_pt.model_dump()
        bs_test['fasting_bs'] = 0
        res = risk_service.predict_risk(PatientClinicalInput(**bs_test))
        gain = round(baseline_risk - (res.risk_score * 100.0), 2)
        interventions.append(ModifiableFactorGain(
            factor_name="Strict Glycemic Control / SGLT2i or GLP-1 RA",
            baseline_value="FBS > 120 mg/dL",
            target_value="FBS <= 100 mg/dL",
            risk_reduction_pct=max(0.0, gain),
            clinical_note="Normalizing fasting glucose and adding cardioprotective agents (e.g. SGLT2 inhibitor) mitigates microvascular endothelial injury."
        ))

    # 4. Exercise tolerance
    if baseline_pt.max_hr < 140:
        hr_test = baseline_pt.model_dump()
        hr_test['max_hr'] = 150
        res = risk_service.predict_risk(PatientClinicalInput(**hr_test))
        gain = round(baseline_risk - (res.risk_score * 100.0), 2)
        interventions.append(ModifiableFactorGain(
            factor_name="Structured Cardiac Rehab & Aerobic Conditioning",
            baseline_value=f"Peak HR {baseline_pt.max_hr} bpm",
            target_value="Peak HR 150 bpm",
            risk_reduction_pct=max(0.0, gain),
            clinical_note="Improves autonomic chronotropic competence, enhances myocardial capillary density, and lowers resting sympathetic tone."
        ))

    interventions.sort(key=lambda x: x.risk_reduction_pct, reverse=True)

    if arr > 0:
        interpretation = (
            f"Achieving the proposed clinical targets lowers cardiovascular risk from "
            f"{baseline_risk:.1f}% to {counterfactual_risk:.1f}% (Absolute Risk Reduction: -{arr:.1f}%, "
            f"Relative Risk Reduction: -{rrr:.1f}%)."
        )
    elif arr < 0:
        interpretation = (
            f"Simulated deterioration increases estimated risk from {baseline_risk:.1f}% to "
            f"{counterfactual_risk:.1f}% (+{abs(arr):.1f}% absolute increase)."
        )
    else:
        interpretation = "Simulated scenario maintains equivalent cardiovascular risk baseline."

    return CounterfactualSimulationResult(
        baseline_risk_pct=round(baseline_risk, 1),
        counterfactual_risk_pct=round(counterfactual_risk, 1),
        absolute_risk_reduction_pct=arr,
        relative_risk_reduction_pct=rrr,
        number_needed_to_treat=nnt,
        interpretation=interpretation,
        top_interventions=interventions
    )
