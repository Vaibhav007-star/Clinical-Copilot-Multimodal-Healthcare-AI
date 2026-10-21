"""
Automated Pytest Suite for Advanced Clinical Copilot Capabilities:
1. Multi-Score Clinical Calculators (Framingham ASCVD, CURB-65, Wells PE, CHA2DS2-VASc)
2. Interactive Counterfactual What-If Risk Simulator
3. Pharmacotherapy Safety Guardrail & Contraindication Checker
4. Multimodal Differential Diagnosis (DDx) Engine
5. HL7 FHIR R4 Interoperability Bundle Builder
"""
import pytest
from modules.risk.schemas import PatientClinicalInput
from modules.risk.calculators import (
    calculate_ascvd_risk,
    calculate_curb65,
    calculate_wells_pe,
    calculate_cha2ds2_vasc,
    compute_multi_score_panel,
    run_counterfactual_simulation,
    CounterfactualSimulationRequest
)
from modules.rag.safety import safety_guardrail_service
from modules.orchestration.ddx import ddx_engine
from modules.orchestration.fhir import fhir_service
from modules.imaging.schemas import ImagingAnalysisResult, FindingScore


@pytest.fixture
def cardiac_patient():
    return PatientClinicalInput(
        patient_id="PT-TEST-01",
        age=65,
        sex="M",
        chest_pain_type="ASY",
        resting_bp=160,
        cholesterol=270,
        fasting_bs=1,
        resting_ecg="LVH",
        max_hr=120,
        exercise_angina="Y",
        oldpeak=2.5,
        st_slope="Flat"
    )


@pytest.fixture
def mock_imaging_cardiomegaly():
    return ImagingAnalysisResult(
        model_name="DenseNet121-CXR",
        model_version="v1.2",
        modality="Chest X-Ray (CXR)",
        task="Multi-label Thoracic Abnormality Screening",
        image_filename="sample_cxr_cardiomegaly.png",
        image_dimensions=(224, 224),
        inference_status="SUCCESS",
        findings=[
            FindingScore(finding="Cardiomegaly", confidence=0.88, severity_category="High"),
            FindingScore(finding="Pleural Effusion", confidence=0.55, severity_category="High"),
            FindingScore(finding="Normal / No Infiltrate", confidence=0.12, severity_category="Normal/Low")
        ],
        top_finding="Cardiomegaly",
        top_confidence=0.88,
        visualization_available=True,
        available_heatmaps={"Cardiomegaly": "path/to/cam.png"}
    )


def test_ascvd_calculator(cardiac_patient):
    res = calculate_ascvd_risk(
        age=cardiac_patient.age,
        sex=cardiac_patient.sex,
        total_chol=cardiac_patient.cholesterol,
        hdl_chol=45,
        sbp=cardiac_patient.resting_bp,
        treated_htn=True,
        diabetes=True
    )
    assert res.ten_year_risk_pct > 15.0
    assert "Risk" in res.risk_category
    assert "statin" in res.clinical_recommendation.lower()


def test_curb65_calculator():
    # Low risk patient
    res_low = calculate_curb65(age=45, confusion=False, bun_elevated=False, resp_rate_high=False, low_bp=False)
    assert res_low.score == 0
    assert "Outpatient" in res_low.site_of_care_recommendation

    # High risk elderly patient
    res_high = calculate_curb65(age=72, confusion=True, bun_elevated=True, resp_rate_high=True, low_bp=True)
    assert res_high.score == 5
    assert "High Risk" in res_high.risk_tier
    assert res_high.mortality_risk_pct > 20.0


def test_wells_pe_calculator():
    res_unlikely = calculate_wells_pe(heart_rate=75)
    assert res_unlikely.score <= 4.0
    assert "Unlikely" in res_unlikely.risk_tier
    assert "D-dimer" in res_unlikely.recommended_workup

    res_likely = calculate_wells_pe(heart_rate=115, dvt_symptoms=True, pe_most_likely=True)
    assert res_likely.score > 4.0
    assert "Likely" in res_likely.risk_tier
    assert "CTPA" in res_likely.recommended_workup or "CT" in res_likely.recommended_workup


def test_cha2ds2_vasc_calculator(cardiac_patient):
    res = calculate_cha2ds2_vasc(
        age=cardiac_patient.age,
        sex=cardiac_patient.sex,
        chf_history=True,
        hypertension=True,
        diabetes=True,
        vascular_disease=True
    )
    assert res.score >= 4
    assert res.annual_stroke_risk_pct > 3.0
    assert "anticoagulation" in res.anticoagulation_recommendation.lower()


def test_multi_score_panel(cardiac_patient):
    panel = compute_multi_score_panel(cardiac_patient)
    assert panel.ascvd.ten_year_risk_pct > 0.0
    assert panel.curb65.score >= 0
    assert panel.wells_pe.score >= 0.0
    assert panel.cha2ds2_vasc.score >= 0


def test_counterfactual_simulation(cardiac_patient):
    req = CounterfactualSimulationRequest(
        baseline_patient=cardiac_patient,
        target_resting_bp=120,
        target_cholesterol=180,
        target_max_hr=150,
        target_fasting_bs=0,
        target_exercise_angina="N"
    )
    sim = run_counterfactual_simulation(req)
    assert sim.baseline_risk_pct > 0
    assert sim.counterfactual_risk_pct < sim.baseline_risk_pct
    assert sim.absolute_risk_reduction_pct > 0
    assert sim.relative_risk_reduction_pct > 0
    assert len(sim.top_interventions) > 0


def test_pharmacotherapy_safety(cardiac_patient):
    # Test bradycardia safety trigger
    brady_pt = cardiac_patient.model_copy(update={"max_hr": 48})
    safety_res = safety_guardrail_service.evaluate_patient_safety(brady_pt)
    assert safety_res.total_alerts > 0
    assert any("Beta-Adrenergic Blockers" in a.drug_class for a in safety_res.alerts)
    assert any(a.severity == "CRITICAL CONTRAINDICATION" for a in safety_res.alerts)


def test_differential_diagnosis_engine(cardiac_patient, mock_imaging_cardiomegaly):
    ddx = ddx_engine.generate_ddx(
        case_id="TEST-CASE-01",
        patient=cardiac_patient,
        imaging=mock_imaging_cardiomegaly
    )
    assert ddx.case_id == "TEST-CASE-01"
    assert "Heart Failure" in ddx.primary_diagnosis.condition_name
    assert ddx.primary_diagnosis.icd10_code == "I50.9"
    assert len(ddx.secondary_diagnoses) > 0
    assert len(ddx.ruled_out_conditions) > 0
    assert len(ddx.recommended_workup) > 0
    assert any("Troponin" in w.test_name for w in ddx.recommended_workup)


def test_fhir_r4_bundle_builder(cardiac_patient, mock_imaging_cardiomegaly):
    ddx = ddx_engine.generate_ddx(
        case_id="TEST-CASE-01",
        patient=cardiac_patient,
        imaging=mock_imaging_cardiomegaly
    )
    bundle = fhir_service.build_fhir_r4_bundle(
        case_id="TEST-CASE-01",
        patient=cardiac_patient,
        imaging=mock_imaging_cardiomegaly,
        ddx=ddx
    )
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["total"] >= 4

    resource_types = [entry["resource"]["resourceType"] for entry in bundle["entry"]]
    assert "Patient" in resource_types
    assert "Observation" in resource_types
    assert "DiagnosticReport" in resource_types
