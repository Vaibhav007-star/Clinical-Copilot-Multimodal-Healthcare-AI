"""
Integration tests for Unified Clinical Case Review Orchestration.
Verifies multimodal correlation, provenance tracking, and graceful handling of missing modules.
"""
from PIL import Image
import pytest

from modules.imaging.inference import imaging_service
from modules.risk.schemas import PatientClinicalInput
from modules.risk.model import risk_service
from modules.rag.generation import guideline_service
from modules.orchestration.clinical_review import orchestrator_service

def test_unified_orchestration_complete_flow():
    # 1. Module A: Imaging
    img = Image.new("RGB", (224, 224), color="gray")
    img_res = imaging_service.analyze(img, "test_cxr.png")
    
    # 2. Module B: Risk
    patient = PatientClinicalInput(
        patient_id="PT-INT-01",
        age=58,
        sex="M",
        chest_pain_type="ASY",
        resting_bp=145,
        cholesterol=260,
        fasting_bs=1,
        max_hr=125,
        exercise_angina="Y",
        oldpeak=1.8,
        resting_ecg="Normal",
        st_slope="Flat"
    )
    risk_res = risk_service.predict(patient)

    # 3. Module C: RAG
    rag_ans = guideline_service.answer_query("When is chest radiography recommended in chest pain?")

    # 4. Orchestration
    summary = orchestrator_service.build_case_summary(
        case_id="CASE-INT-01",
        patient_input=patient,
        imaging_result=img_res,
        risk_result=risk_res,
        guideline_answers=[rag_ans]
    )

    assert summary.overall_status == "COMPLETE"
    assert len(summary.provenance_table) >= 3
    # Check that each module origin is explicitly documented
    sources = [p.source_module for p in summary.provenance_table]
    assert any("Imaging" in s for s in sources)
    assert any("Risk" in s for s in sources)
    assert any("Patient Input" in s for s in sources)

def test_unified_orchestration_partial_flow():
    # Only Patient input, no imaging or risk yet
    patient = PatientClinicalInput(
        patient_id="PT-INT-02",
        age=40,
        sex="F",
        chest_pain_type="ATA",
        resting_bp=118,
        cholesterol=190,
        fasting_bs=0,
        max_hr=165,
        exercise_angina="N",
        oldpeak=0.0,
        resting_ecg="Normal",
        st_slope="Up"
    )
    summary = orchestrator_service.build_case_summary(
        case_id="CASE-INT-02",
        patient_input=patient,
        imaging_result=None,
        risk_result=None,
        guideline_answers=[]
    )
    assert summary.overall_status == "PARTIAL"
    assert len(summary.missing_data_notices) >= 2
