"""
Unit tests for Patient Risk Assessment Module.
Tests schema validation, boundary conditions, inference calibration, and feature explanations.
"""
import pytest
from pydantic import ValidationError

from modules.risk.schemas import PatientClinicalInput, RiskAssessmentResult
from modules.risk.model import risk_service

def test_patient_input_validation_valid():
    p = PatientClinicalInput(
        patient_id="PT-TEST-01",
        age=52,
        sex="M",
        chest_pain_type="ASY",
        resting_bp=130,
        cholesterol=240,
        fasting_bs=0,
        max_hr=150,
        exercise_angina="Y",
        oldpeak=1.5,
        resting_ecg="Normal",
        st_slope="Flat"
    )
    assert p.age == 52
    assert p.sex == "M"

def test_patient_input_validation_invalid_bounds():
    with pytest.raises(ValidationError):
        # Age outside valid clinical range
        PatientClinicalInput(
            age=12,  # min is 18
            sex="M",
            chest_pain_type="ASY",
            resting_bp=130,
            cholesterol=240,
            fasting_bs=0,
            max_hr=150,
            exercise_angina="Y",
            oldpeak=1.5,
            resting_ecg="Normal",
            st_slope="Flat"
        )

def test_risk_prediction_output():
    patient = PatientClinicalInput(
        age=65,
        sex="M",
        chest_pain_type="ASY",
        resting_bp=160,
        cholesterol=290,
        fasting_bs=1,
        max_hr=110,
        exercise_angina="Y",
        oldpeak=2.5,
        resting_ecg="ST",
        st_slope="Flat"
    )
    res = risk_service.predict(patient)
    assert isinstance(res, RiskAssessmentResult)
    assert 0.0 <= res.risk_score <= 1.0
    assert res.risk_category in ["Low Risk", "Moderate Risk", "High Risk"]
    assert len(res.contributing_factors) > 0
    assert "Kaggle" in res.dataset_source
