"""
Data schemas for Patient Risk Assessment Module.
Directly aligned with real clinical features from Kaggle Heart Failure & CVD Prediction dataset.
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

class PatientClinicalInput(BaseModel):
    # Patient Demographics
    patient_id: Optional[str] = Field("PT-1001", description="De-identified patient ID")
    age: int = Field(..., ge=18, le=110, description="Age in years (18-110)")
    sex: Literal["M", "F"] = Field(..., description="Biological sex (M: Male, F: Female)")
    
    # Symptoms & History
    chest_pain_type: Literal["ASY", "NAP", "ATA", "TA"] = Field(
        ...,
        description="Chest Pain Type (ASY: Asymptomatic, NAP: Non-Anginal Pain, ATA: Atypical Angina, TA: Typical Angina)"
    )
    exercise_angina: Literal["Y", "N"] = Field(..., description="Exercise-induced angina (Y: Yes, N: No)")
    
    # Hemodynamics & Lab Measurements
    resting_bp: int = Field(..., ge=70, le=240, description="Resting Blood Pressure in mm Hg (70-240)")
    cholesterol: int = Field(..., ge=80, le=600, description="Serum Cholesterol in mg/dL (80-600)")
    fasting_bs: Literal[0, 1] = Field(..., description="Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)")
    max_hr: int = Field(..., ge=30, le=220, description="Maximum Heart Rate achieved (30-220 bpm)")
    
    # Electrocardiography
    resting_ecg: Literal["Normal", "ST", "LVH"] = Field(
        ...,
        description="Resting ECG findings (Normal, ST: ST-T wave abnormality, LVH: Left Ventricular Hypertrophy)"
    )
    oldpeak: float = Field(..., ge=-3.0, le=7.0, description="ST depression induced by exercise relative to rest (mm)")
    st_slope: Literal["Up", "Flat", "Down"] = Field(
        ...,
        description="Slope of the peak exercise ST segment (Up: Upsloping, Flat: Flat, Down: Downsloping)"
    )

class FeatureContribution(BaseModel):
    feature_name: str = Field(..., description="Clinical attribute name")
    feature_value: str = Field(..., description="Observed patient value")
    importance_weight: float = Field(..., description="Normalized contribution or SHAP/feature weight")
    direction: Literal["Increases Risk", "Decreases Risk", "Neutral"] = Field(..., description="Directional impact on risk")
    clinical_explanation: str = Field(..., description="Clinical rationale behind this risk factor")

class RiskAssessmentResult(BaseModel):
    model_name: str = Field("Calibrated-GradientBoosting-CVD", description="Risk model name")
    dataset_source: str = Field("Kaggle fedesoriano/heart-failure-prediction (918 Patients)", description="Training data source")
    outcome: str = Field("Elevated Cardiovascular / Heart Disease Risk", description="Target clinical outcome")
    prediction_horizon: str = Field("Current Diagnostic / 1-Year Follow-up Horizon", description="Temporal prediction window")
    
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated risk probability [0.0 - 1.0]")
    risk_category: Literal["Low Risk", "Moderate Risk", "High Risk"] = Field(..., description="Categorized clinical risk level")
    contributing_factors: List[FeatureContribution] = Field(default_factory=list, description="Top positive and protective factors")
    
    missing_features: List[str] = Field(default_factory=list, description="Missing required clinical features if any")
    uncertainty_note: str = Field(
        "Risk estimation is derived from observational cohorts. Model explanations represent correlational feature importances, not clinical causation.",
        description="Uncertainty and causality disclaimer"
    )
    limitations: str = Field(
        "Trained on adult cardiology cohort with 918 observations across 5 centers. Not validated for acute myocardial infarction emergency triage or pediatric patients.",
        description="Validated boundaries and limitations"
    )
    is_demonstration: bool = Field(False, description="Whether model ran in fallback demo mode")
