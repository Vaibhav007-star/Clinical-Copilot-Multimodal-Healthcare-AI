"""
Preprocessing pipeline for Patient Risk Assessment.
Handles feature encoding, validation, and serialization matching Kaggle Heart Failure dataset.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from modules.risk.schemas import PatientClinicalInput

FEATURE_COLUMNS = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

# Standard encodings matching clinical Kaggle dataset
SEX_MAP = {'F': 0, 'M': 1}
CHEST_PAIN_MAP = {'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3}
RESTING_ECG_MAP = {'Normal': 0, 'ST': 1, 'LVH': 2}
EXERCISE_ANGINA_MAP = {'N': 0, 'Y': 1}
ST_SLOPE_MAP = {'Up': 0, 'Flat': 1, 'Down': 2}

def patient_input_to_dataframe(patient: PatientClinicalInput) -> pd.DataFrame:
    """Converts a validated PatientClinicalInput Pydantic model into a 1-row DataFrame."""
    data = {
        'Age': [patient.age],
        'Sex': [patient.sex],
        'ChestPainType': [patient.chest_pain_type],
        'RestingBP': [patient.resting_bp],
        'Cholesterol': [patient.cholesterol],
        'FastingBS': [patient.fasting_bs],
        'RestingECG': [patient.resting_ecg],
        'MaxHR': [patient.max_hr],
        'ExerciseAngina': [patient.exercise_angina],
        'Oldpeak': [patient.oldpeak],
        'ST_Slope': [patient.st_slope]
    }
    return pd.DataFrame(data)

def preprocess_dataframe(df: pd.DataFrame, median_cholesterol: float = 237.0) -> np.ndarray:
    """
    Transforms raw clinical DataFrame into numerical matrix for model inference.
    Handles clinical zeroes in serum cholesterol by replacing with median.
    """
    df_clean = df.copy()
    
    # Handle zero cholesterol artifact common in legacy clinical records
    if 'Cholesterol' in df_clean.columns:
        df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, median_cholesterol)
        
    # Apply category mappings
    df_clean['Sex'] = df_clean['Sex'].map(lambda x: SEX_MAP.get(x, 1))
    df_clean['ChestPainType'] = df_clean['ChestPainType'].map(lambda x: CHEST_PAIN_MAP.get(x, 0))
    df_clean['RestingECG'] = df_clean['RestingECG'].map(lambda x: RESTING_ECG_MAP.get(x, 0))
    df_clean['ExerciseAngina'] = df_clean['ExerciseAngina'].map(lambda x: EXERCISE_ANGINA_MAP.get(x, 0))
    df_clean['ST_Slope'] = df_clean['ST_Slope'].map(lambda x: ST_SLOPE_MAP.get(x, 1))
    
    # Return numerical values in strict column order
    return df_clean[FEATURE_COLUMNS].values
