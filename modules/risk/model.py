"""
Machine Learning Risk Assessment Model trained on Kaggle Heart Failure dataset.
Provides calibrated probability estimation, feature attribution, and honest performance reporting.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, brier_score_loss
import joblib

from config.settings import settings
from modules.risk.schemas import (
    PatientClinicalInput,
    RiskAssessmentResult,
    FeatureContribution
)
from modules.risk.preprocessing import (
    FEATURE_COLUMNS,
    patient_input_to_dataframe,
    preprocess_dataframe
)

MODEL_PATH = settings.models_dir / "risk_model.joblib"
METRICS_PATH = settings.models_dir / "risk_metrics.json"

CLINICAL_FEATURE_DESCRIPTIONS = {
    'Age': 'Advanced age correlates with progressive arterial stiffening and atheroma burden.',
    'Sex': 'Male sex exhibits historically higher incidence of premature ischemic heart disease.',
    'ChestPainType': 'Asymptomatic (ASY) ischemia or typical angina (TA) significantly stratifies ischemic risk.',
    'RestingBP': 'Elevated resting systolic pressure increases left ventricular wall stress and cardiac workload.',
    'Cholesterol': 'Dyslipidemia accelerates vascular endothelial plaque formation and coronary stenosis.',
    'FastingBS': 'Hyperglycemia (>120 mg/dL) denotes impaired glucose metabolism and microvascular risk.',
    'RestingECG': 'Resting ST-T abnormality or left ventricular hypertrophy indicates cardiac strain.',
    'MaxHR': 'Chronotropic incompetence (inability to reach expected max heart rate) indicates cardiovascular vulnerability.',
    'ExerciseAngina': 'Exercise-induced angina strongly indicates inducible myocardial ischemia.',
    'Oldpeak': 'Exercise ST segment depression (>1.0 mm) is a key diagnostic indicator of myocardial ischemia.',
    'ST_Slope': 'Flat or downsloping ST segment post-exercise is strongly associated with severe coronary artery disease.'
}

class ClinicalRiskModel:
    def __init__(self):
        self.model: Optional[CalibratedClassifierCV] = None
        self.base_estimator: Optional[GradientBoostingClassifier] = None
        self.feature_importances: Dict[str, float] = {}
        self.evaluation_metrics: Dict[str, float] = {}
        self.median_cholesterol: float = 237.0
        self._initialize()

    def _initialize(self):
        if MODEL_PATH.exists() and METRICS_PATH.exists():
            try:
                bundle = joblib.load(MODEL_PATH)
                self.model = bundle['model']
                self.base_estimator = bundle['base_estimator']
                self.feature_importances = bundle['feature_importances']
                self.median_cholesterol = bundle.get('median_cholesterol', 237.0)
                with open(METRICS_PATH, 'r') as f:
                    self.evaluation_metrics = json.load(f)
                return
            except Exception as e:
                pass
        
        # Train on Kaggle dataset if bundle doesn't exist
        kaggle_csv = settings.datasets_dir / "heart_failure_kaggle.csv"
        if kaggle_csv.exists():
            self.train_and_evaluate(str(kaggle_csv))

    def train_and_evaluate(self, csv_path: str) -> Dict[str, float]:
        """Trains Calibrated Classifier on Kaggle heart failure dataset and evaluates performance."""
        df = pd.read_csv(csv_path)
        
        # Calculate clinical median for non-zero cholesterol
        valid_chol = df[df['Cholesterol'] > 0]['Cholesterol']
        self.median_cholesterol = float(valid_chol.median()) if not valid_chol.empty else 237.0
        
        X = preprocess_dataframe(df, self.median_cholesterol)
        y = df['HeartDisease'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Train base gradient boosting classifier
        base_gb = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )
        base_gb.fit(X_train, y_train)
        self.base_estimator = base_gb
        
        # Calibrate probabilities using isotonic / sigmoid calibration
        calibrated_model = CalibratedClassifierCV(estimator=base_gb, cv=5, method='sigmoid')
        calibrated_model.fit(X_train, y_train)
        self.model = calibrated_model
        
        # Compute test set evaluation metrics
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        auc = float(roc_auc_score(y_test, y_prob))
        acc = float(accuracy_score(y_test, y_pred))
        sens = float(recall_score(y_test, y_pred))
        brier = float(brier_score_loss(y_test, y_prob))
        
        # Calculate feature importances from base estimator
        raw_importances = base_gb.feature_importances_
        sum_imp = float(np.sum(raw_importances))
        self.feature_importances = {
            col: float(imp / sum_imp) for col, imp in zip(FEATURE_COLUMNS, raw_importances)
        }
        
        self.evaluation_metrics = {
            "test_roc_auc": round(auc, 4),
            "test_accuracy": round(acc, 4),
            "test_sensitivity": round(sens, 4),
            "brier_score": round(brier, 4),
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(X_test)),
            "dataset_origin": "Kaggle Heart Failure Prediction (fedesoriano)",
            "validation_method": "Stratified Train/Test (80/20) + 5-Fold Probability Calibration"
        }
        
        # Save model bundle & metrics
        joblib.dump({
            'model': self.model,
            'base_estimator': self.base_estimator,
            'feature_importances': self.feature_importances,
            'median_cholesterol': self.median_cholesterol
        }, MODEL_PATH)
        
        with open(METRICS_PATH, 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=2)
            
        return self.evaluation_metrics

    def predict(self, patient: PatientClinicalInput) -> RiskAssessmentResult:
        """Generates calibrated cardiovascular risk estimate and feature explanations for a patient."""
        if self.model is None:
            self._initialize()
            
        df_patient = patient_input_to_dataframe(patient)
        X_vec = preprocess_dataframe(df_patient, self.median_cholesterol)
        
        # Inference
        prob = float(self.model.predict_proba(X_vec)[0, 1])
        
        # Categorize
        if prob < 0.30:
            category = "Low Risk"
        elif prob <= 0.65:
            category = "Moderate Risk"
        else:
            category = "High Risk"
            
        # Compute feature contributions based on patient values and model weights
        contributions: List[FeatureContribution] = []
        for feature, base_weight in sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True):
            val = getattr(patient, feature.lower() if feature != 'ST_Slope' else 'st_slope', None)
            if val is None:
                # Handle special casing
                if feature == 'ChestPainType': val = patient.chest_pain_type
                elif feature == 'RestingBP': val = patient.resting_bp
                elif feature == 'FastingBS': val = patient.fasting_bs
                elif feature == 'RestingECG': val = patient.resting_ecg
                elif feature == 'MaxHR': val = patient.max_hr
                elif feature == 'ExerciseAngina': val = patient.exercise_angina
            
            # Determine direction
            direction = "Neutral"
            if feature == 'ST_Slope':
                direction = "Increases Risk" if val in ['Flat', 'Down'] else "Decreases Risk"
            elif feature == 'ExerciseAngina':
                direction = "Increases Risk" if val == 'Y' else "Decreases Risk"
            elif feature == 'Oldpeak':
                direction = "Increases Risk" if val > 1.0 else ("Decreases Risk" if val <= 0.2 else "Neutral")
            elif feature == 'ChestPainType':
                direction = "Increases Risk" if val == 'ASY' else "Decreases Risk"
            elif feature == 'MaxHR':
                direction = "Decreases Risk" if val > 140 else "Increases Risk"
            elif feature == 'Age':
                direction = "Increases Risk" if val >= 55 else "Decreases Risk"
            elif feature == 'FastingBS':
                direction = "Increases Risk" if val == 1 else "Decreases Risk"
            elif feature == 'Cholesterol':
                direction = "Increases Risk" if val > 240 else "Decreases Risk"
            elif feature == 'RestingBP':
                direction = "Increases Risk" if val >= 140 else "Decreases Risk"
            elif feature == 'Sex':
                direction = "Increases Risk" if val == 'M' else "Decreases Risk"
                
            contributions.append(FeatureContribution(
                feature_name=feature,
                feature_value=str(val),
                importance_weight=round(base_weight, 4),
                direction=direction,
                clinical_explanation=CLINICAL_FEATURE_DESCRIPTIONS.get(feature, "Clinical risk biomarker.")
            ))

        return RiskAssessmentResult(
            model_name="Calibrated-GradientBoosting-CVD",
            dataset_source="Kaggle fedesoriano/heart-failure-prediction (918 Patients)",
            outcome="Elevated Cardiovascular / Heart Disease Risk",
            prediction_horizon="Current Diagnostic & 1-Year Follow-up Horizon",
            risk_score=round(prob, 4),
            risk_category=category,
            contributing_factors=contributions,
            missing_features=[],
            is_demonstration=False
        )

# Global singleton risk model instance
risk_service = ClinicalRiskModel()
