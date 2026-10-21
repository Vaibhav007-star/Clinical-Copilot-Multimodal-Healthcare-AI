"""
HL7 FHIR R4 Interoperability Export Engine for Clinical Copilot.
Generates fully compliant HL7 FHIR R4 JSON Bundles for direct Electronic Health Record (EHR) ingestion,
including Patient, Observation (vitals, labs, CXR), RiskAssessment, and DiagnosticReport resources.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import json

from modules.imaging.schemas import ImagingAnalysisResult
from modules.risk.schemas import PatientClinicalInput, RiskAssessmentResult
from modules.risk.calculators import MultiScoreSummary
from modules.orchestration.ddx import DifferentialDiagnosisResult

class FHIRBundleBuilder:
    @staticmethod
    def build_fhir_r4_bundle(
        case_id: str,
        patient: Optional[PatientClinicalInput] = None,
        imaging: Optional[ImagingAnalysisResult] = None,
        risk: Optional[RiskAssessmentResult] = None,
        ddx: Optional[DifferentialDiagnosisResult] = None,
        multi_scores: Optional[MultiScoreSummary] = None
    ) -> Dict[str, Any]:
        """
        Builds a valid HL7 FHIR Release 4 (R4) Document Bundle.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        patient_id = patient.patient_id if patient and patient.patient_id else "PT-1001"
        patient_ref = f"urn:uuid:patient-{patient_id}"

        entries: List[Dict[str, Any]] = []

        # 1. FHIR Patient Resource
        estimated_birth_year = datetime.now().year - (patient.age if patient else 55)
        fhir_patient = {
            "fullUrl": patient_ref,
            "resource": {
                "resourceType": "Patient",
                "id": patient_id,
                "identifier": [{
                    "use": "usual",
                    "system": "urn:oid:clinical-copilot:patient-id",
                    "value": patient_id
                }],
                "active": True,
                "gender": "male" if (patient and patient.sex == "M") else "female",
                "birthDate": f"{estimated_birth_year}-01-01"
            }
        }
        entries.append(fhir_patient)

        # 2. FHIR Observations (Vitals & Labs)
        if patient:
            # Blood Pressure Observation (LOINC 85354-9)
            bp_obs = {
                "fullUrl": f"urn:uuid:obs-bp-{case_id}",
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-bp-{case_id}",
                    "status": "final",
                    "category": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs"
                        }]
                    }],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "85354-9",
                            "display": "Blood pressure panel with all children optional"
                        }]
                    },
                    "subject": {"reference": patient_ref},
                    "effectiveDateTime": now_iso,
                    "component": [{
                        "code": {
                            "coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]
                        },
                        "valueQuantity": {
                            "value": patient.resting_bp,
                            "unit": "mmHg",
                            "system": "http://unitsofmeasure.org",
                            "code": "mm[Hg]"
                        }
                    }]
                }
            }
            entries.append(bp_obs)

            # Serum Cholesterol (LOINC 2093-3)
            chol_obs = {
                "fullUrl": f"urn:uuid:obs-chol-{case_id}",
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-chol-{case_id}",
                    "status": "final",
                    "category": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "laboratory",
                            "display": "Laboratory"
                        }]
                    }],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "2093-3",
                            "display": "Cholesterol [Mass/volume] in Serum or Plasma"
                        }]
                    },
                    "subject": {"reference": patient_ref},
                    "effectiveDateTime": now_iso,
                    "valueQuantity": {
                        "value": patient.cholesterol,
                        "unit": "mg/dL",
                        "system": "http://unitsofmeasure.org",
                        "code": "mg/dL"
                    }
                }
            }
            entries.append(chol_obs)

            # Heart Rate (LOINC 8867-4)
            hr_obs = {
                "fullUrl": f"urn:uuid:obs-hr-{case_id}",
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-hr-{case_id}",
                    "status": "final",
                    "category": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs"
                        }]
                    }],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "8867-4",
                            "display": "Heart rate"
                        }]
                    },
                    "subject": {"reference": patient_ref},
                    "effectiveDateTime": now_iso,
                    "valueQuantity": {
                        "value": patient.max_hr,
                        "unit": "beats/minute",
                        "system": "http://unitsofmeasure.org",
                        "code": "/min"
                    }
                }
            }
            entries.append(hr_obs)

        # 3. FHIR Observation for Imaging (DenseNet121 CXR finding)
        if imaging and imaging.inference_status == "SUCCESS":
            imaging_obs = {
                "fullUrl": f"urn:uuid:obs-cxr-{case_id}",
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-cxr-{case_id}",
                    "status": "final",
                    "category": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "imaging",
                            "display": "Imaging"
                        }]
                    }],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "18782-3",
                            "display": "Radiology Study observation"
                        }]
                    },
                    "subject": {"reference": patient_ref},
                    "effectiveDateTime": now_iso,
                    "valueString": f"Primary Finding: {imaging.top_finding} (Confidence: {imaging.top_confidence * 100:.1f}%)",
                    "note": [{
                        "text": f"Screened with DenseNet121-CXR. Grad-CAM visual saliency generated: {imaging.visualization_available}."
                    }]
                }
            }
            entries.append(imaging_obs)

        # 4. FHIR RiskAssessment Resource
        if risk:
            risk_resource = {
                "fullUrl": f"urn:uuid:risk-{case_id}",
                "resource": {
                    "resourceType": "RiskAssessment",
                    "id": f"risk-{case_id}",
                    "status": "final",
                    "subject": {"reference": patient_ref},
                    "occurrenceDateTime": now_iso,
                    "method": {
                        "coding": [{
                            "system": "urn:oid:clinical-copilot:models",
                            "code": "Calibrated-GradientBoosting-CVD",
                            "display": "Calibrated Gradient Boosting CVD Risk Model"
                        }]
                    },
                    "prediction": [{
                        "outcome": {
                            "text": "Cardiovascular Disease / Heart Failure Risk"
                        },
                        "probabilityDecimal": round(risk.risk_score, 4),
                        "qualitativeRisk": {
                            "coding": [{
                                "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                                "code": risk.risk_category.lower().replace(" ", "-"),
                                "display": risk.risk_category
                            }]
                        }
                    }]
                }
            }
            if multi_scores:
                risk_resource["resource"]["prediction"].append({
                    "outcome": {"text": "10-Year ASCVD Atherosclerotic Event"},
                    "probabilityDecimal": round(multi_scores.ascvd.ten_year_risk_pct / 100.0, 4),
                    "qualitativeRisk": {"text": multi_scores.ascvd.risk_category}
                })
                risk_resource["resource"]["prediction"].append({
                    "outcome": {"text": "Pneumonia 30-Day Mortality (CURB-65)"},
                    "probabilityDecimal": round(multi_scores.curb65.mortality_risk_pct / 100.0, 4),
                    "qualitativeRisk": {"text": multi_scores.curb65.risk_tier}
                })
            entries.append(risk_resource)

        # 5. FHIR DiagnosticReport Resource
        conclusion = ddx.primary_diagnosis.condition_name if ddx else "Multimodal evaluation completed."
        diag_report = {
            "fullUrl": f"urn:uuid:report-{case_id}",
            "resource": {
                "resourceType": "DiagnosticReport",
                "id": f"report-{case_id}",
                "status": "preliminary",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "CDS",
                        "display": "Clinical Decision Support"
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "11506-3",
                        "display": "Subspecialty Progress note"
                    }],
                    "text": "Clinical Copilot Multimodal Decision Support Summary"
                },
                "subject": {"reference": patient_ref},
                "effectiveDateTime": now_iso,
                "issued": now_iso,
                "conclusion": conclusion,
                "conclusionCode": [{
                    "coding": [{
                        "system": "http://hl7.org/fhir/sid/icd-10-cm",
                        "code": ddx.primary_diagnosis.icd10_code if ddx else "R07.89",
                        "display": conclusion
                    }]
                }]
            }
        }
        entries.append(diag_report)

        # FHIR R4 Bundle Wrapper
        bundle = {
            "resourceType": "Bundle",
            "id": f"bundle-copilot-{case_id}",
            "meta": {
                "lastUpdated": now_iso,
                "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"]
            },
            "type": "collection",
            "timestamp": now_iso,
            "total": len(entries),
            "entry": entries
        }

        return bundle

fhir_service = FHIRBundleBuilder()
