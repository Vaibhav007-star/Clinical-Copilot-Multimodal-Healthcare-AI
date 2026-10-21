"""
Data schemas for Unified Clinical Case Review & Orchestration.
Maintains strict provenance tracking across imaging, patient risk, and guidelines.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from modules.imaging.schemas import ImagingAnalysisResult
from modules.risk.schemas import RiskAssessmentResult, PatientClinicalInput
from modules.rag.schemas import GuidelineAnswer

class ProvenanceItem(BaseModel):
    source_module: str = Field(..., description="Module A (Imaging), Module B (Risk), Module C (RAG), or User Input")
    element_name: str = Field(..., description="Field or finding name")
    value: str = Field(..., description="Reported value or observation")
    confidence_or_status: str = Field(..., description="Confidence percentage, risk score, or retrieval score")
    provenance_disclaimer: str = Field(..., description="Provenance boundary notice")

class UnifiedCaseSummary(BaseModel):
    case_id: str = Field(..., description="Case identifier")
    timestamp: str = Field(..., description="Creation timestamp")
    
    # Module States
    patient_input: Optional[PatientClinicalInput] = None
    imaging_result: Optional[ImagingAnalysisResult] = None
    risk_result: Optional[RiskAssessmentResult] = None
    guideline_answers: List[GuidelineAnswer] = Field(default_factory=list)
    
    # Synthesis & Audit
    provenance_table: List[ProvenanceItem] = Field(default_factory=list)
    clinical_synthesis_notes: List[str] = Field(default_factory=list)
    unresolved_uncertainties: List[str] = Field(default_factory=list)
    conflicts_or_warnings: List[str] = Field(default_factory=list)
    missing_data_notices: List[str] = Field(default_factory=list)
    
    # Advanced Synthesis Extensions
    differential_diagnosis: Optional[Any] = None
    multi_scores: Optional[Any] = None
    safety_review: Optional[Any] = None
    clinician_attestation: Optional[Dict[str, Any]] = None
    
    overall_status: str = Field("INCOMPLETE", description="COMPLETE, PARTIAL, or INCOMPLETE")
    governance_statement: str = Field(
        "Clinical Copilot Multimodal Review: AI findings from separate modalities are correlated for decision-support "
        "only and do not establish a definitive medical diagnosis. Clinical validation by an attending physician is mandatory.",
        description="Clinical governance and safety statement"
    )
