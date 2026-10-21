"""
Data schemas for Medical Imaging Module.
Follows clinical copilot Master Prompt specifications for structured outputs.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class FindingScore(BaseModel):
    finding: str = Field(..., description="Pathology or radiological pattern name (e.g. Cardiomegaly, Pneumonia)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model probability or calibrated confidence score [0.0 - 1.0]")
    severity_category: str = Field("Normal/Low", description="Risk/severity classification: Normal/Low, Indeterminate, High")
    clinical_note: Optional[str] = Field(None, description="Radiological description of this pattern")

class ImagingAnalysisResult(BaseModel):
    model_name: str = Field("DenseNet121-CXR", description="Underlying vision model identifier")
    model_version: str = Field("v1.2", description="Version of the model checkpoint")
    modality: str = Field("Chest X-Ray (CXR)", description="Medical imaging modality")
    task: str = Field("Multi-label Thoracic Abnormality Screening", description="Radiological analysis task")
    image_filename: str = Field(..., description="Uploaded file name")
    image_dimensions: tuple[int, int] = Field(..., description="Width and height of the processed image")
    inference_status: str = Field("SUCCESS", description="SUCCESS, WARNING, or FAILED")
    findings: List[FindingScore] = Field(default_factory=list, description="List of detected thoracic findings with scores")
    top_finding: Optional[str] = Field(None, description="Primary detected abnormality if any")
    top_confidence: float = Field(0.0, description="Confidence of the primary finding")
    visualization_available: bool = Field(False, description="Whether an anatomical Grad-CAM heatmap was generated")
    heatmap_path: Optional[str] = Field(None, description="Path to the generated Grad-CAM overlay image")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic uncertainty warnings or technical cautions")
    limitations: str = Field(
        "AI screening prototype for research & clinical decision support only. "
        "Not a substitute for official radiologist interpretation or emergency care.",
        description="Explicit clinical limitation disclaimer"
    )
    is_demonstration: bool = Field(False, description="Whether output was generated in fallback demo mode")
