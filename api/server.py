"""
FastAPI Server for Clinical Copilot Web Application.
Serves multimodal AI endpoints (DenseNet121 CXR, Kaggle Risk Model, FAISS Guidelines RAG, Orchestrator)
and serves the rich web application interface.
"""
from pathlib import Path
import base64
import io
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from config.settings import settings
from modules.imaging.schemas import ImagingAnalysisResult
from modules.imaging.preprocessing import validate_image_file
from modules.imaging.inference import imaging_service
from modules.risk.schemas import PatientClinicalInput, RiskAssessmentResult
from modules.risk.model import risk_service
from modules.rag.schemas import GuidelineAnswer
from modules.rag.generation import guideline_service
from modules.rag.retrieval import retrieval_service
from modules.orchestration.schemas import UnifiedCaseSummary
from modules.orchestration.clinical_review import orchestrator_service

app = FastAPI(
    title="Clinical Copilot Web API",
    description="Backend services for Multimodal Clinical Decision Support Web Application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    document_filter: Optional[str] = None

class ReviewPayload(BaseModel):
    case_id: str = "CASE-WEB-DEMO"
    patient_input: Optional[PatientClinicalInput] = None
    imaging_result: Optional[ImagingAnalysisResult] = None
    risk_result: Optional[RiskAssessmentResult] = None
    guideline_answers: List[GuidelineAnswer] = []

@app.get("/api/status")
def get_system_status():
    """Returns real-time status of backend models, data provenance, and environment."""
    return {
        "status": "ONLINE",
        "app_name": "Clinical Copilot",
        "version": settings.app_version,
        "imaging_model": {
            "name": "DenseNet121-CXR",
            "task": "Multi-label Thoracic Screening (7 Pathologies)",
            "saliency": "Grad-CAM Enabled"
        },
        "risk_model": {
            "name": "Calibrated-GradientBoosting-CVD",
            "metrics": risk_service.evaluation_metrics,
            "dataset": "Kaggle Heart Failure Prediction (918 Patients)"
        },
        "rag_service": {
            "embedding_model": settings.embedding_model_name,
            "vector_store": "FAISS (Inner Product / Cosine Sim)",
            "chunks_count": len(retrieval_service.chunks)
        }
    }

@app.get("/api/presets")
def get_demo_presets():
    """Returns curated demonstration patient presets."""
    return [
        {
            "id": "CASE-01",
            "title": "Case 1: Acute Ischemic Heart Disease",
            "description": "62 y/o Male with asymptomatic cardiac ischemia, hypertension, dyslipidemia, and ST-segment depression.",
            "patient": {
                "patient_id": "PT-DEMO-01",
                "age": 62,
                "sex": "M",
                "chest_pain_type": "ASY",
                "resting_bp": 155,
                "cholesterol": 280,
                "fasting_bs": 1,
                "resting_ecg": "ST",
                "max_hr": 118,
                "exercise_angina": "Y",
                "oldpeak": 2.2,
                "st_slope": "Flat"
            },
            "sample_image": "sample_cxr_cardiomegaly.png",
            "suggested_query": "What are the indications for chest radiography in chest pain evaluation?"
        },
        {
            "id": "CASE-02",
            "title": "Case 2: Lower Respiratory Infection (CAP)",
            "description": "45 y/o Female presenting with acute pleuritic non-anginal chest pain and right lower lobe radiographic consolidation.",
            "patient": {
                "patient_id": "PT-DEMO-02",
                "age": 45,
                "sex": "F",
                "chest_pain_type": "NAP",
                "resting_bp": 120,
                "cholesterol": 195,
                "fasting_bs": 0,
                "resting_ecg": "Normal",
                "max_hr": 165,
                "exercise_angina": "N",
                "oldpeak": 0.0,
                "st_slope": "Up"
            },
            "sample_image": "sample_cxr_pneumonia.png",
            "suggested_query": "What are the empiric antimicrobial therapy regimens for community-acquired pneumonia?"
        },
        {
            "id": "CASE-03",
            "title": "Case 3: Low-Risk Non-Cardiac Chest Pain",
            "description": "34 y/o Male athlete presenting with atypical chest pain, optimal hemodynamic metrics, and clear thoracic radiograph.",
            "patient": {
                "patient_id": "PT-DEMO-03",
                "age": 34,
                "sex": "M",
                "chest_pain_type": "ATA",
                "resting_bp": 115,
                "cholesterol": 175,
                "fasting_bs": 0,
                "resting_ecg": "Normal",
                "max_hr": 178,
                "exercise_angina": "N",
                "oldpeak": 0.0,
                "st_slope": "Up"
            },
            "sample_image": "sample_cxr_normal.png",
            "suggested_query": "How does the HEART Score stratify major adverse cardiac event risk?"
        }
    ]

@app.post("/api/imaging/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None)
):
    """Analyzes uploaded or sample CXR image, runs DenseNet121 and returns Grad-CAM heatmap."""
    image_bytes = None
    filename = "uploaded_cxr.png"

    if sample_name:
        sample_path = settings.sample_images_dir / sample_name
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Sample image not found.")
        image_bytes = sample_path.read_bytes()
        filename = sample_name
    elif file:
        image_bytes = await file.read()
        filename = file.filename
    else:
        raise HTTPException(status_code=400, detail="Must provide an image file or sample_name.")

    is_valid, err_msg, pil_img = validate_image_file(image_bytes, filename)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    result = imaging_service.analyze(pil_img, filename)

    # Encode original image to Base64
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    original_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Encode heatmap overlay to Base64 if available
    heatmap_b64 = None
    if result.visualization_available and result.heatmap_path and Path(result.heatmap_path).exists():
        h_bytes = Path(result.heatmap_path).read_bytes()
        heatmap_b64 = f"data:image/png;base64,{base64.b64encode(h_bytes).decode()}"

    return {
        "analysis": result.model_dump(),
        "original_image_base64": original_b64,
        "heatmap_base64": heatmap_b64
    }

@app.post("/api/risk/predict")
def predict_risk(patient: PatientClinicalInput):
    """Predicts calibrated cardiovascular risk using Kaggle-trained Gradient Boosting model."""
    result = risk_service.predict(patient)
    return result.model_dump()

@app.post("/api/guidelines/query")
def query_guidelines(req: QueryRequest):
    """Answers clinical questions grounded in FAISS vector database with page/section citations."""
    answer = guideline_service.answer_query(
        query=req.query,
        document_filter=req.document_filter,
        top_k=req.top_k
    )
    return answer.model_dump()

@app.get("/api/guidelines/inventory")
def list_guidelines():
    """Lists indexed clinical guidelines."""
    docs = []
    for g in settings.guidelines_dir.glob("*.*"):
        docs.append({
            "name": g.stem.replace("_", " ").title(),
            "filename": g.name,
            "size_kb": round(g.stat().st_size / 1024, 1),
            "format": g.suffix.upper()
        })
    return {
        "documents": docs,
        "total_chunks": len(retrieval_service.chunks)
    }

@app.post("/api/orchestration/review")
def generate_unified_review(payload: ReviewPayload):
    """Orchestrates multimodal findings into a consolidated review with strict provenance."""
    summary = orchestrator_service.build_case_summary(
        case_id=payload.case_id,
        patient_input=payload.patient_input,
        imaging_result=payload.imaging_result,
        risk_result=payload.risk_result,
        guideline_answers=payload.guideline_answers
    )
    return summary.model_dump()

# Mount Sample Images static route
app.mount("/data/sample_images", StaticFiles(directory=str(settings.sample_images_dir)), name="sample_images")

# Mount Web Application static files
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
