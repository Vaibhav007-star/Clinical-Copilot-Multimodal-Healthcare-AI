# Clinical Copilot ⚕️
### Multimodal Clinical Assistant with Medical Imaging, Patient Risk Assessment, and RAG-Powered Clinical Guidelines

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)](https://streamlit.io/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Heart%20Failure-20BEFF.svg)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
[![License](https://img.shields.io/badge/License-MIT%20Research-green.svg)](LICENSE)

---

## 🌟 Overview
**Clinical Copilot** is a modular, transparent healthcare AI application designed for clinical decision-support and medical education. It combines three multimodal AI capabilities into a unified clinician workspace:

1. **Module A — Medical Imaging Intelligence:** Frontal Chest X-Ray (CXR) radiograph screening with DenseNet121, multi-pathology confidence scoring, anatomical **Grad-CAM** visual saliency maps, **Multi-Pathology Heatmap Switching**, and **DICOM-Style Radiological Windowing** (Standard, Lung Window, Mediastinal/Soft Tissue, Bone Detail, and Invert/Negative Film).
2. **Module B — Patient Risk & Counterfactual Intelligence:** 
   - Calibrated cardiovascular disease risk calculator trained on the real **Kaggle 918-Patient Heart Failure Cohort** achieving **92.7% Test ROC-AUC**.
   - **Interactive What-If Counterfactual Simulator:** Models prospective clinical interventions with dynamic calculation of Absolute Risk Reduction (ARR), Relative Risk Reduction (RRR), Number Needed to Treat (NNT), and ranked high-yield modifiable factors.
   - **Evidence-Based Multi-Score Panel:** Integrated calculators for **Framingham 10-Yr ASCVD**, **CURB-65 Pneumonia Severity**, **Wells' Criteria for Pulmonary Embolism**, and **CHA2DS2-VASc AFib Stroke Risk**.
3. **Module C — Guidelines RAG & Safety Guardrails:** Semantic search grounded in AHA/ACC Chest Pain, AHA/ACC Heart Failure, and ATS/IDSA CAP guidelines with FAISS vector indexing, traceable citations, and **Automated Pharmacotherapy Contraindication Screening** (e.g., Nitrates + PDE-5 inhibitors, Beta-Blockers in bradycardia, NSAIDs in decompensated HF).
4. **Unified Multimodal Synthesis & Clinical Governance:**
   - **Multimodal AI Differential Diagnosis (DDx) Engine:** Produces primary working diagnoses with ICD-10 codes, probability estimates, rival diagnoses, ruled-out conditions with negative clinical rationale, red-flag alerts, and STAT/urgent workup orders.
   - **HL7 FHIR R4 Interoperability:** Generates standardized FHIR R4 Document Bundles (`Patient`, `Observation`, `RiskAssessment`, `DiagnosticReport`) ready for hospital EHR ingestion.
   - **Electronic Clinician Attestation:** Attending physician sign-off workflow with cryptographic audit stamp minting.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.12 (Windows / Linux / macOS)
- NVIDIA GPU with CUDA support (optional; automatically falls back to CPU if unavailable)

### 2. Environment Activation & Dependencies
```powershell
# Clone or navigate to the project directory
cd "c:\Projects\Multimodal Clinical Assistant with Medical Imaging, Patient Risk Assessment, and RAG-Powered Clinical Guidelines"

# Activate the existing virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Launch Option A — Interactive Web Application (Recommended)
Launch the modern, responsive Web Application with bespoke dark glassmorphism, interactive Grad-CAM heatmap opacity slider, dynamic SVG risk gauge, and real-time RAG:
```powershell
.\venv\Scripts\python.exe -m uvicorn api.server:app --host 0.0.0.0 --port 8080
```
Open your browser to: **[http://localhost:8080](http://localhost:8080)**

### 4. Launch Option B — Streamlit Dashboard
If you prefer running the Streamlit interface:
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Open your browser to: **[http://localhost:8501](http://localhost:8501)**

---

## 📂 Project Architecture

```text
├── app.py                     # Streamlit application entry point & navigation
├── requirements.txt           # Verified project dependencies
├── pytest.ini                 # Test discovery & Python path configuration
├── config/
│   └── settings.py            # Global paths, model settings, and environment configs
├── modules/
│   ├── imaging/               # Module A: Medical Imaging
│   │   ├── inference.py       # DenseNet121 model + Grad-CAM saliency engine
│   │   ├── preprocessing.py   # Image format/size validation & tensor normalization
│   │   └── schemas.py         # Pydantic structured output models
│   ├── risk/                  # Module B: Patient Risk Assessment
│   │   ├── model.py           # Calibrated Gradient Boosting on Kaggle Heart Failure
│   │   ├── preprocessing.py   # Clinical feature encoding & cholesterol imputation
│   │   └── schemas.py         # Patient clinical inputs & feature attribution schemas
│   ├── rag/                   # Module C: RAG-Powered Clinical Guidelines
│   │   ├── ingestion.py       # Document loaders (.txt, .pdf)
│   │   ├── chunking.py        # Semantic chunker with page/section preservation
│   │   ├── retrieval.py       # FAISS vector database + Sentence-Transformers
│   │   ├── generation.py      # Grounded answer synthesizer with citation tagging
│   │   └── schemas.py         # Citations, Chunks, and Grounded Answer schemas
│   └── orchestration/         # Unified Case Review Engine
│       ├── clinical_review.py # Multimodal synthesis & provenance tracker
│       └── schemas.py         # Provenance table & unified case summary schemas
├── ui/                        # Clinical Dashboard Pages
│   ├── dashboard.py           # Page 1: Clinical Overview & Case Presets
│   ├── imaging_page.py        # Page 2: Medical Imaging Analysis & Grad-CAM
│   ├── risk_page.py           # Page 3: Patient Risk Assessment & Gauge Meter
│   ├── guidelines_page.py     # Page 4: Guidelines Assistant & Citation Inspector
│   ├── case_review_page.py    # Page 5: Unified Patient Case Review & Report Export
│   └── transparency_page.py   # Page 6: Model Cards, Validation Metrics & Audit
├── data/
│   ├── datasets/              # Real Kaggle Heart Failure dataset (918 patients)
│   ├── guidelines/            # AHA/ACC Chest Pain, HF & ATS/IDSA CAP guidelines
│   ├── sample_images/         # Preloaded CXR test images (Cardiomegaly, Pneumonia, Normal)
│   └── vector_store/          # FAISS index and chunk metadata JSON
├── tests/                     # Automated Test Suite (Pytest)
│   ├── test_imaging.py        # CXR validation & vision inference tests
│   ├── test_risk.py           # Clinical boundary tests & calibration checks
│   ├── test_rag.py            # FAISS retrieval & honest fallback tests
│   └── test_integration.py    # End-to-end multimodal orchestration tests
└── docs/
    ├── implementation_plan.md # Development phases and technical roadmap
    ├── model_card.md          # Formal model cards & evaluation benchmarks
    └── limitations.md         # Clinical governance and regulatory disclaimers
```

---

## 🧪 Running Automated Tests
```powershell
.\venv\Scripts\pytest.exe -v
```

---

## 🔒 Clinical Governance & Safety Notice
**FOR RESEARCH AND CLINICAL DECISION-SUPPORT DEMONSTRATION ONLY.**
- This application does **not** provide independent diagnostic claims or prescribe pharmacotherapy.
- Model probabilities and feature importances reflect correlational patterns within observational cohorts, not clinical causation.
- Never delay emergent medical triage (e.g., suspected acute coronary syndrome, aortic dissection, tension pneumothorax) for AI software evaluation.
