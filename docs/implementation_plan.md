# Clinical Copilot — Implementation Architecture & Plan

## 1. System Overview
**Clinical Copilot** is a multimodal clinical decision-support platform integrating three core modules:
1. **Module A — Medical Imaging Intelligence:** Chest X-ray analysis with deep learning (DenseNet121 / ResNet50 CXR backbone), multi-pathology confidence scoring (Pneumonia, Cardiomegaly, Atelectasis, Effusion, Consolidation, Infiltration, Normal), Grad-CAM anatomical saliency heatmaps, and diagnostic uncertainty warnings.
2. **Module B — Patient Risk Assessment:** Supervised clinical risk modeling trained directly on the Kaggle 918-patient Heart Failure Clinical Dataset (`fedesoriano/heart-failure-prediction`), utilizing validated clinical features (Age, Sex, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope), ROC-AUC evaluation, and SHAP/feature contribution breakdown.
3. **Module C — RAG-Powered Clinical Guidelines:** Guideline retrieval and question answering grounded in ACC/AHA cardiovascular guidelines and ATS/IDSA respiratory infection guidelines, using FAISS vector indexing, Sentence-Transformers embeddings, chunk metadata tracking (page, section, document), evidence citations, and honest uncertainty fallback when evidence is insufficient.
4. **Unified Clinical Review:** Synthesizes patient inputs, imaging findings, risk scores, and guideline passages into a structured clinical review with strict provenance tracking.

---

## 2. Technical Stack & Hardware Specification
- **OS & Environment:** Windows 11, Python 3.12.10 virtual environment (`venv`).
- **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA capable).
- **Core Libraries:**
  - UI/UX: Streamlit with custom clinical dark/light design system, Plotly interactive charts.
  - Imaging: PyTorch, Torchvision (DenseNet121 / ResNet CXR), Grad-CAM visualizer, PIL.
  - Risk Model: Scikit-learn (RandomForestClassifier / GradientBoostingClassifier / CalibratedClassifierCV), Pandas, NumPy.
  - RAG: Sentence-Transformers (`all-MiniLM-L6-v2`), FAISS-cpu, PyPDF.
  - Architecture: Pydantic schemas, modular services, Pytest automated test suite.

---

## 3. Real Kaggle Datasets & Provenance
- **Dataset 1 (Patient Risk):** `fedesoriano/heart-failure-prediction`
  - Provenance: 918 patient records pooled from 5 clinical heart disease datasets (Cleveland Clinic, Hungarian Institute of Cardiology, University Hospital of Zurich/Switzerland, V.A. Medical Center Long Beach, and St. Carlos University Hospital).
  - Target: Heart Disease / Cardiovascular failure risk (binary classification 0 vs 1).
  - Validation: Train/Test split with Stratified 5-Fold Cross Validation, Calibrated probabilities, and Feature importance.
- **Dataset 2 (Medical Imaging):** Real Chest X-Ray samples (Pneumonia vs Normal vs Cardiomegaly) sourced from Kaggle's Chest X-Ray dataset & NIH ChestX-ray benchmark.
- **Dataset 3 (Guidelines):** Authoritative Clinical Guidelines (AHA/ACC Chest Pain and Heart Failure Guideline Summary, ATS/IDSA Guidelines for Community-Acquired Pneumonia).

---

## 4. Phase-by-Phase Roadmap
- **Phase 1 (Completed):** Environment inspection, venv creation, requirements definition, planning.
- **Phase 2 (In Progress):** Dependency installation, settings module, logging, data directory initialization, Kaggle data ingestion.
- **Phase 3:** Module A (Imaging preprocessing, CXR inference model, Grad-CAM saliency map generation, schema validation).
- **Phase 4:** Module B (Kaggle Heart Failure dataset training, model serialization, risk inference engine, feature attribution).
- **Phase 5:** Module C (Guideline PDF/Text ingestion, chunking with metadata, FAISS vector index, similarity retrieval, evidence-grounded generation).
- **Phase 6:** Unified Orchestrator & Case Review (Data provenance aggregation, conflict detection, uncertainty reporting).
- **Phase 7:** Streamlit UI (Overview, Imaging, Risk, Guidelines, Case Review, Model Transparency).
- **Phase 8:** Automated Pytest validation suite, verification, and documentation.
