# Clinical Copilot — Model Cards

## 1. Medical Imaging Intelligence (DenseNet121-CXR)
- **Model Architecture:** Densely Connected Convolutional Networks (DenseNet-121)
- **Modality:** Frontal Chest Radiographs (Posterior-Anterior PA and Anteroposterior AP)
- **Input Dimensions:** 224 x 224 pixels, 3 color channels (RGB normalized by ImageNet statistics)
- **Pathologies Screened:**
  1. Normal / No Acute Infiltrate
  2. Cardiomegaly
  3. Pneumonia
  4. Pleural Effusion
  5. Pulmonary Consolidation
  6. Atelectasis
  7. Infiltration
- **Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM) computed on the final dense block feature layer (`features.denseblock4.denselayer16.conv2`).
- **Clinical Limitations:**
  - Designed for educational decision support and preliminary triaging.
  - Saliency maps provide feature correlation, not anatomical segmentation or biopsy-level confirmation.
  - Does not evaluate lateral decubitus projections or pediatric radiographs.

---

## 2. Patient Risk Assessment Model (Calibrated Gradient Boosting)
- **Model Architecture:** Gradient Boosting Classifier with 5-Fold Stratified Sigmoid Probability Calibration (`CalibratedClassifierCV`)
- **Target Outcome:** Elevated Cardiovascular Disease / Heart Failure Risk (Binary: 0 vs 1)
- **Dataset Provenance:** Real Kaggle 918-Patient Heart Failure Dataset (`fedesoriano/heart-failure-prediction`) pooled from 5 clinical research centers:
  - Cleveland Clinic Foundation (USA)
  - Hungarian Institute of Cardiology (Budapest)
  - University Hospital Zurich (Switzerland)
  - V.A. Medical Center Long Beach (USA)
  - St. Carlos University Hospital (Madrid)
- **Performance Evaluation (Held-out 20% Stratified Test Set):**
  - **ROC-AUC:** 0.9274
  - **Test Accuracy:** 88.04%
  - **Sensitivity / Recall:** 89.22%
  - **Brier Calibration Score:** 0.0968
- **Input Features (11 Validated Clinical Biomarkers):**
  - Age (Years)
  - Sex (Biological: Male / Female)
  - Chest Pain Type (ASY, NAP, ATA, TA)
  - Resting Blood Pressure (mm Hg)
  - Serum Cholesterol (mg/dL)
  - Fasting Blood Sugar > 120 mg/dL (Binary)
  - Resting ECG (Normal, ST Abnormality, LVH)
  - Maximum Heart Rate Achieved (bpm)
  - Exercise-Induced Angina (Yes / No)
  - ST Depression (Oldpeak in mm)
  - Peak Exercise ST Slope (Up, Flat, Down)

---

## 3. RAG-Powered Clinical Guidelines Retrieval
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic vectors)
- **Vector Search Engine:** FAISS (Facebook AI Similarity Search) Flat Inner Product index
- **Indexed Guidelines:**
  - 2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation and Diagnosis of Chest Pain
  - 2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure
  - 2019 ATS/IDSA Guideline on Community-Acquired Pneumonia
- **Provenance Safeguards:**
  - Citation tags `[1]`, `[2]` with exact page numbers and section headings.
  - Automated evidence sufficiency gate: If similarity falls below threshold, system explicitly reports insufficient evidence rather than hallucinating medical advice.
