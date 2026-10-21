# Clinical Copilot — Clinical Limitations & Risk Governance

## 1. Regulatory Status & Scope of Intended Use
- **Classification:** Research Prototype & Educational Clinical Decision Support (CDS) System.
- **Regulatory Clearance:** Clinical Copilot has **not** been evaluated, certified, or cleared as a Medical Device (SaMD) by the U.S. FDA, European Medicines Agency (EMA), or any other regulatory health authority.
- **Intended Population:** Simulated adult cardiology and pulmonology patient profiles. Not tested or validated for pediatric cohorts, obstetric patients, or acute emergency trauma cases.

---

## 2. Medical Imaging Model Limitations (Module A)
- **Modality Scope:** Only frontal (PA / AP) chest radiographs are supported. Lateral, oblique, decubitus, and CT/MRI slices are unsupported.
- **Grad-CAM Saliency Maps:** Visual heatmaps indicate feature gradients from the final dense convolutional layer contributing to model classification scores. They are **not** anatomical segmentations and must not be used to guide surgical resection, biopsy, or radiation therapy.
- **Clinical Artifacts:** Radiographic artifacts such as surgical clips, pacemakers, jewelry, motion blur, and underexposure/overexposure can degrade model accuracy.

---

## 3. Patient Risk Assessment Model Limitations (Module B)
- **Training Cohort:** 918 adult cardiology patients across five medical centers (Cleveland Clinic, Hungarian Institute, Univ. of Zurich, Long Beach VA, St. Carlos).
- **Correlational Nature:** Feature importances and SHAP-style weights represent statistical associations within the training cohort. They do **not** imply clinical etiology or biological causation.
- **Emergency Exclusions:** Not validated for acute ST-Elevation Myocardial Infarction (STEMI) triage. Patients presenting with hemodynamic collapse, sustained ventricular arrhythmias, or acute pulmonary edema require immediate emergency intervention rather than risk calculator estimation.

---

## 4. Guidelines RAG Limitations (Module C)
- **Knowledge Base Scope:** Grounded exclusively in indexed clinical documents (AHA/ACC Chest Pain, AHA/ACC Heart Failure, ATS/IDSA CAP). Unindexed clinical specialties (e.g., oncology, nephrology, pediatrics) are outside the evidence retrieval boundary.
- **Evidence Sufficiency Gate:** When similarity between clinician inquiry and indexed passages falls below calibrated thresholds, the system intentionally returns an "Insufficient Evidence" statement to prevent hallucination.
- **Local Adaptation:** Hospital-specific formularies, antimicrobial resistance patterns, and regional clinical protocols supersede general guideline recommendations.

---

## 5. Multimodal Orchestration Safeguards
- **Independent Validation:** Findings from one modality do not automatically corroborate or validate another modality.
- **Traceability:** Every finding in the Unified Case Review explicitly displays its origin (User Entry, Vision Model, Supervised Risk Model, or Guideline Citation).
- **Attending Physician Mandate:** All clinical decisions, diagnostic orders, and treatment prescriptions remain the sole responsibility of the licensed attending physician.
