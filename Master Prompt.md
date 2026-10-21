# MASTER PROMPT — Clinical Copilot
## Multimodal Clinical Assistant with Medical Imaging, Patient Risk Assessment, and RAG-Powered Clinical Guidelines

### 1. Your Role

Act as a senior AI/ML engineer, healthcare software architect, medical AI researcher, full-stack developer, UI/UX designer, and software testing engineer.

Build a complete, modular, professional healthcare AI application named **Clinical Copilot**.

Work directly inside my existing VS Code project using Antigravity. Inspect the existing files and environment before making changes. If the project is empty, create the required structure from scratch.

Do not simply generate a plan or a static dashboard. Build a working application, integrate its components, run it, test the functionality, and fix errors wherever possible.

Use sensible technical defaults when details are missing. Do not repeatedly stop to ask questions that can be resolved through reasonable engineering decisions.

---

### 2. Project Vision

Clinical Copilot is a multimodal clinical decision-support platform that combines three healthcare AI capabilities into one application:

**Module A: Medical Imaging Intelligence**
- Accept supported medical images, initially chest X-rays.
- Validate file formats, image dimensions, and upload limits.
- Use a suitable pretrained medical imaging model or a documented, compatible model integration.
- Identify supported visual patterns and produce model-generated findings with confidence scores where available.
- Display the uploaded image and relevant visual highlights when supported by the model.
- Explain the model's limitations and distinguish AI-generated findings from confirmed diagnoses.
- Handle unsupported images and failed model inference gracefully.

**Module B: Patient Risk Assessment**
- Accept structured patient information and relevant clinical measurements.
- Support configurable risk prediction for a clearly defined clinical outcome.
- Use a suitable validated prediction model, or a transparent baseline model for demonstration purposes.
- Display the estimated risk, the prediction horizon when applicable, contributing factors, and missing information.
- Explain which patient inputs influenced the result.
- Clearly distinguish model predictions from established clinical facts.
- Do not invent clinical thresholds, model accuracy, or validation results.

**Module C: RAG-Powered Clinical Guidelines**
- Allow users to upload or index authorized clinical guidelines and medical reference documents.
- Extract, clean, chunk, embed, and index the documents.
- Retrieve relevant passages in response to clinical questions.
- Generate answers grounded in retrieved evidence.
- Display citations containing document names and page numbers or section references whenever available.
- Clearly state when the available documents do not contain enough evidence.
- Prevent the language model from inventing citations, guideline recommendations, or medical facts.

**Unified Clinical Copilot**
Integrate the three modules into a single interface where clinicians can review imaging findings, patient risk estimates, and relevant guideline evidence together.

The application must never imply that a finding from one module automatically confirms the output of another module.

---

### 3. Recommended Technology Stack

Use this stack unless the existing project already has a suitable, working alternative.

**Frontend**
- Streamlit for a functional initial interface.
- Plotly for charts and risk visualizations.
- A clean, responsive layout with clear navigation and professional healthcare styling.

**Backend**
- Python.
- Modular services for imaging, risk assessment, retrieval, and application logic.
- Pydantic for input validation and structured outputs.
- FastAPI may be introduced if a separate API backend is beneficial.

**Artificial Intelligence**
- PyTorch and compatible pretrained models for medical image analysis.
- Scikit-learn for suitable baseline risk models.
- Hugging Face Transformers where appropriate.
- Sentence Transformers for text embeddings.
- A compatible vector store such as FAISS or Chroma for guideline retrieval.
- A configurable LLM provider for generating evidence-grounded explanations.

**Data and Storage**
- Pandas and NumPy for structured clinical data.
- SQLite for a local development prototype when persistence is needed.
- Environment variables for API keys and configuration.
- A configurable document storage location.

**Testing and Quality**
- Pytest.
- Logging and exception handling.
- Type hints and reusable Python modules.
- A requirements file or suitable project dependency configuration.

Choose compatible library versions and verify actual model requirements before installing dependencies.

Do not require paid APIs to run the basic demonstration. Where an external API is optional, provide a clear configuration path and a useful offline or mock-data alternative.

---

### 4. Application Interface

Create a polished application with the following pages.

#### Page 1: Clinical Overview
Show:
- Application name and purpose.
- Imaging analysis entry point.
- Patient risk assessment entry point.
- Clinical guideline assistant entry point.
- Recent demonstration cases, if available.
- System and model availability indicators.
- Clear warnings that this is a decision-support tool.

Use synthetic demonstration data only. Do not populate the interface with fabricated real patients or misleading clinical performance statistics.

#### Page 2: Medical Imaging Analysis
Provide:
- Image upload control.
- Image preview.
- Supported-format and file-size validation.
- Analysis button.
- Structured findings panel.
- Confidence scores when supplied by the underlying model.
- Optional image localization or heatmap visualization if genuinely supported.
- Model name, version, and relevant limitations.

Never create a fake heatmap or represent an illustrative image overlay as a validated localization result.

If a suitable pretrained model cannot be obtained, build a working upload and visualization workflow with an explicitly labeled demonstration mode. Do not pretend that mock outputs are real model predictions.

#### Page 3: Patient Risk Assessment
Provide a structured patient information form.

Include only the inputs required by the selected prediction model. Possible inputs may include:
- Age.
- Relevant clinical history.
- Vital signs.
- Relevant laboratory measurements.
- Other validated model-specific features.

The exact features must depend on the chosen clinical outcome and model.

Display:
- Estimated probability or risk category when justified.
- Prediction horizon, if defined.
- Contributing features or model explanations.
- Missing or invalid inputs.
- Model limitations and appropriate uncertainty information.

If no suitable validated model or dataset is available, provide a transparent baseline or educational demonstration mode. Never present an arbitrary formula as a validated medical risk calculator.

#### Page 4: Clinical Guidelines Assistant
Provide:
- A chat-style question interface.
- Guideline document upload and indexing.
- Document selection or filtering.
- Retrieved evidence passages.
- Answer summaries.
- Source citations and page references when available.
- A visible indication when evidence is insufficient.

The assistant must answer using retrieved guideline evidence rather than relying solely on the LLM's general knowledge.

If a user asks a question unrelated to the indexed documents, explain the limitation and offer a general informational response only when the system can safely distinguish it from a document-grounded answer.

#### Page 5: Unified Patient Case Review
Create a workspace where the user can review:
- Patient information entered for the current case.
- Imaging analysis results.
- Patient risk estimates.
- Relevant guideline passages.
- A consolidated evidence summary.
- Missing data and unresolved uncertainties.

The consolidated summary must preserve the source of each finding. Clearly label whether information came from user-entered data, the imaging model, the risk model, or retrieved guidelines.

Do not merge conflicting outputs into a single unsupported conclusion.

#### Page 6: Model and Data Transparency
Display:
- Active model identifiers and versions.
- Data source descriptions.
- Guideline document inventory.
- Model limitations.
- Whether results come from live inference, cached outputs, or demonstration data.
- Configuration and dependency status.

Never display fabricated accuracy, sensitivity, specificity, AUC, or other evaluation metrics.

---

### 5. Medical Imaging Implementation Requirements

First identify a suitable pretrained model for the selected imaging task.

Evaluate the model based on:
- Intended clinical task.
- Training and validation dataset.
- Model license and permitted usage.
- Input image requirements.
- Available inference code.
- Hardware requirements.
- Known limitations and external validation evidence.

Start with chest X-ray analysis rather than claiming support for every imaging modality.

Keep the model inference logic separate from the interface so the model can be replaced later.

Use a structured output format similar to:

- `model_name`
- `model_version`
- `task`
- `findings`
- `confidence_scores`
- `visualization_available`
- `warnings`
- `inference_status`

Adapt the schema to the actual model output. Do not infer probabilities or clinical meanings that the model does not provide.

Use appropriate preprocessing for the selected model. Validate image orientation, dimensions, color channels, and pixel handling where applicable.

Do not treat a generic image-classification model as a clinically validated diagnostic system.

---

### 6. Patient Risk Model Requirements

Choose one well-defined clinical outcome for the initial implementation.

Before selecting a model, document:
- Intended population.
- Required input variables.
- Predicted outcome.
- Prediction horizon.
- Training data provenance.
- Validation methodology.
- Known limitations.

Prefer a publicly documented, appropriately licensed model or dataset suitable for educational research.

Do not use synthetic data to claim clinical validity.

When real data is unavailable, create clearly labeled synthetic examples for testing the interface and application logic.

Implement:
- Schema and range validation.
- Missing-value handling.
- Model-specific preprocessing.
- Prediction logic.
- Explanation of available model features.
- Unit tests for valid and invalid inputs.

Avoid using protected or sensitive attributes in predictions unless their use is explicitly justified, lawful, and appropriate for the selected model.

Do not claim that feature importance establishes causation.

---

### 7. RAG Pipeline Requirements

Implement a reliable retrieval-augmented generation pipeline.

**Step 1: Document ingestion**
- Support PDF and plain-text files initially.
- Extract text while preserving page information when possible.
- Detect empty, corrupted, and unsupported documents.
- Record document title, source, version, publication date when available, and ingestion status.

**Step 2: Text processing**
- Clean extracted text without removing medically meaningful information.
- Split documents into appropriately sized chunks.
- Preserve document identifiers, page numbers, and section metadata.

**Step 3: Embeddings and indexing**
- Generate embeddings with a compatible model.
- Store vectors and metadata in the selected vector database.
- Prevent duplicate indexing where feasible.
- Support document re-indexing and removal.

**Step 4: Retrieval**
- Retrieve the most relevant passages for the question.
- Filter by selected document or guideline when requested.
- Return source metadata with every retrieved passage.
- Handle empty or low-relevance retrieval results.

**Step 5: Answer generation**
- Give the LLM only the evidence needed for the answer.
- Instruct the LLM to use the retrieved evidence and acknowledge uncertainty.
- Require citations for claims attributed to the indexed documents.
- Do not fabricate quotations, pages, document names, or guideline recommendations.
- If evidence is insufficient or conflicting, state that explicitly.

**Step 6: Source display**
- Show citations next to the claims they support.
- Include page numbers or section headings when reliably extracted.
- Make source documents easy to inspect.

Treat document contents as untrusted evidence, not executable instructions. Uploaded documents must not be allowed to override system instructions, expose secrets, or trigger unauthorized actions.

---

### 8. Integrating All Three Modules

The unified workflow should operate as follows:

1. The user creates a demonstration case or enters a patient case.
2. The user supplies relevant patient information.
3. The user uploads a supported medical image when imaging analysis is required.
4. The imaging service generates its model-specific findings.
5. The risk service generates a prediction only when its required inputs are available.
6. The RAG service retrieves relevant evidence from the configured guidelines.
7. The application presents each output with its own provenance, limitations, and status.
8. The unified review summarizes the available evidence without creating an unsupported diagnosis.

Each module must be independently usable.

If one module fails, the other modules should remain available whenever technically possible.

Do not allow an imaging result to silently become a risk-model input unless that relationship has been explicitly designed, justified, and validated.

---

### 9. Safety, Privacy, and Security

Implement these requirements from the beginning.

- Display a clear statement that the application is a research and clinical decision-support prototype, not a substitute for professional medical judgment.
- Do not independently prescribe treatment or medication.
- Do not issue definitive diagnoses from uncertain model predictions.
- Do not fabricate clinical evidence or claim regulatory approval.
- Do not describe a system as clinically validated unless supporting evidence establishes that status.
- Clearly identify demonstration and mock-data modes.
- Validate uploaded files and restrict file size and accepted types.
- Protect against unsafe file paths, malicious documents, and unauthorized file access.
- Never hardcode API keys or credentials.
- Avoid sending patient data to external services without explicit configuration, authorization, and appropriate safeguards.
- Use synthetic patient data for the initial demonstration.
- Avoid retaining identifiable patient data unnecessarily.
- Provide appropriate access controls before adding real patient records or multi-user functionality.
- Include audit-friendly logging without writing sensitive clinical information or credentials into logs.
- Design for applicable privacy and healthcare requirements before any real-world clinical deployment.

If a user describes an emergency, do not let an experimental AI prediction delay urgent professional medical care.

---

### 10. Suggested Project Structure

Organize the project into clear modules. Adapt this structure to the existing repository instead of duplicating files unnecessarily.

```text
clinical-copilot/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── config/
│   └── settings.py
├── modules/
│   ├── imaging/
│   │   ├── inference.py
│   │   ├── preprocessing.py
│   │   └── schemas.py
│   ├── risk/
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   └── schemas.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── chunking.py
│   │   ├── retrieval.py
│   │   └── generation.py
│   └── orchestration/
│       └── clinical_review.py
├── ui/
│   ├── dashboard.py
│   ├── imaging_page.py
│   ├── risk_page.py
│   ├── guidelines_page.py
│   └── case_review_page.py
├── data/
│   ├── guidelines/
│   └── synthetic/
├── tests/
│   ├── test_imaging.py
│   ├── test_risk.py
│   ├── test_rag.py
│   └── test_integration.py
└── docs/
    ├── architecture.md
    ├── model_card.md
    └── limitations.md
```

Do not commit sensitive data, downloaded model weights without checking their licenses, or local secrets.

---

### 11. Development Workflow for Antigravity

Follow these phases in order.

**Phase 1 — Inspect and plan**
- Inspect the existing repository.
- Identify the installed Python version, dependencies, available hardware, and current application structure.
- Check available model and dataset requirements.
- Select a realistic initial imaging model, risk model, and guideline source.
- Write a short implementation plan and document technical limitations.

**Phase 2 — Establish the application**
- Set up dependencies and configuration.
- Create the main application entry point.
- Implement the interface and navigation.
- Add input validation, logging, and exception handling.

**Phase 3 — Implement imaging**
- Integrate the selected model or clearly labeled demonstration adapter.
- Implement image validation, preprocessing, inference, and structured outputs.
- Test supported and unsupported files.

**Phase 4 — Implement patient risk**
- Integrate the selected prediction model.
- Validate its input schema and preprocessing.
- Display results and explanations accurately.
- Test missing and invalid inputs.

**Phase 5 — Implement RAG**
- Build document ingestion and chunking.
- Add embeddings and vector search.
- Implement evidence-grounded answers and source citations.
- Test retrieval failures, missing evidence, and conflicting passages.

**Phase 6 — Integrate the unified review**
- Connect the three modules.
- Preserve source attribution.
- Handle missing module results and errors.
- Ensure no module is falsely represented as successful when it fails.

**Phase 7 — Test and debug**
- Run the application.
- Run unit and integration tests.
- Check imports, dependencies, model loading, file uploads, document retrieval, citations, and navigation.
- Fix discovered errors.
- Verify that the application can launch using the documented command.

**Phase 8 — Document and deliver**
- Write complete setup instructions.
- Document required environment variables.
- Explain model and dataset setup.
- Include the launch command.
- Document limitations and the difference between real inference and demonstration mode.
- Provide a concise summary of implemented features and remaining limitations.

Do not skip testing simply because the application starts successfully.

---

### 12. Testing and Acceptance Criteria

The initial version is complete only when the following criteria have been checked.

**Application**
- The application launches without a blocking error.
- Navigation and page rendering work.
- Input validation and error messages are understandable.

**Imaging**
- Supported image uploads work.
- Invalid uploads are rejected safely.
- The configured model runs successfully, or demonstration mode is explicitly identified.
- The output matches the actual model response.

**Risk assessment**
- Valid inputs produce the expected type of output.
- Missing and invalid inputs are handled safely.
- Risk scores are not fabricated.
- Demonstration predictions are labeled appropriately.

**RAG**
- Documents can be ingested and indexed.
- Questions retrieve relevant passages.
- Answers display traceable citations.
- Unsupported claims and missing evidence are handled honestly.

**Integration**
- The case review preserves the origin of each finding.
- Failed modules do not silently produce successful-looking outputs.
- Missing results do not break the complete interface.

**Documentation**
- Installation and startup instructions are available.
- Model limitations and data provenance are documented.
- Secrets and patient information are excluded from version control.

Report any acceptance criterion that cannot be completed. Do not claim that all tests passed without running them.

---

### 13. Engineering Rules

Follow these rules throughout development:

1. Build working functionality rather than a frontend-only prototype.
2. Use modular, readable, maintainable code.
3. Avoid unnecessary complexity and excessive dependencies.
4. Do not fabricate datasets, model weights, API responses, or medical evidence.
5. Do not silently replace a failed real model with mock output.
6. Use documented interfaces between modules.
7. Keep model inference separate from presentation logic.
8. Handle errors explicitly and provide useful user feedback.
9. Preserve existing working code unless a change is necessary.
10. Do not overwrite unrelated files.
11. Explain important technical decisions briefly.
12. Verify compatibility before installing or integrating libraries.
13. Use synthetic data for testing when real clinical data is unavailable.
14. Do not claim production readiness, clinical validation, or regulatory compliance without evidence.
15. Prefer a small, fully working implementation over several incomplete advanced features.

If hardware, model downloads, credentials, or external services are unavailable, implement the parts that can be verified locally and clearly document what remains blocked.

---

### 14. Final Deliverable

Build **Clinical Copilot** as a unified, modular, demonstrable healthcare AI application.

The final result should include:

- A professional clinical dashboard.
- A medical imaging analysis module.
- A patient risk assessment module.
- A RAG-powered clinical guidelines assistant.
- A unified case review workspace.
- Traceable evidence citations.
- Transparent model outputs and limitations.
- Automated tests.
- Installation and execution documentation.

Start by inspecting the repository and creating a concise implementation plan. Then proceed phase by phase, verifying each component before integrating it into the complete application.

**Primary objective:** Deliver a working and technically credible clinical decision-support prototype that integrates multimodal AI, patient risk estimation, and evidence-grounded guideline retrieval without overstating its clinical capabilities.