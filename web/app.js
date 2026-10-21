/**
 * Clinical Copilot — Multimodal Web Application Logic
 * Integrates DenseNet121 Imaging, Kaggle-trained Risk Engine, and FAISS Guidelines RAG.
 */

// Application State
const appState = {
    activePatient: {
        patient_id: "PT-1001",
        age: 55,
        sex: "M",
        chest_pain_type: "ASY",
        resting_bp: 135,
        cholesterol: 230,
        fasting_bs: 0,
        resting_ecg: "Normal",
        max_hr: 145,
        exercise_angina: "N",
        oldpeak: 1.0,
        st_slope: "Flat"
    },
    imagingResult: null,
    riskResult: null,
    guidelineAnswers: [],
    selectedImageFile: null,
    selectedSampleName: null,
    imagingChart: null
};

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initPresets();
    initImagingStudio();
    initRiskStudio();
    initGuidelinesStudio();
    initResetButton();
    fetchSystemStatus();
    computeActiveMultiScores(appState.activePatient);
    loadPharmacotherapySafety(appState.activePatient);
});

/* ==========================================================================
   Navigation
   ========================================================================== */
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const viewId = item.getAttribute('data-view');
            navigateTo(viewId);
        });
    });
}

function navigateTo(viewId) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-view') === viewId);
    });

    // Update panels
    document.querySelectorAll('.view-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === viewId);
    });

    // Update header title & breadcrumb
    const titles = {
        'view-overview': { title: 'Clinical Decision-Support Overview', crumb: 'Clinical Copilot / Overview' },
        'view-imaging': { title: 'Medical Imaging Intelligence (CXR)', crumb: 'Clinical Copilot / Medical Imaging' },
        'view-risk': { title: 'Supervised Patient Risk Assessment', crumb: 'Clinical Copilot / Patient Risk' },
        'view-guidelines': { title: 'RAG-Powered Clinical Practice Guidelines', crumb: 'Clinical Copilot / Guidelines RAG' },
        'view-review': { title: 'Unified Multimodal Case Review', crumb: 'Clinical Copilot / Case Review' },
        'view-transparency': { title: 'Model & Data Transparency Cards', crumb: 'Clinical Copilot / Model Transparency' }
    };

    if (titles[viewId]) {
        document.getElementById('viewTitle').textContent = titles[viewId].title;
        document.getElementById('viewBreadcrumb').textContent = titles[viewId].crumb;
    }

    // Refresh review when opening review view
    if (viewId === 'view-review') {
        refreshUnifiedReview();
    }
}

/* ==========================================================================
   Presets & System Status
   ========================================================================== */
async function fetchSystemStatus() {
    try {
        const res = await fetch('/api/status');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('statChunksCount').textContent = `${data.rag_service.chunks_count} Chunks`;
            document.getElementById('systemStatusText').textContent = 'Kaggle & Vision Models Live';
        }
    } catch (e) {
        console.warn('System status poll error:', e);
    }
}

async function initPresets() {
    try {
        const res = await fetch('/api/presets');
        if (!res.ok) return;
        const presets = await res.json();
        
        const container = document.getElementById('presetsContainer');
        container.innerHTML = '';

        presets.forEach(preset => {
            const card = document.createElement('div');
            card.className = 'preset-card';
            card.innerHTML = `
                <div class="preset-card-header">
                    <h4>${preset.title}</h4>
                    <span class="badge badge-cyan">${preset.id}</span>
                </div>
                <p>${preset.description}</p>
                <button class="btn btn-outline btn-sm" onclick="loadPreset('${preset.id}')">Load Case Preset →</button>
            `;
            container.appendChild(card);
        });

        // Store presets in window for lookup
        window.presetsData = presets;
    } catch (e) {
        console.error('Error fetching presets:', e);
    }
}

function loadPreset(presetId) {
    if (!window.presetsData) return;
    const preset = window.presetsData.find(p => p.id === presetId);
    if (!preset) return;

    // Apply patient
    appState.activePatient = { ...preset.patient };
    updateSidebarCase();

    // Populate Risk form
    populateRiskForm(preset.patient);

    // Apply sample image
    appState.selectedSampleName = preset.sample_image;
    appState.selectedImageFile = null;
    document.getElementById('selectSampleImage').value = preset.sample_image;
    
    // Automatically trigger sample preview
    loadSamplePreview(preset.sample_image);

    // Populate suggested query
    document.getElementById('guidelineQueryInput').value = preset.suggested_query;

    // Refresh advanced modules for new preset
    computeActiveMultiScores(preset.patient);
    loadPharmacotherapySafety(preset.patient);

    showToast(`Loaded ${preset.title}! Navigate to Imaging or Risk to inspect.`);
}

function updateSidebarCase() {
    const p = appState.activePatient;
    document.getElementById('sidebarCaseId').textContent = p.patient_id || 'PT-1001';
    document.getElementById('sidebarCaseDetails').textContent = `Demographics: ${p.age} y/o • ${p.sex === 'M' ? 'Male' : 'Female'} • Chest Pain: ${p.chest_pain_type}`;
}

function initResetButton() {
    document.getElementById('btnResetCase').addEventListener('click', () => {
        appState.activePatient = {
            patient_id: "PT-1001",
            age: 55,
            sex: "M",
            chest_pain_type: "ASY",
            resting_bp: 135,
            cholesterol: 230,
            fasting_bs: 0,
            resting_ecg: "Normal",
            max_hr: 145,
            exercise_angina: "N",
            oldpeak: 1.0,
            st_slope: "Flat"
        };
        appState.imagingResult = null;
        appState.riskResult = null;
        appState.guidelineAnswers = [];
        appState.selectedImageFile = null;
        appState.selectedSampleName = null;

        updateSidebarCase();
        populateRiskForm(appState.activePatient);

        // Reset imaging viewer
        document.getElementById('imageCanvasWrapper').style.display = 'none';
        document.getElementById('emptyImageViewer').style.display = 'block';
        document.getElementById('sliderWrapper').style.display = 'none';
        document.getElementById('imagingFindingsCard').style.display = 'none';
        document.getElementById('btnRunImaging').disabled = true;
        document.getElementById('selectSampleImage').value = '';

        // Reset risk gauge
        resetRiskGauge();

        // Reset RAG
        document.getElementById('ragResultCard').style.display = 'none';
        document.getElementById('guidelineQueryInput').value = '';

        showToast('Active case reset to defaults.');
    });
}

/* ==========================================================================
   Medical Imaging Studio
   ========================================================================== */
function initImagingStudio() {
    const dropzone = document.getElementById('cxrDropzone');
    const fileInput = document.getElementById('cxrFileInput');
    const sampleSelect = document.getElementById('selectSampleImage');
    const runBtn = document.getElementById('btnRunImaging');
    const opacitySlider = document.getElementById('heatmapOpacity');

    // Click dropzone to open file browser
    dropzone.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleImageSelection(e.target.files[0], null);
        }
    });

    // Drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleImageSelection(e.dataTransfer.files[0], null);
        }
    });

    // Sample selector
    sampleSelect.addEventListener('change', (e) => {
        if (e.target.value) {
            handleImageSelection(null, e.target.value);
        }
    });

    // Run Imaging
    runBtn.addEventListener('click', executeImagingAnalysis);

    // Opacity slider
    opacitySlider.addEventListener('input', (e) => {
        const val = e.target.value;
        document.getElementById('opacityVal').textContent = `${val}%`;
        const heatmapImg = document.getElementById('cxrHeatmapImage');
        if (heatmapImg) {
            heatmapImg.style.opacity = val / 100;
        }
    });

    // Windowing Preset Handlers
    document.querySelectorAll('.win-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.win-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const win = btn.getAttribute('data-win');
            const baseImg = document.getElementById('cxrBaseImage');
            baseImg.className = '';
            if (win && win !== 'default') {
                baseImg.classList.add(`win-${win}`);
            }
        });
    });
}

function handleImageSelection(file, sampleName) {
    appState.selectedImageFile = file;
    appState.selectedSampleName = sampleName;
    document.getElementById('btnRunImaging').disabled = false;

    if (file) {
        document.getElementById('selectSampleImage').value = '';
        const reader = new FileReader();
        reader.onload = (e) => {
            showImageInViewer(e.target.result);
        };
        reader.readAsDataURL(file);
    } else if (sampleName) {
        loadSamplePreview(sampleName);
    }
}

function loadSamplePreview(sampleName) {
    showImageInViewer(`/data/sample_images/${sampleName}`);
    document.getElementById('btnRunImaging').disabled = false;
}

function showImageInViewer(src) {
    const baseImg = document.getElementById('cxrBaseImage');
    const heatmapImg = document.getElementById('cxrHeatmapImage');
    baseImg.src = src;
    heatmapImg.style.display = 'none';

    document.getElementById('emptyImageViewer').style.display = 'none';
    document.getElementById('imageCanvasWrapper').style.display = 'flex';
    document.getElementById('sliderWrapper').style.display = 'none';
}

async function executeImagingAnalysis() {
    const runBtn = document.getElementById('btnRunImaging');
    const spinner = document.getElementById('imagingSpinner');
    runBtn.disabled = true;
    spinner.style.display = 'inline-block';

    try {
        const formData = new FormData();
        if (appState.selectedImageFile) {
            formData.append('file', appState.selectedImageFile);
        } else if (appState.selectedSampleName) {
            formData.append('sample_name', appState.selectedSampleName);
        } else {
            showToast('Please select or upload a chest X-ray image.', 'error');
            return;
        }

        const res = await fetch('/api/imaging/analyze', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Imaging analysis failed.');
        }

        const data = await res.json();
        appState.imagingResult = data.analysis;
        appState.heatmapsByPathology = data.heatmaps_by_pathology || {};

        // Render base and heatmap
        if (data.original_image_base64) {
            document.getElementById('cxrBaseImage').src = data.original_image_base64;
        }
        if (data.heatmap_base64) {
            const heatmapImg = document.getElementById('cxrHeatmapImage');
            heatmapImg.src = data.heatmap_base64;
            heatmapImg.style.display = 'block';
            heatmapImg.style.opacity = '0.7';
            document.getElementById('sliderWrapper').style.display = 'flex';
        }

        // Show windowing toolbar
        document.getElementById('windowingToolbar').style.display = 'flex';

        // Render Multi-Pathology CAM focus chips
        const selectorRow = document.getElementById('pathologyCamSelector');
        const chipsRow = document.getElementById('pathologyChipsRow');
        if (data.heatmaps_by_pathology && Object.keys(data.heatmaps_by_pathology).length > 0) {
            selectorRow.style.display = 'flex';
            chipsRow.innerHTML = '';
            Object.keys(data.heatmaps_by_pathology).forEach((pName, idx) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = `cam-chip ${pName === data.analysis.top_finding ? 'active' : ''}`;
                btn.textContent = pName;
                btn.onclick = () => {
                    document.querySelectorAll('.cam-chip').forEach(c => c.classList.remove('active'));
                    btn.classList.add('active');
                    const heatmapImg = document.getElementById('cxrHeatmapImage');
                    heatmapImg.src = data.heatmaps_by_pathology[pName];
                    heatmapImg.style.display = 'block';
                };
                chipsRow.appendChild(btn);
            });
        }

        // Render Findings Card & Chart
        renderImagingFindings(data.analysis);
        showToast('DenseNet121 CXR Analysis & Grad-CAM Complete!');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        runBtn.disabled = false;
        spinner.style.display = 'none';
    }
}

function renderImagingFindings(analysis) {
    const card = document.getElementById('imagingFindingsCard');
    card.style.display = 'block';

    const metricsRow = document.getElementById('imagingSummaryMetrics');
    metricsRow.innerHTML = `
        <div class="metric-card cyan-border" style="flex:1;">
            <span class="metric-label">Top Detected Pathology</span>
            <span class="metric-value">${analysis.top_finding}</span>
            <span class="metric-caption">Confidence: ${(analysis.top_confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="metric-card emerald-border" style="flex:1;">
            <span class="metric-label">Model Architecture</span>
            <span class="metric-value">${analysis.model_name}</span>
            <span class="metric-caption">Grad-CAM Convolutional Heatmap</span>
        </div>
    `;

    document.getElementById('imagingLimitationsNote').innerHTML = `
        <strong>Clinical Disclaimer:</strong> ${analysis.limitations}
    `;

    // Render Chart.js Horizontal Bar Chart
    const labels = analysis.findings.map(f => f.finding);
    const confidences = analysis.findings.map(f => (f.confidence * 100).toFixed(1));

    const ctx = document.getElementById('imagingChart').getContext('2d');
    if (appState.imagingChart) {
        appState.imagingChart.destroy();
    }

    appState.imagingChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Model Confidence (%)',
                data: confidences,
                backgroundColor: 'rgba(6, 182, 212, 0.7)',
                borderColor: '#06b6d4',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f8fafc', font: { family: 'Outfit', size: 12 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

/* ==========================================================================
   Patient Risk Studio
   ========================================================================== */
function initRiskStudio() {
    populateRiskForm(appState.activePatient);
}

function populateRiskForm(p) {
    document.getElementById('rf_id').value = p.patient_id || 'PT-1001';
    document.getElementById('rf_age').value = p.age || 55;
    document.getElementById('rf_sex').value = p.sex || 'M';
    document.getElementById('rf_cp').value = p.chest_pain_type || 'ASY';
    document.getElementById('rf_bp').value = p.resting_bp || 135;
    document.getElementById('rf_chol').value = p.cholesterol || 230;
    document.getElementById('rf_fbs').value = p.fasting_bs !== undefined ? p.fasting_bs : 0;
    document.getElementById('rf_hr').value = p.max_hr || 145;
    document.getElementById('rf_ea').value = p.exercise_angina || 'N';
    document.getElementById('rf_ecg').value = p.resting_ecg || 'Normal';
    document.getElementById('rf_oldpeak').value = p.oldpeak !== undefined ? p.oldpeak : 1.0;
    document.getElementById('rf_slope').value = p.st_slope || 'Flat';
}

async function calculateRisk() {
    const spinner = document.getElementById('riskSpinner');
    spinner.style.display = 'inline-block';

    const patient = {
        patient_id: document.getElementById('rf_id').value,
        age: parseInt(document.getElementById('rf_age').value),
        sex: document.getElementById('rf_sex').value,
        chest_pain_type: document.getElementById('rf_cp').value,
        resting_bp: parseInt(document.getElementById('rf_bp').value),
        cholesterol: parseInt(document.getElementById('rf_chol').value),
        fasting_bs: parseInt(document.getElementById('rf_fbs').value),
        max_hr: parseInt(document.getElementById('rf_hr').value),
        exercise_angina: document.getElementById('rf_ea').value,
        resting_ecg: document.getElementById('rf_ecg').value,
        oldpeak: parseFloat(document.getElementById('rf_oldpeak').value),
        st_slope: document.getElementById('rf_slope').value
    };

    appState.activePatient = patient;
    updateSidebarCase();

    try {
        const res = await fetch('/api/risk/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patient)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Risk prediction failed.');
        }

        const data = await res.json();
        appState.riskResult = data;

        // Render Gauge & Features
        renderRiskResults(data);

        // Run Advanced Multi-Scores, Counterfactual Simulation & Safety
        computeActiveMultiScores(patient);
        initSimulationSliders(patient, data.risk_score * 100);
        loadPharmacotherapySafety(patient);

        showToast('Cardiovascular Risk Computed via Kaggle-Trained Model!');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        spinner.style.display = 'none';
    }
}

function renderRiskResults(data) {
    const prob = data.risk_score;
    const pct = Math.round(prob * 100);

    // Color based on category
    let color = '#10b981'; // Green
    if (data.risk_category === 'Moderate Risk') color = '#f59e0b';
    else if (data.risk_category === 'High Risk') color = '#f43f5e';

    // SVG arc stroke-dashoffset: circumference = 251.2
    const path = document.getElementById('gaugePath');
    path.style.stroke = color;
    path.style.strokeDashoffset = 251.2 * (1 - prob);

    const num = document.getElementById('gaugeNumber');
    num.textContent = `${pct}%`;
    num.style.fill = color;

    document.getElementById('gaugeCategory').textContent = data.risk_category.toUpperCase();

    // Meta Box
    document.getElementById('riskMetaBox').style.display = 'block';
    document.getElementById('rm_outcome').textContent = data.outcome;
    document.getElementById('rm_horizon').textContent = data.prediction_horizon;

    // Feature Table
    const tableWrapper = document.getElementById('contributingFeaturesWrapper');
    tableWrapper.style.display = 'block';
    const tbody = document.getElementById('featuresTableBody');
    tbody.innerHTML = '';

    data.contributing_factors.slice(0, 5).forEach(f => {
        const tr = document.createElement('tr');
        const badgeClass = f.direction === 'Increases Risk' ? 'badge-rose' : (f.direction === 'Decreases Risk' ? 'badge-emerald' : 'badge-cyan');
        tr.innerHTML = `
            <td><strong>${f.feature_name}</strong></td>
            <td><code>${f.feature_value}</code></td>
            <td><span class="badge ${badgeClass}">${f.direction}</span></td>
            <td>${(f.importance_weight * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

/* ==========================================================================
   Advanced Clinical Calculators & Multi-Score Panel
   ========================================================================== */
async function computeActiveMultiScores(patient = appState.activePatient) {
    try {
        const res = await fetch('/api/risk/calculators', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patient)
        });
        if (!res.ok) return;
        const scores = await res.json();
        appState.multiScores = scores;

        // ASCVD
        document.getElementById('sc_ascvd_badge').textContent = `${scores.ascvd.ten_year_risk_pct}%`;
        document.getElementById('sc_ascvd_val').textContent = `${scores.ascvd.ten_year_risk_pct}%`;
        document.getElementById('sc_ascvd_cat').textContent = scores.ascvd.risk_category;
        document.getElementById('sc_ascvd_rec').textContent = scores.ascvd.clinical_recommendation;

        // CURB-65
        document.getElementById('sc_curb_badge').textContent = `Score: ${scores.curb65.score}`;
        document.getElementById('sc_curb_val').textContent = `${scores.curb65.score} / 5`;
        document.getElementById('sc_curb_tier').textContent = `${scores.curb65.risk_tier} (30d Mort: ${scores.curb65.mortality_risk_pct}%)`;
        document.getElementById('sc_curb_site').textContent = scores.curb65.site_of_care_recommendation;

        // Wells PE
        document.getElementById('sc_wells_badge').textContent = `Score: ${scores.wells_pe.score}`;
        document.getElementById('sc_wells_val').textContent = `${scores.wells_pe.score} pts`;
        document.getElementById('sc_wells_tier').textContent = `${scores.wells_pe.risk_tier} (~${scores.wells_pe.probability_pct}%)`;
        document.getElementById('sc_wells_rec').textContent = scores.wells_pe.recommended_workup;

        // CHA2DS2-VASc
        document.getElementById('sc_cha_badge').textContent = `Score: ${scores.cha2ds2_vasc.score}`;
        document.getElementById('sc_cha_val').textContent = `${scores.cha2ds2_vasc.score} pts`;
        document.getElementById('sc_cha_cat').textContent = `Annual Stroke Risk: ${scores.cha2ds2_vasc.annual_stroke_risk_pct}%`;
        document.getElementById('sc_cha_rec').textContent = scores.cha2ds2_vasc.anticoagulation_recommendation;
    } catch (e) {
        console.warn('Error computing multi-scores:', e);
    }
}

/* ==========================================================================
   Interactive What-If Counterfactual Simulator
   ========================================================================== */
function initSimulationSliders(patient, baselineRiskPct) {
    document.getElementById('sim_sbp').value = Math.min(190, Math.max(100, patient.resting_bp || 135));
    document.getElementById('sim_sbp_val').textContent = `${patient.resting_bp || 135} mmHg`;
    document.getElementById('sim_chol').value = Math.min(350, Math.max(130, patient.cholesterol || 230));
    document.getElementById('sim_chol_val').textContent = `${patient.cholesterol || 230} mg/dL`;
    document.getElementById('sim_hr').value = Math.min(200, Math.max(80, patient.max_hr || 145));
    document.getElementById('sim_hr_val').textContent = `${patient.max_hr || 145} bpm`;
    document.getElementById('sim_fasting_bs').checked = (patient.fasting_bs === 0);
    document.getElementById('sim_angina_resolved').checked = (patient.exercise_angina === 'N');

    document.getElementById('sim_base_risk').textContent = `${baselineRiskPct.toFixed(1)}%`;
    runCounterfactualSimulation();
}

let simDebounceTimer = null;
function onSimulationSliderChange() {
    const sbp = document.getElementById('sim_sbp').value;
    const chol = document.getElementById('sim_chol').value;
    const hr = document.getElementById('sim_hr').value;
    document.getElementById('sim_sbp_val').textContent = `${sbp} mmHg`;
    document.getElementById('sim_chol_val').textContent = `${chol} mg/dL`;
    document.getElementById('sim_hr_val').textContent = `${hr} bpm`;

    clearTimeout(simDebounceTimer);
    simDebounceTimer = setTimeout(runCounterfactualSimulation, 150);
}

async function runCounterfactualSimulation() {
    try {
        const payload = {
            baseline_patient: appState.activePatient,
            target_resting_bp: parseInt(document.getElementById('sim_sbp').value),
            target_cholesterol: parseInt(document.getElementById('sim_chol').value),
            target_max_hr: parseInt(document.getElementById('sim_hr').value),
            target_fasting_bs: document.getElementById('sim_fasting_bs').checked ? 0 : 1,
            target_exercise_angina: document.getElementById('sim_angina_resolved').checked ? 'N' : 'Y'
        };

        const res = await fetch('/api/risk/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) return;
        const result = await res.json();

        document.getElementById('sim_base_risk').textContent = `${result.baseline_risk_pct}%`;
        document.getElementById('sim_target_risk').textContent = `${result.counterfactual_risk_pct}%`;
        document.getElementById('sim_arr_chip').textContent = `ARR: -${result.absolute_risk_reduction_pct}%`;
        document.getElementById('sim_rrr_chip').textContent = `RRR: -${result.relative_risk_reduction_pct}%`;
        document.getElementById('sim_nnt_chip').textContent = result.number_needed_to_treat ? `NNT: ${result.number_needed_to_treat}` : 'NNT: N/A';
        document.getElementById('sim_summary_text').textContent = result.interpretation;

        // Render interventions
        const wrapper = document.getElementById('simInterventionsWrapper');
        const list = document.getElementById('simInterventionsList');
        if (result.top_interventions && result.top_interventions.length > 0) {
            wrapper.style.display = 'block';
            list.innerHTML = '';
            result.top_interventions.forEach(item => {
                const div = document.createElement('div');
                div.className = 'intervention-item';
                div.innerHTML = `
                    <div class="intervention-item-header">
                        <span>${item.factor_name}</span>
                        <span class="intervention-gain">-${item.risk_reduction_pct}% Risk</span>
                    </div>
                    <div style="color:#94a3b8; font-size:0.75rem; margin-bottom:2px;">
                        Baseline: <code>${item.baseline_value}</code> &rarr; Target: <code>${item.target_value}</code>
                    </div>
                    <div class="intervention-note">${item.clinical_note}</div>
                `;
                list.appendChild(div);
            });
        } else {
            wrapper.style.display = 'none';
        }
    } catch (e) {
        console.warn('Simulation error:', e);
    }
}

/* ==========================================================================
   Pharmacotherapy Safety Guardrail Review
   ========================================================================== */
async function loadPharmacotherapySafety(patient = appState.activePatient) {
    try {
        const res = await fetch('/api/safety/review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patient)
        });
        if (!res.ok) return;
        const data = await res.json();
        appState.safetyReview = data;

        const badge = document.getElementById('safetyStatusBadge');
        if (data.has_critical_blocker) {
            badge.textContent = 'Critical Contraindication';
            badge.className = 'badge badge-rose';
        } else if (data.total_alerts > 0) {
            badge.textContent = `${data.total_alerts} Active Advisories`;
            badge.className = 'badge badge-amber';
        } else {
            badge.textContent = 'Standard Safety Verified';
            badge.className = 'badge badge-emerald';
        }

        const container = document.getElementById('safetyAlertsContainer');
        container.innerHTML = '';
        data.alerts.forEach(alert => {
            const card = document.createElement('div');
            const sevClass = alert.severity.includes('CRITICAL') ? 'critical' : (alert.severity.includes('HIGH') ? 'high' : 'advisory');
            const badgeClass = alert.severity.includes('CRITICAL') ? 'badge-rose' : (alert.severity.includes('HIGH') ? 'badge-amber' : 'badge-cyan');
            card.className = `safety-alert-card ${sevClass}`;
            card.innerHTML = `
                <div class="safety-card-header">
                    <span class="safety-drug-name">${alert.drug_class}</span>
                    <span class="badge ${badgeClass}">${alert.severity}</span>
                </div>
                <div class="safety-rationale">${alert.contraindication_rationale}</div>
                <div class="safety-source-rec">
                    <span><strong>Source:</strong> ${alert.guideline_source}</span>
                    <span><strong>Clinical Action:</strong> ${alert.recommended_alternative_action}</span>
                </div>
            `;
            container.appendChild(card);
        });
    } catch (e) {
        console.warn('Safety review error:', e);
    }
}

function resetRiskGauge() {
    const path = document.getElementById('gaugePath');
    path.style.strokeDashoffset = 251.2;
    document.getElementById('gaugeNumber').textContent = '--%';
    document.getElementById('gaugeNumber').style.fill = '#f8fafc';
    document.getElementById('gaugeCategory').textContent = 'Awaiting Input';
    document.getElementById('riskMetaBox').style.display = 'none';
    document.getElementById('contributingFeaturesWrapper').style.display = 'none';
}

/* ==========================================================================
   Guidelines RAG Studio
   ========================================================================== */
function initGuidelinesStudio() {
    // Enter key triggers search
    document.getElementById('guidelineQueryInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            askGuideline();
        }
    });
}

function fillQuery(text) {
    document.getElementById('guidelineQueryInput').value = text;
    askGuideline();
}

async function askGuideline() {
    const input = document.getElementById('guidelineQueryInput');
    const query = input.value.trim();
    if (!query) {
        showToast('Please enter a clinical question.', 'error');
        return;
    }

    const spinner = document.getElementById('ragSpinner');
    const btn = document.getElementById('btnAskGuideline');
    spinner.style.display = 'inline-block';
    btn.disabled = true;

    const filterVal = document.getElementById('guidelineFilter').value;
    const topK = parseInt(document.getElementById('guidelineTopK').value);

    try {
        const res = await fetch('/api/guidelines/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                top_k: topK,
                document_filter: filterVal || null
            })
        });

        if (!res.ok) throw new Error('Guidelines query failed.');
        const answer = await res.json();
        
        appState.guidelineAnswers.push(answer);
        renderGuidelineAnswer(answer);
        showToast('Grounded Guideline Evidence Retrieved!');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        spinner.style.display = 'none';
        btn.disabled = false;
    }
}

function renderGuidelineAnswer(answer) {
    const card = document.getElementById('ragResultCard');
    card.style.display = 'block';

    const alert = document.getElementById('insufficientAlert');
    const answerBox = document.getElementById('synthesizedAnswerBox');
    const citationsContainer = document.getElementById('citationsContainer');

    if (!answer.evidence_sufficient) {
        alert.style.display = 'flex';
        document.getElementById('insufficientText').textContent = answer.answer;
        answerBox.style.display = 'none';
        citationsContainer.innerHTML = '';
        return;
    }

    alert.style.display = 'none';
    answerBox.style.display = 'block';
    answerBox.textContent = answer.answer;
    document.getElementById('ragModelBadge').textContent = answer.generator_model;

    // Render Citations
    citationsContainer.innerHTML = '';
    answer.citations.forEach(c => {
        const cCard = document.createElement('div');
        cCard.className = 'citation-card';
        cCard.innerHTML = `
            <div class="citation-header">
                <h5>[${c.citation_id}] ${c.document_title} — Page ${c.page_number || 'N/A'}</h5>
                <span class="badge badge-cyan">Sim: ${c.similarity_score.toFixed(3)}</span>
            </div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:4px;">Section: ${c.section_heading || 'General Guidance'}</div>
            <div class="citation-snippet">"${c.snippet}"</div>
        `;
        citationsContainer.appendChild(cCard);
    });
}

/* ==========================================================================
   Unified Case Review Workspace
   ========================================================================== */
async function refreshUnifiedReview() {
    const payload = {
        case_id: appState.activePatient.patient_id || 'CASE-WEB-DEMO',
        patient_input: appState.activePatient,
        imaging_result: appState.imagingResult,
        risk_result: appState.riskResult,
        guideline_answers: appState.guidelineAnswers
    };

    try {
        const res = await fetch('/api/orchestration/review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) return;
        const summary = await res.json();
        window.currentSummary = summary;

        document.getElementById('reviewCaseTitle').textContent = `Case Summary: ${summary.case_id}`;
        document.getElementById('reviewTimestamp').textContent = `Review Timestamp: ${summary.timestamp}`;
        
        const badge = document.getElementById('reviewStatusBadge');
        badge.textContent = summary.overall_status;
        badge.className = `badge ${summary.overall_status === 'COMPLETE' ? 'badge-emerald' : 'badge-amber'}`;

        // Render Clinician Attestation Stamp if attested
        const stampBox = document.getElementById('attestationStampBox');
        if (appState.attestation) {
            stampBox.style.display = 'flex';
            document.getElementById('stampStatus').textContent = `CLINICAL ATTESTATION: ${appState.attestation.decision_status.toUpperCase()}`;
            document.getElementById('stampAuditId').textContent = appState.attestation.audit_stamp;
            document.getElementById('stampDetails').textContent = `Attending: ${appState.attestation.clinician_name} | ${appState.attestation.npi_or_license} | ${appState.attestation.institution} | Signed: ${new Date(appState.attestation.signed_at).toLocaleString()}`;
        } else {
            stampBox.style.display = 'none';
        }

        // Render Multimodal Differential Diagnosis (DDx)
        if (summary.differential_diagnosis) {
            const ddx = summary.differential_diagnosis;
            appState.differentialDiagnosis = ddx;

            // Red Flags
            const redFlagsBox = document.getElementById('redFlagsContainer');
            if (ddx.red_flag_alerts && ddx.red_flag_alerts.length > 0) {
                redFlagsBox.style.display = 'flex';
                redFlagsBox.innerHTML = '';
                ddx.red_flag_alerts.forEach(flag => {
                    const d = document.createElement('div');
                    d.className = 'red-flag-alert';
                    d.innerHTML = `<span>🚨</span><span><strong>Clinical Warning:</strong> ${flag}</span>`;
                    redFlagsBox.appendChild(d);
                });
            } else {
                redFlagsBox.style.display = 'none';
            }

            // Primary Diagnosis
            if (ddx.primary_diagnosis) {
                document.getElementById('ddxPrimaryName').textContent = ddx.primary_diagnosis.condition_name;
                document.getElementById('ddxIcdCode').textContent = `ICD-10: ${ddx.primary_diagnosis.icd10_code}`;
                document.getElementById('ddxProbEst').textContent = `${ddx.primary_diagnosis.probability_estimate_pct}% Probability`;
                document.getElementById('ddxPrimaryPatho').textContent = ddx.primary_diagnosis.pathophysiological_summary;

                const suppBox = document.getElementById('ddxSupportingBox');
                const suppList = document.getElementById('ddxSupportingList');
                if (ddx.primary_diagnosis.supporting_findings && ddx.primary_diagnosis.supporting_findings.length > 0) {
                    suppBox.style.display = 'block';
                    suppList.innerHTML = '';
                    ddx.primary_diagnosis.supporting_findings.forEach(sf => {
                        const li = document.createElement('li');
                        li.textContent = sf;
                        suppList.appendChild(li);
                    });
                } else {
                    suppBox.style.display = 'none';
                }
            }

            // Rival Diagnoses
            const rivalList = document.getElementById('ddxRivalList');
            rivalList.innerHTML = '';
            if (ddx.secondary_diagnoses && ddx.secondary_diagnoses.length > 0) {
                ddx.secondary_diagnoses.forEach(r => {
                    const item = document.createElement('div');
                    item.className = 'rival-item';
                    item.innerHTML = `
                        <div class="item-top-row">
                            <span>${r.condition_name}</span>
                            <span class="badge badge-purple">${r.icd10_code} (${r.probability_estimate_pct}%)</span>
                        </div>
                        <div class="item-desc">${r.pathophysiological_summary}</div>
                    `;
                    rivalList.appendChild(item);
                });
            } else {
                rivalList.innerHTML = '<p class="text-secondary" style="font-size:0.85rem;">No rival conditions currently flagged.</p>';
            }

            // Ruled-out conditions
            const ruledList = document.getElementById('ddxRuledOutList');
            ruledList.innerHTML = '';
            if (ddx.ruled_out_conditions && ddx.ruled_out_conditions.length > 0) {
                ddx.ruled_out_conditions.forEach(ro => {
                    const item = document.createElement('div');
                    item.className = 'ruled-out-item';
                    item.innerHTML = `
                        <div class="item-top-row">
                            <span>${ro.condition_name}</span>
                            <span class="badge badge-cyan">${ro.icd10_code}</span>
                        </div>
                        <div class="item-desc">${ro.rationale}</div>
                    `;
                    ruledList.appendChild(item);
                });
            } else {
                ruledList.innerHTML = '<p class="text-secondary" style="font-size:0.85rem;">None ruled out.</p>';
            }

            // Workup Orders
            const workupCard = document.getElementById('workupCard');
            const workupTbody = document.getElementById('workupTableBody');
            if (ddx.recommended_workup && ddx.recommended_workup.length > 0) {
                workupCard.style.display = 'block';
                workupTbody.innerHTML = '';
                ddx.recommended_workup.forEach(w => {
                    const tr = document.createElement('tr');
                    const badgeClass = w.urgency.includes('STAT') ? 'badge-rose' : (w.urgency.includes('Urgent') ? 'badge-amber' : 'badge-cyan');
                    tr.innerHTML = `
                        <td><strong>${w.test_name}</strong></td>
                        <td>${w.category}</td>
                        <td><span class="badge ${badgeClass}">${w.urgency}</span></td>
                        <td>${w.clinical_justification}</td>
                    `;
                    workupTbody.appendChild(tr);
                });
            } else {
                workupCard.style.display = 'none';
            }
        }

        // Correlation Alerts
        const alertBox = document.getElementById('correlationAlertBox');
        if (summary.clinical_synthesis_notes.length > 0) {
            alertBox.style.display = 'flex';
            document.getElementById('correlationAlertText').innerHTML = `
                <strong>Multimodal Correlation Finding:</strong> ${summary.clinical_synthesis_notes[0]}
            `;
        } else {
            alertBox.style.display = 'none';
        }

        // Provenance Table
        const tbody = document.getElementById('provenanceTableBody');
        tbody.innerHTML = '';
        summary.provenance_table.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><span class="badge badge-cyan">${row.source_module}</span></td>
                <td><strong>${row.element_name}</strong></td>
                <td>${row.value}</td>
                <td><code>${row.confidence_or_status}</code></td>
                <td><span style="font-size:0.75rem; color:#94a3b8;">${row.provenance_disclaimer}</span></td>
            `;
            tbody.appendChild(tr);
        });

        // Notes and Uncertainties
        const notesList = document.getElementById('synthesisNotesList');
        notesList.innerHTML = '';
        if (summary.clinical_synthesis_notes.length > 0) {
            summary.clinical_synthesis_notes.forEach(n => {
                const li = document.createElement('li');
                li.textContent = n;
                notesList.appendChild(li);
            });
        } else {
            notesList.innerHTML = '<li>Run Medical Imaging and Risk Model to generate synthesis notes.</li>';
        }

        const uncList = document.getElementById('uncertaintiesList');
        uncList.innerHTML = '';
        if (summary.unresolved_uncertainties.length > 0) {
            summary.unresolved_uncertainties.forEach(u => {
                const li = document.createElement('li');
                li.textContent = u;
                uncList.appendChild(li);
            });
        } else {
            uncList.innerHTML = '<li>No unresolved diagnostic uncertainties currently flagged.</li>';
        }

    } catch (e) {
        console.error('Error refreshing review:', e);
    }
}

/* ==========================================================================
   FHIR R4 Export & Clinician Attestation
   ========================================================================== */
async function exportFHIRBundle() {
    const payload = {
        case_id: appState.activePatient.patient_id || 'PT-1001',
        patient_input: appState.activePatient,
        imaging_result: appState.imagingResult,
        risk_result: appState.riskResult,
        guideline_answers: appState.guidelineAnswers
    };

    try {
        const res = await fetch('/api/export/fhir', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('FHIR export failed.');
        const bundle = await res.json();
        const jsonStr = JSON.stringify(bundle, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `case_${payload.case_id.toLowerCase()}_fhir_r4.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('HL7 FHIR Release 4 JSON Bundle exported successfully!');
    } catch (e) {
        showToast(e.message, 'error');
    }
}

function openAttestationModal() {
    document.getElementById('attestCaseIdDisplay').textContent = appState.activePatient.patient_id || 'PT-1001';
    document.getElementById('attestationModal').style.display = 'flex';
}

function closeAttestationModal() {
    document.getElementById('attestationModal').style.display = 'none';
}

async function submitAttestation() {
    const caseId = appState.activePatient.patient_id || 'PT-1001';
    const payload = {
        case_id: caseId,
        clinician_name: document.getElementById('att_name').value,
        npi_or_license: document.getElementById('att_npi').value,
        institution: document.getElementById('att_inst').value,
        decision_status: document.getElementById('att_status').value,
        notes: document.getElementById('att_notes').value
    };

    try {
        const res = await fetch('/api/attestation/sign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Attestation failed.');
        const data = await res.json();
        appState.attestation = data;

        closeAttestationModal();

        // Show attestation stamp in review
        const stampBox = document.getElementById('attestationStampBox');
        stampBox.style.display = 'flex';
        document.getElementById('stampStatus').textContent = `CLINICAL ATTESTATION: ${data.decision_status.toUpperCase()}`;
        document.getElementById('stampAuditId').textContent = data.audit_stamp;
        document.getElementById('stampDetails').textContent = `Attending: ${data.clinician_name} | ${data.npi_or_license} | ${data.institution} | Signed: ${new Date(data.signed_at).toLocaleString()}`;

        showToast('Formal Attestation & Cryptographic Audit Stamp Recorded!');
    } catch (e) {
        showToast(e.message, 'error');
    }
}

function exportCaseReport() {
    if (!window.currentSummary) {
        showToast('Please open the Case Review tab to generate the report first.', 'error');
        return;
    }

    const s = window.currentSummary;
    const text = `================================================================================
CLINICAL COPILOT MULTIMODAL CASE REVIEW
Case ID: ${s.case_id}
Date: ${s.timestamp}
Overall Status: ${s.overall_status}
================================================================================

1. PATIENT DEMOGRAPHICS & CLINICAL INTAKE
Age: ${s.patient_input ? s.patient_input.age : 'N/A'}, Sex: ${s.patient_input ? s.patient_input.sex : 'N/A'}, Chest Pain: ${s.patient_input ? s.patient_input.chest_pain_type : 'N/A'}
Resting BP: ${s.patient_input ? s.patient_input.resting_bp + ' mmHg' : 'N/A'}, Cholesterol: ${s.patient_input ? s.patient_input.cholesterol + ' mg/dL' : 'N/A'}

2. MODULE A: MEDICAL IMAGING INTELLIGENCE (CXR)
Model: ${s.imaging_result ? s.imaging_result.model_name : 'No imaging conducted'}
Top Pathology Finding: ${s.imaging_result ? s.imaging_result.top_finding + ' (' + (s.imaging_result.top_confidence * 100).toFixed(1) + '%)' : 'N/A'}

3. MODULE B: PATIENT RISK MODEL
Model: ${s.risk_result ? s.risk_result.model_name : 'No risk score computed'}
Predicted Risk: ${s.risk_result ? s.risk_result.risk_category + ' (' + (s.risk_result.risk_score * 100).toFixed(1) + '%)' : 'N/A'}
Training Cohort: ${s.risk_result ? s.risk_result.dataset_source : 'N/A'}

4. CLINICAL SYNTHESIS & MULTIMODAL CORRELATIONS
${s.clinical_synthesis_notes.map(n => '- ' + n).join('\n')}

5. DOCUMENTED UNCERTAINTIES & WARNINGS
${s.unresolved_uncertainties.map(u => '- ' + u).join('\n')}

================================================================================
CLINICAL GOVERNANCE STATEMENT:
${s.governance_statement}
================================================================================`;

    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clinical_copilot_case_${s.case_id.toLowerCase()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Structured Case Report downloaded.');
}

/* ==========================================================================
   Toast Notification Helper
   ========================================================================== */
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = 'toast';
    if (type === 'error') {
        toast.style.borderLeftColor = '#f43f5e';
    }
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
