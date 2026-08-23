/**
 * AgriVision AI - Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const fileMeta = document.getElementById('file-meta');
    const btnClearImg = document.getElementById('btn-clear-img');
    const btnAnalyze = document.getElementById('btn-analyze');
    const loadingOverlay = document.getElementById('loading-overlay');

    // Tabs
    const tabUpload = document.getElementById('tab-upload');
    const tabCamera = document.getElementById('tab-camera');
    const tabSamples = document.getElementById('tab-samples');
    const viewUpload = document.getElementById('view-upload');
    const viewCamera = document.getElementById('view-camera');
    const viewSamples = document.getElementById('view-samples');
    const sampleGrid = document.getElementById('sample-grid');

    // Camera
    const cameraFeed = document.getElementById('camera-feed');
    const cameraCanvas = document.getElementById('camera-canvas');
    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnCapturePhoto = document.getElementById('btn-capture-photo');
    let cameraStream = null;

    // Results UI
    const emptyState = document.getElementById('empty-state');
    const resultsContent = document.getElementById('results-content');
    const primaryCard = document.getElementById('primary-diagnosis-card');
    const resPlantName = document.getElementById('res-plant-name');
    const resDiseaseName = document.getElementById('res-disease-name');
    const resStatusTag = document.getElementById('res-status-tag');
    const resStatusText = document.getElementById('res-status-text');
    const resConfidenceVal = document.getElementById('res-confidence-val');
    const resConfidenceBar = document.getElementById('res-confidence-bar');
    const resCauseVal = document.getElementById('res-cause-val');
    const resSeverityVal = document.getElementById('res-severity-val');
    const topkList = document.getElementById('topk-list');
    const resSymptoms = document.getElementById('res-symptoms');
    const resOrganic = document.getElementById('res-organic');
    const resChemical = document.getElementById('res-chemical');
    const resPrevention = document.getElementById('res-prevention');
    const reportTimestamp = document.getElementById('report-timestamp');

    // Grad-CAM UI elements
    const gradcamSection    = document.getElementById('gradcam-section');
    const gradcamStatusBadge= document.getElementById('gradcam-status-badge');
    const gradcamErrorNotice= document.getElementById('gradcam-error-notice');
    const gradcamErrorText  = document.getElementById('gradcam-error-text');
    const gradcamOriginal   = document.getElementById('gradcam-original');
    const gradcamHeatmap    = document.getElementById('gradcam-heatmap');
    const gradcamOverlay    = document.getElementById('gradcam-overlay');
    const gradcamLayerName  = document.getElementById('gradcam-layer-name');
    const gradcamGrid       = document.getElementById('gradcam-grid');

    // Encyclopedia Modal
    const btnEncyclopedia = document.getElementById('btn-encyclopedia');
    const encyclopediaModal = document.getElementById('encyclopedia-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const encyclopediaSearch = document.getElementById('encyclopedia-search');
    const encyclopediaList = document.getElementById('encyclopedia-list');
    const filterPills = document.querySelectorAll('.filter-pill');

    // State Variables
    let selectedFile = null;
    let selectedBase64 = null;
    let allEncyclopediaData = [];
    let currentFilter = 'all';

    // Sample Images Definition
    const sampleImages = [
        {
            title: "Tomato Early Blight",
            plant: "Tomato",
            filename: "tomato_early_blight.jpg",
            class_name: "tomato_early_blight"
        },
        {
            title: "Healthy Apple Leaf",
            plant: "Apple",
            filename: "healthy_apple.jpg",
            class_name: "healthy_apple"
        },
        {
            title: "Corn Common Rust",
            plant: "Corn",
            filename: "corn_common_rust.jpg",
            class_name: "corn_common_rust"
        },
        {
            title: "Lemon Citrus Canker",
            plant: "Lemon",
            filename: "lemon_citrus_canker.jpg",
            class_name: "lemon_citrus_canker"
        },
        {
            title: "Potato Late Blight",
            plant: "Potato",
            filename: "potato_late_blight.jpg",
            class_name: "potato_late_blight"
        },
        {
            title: "Wheat Yellow Rust",
            plant: "Wheat",
            filename: "wheat_yellow_rust.jpg",
            class_name: "wheat_yellow_rust"
        }
    ];

    // =========================================================================
    // Initialization & Sample Setup
    // =========================================================================

    function initSamples() {
        sampleGrid.innerHTML = '';
        sampleImages.forEach(sample => {
            const card = document.createElement('div');
            card.className = 'sample-card';
            card.innerHTML = `
                <div class="sample-img-wrap">
                    <img src="/static/samples/${sample.filename}" alt="${sample.title}" onerror="this.src='/static/samples/placeholder.png'">
                </div>
                <div class="sample-title">${sample.title}</div>
            `;
            card.addEventListener('click', () => loadSampleImage(sample));
            sampleGrid.appendChild(card);
        });
    }

    async function loadSampleImage(sample) {
        try {
            showLoading(true);
            const response = await fetch(`/static/samples/${sample.filename}`);
            if (!response.ok) throw new Error("Sample image not found on server");
            const blob = await response.blob();
            const file = new File([blob], sample.filename, { type: "image/jpeg" });
            handleFileSelect(file);
            showToast(`Loaded sample: ${sample.title}`, 'success');
        } catch (e) {
            console.error(e);
            showToast("Failed to load sample leaf.", "error");
        } finally {
            showLoading(false);
        }
    }

    // =========================================================================
    // Tab Navigation
    // =========================================================================

    function switchTab(activeTab) {
        [tabUpload, tabCamera, tabSamples].forEach(t => t.classList.remove('active'));
        [viewUpload, viewCamera, viewSamples].forEach(v => v.classList.add('hidden'));

        if (activeTab === 'upload') {
            tabUpload.classList.add('active');
            viewUpload.classList.remove('hidden');
            stopCamera();
        } else if (activeTab === 'camera') {
            tabCamera.classList.add('active');
            viewCamera.classList.remove('hidden');
        } else if (activeTab === 'samples') {
            tabSamples.classList.add('active');
            viewSamples.classList.remove('hidden');
            stopCamera();
        }
    }

    tabUpload.addEventListener('click', () => switchTab('upload'));
    tabCamera.addEventListener('click', () => switchTab('camera'));
    tabSamples.addEventListener('click', () => switchTab('samples'));

    // =========================================================================
    // File Drag & Drop and Upload
    // =========================================================================

    btnBrowse.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            showToast("Please upload an image file (JPEG, PNG, WEBP).", "error");
            return;
        }

        selectedFile = file;
        selectedBase64 = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            fileMeta.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            previewContainer.classList.remove('hidden');
            previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };
        reader.readAsDataURL(file);
    }

    btnClearImg.addEventListener('click', () => {
        selectedFile = null;
        selectedBase64 = null;
        previewImg.src = '';
        fileInput.value = '';
        previewContainer.classList.add('hidden');
    });

    // =========================================================================
    // Live Camera Feed & Capture
    // =========================================================================

    btnStartCamera.addEventListener('click', async () => {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            cameraFeed.srcObject = cameraStream;
            btnStartCamera.classList.add('hidden');
            btnCapturePhoto.classList.remove('hidden');
            showToast("Camera activated. Point at plant leaf.", "success");
        } catch (err) {
            console.error("Camera access failed:", err);
            showToast("Camera access denied or unavailable.", "error");
        }
    });

    btnCapturePhoto.addEventListener('click', () => {
        if (!cameraStream) return;

        cameraCanvas.width = cameraFeed.videoWidth || 640;
        cameraCanvas.height = cameraFeed.videoHeight || 480;
        const ctx = cameraCanvas.getContext('2d');
        ctx.drawImage(cameraFeed, 0, 0, cameraCanvas.width, cameraCanvas.height);

        const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.92);
        previewImg.src = dataUrl;
        selectedBase64 = dataUrl;
        selectedFile = null;
        fileMeta.textContent = `Live Camera Capture (${cameraCanvas.width}x${cameraCanvas.height})`;
        previewContainer.classList.remove('hidden');

        stopCamera();
        switchTab('upload');
        showToast("Leaf captured! Click 'Run AI Diagnosis'", "success");
    });

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
            cameraFeed.srcObject = null;
            btnStartCamera.classList.remove('hidden');
            btnCapturePhoto.classList.add('hidden');
        }
    }

    // =========================================================================
    // Run AI Diagnosis & Prediction
    // =========================================================================

    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile && !selectedBase64) {
            showToast("Please select or capture a plant leaf image first.", "error");
            return;
        }

        try {
            showLoading(true);
            let responseData;

            if (selectedFile) {
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('top_k', '5');

                const response = await fetch('/predict-with-gradcam', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || "Diagnosis failed on server");
                }
                responseData = await response.json();
            } else if (selectedBase64) {
                const response = await fetch('/predict-base64', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: selectedBase64, top_k: 5 })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || "Diagnosis failed on server");
                }
                responseData = await response.json();
            }

            renderDiagnosisResults(responseData);
            showToast("Diagnosis completed successfully!", "success");

        } catch (error) {
            console.error("Diagnosis error:", error);
            showToast(error.message || "Failed to process image.", "error");
        } finally {
            showLoading(false);
        }
    });

    function renderDiagnosisResults(data) {
        if (!data || !data.prediction) return;

        const pred = data.prediction;
        const isHealthy = pred.is_healthy;

        emptyState.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        // Plant & Condition
        resPlantName.textContent = pred.plant;
        resDiseaseName.textContent = pred.condition;

        // Health Status Tag & Theme
        if (isHealthy) {
            primaryCard.className = 'diagnosis-card';
            resStatusTag.className = 'status-indicator healthy';
            resStatusTag.innerHTML = '<i class="fa-solid fa-circle-check"></i> <span>Healthy</span>';
            resSeverityVal.className = 'meta-value severity-badge severity-none';
        } else {
            primaryCard.className = 'diagnosis-card infected';
            resStatusTag.className = 'status-indicator infected';
            resStatusTag.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> <span>Infected</span>';
            
            const sev = (pred.severity || 'Medium').toLowerCase();
            resSeverityVal.className = `meta-value severity-badge severity-${sev}`;
        }

        // Confidence
        resConfidenceVal.textContent = `${pred.confidence}%`;
        resConfidenceBar.style.width = `${pred.confidence}%`;

        // Metadata
        resCauseVal.textContent = pred.cause || 'N/A';
        resSeverityVal.textContent = pred.severity || 'None';

        // Top 5 Probabilities
        topkList.innerHTML = '';
        if (data.top_k_predictions && data.top_k_predictions.length > 0) {
            data.top_k_predictions.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'topk-item';
                itemDiv.innerHTML = `
                    <div class="topk-row">
                        <span class="topk-name">
                            <span class="topk-rank">#${item.rank}</span>
                            ${item.display_name}
                        </span>
                        <span class="topk-pct">${item.confidence}%</span>
                    </div>
                    <div class="topk-bar-track">
                        <div class="topk-bar-fill" style="width: ${item.confidence}%"></div>
                    </div>
                `;
                topkList.appendChild(itemDiv);
            });
        }

        // Prescription Sections
        resSymptoms.textContent = pred.symptoms || "No specific symptoms recorded.";
        resOrganic.textContent = pred.organic_treatment || "No specific organic treatment required.";
        resChemical.textContent = pred.chemical_treatment || "No chemical application required.";
        resPrevention.textContent = pred.prevention || "Maintain standard clean cultivation practices.";

        reportTimestamp.textContent = new Date().toLocaleTimeString();

        // Render Grad-CAM section if data is present
        if (data.gradcam) {
            renderGradcam(data.gradcam);
        } else {
            // Hide gradcam section if gradcam data is missing
            gradcamSection.style.display = 'none';
        }

        resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // =========================================================================
    // Grad-CAM Renderer
    // =========================================================================

    function renderGradcam(gcam) {
        // Always show the section once a prediction comes in
        gradcamSection.style.display = 'block';

        // Reset state
        gradcamErrorNotice.classList.add('hidden');
        gradcamGrid.style.display = 'grid';

        if (gcam.target_layer) {
            gradcamLayerName.textContent = gcam.target_layer;
        }

        if (gcam.success) {
            // ── Success ──
            gradcamStatusBadge.className = 'gradcam-status-badge success';
            gradcamStatusBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> Grad-CAM Ready';

            gradcamOriginal.src = gcam.original_image || '';
            gradcamHeatmap.src  = gcam.heatmap        || '';
            gradcamOverlay.src  = gcam.overlay         || '';

        } else {
            // ── Failure (prediction still shown; gradcam section shows error) ──
            gradcamStatusBadge.className = 'gradcam-status-badge error';
            gradcamStatusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Grad-CAM Error';

            gradcamErrorNotice.classList.remove('hidden');
            gradcamErrorText.textContent = gcam.error || 'Grad-CAM could not be generated.';

            // Clear images
            gradcamOriginal.src = '';
            gradcamHeatmap.src  = '';
            gradcamOverlay.src  = '';
            gradcamGrid.style.display = 'none';
        }
    }

    // =========================================================================
    // 86 Disease Encyclopedia Modal
    // =========================================================================

    btnEncyclopedia.addEventListener('click', async () => {
        encyclopediaModal.classList.remove('hidden');
        if (allEncyclopediaData.length === 0) {
            await fetchEncyclopedia();
        }
    });

    btnCloseModal.addEventListener('click', () => encyclopediaModal.classList.add('hidden'));

    encyclopediaModal.addEventListener('click', (e) => {
        if (e.target === encyclopediaModal) {
            encyclopediaModal.classList.add('hidden');
        }
    });

    async function fetchEncyclopedia() {
        try {
            const res = await fetch('/api/classes');
            const data = await res.json();
            allEncyclopediaData = data.classes;
            renderEncyclopediaList();
        } catch (e) {
            console.error("Failed to load encyclopedia:", e);
        }
    }

    function renderEncyclopediaList() {
        const query = (encyclopediaSearch.value || '').toLowerCase().trim();
        encyclopediaList.innerHTML = '';

        const filtered = allEncyclopediaData.filter(item => {
            const matchesQuery = item.display_name.toLowerCase().includes(query) ||
                                 item.plant.toLowerCase().includes(query) ||
                                 item.condition.toLowerCase().includes(query);
            if (!matchesQuery) return false;

            if (currentFilter === 'healthy') return item.is_healthy;
            if (currentFilter === 'diseased') return !item.is_healthy;
            return true;
        });

        if (filtered.length === 0) {
            encyclopediaList.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 2rem;">No matching plant diseases found.</div>';
            return;
        }

        filtered.forEach(item => {
            const card = document.createElement('div');
            card.className = 'encyclopedia-card';
            card.style.cursor = 'pointer';
            card.innerHTML = `
                <div class="enc-top">
                    <div>
                        <div class="enc-plant">${item.plant}</div>
                        <div class="enc-title">${item.condition}</div>
                    </div>
                    <span class="enc-status ${item.is_healthy ? 'healthy' : 'infected'}">
                        ${item.is_healthy ? 'Healthy' : 'Diseased'}
                    </span>
                </div>
                <div class="enc-desc">Trained neural class identifier: <code>${item.raw_class}</code></div>
            `;
            
            card.addEventListener('click', async () => {
                // Highlight active card
                document.querySelectorAll('.encyclopedia-card').forEach(c => c.style.borderColor = 'var(--card-border)');
                card.style.borderColor = 'var(--emerald-400)';
                
                const detailContainer = document.getElementById('encyclopedia-detail');
                const title = document.getElementById('enc-detail-title');
                const symptoms = document.getElementById('enc-detail-symptoms');
                const organic = document.getElementById('enc-detail-organic');
                const chemical = document.getElementById('enc-detail-chemical');
                const prevention = document.getElementById('enc-detail-prevention');
                
                detailContainer.style.display = 'block';
                title.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading details...';
                symptoms.textContent = '';
                organic.textContent = '';
                chemical.textContent = '';
                prevention.textContent = '';
                
                try {
                    const res = await fetch(`/api/class-info/${item.raw_class}`);
                    if (!res.ok) throw new Error('Failed to fetch class info');
                    const info = await res.json();
                    
                    title.textContent = item.display_name;
                    symptoms.textContent = info.symptoms || 'Information not available.';
                    organic.textContent = info.organic_treatment || 'Information not available.';
                    chemical.textContent = info.chemical_treatment || 'Information not available.';
                    prevention.textContent = info.prevention || 'Information not available.';
                } catch (e) {
                    console.error("Error fetching class info:", e);
                    title.textContent = item.display_name;
                    symptoms.textContent = "Error loading details. Please try again.";
                }
            });

            encyclopediaList.appendChild(card);
        });
    }

    encyclopediaSearch.addEventListener('input', renderEncyclopediaList);

    filterPills.forEach(pill => {
        pill.addEventListener('click', () => {
            filterPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentFilter = pill.getAttribute('data-filter');
            renderEncyclopediaList();
        });
    });

    // =========================================================================
    // UI Helpers & Toasts
    // =========================================================================

    function showLoading(show) {
        if (show) {
            loadingOverlay.classList.remove('hidden');
        } else {
            loadingOverlay.classList.add('hidden');
        }
    }

    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = 'fa-circle-info';
        if (type === 'success') icon = 'fa-circle-check';
        if (type === 'error') icon = 'fa-triangle-exclamation';

        toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3800);
    }

    // Initialize sample gallery
    initSamples();
});
