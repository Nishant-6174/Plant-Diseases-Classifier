/**
 * AgriVision AI - Frontend Application Logic
 * ---------------------------------------------------------
 * Supports:
 *  - Image upload
 *  - Drag & drop
 *  - Live camera capture
 *  - Sample images
 *  - AI prediction
 *  - Top-K predictions
 *  - Grad-CAM explainability
 *  - Disease encyclopedia
 *  - Toast notifications
 */

document.addEventListener('DOMContentLoaded', () => {

    // ============================================================
    // DOM ELEMENTS
    // ============================================================

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const fileMeta = document.getElementById('file-meta');
    const btnClearImg = document.getElementById('btn-clear-img');
    const btnAnalyze = document.getElementById('btn-analyze');
    const loadingOverlay = document.getElementById('loading-overlay');

    // ============================================================
    // TABS
    // ============================================================

    const tabUpload = document.getElementById('tab-upload');
    const tabCamera = document.getElementById('tab-camera');
    const tabSamples = document.getElementById('tab-samples');

    const viewUpload = document.getElementById('view-upload');
    const viewCamera = document.getElementById('view-camera');
    const viewSamples = document.getElementById('view-samples');

    const sampleGrid = document.getElementById('sample-grid');

    // ============================================================
    // CAMERA
    // ============================================================

    const cameraFeed = document.getElementById('camera-feed');
    const cameraCanvas = document.getElementById('camera-canvas');
    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnCapturePhoto = document.getElementById('btn-capture-photo');

    let cameraStream = null;

    // ============================================================
    // RESULTS UI
    // ============================================================

    const emptyState = document.getElementById('empty-state');
    const resultsContent = document.getElementById('results-content');

    const primaryCard = document.getElementById('primary-diagnosis-card');

    const resPlantName = document.getElementById('res-plant-name');
    const resDiseaseName = document.getElementById('res-disease-name');
    const resStatusTag = document.getElementById('res-status-tag');
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

    // ============================================================
    // GRAD-CAM UI
    // ============================================================

    const gradcamSection = document.getElementById('gradcam-section');
    const gradcamStatusBadge = document.getElementById('gradcam-status-badge');
    const gradcamErrorNotice = document.getElementById('gradcam-error-notice');
    const gradcamErrorText = document.getElementById('gradcam-error-text');

    const gradcamOriginal = document.getElementById('gradcam-original');
    const gradcamHeatmap = document.getElementById('gradcam-heatmap');
    const gradcamOverlay = document.getElementById('gradcam-overlay');

    const gradcamLayerName = document.getElementById('gradcam-layer-name');
    const gradcamGrid = document.getElementById('gradcam-grid');

    // ============================================================
    // ENCYCLOPEDIA
    // ============================================================

    const btnEncyclopedia = document.getElementById('btn-encyclopedia');
    const encyclopediaModal = document.getElementById('encyclopedia-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');

    const encyclopediaSearch = document.getElementById('encyclopedia-search');
    const encyclopediaList = document.getElementById('encyclopedia-list');

    const filterPills = document.querySelectorAll('.filter-pill');

    // ============================================================
    // APPLICATION STATE
    // ============================================================

    let selectedFile = null;
    let selectedBase64 = null;

    let allEncyclopediaData = [];
    let currentFilter = 'all';

    // ============================================================
    // SAMPLE IMAGES
    // ============================================================

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

    // ============================================================
    // INITIALIZATION
    // ============================================================

    function initSamples() {

        if (!sampleGrid) return;

        sampleGrid.innerHTML = '';

        sampleImages.forEach(sample => {

            const card = document.createElement('div');

            card.className = 'sample-card';

            card.innerHTML = `
                <div class="sample-img-wrap">
                    <img
                        src="/static/samples/${sample.filename}"
                        alt="${sample.title}"
                        onerror="this.src='/static/samples/placeholder.png'"
                    >
                </div>

                <div class="sample-title">
                    ${sample.title}
                </div>
            `;

            card.addEventListener('click', () => {
                loadSampleImage(sample);
            });

            sampleGrid.appendChild(card);
        });
    }

    // ============================================================
    // LOAD SAMPLE IMAGE
    // ============================================================

    async function loadSampleImage(sample) {

        try {

            showLoading(true);

            const response = await fetch(
                `/static/samples/${sample.filename}`
            );

            if (!response.ok) {
                throw new Error("Sample image not found on server");
            }

            const blob = await response.blob();

            const file = new File(
                [blob],
                sample.filename,
                {
                    type: blob.type || "image/jpeg"
                }
            );

            handleFileSelect(file);

            showToast(
                `Loaded sample: ${sample.title}`,
                "success"
            );

        } catch (error) {

            console.error("Sample loading error:", error);

            showToast(
                "Failed to load sample leaf.",
                "error"
            );

        } finally {

            showLoading(false);
        }
    }

    // ============================================================
    // TAB NAVIGATION
    // ============================================================

    function switchTab(activeTab) {

        [
            tabUpload,
            tabCamera,
            tabSamples
        ].forEach(tab => {
            if (tab) tab.classList.remove('active');
        });

        [
            viewUpload,
            viewCamera,
            viewSamples
        ].forEach(view => {
            if (view) view.classList.add('hidden');
        });

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

    tabUpload.addEventListener(
        'click',
        () => switchTab('upload')
    );

    tabCamera.addEventListener(
        'click',
        () => switchTab('camera')
    );

    tabSamples.addEventListener(
        'click',
        () => switchTab('samples')
    );

    // ============================================================
    // FILE UPLOAD
    // ============================================================

    btnBrowse.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', event => {

        if (
            event.target.files &&
            event.target.files.length > 0
        ) {
            handleFileSelect(event.target.files[0]);
        }
    });

    // ============================================================
    // DRAG & DROP
    // ============================================================

    ['dragenter', 'dragover'].forEach(eventName => {

        dropzone.addEventListener(eventName, event => {

            event.preventDefault();
            event.stopPropagation();

            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {

        dropzone.addEventListener(eventName, event => {

            event.preventDefault();
            event.stopPropagation();

            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', event => {

        const files = event.dataTransfer.files;

        if (files && files.length > 0) {

            handleFileSelect(files[0]);
        }
    });

    // ============================================================
    // HANDLE SELECTED FILE
    // ============================================================

    function handleFileSelect(file) {

        if (!file) return;

        if (!file.type.startsWith('image/')) {

            showToast(
                "Please upload an image file (JPEG, PNG, WEBP).",
                "error"
            );

            return;
        }

        selectedFile = file;
        selectedBase64 = null;

        const reader = new FileReader();

        reader.onload = event => {

            previewImg.src = event.target.result;

            fileMeta.textContent =
                `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

            previewContainer.classList.remove('hidden');

            previewContainer.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest'
            });
        };

        reader.onerror = () => {

            showToast(
                "Unable to read the selected image.",
                "error"
            );
        };

        reader.readAsDataURL(file);
    }

    // ============================================================
    // CLEAR IMAGE
    // ============================================================

    btnClearImg.addEventListener('click', () => {

        selectedFile = null;
        selectedBase64 = null;

        previewImg.src = '';

        fileInput.value = '';

        previewContainer.classList.add('hidden');

        // Also hide previous Grad-CAM
        if (gradcamSection) {
            gradcamSection.style.display = 'none';
        }

        // Return result panel to empty state
        if (resultsContent) {
            resultsContent.classList.add('hidden');
        }

        if (emptyState) {
            emptyState.classList.remove('hidden');
        }
    });

    // ============================================================
    // LIVE CAMERA
    // ============================================================

    btnStartCamera.addEventListener(
        'click',
        async () => {

            try {

                if (
                    !navigator.mediaDevices ||
                    !navigator.mediaDevices.getUserMedia
                ) {
                    throw new Error(
                        "Camera API is not supported by this browser."
                    );
                }

                cameraStream =
                    await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: {
                                ideal: 'environment'
                            },
                            width: {
                                ideal: 1280
                            },
                            height: {
                                ideal: 720
                            }
                        },
                        audio: false
                    });

                cameraFeed.srcObject = cameraStream;

                await cameraFeed.play();

                btnStartCamera.classList.add('hidden');

                btnCapturePhoto.classList.remove('hidden');

                showToast(
                    "Camera activated. Point at plant leaf.",
                    "success"
                );

            } catch (error) {

                console.error(
                    "Camera access failed:",
                    error
                );

                showToast(
                    "Camera access denied or unavailable.",
                    "error"
                );
            }
        }
    );

    // ============================================================
    // CAPTURE CAMERA PHOTO
    // ============================================================

    btnCapturePhoto.addEventListener(
        'click',
        () => {

            if (!cameraStream) {

                showToast(
                    "Camera is not active.",
                    "error"
                );

                return;
            }

            const width =
                cameraFeed.videoWidth || 640;

            const height =
                cameraFeed.videoHeight || 480;

            cameraCanvas.width = width;
            cameraCanvas.height = height;

            const context =
                cameraCanvas.getContext('2d');

            context.drawImage(
                cameraFeed,
                0,
                0,
                width,
                height
            );

            const dataUrl =
                cameraCanvas.toDataURL(
                    'image/jpeg',
                    0.92
                );

            previewImg.src = dataUrl;

            selectedBase64 = dataUrl;
            selectedFile = null;

            fileMeta.textContent =
                `Live Camera Capture (${width}x${height})`;

            previewContainer.classList.remove('hidden');

            stopCamera();

            switchTab('upload');

            showToast(
                "Leaf captured! Click 'Run AI Diagnosis'.",
                "success"
            );
        }
    );

    // ============================================================
    // STOP CAMERA
    // ============================================================

    function stopCamera() {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(track => track.stop());

            cameraStream = null;
        }

        if (cameraFeed) {
            cameraFeed.srcObject = null;
        }

        if (btnStartCamera) {
            btnStartCamera.classList.remove('hidden');
        }

        if (btnCapturePhoto) {
            btnCapturePhoto.classList.add('hidden');
        }
    }

    // ============================================================
    // RUN AI DIAGNOSIS
    // ============================================================

    btnAnalyze.addEventListener(
        'click',
        async () => {

            if (!selectedFile && !selectedBase64) {

                showToast(
                    "Please select or capture a plant leaf image first.",
                    "error"
                );

                return;
            }

            try {

                showLoading(true);

                let responseData;

                // ------------------------------------------------
                // NORMAL UPLOADED FILE
                // Uses Grad-CAM endpoint
                // ------------------------------------------------

                if (selectedFile) {

                    const formData = new FormData();

                    formData.append(
                        'file',
                        selectedFile
                    );

                    formData.append(
                        'top_k',
                        '5'
                    );

                    const response =
                        await fetch(
                            '/predict-with-gradcam',
                            {
                                method: 'POST',
                                body: formData
                            }
                        );

                    responseData =
                        await parseApiResponse(response);
                }

                // ------------------------------------------------
                // CAMERA IMAGE
                // Uses BASE64 GRAD-CAM endpoint if available
                // ------------------------------------------------

                else if (selectedBase64) {

                    /*
                     * IMPORTANT:
                     *
                     * We first try the Grad-CAM base64 endpoint.
                     * If your backend has not yet added this endpoint,
                     * we automatically fall back to /predict-base64.
                     */

                    let response =
                        await fetch(
                            '/predict-base64-with-gradcam',
                            {
                                method: 'POST',
                                headers: {
                                    'Content-Type':
                                        'application/json'
                                },
                                body: JSON.stringify({
                                    image: selectedBase64,
                                    top_k: 5
                                })
                            }
                        );

                    // ------------------------------------------------
                    // BACKWARD COMPATIBILITY
                    // ------------------------------------------------

                    if (response.status === 404) {

                        console.warn(
                            "Base64 Grad-CAM endpoint not found. Falling back to /predict-base64."
                        );

                        response =
                            await fetch(
                                '/predict-base64',
                                {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type':
                                            'application/json'
                                    },
                                    body: JSON.stringify({
                                        image: selectedBase64,
                                        top_k: 5
                                    })
                                }
                            );
                    }

                    responseData =
                        await parseApiResponse(response);
                }

                // ------------------------------------------------
                // RENDER RESULTS
                // ------------------------------------------------

                renderDiagnosisResults(
                    responseData
                );

                showToast(
                    "Diagnosis completed successfully!",
                    "success"
                );

            } catch (error) {

                console.error(
                    "Diagnosis error:",
                    error
                );

                showToast(
                    error.message ||
                    "Failed to process image.",
                    "error"
                );

            } finally {

                showLoading(false);
            }
        }
    );

    // ============================================================
    // API RESPONSE PARSER
    // ============================================================

    async function parseApiResponse(response) {

        let data = null;

        try {
            data = await response.json();
        } catch (error) {

            throw new Error(
                `Server returned an invalid response (${response.status}).`
            );
        }

        if (!response.ok) {

            let message =
                "Diagnosis failed on server.";

            if (data) {

                if (typeof data.detail === 'string') {
                    message = data.detail;
                }

                else if (
                    data.detail &&
                    typeof data.detail === 'object'
                ) {
                    message =
                        JSON.stringify(data.detail);
                }

                else if (data.message) {
                    message = data.message;
                }
            }

            throw new Error(message);
        }

        return data;
    }

    // ============================================================
    // RENDER DIAGNOSIS RESULTS
    // ============================================================

    function renderDiagnosisResults(data) {

        if (!data || !data.prediction) {

            showToast(
                "Server returned no prediction.",
                "error"
            );

            return;
        }

        const pred = data.prediction;

        const isHealthy =
            Boolean(pred.is_healthy);

        // --------------------------------------------------------
        // SHOW RESULTS
        // --------------------------------------------------------

        emptyState.classList.add('hidden');

        resultsContent.classList.remove('hidden');

        // --------------------------------------------------------
        // PLANT / DISEASE
        // --------------------------------------------------------

        resPlantName.textContent =
            pred.plant || "Unknown Plant";

        resDiseaseName.textContent =
            pred.condition || "Unknown Condition";

        // --------------------------------------------------------
        // HEALTH STATUS
        // --------------------------------------------------------

        if (isHealthy) {

            primaryCard.className =
                'diagnosis-card';

            resStatusTag.className =
                'status-indicator healthy';

            resStatusTag.innerHTML =
                '<i class="fa-solid fa-circle-check"></i> <span>Healthy</span>';

            resSeverityVal.className =
                'meta-value severity-badge severity-none';

        } else {

            primaryCard.className =
                'diagnosis-card infected';

            resStatusTag.className =
                'status-indicator infected';

            resStatusTag.innerHTML =
                '<i class="fa-solid fa-triangle-exclamation"></i> <span>Infected</span>';

            const severity =
                (
                    pred.severity ||
                    'Medium'
                ).toLowerCase();

            resSeverityVal.className =
                `meta-value severity-badge severity-${severity}`;
        }

        // --------------------------------------------------------
        // CONFIDENCE
        // --------------------------------------------------------

        const confidence =
            Number(pred.confidence) || 0;

        const safeConfidence =
            Math.max(
                0,
                Math.min(
                    100,
                    confidence
                )
            );

        resConfidenceVal.textContent =
            `${safeConfidence}%`;

        resConfidenceBar.style.width =
            `${safeConfidence}%`;

        // --------------------------------------------------------
        // METADATA
        // --------------------------------------------------------

        resCauseVal.textContent =
            pred.cause || 'N/A';

        resSeverityVal.textContent =
            pred.severity || 'None';

        // --------------------------------------------------------
        // TOP 5 PREDICTIONS
        // --------------------------------------------------------

        renderTopK(
            data.top_k_predictions
        );

        // --------------------------------------------------------
        // TREATMENT INFORMATION
        // --------------------------------------------------------

        resSymptoms.textContent =
            pred.symptoms ||
            "No specific symptoms recorded.";

        resOrganic.textContent =
            pred.organic_treatment ||
            "No specific organic treatment required.";

        resChemical.textContent =
            pred.chemical_treatment ||
            "No chemical application required.";

        resPrevention.textContent =
            pred.prevention ||
            "Maintain standard clean cultivation practices.";

        // --------------------------------------------------------
        // TIMESTAMP
        // --------------------------------------------------------

        reportTimestamp.textContent =
            new Date().toLocaleTimeString();

        // --------------------------------------------------------
        // GRAD-CAM
        // --------------------------------------------------------

        if (data.gradcam) {

            renderGradcam(
                data.gradcam
            );

        } else {

            gradcamSection.style.display =
                'none';
        }

        // --------------------------------------------------------
        // SCROLL RESULTS INTO VIEW
        // --------------------------------------------------------

        resultsContent.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }

    // ============================================================
    // TOP-K RENDERER
    // ============================================================

    function renderTopK(predictions) {

        topkList.innerHTML = '';

        if (
            !Array.isArray(predictions) ||
            predictions.length === 0
        ) {

            topkList.innerHTML = `
                <div class="topk-empty">
                    No probability ranking available.
                </div>
            `;

            return;
        }

        predictions.forEach(item => {

            const itemDiv =
                document.createElement('div');

            itemDiv.className =
                'topk-item';

            const confidence =
                Number(item.confidence) || 0;

            const safeConfidence =
                Math.max(
                    0,
                    Math.min(
                        100,
                        confidence
                    )
                );

            itemDiv.innerHTML = `
                <div class="topk-row">

                    <span class="topk-name">

                        <span class="topk-rank">
                            #${item.rank ?? ''}
                        </span>

                        ${escapeHtml(
                            item.display_name ||
                            item.condition ||
                            item.raw_class ||
                            'Unknown'
                        )}

                    </span>

                    <span class="topk-pct">
                        ${safeConfidence}%
                    </span>

                </div>

                <div class="topk-bar-track">

                    <div
                        class="topk-bar-fill"
                        style="width: ${safeConfidence}%"
                    ></div>

                </div>
            `;

            topkList.appendChild(
                itemDiv
            );
        });
    }

    // ============================================================
    // GRAD-CAM RENDERER
    // ============================================================

    function renderGradcam(gcam) {

        if (!gradcamSection) return;

        // Always show section
        gradcamSection.style.display =
            'block';

        // Reset
        gradcamErrorNotice.classList.add(
            'hidden'
        );

        gradcamGrid.style.display =
            'grid';

        // --------------------------------------------------------
        // TARGET LAYER
        // --------------------------------------------------------

        if (gcam.target_layer) {

            gradcamLayerName.textContent =
                gcam.target_layer;
        }

        // --------------------------------------------------------
        // SUCCESS
        // --------------------------------------------------------

        if (gcam.success) {

            gradcamStatusBadge.className =
                'gradcam-status-badge success';

            gradcamStatusBadge.innerHTML =
                '<i class="fa-solid fa-circle-check"></i> Grad-CAM Ready';

            gradcamOriginal.src =
                gcam.original_image || '';

            gradcamHeatmap.src =
                gcam.heatmap || '';

            gradcamOverlay.src =
                gcam.overlay || '';

            return;
        }

        // --------------------------------------------------------
        // FAILURE
        // --------------------------------------------------------

        gradcamStatusBadge.className =
            'gradcam-status-badge error';

        gradcamStatusBadge.innerHTML =
            '<i class="fa-solid fa-triangle-exclamation"></i> Grad-CAM Error';

        gradcamErrorNotice.classList.remove(
            'hidden'
        );

        gradcamErrorText.textContent =
            gcam.error ||
            'Grad-CAM could not be generated.';

        gradcamOriginal.src = '';
        gradcamHeatmap.src = '';
        gradcamOverlay.src = '';

        gradcamGrid.style.display =
            'none';
    }

    // ============================================================
    // ENCYCLOPEDIA MODAL
    // ============================================================

    btnEncyclopedia.addEventListener(
        'click',
        async () => {

            encyclopediaModal.classList.remove(
                'hidden'
            );

            if (
                allEncyclopediaData.length === 0
            ) {

                await fetchEncyclopedia();
            }
        }
    );

    btnCloseModal.addEventListener(
        'click',
        () => {

            encyclopediaModal.classList.add(
                'hidden'
            );
        }
    );

    encyclopediaModal.addEventListener(
        'click',
        event => {

            if (
                event.target ===
                encyclopediaModal
            ) {

                encyclopediaModal.classList.add(
                    'hidden'
                );
            }
        }
    );

    // ============================================================
    // FETCH ENCYCLOPEDIA
    // ============================================================

    async function fetchEncyclopedia() {

        try {

            const response =
                await fetch(
                    '/api/classes'
                );

            const data =
                await parseApiResponse(
                    response
                );

            allEncyclopediaData =
                Array.isArray(data.classes)
                    ? data.classes
                    : [];

            renderEncyclopediaList();

        } catch (error) {

            console.error(
                "Failed to load encyclopedia:",
                error
            );

            encyclopediaList.innerHTML = `
                <div style="
                    grid-column:1/-1;
                    text-align:center;
                    padding:2rem;
                    color:var(--text-muted);
                ">
                    Failed to load disease encyclopedia.
                    <br>
                    Please try again.
                </div>
            `;

            showToast(
                "Failed to load disease encyclopedia.",
                "error"
            );
        }
    }

    // ============================================================
    // RENDER ENCYCLOPEDIA
    // ============================================================

    function renderEncyclopediaList() {

        const query =
            (
                encyclopediaSearch.value ||
                ''
            )
                .toLowerCase()
                .trim();

        encyclopediaList.innerHTML = '';

        const filtered =
            allEncyclopediaData.filter(item => {

                const displayName =
                    String(
                        item.display_name || ''
                    ).toLowerCase();

                const plant =
                    String(
                        item.plant || ''
                    ).toLowerCase();

                const condition =
                    String(
                        item.condition || ''
                    ).toLowerCase();

                const matchesQuery =
                    displayName.includes(query) ||
                    plant.includes(query) ||
                    condition.includes(query);

                if (!matchesQuery) {
                    return false;
                }

                if (
                    currentFilter === 'healthy'
                ) {
                    return Boolean(
                        item.is_healthy
                    );
                }

                if (
                    currentFilter === 'diseased'
                ) {
                    return !Boolean(
                        item.is_healthy
                    );
                }

                return true;
            });

        if (filtered.length === 0) {

            encyclopediaList.innerHTML = `
                <div style="
                    grid-column:1/-1;
                    text-align:center;
                    color:var(--text-muted);
                    padding:2rem;
                ">
                    No matching plant diseases found.
                </div>
            `;

            return;
        }

        filtered.forEach(item => {

            const card =
                document.createElement('div');

            card.className =
                'encyclopedia-card';

            card.style.cursor =
                'pointer';

            card.innerHTML = `
                <div class="enc-top">

                    <div>

                        <div class="enc-plant">
                            ${escapeHtml(
                                item.plant || 'Unknown'
                            )}
                        </div>

                        <div class="enc-title">
                            ${escapeHtml(
                                item.condition || 'Unknown'
                            )}
                        </div>

                    </div>

                    <span class="enc-status ${
                        item.is_healthy
                            ? 'healthy'
                            : 'infected'
                    }">

                        ${
                            item.is_healthy
                                ? 'Healthy'
                                : 'Diseased'
                        }

                    </span>

                </div>

                <div class="enc-desc">

                    Trained neural class identifier:

                    <code>
                        ${escapeHtml(
                            item.raw_class || ''
                        )}
                    </code>

                </div>
            `;

            card.addEventListener(
                'click',
                () => {

                    selectEncyclopediaItem(
                        item,
                        card
                    );
                }
            );

            encyclopediaList.appendChild(
                card
            );
        });
    }

    // ============================================================
    // ENCYCLOPEDIA ITEM DETAILS
    // ============================================================

    async function selectEncyclopediaItem(
        item,
        card
    ) {

        document
            .querySelectorAll(
                '.encyclopedia-card'
            )
            .forEach(c => {

                c.style.borderColor =
                    'var(--card-border)';
            });

        card.style.borderColor =
            'var(--emerald-400)';

        const detailContainer =
            document.getElementById(
                'encyclopedia-detail'
            );

        const title =
            document.getElementById(
                'enc-detail-title'
            );

        const symptoms =
            document.getElementById(
                'enc-detail-symptoms'
            );

        const organic =
            document.getElementById(
                'enc-detail-organic'
            );

        const chemical =
            document.getElementById(
                'enc-detail-chemical'
            );

        const prevention =
            document.getElementById(
                'enc-detail-prevention'
            );

        detailContainer.style.display =
            'block';

        title.innerHTML =
            '<i class="fa-solid fa-spinner fa-spin"></i> Loading details...';

        symptoms.textContent = '';
        organic.textContent = '';
        chemical.textContent = '';
        prevention.textContent = '';

        try {

            const rawClass =
                encodeURIComponent(
                    item.raw_class
                );

            const response =
                await fetch(
                    `/api/class-info/${rawClass}`
                );

            const info =
                await parseApiResponse(
                    response
                );

            title.textContent =
                item.display_name ||
                item.condition ||
                'Condition';

            symptoms.textContent =
                info.symptoms ||
                'Information not available.';

            organic.textContent =
                info.organic_treatment ||
                'Information not available.';

            chemical.textContent =
                info.chemical_treatment ||
                'Information not available.';

            prevention.textContent =
                info.prevention ||
                'Information not available.';

        } catch (error) {

            console.error(
                "Error fetching class info:",
                error
            );

            title.textContent =
                item.display_name ||
                item.condition ||
                'Condition';

            symptoms.textContent =
                "Error loading details. Please try again.";

            organic.textContent = '';
            chemical.textContent = '';
            prevention.textContent = '';
        }
    }

    // ============================================================
    // ENCYCLOPEDIA SEARCH
    // ============================================================

    encyclopediaSearch.addEventListener(
        'input',
        renderEncyclopediaList
    );

    // ============================================================
    // ENCYCLOPEDIA FILTERS
    // ============================================================

    filterPills.forEach(pill => {

        pill.addEventListener(
            'click',
            () => {

                filterPills.forEach(p => {
                    p.classList.remove(
                        'active'
                    );
                });

                pill.classList.add(
                    'active'
                );

                currentFilter =
                    pill.getAttribute(
                        'data-filter'
                    ) || 'all';

                renderEncyclopediaList();
            }
        );
    });

    // ============================================================
    // LOADING STATE
    // ============================================================

    function showLoading(show) {

        if (!loadingOverlay) return;

        if (show) {
            loadingOverlay.classList.remove(
                'hidden'
            );
        } else {
            loadingOverlay.classList.add(
                'hidden'
            );
        }
    }

    // ============================================================
    // TOAST NOTIFICATIONS
    // ============================================================

    function showToast(
        message,
        type = 'info'
    ) {

        const container =
            document.getElementById(
                'toast-container'
            );

        if (!container) return;

        const toast =
            document.createElement('div');

        toast.className =
            `toast ${type}`;

        let icon =
            'fa-circle-info';

        if (type === 'success') {
            icon = 'fa-circle-check';
        }

        if (type === 'error') {
            icon =
                'fa-triangle-exclamation';
        }

        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${escapeHtml(
                String(message)
            )}</span>
        `;

        container.appendChild(
            toast
        );

        setTimeout(() => {

            toast.style.opacity = '0';
            toast.style.transform =
                'translateY(10px)';
            toast.style.transition =
                'all 0.3s ease';

            setTimeout(() => {

                if (toast.parentNode) {
                    toast.remove();
                }

            }, 300);

        }, 3800);
    }

    // ============================================================
    // BASIC HTML ESCAPE
    // ============================================================

    function escapeHtml(value) {

        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // ============================================================
    // INITIALIZE
    // ============================================================

    initSamples();

    console.log(
        "AgriVision AI frontend initialized successfully."
    );
});