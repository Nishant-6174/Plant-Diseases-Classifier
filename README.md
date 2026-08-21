# 🌿 AgriVision AI — Production-Grade Plant Disease Classifier & Pathology Suite

[![CI/CD Pipeline](https://github.com/your-org/plant-disease-classifier/actions/workflows/ci_cd.yaml/badge.svg)](https://github.com/your-org/plant-disease-classifier/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AgriVision AI** is an end-to-end, industrial-grade deep learning solution for automated plant leaf disease diagnosis and treatment prescription across **86 plant pathology categories** covering 19+ vital agricultural crops (Tomato, Potato, Apple, Corn, Grape, Cotton, Wheat, Rice, Lemon, etc.).

---

## 📸 System Interface & Capabilities

- **Multi-Input Leaf Studio**: Drag-and-drop file upload, live webcam / field camera capture, and 1-click sample leaf demonstrations.
- **Real-Time Pathology Diagnosis**: Predicts primary condition with confidence percentage, disease severity level, and pathogen classification (Fungal, Bacterial, Viral, Pest, Deficiency).
- **Top-5 Probability Distribution**: Visualizes multi-class uncertainty across all potential candidate categories.
- **Clinical Prescription & Treatment Protocols**: Instant recommendations covering:
  - 📋 **Visual Symptoms & Pathogen Cause**
  - 🌿 **Organic & Biological Solutions** (e.g. Neem oil, Trichoderma viride, Bacillus subtilis)
  - 🧪 **Chemical Treatment & Active Fungicides** (e.g. Mancozeb, Propiconazole, Copper Oxychloride)
  - 🛡️ **Agronomic Prevention Protocols** (Crop rotation, canopy ventilation, drip irrigation)
- **Searchable Encyclopedia**: Interactive catalog for browsing all 86 supported plant health conditions.

---

## 🏗️ Architecture & Component Design

```
Plant-Disease-Classifier/
├── .github/
│   └── workflows/
│       ├── ci_cd.yaml             # Continuous Integration & Docker Build/Push
│       └── deploy.yaml            # Automated Continuous Deployment to Cloud VM
├── logs/                          # Rotating timestamped application execution logs
├── src/
│   ├── components/
│   │   ├── model_loader.py        # Thread-safe singleton model cache with warm-up
│   │   └── predictor.py           # Preprocessor, tensor inference & remedy enrichment
│   ├── constants/
│   │   └── __init__.py            # Image dimensions, model path & system constants
│   ├── data/
│   │   ├── disease_classes.json   # 86 trained class labels
│   │   └── disease_info.json      # Pathology & treatment knowledge base for 86 classes
│   ├── pipeline/
│   │   └── predict_pipeline.py    # Single, Base64 & Batch inference orchestrator
│   ├── utils/
│   │   └── common.py              # File I/O, base64 decoding & metadata parsing
│   ├── exception.py               # Custom Exception handler with traceback capture
│   └── logger.py                  # Production rotating logger
├── static/
│   ├── css/style.css              # Custom responsive stylesheet & glassmorphism theme
│   ├── js/main.js                 # Asynchronous client logic, camera & charts
│   └── samples/                   # Demonstration leaf images for 1-click tests
├── templates/
│   └── index.html                 # Main diagnostic dashboard template
├── tests/
│   ├── test_api.py                # FastAPI endpoint integration tests
│   ├── test_exception.py          # Custom exception unit tests
│   ├── test_logger.py             # Logger disk verification tests
│   └── test_predictor.py          # Tensor shapes & inference tests
├── app.py                         # Production FastAPI application & ASGI server
├── Dockerfile                     # Optimized Python 3.10-slim container image
├── docker-compose.yml             # Single-command local/server orchestration
├── requirements.txt               # Pinned production dependencies
├── setup.py                       # Python package configuration
└── plant_disease_efficientnetb0_final.keras  # Fine-tuned neural weights (41.9 MB)
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/plant-disease-classifier.git
cd plant-disease-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies & editable package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Application
```bash
# Start the FastAPI server on port 8000
python app.py
# Or using uvicorn:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.
Interactive Swagger API documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🐳 Docker Deployment

### Run via Docker Compose (Recommended)
```bash
# Build and start container in detached mode
docker-compose up --build -d

# View real-time logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Run via Docker CLI
```bash
# Build Docker image
docker build -t plant-disease-classifier:latest .

# Run container on port 8000
docker run -d -p 8000:8000 --name plant_disease_app plant-disease-classifier:latest
```

---

## 🔌 API Reference & Endpoints

### 1. Single Image Upload
`POST /predict` (Multipart/form-data)
```bash
curl -X POST "http://localhost:8000/predict?top_k=5" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_leaf.jpg"
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class_name": "tomato_early_blight",
    "display_name": "Tomato - Early Blight",
    "plant": "Tomato",
    "condition": "Early Blight",
    "is_healthy": false,
    "confidence": 98.74,
    "probability": 0.9874,
    "status": "Infected",
    "severity": "High",
    "cause": "Fungal Pathogen (Alternaria solani)",
    "symptoms": "Concentric dark brown target spots on lower leaves...",
    "organic_treatment": "Apply copper octanoate or Trichoderma bio-fungicide...",
    "chemical_treatment": "Mancozeb 75% WP or Chlorothalonil spray...",
    "prevention": "Rotate crops and avoid overhead watering."
  },
  "top_k_predictions": [
    {
      "rank": 1,
      "class_name": "tomato_early_blight",
      "display_name": "Tomato - Early Blight",
      "confidence": 98.74
    }
  ]
}
```

### 2. Base64 Inference (Webcam / Mobile)
`POST /predict-base64` (JSON)
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
  "top_k": 5
}
```

### 3. Batch Image Inference
`POST /batch-predict` (Multipart list of files)

### 4. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "service": "Plant Disease Classifier API",
  "model_loaded": true,
  "total_classes": 86,
  "uptime_seconds": 1240,
  "framework": "TensorFlow / Keras (EfficientNetB0)"
}
```

---

## 🧪 Automated Testing

Execute the automated test suite:
```bash
pytest tests/ -v
```

---

## ☁️ Continuous Integration & Server Deployment

### GitHub Actions Secrets Configuration
Set up the following secrets in your GitHub repository (`Settings -> Secrets and variables -> Actions`):

| Secret Key | Description |
| :--- | :--- |
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token |
| `SERVER_HOST` | Target server IP address or domain |
| `SERVER_USER` | Target SSH user (e.g. `ubuntu`) |
| `SERVER_SSH_KEY` | Private SSH key for automated deployment |

Every push to `main` executes unit tests, builds the optimized Docker image, pushes to Docker Hub, and triggers deployment on your cloud host.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
