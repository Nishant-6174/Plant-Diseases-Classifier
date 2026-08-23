# 🌿 AgriVision AI — Plant Disease Classifier

[![CI/CD Pipeline](https://github.com/Nishant-6174/plant-disease-classifier/actions/workflows/ci_cd.yaml/badge.svg)](https://github.com/Nishant-6174/plant-disease-classifier/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg?logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker\&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AgriVision AI** is an end-to-end deep learning application for plant leaf disease classification using **EfficientNetB0**, exposed through a **FastAPI REST API** with **Grad-CAM explainability**, structured metadata, image validation, automated testing, and Docker support.

---

## 📌 Overview

Plant disease diagnosis from leaf images is a practical computer-vision problem where a trained deep learning model can assist in identifying disease categories from visual symptoms.

This project combines:

* 🧠 **EfficientNetB0** for image classification
* 🚀 **FastAPI** for REST API deployment
* 🔍 **Grad-CAM** for model explainability
* 🛡️ **Input validation** before model inference
* 📊 **Top-K predictions** for probability comparison
* 📚 Disease metadata and treatment information
* 🧪 Automated testing with **pytest**
* 📝 Structured application logging
* 🐳 Docker support
* ⚙️ CI/CD workflow support

The system is designed as a **portfolio-level end-to-end machine-learning application**, covering model inference, API development, validation, testing, and deployment infrastructure.

---

## ✨ Key Features

### 🧠 EfficientNetB0 Disease Classification

The application uses a fine-tuned **EfficientNetB0** model to classify plant leaf images into the trained disease categories.

The prediction pipeline handles:

1. Image loading
2. Image preprocessing
3. Model inference
4. Class mapping
5. Confidence calculation
6. Top-K prediction generation
7. Disease metadata enrichment

---

### 🔍 Grad-CAM Explainability

The `/predict-with-gradcam` endpoint provides Grad-CAM-based visual explanations.

Grad-CAM helps visualize the regions of an input image that contributed most strongly to the model's prediction.

This makes the system more interpretable than a simple classification API.

---

### 🛡️ Input Image Validation

A conservative validation layer has been added before model inference.

The validator is implemented in:

```text
src/utils/image_validator.py
```

It combines three visual signals:

| Signal                      | Purpose                                                                    |
| --------------------------- | -------------------------------------------------------------------------- |
| HSV organic-colour analysis | Detects the presence of plant-like green, yellow, brown and rust/red tones |
| Laplacian variance          | Measures image texture                                                     |
| Canny edge density          | Measures structural content and edges                                      |

The purpose of this layer is to reject **obviously unsuitable or featureless inputs** before they reach the disease classifier.

For example, completely solid blue, gray, black, or white images can be rejected.

### Important limitation

The validator is a **heuristic**, not a dedicated plant-leaf object detector or segmentation model.

It is intentionally conservative:

> **Reject obviously unsuitable images while minimizing false rejection of legitimate healthy or diseased leaf images.**

A real photograph of a car, person, or phone may still pass the heuristic because such images contain edges, textures, and potentially plant-like colours.

Therefore, the validation layer should **not** be described as a guaranteed detector of every non-leaf object.

---

## 🖥️ Application Capabilities

The application provides a web-based interface supporting:

* Drag-and-drop image upload
* Sample leaf images
* Disease prediction
* Confidence scores
* Top-K predictions
* Disease information
* Treatment/prevention information
* Grad-CAM visualization
* API-based inference

---

## 🏗️ Project Architecture

```text
Plant-Disease-Classifier/
│
├── .github/
│   └── workflows/
│       ├── ci_cd.yaml
│       └── deploy.yaml
│
├── logs/
│   └── Application log files
│
├── src/
│   ├── components/
│   │   ├── model_loader.py
│   │   └── predictor.py
│   │
│   ├── constants/
│   │   └── __init__.py
│   │
│   ├── data/
│   │   ├── disease_classes.json
│   │   └── disease_info.json
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── utils/
│   │   ├── common.py
│   │   └── image_validator.py
│   │
│   ├── exception.py
│   └── logger.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── samples/
│       ├── tomato_early_blight.jpg
│       ├── corn_common_rust.jpg
│       ├── potato_late_blight.jpg
│       └── healthy_apple.jpg
│
├── templates/
│   └── index.html
│
├── tests/
│   ├── test_api.py
│   ├── test_exception.py
│   ├── test_gradcam_quick.py
│   ├── test_logger.py
│   ├── test_predictor.py
│   └── test_validation.py
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── walkthrough.md
└── plant_disease_efficientnetb0_final.keras
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Nishant-6174/plant-disease-classifier.git

cd plant-disease-classifier
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project is configured as an editable Python package:

```bash
pip install -e .
```

---

## ▶️ Running the Application

Start the FastAPI application:

```bash
python app.py
```

Or use Uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000
```

Interactive Swagger API documentation:

```text
http://localhost:8000/docs
```

Alternative ReDoc documentation:

```text
http://localhost:8000/redoc
```

---

# 🔌 API Endpoints

## 1. Predict Plant Disease

### Endpoint

```text
POST /predict
```

Accepts a multipart image upload.

Example:

```bash
curl -X POST "http://localhost:8000/predict?top_k=5" \
     -H "accept: application/json" \
     -F "file=@sample_leaf.jpg"
```

Successful response:

```json
{
  "success": true,
  "prediction": {
    "class_name": "tomato_early_blight",
    "display_name": "Tomato - Early Blight",
    "confidence": 98.74
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

The exact response fields depend on the current implementation and disease metadata.

---

## 2. Base64 Prediction

### Endpoint

```text
POST /predict-base64
```

Designed for applications such as:

* Webcam interfaces
* Mobile applications
* Front-end image capture
* Base64-based clients

Example request:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
  "top_k": 5
}
```

---

## 3. Prediction with Grad-CAM

### Endpoint

```text
POST /predict-with-gradcam
```

This endpoint performs prediction and generates a Grad-CAM explanation.

For valid leaf images:

```text
HTTP 200
```

For images rejected by the validation layer:

```text
HTTP 400
```

The classifier and Grad-CAM generation are skipped when validation rejects the image.

---

## 4. Batch Prediction

### Endpoint

```text
POST /batch-predict
```

Designed for processing multiple uploaded images.

> The current input-validation layer is integrated into `/predict`, `/predict-base64`, and `/predict-with-gradcam`. Batch validation is outside the current validation scope.

---

## 5. Health Check

### Endpoint

```text
GET /health
```

Example:

```json
{
  "status": "healthy",
  "service": "Plant Disease Classifier API",
  "model_loaded": true,
  "total_classes": 86,
  "framework": "TensorFlow / Keras (EfficientNetB0)"
}
```

The exact fields depend on the current application implementation.

---

# 🛡️ Invalid Image Handling

If the validation layer identifies an obviously unsuitable image, the API returns:

```text
HTTP 400
```

Example:

```json
{
  "success": false,
  "error_type": "invalid_image",
  "message": "Invalid image. Please upload a clear image of a plant leaf."
}
```

This validation occurs **before model inference**.

Therefore, obviously invalid images do not unnecessarily reach the EfficientNetB0 classifier.

---

# 🧪 Automated Testing

The project includes automated tests covering:

* API health endpoint
* Disease class endpoint
* Prediction endpoint
* Base64 prediction
* Grad-CAM
* Exception handling
* Logging
* Image preprocessing
* Prediction output structure
* Image validation
* Invalid-image API responses
* Grad-CAM validation behavior

Run the complete test suite:

```bash
pytest -v
```

Current verified result:

```text
35 passed, 1 warning
```

The warning currently comes from a dependency deprecation notice involving the FastAPI/Starlette test client stack and does not represent a failing test.

---

## 🧪 Validation Test Coverage

The validation tests in:

```text
tests/test_validation.py
```

cover both unit and integration behavior.

### Valid inputs

* Tomato Early Blight
* Corn Common Rust
* Potato Late Blight
* Healthy Apple
* Synthetic green image
* Synthetic yellow/brown image

### Invalid inputs

* Solid blue image
* Solid gray image
* Solid black image
* Solid white image

### Endpoint behavior

The tests verify that:

```text
Valid image
     ↓
Validation
     ↓
EfficientNetB0
     ↓
Prediction
```

while:

```text
Obviously invalid image
     ↓
Validation
     ↓
HTTP 400
     ↓
Model inference skipped
```

For Grad-CAM, the tests additionally verify that a rejected image does not contain a `gradcam` response field.

---

# 📊 Model

The project uses:

```text
EfficientNetB0
```

through TensorFlow/Keras.

The trained model file is:

```text
plant_disease_efficientnetb0_final.keras
```

The model is loaded through the project's model-loading/prediction pipeline.

Disease class metadata is maintained separately in:

```text
src/data/disease_classes.json
```

Additional disease information is stored in:

```text
src/data/disease_info.json
```

---

# 🔍 Explainability

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations of model predictions.

Instead of only returning:

```text
Tomato Early Blight — 98.7%
```

the system can also provide an explanation map showing which image regions contributed most strongly to the prediction.

This is particularly useful when evaluating whether the model is focusing on visually meaningful leaf regions.

---

# 📝 Logging

Application logging is implemented through:

```text
src/logger.py
```

Logs are stored under:

```text
logs/
```

The project also includes automated logger tests.

---

# 🐳 Docker

The project includes Docker support.

### Build the image

```bash
docker build -t plant-disease-classifier:latest .
```

### Run the container

```bash
docker run -d \
  -p 8000:8000 \
  --name plant_disease_app \
  plant-disease-classifier:latest
```

The application can then be accessed at:

```text
http://localhost:8000
```

---

## Docker Compose

If `docker-compose.yml` is available:

```bash
docker-compose up --build
```

Run in detached mode:

```bash
docker-compose up --build -d
```

View logs:

```bash
docker-compose logs -f
```

Stop the application:

```bash
docker-compose down
```

---

# ⚙️ CI/CD

The repository contains GitHub Actions workflows under:

```text
.github/workflows/
```

These workflows can be used for automated project checks, testing, container builds, and deployment depending on the configured workflow files and repository secrets.

Typical deployment secrets may include:

| Secret            | Purpose                      |
| ----------------- | ---------------------------- |
| `DOCKER_USERNAME` | Docker registry username     |
| `DOCKER_PASSWORD` | Docker registry access token |
| `SERVER_HOST`     | Deployment server            |
| `SERVER_USER`     | SSH deployment user          |
| `SERVER_SSH_KEY`  | SSH private key              |

> Only configure deployment secrets if the corresponding workflow is actually enabled and intended to be used.

---

# 📁 Important Files

| File                               | Purpose                                     |
| ---------------------------------- | ------------------------------------------- |
| `app.py`                           | FastAPI application and API endpoints       |
| `src/components/model_loader.py`   | Model loading and initialization            |
| `src/components/predictor.py`      | Prediction and preprocessing logic          |
| `src/pipeline/predict_pipeline.py` | Prediction pipeline orchestration           |
| `src/utils/common.py`              | Common utilities                            |
| `src/utils/image_validator.py`     | Conservative input-image validation         |
| `src/data/disease_classes.json`    | Disease class definitions                   |
| `src/data/disease_info.json`       | Disease information and metadata            |
| `src/logger.py`                    | Application logging                         |
| `src/exception.py`                 | Custom exception handling                   |
| `tests/`                           | Automated test suite                        |
| `walkthrough.md`                   | Input-validation implementation walkthrough |
| `Dockerfile`                       | Container configuration                     |
| `docker-compose.yml`               | Docker Compose configuration                |
| `requirements.txt`                 | Python dependencies                         |

---

# ⚠️ Limitations

This project is a machine-learning classification system and has several important limitations.

### 1. Dataset Dependence

Model performance depends heavily on:

* Training dataset quality
* Class distribution
* Image quality
* Lighting conditions
* Background variation
* Camera characteristics

### 2. Image Validation

The current validation layer is a conservative heuristic.

It is **not**:

* A dedicated object detector
* A semantic segmentation model
* A plant-leaf recognition model
* A guarantee that every non-leaf image will be rejected

Its purpose is primarily to reject clearly unsuitable or featureless inputs.

### 3. Model Predictions

A high confidence score does not necessarily mean that a prediction is correct.

The model may fail when presented with images significantly different from its training distribution.

### 4. Treatment Information

Treatment and prevention information provided by the application should be considered **informational guidance** and not a replacement for diagnosis or advice from a qualified agricultural professional.

---

# 🚧 Future Improvements

Possible future improvements include:

* Dedicated plant/leaf object detection
* Leaf segmentation before classification
* Stronger out-of-distribution detection
* Confidence calibration
* Larger and more diverse datasets
* Field-condition image testing
* Mobile deployment
* Model quantization
* Edge-device inference
* Improved Grad-CAM visualization
* Monitoring model performance after deployment
* Per-image validation for batch prediction
* More comprehensive adversarial/non-leaf test cases

---

# 📖 Validation Walkthrough

A detailed explanation of the image-validation implementation is available in:

```text
walkthrough.md
```

It documents:

* Why input validation was added
* The HSV colour heuristic
* Texture analysis
* Edge-density analysis
* Validation behavior
* API integration
* Test results
* Current limitations

---

# 📜 License

This project is distributed under the MIT License.

See:

```text
LICENSE
```

for the complete license text.

---

# 👨‍💻 Author

**Nishant Kumar**

Plant Disease Classification • Deep Learning • Computer Vision • FastAPI • MLOps

GitHub:

https://github.com/Nishant-6174

---

## ⭐ Project Status

**Current automated test status:**

```text
35 passed, 1 warning
```

The project currently provides a complete inference workflow:

```text
                ┌─────────────────────┐
                │   Image Input       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Image Validation    │
                │ HSV + Texture +     │
                │ Edge Density        │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
             Invalid                Valid
                │                     │
                ▼                     ▼
          HTTP 400              EfficientNetB0
                                      │
                                      ▼
                              Disease Prediction
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                      Top-K Results          Grad-CAM
                           │                     │
                           └──────────┬──────────┘
                                      │
                                      ▼
                              API / Web Interface
```

---

**Built as an end-to-end deep learning and deployment project for plant disease classification.**
