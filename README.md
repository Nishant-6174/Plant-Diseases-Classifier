# 🌿 AgriVision AI — Plant Disease Classifier

[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker\&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AgriVision AI** is an end-to-end deep learning application for plant disease classification using a fine-tuned **EfficientNetB0** model, exposed through a **FastAPI backend** and an interactive web interface with **image upload, live camera capture, sample images, Top-K predictions, disease information, treatment guidance, and Grad-CAM explainability**.

---

## 📌 Overview

Plant disease identification from leaf images is an important computer-vision problem in modern agriculture.

AgriVision AI uses transfer learning with EfficientNetB0 to classify plant leaf images across **86 disease and healthy classes** covering **19+ major crop varieties**.

The project combines:

* 🧠 Fine-tuned EfficientNetB0
* 🚀 FastAPI backend
* 🖥️ Interactive web interface
* 📤 Drag-and-drop image upload
* 📷 Live camera capture
* 🖼️ Sample leaf demonstrations
* 📊 Top-K prediction probabilities
* 🔍 Grad-CAM explainability
* 📚 86-category disease encyclopedia
* 💊 Treatment and prevention information
* 📝 Structured application logging
* 🛡️ Image validation
* 🐳 Docker support
* 🧪 Automated testing infrastructure
* ⚙️ Deployment-ready project structure

The application is designed as a **portfolio-level end-to-end machine-learning project**, demonstrating the complete path from trained model to usable web application and API.

---

# ✨ Key Features

## 🧠 EfficientNetB0 Classification

The core classifier uses a fine-tuned **EfficientNetB0** model.

The prediction pipeline performs:

1. Image loading
2. Image preprocessing
3. Tensor preparation
4. EfficientNetB0 inference
5. Class mapping
6. Confidence calculation
7. Top-K prediction generation
8. Disease metadata enrichment

The application supports **86 trained classes**.

---

## 🖥️ Interactive Web Application

The frontend is implemented using:

* HTML
* CSS
* Vanilla JavaScript
* Jinja2 templates
* FastAPI

The main interface is located at:

```text
templates/index.html
```

Frontend assets are separated into:

```text
static/
├── css/
│   └── style.css
├── js/
│   └── main.js
└── samples/
```

The frontend communicates directly with the FastAPI backend using JavaScript `fetch()` requests.

---

## 📤 Image Upload

Users can:

* Browse for an image
* Drag and drop an image
* Preview the selected image
* Remove the image
* Run AI diagnosis

Supported formats:

```text
JPEG
JPG
PNG
WEBP
```

---

## 📷 Live Camera Capture

The web interface supports real-time camera capture through the browser.

Workflow:

```text
Live Camera
     ↓
Capture Leaf
     ↓
Preview Image
     ↓
Run AI Diagnosis
     ↓
FastAPI
     ↓
EfficientNetB0
```

Camera images are captured in the browser and sent to the backend using the Base64 prediction workflow.

---

## 🖼️ Sample Leaf Gallery

The application includes demonstration images for quick testing.

Current sample images include:

```text
tomato_early_blight.jpg
healthy_apple.jpg
corn_common_rust.jpg
lemon_citrus_canker.jpg
potato_late_blight.jpg
wheat_yellow_rust.jpg
```

Sample images are stored under:

```text
static/samples/
```

Users can select a sample directly from the interface and run a diagnosis.

---

# 🔍 Grad-CAM Explainability

AgriVision AI includes **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Instead of returning only:

```text
Tomato — Early Blight
Confidence: 98.7%
```

the application can also generate visual explanations showing the regions of the leaf that contributed most strongly to the prediction.

The web interface displays:

1. Original image
2. Grad-CAM heatmap
3. Grad-CAM overlay

Example workflow:

```text
Input Leaf
    ↓
EfficientNetB0
    ↓
Prediction
    ↓
Grad-CAM
    ↓
Heatmap + Overlay
```

The current Grad-CAM implementation uses the model's configured activation layer, exposed in the interface as the target Grad-CAM layer.

---

# 📚 Disease Encyclopedia

The application contains an interactive encyclopedia covering all **86 supported classes**.

Users can:

* Search by crop
* Search by disease
* Filter healthy classes
* Filter diseased classes
* Open individual disease information
* View symptoms
* View management information
* View treatment information
* View prevention information

The encyclopedia is dynamically loaded from the backend rather than being hard-coded into the frontend.

---

# 🛡️ Image Validation

The project includes an image-validation layer before model inference.

The implementation is located at:

```text
src/utils/image_validator.py
```

The validator uses visual heuristics such as:

* HSV colour analysis
* Image texture
* Laplacian variance
* Canny edge density

Its purpose is to reject **obviously unsuitable or featureless images** before they reach the disease classifier.

For example, completely uniform images such as:

```text
Solid black
Solid white
Solid gray
Solid blue
```

can be rejected.

### Important limitation

The validation system is a **heuristic validation layer**.

It is **not**:

* A plant detector
* A leaf object detector
* A semantic segmentation model
* A guaranteed non-leaf image detector
* A replacement for a dedicated computer-vision OOD system

A photograph of a non-leaf object may still pass validation if it contains sufficient texture, edges, or plant-like colours.

Therefore, the purpose of this component is:

> **Reject obviously unsuitable images while minimizing unnecessary rejection of legitimate plant leaf images.**

---

# 🏗️ Current Project Architecture

```text
Plant-Disease-Classifier/
│
├── .github/
│   └── workflows/
│       └── ...
│
├── logs/
│   └── Application log files
│
├── src/
│   │
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
│   │
│   ├── js/
│   │   └── main.js
│   │
│   └── samples/
│       ├── corn_common_rust.jpg
│       ├── healthy_apple.jpg
│       ├── lemon_citrus_canker.jpg
│       ├── placeholder.png
│       ├── potato_late_blight.jpg
│       ├── tomato_early_blight.jpg
│       └── wheat_yellow_rust.jpg
│
├── templates/
│   └── index.html
│
├── tests/
│   └── ...
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

# 🔄 Application Architecture

The complete application flow is:

```text
                    ┌──────────────────────┐
                    │    Web Browser       │
                    │                      │
                    │  index.html          │
                    │  style.css            │
                    │  main.js              │
                    └──────────┬───────────┘
                               │
                 Upload / Camera / Sample
                               │
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │       app.py         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Input Validation   │
                    │                      │
                    │ HSV + Texture +      │
                    │ Edge Analysis        │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                  Invalid                Valid
                    │                     │
                    ▼                     ▼
                 HTTP 400          Prediction Pipeline
                                          │
                                          ▼
                                  EfficientNetB0
                                          │
                                          ▼
                                  Class Prediction
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                          ▼               ▼               ▼
                       Top-K          Metadata        Grad-CAM
                       Results        Information     Explanation
                          │               │               │
                          └───────────────┼───────────────┘
                                          │
                                          ▼
                                  FastAPI JSON Response
                                          │
                                          ▼
                                  Interactive Web UI
```

---

# 🌐 Frontend Architecture

The current frontend is intentionally separated from the backend.

## HTML

```text
templates/index.html
```

Responsible for:

* Page structure
* Upload interface
* Camera interface
* Sample gallery
* Diagnosis cards
* Grad-CAM section
* Encyclopedia modal
* Footer

## CSS

```text
static/css/style.css
```

Responsible for:

* Layout
* Responsive design
* Cards
* Buttons
* Diagnosis results
* Grad-CAM presentation
* Modal interface
* Loading states
* Toast notifications

## JavaScript

```text
static/js/main.js
```

Responsible for:

* Upload handling
* Drag-and-drop
* Camera capture
* Sample image loading
* API requests
* Diagnosis rendering
* Top-K results
* Grad-CAM rendering
* Encyclopedia search
* Disease information retrieval
* Toast notifications
* Loading states

---

# 🔌 API Endpoints

The application exposes a REST API through FastAPI.

## `GET /`

Returns the main AgriVision AI web application.

---

## `GET /health`

Checks application and model status.

Example:

```text
GET /health
```

The endpoint is useful for:

* Deployment health checks
* Container health monitoring
* Application diagnostics

---

## `POST /predict`

Performs plant disease prediction using a multipart image upload.

Example:

```bash
curl -X POST "http://localhost:8000/predict?top_k=5" \
     -H "accept: application/json" \
     -F "file=@sample_leaf.jpg"
```

---

## `POST /predict-base64`

Performs prediction using a Base64-encoded image.

This endpoint is particularly useful for:

* Browser camera capture
* Frontend applications
* Mobile clients
* Base64 image workflows

Example request:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
  "top_k": 5
}
```

---

## `POST /predict-with-gradcam`

Performs disease prediction together with Grad-CAM explainability.

Workflow:

```text
Image
 ↓
Validation
 ↓
Prediction
 ↓
Grad-CAM
 ↓
JSON response containing prediction + explanation
```

This endpoint is used by the web application's uploaded-image diagnosis workflow.

---

## `POST /batch-predict`

Provides batch prediction functionality for multiple images.

This endpoint is intended for API-based batch inference.

---

## `GET /api/classes`

Returns the supported disease/healthy class information used by the encyclopedia.

The frontend calls this endpoint dynamically.

---

## `GET /api/class-info/{class_name}`

Returns detailed information for a particular class.

The response can contain information such as:

* Symptoms
* Organic/general management
* Chemical treatment
* Prevention

This endpoint powers the interactive disease encyclopedia.

---

# 📖 API Documentation

FastAPI automatically provides interactive API documentation.

### Swagger UI

```text
http://localhost:8000/docs
```

### ReDoc

```text
http://localhost:8000/redoc
```

Swagger can be used to test API endpoints directly without the frontend.

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/Nishant-6174/plant-disease-classifier.git
cd plant-disease-classifier
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project is configured as an installable package:

```bash
pip install -e .
```

---

# ▶️ Running the Application

Start the application with:

```bash
python app.py
```

Alternatively:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

For local development with automatic reload:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open the application:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

---

# 🧪 Testing

The project contains automated tests for important backend components.

Run:

```bash
pytest -v
```

The test suite covers areas such as:

* API behavior
* Prediction logic
* Model/pipeline behavior
* Grad-CAM
* Image validation
* Exception handling
* Logging
* Response structures

Always run the test suite before pushing a new version to GitHub.

---

# 📊 Model

The project uses:

```text
EfficientNetB0
```

through TensorFlow/Keras.

The trained model is:

```text
plant_disease_efficientnetb0_final.keras
```

The model is loaded through:

```text
src/components/model_loader.py
```

Prediction logic is implemented through the project's prediction components and pipeline.

---

# 🗂️ Disease Metadata

Class definitions are stored in:

```text
src/data/disease_classes.json
```

Disease information is stored in:

```text
src/data/disease_info.json
```

This separation allows the prediction system and encyclopedia to use structured disease metadata without hard-coding the information into the frontend.

---

# 📝 Logging

Application logging is implemented through:

```text
src/logger.py
```

Application logs are stored under:

```text
logs/
```

Logging is used for:

* Application startup
* Model initialization
* Prediction events
* Errors
* Pipeline status
* Debugging

---

# 🐳 Docker

The repository includes Docker support.

Build the image:

```bash
docker build -t plant-disease-classifier .
```

Run the container:

```bash
docker run -d \
  -p 8000:8000 \
  --name plant_disease_app \
  plant-disease-classifier
```

Open:

```text
http://localhost:8000
```

---

# 🐳 Docker Compose

If using the included Compose configuration:

```bash
docker compose up --build
```

Run in detached mode:

```bash
docker compose up --build -d
```

View logs:

```bash
docker compose logs -f
```

Stop the application:

```bash
docker compose down
```

---

# ⚙️ Deployment Architecture

The application is structured so that the same FastAPI application can be used for local development, Docker execution, and cloud deployment.

The production architecture is:

```text
                         Internet
                            │
                            ▼
                    ┌───────────────┐
                    │ Cloud / Server│
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Uvicorn     │
                    │   FastAPI     │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       Web Application               REST API
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                    Prediction Pipeline
                            │
                            ▼
                       EfficientNetB0
```

The application listens on:

```text
0.0.0.0:8000
```

which allows it to run inside a container or on a remote server.

---

# 🔄 CI/CD

The project can use GitHub Actions for automated workflows.

Workflow files are located under:

```text
.github/workflows/
```

Typical CI/CD stages include:

```text
Git Push
   ↓
GitHub Actions
   ↓
Install Dependencies
   ↓
Run Tests
   ↓
Build Application / Docker Image
   ↓
Deployment
```

Deployment-specific secrets should only be configured when required by the selected deployment platform.

---

# 📁 Important Files

| File                               | Purpose                             |
| ---------------------------------- | ----------------------------------- |
| `app.py`                           | FastAPI application and API routes  |
| `templates/index.html`             | Main web application interface      |
| `static/css/style.css`             | Frontend styling                    |
| `static/js/main.js`                | Frontend application logic          |
| `static/samples/`                  | Demonstration leaf images           |
| `src/components/model_loader.py`   | Model loading and initialization    |
| `src/components/predictor.py`      | Prediction logic                    |
| `src/pipeline/predict_pipeline.py` | Prediction pipeline                 |
| `src/utils/common.py`              | Common utility functions            |
| `src/utils/image_validator.py`     | Input image validation              |
| `src/data/disease_classes.json`    | Supported class definitions         |
| `src/data/disease_info.json`       | Disease information                 |
| `src/logger.py`                    | Application logging                 |
| `src/exception.py`                 | Custom exception handling           |
| `tests/`                           | Automated tests                     |
| `Dockerfile`                       | Docker configuration                |
| `docker-compose.yml`               | Docker Compose configuration        |
| `requirements.txt`                 | Python dependencies                 |
| `setup.py`                         | Python package configuration        |
| `walkthrough.md`                   | Technical walkthrough/documentation |

---

# ⚠️ Limitations

## 1. Dataset Dependence

Model performance depends on:

* Training data quality
* Class distribution
* Image quality
* Lighting
* Background variation
* Camera characteristics
* Similarity between deployment images and training data

---

## 2. Image Validation

The current image validator is heuristic-based.

It is not a dedicated plant or leaf detection model.

Therefore, it cannot guarantee rejection of every non-leaf image.

---

## 3. Model Confidence

A high confidence score does not guarantee a correct prediction.

The model may perform poorly on images that are significantly different from its training distribution.

---

## 4. Field Conditions

Images captured in controlled datasets may differ from real agricultural field conditions.

Real-world performance can be affected by:

* Poor lighting
* Occlusion
* Multiple leaves
* Complex backgrounds
* Blur
* Camera quality
* Disease stages not represented in training data

---

## 5. Treatment Information

Treatment and prevention information provided by the application is intended for **informational purposes**.

It should not replace advice from a qualified agricultural professional.

---

# 🚧 Future Improvements

Potential future improvements include:

* Dedicated plant/leaf object detection
* Leaf segmentation
* Stronger out-of-distribution detection
* Confidence calibration
* Larger and more diverse datasets
* Real-world field-condition testing
* Mobile deployment
* Model quantization
* Edge-device inference
* Improved Grad-CAM visualization
* Model monitoring
* Batch image validation
* More comprehensive non-leaf image testing
* Authentication and rate limiting for public API deployment
* Production database for disease metadata
* Cloud-based model serving

---

# 📖 Technical Documentation

Additional implementation documentation can be found in:

```text
walkthrough.md
```

This documentation provides deeper information about the project's image-validation and inference workflow.

---

# 🔐 Security Considerations for Deployment

Before exposing the application publicly, production configuration should consider:

* CORS restrictions
* Request-size limits
* File-type validation
* Rate limiting
* Authentication where appropriate
* HTTPS
* Secure environment variables
* Logging and monitoring
* Resource limits for uploaded images

The current development configuration may intentionally use permissive settings such as:

```python
allow_origins=["*"]
```

For a public production deployment, this should be restricted according to the actual frontend/client origin.

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

Plant Disease Classification • Deep Learning • Computer Vision • FastAPI • Explainable AI

GitHub:

https://github.com/Nishant-6174

---

# ⭐ Project Status

**AgriVision AI is currently a functional end-to-end plant disease classification application.**

The current workflow is:

```text
┌──────────────────────────┐
│       User Image         │
│                          │
│ Upload / Camera / Sample │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│       FastAPI            │
│         app.py           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Image Validation      │
└────────────┬─────────────┘
             │
        ┌────┴────┐
        │         │
     Invalid     Valid
        │         │
        ▼         ▼
    HTTP 400  Prediction
                  │
                  ▼
           EfficientNetB0
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
      Top-K   Metadata   Grad-CAM
        │         │         │
        └─────────┼─────────┘
                  │
                  ▼
           JSON Response
                  │
                  ▼
        Interactive Web UI
```

---

## 🌱 What This Project Demonstrates

AgriVision AI demonstrates an end-to-end machine-learning engineering workflow:

```text
Deep Learning Model
        ↓
Model Loading
        ↓
Prediction Pipeline
        ↓
Image Validation
        ↓
FastAPI REST API
        ↓
Interactive Frontend
        ↓
Explainable AI
        ↓
Automated Testing
        ↓
Docker
        ↓
CI/CD
        ↓
Deployment
```

**Built as an end-to-end deep learning and deployment project for plant disease classification.**
