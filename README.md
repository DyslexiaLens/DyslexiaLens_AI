# DyslexiaLens AI — Sistem Deteksi Disleksia Berbasis Deep Learning

**Capstone Project - Dicoding Generasi Bangkit 2024**

Sistem AI comprehensive untuk mendeteksi dan menganalisis disleksia melalui analisis tulisan tangan dengan arsitektur multi-modal yang didukung oleh tiga divisi pengembangan (AI Engineer, Data Scientist, Backend Engineer).

---

## 📊 Project Overview

```
DyslexiaLens_AI/
│
├── 🤖 Character Recognition/          # AI Engineer - OCR & Char Detection
│   ├── notebooks/                     # Training notebooks (EDA, modeling)
│   ├── inference and fastAPI/         # FastAPI inference server
│   ├── models/                        # Trained EMNIST models
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md                      # Setup & deployment guide
│
├── 📈 DyslexiaScreening/              # Data Scientist - Detection & Analysis
│   ├── notebooks/                     # Prototyping & analysis
│   ├── Inference & FastAPI/           # Inference engine
│   ├── models/                        # Dyslexia classification models
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md                      # Feature engineering docs
│
└── 🏗️ HuggingFace DyslexiaLens/       # Backend Engineer - Production API
    ├── app/                           # FastAPI application
    │   ├── api/v1/                   # RESTful endpoints
    │   ├── services/                 # Core ML inference logic
    │   ├── schemas/                  # Pydantic models
    │   ├── core/                     # Config & security
    │   └── utils/                    # Image processing
    ├── models/                        # Production models
    ├── Dockerfile                     # Container definition
    ├── docker-compose.yml             # Multi-service orchestration
    ├── requirements.txt
    ├── .env.example
    └── README.md                      # Production deployment guide
```

---

## 🎯 Sistem AI - 3 Pipeline Independen

### 1️⃣ Character Recognition (OCR)
**Divisi**: AI Engineer | **Status**: Training & Inference
- **Input**: Gambar handwriting dari lembar kerja
- **Output**: Teks terdeteksi (A-Z)
- **Model**: CNN berbasis EMNIST dataset
- **Tech**: TensorFlow, FastAPI
- 📖 [Dokumentasi → Character Recognition/README.md](Character%20Recognition/README.md)

### 2️⃣ Dyslexia Detection (Multi-Modal Classification)
**Divisi**: Data Scientist | **Status**: Modeling & Evaluation
- **Input**: Character image + 6 mechanical features (stroke density, symmetry, dll)
- **Output**: Binary classification (Normal/Dyslexic) + Severity score
- **Model**: Late Fusion CNN (Multi-Task Learning)
- **Dataset**: Gambo EMNIST Balanced
- 📖 [Dokumentasi → DyslexiaScreening/README.md](DyslexiaScreening/README.md)

### 3️⃣ Production Backend (Multi-Service API)
**Divisi**: Backend Engineer | **Status**: Production-ready
- **Architecture**: Decoupled microservices
- **Endpoints**: Dyslexia detection, OCR, Sentence generation (Gemini AI)
- **Deployment**: Docker + docker-compose
- **Features**: Authentication, health checks, structured logging
- 📖 [Dokumentasi → HuggingFace DyslexiaLens/README.md](HuggingFace%20DyslexiaLens/README.md)

---

## 🚀 Quick Start

### Option 1: Lokal Development (Per-divisi)

```bash
# Character Recognition
cd "Character Recognition"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Run notebook atau FastAPI

# DyslexiaScreening
cd "../DyslexiaScreening"
# Same setup...

# HuggingFace DyslexiaLens
cd "../HuggingFace DyslexiaLens"
# Same setup...
```

### Option 2: Production Deployment (Docker)

```bash
cd "HuggingFace DyslexiaLens"
docker-compose up -d

# API running at: http://localhost:7860
# Docs at: http://localhost:7860/docs
```

---

## 📦 Model Management

### Akses Model dari Google Drive

🔗 **[Model Storage - Google Drive](https://drive.google.com/drive/folders/1FHs447QrYQK6CAnXrz_rIa2Zcc6LnjWW?usp=sharing)**

- **Akun Wajib**: capstone@student.devacademy.id
- **Models**:
  - `dyslexialens_model.keras` - Dyslexia Detection
  - `ocr_model_v4.keras` - OCR EMNIST

Load model:
```python
from tensorflow import keras
model = keras.models.load_model('models/dyslexialens_model.keras')
```

---

## 🛠 Technology Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | FastAPI + Uvicorn |
| **ML/DL** | TensorFlow 2.16.1, Keras |
| **Vision** | OpenCV |
| **LLM** | Google Gemini 1.5 Flash |
| **Container** | Docker, Docker Compose |
| **Data** | Pydantic, Pandas |
| **Notebooks** | Jupyter, Kaggle |

---

## 📋 Environment Configuration

Setiap divisi memiliki `.env.example`. Copy ke `.env` dan konfigurasi:

**HuggingFace DyslexiaLens** (yang paling penting):
```bash
GEMINI_API_KEY=your_google_ai_studio_key
DYSLEXIA_MODEL_PATH=./models/dyslexialens_model.keras
OCR_MODEL_PATH=./models/ocr_model_v4.keras
```

⚠️ **Security**: Never commit `.env` dengan credentials aktif!

---

## 📞 Development Teams

| Divisi | Peran | Repo |
|--------|-------|------|
| 🤖 **AI Engineer** | OCR Development | `Character Recognition/` |
| 📊 **Data Scientist** | Model Training & Feature Engineering | `DyslexiaScreening/` |
| 🏗️ **Backend Engineer** | Production API & Infrastructure | `HuggingFace DyslexiaLens/` |

---

## 🔍 Repository Structure Details

### ✅ Checklist Per Divisi

#### Character Recognition (AI Engineer)
- [x] Jupyter notebooks (training, EDA)
- [x] Source code (inference, FastAPI)
- [x] EMNIST-trained models (.keras)
- [x] requirements.txt
- [x] .env.example (NO credentials)
- [x] .gitignore
- [x] README dengan link Google Drive model

#### HuggingFace DyslexiaLens (Backend Engineer)
- [x] FastAPI application (structured)
- [x] requirements.txt (production)
- [x] .env.example (NO credentials - FIXED!)
- [x] .gitignore (complete)
- [x] Dockerfile (optimized)
- [x] docker-compose.yml (multi-service)
- [x] Comprehensive README

---

## 🎓 How to Use This Repository

1. **For Training**: Buka notebook di divisi masing-masing
2. **For Inference**: Run FastAPI server di divisi masing-masing
3. **For Production**: Deploy dengan docker-compose di `HuggingFace DyslexiaLens/`
4. **For Integration**: Call API endpoints dari backend kemudian dipanggil lagi dari frontend

---

## 📝 Documentation

Setiap direktori divisi memiliki **README lengkap** dengan:
- Deskripsi project
- Setup environment
- Cara menjalankan
- Technology stack
- Notes & troubleshooting

---

## ⚠️ Important Notes

1. **Model Files**: Disimpan di Google Drive (gitignore)
2. **API Keys**: Gunakan `.env` untuk credentials (never commit!)
3. **Development**: Setiap divisi bisa develop independently
4. **Production**: Deploy melalui `HuggingFace DyslexiaLens/` dengan docker
5. **Collaboration**: Gunakan API contracts (Pydantic schemas)

---

## 📞 Support

Untuk issues atau questions:
- Check README di divisi masing-masing
- Review `.env.example` untuk konfigurasi
- Verify requirements.txt dependencies

**Last Updated**: 2026-06-05
