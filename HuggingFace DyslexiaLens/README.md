# DyslexiaLens Backend API - Production Inference Pipeline

**Infrastruktur backend siap produksi** untuk sistem deteksi disleksia multi-modal dengan integrasi Gemini AI untuk generasi kalimat latihan klinis.

## 🚀 Ringkasan Fitur Utama

API ini menyediakan 3 pipeline AI independen:

1. **Dyslexia Detection (Multi-Modal)**: Late Fusion CNN menerima gambar + 6 fitur mekanis tulisan tangan → klasifikasi biner + severity score
2. **OCR Grid Extraction**: Deteksi otomatis lembar kerja (OpenCV) → ekstraksi grid karakter → recognition (EMNIST)
3. **Clinical Sentence Generation**: Gemini 1.5 Flash → kalimat latihan dinamis dengan validasi layout (5 kata, maks 8 karakter/kata)

## 📁 Struktur Direktori

```
.
├── app/
│   ├── api/v1/
│   │   ├── endpoints/          # Handlers: ai.py, dyslexia.py, ocr.py
│   │   └── router.py
│   ├── core/                   # Config & security utilities
│   ├── schemas/                # Pydantic data validation
│   ├── services/               # Core inference logic
│   ├── utils/                  # Image processing helpers
│   └── main.py                 # FastAPI app entry point
├── models/
│   ├── dyslexialens_model.keras    # Main detection model
│   └── ocr_model_v4.keras          # OCR EMNIST model
├── Dockerfile                  # Container build spec
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Python dependencies
├── .env.example               # Configuration template (NO credentials)
└── README.md                  # This file
```

## 🔧 Setup & Deployment

### Local Development

```bash
# 1. Clone & setup venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key_here

# 4. Run server
uvicorn app.main:app --reload --port 7860
```

API docs: http://localhost:7860/docs

### Docker (Recommended for Production)

```bash
# Build image
docker build -t dyslexialens-backend .

# Run container
docker run -d -p 7860:7860 --env-file .env dyslexialens-backend

# Or with docker-compose
docker-compose up -d
```

## 📋 Environment Configuration

Copy `.env.example` → `.env` dan isi:

```
GEMINI_API_KEY=your_gemini_studio_key
DYSLEXIA_MODEL_PATH=./models/dyslexialens_model.keras
OCR_MODEL_PATH=./models/ocr_model_v4.keras
```

## 🧠 API Endpoints

- `POST /api/v1/dyslexia/detect` - Deteksi disleksia
- `POST /api/v1/ocr/extract` - Ekstraksi grid & OCR
- `POST /api/v1/ai/generate-sentence` - Generate kalimat latihan
- `GET /docs` - Swagger UI interaktif

## 🛠 Technologies

- **Framework**: FastAPI + Uvicorn
- **ML/DL**: TensorFlow 2.16.1, Keras
- **Vision**: OpenCV
- **LLM**: Google Gemini 1.5 Flash
- **Container**: Docker
- **Data**: Pydantic schemas

## 📦 Dependencies

Lihat `requirements.txt` untuk daftar lengkap. Key packages:
- fastapi, uvicorn
- tensorflow-cpu
- opencv-python-headless
- google-generativeai
- numpy, scikit-learn

## ⚠️ Security Notes

- **NEVER commit `.env`** dengan credentials
- API key hanya di `.env.example` tanpa nilai (gunakan template)
- Production: gunakan env variable secret manager
- HTTPS recommended untuk deployment
