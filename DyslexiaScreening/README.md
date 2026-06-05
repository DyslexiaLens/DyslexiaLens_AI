# DyslexiaScreening - Detection & Analysis

Module untuk deteksi dan screening disleksia berdasarkan analisis tulisan tangan dan pattern karakteristik.

## Deskripsi Project

Project ini fokus pada:
- **Feature Extraction**: Ekstraksi fitur mekanis dari tulisan tangan (stroke density, center of mass, symmetry, dll)
- **Classification**: Klasifikasi biner (Normal vs Corrected/Dyslexic)
- **Severity Scoring**: Estimasi tingkat keparahan disleksia
- **API Inference**: FastAPI untuk real-time prediction

## Struktur Direktori

```
DyslexiaScreening/
├── notebooks/               # Jupyter notebooks untuk EDA & modeling
│   ├── DyslexiaPrototype.ipynb
│   └── dicoding-capstone-reiko-emnist-v5.ipynb
├── Inference & FastAPI/     # Source code inference & API
│   ├── fast_api.py         # FastAPI application
│   ├── inference.py        # Inference logic
│   └── feature_extractor.py # Feature extraction utilities
├── models/                  # Pre-trained models
│   ├── dyslexialens_model.keras      # Main classification model
│   ├── dyslexia_model.keras
│   └── ckpt_acn_only.keras
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
└── README.md              # This file
```

## Setup Environment

### 1. Clone dan Setup

```bash
cd DyslexiaScreening
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Environment

```bash
cp .env.example .env
# Edit .env dengan setting yang sesuai
```

## Cara Menjalankan

### Exploratory Data Analysis

```bash
cd notebooks/
jupyter notebook DyslexiaPrototype.ipynb
```

### Inference API

```bash
cd "Inference & FastAPI"
uvicorn fast_api:app --reload --port 8000
```

Akses: `http://localhost:8000/docs`

## Features Extracted

- **Stroke Density**: Kepadatan stroke dalam karakter
- **Center of Mass**: Pusat gravitasi tulisan
- **Bounding Box Ratio**: Rasio dimensi bounding box
- **Stroke Transitions**: Jumlah transisi stroke
- **Horizontal Symmetry**: Analisis simetri horizontal

## Technologies

- TensorFlow 2.16.1 (Keras)
- FastAPI
- OpenCV
- scikit-learn
- SciPy

## Model Information

Model menggunakan Late Fusion CNN architecture untuk multi-input processing:
- Input: Character images + 6 mechanical features
- Output: Binary classification + severity score
- Training: Gambo EMNIST Balanced Dataset

## Notes

- API berjalan pada port 8000 default
- GPU support tersedia dengan `tensorflow-gpu`
- Feature extraction otomatis dari image input
