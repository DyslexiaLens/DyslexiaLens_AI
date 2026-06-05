# Character Recognition - DyslexiaLens AI

Module OCR untuk mengenali karakter/huruf secara otomatis menggunakan model deep learning berbasis CNN.

## Deskripsi Project

Project ini fokus pada:
- **Training model**: Melatih model CNN untuk mengenali karakter (EMNIST dataset)
- **Inference**: Melakukan prediksi karakter dari gambar
- **API Inference**: Menyediakan FastAPI untuk integration dengan backend

## Struktur Direktori

```
Character Recognition/
├── notebook/                  # Jupyter notebooks untuk training & EDA
│   ├── dicoding-capstone.ipynb
│   └── dicoding-capstone (1-3).ipynb
├── inference and fastAPI/    # Source code inference & API
│   ├── app.py               # FastAPI application (OCR inference)
│   ├── app2.py              # Alternative FastAPI implementation
│   └── inference.py         # Inference utilities
├── models/                   # Pre-trained model weights
│   ├── best_model.keras
│   ├── best_model (2).keras
│   └── best_model (4).keras
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template (no credentials)
└── README.md               # This file
```

## Setup Environment

### 1. Clone dan Setup

```bash
cd "Character Recognition"
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
# Edit .env sesuai kebutuhan Anda
```

## Cara Menjalankan

### Training Model

```bash
cd notebook/
jupyter notebook dicoding-capstone.ipynb
```

### Inference API (FastAPI)

```bash
cd "inference and fastAPI"
uvicorn app:app --reload --port 8000
```

Akses dokumentasi API: `http://localhost:8000/docs`

## Model

Model disimpan di Google Drive dan dapat diakses melalui link berikut:

🔗 **[Model Download Link - Google Drive](https://drive.google.com/drive/folders/1FHs447QrYQK6CAnXrz_rIa2Zcc6LnjWW?usp=sharing)**

### Cara Load Model

```python
from tensorflow import keras
model = keras.models.load_model('models/best_model.keras')
```

### Akses Model

Pastikan akun **capstone@student.devacademy.id** memiliki akses read untuk folder model di Google Drive.

## Technologies

- TensorFlow 2.16.1 (Keras)
- FastAPI
- OpenCV
- NumPy, Pandas
- scikit-learn

## Notes

- Model dilatih menggunakan EMNIST dataset
- Dukungan GPU tersedia dengan `tensorflow-gpu`
- API berjalan pada port 8000 secara default
