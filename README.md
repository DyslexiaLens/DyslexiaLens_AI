
# DyslexiaLens — Pipeline Inferensi AI & Infrastruktur Backend Produksi

Berkas ini adalah dokumentasi teknis utama untuk infrastruktur backend proyek **DyslexiaLens** (Dicoding Capstone CC26-PSU052). Repositori ini mengisolasi seluruh logika mesin kecerdasan buatan, pemrosesan gambar, dan perutean API ke dalam arsitektur siap produksi berbasis Docker.


## Ringkasan Fitur Utama Sistem AI
Aplikasi backend ini melayani tiga fungsi klinis utama yang bekerja secara independen melalui arsitektur terpisah (*Decoupled Architecture*):

1. **Deteksi Disleksia Multi-Modal (Model A):** Menggunakan arsitektur *Late Fusion CNN* (Keras Functional API) yang menerima input ganda berupa potongan gambar huruf dan 6 fitur geometri mekanis tulisan tangan. Model ini mengeksekusi *Multi-Task Learning* untuk mengeluarkan status klasifikasi biner sekaligus estimasi tingkat keparahan (*Severity Score*).
2. **Ekstraksi Grid & Lembar Kerja OCR (Model B):** Menggunakan kombinasi pemrosesan citra tradisional (OpenCV) untuk mendeteksi koordinat lembar kerja fisik, memotong sel karakter secara otomatis, dan menerjemahkannya menjadi teks (A-Z) menggunakan model *Optical Character Recognition* (OCR).
3. **Sintesis Kalimat Latihan Klinis (Generative AI):** Mengintegrasikan Gemini 1.5 Flash untuk menciptakan kalimat tes secara dinamis. Dilengkapi dengan *validation loop* internal berbasis Python untuk memastikan teks mematuhi batas tata letak kertas yang ketat (tepat 5 kata, maksimal 8 karakter per kata).


## Struktur Berkas Proyek

```text
.
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/      # Handler API (ai.py, dyslexia.py, ocr.py)
│   │       └── router.py       # Gerbang perutean utama API v1
│   ├── core/                   # Utilitas keamanan dan pembacaan konfigurasi
│   ├── schemas/                # Skema validasi data input/output (Pydantic)
│   ├── services/               # Logika inti inferensi model dan interaksi API Cloud
│   └── utils/                  # Skrip pembantu pemrosesan citra (OpenCV)
├── models/
│   ├── dyslexialens_model.keras # Bobot model pendeteksi utama (Multi-Task)
│   └── ocr_model_v4.keras       # Bobot model pengenal abjad EMNIST
├── Dockerfile                  # Konstruksi kontainer isolasi lingkungan sistem
├── requirements.txt            # Daftar pustaka dan dependensi Python produksi
├── .env.example                # Cetak biru variabel lingkungan (API Key)
└── README.md                   # Dokumentasi teknis proyek

```

---

## Panduan Memulai & Pengembangan Lokal

### 1. Pengaturan Variabel Lingkungan

Salin berkas `.env.example` menjadi `.env` dan masukkan token rahasia Google AI Studio Anda:

```bash
cp .env.example .env
# Buka file .env dan isi: GEMINI_API_KEY="Kunci_Rahasia_Gemini_Anda"

```

### 2. Eksekusi Menggunakan Docker (Rekomendasi Produksi)

Untuk membangun lingkungan sistem yang terisolasi tanpa perlu menginstal TensorFlow secara lokal di komputer Anda:

```bash
# Membangun image kontainer
docker build -t dyslexialens-backend .

# Menjalankan kontainer pada port 8000
docker run -d -p 8000:8000 --env-file .env dyslexialens-backend

```

Setelah aktif, dokumentasi API interaktif dapat diakses langsung melalui peramban di tautan `http://localhost:8000/docs` (Swagger UI).
