import io
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────
# 1. ARSITEKTUR CUSTOM LAYER (Wajib didefinisikan agar model dapat di-load)
# ──────────────────────────────────────────────────────────────────────
class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(max(1, channels // self.reduction_ratio), activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, channels))
    def call(self, x):
        return x * self.reshape(self.fc2(self.fc1(self.gap(x))))
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction_ratio": self.reduction_ratio})
        return cfg

class LabelSmoothingCrossEntropy(keras.losses.Loss):
    def __init__(self, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing
    def call(self, y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true, tf.cast(num_classes, tf.int32))
        smooth_labels = y_true_oh * (1.0 - self.smoothing) + self.smoothing / num_classes
        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        loss = -tf.reduce_sum(smooth_labels * log_probs, axis=-1)
        return tf.reduce_mean(loss)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"smoothing": self.smoothing})
        return cfg

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

# ──────────────────────────────────────────────────────────────────────
# 2. INISIALISASI FASTAPI & LOAD MODEL
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Handwriting Grid OCR API",
    description="API untuk mendeteksi teks huruf alfabet bertulisan tangan di dalam formulir/kertas berpetak (Grid).",
    version="1.0.0"
)

MODEL_PATH = "best_model (4).keras"

try:
    model = keras.models.load_model(MODEL_PATH, custom_objects={
        "ChannelAttention": ChannelAttention, 
        "LabelSmoothingCrossEntropy": LabelSmoothingCrossEntropy, 
        "mish": mish
    })
    print(f"✓ Model '{MODEL_PATH}' berhasil dimuat ke dalam memori API.")
except Exception as e:
    print(f"✗ Gagal memuat model. Pastikan file berada di direktori yang sama. Error: {e}")
    model = None

# Skema JSON untuk Response API
class OCRResponse(BaseModel):
    status: str
    total_rows_detected: int
    result_text: str

# ──────────────────────────────────────────────────────────────────────
# 3. PIPELINE INFERENCE PIPELINE (CORE LOGIC)
# ──────────────────────────────────────────────────────────────────────
def pipeline_predict_grid(image_bytes: bytes) -> tuple[int, str]:
    """
    Memproses byte gambar raw, membagi kotak grid, dan mengekstrak karakter huruf.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model neural network belum siap atau gagal dimuat.")

    # Konversi byte gambar menjadi format matriks OpenCV (BGR)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Format file rusak atau bukan merupakan gambar yang valid.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Deteksi Grid Mask menggunakan Teknik Morfologi
    scale = 28
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
    
    mask_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_k, iterations=2)
    mask_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_k, iterations=2)
    mask = cv2.addWeighted(mask_h, 0.5, mask_v, 0.5, 0)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]

    # Ekstraksi Bounding Boxes Kontur Kotak Petak
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if 20 < cv2.boundingRect(c)[2] < 200 and 20 < cv2.boundingRect(c)[3] < 200]
    
    if not boxes:
        return 0, ""

    # Kelompokkan Kotak Berdasarkan Koordinat Baris Sumbu Y (Row-by-Row Structuring)
    boxes = sorted(boxes, key=lambda b: b[1])
    rows, cur_row, prev_y = [], [], boxes[0][1]
    
    for b in boxes:
        if abs(b[1] - prev_y) > 20:
            rows.append(sorted(cur_row, key=lambda b: b[0]))
            cur_row, prev_y = [b], b[1]
        else:
            cur_row.append(b)
    if cur_row: 
        rows.append(sorted(cur_row, key=lambda b: b[0]))

    # Looping Prediksi Karakter per Sel Grid
    extracted_text = []
    
    for row in rows:
        row_text, has_content, space_allocated = "", False, False

        for (x, y, w, h) in row:
            # Berikan padding margin kecil ke dalam sel (crop karakter murni)
            cell = thresh[y+5:y+h-5, x+5:x+w-5]
            if cell.size == 0: 
                continue

            # Hitung rasio kepadatan piksel putih untuk deteksi spasi/kotak kosong
            density = (cv2.countNonZero(cell) / (cell.shape[0] * cell.shape[1]) * 100)
            is_blank = density <= 1.5
            
            if not is_blank:
                # Normalisasi dimensi dan skala piksel menjadi (0.0 - 1.0) sesuai prasyarat EMNIST
                cell_in = cv2.resize(cell, (28, 28)).astype(np.float32) / 255.0
                cell_in = cell_in.reshape(1, 28, 28, 1)
                
                # Inference jalankan ke model CNN
                pred = model.predict(cell_in, verbose=0)
                idx = np.argmax(pred, axis=-1)[0]
                
                # Konversi indeks (0-25) menjadi representasi teks String (A-Z)
                row_text += chr(65 + idx)
                has_content, space_allocated = True, False
            elif has_content and not space_allocated:
                row_text += " "
                space_allocated = True

        if row_text.strip():
            extracted_text.append(row_text.rstrip())

    final_output_string = "\n".join(extracted_text)
    return len(rows), final_output_string

# ──────────────────────────────────────────────────────────────────────
# 4. ENDPOINT API DEFINITION
# ──────────────────────────────────────────────────────────────────────
@app.post("/predict-grid", response_model=OCRResponse)
async def predict_grid_endpoint(file: UploadFile = File(...)):
    """
    Endpoint utama untuk memprediksi berkas citra formulir grid tulisan tangan.
    Menerima file multi-part (JPG/PNG).
    """
    # Validasi format ekstensi berkas input
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Format dokumen tidak didukung. Unggah berkas gambar dalam ekstensi PNG atau JPG/JPEG."
        )

    try:
        image_bytes = await file.read()
        total_rows, text_result = pipeline_predict_grid(image_bytes)
        
        return OCRResponse(
            status="success",
            total_rows_detected=total_rows,
            result_text=text_result
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error sewaktu OCR: {str(e)}")

# Driver lokal (Opsional - untuk mengecek via terminal terminal)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)