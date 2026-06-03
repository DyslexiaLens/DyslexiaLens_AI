import os
import base64
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# ─────────────────────────────────────────
# 1. DEFINISI CUSTOM LAYER CORNERSTONES
# ─────────────────────────────────────────
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
        scale = self.gap(x)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        scale = self.reshape(scale)
        return x * scale

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
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[-1] == 1):
            y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), tf.cast(num_classes, tf.int32))
        smooth_labels = y_true * (1.0 - self.smoothing) + self.smoothing / num_classes
        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        loss = -tf.reduce_sum(smooth_labels * log_probs, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"smoothing": self.smoothing})
        return cfg

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

# Map index EMNIST Anda ke Karakter Nyata (Sesuaikan urutan ini dengan label training Anda!)
EMNIST_MAPPING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 

# ─────────────────────────────────────────
# 2. INITIALIZE FASTAPI & LOAD MODEL
# ─────────────────────────────────────────
app = FastAPI(title="EMNIST Complete Grid Sheet OCR API")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY = os.environ.get("MY_SECRET_API_KEY", "KunciRahasia123!")

model = keras.models.load_model(
    "best_model.keras",
    custom_objects={
        "ChannelAttention": ChannelAttention,
        "LabelSmoothingCrossEntropy": LabelSmoothingCrossEntropy,
        "mish": mish
    }
)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY: return api_key_header
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")

# Skema Request Input untuk menerima gambar Base64 dari Web Frontend
class ImageInput(BaseModel):
    image_base64: str

@app.get("/")
def home():
    return {"status": "OCR Engine Active"}

# ─────────────────────────────────────────
# 3. ENDPOINT INFERENCE DENGAN LOGIKA GRID
# ─────────────────────────────────────────
@app.post("/predict-sheet", dependencies=[Security(get_api_key)])
def predict_sheet(payload: ImageInput):
    try:
        # Decode gambar base64 yang dikirim frontend ke format OpenCV Mat
        encoded_data = payload.image_base64.split(",")[-1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Format gambar tidak valid.")

        # Preprocessing ke Grayscale dan Thresholding biner
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # --- LOGIKA SEPARASI GRID ANDA ---
        scale = 28 
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        clean_grid_mask = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0)
        clean_grid_mask = cv2.threshold(clean_grid_mask, 10, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(clean_grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 20 < w < 200 and 20 < h < 200:
                valid_boxes.append((x, y, w, h))

        if not valid_boxes:
            return {"status": "success", "text": ""}

        # Sort & Grouping kotak ke baris-baris (Row-by-Row)
        valid_boxes = sorted(valid_boxes, key=lambda b: b[1])
        rows = []
        current_row = []
        prev_y = valid_boxes[0][1]
        cell_height_threshold = 20  

        for box in valid_boxes:
            if abs(box[1] - prev_y) > cell_height_threshold:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
                prev_y = box[1]
            else:
                current_row.append(box)
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        # --- PROSES EVALUASI KOTAK BLANK VS MODEL PREDICT ---
        extracted_text = []

        for row in rows:
            space_allocated = False
            has_content_yet = False
            row_text = ""
            
            for x, y, w, h in row:
                padding = 5
                cropped_thresh_cell = thresh[y+padding : y+h-padding, x+padding : x+w-padding]
                
                # Proteksi ukuran jika cropping di tepi gambar bermasalah
                if cropped_thresh_cell.size == 0:
                    continue
                    
                text_pixel_count = cv2.countNonZero(cropped_thresh_cell)
                total_pixels = cropped_thresh_cell.shape[0] * cropped_thresh_cell.shape[1]
                text_ratio = (text_pixel_count / total_pixels) * 100
                
                is_blank = text_ratio <= 1.5
                
                if not is_blank:
                    # ADA ISI -> Resize kotak ke (28, 28) untuk dimasukkan ke model Keras
                    cell_resized = cv2.resize(cropped_thresh_cell, (28, 28), interpolation=cv2.INTER_AREA)
                    
                    # Normalisasi data pixel 0.0 - 1.0
                    cell_input = cell_resized.astype(np.float32) / 255.0
                    cell_input = np.expand_dims(cell_input, axis=(0, -1)) # Shape menjadi (1, 28, 28, 1)
                    
                    # Prediksi menggunakan model
                    logits = model.predict(cell_input, verbose=0)
                    pred_idx = np.argmax(logits, axis=-1)[0]
                    
                    # Terjemahkan index angka ke karakter asli string
                    char_result = EMNIST_MAPPING[pred_idx] if pred_idx < len(EMNIST_MAPPING) else "?"
                    row_text += char_result
                    
                    has_content_yet = True
                    space_allocated = False 
                else:
                    # KOTAK BLANK -> Simpan spasi hanya untuk space pertama pasca konten teks
                    if has_content_yet and not space_allocated:
                        row_text += " "  # <-- DI SINI JADI SPASI
                        space_allocated = True
            
            if row_text.strip():  # Masukkan baris jika tidak sepenuhnya kosong
                extracted_text.append(row_text.rstrip())

        # Gabungkan semua baris teks dengan pemisah baris baru (\n)
        final_output_string = "\n".join(extracted_text)

        return {
            "status": "success",
            "result_text": final_output_string
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses kertas grid: {str(e)}")