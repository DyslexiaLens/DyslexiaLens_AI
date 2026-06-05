from pydantic import BaseModel, Field


class OcrPredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded grid sheet image")


class OcrPredictResponse(BaseModel):
    status: str
    result_text: str


class OcrGridPredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded handwritten grid form image")


class OcrGridPredictResponse(BaseModel):
    status: str
    total_rows_detected: int
    result_text: str
