import os
import sys
import time
import webbrowser
import threading
from pathlib import Path
from typing import Optional
import urllib.request
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Suppress TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "vertex_classifier_quant.tflite"
INDEX_PATH = BASE_DIR / "index.html"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

DEFAULT_THRESHOLD = 0.70

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(str(INDEX_PATH), media_type="text/html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/validate")
async def validate_image(file: UploadFile = File(...), threshold: Optional[float] = None):
    cutoff = float(threshold) if isinstance(threshold, (int, float)) else DEFAULT_THRESHOLD
    
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to decode image file.")

    input_data = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    input_data = tf.keras.applications.mobilenet_v3.preprocess_input(input_data)

    start_time = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Extract binary prediction score
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    confidence = max(0.0, min(1.0, float(prediction)))
    is_valid = bool(confidence >= cutoff)

    return {
        "status": "APPROVED" if is_valid else "REJECTED",
        "is_valid": is_valid,
        "confidence": round(confidence * 100, 2),
        "threshold": round(cutoff, 4),
        "inference_time_ms": inference_time_ms,
        "message": (
            "Valid vertex scalp scan detected." if is_valid 
            else "Image rejected. Region does not meet vertex standards."
        )
    }

def launch_browser(url: str = "http://127.0.0.1:5003"):
    for _ in range(50):
        time.sleep(0.4)
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as resp:
                if resp.status == 200: break
        except Exception: pass
    try: webbrowser.open(url)
    except Exception: pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5003"))
    threading.Thread(target=launch_browser, args=(f"http://127.0.0.1:{port}",), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port)