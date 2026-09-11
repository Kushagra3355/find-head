import os
import sys

# Ensure UTF-8 output on Windows consoles to prevent charmap encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import webbrowser
import threading
from pathlib import Path
from typing import Optional
import urllib.request

# Suppress TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Define base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "vertex_classifier_quant.tflite"
INDEX_PATH = BASE_DIR / "index.html"

# Initialize FastAPI application with Swagger/OpenAPI disabled
app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and allocate TFLite model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Warm up model with a zero tensor so the first request is instant
try:
    warmup_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], warmup_tensor)
    interpreter.invoke()
except Exception as e:
    print(f"Warning during model warmup: {e}")

DEFAULT_THRESHOLD = 0.70  # 85% confidence required to pass


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serves the main interactive testing frontend."""
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(str(INDEX_PATH), media_type="text/html")


@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend status and model readiness."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "input_shape": input_details[0]['shape'].tolist(),
        "default_threshold": DEFAULT_THRESHOLD
    }


@app.post("/validate")
async def validate_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = None
):
    """
    Validates whether the uploaded photo is a valid vertex scalp image.
    Accepts image file (JPEG, PNG, WEBP, etc.) and optional threshold.
    """
    if isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0:
        cutoff = float(threshold)
    else:
        cutoff = DEFAULT_THRESHOLD

    # 1. Validate content type or fallback to image reading
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to decode image file. Please provide a valid JPEG, PNG, or WebP.")

    # 2. Resize to exact dimensions expected by model (224x224)
    image = image.resize((224, 224), Image.Resampling.BILINEAR)

    # 3. Preprocess the image (Let TensorFlow handle the math)
    input_data = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    input_data = tf.keras.applications.mobilenet_v3.preprocess_input(input_data)

    # 4. Measure inference latency and run prediction
    start_time = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # 5. Extract prediction confidence score
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    confidence = float(prediction)
    # Clamp confidence to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))
    is_valid = bool(confidence >= cutoff)

    # 6. Return structured diagnostic response
    return {
        "status": "APPROVED" if is_valid else "REJECTED",
        "is_valid": is_valid,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(confidence, 6),
        "threshold": round(cutoff, 4),
        "inference_time_ms": inference_time_ms,
        "message": (
            "Valid vertex scalp scan detected."
            if is_valid
            else "Image rejected. Angle, lighting, or region does not meet vertex scalp standards."
        )
    }


def launch_browser(url: str = "http://127.0.0.1:8000"):
    """Waits until the server is ready, then automatically opens the frontend in the default browser."""
    for _ in range(50):
        time.sleep(0.4)
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass

    print(f"[+] Opening frontend automatically in default browser: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[!] Could not open browser automatically: {e}")


if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000
    server_url = f"http://{host}:{port}"

    # Launch browser in a background daemon thread
    threading.Thread(target=launch_browser, args=(server_url,), daemon=True).start()

    print("\n" + "=" * 65)
    print("   SCALP IMAGE VALIDATOR (FastAPI Backend & Testing UI)   ")
    print(f"   -> Web Frontend : {server_url}")
    print("   -> Status       : Auto-opening frontend in your browser...")
    print("=" * 65 + "\n")

    # Start the FastAPI server via Uvicorn
    uvicorn.run(app, host=host, port=port)