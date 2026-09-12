import os
import io
import time
import threading
import webbrowser
import urllib.request
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 1. Initialize API
app = FastAPI(title="Norwood Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Norwood Model
MODEL_PATH = "model/norwood_large_tuned.tflite"
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading model: Make sure {MODEL_PATH} exists!")
    raise e

# 3. Serve Frontend HTML
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return FileResponse("frontend.html", media_type="text/html")

@app.get("/health")
async def health():
    return {"status": "ok"}

# 4. Prediction Endpoint
@app.post("/predict")
async def predict_norwood(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    
    input_data = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    input_data = tf.keras.applications.mobilenet_v3.preprocess_input(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    best_index = int(np.argmax(predictions))
    
    return {
        "norwood_stage": f"Stage {best_index + 1}",
        "confidence_percentage": round(float(predictions[best_index]) * 100, 2),
        "all_probabilities": [round(float(p) * 100, 2) for p in predictions]
    }

# 5. Auto-Launch Browser Logic
def launch_browser(url: str):
    for _ in range(30):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    # Start the browser thread right before the server starts
    threading.Thread(target=launch_browser, args=("http://127.0.0.1:5050",), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=5050)