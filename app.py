import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow.lite as tflite


# 1. INITIALIZE FASTAPI & LOAD TFLITE MODEL

app = FastAPI(title="Head Detection Inference API")

MODEL_PATH = "model/model.tflite"

interpreter = None
input_details = None
output_details = None

if os.path.exists(MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Loaded {MODEL_PATH} successfully.")
else:
    print(f"⚠️ Warning: '{MODEL_PATH}' not found in current directory. Place your .tflite file here.")


# 2. EMBEDDED FRONTEND (HTML + INLINE CSS)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Head Detection Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
    .container { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; width: 100%; max-width: 820px; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }
    h1 { font-size: 24px; font-weight: 700; color: #38bdf8; margin-bottom: 8px; }
    p.subtitle { font-size: 14px; color: #94a3b8; margin-bottom: 24px; }
    .dropzone { border: 2px dashed #475569; border-radius: 8px; padding: 32px; text-align: center; cursor: pointer; transition: 0.2s ease; background: #0f172a; margin-bottom: 20px; }
    .dropzone:hover { border-color: #38bdf8; background: #132039; }
    input[type="file"] { display: none; }
    .btn { background-color: #38bdf8; color: #0f172a; border: none; font-weight: 600; padding: 10px 20px; border-radius: 6px; cursor: pointer; transition: 0.2s; }
    .btn:hover { background-color: #0ea5e9; }
    .btn:disabled { background-color: #475569; cursor: not-allowed; }
    .preview-container { display: flex; flex-direction: column; gap: 16px; align-items: center; margin-top: 20px; }
    .image-box { max-width: 100%; border-radius: 8px; overflow: hidden; border: 1px solid #334155; display: none; }
    .image-box img { width: 100%; height: auto; display: block; }
    .status-badge { display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 13px; font-weight: 500; margin-top: 12px; }
    .badge-success { background-color: #064e3b; color: #34d399; }
    .badge-empty { background-color: #7f1d1d; color: #f87171; }
    .loader { border: 4px solid #334155; border-top: 4px solid #38bdf8; border-radius: 50%; width: 32px; height: 32px; animation: spin 1s linear infinite; display: none; margin: 16px auto; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>

<div class="container">
  <h1>Top-of-Head Detector</h1>
  <p class="subtitle">Deploy custom SSD MobileNet / Edge TFLite model inference in real-time.</p>

  <label for="imageInput" class="dropzone" id="dropArea">
    <p style="font-weight: 500; font-size: 16px;">Click or Drag & Drop image here</p>
    <p style="font-size: 13px; color: #64748b; margin-top: 6px;">Supports JPG, PNG, WEBP</p>
    <input type="file" id="imageInput" accept="image/*" />
  </label>

  <div style="display: flex; justify-content: flex-end;">
    <button class="btn" id="runBtn" disabled>Run Inference</button>
  </div>

  <div class="loader" id="loader"></div>

  <div class="preview-container">
    <div id="status"></div>
    <div class="image-box" id="imageBox">
      <img id="resultImage" src="" alt="Detection Preview" />
    </div>
  </div>
</div>

<script>
  const fileInput = document.getElementById('imageInput');
  const runBtn = document.getElementById('runBtn');
  const dropArea = document.getElementById('dropArea');
  const loader = document.getElementById('loader');
  const imageBox = document.getElementById('imageBox');
  const resultImage = document.getElementById('resultImage');
  const statusDiv = document.getElementById('status');

  let selectedFile = null;

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      selectedFile = e.target.files[0];
      dropArea.querySelector('p').innerText = selectedFile.name;
      runBtn.disabled = false;
    }
  });

  runBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    loader.style.display = 'block';
    imageBox.style.display = 'none';
    statusDiv.innerHTML = '';
    runBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/detect', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      loader.style.display = 'none';
      runBtn.disabled = false;

      if (response.ok) {
        resultImage.src = 'data:image/jpeg;base64,' + data.image_base64;
        imageBox.style.display = 'block';

        if (data.detections_count > 0) {
          statusDiv.innerHTML = `<span class="status-badge badge-success">Detected ${data.detections_count} head(s)</span>`;
        } else {
          statusDiv.innerHTML = `<span class="status-badge badge-empty">No head detected above confidence threshold</span>`;
        }
      } else {
        alert(data.error || 'Inference error');
      }
    } catch (err) {
      loader.style.display = 'none';
      runBtn.disabled = false;
      alert('Failed to connect to detection backend.');
    }
  });
</script>

</body>
</html>
"""


# 3. ROUTES

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Renders the single-page HTML interface with inline styling."""
    return HTMLResponse(content=HTML_CONTENT)

# ==========================================
# UPDATED INFERENCE ROUTE FOR YOLOV8 TFLITE
# ==========================================
@app.post("/detect")
async def detect_heads(file: UploadFile = File(...)):
    """Receives an image, formats for NCHW/NHWC, runs TFLite inference, and draws bounding boxes."""
    if interpreter is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model file 'model.tflite' not found on server."}
        )

    # 1. Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_w, orig_h = image.size
    np_image = np.array(image)

    # 2. Determine input dimensions and channel order dynamically
    in_shape = input_details[0]['shape']  # e.g., [1, 3, 320, 320] or [1, 320, 320, 3]
    dtype = input_details[0]['dtype']

    if in_shape[1] == 3:  # NCHW format [1, 3, H, W]
        in_height, in_width = in_shape[2], in_shape[3]
        is_nchw = True
    else:  # NHWC format [1, H, W, 3]
        in_height, in_width = in_shape[1], in_shape[2]
        is_nchw = False

    # Resize image to expected dimensions
    resized = cv2.resize(np_image, (in_width, in_height))

    # Normalize image to float32 [0.0, 1.0] (Standard for YOLO)
    if dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)

    # Rearrange channels to NCHW [1, 3, H, W] if needed
    if is_nchw:
        input_data = np.transpose(input_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    
    input_data = np.expand_dims(input_data, axis=0)

    # 3. Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 4. Extract Raw YOLO Outputs
    raw_output = interpreter.get_tensor(output_details[0]['index'])
    # Output shape is typically [1, 5, num_boxes] or [1, num_boxes, 5]
    if raw_output.shape[1] < raw_output.shape[2]:
        output = raw_output[0].T  # Transpose to [num_boxes, 5]
    else:
        output = raw_output[0]

    boxes = []
    confidences = []
    confidence_threshold = 0.35
    score_threshold = 0.45
    nms_threshold = 0.50

    for pred in output:
        # pred structure: [cx, cy, w, h, class_scores...]
        scores = pred[4:]
        confidence = float(np.max(scores)) if len(scores) > 0 else float(pred[4])

        if confidence >= confidence_threshold:
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

            # Scale normalized or pixel coordinates back to original image
            if cx <= 1.0 and cy <= 1.0:
                cx, cy, w, h = cx * orig_w, cy * orig_h, w * orig_w, h * orig_h
            else:
                cx = (cx / in_width) * orig_w
                cy = (cy / in_height) * orig_h
                w = (w / in_width) * orig_w
                h = (h / in_height) * orig_h

            left = int(cx - w / 2)
            top = int(cy - h / 2)
            width = int(w)
            height = int(h)

            boxes.append([left, top, width, height])
            confidences.append(confidence)

    # 5. Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

    detections = []
    display_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    if len(indices) > 0:
        for idx in indices.flatten():
            bx, by, bw, bh = boxes[idx]
            conf = confidences[idx]

            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(orig_w, bx + bw)
            y2 = min(orig_h, by + bh)

            detections.append({
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

            # Draw green bounding box & label
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"Head: {conf:.2f}"
            cv2.putText(
                display_img,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # 6. Encode result to Base64
    _, buffer = cv2.imencode(".jpg", display_img)
    b64_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "detections_count": len(detections),
        "detections": detections,
        "image_base64": b64_image
    }


# 4. SERVER RUNNER

if __name__ == "__main__":
    print("🚀 Starting server on http://localhost:5005 ...")
    uvicorn.run("app:app", host="localhost", port=5005, reload=True)