import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join("model", "nails_seg_s_yolov8_v1_float16.tflite")

NAIL_COLOR = (30, 30, 200)
ALPHA = 0.5
MIN_CONTOUR = 100
CHAIKIN_ITERS = 5
DILATION_PIXELS = 5
FEATHER = 5
CONF_THRESHOLD = 0.25

# ---------------- LOAD MODEL ----------------
st.write("🔄 Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
st.success("✅ Model loaded successfully!")


# ---------------- HELPERS ----------------
def chaikin_smooth(pts, iterations=3):
    pts = np.array(pts, dtype=np.float32)
    for _ in range(iterations):
        new = []
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new.extend([Q, R])
        pts = np.array(new)
    return pts.astype(np.int32)


def smooth_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    refined = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR:
            continue
        hull = cv2.convexHull(c)
        smooth = chaikin_smooth(hull[:, 0, :], CHAIKIN_ITERS)
        cv2.fillPoly(refined, [smooth], 255)
    return refined


def process_nails(image):
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    H, W = image.shape[:2]

    # Preprocess
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    detection_output = interpreter.get_tensor(output_details[0]['index'])
    mask_protos = interpreter.get_tensor(output_details[1]['index'])

    # Parse detections
    detection_output = np.transpose(detection_output, (0, 2, 1))[0]
    boxes = detection_output[:, :4]
    scores = detection_output[:, 4]
    mask_coeffs = detection_output[:, 5:]

    valid_indices = scores > CONF_THRESHOLD
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    mask_coeffs = mask_coeffs[valid_indices]

    if len(boxes) == 0:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_protos = mask_protos[0]
    overlay = image.copy()

    for i in range(len(boxes)):
        mask_coef = mask_coeffs[i]
        mask = np.matmul(mask_protos, mask_coef)
        mask = 1 / (1 + np.exp(-mask))
        mask_640 = cv2.resize(mask, (640, 640))

        cx, cy, w_box, h_box = boxes[i]
        x1 = int((cx - w_box / 2) * 640)
        y1 = int((cy - h_box / 2) * 640)
        x2 = int((cx + w_box / 2) * 640)
        y2 = int((cy + h_box / 2) * 640)

        full_mask = np.zeros((640, 640), dtype=np.float32)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(640, x2), min(640, y2)

        if x2 > x1 and y2 > y1:
            full_mask[y1:y2, x1:x2] = cv2.resize(mask_640[y1:y2, x1:x2], (x2 - x1, y2 - y1))
        else:
            full_mask = mask_640

        raw = (full_mask * 255).astype(np.uint8)
        raw = cv2.resize(raw, (W, H))
        _, raw = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)

        refined = smooth_mask(raw)
        kernel = np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8)
        refined = cv2.dilate(refined, kernel, iterations=1)

        feather = max(1, FEATHER if FEATHER % 2 == 1 else FEATHER + 1)
        alpha_mask = cv2.GaussianBlur(refined, (feather, feather), 0).astype(np.float32) / 255.0

        color_layer = np.zeros_like(image, dtype=np.uint8)
        color_layer[:, :] = NAIL_COLOR
        alpha_3 = alpha_mask[:, :, None]
        overlay = (overlay * (1 - alpha_3) + color_layer * alpha_3 * ALPHA).astype(np.uint8)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="💅 Nail Segmentation Tester", layout="wide")

st.title("💅 Nail Segmentation Tester")
st.markdown("📸 Take a snapshot or upload an image — works smoothly without lag!")

col1, col2 = st.columns(2)

with col1:
    uploaded_img = st.camera_input("📸 Take a Snapshot") or st.file_uploader("📤 Or Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.session_state['input_image'] = img_rgb
else:
    img_rgb = st.session_state.get('input_image', None)

if img_rgb is not None:
    with st.spinner("🎨 Processing nails..."):
        output = process_nails(img_rgb)
    col2.image(output, caption="✨ Processed Result", use_container_width=True)
else:
    col2.info("Please capture or upload an image to start.")

st.markdown("---")
st.markdown("""
### 📱 How to use:
1. **📸 Click "Take a Snapshot"** — capture live image  
2. **📤 Or upload** from your gallery  
3. **⚡ Nails are processed instantly!**  
4. **🔄 Try multiple snapshots** for better angles  
""")
