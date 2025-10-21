"""
Live Nail Segmentation (TFLite YOLOv8-seg logic) with Streamlit + streamlit-webrtc.

Place your TFLite model at repo root (same folder as this app) and set MODEL_FILENAME below.
"""

import os
import time
from typing import Tuple
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

# webrtc imports
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---------------- CONFIG / MODEL FILE ----------------
MODEL_FILENAME = "nails_seg_s_yolov8_v1_float32.tflite"  # change to your exact filename if different
MODEL_PATH = os.path.join(".", MODEL_FILENAME)

# Default processing params (same logic as your script)
NAIL_COLOR = (30, 30, 200)   # BGR
ALPHA = 0.5
MIN_CONTOUR = 100
CHAIKIN_ITERS = 5
DILATION_PIXELS = 5
FEATHER = 5
CONF_THRESHOLD = 0.25

# Optional: RTC config for streamlit-webrtc (use default TURN if not needed)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ---------------- HELPERS (EXACT SAME LOGIC YOU PROVIDED) ----------------
def chaikin_smooth(pts, iterations=3):
    pts = np.array(pts, dtype=np.float32)
    for _ in range(iterations):
        new = []
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i+1) % len(pts)]
            Q = 0.75*p0 + 0.25*p1
            R = 0.25*p0 + 0.75*p1
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
        smooth = chaikin_smooth(hull[:,0,:], CHAIKIN_ITERS)
        cv2.fillPoly(refined, [smooth], 255)
    return refined

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------- Video Processor (runs in worker thread) ----------------
class NailProcessor(VideoProcessorBase):
    def __init__(self, model_path: str, params: dict):
        # Load interpreter inside the worker so it's local to processor thread
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Copy parameters (so UI updates won't affect ongoing processing until restart)
        self.params = params.copy()

        # small morphological kernel to clean specks
        self.kernel_small = np.ones((3,3), np.uint8)

    def recv(self, frame):
        """
        frame: av.VideoFrame
        Returns: av.VideoFrame processed
        """
        img_bgr = frame.to_ndarray(format="bgr24")
        H_orig, W_orig = img_bgr.shape[:2]

        try:
            # Preprocess (same as your script)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            img_normalized = img_resized.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img_normalized, axis=0)

            # Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()

            # Read outputs (assume same layout: detections then mask_protos)
            detection_output = self.interpreter.get_tensor(self.output_details[0]['index'])
            mask_protos = self.interpreter.get_tensor(self.output_details[1]['index'])

            # Parse
            detection_output = np.transpose(detection_output, (0, 2, 1))[0]  # [N, 37] style
            boxes = detection_output[:, :4]
            scores = detection_output[:, 4]
            mask_coeffs = detection_output[:, 5:]

            # filter by CONF
            conf_th = self.params.get("CONF_THRESHOLD", CONF_THRESHOLD)
            valid = scores > conf_th
            boxes = boxes[valid]
            scores = scores[valid]
            mask_coeffs = mask_coeffs[valid]

            # if no detections, just return original frame
            if len(boxes) == 0:
                return frame

            # prepare protos
            mask_protos = mask_protos[0]  # [160,160,32] (expected)

            overlay = img_bgr.copy()

            # Per-detection mask creation (follows your script)
            for i in range(len(boxes)):
                mask_coef = mask_coeffs[i]  # [32]
                mask = np.matmul(mask_protos, mask_coef)  # [160,160]
                mask = sigmoid(mask)
                mask_640 = cv2.resize(mask, (640, 640))

                cx, cy, w_box, h_box = boxes[i]
                x1 = int((cx - w_box/2) * 640)
                y1 = int((cy - h_box/2) * 640)
                x2 = int((cx + w_box/2) * 640)
                y2 = int((cy + h_box/2) * 640)

                full_mask = np.zeros((640, 640), dtype=np.float32)
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(640, x2), min(640, y2)

                if x2c > x1c and y2c > y1c:
                    # copy region from mask_640
                    full_mask[y1c:y2c, x1c:x2c] = cv2.resize(mask_640[y1c:y2c, x1c:x2c], (x2c - x1c, y2c - y1c))
                else:
                    full_mask = mask_640

                # to uint8 and resize to original frame
                raw = (full_mask * 255).astype(np.uint8)
                raw = cv2.resize(raw, (W_orig, H_orig))

                # threshold
                _, raw = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)

                # clean tiny specks -> optional small open
                raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, self.kernel_small, iterations=1)

                # refine contours using your smooth_mask (Chaikin + convex hull)
                refined = smooth_mask(raw)

                # dilation for better coverage
                dil = self.params.get("DILATION_PIXELS", DILATION_PIXELS)
                kernel = np.ones((dil, dil), np.uint8)
                refined = cv2.dilate(refined, kernel, iterations=1)

                # feather for blending
                feather_k = max(1, self.params.get("FEATHER", FEATHER) if self.params.get("FEATHER", FEATHER) % 2 == 1 else self.params.get("FEATHER", FEATHER) + 1)
                alpha_mask = cv2.GaussianBlur(refined, (feather_k, feather_k), 0).astype(np.float32) / 255.0

                color_layer = np.zeros_like(overlay, dtype=np.uint8)
                color_layer[:, :] = self.params.get("NAIL_COLOR", NAIL_COLOR)
                alpha_3 = alpha_mask[:, :, None]

                # overlay composition — follow your original blending math (keeps overlay BGR)
                overlay = (overlay * (1 - alpha_3) + color_layer * alpha_3 * self.params.get("ALPHA", ALPHA)).astype(np.uint8)

            # Return processed frame as av.VideoFrame (bgr24)
            return av.VideoFrame.from_ndarray(overlay, format="bgr24")

        except Exception as e:
            # if any error, fallback to original frame (prevents worker crash)
            print("Processing error:", e)
            return frame

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="💅 Live Nail Segmentation", layout="wide")
st.title("💅 Live Nail Segmentation (TFLite YOLOv8-seg logic)")

# Basic model check
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at repo root: {MODEL_PATH}\nUpload your TFLite model with this name or change MODEL_FILENAME in the script.")
    st.stop()

# Sidebar controls (live-adjustable only before starting; processor captures snapshot of params on start)
st.sidebar.header("Processing Settings")
conf_th = st.sidebar.slider("Confidence threshold", 0.0, 1.0, float(CONF_THRESHOLD), 0.01)
alpha = st.sidebar.slider("Overlay alpha", 0.0, 1.0, float(ALPHA), 0.05)
dilation = st.sidebar.slider("Dilation (px)", 1, 25, int(DILATION_PIXELS), 1)
feather = st.sidebar.slider("Feather kernel (odd)", 1, 31, int(FEATHER), 2)
nail_color = st.sidebar.color_picker("Nail color (HEX)", "#1e1eb8")  # hex => convert to BGR shortly
# convert hex to BGR tuple
hexc = nail_color.lstrip("#")
nail_color_bgr = (int(hexc[4:6], 16), int(hexc[2:4], 16), int(hexc[0:2], 16))

st.sidebar.markdown("---")
st.sidebar.info("Start the live webcam stream below. The processor uses the same mask-refinement logic (chaikin smoothing, convex hull, dilation, feathering).")

# prepare params dict for the worker
params = {
    "CONF_THRESHOLD": conf_th,
    "ALPHA": alpha,
    "DILATION_PIXELS": dilation,
    "FEATHER": feather,
    "NAIL_COLOR": nail_color_bgr
}

# Start live stream with processor factory (processor created with model loaded inside worker)
webrtc_ctx = webrtc_streamer(
    key="nail-seg-live",
    mode="live",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: NailProcessor(MODEL_PATH, params),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

st.markdown("---")
st.markdown("**Tips & Notes**:")
st.write("- This uses the same exact mask creation & refinement steps you provided (sigmoid on protos*coeffs, resize, threshold, Chaikin smoothing, convex hull).")
st.write("- If masks look odd: try lowering `Dilation` or increasing `FEATHER`, and ensure the correct TFLite file (`MODEL_FILENAME`) is used (float32 vs float16).")
st.write("- If your model output shapes differ, please paste the printed shapes from a local inference run and I will adapt parsing accordingly.")
