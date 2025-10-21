# app_cam_snapshot.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os

MODEL_FILENAME = "nails_seg_s_yolov8_v1_float32.tflite"
MODEL_PATH = os.path.join(".", MODEL_FILENAME)

# same config as your script:
NAIL_COLOR = (30, 30, 200)
ALPHA = 0.5
MIN_CONTOUR = 100
CHAIKIN_ITERS = 5
DILATION_PIXELS = 5
FEATHER = 5
CONF_THRESHOLD = 0.25

# helpers (chaikin/smooth_mask/sigmoid) -> paste same functions you used
def chaikin_smooth(pts, iterations=3):
    pts = np.array(pts, dtype=np.float32)
    for _ in range(iterations):
        new=[]
        for i in range(len(pts)):
            p0=pts[i]
            p1=pts[(i+1)%len(pts)]
            Q=0.75*p0+0.25*p1
            R=0.25*p0+0.75*p1
            new.extend([Q,R])
        pts=np.array(new)
    return pts.astype(np.int32)

def smooth_mask(mask):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    refined=np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR:
            continue
        hull = cv2.convexHull(c)
        smooth = chaikin_smooth(hull[:,0,:], CHAIKIN_ITERS)
        cv2.fillPoly(refined, [smooth], 255)
    return refined

def sigmoid(x): return 1/(1+np.exp(-x))

@st.cache_resource
def load_interpreter(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

st.set_page_config(layout="wide", page_title="Nail Live (snapshot)")
st.title("💅 Nail Segmentation — Snapshot Live")

try:
    interpreter, input_details, output_details = load_interpreter(MODEL_PATH)
    st.success("Loaded TFLite model")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    cam_file = st.camera_input("📸 Take Snapshot (Allow camera)")
    uploaded = st.file_uploader("Or upload image", type=["jpg","jpeg","png"])

img = None
if cam_file:
    file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def process_frame(img_bgr):
    # paste your exact inference + mask logic (identical to your working script)
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640,640))
    input_tensor = np.expand_dims(img_resized.astype(np.float32)/255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    det_out = interpreter.get_tensor(output_details[0]['index'])
    mask_protos = interpreter.get_tensor(output_details[1]['index'])
    det_out = np.transpose(det_out, (0,2,1))[0]
    boxes = det_out[:, :4]
    scores = det_out[:, 4]
    mask_coeffs = det_out[:, 5:]
    valid = scores > CONF_THRESHOLD
    boxes = boxes[valid]; mask_coeffs = mask_coeffs[valid]
    if len(boxes)==0:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_protos = mask_protos[0]; overlay = img_bgr.copy()
    for i in range(len(boxes)):
        mask_coef = mask_coeffs[i]
        mask = np.matmul(mask_protos, mask_coef)
        mask = sigmoid(mask)
        mask_640 = cv2.resize(mask, (640,640))
        cx,cy,w_box,h_box = boxes[i]
        x1=int((cx-w_box/2)*640); y1=int((cy-h_box/2)*640)
        x2=int((cx+w_box/2)*640); y2=int((cy+h_box/2)*640)
        full_mask = np.zeros((640,640), dtype=np.float32)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(640,x2), min(640,y2)
        if x2>x1 and y2>y1:
            full_mask[y1:y2,x1:x2] = cv2.resize(mask_640[y1:y2,x1:x2], (x2-x1, y2-y1))
        else:
            full_mask = mask_640
        raw = (full_mask*255).astype(np.uint8)
        raw = cv2.resize(raw, (W,H))
        _, raw = cv2.threshold(raw,127,255,cv2.THRESH_BINARY)
        refined = smooth_mask(raw)
        kernel = np.ones((DILATION_PIXELS,DILATION_PIXELS), np.uint8)
        refined = cv2.dilate(refined, kernel, iterations=1)
        feather_k = max(1, FEATHER if FEATHER%2==1 else FEATHER+1)
        alpha_mask = cv2.GaussianBlur(refined, (feather_k, feather_k), 0).astype(np.float32)/255.0
        color_layer = np.zeros_like(overlay, dtype=np.uint8); color_layer[:,:] = NAIL_COLOR
        alpha_3 = alpha_mask[:,:,None]
        overlay = (overlay*(1-alpha_3) + color_layer*alpha_3*ALPHA).astype(np.uint8)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

with col2:
    if img is not None:
        st.write("Processing snapshot...")
        out = process_frame(img)
        st.image(out, use_column_width=True)
        st.download_button("Download result", data=cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))[1].tobytes(), file_name="nail_result.png", mime="image/png")
    else:
        st.info("Use camera or upload image to process.")
