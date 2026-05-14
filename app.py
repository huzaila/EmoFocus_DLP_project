import pandas as pd
import numpy as np
import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
import plotly.graph_objects as go
import time
import threading
from focus_engine import FocusEngine

EMOTIONS    = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH  = "emofocus_model.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
@st.cache_resource
def load_model():
    import timm
    backbone    = timm.create_model('resnet50', pretrained=False, num_classes=0)
    in_features = backbone.num_features

    head = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 7)
    )

    class EmotionModel(nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.backbone = b
            self.head     = h
        def forward(self, x):
            return self.head(self.backbone(x))

    model = EmotionModel(backbone, head)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

# Face detector 
@st.cache_resource
def get_face_detector():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(min_detection_confidence=0.5)

# Transform 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Helpers
def crop_face(frame, detector):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if results.detections:
        d  = results.detections[0].location_data.relative_bounding_box
        h, w = frame.shape[:2]
        x1 = max(0, int(d.xmin * w))
        y1 = max(0, int(d.ymin * h))
        x2 = min(w, x1 + int(d.width * w))
        y2 = min(h, y1 + int(d.height * h))
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None

def predict_emotion(model, face_img):
    tensor = transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out   = model(tensor)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    return EMOTIONS[probs.argmax()], probs

# Background camera thread 
def camera_loop(stop_event, model, detector, engine, start_time, state):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        emotion    = "neutral"
        face, bbox = crop_face(frame, detector)

        if face is not None and face.size > 0:
            try:
                emotion, _ = predict_emotion(model, face)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion.upper(), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except:
                pass

        elapsed              = time.time() - start_time
        focus_score, alert   = engine.update(emotion, elapsed)

        state["frame"]   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        state["emotion"] = emotion
        state["score"]   = focus_score
        state["timeline"].append((elapsed, emotion, focus_score))
        if alert:
            state["alert"] = alert

        time.sleep(0.05)

    cap.release()

# Session state defaults 
defaults = {
    "running":       False,
    "engine":        FocusEngine(),
    "start_time":    None,
    "last_alert":    None,
    "chart_counter": 0,
    "shared": {
        "frame": None, "emotion": "neutral",
        "score": 50.0, "alert": None, "timeline": []
    },
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Page layout
st.set_page_config(page_title="EmoFocus", layout="wide")
st.title("🎓 EmoFocus Student Engagement Monitor")

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    start_btn = st.button("▶ Start Session", use_container_width=True)
with btn_col2:
    stop_btn  = st.button("⏹ Stop Session",  use_container_width=True)

col1, col2 = st.columns([2, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    score_placeholder   = st.empty()
    emotion_placeholder = st.empty()
    alert_placeholder   = st.empty()

timeline_placeholder = st.empty()

# Start 
if start_btn and not st.session_state.running:
    st.session_state.running       = True
    st.session_state.engine        = FocusEngine()
    st.session_state.start_time    = time.time()
    st.session_state.last_alert    = None
    st.session_state.chart_counter = 0
    st.session_state.shared = {
        "frame": None, "emotion": "neutral",
        "score": 50.0, "alert": None, "timeline": []
    }

    stop_event = threading.Event()
    st.session_state.stop_event = stop_event

    t = threading.Thread(
        target=camera_loop,
        args=(stop_event, load_model(), get_face_detector(),
              st.session_state.engine, st.session_state.start_time,
              st.session_state.shared),
        daemon=True
    )
    t.start()
    st.session_state.thread = t

# Stop 
if stop_btn and st.session_state.running:
    st.session_state.running = False
    if "stop_event" in st.session_state:
        st.session_state.stop_event.set()
    st.rerun()

# Live display 
if st.session_state.running:
    shared  = st.session_state.shared
    frame   = shared.get("frame")
    emotion = shared.get("emotion", "neutral")
    score   = shared.get("score", 50.0)
    alert   = shared.get("alert")
    tl      = list(shared.get("timeline", []))

    if frame is not None:
        frame_placeholder.image(frame, channels="RGB", width=640)
    else:
        frame_placeholder.info("Waiting for camera...")

    score_placeholder.metric("Focus Score",      f"{score:.1f}/100")
    emotion_placeholder.metric("Current Emotion", emotion.capitalize())

    if alert:
        alert_placeholder.warning(f"💡 {alert}")
        shared["alert"] = None

    if len(tl) > 1:
        st.session_state.chart_counter += 1
        times  = [t[0] for t in tl[-100:]]
        scores = [t[2] for t in tl[-100:]]
        fig = go.Figure(go.Scatter(
            x=times, y=scores, mode='lines',
            line=dict(color='royalblue', width=2)
        ))
        fig.update_layout(
            title="Focus Score Over Time",
            xaxis_title="Time (s)", yaxis_title="Focus Score",
            yaxis=dict(range=[0, 100]), height=250, margin=dict(t=40)
        )
        timeline_placeholder.plotly_chart(
            fig, key=f"chart_{st.session_state.chart_counter}"
        )

    time.sleep(0.1)
    st.rerun()

# Session Summary 
if not st.session_state.running and st.session_state.start_time:
    shared   = st.session_state.shared
    total    = time.time() - st.session_state.start_time
    summary  = st.session_state.engine.get_summary(total)
    dominant = summary.get("dominant_emotion", "neutral")
    tip      = summary.get("session_tip", "")

    st.subheader("📋 Session Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration",         f"{int(total//60)}m {int(total%60)}s")
    c2.metric("Overall Focus",    f"{summary['overall_focus']}/100")
    c3.metric("Alerts Triggered", summary["alert_count"])

    if summary.get("emotion_times"):
        st.write("**Time per Emotion (seconds):**")
        st.bar_chart(summary["emotion_times"])

    # Post-session recommendation 
    if tip:
        st.markdown("---")
        st.subheader("📌 Session Recommendation")
        st.info(f"**Dominant emotion this session: {dominant.capitalize()}**\n\n{tip}")

    # Real-time alert log 
    if summary["alerts"]:
        st.markdown("---")
        st.write("**Real-Time Alert Log:**")
        for ts, msg in summary["alerts"]:
            st.write(f"• {int(ts)}s — {msg}")