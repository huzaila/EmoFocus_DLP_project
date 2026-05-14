# EmoFocus — Real-Time Student Engagement Monitor

A deep learning system that monitors student engagement during study sessions 
by classifying facial emotions through a live webcam feed. Built as a semester 
project for Deep Learning (2026).

**Group:** Maham Mohsin · Jawairia Hammad · Huzaila Asif

---

## What It Does

EmoFocus reads a student's facial expressions in real time, classifies them 
into 7 emotion categories, and computes a rolling focus score (0–100) that 
updates continuously throughout the session. When the system detects sustained 
frustration or disengagement, it fires a contextual alert. A Streamlit dashboard 
shows the live feed, focus score, emotion timeline, and a full session summary.

No language models or external APIs are used at any stage — the system is 
entirely computer vision based and runs fully locally on the student's machine. 
Webcam frames are never saved to disk; only emotion labels are logged.

---

## Model

- **Architecture:** ResNet-50 (ImageNet pretrained) with a custom classification head  
  — FC layers of 1024 and 512 units, batch normalisation, dropout at 0.5 / 0.3
- **Dataset:** FER-2013 — 35,887 grayscale images across 7 emotion classes
- **Training:** Two-phase fine-tuning over 40 epochs on Google Colab T4 GPU  
  — Phase 1: head only; Phase 2: full backbone unfrozen  
  — AdamW with cosine annealing, label smoothing 0.1, gradient clipping 1.0  
  — Differential learning rates: 3e-5 backbone / 3e-4 head  
  — Weighted random sampling to handle Disgust class imbalance (436 vs 7,215 Happy samples)
- **Input:** Grayscale → 3-channel, resized to 112×112, ImageNet normalised
- **Test accuracy:** 73% on FER-2013 held-out test set (7,178 images)

---

## Focus Scoring Engine

Each detected emotion is mapped to a weight. A rolling window of the last 30 
detections computes the focus score. If a negative-weight emotion persists for 
10 or more consecutive frames, a contextual alert fires.

| Emotion  | Weight | Interpretation       |
|----------|--------|----------------------|
| Happy    | +1.0   | High engagement      |
| Neutral  | +0.5   | Normal study state   |
| Surprise | +0.3   | Active attention     |
| Fear     | −0.3   | Possible confusion   |
| Sad      | −0.5   | Low engagement       |
| Angry    | −0.7   | Frustration          |
| Disgust  | −0.8   | Strong disengagement |

---

## Real-Time Pipeline
Webcam → Face Detection (OpenCV Haar Cascade) → Preprocessing →
ResNet-50 Inference → Focus Engine → Streamlit Dashboard

- Capture: OpenCV at 640×480, ~20 fps  
- Face detection: OpenCV Haar Cascade (frontal face)  
- Background camera thread keeps frames flowing independently of Streamlit reruns  

---

## Stack

- PyTorch + torchvision — model and inference  
- OpenCV — webcam capture and face detection  
- Streamlit — live dashboard  
- Plotly — focus score timeline chart  

---

## Setup

```bash
pip install torch torchvision streamlit opencv-python plotly
```

Place `emofocus_model.pth` in the same folder as `app.py`, then:

```bash
streamlit run app.py
```

---

## Results

| Method               | Backbone       | Accuracy | Real-Time | Dashboard | Alerts |
|----------------------|----------------|----------|-----------|-----------|--------|
| EmoFocus (final)     | ResNet-50      | 73%      | Yes       | Yes       | Yes    |
| EmoFocus (baseline)  | EfficientNet-B0| 48%      | Yes       | Yes       | Yes    |
| Khaireddin & Chen '21| VGG-13         | 73.3%    | No        | No        | No     |
| Minaee et al. '21    | AttentionNet   | 70.0%    | No        | No        | No     |
| Human baseline       | —              | 65%      | —         | —         | —      |


## Download Model
Download `emofocus_model.pth` from [Releases](../../releases) 
and place it in the same folder as `app.py` before running.
