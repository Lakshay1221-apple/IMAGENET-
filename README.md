# 👁️ Compound AI Vision System

> A high-performance, multi-engine computer vision pipeline combining object detection, open-vocabulary reasoning, and one-shot facial recognition — all running locally with real-time inference.

---

## 🚀 Overview

This project is a **Compound AI System** that integrates multiple specialized models into a unified real-time pipeline.

Instead of relying on a single monolithic model, this system distributes tasks across multiple AI engines:

* 🎯 **Object Detection (YOLO-based)**
* 🌍 **Zero-Shot Open Vocabulary Detection**
* 🧠 **One-Shot Facial Recognition (Vector Search)**
* 🖥️ **Asynchronous GUI for real-time interaction**

> Built for **edge devices** and optimized for **GPU acceleration (GTX 1650, 4GB VRAM)**.

---

## 🧩 System Architecture

```
Camera Input → Frame Processing Pipeline
               ↓
     ┌───────────────────────┐
     │   Localization Engine │ → Bounding Boxes (Known Objects)
     └───────────────────────┘
               ↓
     ┌───────────────────────┐
     │   Open-World Engine   │ → Detects unseen objects via text prompts
     └───────────────────────┘
               ↓
     ┌───────────────────────┐
     │   Identity Engine     │ → Face embeddings + matching
     └───────────────────────┘
               ↓
         UI Rendering Layer
```

---

## ⚙️ Tech Stack

### Core AI

* YOLO (Ultralytics)
* YOLO-World (Open Vocabulary Detection)
* Siamese Network (Face Embeddings)

### Computer Vision

* OpenCV
* NumPy

### Backend / System

* Python
* Async Processing

### UI Layer

* CustomTkinter
* Pillow

### Hardware

* NVIDIA GPU (CUDA-enabled)
* Optimized for low VRAM environments

---

## 🧠 Key Features

### 🔹 1. Real-Time Multi-Model Inference

Runs multiple AI models simultaneously without blocking UI.

### 🔹 2. Zero-Shot Object Detection

Detect objects **never seen during training** using text prompts.

### 🔹 3. One-Shot Face Recognition

Register a person once → system recognizes instantly.

### 🔹 4. Vector Database (In-Memory)

* No retraining required
* Fast similarity search using embeddings

### 🔹 5. Edge Deployment Ready

Designed for:

* Low VRAM GPUs
* Local inference (no cloud dependency)

---

## 📂 Project Structure

```
IMAGE_NET/
│
├── dashboard.py        # Main GUI application
├── test_world.py       # Zero-shot detection module
├── train.py            # Training pipeline
├── resume.py           # Resume training
├── test_cam.py         # Camera testing script
│
├── models/             # Trained weights
├── data/               # Dataset
└── utils/              # Helper functions
```

---

## 🚀 Installation

### 1. System Setup (Linux)

```bash
sudo pacman -S cmake base-devel tk
```

For Ubuntu:

```bash
sudo apt install cmake build-essential python3-tk
```

---

### 2. Python Setup

```bash
python -m venv vision_env
source vision_env/bin/activate

pip install ultralytics face_recognition opencv-python numpy customtkinter Pillow
```

---

### 3. Run the Application

```bash
python dashboard.py
```

---

## 🧪 Modules

### 🎥 Vision Dashboard

* Real-time detection + face registration
* Live embedding storage

---

### 🌍 YOLO-World Module

```bash
python test_world.py
```

* Add custom text prompts
* Detect unseen objects

---

### 🏋️ Training Pipeline

```bash
python train.py
```

* Dataset loading
* Model training

```bash
python resume.py
```

* Resume interrupted training

---

## 📊 Model Training Details

* Dataset Size: ~7,200 images (~885MB)
* Epochs: 50
* Image Size: 640
* GPU: GTX 1650 (4GB)

### ⚠️ Challenges Solved

* Class overlap in dataset
* Imbalanced classes
* Limited labeled data

👉 Solution:

* Integrated **YOLO-World** for open-vocabulary robustness

---

## 🔬 Face Recognition Engine

Uses **vector similarity instead of classification**.

Distance metric:

[
d(p, q) = \sqrt{\sum_{i=1}^{128} (q_i - p_i)^2}
]

* Threshold ≤ 0.5 → Match
* Instant identity verification
* No retraining required

---

## 🎯 Why This Project Matters

This is not just a model — it's a **system design project**.

It demonstrates:

* Multi-model orchestration
* Real-time ML pipelines
* Edge AI optimization
* Vector search systems
* Practical deployment thinking

---

## 🧭 Future Improvements

* Persistent vector database (FAISS / SQLite)
* Model quantization (ONNX / TensorRT)
* Web-based dashboard (React + FastAPI)
* Multi-camera support
* Tracking (DeepSORT / ByteTrack)

---

## 👨‍💻 Author

**Lakshay Raj**

* AI/ML Enthusiast
* Focus: Full Stack + Machine Learning + Systems

---

## ⭐ Final Note

This project represents a shift from:

> "training models" → "building intelligent systems"

And that’s exactly what top AI engineers do.
