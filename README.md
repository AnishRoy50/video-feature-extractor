# Video Feature Extraction Tool 🎥

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive Python-based tool for extracting and analyzing various features from video files using computer vision and deep learning techniques. This tool automatically analyzes videos to detect motion, text, scene cuts, and object/person presence.

## 📑 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Implemented Features](#-implemented-features)
- [Output Format](#-output-format)
- [License](#-license)

## ✨ Features

This tool provides four main analysis modules that work together to extract comprehensive video features:

### 1. **Motion Analysis** 🏃
- Quantifies average motion intensity using Optical Flow (Farneback method)
- Dense optical flow computation between consecutive frames
- Returns normalized motion magnitude across the entire video
- Useful for detecting action scenes vs static shots

### 2. **Text Detection (OCR)** 📝
- Detects and extracts text from video frames using EasyOCR
- Calculates text presence ratio (percentage of frames containing text)
- Extracts and lists unique keywords found throughout the video
- **GPU acceleration** support for faster processing
- **Multi-language support** (default: English)
- Configurable confidence thresholds

### 3. **Shot Cut Detection** ✂️
- Identifies hard cuts (abrupt scene transitions) in videos
- Frame-to-frame pixel difference analysis
- Returns total count of cuts and their specific frame indices
- Configurable threshold to avoid false positives
- Minimum scene length parameter to filter out flickers

### 4. **Person vs Object Dominance** 👤
- Analyzes the presence of people versus objects using **YOLOv8**
- Provides detailed detection counts and statistical ratios
- Calculates person-to-object dominance metrics
- Uses COCO-trained YOLO models for 80+ object classes
- Frame sampling for efficient processing

## 📋 Prerequisites
- **Python 3.8 or higher** installed
- **pip** package manager

### Required Libraries

- `opencv-python` - Computer vision and video processing
- `numpy` - Numerical computations
- `easyocr` - Optical character recognition
- `torch` - PyTorch deep learning framework
- `ultralytics` - YOLOv8 object detection

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/video-feature-extractor.git
cd video-feature-extractor
```

### Step 2: Install Dependencies

#### Option A: Using pip

```bash
pip install opencv-python numpy easyocr torch ultralytics
```
#### Option B: For Google Colab
```python
!pip install opencv-python numpy easyocr ultralytics
```

Then mount Google Drive to access video files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Verify Installation

Run this to check if all packages are installed correctly:

```python
import cv2
import numpy as np
import easyocr
import torch
from ultralytics import YOLO
print("All dependencies installed successfully!")
print(f"GPU Available: {torch.cuda.is_available()}")
```

## 🎬 How to Run

### Method 1: Using Jupyter Notebook

1. **Open the notebook**:
   ```bash
   jupyter notebook video-feature-extraction-tool.ipynb
   ```

2. **Run the cells in order**:
   - Cell 1: Install dependencies
   - Cell 2: Import libraries
   - Cells 3-10: Function definitions (run all)
   - Cell 11: Modify the video path and run

3. **Update the video path** in the last cell:
   ```python
   video_path = "path/to/your/video.mp4"  # Change this to the video file
   ```

4. **Run the analysis**:
   - The tool will process the video and display progress
   - Results will be printed and saved as JSON

### Method 2: Google Colab

1. Upload the notebook to Google Colab
2. Upload your video to Google Drive
3. Mount Drive and update the path:
   ```python
   video_path = "/content/drive/MyDrive/videos/sample.mp4"
   ```
4. Run all cells

## 🔧 Implemented Features

### Feature 1: Motion Analysis (`analyze_motion`)

**Purpose**: Quantify the overall motion/activity level in a video.

**How it works**:
- Converts frames to grayscale
- Computes dense optical flow using Farneback algorithm
- Calculates motion magnitude between consecutive frames
- Returns average motion across entire video
---

### Feature 2: Text Detection (`analyze_text_ocr_easyocr`)

**Purpose**: Detect and extract text appearing in video frames.

**How it works**:
- Samples frames at regular intervals (configurable)
- Uses EasyOCR to detect text in each sampled frame
- Filters results by confidence threshold
- Extracts unique keywords (alphabetic, length ≥ 3)
- Calculates ratio of frames containing text
---

### Feature 3: Hard Cut Detection (`detect_hard_cuts`)

**Purpose**: Identify scene transitions and shot changes.

**How it works**:
- Computes absolute pixel difference between consecutive frames
- Detects cuts when difference exceeds threshold
- Implements minimum scene length to avoid false positives
- Records frame indices where cuts occur
---

### Feature 4: Person vs Object Dominance (`analyze_person_object_dominance`)

**Purpose**: Analyze the presence and ratio of people versus objects in video.

**How it works**:
- Uses YOLOv8 pre-trained on COCO dataset
- Samples frames at regular intervals
- Detects and classifies objects (80 classes)
- Separates "person" class from other objects
- Calculates various dominance metrics
---
## 📊 Output Format

The tool returns a dictionary with the following structure:

```json
{
  "file": "path/to/video.mp4",
  "hard_cuts": 5,
  "cut_frames": [120, 245, 387, 521, 698],
  "average_motion": 2.45,
  "text_present_ratio": 0.35,
  "keywords": ["hello", "world", "example", "text"],
  "person_detections": 142,
  "object_detections": 89,
  "person_ratio": 0.615,
  "object_ratio": 0.385,
  "person_to_object_ratio": 1.596
}
```
This tool also generates a .json file with the outputs in the specified location. 

### Output Fields Explained

| Field | Description |
|-------|-------------|
| `file` | Path to the analyzed video file |
| `hard_cuts` | Number of detected hard cuts/scene transitions |
| `cut_frames` | List of frame indices where cuts occur |
| `average_motion` | Average motion magnitude (0-255 scale) |
| `text_present_ratio` | Fraction of sampled frames containing text (0-1) |
| `keywords` | Unique words extracted from video text |
| `person_detections` | Total number of person detections |
| `object_detections` | Total number of non-person object detections |
| `person_ratio` | Ratio of person detections to total detections |
| `object_ratio` | Ratio of object detections to total detections |
| `person_to_object_ratio` | Person detections divided by object detections |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citations & Acknowledgments

This project uses the following open-source libraries:

- **OpenCV**: Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
- **YOLOv8**: Ultralytics (2023). YOLOv8: A new state-of-the-art computer vision model.
- **EasyOCR**: JaidedAI (2020). EasyOCR: Ready-to-use OCR with 80+ languages.

## 🔗 Useful Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [EasyOCR GitHub Repository](https://github.com/JaidedAI/EasyOCR)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [COCO Dataset](https://cocodataset.org/) - Object classes used by YOLO

## 📩 Contact
- **Email**: royanish.career@gmail.com

---
