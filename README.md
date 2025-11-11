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
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
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

#### Option A: Using pip (Recommended)

```bash
pip install opencv-python numpy easyocr torch ultralytics
```

#### Option B: Using requirements.txt

```bash
pip install -r requirements.txt
```

#### Option C: For Google Colab

If you're using Google Colab, run this in a code cell:

```python
!pip install opencv-python numpy easyocr ultralytics
```

Then mount your Google Drive to access video files:

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
   jupyter notebook video-extraction-tool.ipynb
   ```

2. **Run the cells in order**:
   - Cell 1: Install dependencies
   - Cell 2: Import libraries
   - Cells 3-10: Function definitions (run all)
   - Cell 11: Modify the video path and run

3. **Update the video path** in the last cell:
   ```python
   video_path = "path/to/your/video.mp4"  # Change this to your video file
   ```

4. **Run the analysis**:
   - The tool will process your video and display progress
   - Results will be printed and saved as JSON

### Method 2: Using Python Script

1. **Create a Python script** (`analyze_video.py`):
   ```python
   import json
   from video_extraction_tool import extract_video_features
   
   # Specify your video path
   video_path = "sample_video.mp4"
   
   # Extract features
   features = extract_video_features(video_path)
   
   # Display results
   print(json.dumps(features, indent=2))
   
   # Save to file
   with open("output_features.json", "w") as f:
       json.dump(features, f, indent=2)
   ```

2. **Run the script**:
   ```bash
   python analyze_video.py
   ```

### Method 4: Google Colab

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

**Use Cases**: 
- Detecting action vs dialogue scenes
- Identifying camera movement
- Video content classification

**Example**:
```python
avg_motion = analyze_motion("video.mp4")
print(f"Average Motion: {avg_motion:.2f}")
# Output: Average Motion: 3.45
```

---

### Feature 2: Text Detection (`analyze_text_ocr_easyocr`)

**Purpose**: Detect and extract text appearing in video frames.

**How it works**:
- Samples frames at regular intervals (configurable)
- Uses EasyOCR to detect text in each sampled frame
- Filters results by confidence threshold
- Extracts unique keywords (alphabetic, length ≥ 3)
- Calculates ratio of frames containing text

**Use Cases**:
- Subtitle extraction
- On-screen text indexing
- Content categorization
- Accessibility features

**Example**:
```python
text_info = analyze_text_ocr_easyocr("video.mp4")
print(f"Text Present Ratio: {text_info['text_present_ratio']:.2%}")
print(f"Keywords: {text_info['keywords'][:10]}")  # First 10 keywords
```

---

### Feature 3: Hard Cut Detection (`detect_hard_cuts`)

**Purpose**: Identify scene transitions and shot changes.

**How it works**:
- Computes absolute pixel difference between consecutive frames
- Detects cuts when difference exceeds threshold
- Implements minimum scene length to avoid false positives
- Records frame indices where cuts occur

**Use Cases**:
- Automatic video segmentation
- Scene boundary detection
- Video editing assistance
- Content structure analysis

**Example**:
```python
cuts = detect_hard_cuts("video.mp4")
print(f"Total Cuts: {cuts['hard_cuts']}")
print(f"Cut at frames: {cuts['cut_frames']}")
# Output: Total Cuts: 5
# Output: Cut at frames: [120, 245, 387, 521, 698]
```

---

### Feature 4: Person vs Object Dominance (`analyze_person_object_dominance`)

**Purpose**: Analyze the presence and ratio of people versus objects in video.

**How it works**:
- Uses YOLOv8 pre-trained on COCO dataset
- Samples frames at regular intervals
- Detects and classifies objects (80 classes)
- Separates "person" class from other objects
- Calculates various dominance metrics

**Use Cases**:
- Content type classification (person-focused vs object-focused)
- Audience targeting
- Video categorization
- Analytics for marketing

**Example**:
```python
dominance = analyze_person_object_dominance("video.mp4")
print(f"Person Detections: {dominance['person_detections']}")
print(f"Object Detections: {dominance['object_detections']}")
print(f"Person Ratio: {dominance['person_ratio']:.2%}")
# Output: Person Detections: 142
# Output: Object Detections: 89
# Output: Person Ratio: 61.47%
```

---

### Combined Analysis (`extract_video_features`)

**Purpose**: Run all feature extraction methods in a single function call.

**How it works**:
- Orchestrates all four analysis functions
- Provides progress updates
- Combines results into unified output
- Optimizes processing with configurable frame sampling

**Example**:
```python
# Analyze with default settings
features = extract_video_features("video.mp4")

# Analyze with custom settings
features = extract_video_features(
    video_path="video.mp4",
    ocr_frame_step=15,        # Sample every 15th frame for OCR
    det_frame_step=10,        # Sample every 10th frame for YOLO
    yolo_model="yolov8x.pt"   #
)
```

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

## ⚙️ Configuration Options

### Motion Analysis
- No configuration needed - uses optimized Farneback parameters

### Text Detection (OCR)
- `ocr_frame_step`: Process every Nth frame (default: 10)
- `languages`: List of language codes (default: ['en'])
- `min_conf`: Minimum confidence threshold (default: 0.5)

### Shot Cut Detection
- `diff_threshold`: Pixel difference threshold for cut detection (default: 30.0)
- `min_scene_length`: Minimum frames between cuts (default: 5)

### Person/Object Detection
- `model_path`: YOLO model to use (default: "yolov8x.pt")
  - Options: "yolov8n.pt" (nano), "yolov8s.pt" (small), "yolov8m.pt" (medium), "yolov8l.pt" (large), "yolov8x.pt" (extra large)
- `det_frame_step`: Process every Nth frame (default: 10)
- `conf_thres`: Minimum confidence for detections (default: 0.5)

## 💡 Examples

### Example 1: Quick Analysis

```python
from video_extraction_tool import extract_video_features
import json

# Analyze video with default settings
features = extract_video_features("sample_video.mp4")

# Display results
print(json.dumps(features, indent=2))
```

### Example 2: Custom Configuration

```python
# High-accuracy mode (slower)
features = extract_video_features(
    video_path="important_video.mp4",
    ocr_frame_step=5,         # Check every 5th frame
    det_frame_step=5,         # Check every 5th frame
    yolo_model="yolov8x.pt"   # Largest, most accurate model
)
```

### Example 3: Fast Processing Mode

```python
# Speed-optimized (faster, less accurate)
features = extract_video_features(
    video_path="long_video.mp4",
    ocr_frame_step=30,        # Check every 30th frame
    det_frame_step=30,        # Check every 30th frame
    yolo_model="yolov8x.pt"  
)
```

### Example 4: Batch Processing

```python
import os
import json

video_folder = "videos/"
output_folder = "results/"

for video_file in os.listdir(video_folder):
    if video_file.endswith((".mp4", ".avi", ".mov")):
        video_path = os.path.join(video_folder, video_file)
        print(f"\nProcessing: {video_file}")
        
        features = extract_video_features(video_path)
        
        # Save results
        output_file = os.path.join(output_folder, f"{video_file}_features.json")
        with open(output_file, "w") as f:
            json.dump(features, f, indent=2)
        
        print(f"Saved to: {output_file}")
```

### Example 5: Using Individual Functions

```python
# Only detect cuts
from video_extraction_tool import detect_hard_cuts
cuts = detect_hard_cuts("video.mp4", diff_threshold=35.0)
print(f"Found {cuts['hard_cuts']} scene transitions")

# Only analyze motion
from video_extraction_tool import analyze_motion
motion = analyze_motion("video.mp4")
if motion > 5.0:
    print("High-action video")
elif motion > 2.0:
    print("Moderate action")
else:
    print("Low-action/static video")
```
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

## � Contact
- **Email**: royanish.career@gmail.com

---
