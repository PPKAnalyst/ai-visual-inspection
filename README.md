# AI Visual Inspection for Manufacturing Defect Detection

This project explores how computer vision can be applied to real-world manufacturing quality inspection problems using YOLOv8.


## Problem

Manual inspection on manufacturing shop floors is:
- time-consuming  
- inconsistent  
- difficult to scale  

Detecting surface defects like cracks, inclusions, or patches requires trained human inspectors and can lead to variability in quality.


## Motivation

During my manufacturing co-op, I designed a gauge to verify correct Belleville washer assembly orientation.

This made me think:

👉 Can computer vision automate such inspection tasks?

This project is my first step toward building AI-driven inspection systems for manufacturing.


## What this project does
- Converts XML annotations to YOLO format  
- Trains a YOLOv8 model on NEU-DET dataset  
- Runs inference on new images  
- Saves annotated predictions  


## Tech Stack
- Python  
- YOLOv8 (Ultralytics)  
- PyTorch  
- OpenCV  
- Google Colab (GPU training)


## ⚙️ How It Works

1. Convert dataset annotations (XML → YOLO format)
2. Train YOLOv8 model on NEU-DET dataset
3. Run inference on validation images
4. Generate bounding boxes with confidence scores
5. Save annotated outputs


## Pipeline

Image → Preprocessing → YOLOv8 Training → Inference → Defect Detection Output

## Results

- Precision: 0.706  
- Recall: 0.692  
- mAP@50: 0.748  
- mAP@50-95: 0.396  

Model successfully detected multiple defect types including:
- crazing  
- inclusion  
- patches  
- pitted surface  


## Sample Outputs

### Crazing Detection
![Crazing](assets/crazing_prediction.jpg)

### Inclusion Detection
![Inclusion](assets/inclusion_multi_prediction.jpg)

### Patches Detection
![Patches](assets/patches_prediction.jpg)

---
