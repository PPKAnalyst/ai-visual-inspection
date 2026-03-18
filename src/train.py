"""
train.py

Fine-tunes a YOLOv8 nano model on the NEU-DET surface defect dataset.

Usage:
    python src/train.py

Outputs are saved to:
    models/neu_det_v1/
        weights/best.pt       ← best checkpoint (use this for prediction)
        weights/last.pt       ← final checkpoint
        results.csv           ← per-epoch metrics
        confusion_matrix.png
        PR_curve.png
        ... (other plots)
"""

import os
import torch
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────

# Path to the YOLO dataset config created by convert_labels.py
DATA_YAML = os.path.join("data", "NEU-DET-YOLO", "neu_det.yaml")

# Starting weights: yolov8n.pt is the nano (smallest) model pre-trained on COCO.
# Ultralytics downloads it automatically if not present; we already have it.
MODEL_WEIGHTS = "yolov8n.pt"

# All training outputs go into models/neu_det_v1/
# Use absolute path so Ultralytics doesn't silently prepend runs/detect/ to it.
OUTPUT_PROJECT = os.path.abspath("models")
OUTPUT_NAME    = "neu_det_v1"


# ── Device selection ───────────────────────────────────────────────────────────

def get_device():
    """
    Pick the best available compute device:
      - 'mps'  on Apple Silicon Macs (fast GPU-like acceleration)
      - 'cuda' on machines with an NVIDIA GPU
      - 'cpu'  as the fallback
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Training ───────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print("=" * 60)
    print("YOLOv8 Training — NEU-DET Surface Defect Detection")
    print("=" * 60)
    print(f"  Device        : {device}")
    print(f"  Starting model: {MODEL_WEIGHTS}")
    print(f"  Dataset config: {DATA_YAML}")
    print(f"  Output folder : {OUTPUT_PROJECT}/{OUTPUT_NAME}/")
    print()

    # Load the pre-trained YOLOv8 nano model.
    # The weights file already exists in the project root from our earlier test.
    print("Loading model weights...")
    model = YOLO(MODEL_WEIGHTS)

    # ── Start training ─────────────────────────────────────────────────────────
    print("Starting training...\n")

    results = model.train(

        # ----- Data -----
        data=DATA_YAML,         # path to neu_det.yaml
        imgsz=640,              # resize all images to 640×640 before feeding the model

        # ----- Training duration -----
        epochs=50,              # number of full passes through the training set
        patience=10,            # stop early if validation mAP doesn't improve for
                                # this many consecutive epochs (saves time)

        # ----- Batch & workers -----
        batch=8,                # images processed together in one forward pass;
                                # reduced from 16 — MPS on M1 hits memory pressure at 16
        workers=0,              # data-loading threads; 0 avoids multiprocessing
                                # issues on macOS (safe default for MPS)

        # ----- Optimizer -----
        optimizer="SGD",        # stochastic gradient descent; reliable for YOLO
        lr0=0.01,               # initial learning rate
        lrf=0.01,               # final learning rate = lr0 × lrf (cosine schedule)
        momentum=0.937,         # SGD momentum; helps training stay on course
        weight_decay=0.0005,    # L2 regularization; penalizes very large weights
                                # to reduce overfitting

        # ----- Hardware -----
        device=device,          # 'mps' on Apple Silicon, 'cpu' as fallback

        # ----- Output -----
        project=OUTPUT_PROJECT, # parent folder for results
        name=OUTPUT_NAME,       # subfolder name; full path: models/neu_det_v1/
        exist_ok=True,          # overwrite previous run with the same name
                                # (useful when re-running after adjustments)

        # ----- Misc -----
        pretrained=True,        # start from COCO-pretrained weights (transfer learning)
        verbose=True,           # print per-epoch metrics to the terminal
    )

    # ── Done ───────────────────────────────────────────────────────────────────
    best_model_path = os.path.join(OUTPUT_PROJECT, OUTPUT_NAME, "weights", "best.pt")
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Best model saved to : {best_model_path}")
    print(f"  All results in      : {OUTPUT_PROJECT}/{OUTPUT_NAME}/")
    print("\nNext step: use best.pt in src/predict.py to run inference.")
    print("=" * 60)


if __name__ == "__main__":
    main()
