"""
predict.py

Runs YOLOv8 inference on all images in an input folder and saves
annotated output images to an output folder.

Usage (local):
    python src/predict.py

Usage (Colab or custom paths):
    Edit MODEL_PATH, INPUT_DIR, and OUTPUT_DIR at the top of this file,
    or import and call run_inference() directly with your own paths.

Output:
    - Annotated images saved to OUTPUT_DIR
    - Per-image detection summary printed to the terminal
"""

import os
from pathlib import Path
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
# Change these three paths to match your environment.
#
# Local (default after running src/train.py):
#   MODEL_PATH = "models/neu_det_v1/weights/best.pt"
#
# Google Colab (default Ultralytics output location):
#   MODEL_PATH = "/content/runs/detect/train/weights/best.pt"

MODEL_PATH = "models/neu_det_v1/weights/best.pt"
INPUT_DIR  = "data/NEU-DET-YOLO/images/val"   # folder of images to run inference on
OUTPUT_DIR = "outputs/predictions"             # where annotated images are saved

# Image extensions the script will process
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Minimum confidence score to display a detection (0.0–1.0)
CONFIDENCE_THRESHOLD = 0.5


# ── Core functions ─────────────────────────────────────────────────────────────

def load_model(model_path: str) -> YOLO:
    """
    Load a YOLOv8 model from the given weights file.

    Args:
        model_path: path to a .pt weights file (e.g. best.pt)

    Returns:
        Loaded YOLO model ready for inference
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}\n"
            "Make sure training has finished and the path is correct."
        )
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    print(f"Model loaded. Classes: {list(model.names.values())}\n")
    return model


def get_image_paths(input_dir: str) -> list[Path]:
    """
    Collect all image file paths from a directory (non-recursive).

    Args:
        input_dir: path to folder containing images

    Returns:
        Sorted list of Path objects for each image file found
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No images found in: {input_dir}")

    return images


def print_detections(image_name: str, result) -> None:
    """
    Print a human-readable summary of detections for one image.

    Args:
        image_name: filename of the image (used in the printed header)
        result:     single Ultralytics Results object from model inference
    """
    boxes = result.boxes
    names = result.names  # dict mapping class ID → class name

    if boxes is None or len(boxes) == 0:
        print(f"  [{image_name}] No detections above threshold.")
        return

    print(f"  [{image_name}] {len(boxes)} detection(s):")
    for box in boxes:
        class_id   = int(box.cls.item())
        class_name = names[class_id]
        confidence = float(box.conf.item())
        coords     = box.xyxy[0].tolist()  # [x1, y1, x2, y2] in pixels
        print(
            f"    • {class_name:<20} conf={confidence:.2f}  "
            f"box=[{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]"
        )


def run_inference(
    model_path: str,
    input_dir:  str,
    output_dir: str,
    conf:       float = CONFIDENCE_THRESHOLD,
) -> None:
    """
    Run inference on every image in input_dir and save annotated results.

    Args:
        model_path: path to the trained .pt weights file
        input_dir:  folder containing images to process
        output_dir: folder where annotated output images are saved
        conf:       minimum confidence threshold for detections
    """
    # ── Setup ──────────────────────────────────────────────────────────────────
    model = load_model(model_path)

    image_paths = get_image_paths(input_dir)
    print(f"Found {len(image_paths)} image(s) in: {input_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotated images to: {output_dir}\n")

    # ── Inference loop ─────────────────────────────────────────────────────────
    total_detections = 0

    for i, img_path in enumerate(image_paths, start=1):
        print(f"[{i}/{len(image_paths)}] Processing: {img_path.name}")

        # Run inference on a single image.
        # conf= filters out boxes below the confidence threshold.
        # verbose=False suppresses Ultralytics' own per-image logging.
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=0.5,    # IoU threshold for NMS: discard overlapping boxes above this overlap
            verbose=False,
        )

        result = results[0]  # one result per image

        # Print detections to terminal
        print_detections(img_path.name, result)

        # Save the annotated image (bounding boxes drawn by Ultralytics)
        output_path = output_dir / img_path.name
        result.save(filename=str(output_path))

        # Accumulate total detection count
        if result.boxes is not None:
            total_detections += len(result.boxes)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Inference complete.")
    print(f"  Images processed : {len(image_paths)}")
    print(f"  Total detections : {total_detections}")
    print(f"  Output folder    : {output_dir.resolve()}")
    print("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_inference(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )
