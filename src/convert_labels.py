"""
convert_labels.py

Converts the NEU-DET dataset from Pascal VOC XML format to YOLO format.

What it does:
  1. Reads every XML annotation from data/NEU-DET/{train,validation}/annotations/
  2. Converts each bounding box to YOLO format (normalized center x, center y, width, height)
  3. Writes one .txt label file per image to data/NEU-DET-YOLO/labels/{train,val}/
  4. Copies the matching image to data/NEU-DET-YOLO/images/{train,val}/
  5. Prints a summary at the end

YOLO label format (one line per object):
  <class_id> <x_center> <y_center> <width> <height>
  All values are normalized between 0 and 1 relative to the image size.

Known dataset quirks handled here:
  - Some <filename> tags are missing the .jpg extension (e.g. pitted_surface_1
    instead of pitted_surface_1.jpg). We derive the image name from the XML
    filename instead of trusting the <filename> tag.
  - One annotation (crazing_240.xml) is filed under validation/annotations but
    its image lives in train/images/. A global image index built at startup
    finds it regardless of which split folder it came from.
"""

import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────

# Root of the original (untouched) NEU-DET dataset
SRC_ROOT = os.path.join("data", "NEU-DET")

# Root of the new YOLO-format dataset we will create
DST_ROOT = os.path.join("data", "NEU-DET-YOLO")

# Map split names: original folder name → YOLO folder name
SPLITS = {
    "train":      "train",
    "validation": "val",
}

# ── Class mapping ──────────────────────────────────────────────────────────────
# Must match the order in neu_det.yaml exactly.
CLASS_NAMES = [
    "crazing",          # 0
    "inclusion",        # 1
    "patches",          # 2
    "pitted_surface",   # 3
    "rolled-in_scale",  # 4
    "scratches",        # 5
]

# Build a reverse lookup: class name → integer ID
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ── Global image index ─────────────────────────────────────────────────────────

def build_image_index():
    """
    Walk every class subfolder in every split of the source dataset and build
    a dict mapping image stem → absolute file path.

      e.g. 'crazing_1'        → '.../train/images/crazing/crazing_1.jpg'
           'pitted_surface_12' → '.../train/images/pitted_surface/pitted_surface_12.jpg'

    Building the index once at startup means:
      - We never depend on the <filename> tag (which is missing .jpg in some files).
      - We find images even if their XML was placed in the wrong split folder
        (e.g. crazing_240.xml is in validation/annotations but the image is in
        train/images/crazing/).
    """
    index = {}  # stem → full path
    for split in SPLITS:
        img_root = os.path.join(SRC_ROOT, split, "images")
        for cls_folder in os.listdir(img_root):
            cls_path = os.path.join(img_root, cls_folder)
            if not os.path.isdir(cls_path):
                continue
            for img_file in os.listdir(cls_path):
                stem = os.path.splitext(img_file)[0]  # strip .jpg
                index[stem] = os.path.join(cls_path, img_file)
    return index


# ── Conversion helper ──────────────────────────────────────────────────────────

def convert_box(size, box):
    """
    Convert a Pascal VOC bounding box to YOLO format.

    Pascal VOC box: (xmin, ymin, xmax, ymax) in pixels
    YOLO format:    (x_center, y_center, width, height) normalized 0–1

    Args:
        size: (image_width, image_height) in pixels
        box:  (xmin, ymin, xmax, ymax) in pixels

    Returns:
        (x_center, y_center, width, height) as floats normalized by image size
    """
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width    = (xmax - xmin) / img_w
    height   = (ymax - ymin) / img_h

    return x_center, y_center, width, height


def parse_xml(xml_path):
    """
    Parse a Pascal VOC XML file and return image info + all bounding boxes.

    Args:
        xml_path: path to the .xml annotation file

    Returns:
        filename:  the image filename recorded in the XML (e.g. 'crazing_1.jpg')
        img_size:  (width, height) of the image in pixels
        objects:   list of (class_name, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Image filename as stored in the XML
    filename = root.find("filename").text

    # Image dimensions
    size_node = root.find("size")
    img_w = int(size_node.find("width").text)
    img_h = int(size_node.find("height").text)

    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        bndbox     = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        objects.append((class_name, xmin, ymin, xmax, ymax))

    return filename, (img_w, img_h), objects


# ── Main conversion loop ───────────────────────────────────────────────────────

def convert_split(src_split, dst_split, image_index):
    """
    Process one dataset split (e.g. train or validation → val).

    Args:
        src_split:   folder name in the original dataset ('train' or 'validation')
        dst_split:   folder name in the YOLO dataset ('train' or 'val')
        image_index: dict of {stem → full image path} built by build_image_index()

    Returns:
        stats dict with counts for this split
    """
    ann_dir = os.path.join(SRC_ROOT, src_split, "annotations")

    dst_img_dir = os.path.join(DST_ROOT, "images", dst_split)
    dst_lbl_dir = os.path.join(DST_ROOT, "labels", dst_split)

    # Counters for the summary
    stats = {
        "images_copied":  0,
        "labels_written": 0,
        "missing_images": [],
        "unknown_classes": [],
        "class_counts": defaultdict(int),
    }

    xml_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".xml"))
    print(f"\n[{src_split}] Found {len(xml_files)} XML files in {ann_dir}")

    for xml_file in xml_files:
        xml_path = os.path.join(ann_dir, xml_file)

        # ── Parse XML ──────────────────────────────────────────────────────────
        try:
            _xml_filename, img_size, objects = parse_xml(xml_path)
        except Exception as e:
            print(f"  WARNING: Could not parse {xml_file}: {e}")
            continue

        # ── Derive image stem from the XML filename (not the <filename> tag) ───
        # Fix for Bug 1: some <filename> tags are missing the .jpg extension
        # (e.g. pitted_surface_1 instead of pitted_surface_1.jpg).
        # The XML file is always named correctly, so we use that as the source
        # of truth.  e.g. 'patches_22.xml' → stem 'patches_22'
        stem = os.path.splitext(xml_file)[0]

        # ── Look up the image in the global index ──────────────────────────────
        # Fix for Bug 2: the index spans all splits, so crazing_240.xml (filed
        # under validation/annotations) can still find its image in
        # train/images/crazing/.
        src_img_path = image_index.get(stem)

        if src_img_path is None:
            print(f"  WARNING: Image not found for {xml_file} (stem='{stem}')")
            stats["missing_images"].append(xml_file)
            continue

        img_filename = os.path.basename(src_img_path)  # e.g. 'patches_22.jpg'

        # ── Build YOLO label lines ─────────────────────────────────────────────
        label_lines = []
        for (cls_name, xmin, ymin, xmax, ymax) in objects:
            if cls_name not in CLASS_TO_ID:
                print(f"  WARNING: Unknown class '{cls_name}' in {xml_file}, skipping object.")
                stats["unknown_classes"].append((xml_file, cls_name))
                continue

            class_id = CLASS_TO_ID[cls_name]
            x_c, y_c, w, h = convert_box(img_size, (xmin, ymin, xmax, ymax))
            label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            stats["class_counts"][cls_name] += 1

        if not label_lines:
            # No valid objects found — skip this file entirely
            print(f"  WARNING: No valid objects in {xml_file}, skipping.")
            continue

        # ── Write label .txt ───────────────────────────────────────────────────
        label_filename = stem + ".txt"
        dst_lbl_path   = os.path.join(dst_lbl_dir, label_filename)
        with open(dst_lbl_path, "w") as f:
            f.write("\n".join(label_lines) + "\n")
        stats["labels_written"] += 1

        # ── Copy image ─────────────────────────────────────────────────────────
        dst_img_path = os.path.join(dst_img_dir, img_filename)
        shutil.copy2(src_img_path, dst_img_path)
        stats["images_copied"] += 1

    return stats


def main():
    print("=" * 60)
    print("NEU-DET → YOLO Format Conversion")
    print("=" * 60)

    # Build the global image index once before processing any split.
    # This makes image lookup independent of which split an XML lives in.
    print("\nBuilding global image index...")
    image_index = build_image_index()
    print(f"  Found {len(image_index)} images across all splits.")

    total_images  = 0
    total_labels  = 0
    total_missing = []
    total_classes = defaultdict(int)

    # Process each split
    for src_split, dst_split in SPLITS.items():
        stats = convert_split(src_split, dst_split, image_index)

        total_images  += stats["images_copied"]
        total_labels  += stats["labels_written"]
        total_missing += stats["missing_images"]
        for cls, count in stats["class_counts"].items():
            total_classes[cls] += count

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE — Summary")
    print("=" * 60)
    print(f"  Total images copied : {total_images}")
    print(f"  Total labels written: {total_labels}")
    print(f"  Missing image files : {len(total_missing)}")
    if total_missing:
        for m in total_missing:
            print(f"    - {m}")

    print("\n  Bounding box counts per class:")
    for cls in CLASS_NAMES:
        print(f"    {cls:<20} {total_classes.get(cls, 0)}")

    print("\nOutput dataset location:")
    print(f"  {os.path.abspath(DST_ROOT)}")


if __name__ == "__main__":
    main()
