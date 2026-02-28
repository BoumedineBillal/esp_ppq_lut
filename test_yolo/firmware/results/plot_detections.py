"""
plot_detections.py — Firmware Detection Visualizer
====================================================
Paste your firmware serial log into RAW_LOG_DATA below.
The script will:
  1. Parse '=== Testing: <image.jpg> ===' headers + detection lines
  2. Load the corresponding image from ../main/test_data/
  3. Apply the same nearest-neighbor letterbox preprocess as the ESP-DL firmware
     (so bounding box coordinates are correct — they live in 512x512 space)
  4. Plot detections with colored bounding boxes + bold labels
"""

import re
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================
# CONFIGURATION
# =============================================

MODEL_W = 512
MODEL_H = 512

# Images are stored here (same folder as the raw_rgb_person.h)
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "main", "test_data")

COLORS = ['#FF3333', '#3399FF', '#33CC66', '#FF9900', '#CC33FF', '#00CCCC']

# =============================================
# PASTE FIRMWARE LOG HERE
# =============================================

RAW_LOG_DATA = """
I (6113) YOLO26: === Testing: person.jpg ===
I (6113) YOLO26: [category: person, score: 0.91, x1: 328, y1: 173, x2: 404, y2: 379]
I (6113) YOLO26: [category: bicycle, score: 0.83, x1: 189, y1: 307, x2: 382, y2: 409]
I (6123) YOLO26: [category: bicycle, score: 0.41, x1: 121, y1: 133, x2: 194, y2: 182]
I (6133) YOLO26: [category: person, score: 0.34, x1: 147, y1: 196, x2: 175, y2: 221]
"""

# =============================================
# ESP-DL BIT-EXACT NEAREST-NEIGHBOR LETTERBOX
# =============================================

def espdl_preprocess(img_bgr: np.ndarray, target_w: int = MODEL_W, target_h: int = MODEL_H) -> np.ndarray:
    """
    Replicates the ESP-DL ImagePreprocessor letterbox exactly:
      - Nearest-neighbor resize preserving aspect ratio
      - Zero (128 gray) padding to fill the target canvas
    Returns a uint8 RGB image of shape (target_h, target_w, 3).
    """
    src_h, src_w = img_bgr.shape[:2]

    # Compute scale (same formula as ESP-DL)
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)   # floor, same as C int cast
    new_h = int(src_h * scale)

    # Nearest-neighbor resize (CV_INTER_NEAREST matches firmware)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Center padding
    pad_top    = (target_h - new_h) // 2
    pad_left   = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right  = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)   # ESP-DL uses 114 gray pad
    )

    # BGR → RGB for matplotlib display
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


# =============================================
# LOG PARSER
# =============================================

def parse_log(log_text: str) -> dict:
    """
    Returns { "image.jpg": [ {class, score, x1, y1, x2, y2}, ... ], ... }
    Parses lines produced by the generated main.cpp:
      I (...) YOLO26: === Testing: <img> ===
      I (...) YOLO26: [category: X, score: Y.YY, x1: A, y1: B, x2: C, y2: D]
    """
    results = {}
    current_image = None

    img_pat = re.compile(r"=== Testing:\s+(.+?)\s+===")
    det_pat = re.compile(
        r"\[category:\s*(.+?),\s*score:\s*([\d.]+),\s*"
        r"x1:\s*(-?\d+),\s*y1:\s*(-?\d+),\s*x2:\s*(-?\d+),\s*y2:\s*(-?\d+)\]"
    )

    for line in log_text.strip().splitlines():
        m = img_pat.search(line)
        if m:
            current_image = m.group(1).strip()
            results[current_image] = []
            continue

        if current_image:
            m = det_pat.search(line)
            if m:
                results[current_image].append({
                    "class":  m.group(1).strip(),
                    "score":  float(m.group(2)),
                    "x1":     int(m.group(3)),
                    "y1":     int(m.group(4)),
                    "x2":     int(m.group(5)),
                    "y2":     int(m.group(6)),
                })

    return results


# =============================================
# VISUALIZER
# =============================================

def visualize(results: dict):
    for img_name, detections in results.items():
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"[ERROR] Image not found: {img_path}")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[ERROR] cv2 could not read: {img_path}")
            continue

        print(f"\n{img_name} — {len(detections)} detections")

        # Apply the same preprocessing as the firmware
        canvas_rgb = espdl_preprocess(img_bgr, MODEL_W, MODEL_H)

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(canvas_rgb)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            score = det["score"]
            cls   = det["class"]

            # Clamp to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(MODEL_W, x2); y2 = min(MODEL_H, y2)

            color = COLORS[i % len(COLORS)]
            print(f"  [{i+1}] {cls} {score:.2f}  box=[{x1},{y1},{x2},{y2}]")

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, max(y1 - 6, 0), f"{cls}  {score:.2f}",
                color="white", fontsize=11, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=2)
            )

        ax.axis("off")
        ax.set_title(f"YOLO26n — {img_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save before show
        stem = os.path.splitext(img_name)[0]
        save_path = os.path.join(os.path.dirname(__file__), f"{stem}_firmware.jpg")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

        plt.show()


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    parsed = parse_log(RAW_LOG_DATA)
    if not parsed:
        print("No detections found — check log format and '=== Testing: <img> ===' header.")
    else:
        visualize(parsed)
