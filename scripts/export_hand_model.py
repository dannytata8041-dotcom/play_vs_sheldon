#!/usr/bin/env python
"""Download YOLO11n-pose-hands weights and export to ONNX for Metis.

Usage:
    python scripts/export_hand_model.py

This will:
1. Download best.pt from chrismuntean/YOLO11n-pose-hands
2. Export to ONNX format (weights/yolo11n-pose-hands.onnx)
3. Verify the export by running a dummy inference

Requirements:
    pip install ultralytics
"""

import os
import sys
import urllib.request

REPO_URL = (
    "https://github.com/chrismuntean/YOLO11n-pose-hands/raw/"
    "bda894403f378d2a298d2f88ae9d5ed6d4e9f8e3/runs/pose/train/weights/best.pt"
)
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
PT_PATH = os.path.join(WEIGHTS_DIR, "yolo11n-pose-hands.pt")
ONNX_PATH = os.path.join(WEIGHTS_DIR, "yolo11n-pose-hands.onnx")


def download_weights():
    """Download .pt weights from GitHub if not already present."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if os.path.exists(PT_PATH):
        print(f"Weights already exist: {PT_PATH}")
        return

    print(f"Downloading YOLO11n-pose-hands weights...")
    print(f"  From: {REPO_URL}")
    print(f"  To:   {PT_PATH}")

    urllib.request.urlretrieve(REPO_URL, PT_PATH)
    size_mb = os.path.getsize(PT_PATH) / (1024 * 1024)
    print(f"  Done ({size_mb:.1f} MB)")


def export_onnx():
    """Export .pt to ONNX for Axelera Metis AIPU."""
    if os.path.exists(ONNX_PATH):
        print(f"ONNX already exists: {ONNX_PATH}")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"Loading model: {PT_PATH}")
    model = YOLO(PT_PATH)

    print(f"Exporting to ONNX...")
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=13,
        dynamic=False,
    )

    # ultralytics exports next to the .pt file
    exported = PT_PATH.replace(".pt", ".onnx")
    if exported != ONNX_PATH and os.path.exists(exported):
        os.rename(exported, ONNX_PATH)

    print(f"Exported: {ONNX_PATH}")
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")


def verify():
    """Quick sanity check on the ONNX export."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("Skipping verification (onnxruntime not installed)")
        return

    print("Verifying ONNX model...")
    sess = ort.InferenceSession(ONNX_PATH)
    inp = sess.get_inputs()[0]
    print(f"  Input: {inp.name}, shape={inp.shape}, dtype={inp.type}")

    for out in sess.get_outputs():
        print(f"  Output: {out.name}, shape={out.shape}")

    dummy = np.random.randn(*inp.shape).astype(np.float32)
    results = sess.run(None, {inp.name: dummy})
    print(f"  Inference OK — output shape: {results[0].shape}")


if __name__ == "__main__":
    download_weights()
    export_onnx()
    verify()
    print("\nDone! The model is ready for Metis at:")
    print(f"  {ONNX_PATH}")
