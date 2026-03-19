#!/usr/bin/env python
"""POV Label Tracker — entry point.

Runs YOLOv8n on Axelera Metis AIPU for object detection,
then OCR on CPU (Tesseract) for text extraction from detected labels.

Usage:
    # Activate Voyager SDK first
    cd /home/orangepi/voyager-sdk && source venv/bin/activate

    # Run with RTSP camera (reads from config/camera.json by default)
    python main.py

    # Run with USB camera
    python main.py --source /dev/video0

    # Run with video file
    python main.py --source /path/to/video.mp4

    # Demo mode (no hardware)
    python main.py --demo
"""

import argparse
import json
import logging
import os
import sys
import time

# Add src dir to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from stability_gate import StabilityGate
from hand_trigger import HandTrigger
from parser import parse
from db import get_connection, init_db, upsert_product, log_scan
from alerts import evaluate, log_alerts
from ocr import extract_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROFILE_PATH = os.path.join(BASE_DIR, "config", "household_profile.json")
CAMERA_CONFIG = os.path.join(BASE_DIR, "config", "camera.json")
DB_PATH = os.path.join(BASE_DIR, "label_tracker.db")
PIPELINE_YAML = os.path.join(BASE_DIR, "config", "pipeline.yaml")
HAND_PIPELINE_YAML = os.path.join(BASE_DIR, "config", "hand-keypoints.yaml")


def load_camera_config():
    """Load camera source and settings from config/camera.json."""
    if os.path.exists(CAMERA_CONFIG):
        with open(CAMERA_CONFIG) as f:
            return json.load(f)
    return {}


def load_profile(path):
    """Load household allergen profile from JSON."""
    with open(path) as f:
        return json.load(f)


def process_ocr_result(conn, profile, raw_text, confidence, thumbnail_path=None):
    """Full processing pipeline for one OCR result."""
    parsed = parse(raw_text, confidence)
    logger.info(
        "Parsed: name=%s, expiry=%s, allergens=%s",
        parsed["name"], parsed["expiry_date"], parsed["allergens"],
    )

    product_id = upsert_product(conn, parsed, thumbnail_path)

    product = {
        "name": parsed["name"],
        "expiry_date": parsed["expiry_date"],
        "allergens": parsed["allergens"],
    }
    alerts_list = evaluate(product, profile)
    log_scan(conn, product_id, parsed, alerts_list)

    if alerts_list:
        log_alerts(alerts_list)

    return parsed, alerts_list


def run_metis_pipeline(args, conn, profile):
    """Run YOLOv8n on Metis with camera source, OCR on CPU."""
    try:
        from axelera.app import config, display
        from axelera.app.stream import create_inference_stream
    except ImportError:
        logger.error(
            "Voyager SDK not available. Activate it first:\n"
            "  cd /home/orangepi/voyager-sdk && source venv/bin/activate"
        )
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not available. Install with: pip install opencv-python")
        sys.exit(1)

    source = args.source
    logger.info("Creating inference stream: network=yolov8n-coco, source=%s", source)

    stream_kwargs = build_stream_kwargs("yolov8n-coco", source, args.rtsp_latency)
    stream = create_inference_stream(**stream_kwargs)
    gate = StabilityGate()

    logger.info("Pipeline started. Press Ctrl+C to stop.")
    logger.info("Detecting products and reading labels on Metis AIPU...")

    thumbnail_dir = os.path.join(BASE_DIR, "thumbnails")
    os.makedirs(thumbnail_dir, exist_ok=True)

    def main_loop(window, stream):
        try:
            for frame_result in stream:
                # Get frame as numpy array for display and cropping
                rgb_img = frame_result.image.asarray()
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                # Process each detection from YOLO on Metis
                for det in frame_result.detections:
                    box = list(det.box)  # [x1, y1, x2, y2]
                    detection = {"box": box, "label": det.label.name, "score": det.score}

                    # Draw detection box on display frame
                    if not args.headless:
                        bx1, by1, bx2, by2 = [int(v) for v in box]
                        cv2.rectangle(bgr_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        cv2.putText(
                            bgr_img,
                            f"{det.label.name} {det.score:.2f}",
                            (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                        )

                    # Stability gate: only trigger OCR when object is held steady
                    trigger_box = gate.update(detection)
                    if trigger_box is None:
                        continue

                    # Object is stable — crop and run OCR on CPU
                    x1, y1, x2, y2 = [int(v) for v in trigger_box]
                    h, w = rgb_img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue  # crop too small

                    crop = rgb_img[y1:y2, x1:x2]

                    # Save thumbnail
                    ts = int(time.time())
                    thumb_path = os.path.join(thumbnail_dir, f"crop_{ts}.jpg")
                    cv2.imwrite(thumb_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                    # Run OCR on CPU
                    raw_text, confidence = extract_text(crop)
                    if not raw_text.strip():
                        logger.debug("OCR returned empty text, skipping")
                        continue

                    logger.info("OCR text: %s", raw_text[:100])

                    # Process through parser → db → alerts
                    process_ocr_result(
                        conn, profile, raw_text, confidence, thumb_path
                    )

                # Show frame with cv2 (same pattern as simple.py)
                if not args.headless:
                    display_img = cv2.resize(bgr_img, (960, 540))
                    cv2.imshow("POV Label Tracker", display_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Manual snapshot: OCR the full frame
                        logger.info("Manual snapshot — running OCR on full frame...")
                        raw_text, confidence = extract_text(
                            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        )
                        if raw_text.strip():
                            logger.info("OCR text:\n%s", raw_text)
                            snap_path = os.path.join(
                                thumbnail_dir, f"snap_{int(time.time())}.jpg"
                            )
                            cv2.imwrite(snap_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                            process_ocr_result(
                                conn, profile, raw_text, confidence, snap_path
                            )
                        else:
                            logger.info("No text found in snapshot")

        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
        finally:
            if not args.headless:
                cv2.destroyAllWindows()

    with display.App(renderer=False) as app:
        app.start_thread(main_loop, (None, stream), name="LabelTracker")
        app.run()

    stream.stop()
    logger.info("Pipeline stopped.")


def run_hand_pipeline(args, conn, profile):
    """Run YOLO11n-pose-hands on Metis — gesture control + OCR trigger."""
    try:
        from axelera.app import config, display
        from axelera.app.stream import create_inference_stream
    except ImportError:
        logger.error(
            "Voyager SDK not available. Activate it first:\n"
            "  cd /home/orangepi/voyager-sdk && source venv/bin/activate"
        )
        sys.exit(1)

    import cv2
    import numpy as np

    # Register HandKeypointsMeta in AxTaskMeta._subclasses before stream starts
    from decode_handpose import HandKeypointsMeta  # noqa: F401

    source = args.source
    logger.info("Creating YOLO11n-pose-hands stream, source=%s", source)

    stream_kwargs = build_stream_kwargs(HAND_PIPELINE_YAML, source, args.rtsp_latency)
    stream = create_inference_stream(**stream_kwargs)
    trigger = HandTrigger()

    logger.info("Hand gesture pipeline started. Press Ctrl+C to stop.")
    logger.info("Gestures: HOLD=scan, THUMBS_UP=confirm, OPEN_PALM=pause, FIST=resume")

    thumbnail_dir = os.path.join(BASE_DIR, "thumbnails")
    os.makedirs(thumbnail_dir, exist_ok=True)

    # Hand skeleton connections for drawing
    SKELETON = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (0, 9), (9, 10), (10, 11), (11, 12),   # middle
        (0, 13), (13, 14), (14, 15), (15, 16), # ring
        (0, 17), (17, 18), (18, 19), (19, 20), # pinky
        (5, 9), (9, 13), (13, 17),             # palm
    ]

    def draw_hand(img, keypoints, gesture_text):
        """Draw hand skeleton and gesture label on the frame."""
        # Draw skeleton lines
        for a, b in SKELETON:
            if keypoints[a][2] > 0.3 and keypoints[b][2] > 0.3:
                pa = (int(keypoints[a][0]), int(keypoints[a][1]))
                pb = (int(keypoints[b][0]), int(keypoints[b][1]))
                cv2.line(img, pa, pb, (0, 200, 200), 2)

        # Draw keypoints
        for kp in keypoints:
            if kp[2] > 0.3:
                cv2.circle(img, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), -1)

        # Show gesture label near wrist
        if keypoints[0][2] > 0.3:
            wx, wy = int(keypoints[0][0]), int(keypoints[0][1])
            cv2.putText(
                img, gesture_text,
                (wx - 30, wy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

    def main_loop(window, stream):
        # Create DB connection in this thread (SQLite requires same-thread access)
        thread_conn = get_connection(args.db)
        init_db(thread_conn)
        try:
            for frame_result in stream:
                rgb_img = frame_result.image.asarray()
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                # Process hand keypoint detections
                # Attribute name matches pipeline stage name in hand-keypoints.yaml
                for det in frame_result.hand_keypoints:
                    keypoints = np.array(det.keypoints)  # (21, 3) array

                    # Run gesture recognition + stability check
                    result = trigger.update(keypoints, rgb_img.shape)

                    # Draw hand skeleton on display
                    if not args.headless:
                        draw_hand(bgr_img, keypoints, result["gesture"])
                        # Draw bounding box
                        if hasattr(det, 'box'):
                            bx1, by1, bx2, by2 = [int(v) for v in det.box]
                            color = (0, 0, 255) if trigger.paused else (0, 255, 0)
                            cv2.rectangle(bgr_img, (bx1, by1), (bx2, by2), color, 2)

                    # Handle scan trigger
                    region = result["region"]
                    if region is None:
                        continue

                    x1, y1, x2, y2 = region
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue

                    crop = rgb_img[y1:y2, x1:x2]
                    ts = int(time.time())
                    thumb_path = os.path.join(thumbnail_dir, f"hand_{ts}.jpg")
                    cv2.imwrite(thumb_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                    raw_text, confidence = extract_text(crop)
                    if not raw_text.strip():
                        logger.debug("OCR returned empty text, skipping")
                        continue

                    logger.info("OCR text: %s", raw_text[:100])
                    process_ocr_result(thread_conn, profile, raw_text, confidence, thumb_path)

                if not args.headless:
                    # Show pause overlay
                    if trigger.paused:
                        cv2.putText(
                            bgr_img, "PAUSED (show fist to resume)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        )

                    display_img = cv2.resize(bgr_img, (960, 540))
                    cv2.imshow("POV Label Tracker", display_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        logger.info("Manual snapshot — running OCR on full frame...")
                        raw_text, confidence = extract_text(
                            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        )
                        if raw_text.strip():
                            logger.info("OCR text:\n%s", raw_text)
                            snap_path = os.path.join(
                                thumbnail_dir, f"snap_{int(time.time())}.jpg"
                            )
                            cv2.imwrite(snap_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                            process_ocr_result(
                                thread_conn, profile, raw_text, confidence, snap_path
                            )
                        else:
                            logger.info("No text found in snapshot")

                if window.is_closed:
                    break

        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
        finally:
            thread_conn.close()
            if not args.headless:
                cv2.destroyAllWindows()

    with display.App(renderer=True) as app:
        wnd = app.create_window("POV Label Tracker", (960, 540))
        app.start_thread(main_loop, (wnd, stream), name="HandTracker")
        app.run()


def run_demo_mode(conn, profile):
    """Demo mode: process sample OCR texts without hardware."""
    samples = [
        (
            "Organic Whole Milk\nBEST BEFORE 20/03/2026\nCONTAINS: MILK\nMay contain: soy",
            0.92,
        ),
        (
            "Crunchy Peanut Butter\nUSE BY 15/06/2026\nCONTAINS: PEANUTS\nMay contain: tree nuts",
            0.88,
        ),
        (
            "Sourdough Bread\nBB 10/03/2026\nCONTAINS: WHEAT, GLUTEN",
            0.85,
        ),
    ]

    print("\n--- Label Tracker Demo Mode ---\n")
    for raw_text, confidence in samples:
        print(f"Processing: {raw_text.split(chr(10))[0]}")
        parsed, alerts_list = process_ocr_result(conn, profile, raw_text, confidence)
        if not alerts_list:
            print("  No alerts.")
        print()


def build_stream_kwargs(network, source, rtsp_latency=500):
    """Build kwargs for create_inference_stream supporting all Voyager source types.

    Supported source formats:
      - RTSP stream:       rtsp://user:password@ip:port/stream
      - Video file:        video.mp4  or  video.mp4@30  or  video.mp4@auto
      - Image directory:   path/to/images/  (scans recursively)
      - Single image:      image.jpg
    """
    kwargs = {
        "network": network,
        "sources": [source],
    }
    if source.startswith("rtsp://"):
        kwargs["rtsp_latency"] = rtsp_latency
    return kwargs


def main():
    cam_cfg = load_camera_config()
    ap = argparse.ArgumentParser(
        description="POV Label Tracker (Metis AIPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
source formats (Voyager SDK):
  RTSP stream:      rtsp://user:password@ip:port/stream
  Video file:       video.mp4  or  video.mp4@30  or  video.mp4@auto
  Image directory:  path/to/images/  (recursive)
  Single image:     image.jpg
""",
    )
    ap.add_argument(
        "--source",
        default=cam_cfg.get("source", "/dev/video0"),
        help="RTSP URL, video file (file.mp4[@fps|auto]), image dir, or single image",
    )
    ap.add_argument("--mode", choices=["object", "hand"], default="object",
                    help="Detection mode: 'object' (yolov8n-coco) or 'hand' (hand keypoints)")
    ap.add_argument("--demo", action="store_true", help="Demo mode (no hardware)")
    ap.add_argument("--headless", action="store_true", help="No display window")
    ap.add_argument("--rtsp-latency", type=int,
                    default=cam_cfg.get("rtsp_latency", 500), help="RTSP latency in ms")
    ap.add_argument("--profile", default=PROFILE_PATH, help="Household profile JSON")
    ap.add_argument("--db", default=DB_PATH, help="SQLite database path")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    logger.info("Loaded household profile with %d members", len(profile["members"]))

    conn = get_connection(args.db)
    init_db(conn)

    if args.demo:
        run_demo_mode(conn, profile)
    elif args.mode == "hand":
        run_hand_pipeline(args, conn, profile)
    else:
        run_metis_pipeline(args, conn, profile)

    conn.close()


if __name__ == "__main__":
    main()
