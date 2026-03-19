#!/usr/bin/env python
"""Rock Paper Scissors Lizard Spock — You vs Sheldon Cooper (on Metis AIPU).

Uses YOLOv10n trained on HaGRID dataset for direct gesture classification.

Rules (as explained by Sheldon):
  "Scissors cuts paper, paper covers rock, rock crushes lizard,
   lizard poisons Spock, Spock smashes scissors, scissors decapitates lizard,
   lizard eats paper, paper disproves Spock, Spock vaporizes rock,
   and as it always has, rock crushes scissors."

Usage:
    cd /home/orangepi/voyager-sdk && source venv/bin/activate
    cd /home/orangepi/Downloads/pov
    python src/rps_game.py
"""

import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
HAGRID_PIPELINE = os.path.join(BASE_DIR, "config", "hagrid-gestures.yaml")
CAMERA_CONFIG = os.path.join(BASE_DIR, "config", "camera.json")


def load_camera_config():
    """Load camera source and settings from config/camera.json."""
    if os.path.exists(CAMERA_CONFIG):
        with open(CAMERA_CONFIG) as f:
            return json.load(f)
    return {}

# Map HaGRID gesture classes to RPSLS moves
GESTURE_TO_MOVE = {
    "rock": "ROCK",
    "fist": "ROCK",
    "peace": "SCISSORS",
    "peace_inverted": "SCISSORS",
    "palm": "PAPER",
    "stop": "PAPER",
    "stop_inverted": "PAPER",
    "grip": "LIZARD",
    "grabbing": "LIZARD",
    "two_up": "SPOCK",
    "two_up_inverted": "SPOCK",
    "four": "SPOCK",
}

CHOICES = ["ROCK", "PAPER", "SCISSORS", "LIZARD", "SPOCK"]

# Each move beats these two
BEATS = {
    "ROCK":     ["SCISSORS", "LIZARD"],
    "PAPER":    ["ROCK", "SPOCK"],
    "SCISSORS": ["PAPER", "LIZARD"],
    "LIZARD":   ["PAPER", "SPOCK"],
    "SPOCK":    ["SCISSORS", "ROCK"],
}

# Flavor text — Sheldon style
WIN_VERB = {
    ("ROCK", "SCISSORS"):     "crushes",
    ("ROCK", "LIZARD"):       "crushes",
    ("PAPER", "ROCK"):        "covers",
    ("PAPER", "SPOCK"):       "disproves",
    ("SCISSORS", "PAPER"):    "cuts",
    ("SCISSORS", "LIZARD"):   "decapitates",
    ("LIZARD", "PAPER"):      "eats",
    ("LIZARD", "SPOCK"):      "poisons",
    ("SPOCK", "SCISSORS"):    "smashes",
    ("SPOCK", "ROCK"):        "vaporizes",
}

GESTURE_HINTS = {
    "ROCK":     "fist",
    "PAPER":    "open palm",
    "SCISSORS": "peace sign",
    "LIZARD":   "grip / grab",
    "SPOCK":    "4 fingers / vulcan",
}

# --- Sheldon quotes ---

SHELDON_WIN = [
    "Bazinga!",
    "I win. Obviously.",
    "As I predicted.",
    "You can't beat me. I'm Sheldon Cooper.",
    "Another victory for science.",
    "I'm not crazy. My mother had me tested.",
    "That's my spot... on top of the leaderboard.",
    "Bazinga! Did you really think you could win?",
    "I have an IQ of 187. What did you expect?",
    "This is hardly a challenge.",
]

SHELDON_LOSE = [
    "That's... statistically improbable.",
    "I demand a do-over!",
    "This game is flawed.",
    "You clearly cheated. I'm filing a complaint.",
    "I'm not upset. I just need to recalibrate.",
    "Impossible! Let me check the rules again.",
    "I blame cosmic rays.",
    "This wouldn't have happened on Vulcan.",
    "Fine. But this changes nothing about my superiority.",
    "I'll be in my room. Knock three times.",
]

SHELDON_DRAW = [
    "Great minds think alike. Well... one great mind.",
    "A tie? How pedestrian.",
    "Even a broken clock is right twice a day.",
    "We appear to have reached an impasse.",
    "Fascinating. You matched my intellect. Briefly.",
]

SHELDON_WAITING = [
    "I'm waiting...",
    "Sometime today would be nice.",
    "Any day now.",
    "Tick tock.",
    "The anticipation is... tedious.",
    "Show me what you've got. It won't matter.",
    "I've already calculated all possible outcomes.",
]

SHELDON_COUNTDOWN = [
    "Rock, paper, scissors, lizard, Spock!",
    "Prepare to lose.",
    "Here we go...",
    "May the best mind win. That's me.",
]

# Game states
WAITING = "waiting"
COUNTDOWN = "countdown"
RESULT = "result"

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)

MOVE_COLOR = {
    "ROCK": WHITE,
    "PAPER": CYAN,
    "SCISSORS": YELLOW,
    "LIZARD": GREEN,
    "SPOCK": MAGENTA,
}


ASSETS_DIR = os.path.join(BASE_DIR, "assets", "sheldon")


def load_sheldon_images():
    """Load Sheldon reaction images, resize to consistent height."""
    import cv2
    images = {}
    files = {
        "win": "Bazinga.jpg",
        "lose": "lose.jpg",
        "draw": "draw.jpg",
        "waiting": "waiting.jpg",
    }
    for key, filename in files.items():
        path = os.path.join(ASSETS_DIR, filename)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Resize to 200px height, keep aspect ratio
                target_h = 200
                scale = target_h / img.shape[0]
                target_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (target_w, target_h))
                images[key] = img
                logger.info("Loaded sheldon/%s (%dx%d)", filename, target_w, target_h)
    return images


def overlay_image(frame, img, x, y):
    """Overlay an image onto the frame at position (x, y), handling bounds."""
    fh, fw = frame.shape[:2]
    ih, iw = img.shape[:2]

    # Clamp to frame bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + iw)
    y2 = min(fh, y + ih)

    if x2 <= x1 or y2 <= y1:
        return

    # Source region
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    if img.shape[2] == 4:
        # Has alpha channel
        alpha = img[sy1:sy2, sx1:sx2, 3:4] / 255.0
        rgb = img[sy1:sy2, sx1:sx2, :3]
        frame[y1:y2, x1:x2] = (rgb * alpha + frame[y1:y2, x1:x2] * (1 - alpha)).astype(frame.dtype)
    else:
        frame[y1:y2, x1:x2] = img[sy1:sy2, sx1:sx2]


def get_label(det):
    """Extract gesture label string from a detection object."""
    try:
        return det.label.name
    except (NotImplementedError, AttributeError):
        pass
    labels = det._meta.labels
    try:
        return labels(det.class_id).name
    except (TypeError, ValueError):
        pass
    try:
        return labels[det.class_id]
    except (KeyError, IndexError):
        return str(det.class_id)


def check_winner(player, sheldon):
    """Returns ('player'|'sheldon'|'draw', description)."""
    if player == sheldon:
        return "draw", "Same move!"
    if sheldon in BEATS[player]:
        verb = WIN_VERB.get((player, sheldon), "beats")
        return "player", f"{player} {verb} {sheldon}!"
    else:
        verb = WIN_VERB.get((sheldon, player), "beats")
        return "sheldon", f"{sheldon} {verb} {player}!"


def run_game(args):
    try:
        from axelera.app.stream import create_inference_stream
    except ImportError:
        logger.error("Voyager SDK not available. Activate venv first.")
        sys.exit(1)

    import cv2

    source = args.source
    stream_kwargs = {
        "network": HAGRID_PIPELINE,
        "sources": [source],
    }
    if source.startswith("rtsp://"):
        stream_kwargs["rtsp_latency"] = args.rtsp_latency

    stream = create_inference_stream(**stream_kwargs)

    sheldon_imgs = load_sheldon_images()
    logger.info("Loaded %d Sheldon images", len(sheldon_imgs))
    logger.info("Rock Paper Scissors Lizard Spock — You vs Sheldon Cooper!")

    # Game state
    state = WAITING
    countdown_start = 0.0
    player_move = None
    sheldon_move = None
    result_text = ""
    result_desc = ""
    result_color = WHITE
    result_time = 0.0
    sheldon_quote = ""
    score_player = 0
    score_sheldon = 0
    score_draw = 0
    rounds = 0
    waiting_quote = random.choice(SHELDON_WAITING)
    waiting_quote_time = time.monotonic()
    countdown_quote = ""

    # Gesture stability tracking
    last_move = None
    stable_since = 0.0
    STABLE_TIME = 0.5
    COUNTDOWN_SEC = 3
    RESULT_SEC = 5

    try:
        for frame_result in stream:
            rgb_img = frame_result.image.asarray()
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            h, w = bgr_img.shape[:2]
            now = time.monotonic()

            # --- Process detections ---
            current_move = None
            current_gesture = None
            best_conf = 0.0

            for det in frame_result.gesture_detections:
                label = get_label(det)
                conf = det.score
                box = [int(v) for v in det.box]

                is_game = label in GESTURE_TO_MOVE
                color = GREEN if is_game else (60, 60, 60)
                cv2.rectangle(bgr_img, (box[0], box[1]), (box[2], box[3]), color, 2)
                disp = f"{GESTURE_TO_MOVE[label]}" if is_game else label
                cv2.putText(bgr_img, f"{disp} {conf:.2f}",
                            (box[0], box[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_game and conf > best_conf:
                    best_conf = conf
                    current_move = GESTURE_TO_MOVE[label]
                    current_gesture = label

            # --- Gesture stability ---
            if current_move and current_move == last_move:
                if stable_since == 0:
                    stable_since = now
            else:
                stable_since = now
            last_move = current_move
            stable_duration = now - stable_since if stable_since else 0

            # --- State machine ---
            if state == WAITING:
                # Rotate Sheldon's waiting quotes
                if now - waiting_quote_time > 4.0:
                    waiting_quote = random.choice(SHELDON_WAITING)
                    waiting_quote_time = now

                if current_move and stable_duration >= STABLE_TIME:
                    state = COUNTDOWN
                    countdown_start = now
                    countdown_quote = random.choice(SHELDON_COUNTDOWN)

            elif state == COUNTDOWN:
                elapsed = now - countdown_start
                remaining = COUNTDOWN_SEC - elapsed

                if remaining <= 0:
                    player_move = current_move if current_move else last_move or "ROCK"
                    sheldon_move = random.choice(CHOICES)
                    rounds += 1

                    winner, desc = check_winner(player_move, sheldon_move)
                    result_desc = desc
                    if winner == "draw":
                        result_text = "DRAW!"
                        result_color = YELLOW
                        score_draw += 1
                        sheldon_quote = random.choice(SHELDON_DRAW)
                    elif winner == "player":
                        result_text = "YOU WIN!"
                        result_color = GREEN
                        score_player += 1
                        sheldon_quote = random.choice(SHELDON_LOSE)
                    else:
                        result_text = "SHELDON WINS!"
                        result_color = RED
                        score_sheldon += 1
                        sheldon_quote = random.choice(SHELDON_WIN)

                    result_time = now
                    state = RESULT
                    last_move = None
                    stable_since = 0
                    logger.info("Round %d: You=%s vs Sheldon=%s -> %s",
                                rounds, player_move, sheldon_move, result_text)

            elif state == RESULT:
                if now - result_time >= RESULT_SEC:
                    state = WAITING
                    player_move = None
                    sheldon_move = None
                    waiting_quote = random.choice(SHELDON_WAITING)
                    waiting_quote_time = now

            # --- Draw HUD ---

            # Scoreboard — top left
            cv2.rectangle(bgr_img, (0, 0), (440, 75), BLACK, -1)
            cv2.putText(bgr_img, f"YOU {score_player}  -  {score_sheldon} SHELDON",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            cv2.putText(bgr_img, f"Draws: {score_draw}  |  Round {rounds + 1}",
                        (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Title — top right
            cv2.rectangle(bgr_img, (w - 360, 0), (w, 30), BLACK, -1)
            cv2.putText(bgr_img, "ROCK PAPER SCISSORS LIZARD SPOCK",
                        (w - 355, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MAGENTA, 1)

            if state == WAITING:
                # Sheldon quote bubble
                cv2.rectangle(bgr_img, (0, h - 110), (w, h), BLACK, -1)

                # Sheldon waiting image — bottom left
                if "waiting" in sheldon_imgs:
                    overlay_image(bgr_img, sheldon_imgs["waiting"], 10, h - 210)

                # Quote
                q_sz = cv2.getTextSize(f'Sheldon: "{waiting_quote}"',
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                cv2.putText(bgr_img, f'Sheldon: "{waiting_quote}"',
                            ((w - q_sz[0]) // 2, h - 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2)

                if current_move:
                    # Progress bar
                    bar_pct = min(stable_duration / STABLE_TIME, 1.0)
                    bar_w = int(300 * bar_pct)
                    mc = MOVE_COLOR.get(current_move, GREEN)
                    cv2.rectangle(bgr_img, (w // 2 - 150, h - 50),
                                  (w // 2 - 150 + bar_w, h - 35), mc, -1)
                    cv2.rectangle(bgr_img, (w // 2 - 150, h - 50),
                                  (w // 2 + 150, h - 35), WHITE, 2)
                    cv2.putText(bgr_img, f"{current_move}",
                                (w // 2 - 50, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mc, 2)
                else:
                    # Gesture hints
                    hints = "  |  ".join(f"{m}: {GESTURE_HINTS[m]}" for m in CHOICES)
                    sz = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.putText(bgr_img, hints,
                                ((w - sz[0]) // 2, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            elif state == COUNTDOWN:
                remaining = max(0, COUNTDOWN_SEC - (now - countdown_start))
                num = str(int(remaining) + 1)
                sz = cv2.getTextSize(num, cv2.FONT_HERSHEY_DUPLEX, 6, 6)[0]
                cx, cy = (w - sz[0]) // 2, (h + sz[1]) // 2

                # Big number with shadow
                cv2.putText(bgr_img, num, (cx + 3, cy + 3),
                            cv2.FONT_HERSHEY_DUPLEX, 6, BLACK, 8)
                cv2.putText(bgr_img, num, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 6, YELLOW, 6)

                # Sheldon's countdown quote
                cv2.rectangle(bgr_img, (0, h - 50), (w, h), BLACK, -1)
                q_sz = cv2.getTextSize(f'Sheldon: "{countdown_quote}"',
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.putText(bgr_img, f'Sheldon: "{countdown_quote}"',
                            ((w - q_sz[0]) // 2, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)

                # Show locked move
                if current_move:
                    mc = MOVE_COLOR.get(current_move, WHITE)
                    cv2.putText(bgr_img, current_move,
                                (cx - 30, cy + 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, mc, 2)

            elif state == RESULT:
                # Dark overlay
                overlay = bgr_img.copy()
                cv2.rectangle(overlay, (0, h // 6), (w, 5 * h // 6 + 30), BLACK, -1)
                cv2.addWeighted(overlay, 0.8, bgr_img, 0.2, 0, bgr_img)

                cy = h // 3

                # Sheldon reaction image — right side
                if "SHELDON" in result_text and "win" in sheldon_imgs:
                    overlay_image(bgr_img, sheldon_imgs["win"], w - sheldon_imgs["win"].shape[1] - 20, cy - 60)
                elif "YOU" in result_text and "lose" in sheldon_imgs:
                    overlay_image(bgr_img, sheldon_imgs["lose"], w - sheldon_imgs["lose"].shape[1] - 20, cy - 60)
                elif "DRAW" in result_text and "draw" in sheldon_imgs:
                    overlay_image(bgr_img, sheldon_imgs["draw"], w - sheldon_imgs["draw"].shape[1] - 20, cy - 60)

                # YOU vs SHELDON — left side
                pc = MOVE_COLOR.get(player_move, WHITE)
                sc = MOVE_COLOR.get(sheldon_move, WHITE)

                cv2.putText(bgr_img, f"YOU:  {player_move}",
                            (w // 8, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, pc, 3)
                cv2.putText(bgr_img, f"SHELDON:  {sheldon_move}",
                            (w // 8, cy + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, sc, 3)

                # Win description
                cv2.putText(bgr_img, result_desc,
                            (w // 8, cy + 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                # Result text — center
                sz = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, 2.2, 3)[0]
                rx = (w - sz[0]) // 2
                cv2.putText(bgr_img, result_text, (rx, cy + 165),
                            cv2.FONT_HERSHEY_DUPLEX, 2.2, result_color, 3)

                # Sheldon quote
                q_sz = cv2.getTextSize(f'"{sheldon_quote}"',
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(bgr_img, f'"{sheldon_quote}"',
                            ((w - q_sz[0]) // 2, cy + 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

            # --- Display ---
            display_img = cv2.resize(bgr_img, (960, 540))
            cv2.imshow("You vs Sheldon Cooper", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        stream.stop()
        logger.info("Final score: YOU %d - %d SHELDON (%d draws in %d rounds)",
                    score_player, score_sheldon, score_draw, rounds)


def main():
    cam_cfg = load_camera_config()
    ap = argparse.ArgumentParser(
        description="Rock Paper Scissors Lizard Spock — You vs Sheldon Cooper",
    )
    ap.add_argument(
        "--source",
        default=cam_cfg.get("source", "/dev/video0"),
        help="Camera source (RTSP, /dev/videoN, file.mp4)",
    )
    ap.add_argument("--rtsp-latency", type=int,
                    default=cam_cfg.get("rtsp_latency", 500))
    args = ap.parse_args()
    run_game(args)


if __name__ == "__main__":
    main()
