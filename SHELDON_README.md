# Rock Paper Scissors Lizard Spock — You vs Sheldon Cooper

A real-time hand gesture game powered by the Axelera Metis AIPU. Show your hand to the camera and play Rock Paper Scissors Lizard Spock against Sheldon Cooper — complete with Bazinga quotes, reaction images, and a live scoreboard.

All inference runs on-device. No cloud. No external APIs.

---

## How It Works

1. A YOLOv10n model (trained on HaGRID dataset) detects your hand gesture through the camera.
2. Hold a gesture steady for **0.5 seconds** to lock it in.
3. A **3-second countdown** begins.
4. Sheldon randomly picks his move and the result is displayed with a snarky quote.
5. After **5 seconds** the next round starts automatically.

Press **q** to quit.

---

## Gesture Mapping

| Hand Gesture         | Game Move    |
|----------------------|--------------|
| Fist / Rock          | ROCK         |
| Open palm / Stop     | PAPER        |
| Peace sign (V)       | SCISSORS     |
| Grip / Grabbing      | LIZARD       |
| Two fingers up / Four| SPOCK        |

---

## Rules

> *"Scissors cuts paper, paper covers rock, rock crushes lizard,
> lizard poisons Spock, Spock smashes scissors, scissors decapitates lizard,
> lizard eats paper, paper disproves Spock, Spock vaporizes rock,
> and as it always has, rock crushes scissors."*
> — Sheldon Cooper

Each move beats exactly two others:

| Move     | Beats              |
|----------|---------------------|
| Rock     | Scissors, Lizard    |
| Paper    | Rock, Spock         |
| Scissors | Paper, Lizard       |
| Lizard   | Paper, Spock        |
| Spock    | Scissors, Rock      |

---

## Requirements

- **Hardware:** Axelera Metis AIPU + USB or RTSP camera
- **OS:** Ubuntu 22.04
- **SDK:** Axelera Voyager SDK (with GStreamer)
- **Python:** 3.x with OpenCV (`cv2`)
- **Model weights:** `weights/yolov10n-hagrid-gestures.onnx` (34-class HaGRID gesture detector)

---

## Quick Start

```bash
# Activate the Voyager SDK environment
cd /home/orangepi/voyager-sdk && source venv/bin/activate

# Run the game (default: RTSP camera)
cd /home/orangepi/Downloads/pov
python src/rps_game.py

# Or use a USB camera
python src/rps_game.py --source /dev/video0

# Or use a video file
python src/rps_game.py --source test_video.mp4
```

### CLI Options

| Flag              | Default                                                  | Description                        |
|-------------------|----------------------------------------------------------|------------------------------------|
| `--source`        | From `config/camera.json`, or `/dev/video0`              | Camera source (RTSP, device, file) |
| `--rtsp-latency`  | `500`                                                    | RTSP latency in ms                 |

---

## Project Files

```
src/rps_game.py                        # Game logic and HUD rendering
config/hagrid-gestures.yaml            # Voyager SDK pipeline definition
config/hagrid.names                    # 34 HaGRID gesture class names
weights/yolov10n-hagrid-gestures.onnx  # YOLOv10n model weights (not in repo)
assets/sheldon/                        # Sheldon reaction images
├── Bazinga.jpg                        #   Sheldon wins
├── lose.jpg                           #   You win
├── draw.jpg                           #   Draw
└── waiting.jpg                        #   Waiting for your move
```

---

## HUD Overview

- **Top-left:** Scoreboard (your score vs Sheldon's, draws, round number)
- **Top-right:** Game title
- **Bottom:** Sheldon quote bubble + gesture hints or stability progress bar
- **Result screen:** Move comparison, win/lose/draw banner, Sheldon reaction image + quote

---

## Sheldon's Personality

Sheldon has unique quote pools for each situation:

- **Waiting:** *"I'm waiting..."*, *"Tick tock."*, *"I've already calculated all possible outcomes."*
- **Wins:** *"Bazinga!"*, *"I win. Obviously."*, *"I have an IQ of 187. What did you expect?"*
- **Loses:** *"That's... statistically improbable."*, *"I demand a do-over!"*, *"I blame cosmic rays."*
- **Draws:** *"Great minds think alike. Well... one great mind."*, *"A tie? How pedestrian."*
