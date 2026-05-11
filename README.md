# Rio_meme

> 🌐 **English** · [简体中文](README.zh-CN.md)

A webcam toy that mirrors your hands and face with a meme of **Rio**, an Australian Shepherd. Show a peace sign — peace-sign Rio pops up. Stick your tongue out — tongue-out Rio pops up. Make a heart with both hands — giant heart Rio.

Runs locally on a Mac. No accounts, no servers, no upload — everything happens on your laptop. Built on Google's MediaPipe Tasks API.

![idle](docs/screenshots/ui-idle.png)

| Hand gesture | Face expression | Two-hand combo |
|---|---|---|
| ![gesture](docs/screenshots/ui-gesture.png) | ![face](docs/screenshots/ui-face.png) | ![combo](docs/screenshots/ui-combo.png) |

---

## Quick start

### macOS

1. Download the project (Code → Download ZIP, or `git clone`).
2. Double-click **`start.command`**.
3. First run: macOS may say *"start.command cannot be opened because Apple cannot verify it"*. Fix:
   - **Right-click** `start.command` → **Open** → **Open** in the warning dialog.
   - Or: System Settings → Privacy & Security → scroll to the bottom → **Open Anyway**.
4. The launcher will:
   - Find or install Python 3.9+ (opens the python.org installer if missing).
   - Create a local `.venv/` and `pip install` opencv-python + mediapipe + numpy.
   - Download the two MediaPipe `.task` model files (~12 MB).
   - Launch the app.
5. macOS will ask for camera permission — say yes, then **fully quit your terminal (Cmd+Q)** and double-click `start.command` again. (macOS only delivers the new permission to *new* processes.)

After the first run everything is cached. Subsequent launches are 1–2 seconds.

### Other platforms

Linux / Windows aren't auto-supported by `start.command`, but the app itself is cross-platform:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install opencv-python mediapipe numpy
curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
curl -L -o face_landmarker.task   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python hand_meme.py
```

Press **q** or **Esc** to quit.

---

## What it detects (28 events)

### Hand gestures — 14, detected on either hand independently

![hand gestures](docs/screenshots/gallery-hands.png)

`Thumb_Up` 👍 · `Thumb_Down` 👎 · `Open_Palm` ✋ · `Closed_Fist` ✊ · `Pointing_Up` ☝️ · `Victory` ✌️ · `OK` 👌 · `Rock` 🤘 · `Call_Me` 🤙 · `Three` 🤟 · `Middle_Finger` 🖕 · `Finger_Heart` 🫶 · (`ILoveYou`, `Four` — meme images optional)

The first 7 come from MediaPipe's built-in gesture recognizer. The rest are custom landmark-geometry classifiers in [`hand_meme.py`](hand_meme.py).

### Two-hand combos — 2

![two-hand combos](docs/screenshots/gallery-combos.png)

`Heart` 💕 · `Clap` 👏

When a two-hand combo fires, it overrides both per-hand panels.

### Face expressions — 12

![face expressions](docs/screenshots/gallery-faces.png)

`Smile` · `Laugh` · `Sad` · `Angry` · `Surprised` · `Kiss` · `Wink` · `Blink` · `Tongue_Out` · `Cheek_Puff` · `Eyebrow_Raise` · `Neutral`

These come from MediaPipe's FaceLandmarker blendshape coefficients (52 ARKit-style values), combined into a small rule-based classifier.

---

## How it works

```
   webcam frame
        │
        ├──► GestureRecognizer (gesture_recognizer.task)
        │        ├─► per-hand built-in label
        │        └─► per-hand landmarks ─► custom landmark classifier
        │                                       │
        │                                       └─► OK / Rock / Call_Me / Three / Four /
        │                                            Finger_Heart / Middle_Finger
        │
        └──► FaceLandmarker   (face_landmarker.task)
                 └─► 52 blendshape coefficients ─► rule-based expression classifier
                                                       │
                                                       └─► Smile / Sad / Surprised / Kiss / …

   ┌─────────────────────────────────────────────────────────────────────┐
   │  per-channel debouncer (Smoother — latches after N consistent frames)│
   └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                priority pick: combo > most-recently-changed channel
                                       │
                                       ▼
                                draw the meme
```

Everything runs on the CPU (XNNPACK delegate). On a 2021 MacBook Pro M1 Pro it sits around 30 FPS.

---

## Customizing memes

Each event maps to one PNG in [`memes/`](memes/). Filenames are fixed — drop in your own image with the right filename and the next run picks it up:

| File | Used for |
|---|---|
| `thumbs_up.png` | Thumb_Up gesture |
| `smile.png` | Smile expression |
| `heart.png` | Two-hand Heart combo |
| `kiss.png` | Kiss expression |
| `idle.png` | Default / fallback when something's missing |
| … | (see [`hand_meme.py`](hand_meme.py) `GESTURE_TO_MEME`, `TWO_HAND_COMBOS`, `EXPRESSION_TO_MEME`) |

If a file is missing, the script logs a warning and uses `idle.png` for that event. Transparent PNGs are supported (alpha is composited onto a warm off-white background).

---

## Tuning detection

If a gesture or expression doesn't fire when you expect it to, the thresholds live in two functions:

- [`classify_custom_gesture`](hand_meme.py) — landmark-geometry rules for OK / Rock / Call_Me / Three / Four / Finger_Heart / Middle_Finger.
- [`classify_expression`](hand_meme.py) — blendshape thresholds for the 12 face expressions.

Each rule has a comment explaining what signal it's reading. Lower a threshold to make an event fire more easily; raise it to reduce false positives.

The two-hand combo detector is in [`detect_two_hand_combo`](hand_meme.py).

---

## Project layout

```
Rio_meme/
├── start.command            ← double-click on macOS
├── hand_meme.py             ← the app
├── memes/                   ← per-event meme images (PNG)
├── tools/
│   └── make_screenshots.py  ← regenerates docs/ previews from memes/
├── docs/
│   └── screenshots/         ← README images (no real faces)
├── README.md
├── LICENSE
└── .gitignore               (excludes .venv, *.task, python-installer.pkg)
```

The `.task` model files and `python-installer.pkg` are intentionally **not** committed:
- The two `.task` files are auto-downloaded by `start.command` on first run.
- `python-installer.pkg` (44 MB) is meant to be dropped in locally when you want to share the folder as a self-contained zip with non-technical friends. See [Distribution](#distribution).

---

## Distribution

To share with a friend who doesn't have Python:

1. Download the Python macOS installer from <https://www.python.org/downloads/macos/> and rename it to `python-installer.pkg`. Place it next to `start.command`.
2. Zip the whole project folder.
3. Send it (AirDrop, iMessage, etc.).
4. Tell them to unzip, then **right-click `start.command` → Open** the first time (see Quick Start step 3).

Total zip size ≈ 100 MB (Python installer 44 MB + meme images ~50 MB).

---

## Credits

- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide) — Google's on-device ML framework. The `gesture_recognizer` and `face_landmarker` models are the core.
- Meme images in this repo are of **Rio**, an Australian Shepherd, generated with ChatGPT image generation. Swap them for whatever (or whoever) you want.

## License

MIT. See [`LICENSE`](LICENSE).
