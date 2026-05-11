"""
Generate README screenshots — mock-up of the live UI without using anyone's webcam.
The webcam thumbnail is replaced by a neutral placeholder so screenshots don't include
the user's face.

Run from project root:
    python tools/make_screenshots.py
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MEMES = ROOT / "memes"
OUT = ROOT / "docs" / "screenshots"


def composite_meme(canvas, meme_img, x, y, side):
    img = cv2.resize(meme_img, (side, side), interpolation=cv2.INTER_AREA)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        roi = canvas[y:y + side, x:x + side].astype(np.float32)
        canvas[y:y + side, x:x + side] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        canvas[y:y + side, x:x + side] = img


def render_screenshot(meme_name: str, label: str) -> np.ndarray:
    """Render a UI mock-up: full-window meme + webcam placeholder + bottom label."""
    w, h = 1280, 720
    canvas = np.full((h, w, 3), (235, 240, 245), dtype=np.uint8)

    meme = cv2.imread(str(MEMES / f"{meme_name}.png"), cv2.IMREAD_UNCHANGED)
    if meme is None:
        raise FileNotFoundError(MEMES / f"{meme_name}.png")
    side = min(w, h) - 60
    composite_meme(canvas, meme, (w - side) // 2, (h - side) // 2, side)

    # Webcam-placeholder block (no real footage).
    mw, mh = 280, 158
    margin = 22
    mx = w - mw - margin
    my = margin
    border = 4
    cv2.rectangle(canvas, (mx - border, my - border),
                  (mx + mw + border, my + mh + border), (255, 255, 255), -1)
    canvas[my:my + mh, mx:mx + mw] = (130, 138, 150)
    cv2.putText(canvas, "your webcam", (mx + 62, my + mh // 2 + 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (225, 228, 235), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (mx - border, my - border),
                  (mx + mw + border, my + mh + border), (210, 210, 215), 1, cv2.LINE_AA)

    if label:
        fscale = 1.1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fscale, 2)
        tx = (w - tw) // 2
        ty = h - 36
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                    fscale, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                    fscale, (60, 60, 70), 2, cv2.LINE_AA)
    return canvas


def make_gallery(meme_names, cols=5, tile=240, gap=12, pad=24) -> np.ndarray:
    """Grid of meme thumbnails — used as the 'detectable events' gallery in the README."""
    rows = (len(meme_names) + cols - 1) // cols
    w = pad * 2 + cols * tile + (cols - 1) * gap
    h = pad * 2 + rows * (tile + 32) + (rows - 1) * gap
    canvas = np.full((h, w, 3), (245, 246, 250), dtype=np.uint8)

    for i, name in enumerate(meme_names):
        r, c = divmod(i, cols)
        x = pad + c * (tile + gap)
        y = pad + r * (tile + 32 + gap)
        img = cv2.imread(str(MEMES / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            cv2.rectangle(canvas, (x, y), (x + tile, y + tile), (220, 220, 225), -1)
        else:
            composite_meme(canvas, img, x, y, tile)
        label = name.replace("_", " ").title()
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.putText(canvas, label, (x + (tile - tw) // 2, y + tile + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (60, 60, 70), 1, cv2.LINE_AA)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Three UI mock-ups: idle, a hand gesture, a face expression.
    cv2.imwrite(str(OUT / "ui-idle.png"),     render_screenshot("idle", ""))
    cv2.imwrite(str(OUT / "ui-gesture.png"),  render_screenshot("thumbs_up", "Thumbs Up"))
    cv2.imwrite(str(OUT / "ui-face.png"),     render_screenshot("laugh", "Laugh"))
    cv2.imwrite(str(OUT / "ui-combo.png"),    render_screenshot("heart", "Heart"))

    # Meme gallery — grouped by category for the README.
    cv2.imwrite(str(OUT / "gallery-hands.png"), make_gallery([
        "thumbs_up", "thumbs_down", "open_palm", "fist", "point",
        "peace", "ok", "rock", "call_me", "three",
        "middle_finger", "finger_heart",
    ], cols=6, tile=200))
    cv2.imwrite(str(OUT / "gallery-combos.png"), make_gallery(
        ["heart", "clap"], cols=2, tile=240))
    cv2.imwrite(str(OUT / "gallery-faces.png"), make_gallery([
        "smile", "laugh", "sad", "angry", "surprised", "kiss",
        "wink", "blink", "tongue_out", "cheek_puff", "eyebrow_raise", "neutral",
    ], cols=6, tile=200))

    print(f"Wrote screenshots to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
