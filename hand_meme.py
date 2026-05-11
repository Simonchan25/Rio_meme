"""
Hand gesture + face expression meme display.
Show your hands and face to the camera; matching memes pop up on the right.

Uses MediaPipe Tasks API:
  - GestureRecognizer (gesture_recognizer.task)  -> up to 2 hands
  - FaceLandmarker   (face_landmarker.task)      -> 1 face, with blendshapes

Detected events (27):
  Per-hand gestures (13):
    Thumb_Up, Thumb_Down, Open_Palm, Closed_Fist, Pointing_Up, Victory,
    ILoveYou, OK, Rock, Call_Me, Three, Four, Finger_Heart
  Two-hand combos (2):  Heart, Clap
  Face expressions (12):
    Smile, Laugh, Sad, Angry, Surprised, Kiss, Wink, Blink,
    Tongue_Out, Cheek_Puff, Eyebrow_Raise, Neutral
"""

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = Path(__file__).parent
MEMES_DIR = ROOT / "memes"
GESTURE_MODEL_PATH = ROOT / "gesture_recognizer.task"
FACE_MODEL_PATH = ROOT / "face_landmarker.task"
WINDOW_NAME = "Hand Meme — press q to quit"

GESTURE_TO_MEME = {
    "Thumb_Up":      "thumbs_up",
    "Thumb_Down":    "thumbs_down",
    "Open_Palm":     "open_palm",
    "Closed_Fist":   "fist",
    "Pointing_Up":   "point",
    "Victory":       "peace",
    "ILoveYou":      "iloveyou",
    "OK":            "ok",
    "Rock":          "rock",
    "Call_Me":       "call_me",
    "Three":         "three",
    "Four":          "four",
    "Finger_Heart":  "finger_heart",
    "Middle_Finger": "middle_finger",
    "None":          "idle",
}

# Two-hand combos override the individual hand panels when active.
TWO_HAND_COMBOS = {
    "Heart": "heart",
    "Clap":  "clap",
}

EXPRESSION_TO_MEME = {
    "Smile":         "smile",
    "Laugh":         "laugh",
    "Sad":           "sad",
    "Angry":         "angry",
    "Surprised":     "surprised",
    "Kiss":          "kiss",
    "Wink":          "wink",
    "Blink":         "blink",
    "Tongue_Out":    "tongue_out",
    "Cheek_Puff":    "cheek_puff",
    "Eyebrow_Raise": "eyebrow_raise",
    "Neutral":       "neutral",
}


def load_memes():
    """Load every referenced PNG; fall back to idle.png for any that are missing."""
    needed = (set(GESTURE_TO_MEME.values())
              | set(TWO_HAND_COMBOS.values())
              | set(EXPRESSION_TO_MEME.values())
              | {"idle"})
    memes = {}
    missing = []
    for name in sorted(needed):
        path = MEMES_DIR / f"{name}.png"
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            missing.append(name)
        else:
            memes[name] = img
    if "idle" not in memes:
        sys.exit(f"Required image missing: {MEMES_DIR}/idle.png")
    if missing:
        print(f"NOTE: falling back to idle.png for {len(missing)} missing meme(s): {missing}")
        for name in missing:
            memes[name] = memes["idle"]
    return memes


# ---------- Drawing ----------

def put_label(img, text, pos, scale=0.9, color=(40, 40, 40), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_meme_panel(meme_img, panel_size, title=None, subtitle=None):
    w, h = panel_size
    panel = np.full((h, w, 3), (235, 240, 245), dtype=np.uint8)
    side = max(40, min(w, h) - 60)
    img = cv2.resize(meme_img, (side, side), interpolation=cv2.INTER_AREA)
    x = (w - side) // 2
    y = (h - side) // 2
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        roi = panel[y:y + side, x:x + side].astype(np.float32)
        panel[y:y + side, x:x + side] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        panel[y:y + side, x:x + side] = img
    if title:
        put_label(panel, title, (16, 32), scale=0.7)
    if subtitle:
        put_label(panel, subtitle, (16, h - 16), scale=0.55, thickness=1)
    return panel


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand(frame, landmarks, color=(60, 200, 80)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, color, -1, cv2.LINE_AA)


def draw_face_mesh(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (180, 180, 255), -1, cv2.LINE_AA)


# ---------- Geometry helpers for custom gesture detection ----------

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _norm_pts(landmarks, w, h):
    return [(lm.x * w, lm.y * h) for lm in landmarks]


def _finger_extended(pts, tip, pip, mcp):
    """Finger is extended when the tip is significantly farther from its MCP joint
    than the PIP joint is. MCP-anchored so it doesn't break when the hand tilts."""
    return _dist(pts[tip], pts[mcp]) > _dist(pts[pip], pts[mcp]) * 1.15


def _thumb_extended(pts):
    """Thumb tip is far from the thumb-MCP relative to where the IP joint sits."""
    return _dist(pts[4], pts[2]) > _dist(pts[3], pts[2]) * 1.40


def classify_custom_gesture(landmarks, w, h):
    """Detect gestures MediaPipe's built-in classifier doesn't reliably cover.
    Returns one of: OK, Rock, Call_Me, Three, Four, Finger_Heart, or None.
    """
    pts = _norm_pts(landmarks, w, h)
    palm = _dist(pts[0], pts[9])
    if palm < 1:
        return None

    idx_up = _finger_extended(pts, 8, 6, 5)
    mid_up = _finger_extended(pts, 12, 10, 9)
    rng_up = _finger_extended(pts, 16, 14, 13)
    pky_up = _finger_extended(pts, 20, 18, 17)
    thb_up = _thumb_extended(pts)
    # Middle_Finger 🖕 — checked BEFORE the pinch block, because in a curled fist the thumb
    # can land near the (also-curled) index tip and accidentally trigger the OK/Finger_Heart
    # branch. mid up + idx and pky folded is the strongest single-frame signal; ring and
    # thumb are unconstrained (ring is biologically coupled to middle, thumb often rests
    # at the side).
    if mid_up and not idx_up and not pky_up:
        return "Middle_Finger"

    # Looser pinch — the Korean finger-heart has the thumb crossing the SIDE of the index
    # rather than tip-to-tip, and palm-in poses have larger 2D distances due to perspective.
    pinch = _dist(pts[4], pts[8]) < palm * 0.55

    # Thumb+index pinch: OK vs Finger_Heart.
    # Closed_Fist with thumb laid across the palm also gets a tip-to-tip pinch — guard
    # against that by requiring the thumb to visibly stick out from the wrist. Threshold
    # loose enough to cover palm-in poses where the thumb foreshortens in 2D projection.
    thumb_visible = _dist(pts[4], pts[0]) > palm * 0.6
    if pinch and thumb_visible:
        n_other_up = int(mid_up) + int(rng_up) + int(pky_up)
        if n_other_up >= 2:
            return "OK"
        # Allow up to one of mid/ring/pinky to stray up — the pinky in particular tends
        # to escape the fold when making a finger heart.
        if n_other_up <= 1:
            return "Finger_Heart"

    # Rock 🤘: index + pinky up, middle + ring folded, thumb folded.
    if idx_up and pky_up and not mid_up and not rng_up and not thb_up:
        return "Rock"

    # Call_Me 🤙: thumb + pinky out, rest folded.
    if thb_up and pky_up and not idx_up and not mid_up and not rng_up:
        return "Call_Me"

    # Three: index + middle + ring up, pinky folded.
    if idx_up and mid_up and rng_up and not pky_up:
        return "Three"

    # Four: 4 fingers up, thumb tucked.
    if idx_up and mid_up and rng_up and pky_up and not thb_up:
        return "Four"

    return None


def detect_two_hand_combo(left_landmarks, right_landmarks, w, h):
    if left_landmarks is None or right_landmarks is None:
        return None
    l = _norm_pts(left_landmarks, w, h)
    r = _norm_pts(right_landmarks, w, h)
    palm = max(_dist(l[0], l[9]), _dist(r[0], r[9]))
    if palm < 1:
        return None

    # Count extended fingers per hand (mid+ring+pinky). Used to distinguish Heart (curled)
    # from Clap (extended).
    def n_open(pts):
        return (int(_finger_extended(pts, 12, 10, 9)) +
                int(_finger_extended(pts, 16, 14, 13)) +
                int(_finger_extended(pts, 20, 18, 17)))
    l_open, r_open = n_open(l), n_open(r)
    most_closed = l_open <= 1 and r_open <= 1
    any_open = l_open >= 1 or r_open >= 1

    # Heart 💕: index tips close, thumb tips close, indexes above thumbs,
    # and the other fingers folded (otherwise it's just a clap).
    idx_close = _dist(l[8], r[8]) < palm * 0.8
    thb_close = _dist(l[4], r[4]) < palm * 0.8
    idx_above_thb = (l[8][1] + r[8][1]) / 2 < (l[4][1] + r[4][1]) / 2 - palm * 0.15
    if idx_close and thb_close and idx_above_thb and most_closed:
        return "Heart"

    # Clap 👏: hands close together, with at least some fingers extended somewhere
    # (so two fists side-by-side don't fire).
    if _dist(l[9], r[9]) < palm * 1.2 and any_open:
        return "Clap"

    return None


# ---------- Face expression ----------

def _bs(blendshapes, name):
    for b in blendshapes:
        if b.category_name == name:
            return b.score
    return 0.0


def classify_expression(blendshapes):
    if not blendshapes:
        return "Neutral"

    smile = (_bs(blendshapes, "mouthSmileLeft") + _bs(blendshapes, "mouthSmileRight")) / 2
    frown = (_bs(blendshapes, "mouthFrownLeft") + _bs(blendshapes, "mouthFrownRight")) / 2
    jaw_open = _bs(blendshapes, "jawOpen")
    brow_down = (_bs(blendshapes, "browDownLeft") + _bs(blendshapes, "browDownRight")) / 2
    brow_up_in = _bs(blendshapes, "browInnerUp")
    brow_up_out = (_bs(blendshapes, "browOuterUpLeft") + _bs(blendshapes, "browOuterUpRight")) / 2
    eye_wide = (_bs(blendshapes, "eyeWideLeft") + _bs(blendshapes, "eyeWideRight")) / 2
    blink_l = _bs(blendshapes, "eyeBlinkLeft")
    blink_r = _bs(blendshapes, "eyeBlinkRight")
    pucker = _bs(blendshapes, "mouthPucker")
    funnel = _bs(blendshapes, "mouthFunnel")
    tongue = _bs(blendshapes, "tongueOut")
    cheek_puff = _bs(blendshapes, "cheekPuff")
    # MediaPipe's cheekPuff is unreliable; mouth-press fires when lips are sealed against
    # internal pressure, which is a usable proxy for puffed cheeks.
    mouth_press = (_bs(blendshapes, "mouthPressLeft") + _bs(blendshapes, "mouthPressRight")) / 2
    mouth_close = _bs(blendshapes, "mouthClose")
    # Lower lip pull-down is a strong tongue-out signal; tongueOut blendshape itself is
    # under-trained in MediaPipe and often stays near 0.
    mouth_lower_down = (_bs(blendshapes, "mouthLowerDownLeft") + _bs(blendshapes, "mouthLowerDownRight")) / 2

    # Order matters: check the most specific signals first.
    if tongue > 0.15:
        return "Tongue_Out"
    # Fallback: tongue out without the tongueOut blendshape firing — mouth ajar, lower lip
    # pulled, no smile.
    if jaw_open > 0.18 and mouth_lower_down > 0.4 and smile < 0.2:
        return "Tongue_Out"
    if cheek_puff > 0.20:
        return "Cheek_Puff"
    # Fallback when cheekPuff doesn't fire: sealed mouth (lips pressed tight) with no other
    # expression — the lip-press signature of puffed cheeks.
    if (mouth_press > 0.30 or mouth_close > 0.40) and jaw_open < 0.10 \
            and smile < 0.15 and frown < 0.15 and pucker < 0.30:
        return "Cheek_Puff"
    # Wink: one eye notably more closed than the other. Loose because real winks are quick
    # and the model rarely drives blink past 0.5 for the closing eye.
    if abs(blink_l - blink_r) > 0.25 and max(blink_l, blink_r) > 0.35:
        return "Wink"
    if blink_l > 0.55 and blink_r > 0.55:
        return "Blink"
    if jaw_open > 0.45 and (brow_up_in > 0.3 or brow_up_out > 0.3 or eye_wide > 0.3):
        return "Surprised"
    if smile > 0.45 and jaw_open > 0.25:
        return "Laugh"
    # Kiss: lips puckered FORWARD (funnel) — not just pursed shut. A neutral resting face
    # can read pucker around 0.6, so require either very-strong pucker, or moderate pucker
    # combined with mouthFunnel (lips forming a tube — the protruding kiss shape).
    if smile < 0.2 and jaw_open < 0.15:
        if pucker > 0.80:
            return "Kiss"
        if pucker > 0.55 and funnel > 0.35:
            return "Kiss"
    # Angry: brow-down is the strongest signal, but the model often only drives it to ~0.2
    # even for a clearly-angry face, so the threshold is quite generous.
    if smile < 0.2 and pucker < 0.4 and (
        brow_down > 0.18
        or (brow_down > 0.10 and frown > 0.20)
    ):
        return "Angry"
    # Sad — must come BEFORE Eyebrow_Raise, because sad faces also lift the inner brow.
    # Use multiple signals because mouthFrown is unreliable for some faces:
    #   - classic frown (mouth corners down)
    #   - sad inner-brow lift (inner brow rises but OUTER stays neutral — distinct from
    #     surprise/question, which raises both inner and outer)
    #   - lower lip pulled down
    if smile < 0.15 and pucker < 0.3 and (
        frown > 0.25
        or (brow_up_in > 0.35 and brow_up_out < 0.25)
        or mouth_lower_down > 0.30
    ):
        return "Sad"
    if (brow_up_in > 0.45 or brow_up_out > 0.45) and smile < 0.3 and jaw_open < 0.25:
        return "Eyebrow_Raise"
    if smile > 0.4:
        return "Smile"
    return "Neutral"


# ---------- Smoothing ----------

class Smoother:
    """Latch a label once it's been the top vote for N consecutive frames.
    Tracks `last_change` (monotonic time) so we can order channels by recency."""
    def __init__(self, window=4, initial="None"):
        self.window = window
        self.recent = []
        self.current = initial
        self.last_change = time.monotonic()

    def update(self, value):
        self.recent.append(value)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        if len(self.recent) == self.window and len(set(self.recent)) == 1:
            new = self.recent[0]
            if new != self.current:
                self.current = new
                self.last_change = time.monotonic()
        return self.current


# ---------- Main ----------

def main():
    if not GESTURE_MODEL_PATH.exists():
        sys.exit(f"Gesture model missing: {GESTURE_MODEL_PATH}")
    if not FACE_MODEL_PATH.exists():
        sys.exit(
            f"Face model missing: {FACE_MODEL_PATH}\n"
            "Download with:\n"
            "  curl -L -o face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    memes = load_memes()

    gesture_options = mp_vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(GESTURE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        # Lower thresholds so MediaPipe keeps both hands when they get close
        # (e.g. forming a heart) instead of dropping one as a false alarm.
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(gesture_options)

    face_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open webcam. Grant camera permission in System Settings → Privacy & Security → Camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Face expressions are often transient (wink, brief frown). Window=3 so they latch
    # within ~100ms instead of dropping.
    face_smoother = Smoother(window=3, initial="Neutral")
    # Snappier hand response — quick gestures shouldn't get dropped by long debouncing.
    left_smoother = Smoother(window=3, initial="None")
    right_smoother = Smoother(window=3, initial="None")
    # Combo latches faster so quick two-hand poses don't get missed.
    combo_smoother = Smoother(window=2, initial="None")

    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - t_start) * 1000)

        gesture_result = gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Frame was horizontally flipped, so MediaPipe's "Right" hand is on the screen-LEFT.
        per_hand = {"screen_left":  {"label": "None", "landmarks": None},
                    "screen_right": {"label": "None", "landmarks": None}}

        if gesture_result.gestures:
            for i, gestures in enumerate(gesture_result.gestures):
                top = gestures[0] if gestures else None
                built_in = top.category_name if top else "None"
                built_in_score = top.score if top else 0.0
                lms = gesture_result.hand_landmarks[i] if i < len(gesture_result.hand_landmarks) else None
                if lms is None:
                    continue

                # Custom classifier takes precedence since it covers labels MediaPipe doesn't know
                # (OK, Rock, Call_Me, Three, Four, Finger_Heart).
                custom = classify_custom_gesture(lms, w, h)
                if custom is not None:
                    label = custom
                elif built_in_score >= 0.5 and built_in and built_in != "None":
                    label = built_in
                else:
                    label = "None"

                handedness = ""
                if gesture_result.handedness and i < len(gesture_result.handedness):
                    handedness = gesture_result.handedness[i][0].category_name
                slot = "screen_left" if handedness == "Right" else "screen_right"
                if per_hand[slot]["landmarks"] is not None:
                    slot = "screen_right" if slot == "screen_left" else "screen_left"
                per_hand[slot] = {"label": label, "landmarks": lms}

        raw_combo = detect_two_hand_combo(
            per_hand["screen_left"]["landmarks"],
            per_hand["screen_right"]["landmarks"],
            w, h,
        ) or "None"
        combo = combo_smoother.update(raw_combo)

        if per_hand["screen_left"]["landmarks"] is not None:
            draw_hand(frame, per_hand["screen_left"]["landmarks"], color=(60, 200, 80))
        if per_hand["screen_right"]["landmarks"] is not None:
            draw_hand(frame, per_hand["screen_right"]["landmarks"], color=(80, 160, 255))

        left_label = left_smoother.update(per_hand["screen_left"]["label"])
        right_label = right_smoother.update(per_hand["screen_right"]["label"])

        raw_expression = "Neutral"
        if face_result.face_blendshapes:
            raw_expression = classify_expression(face_result.face_blendshapes[0])
            if face_result.face_landmarks:
                draw_face_mesh(frame, face_result.face_landmarks[0])
        expression = face_smoother.update(raw_expression)

        # ---- Pick ONE meme to display, by priority ----
        # 1) Two-hand combo wins outright.
        # 2) Otherwise, among {left hand, right hand, face} pick the active channel
        #    whose smoother updated most recently. "Active" means non-default.
        candidates = []
        if combo in TWO_HAND_COMBOS:
            candidates.append((combo_smoother.last_change,
                               TWO_HAND_COMBOS[combo], combo))
        else:
            if left_label != "None":
                candidates.append((left_smoother.last_change,
                                   GESTURE_TO_MEME.get(left_label, "idle"),
                                   left_label))
            if right_label != "None":
                candidates.append((right_smoother.last_change,
                                   GESTURE_TO_MEME.get(right_label, "idle"),
                                   right_label))
            if expression != "Neutral":
                candidates.append((face_smoother.last_change,
                                   EXPRESSION_TO_MEME.get(expression, "neutral"),
                                   expression))

        if candidates:
            _, active_meme_name, active_event = max(candidates, key=lambda c: c[0])
        else:
            active_meme_name, active_event = "idle", ""

        # ---- Build display: big meme + small webcam picture-in-picture ----
        display_w, display_h = 1280, 720
        display = render_meme_panel(memes[active_meme_name], (display_w, display_h))

        # Small webcam in the top-right corner, with a clean light frame.
        mini_w = 280
        mini_h = int(mini_w * h / w) if w else 158
        mini = cv2.resize(frame, (mini_w, mini_h), interpolation=cv2.INTER_AREA)
        margin = 22
        my = margin
        mx = display_w - mini_w - margin
        border = 4
        cv2.rectangle(display,
                      (mx - border, my - border),
                      (mx + mini_w + border, my + mini_h + border),
                      (255, 255, 255), -1)
        display[my:my + mini_h, mx:mx + mini_w] = mini
        cv2.rectangle(display,
                      (mx - border, my - border),
                      (mx + mini_w + border, my + mini_h + border),
                      (210, 210, 215), 1, cv2.LINE_AA)

        # Subtle event label centered at the bottom — only when something is happening.
        if active_event:
            text = active_event.replace("_", " ")
            fscale = 1.1
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fscale, 2)
            tx = (display_w - tw) // 2
            ty = display_h - 36
            cv2.putText(display, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                        fscale, (255, 255, 255), 6, cv2.LINE_AA)
            cv2.putText(display, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                        fscale, (60, 60, 70), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    gesture_recognizer.close()
    face_landmarker.close()


if __name__ == "__main__":
    main()
