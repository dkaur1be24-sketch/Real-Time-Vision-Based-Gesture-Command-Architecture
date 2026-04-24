import cv2
import dlib
import face_recognition
import mediapipe as mp
import webbrowser
import pyautogui
import time
from scipy.spatial import distance

# ─────────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────────
PREDICTOR_PATH        = r"C:\Users\Diljeet\OneDrive\Desktop\edp_project\shape_predictor_68_face_landmarks.dat"
YOUTUBE_URL           = "https://www.youtube.com/watch?v=xzUVPN68Ym4"

# Smile threshold  (MAR = mouth width / mouth height)
# Lower MAR = wider open mouth. Tune by watching the MAR readout on screen.
MAR_THRESHOLD         = 2.8

# Debounce frames
SMILE_FRAMES_REQUIRED = 50     # ~0.67s at 30fps

# Volume cooldown (seconds between consecutive volume actions)
VOLUME_COOLDOWN       = 1.2

# V-sign (exit) — must hold for this many frames to avoid accidents
VSIGN_FRAMES_REQUIRED = 25     # ~0.83s

# Face recognition
MATCH_TOLERANCE       = 0.5
ENCODING_INTERVAL     = 10     # re-run face_recognition every N frames

# ID system
ACTIVE_USER_ID        = "User #1"   # label shown on the locked face's bounding box

# ─────────────────────────────────────────────────────────────
#  MOUTH LANDMARK INDEXES  (dlib 68-point model)
# ─────────────────────────────────────────────────────────────
MOUTH_LEFT   = 48
MOUTH_RIGHT  = 54
MOUTH_TOP    = 51
MOUTH_BOTTOM = 57

# ─────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────
print("[INFO] Loading models...")

# dlib — face detection + landmarks (for smile)
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# MediaPipe — hand tracking
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6
)

print("[INFO] Models loaded.")

# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def mouth_aspect_ratio(landmarks):
    left   = (landmarks.part(MOUTH_LEFT).x,   landmarks.part(MOUTH_LEFT).y)
    right  = (landmarks.part(MOUTH_RIGHT).x,  landmarks.part(MOUTH_RIGHT).y)
    top    = (landmarks.part(MOUTH_TOP).x,     landmarks.part(MOUTH_TOP).y)
    bottom = (landmarks.part(MOUTH_BOTTOM).x,  landmarks.part(MOUTH_BOTTOM).y)
    width  = distance.euclidean(left, right)
    height = distance.euclidean(top, bottom)
    return (width / height) if height != 0 else 0


def count_fingers(hand_landmarks, handedness_label):
    """
    Returns (finger_count, fingers_up_list).

    fingers_up_list is a list of 5 booleans:
      [thumb, index, middle, ring, pinky]

    Tip landmarks : thumb=4, index=8, middle=12, ring=16, pinky=20
    PIP landmarks : thumb=3, index=6, middle=10, ring=14, pinky=18
    """
    lm = hand_landmarks.landmark

    # Fingers: tip y < pip y  → finger is up (y increases downward in image)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = [lm[t].y < lm[p].y for t, p in zip(tips, pips)]

    # Thumb: compare x-axis (direction depends on which hand)
    if handedness_label == "Right":
        thumb_up = lm[4].x < lm[3].x
    else:
        thumb_up = lm[4].x > lm[3].x

    fingers = [thumb_up] + fingers   # [thumb, index, middle, ring, pinky]
    return sum(fingers), fingers


def is_open_palm(finger_count, fingers):
    """All 5 fingers extended."""
    return finger_count == 5


def is_closed_fist(finger_count, fingers):
    """All fingers folded (thumb may or may not be tucked — allow thumb variation)."""
    # index, middle, ring, pinky all closed is enough
    return not any(fingers[1:])


def is_v_sign(finger_count, fingers):
    """Index and middle up, ring and pinky down, thumb down."""
    thumb, index, middle, ring, pinky = fingers
    return (index and middle) and (not ring) and (not pinky) and (not thumb)


def draw_progress_bar(frame, value, maximum, x, y, w, h, color):
    progress = int((min(value, maximum) / maximum) * w)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    if progress > 0:
        cv2.rectangle(frame, (x, y), (x + progress, y + h), color, -1)


def put_label(frame, text, pos, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# ─────────────────────────────────────────────────────────────
#  STATE VARIABLES
# ─────────────────────────────────────────────────────────────
locked_encoding    = None
last_encoding      = None

smile_counter      = 0
youtube_opened     = False

vsign_counter      = 0

last_volume_time   = 0

frame_count        = 0

# ─────────────────────────────────────────────────────────────
#  START CAMERA
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

print("[INFO] Camera started.")
print("─────────────────────────────────────────────────")
print("  GESTURES:")
print("  😊 Smile (hold)       → Play YouTube")
print("  ✋ Open palm           → Volume UP")
print("  ✊ Closed fist         → Volume DOWN")
print("  ✌  V-sign (hold)       → Close tab + Exit")
print("─────────────────────────────────────────────────")
print("  Press R to reset face lock | Press Q to quit")

# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)   # mirror for natural feel

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w = frame.shape[:2]

    # ══════════════════════════════════════════════════════════
    #  SECTION A — FACE RECOGNITION + SMILE (dlib)
    #  Every face gets a bounding box + label.
    #  Only the locked face (User #1) triggers gestures.
    # ══════════════════════════════════════════════════════════
    faces = detector(gray)

    active_face      = None   # the dlib rect of the locked user (if visible)
    active_landmarks = None

    if len(faces) == 0:
        smile_counter = 0
        put_label(frame, "No face detected", (20, 40), (0, 0, 255))

    else:
        # ── Throttled bulk encoding (all faces at once) ────────
        all_encodings = []
        if frame_count % ENCODING_INTERVAL == 0:
            all_encodings = face_recognition.face_encodings(rgb)
            # cache only if we got one per face
            if len(all_encodings) == len(faces):
                last_encoding = all_encodings[0]   # cache first face's encoding
        else:
            # between refresh frames: use cached encoding for face[0] only
            if last_encoding is not None:
                all_encodings = [last_encoding] + [None] * (len(faces) - 1)

        # ── Lock first person seen ─────────────────────────────
        if locked_encoding is None and len(all_encodings) > 0 and all_encodings[0] is not None:
            locked_encoding = all_encodings[0]
            print(f"[INFO] {ACTIVE_USER_ID} locked.")

        # ── Draw a box around EVERY face with an ID label ──────
        for idx, face in enumerate(faces):
            x1, y1 = face.left(),  face.top()
            x2, y2 = face.right(), face.bottom()

            enc = all_encodings[idx] if idx < len(all_encodings) else None

            # Determine if this face matches the locked user
            is_active = False
            if locked_encoding is not None and enc is not None:
                is_active = face_recognition.compare_faces(
                    [locked_encoding], enc, tolerance=MATCH_TOLERANCE
                )[0]
            elif locked_encoding is None:
                is_active = False

            if is_active:
                # Green box + "User #1 (Active)"
                box_color  = (0, 220, 80)
                label_text = f"{ACTIVE_USER_ID}  [ACTIVE]"
                active_face = face
            else:
                # Red box + "Stranger" or numbered unknown
                box_color  = (0, 0, 220)
                stranger_n = idx + 1 if locked_encoding is not None else idx
                label_text = f"Stranger #{stranger_n}  [ignored]"
                if locked_encoding is None:
                    label_text = "Identifying..."

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label background pill above the box
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            pill_y1 = max(y1 - th - 10, 0)
            pill_y2 = max(y1 - 2, th + 4)
            cv2.rectangle(frame, (x1, pill_y1), (x1 + tw + 8, pill_y2), box_color, -1)
            cv2.putText(frame, label_text,
                        (x1 + 4, pill_y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        # ── Only process smile for the locked (active) face ────
        if active_face is None:
            # Locked user not visible this frame
            smile_counter = 0
            put_label(frame, f"{ACTIVE_USER_ID} not in frame", (20, 40), (0, 180, 255))
        else:
            put_label(frame, f"{ACTIVE_USER_ID} — controlling", (20, 40), (0, 220, 80))

            active_landmarks = predictor(gray, active_face)

            # Draw face landmarks only on active user
            for i in range(68):
                cv2.circle(frame,
                           (active_landmarks.part(i).x, active_landmarks.part(i).y),
                           1, (0, 220, 100), -1)

            mar = mouth_aspect_ratio(active_landmarks)

            # ── GESTURE 1: SMILE → Play YouTube ───────────
            if mar > MAR_THRESHOLD:
                smile_counter += 1
                draw_progress_bar(frame, smile_counter,
                                  SMILE_FRAMES_REQUIRED,
                                  20, 62, 220, 13, (0, 255, 120))
                put_label(frame,
                          f"Smiling... ({smile_counter}/{SMILE_FRAMES_REQUIRED})",
                          (20, 58), (0, 255, 120), 0.5)

                if smile_counter >= SMILE_FRAMES_REQUIRED and not youtube_opened:
                    print("[ACTION] Smile detected — opening YouTube!")
                    webbrowser.open(YOUTUBE_URL)
                    youtube_opened = True
            else:
                smile_counter  = 0
                youtube_opened = False

            # MAR debug readout
            put_label(frame,
                      f"MAR: {mar:.2f}  (threshold > {MAR_THRESHOLD})",
                      (20, h - 15), (255, 230, 0), 0.45, 1)

    # ══════════════════════════════════════════════════════════
    #  SECTION B — HAND GESTURES (MediaPipe)
    # ══════════════════════════════════════════════════════════
    hand_result = hands_model.process(rgb)

    gesture_label = ""
    now = time.time()

    if hand_result.multi_hand_landmarks:
        hand_lm       = hand_result.multi_hand_landmarks[0]
        handedness    = hand_result.multi_handedness[0].classification[0].label

        # Draw hand skeleton
        mp_drawing.draw_landmarks(
            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 200, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(200, 80, 255), thickness=2)
        )

        finger_count, fingers = count_fingers(hand_lm, handedness)

        # ── GESTURE 2: OPEN PALM → Volume Up ──────────────────
        if is_open_palm(finger_count, fingers):
            gesture_label = "Open palm — Volume UP"
            draw_progress_bar(frame, 1, 1, 20, 97, 220, 13, (0, 210, 255))
            put_label(frame, gesture_label, (20, 93), (0, 210, 255), 0.55)

            if now - last_volume_time > VOLUME_COOLDOWN:
                print("[ACTION] Open palm — Volume UP")
                pyautogui.press('volumeup')
                pyautogui.press('volumeup')
                last_volume_time = now

            vsign_counter = 0   # reset exit counter

        # ── GESTURE 3: CLOSED FIST → Volume Down ──────────────
        elif is_closed_fist(finger_count, fingers):
            gesture_label = "Closed fist — Volume DOWN"
            draw_progress_bar(frame, 1, 1, 20, 127, 220, 13, (255, 140, 0))
            put_label(frame, gesture_label, (20, 123), (255, 140, 0), 0.55)

            if now - last_volume_time > VOLUME_COOLDOWN:
                print("[ACTION] Closed fist — Volume DOWN")
                pyautogui.press('volumedown')
                pyautogui.press('volumedown')
                last_volume_time = now

            vsign_counter = 0

        # ── GESTURE 4: V-SIGN → Close tab + Exit ──────────────
        elif is_v_sign(finger_count, fingers):
            vsign_counter += 1
            progress_color = (80, 80, 255)
            draw_progress_bar(frame, vsign_counter,
                              VSIGN_FRAMES_REQUIRED,
                              20, 157, 220, 13, progress_color)
            put_label(frame,
                      f"V-sign — Hold to EXIT ({vsign_counter}/{VSIGN_FRAMES_REQUIRED})",
                      (20, 153), (120, 120, 255), 0.5)

            if vsign_counter >= VSIGN_FRAMES_REQUIRED:
                print("[ACTION] V-sign confirmed — closing tab and exiting!")
                pyautogui.hotkey('ctrl', 'w')
                time.sleep(0.3)
                cap.release()
                cv2.destroyAllWindows()
                hands_model.close()
                exit()

        else:
            vsign_counter = 0   # reset if hand is present but no matching gesture

        # Finger count debug
        put_label(frame,
                  f"Fingers: {finger_count}  Hand: {handedness}",
                  (20, h - 35), (200, 200, 255), 0.45, 1)

    else:
        vsign_counter = 0   # reset exit counter when no hand visible
        put_label(frame, "No hand detected", (w - 200, 40), (120, 120, 120), 0.5, 1)

    # ── HUD header ──────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 22), (0, 0, 0), -1)
    put_label(frame,
              "SMILE=Play  |  Palm=Vol+  |  Fist=Vol-  |  V-sign=Exit  |  R=Reset  Q=Quit",
              (6, 16), (180, 180, 180), 0.38, 1)

    cv2.imshow("Gesture Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Quit by user.")
        break
    elif key == ord('r'):
        locked_encoding = None
        last_encoding   = None
        smile_counter   = 0
        vsign_counter   = 0
        youtube_opened  = False
        frame_count     = 0
        print(f"[INFO] Face lock reset. Next face will become {ACTIVE_USER_ID}.")

# ─────────────────────────────────────────────────────────────
#  CLEANUP
# ─────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands_model.close()
print("[INFO] Camera released. Goodbye.")
