import cv2
import face_recognition
import mediapipe as mp
import webbrowser
import pyautogui
import time

# ─────────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────────
YOUTUBE_URL           = "https://www.youtube.com/watch?v=xzUVPN68Ym4"

# Smile threshold using MediaPipe Face Mesh
# Mouth Aspect Ratio (MAR) = mouth width / mouth height
# Higher MAR = wider smile (mouth open horizontally more than vertically)
MAR_THRESHOLD         = 1.8

# Debounce frames
SMILE_FRAMES_REQUIRED = 50    # ~0.67s at 30fps

# Volume cooldown (seconds between consecutive volume actions)
VOLUME_COOLDOWN       = 1.2

# V-sign (exit) — must hold for this many frames to avoid accidents
VSIGN_FRAMES_REQUIRED = 25     # ~0.83s

# Face recognition
MATCH_TOLERANCE       = 0.5
ENCODING_INTERVAL     = 10     # re-run face_recognition every N frames

# ID system
ACTIVE_USER_ID        = "User #1"

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE FACE MESH — MOUTH LANDMARK INDEXES
#  Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ─────────────────────────────────────────────────────────────
# Outer lip corners (left/right from camera perspective)
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
# Outer lip top & bottom (vertical opening)
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14

# ─────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────
print("[INFO] Loading models...")

# MediaPipe — Hand tracking
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6
)

# MediaPipe — Face Mesh (replaces dlib for smile detection)
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=4,            # detect up to 4 faces
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("[INFO] Models loaded.")

# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def mouth_aspect_ratio(landmarks, img_w, img_h):
    """
    Compute MAR using MediaPipe Face Mesh normalized landmarks.
    Converts normalized [0,1] coords to pixel coords first.
    """
    def px(lm_idx):
        lm = landmarks.landmark[lm_idx]
        return int(lm.x * img_w), int(lm.y * img_h)

    lx, ly  = px(MOUTH_LEFT)
    rx, ry  = px(MOUTH_RIGHT)
    tx, ty  = px(MOUTH_TOP)
    bx, by  = px(MOUTH_BOTTOM)

    width  = ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5
    height = ((bx - tx) ** 2 + (by - ty) ** 2) ** 0.5

    return (width / height) if height != 0 else 0


def count_fingers(hand_landmarks, handedness_label):
    """
    Returns (finger_count, fingers_up_list).
    fingers_up_list = [thumb, index, middle, ring, pinky]
    """
    lm   = hand_landmarks.landmark
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = [lm[t].y < lm[p].y for t, p in zip(tips, pips)]

    if handedness_label == "Right":
        thumb_up = lm[4].x < lm[3].x
    else:
        thumb_up = lm[4].x > lm[3].x

    fingers = [thumb_up] + fingers   # [thumb, index, middle, ring, pinky]
    return sum(fingers), fingers


def is_open_palm(finger_count, fingers):
    return finger_count == 5


def is_closed_fist(finger_count, fingers):
    return not any(fingers[1:])


def is_v_sign(finger_count, fingers):
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
locked_encoding  = None
last_encoding    = None

smile_counter    = 0
youtube_opened   = False

vsign_counter    = 0
last_volume_time = 0
frame_count      = 0

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

    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # ══════════════════════════════════════════════════════════
    #  SECTION A — FACE RECOGNITION (face_recognition lib)
    #              + SMILE DETECTION (MediaPipe Face Mesh)
    # ══════════════════════════════════════════════════════════

    # ── A1: face_recognition for identity locking ─────────────
    # Downscale for speed (face_recognition is slow at full res)
    small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
    face_locations = face_recognition.face_locations(small)

    all_encodings = []
    if face_locations:
        if frame_count % ENCODING_INTERVAL == 0:
            all_encodings = face_recognition.face_encodings(small, face_locations)
            if all_encodings:
                last_encoding = all_encodings[0]
        else:
            if last_encoding is not None:
                all_encodings = [last_encoding] + [None] * (len(face_locations) - 1)

    # Lock first person seen
    if locked_encoding is None and all_encodings and all_encodings[0] is not None:
        locked_encoding = all_encodings[0]
        print(f"[INFO] {ACTIVE_USER_ID} locked.")

    # ── A2: MediaPipe Face Mesh for smile + bounding boxes ────
    mesh_result = face_mesh.process(rgb)

    active_mar    = None
    active_box_drawn = False

    if mesh_result.multi_face_landmarks:
        num_mesh_faces  = len(mesh_result.multi_face_landmarks)
        num_recog_faces = len(face_locations)

        for idx, face_lm in enumerate(mesh_result.multi_face_landmarks):
            # ── Derive bounding box from face mesh ─────────────
            xs = [lm.x for lm in face_lm.landmark]
            ys = [lm.y for lm in face_lm.landmark]
            x1 = int(min(xs) * w)
            y1 = int(min(ys) * h)
            x2 = int(max(xs) * w)
            y2 = int(max(ys) * h)

            # Clamp to frame
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)

            # ── Match mesh face to recognized face by position ──
            # Mesh faces & face_recognition faces are ordered similarly.
            # We match by index (both are top-to-bottom sorted).
            is_active = False
            if locked_encoding is not None and idx < len(all_encodings):
                enc = all_encodings[idx]
                if enc is not None:
                    is_active = face_recognition.compare_faces(
                        [locked_encoding], enc, tolerance=MATCH_TOLERANCE
                    )[0]
            elif locked_encoding is None:
                is_active = False

            if is_active:
                box_color  = (0, 220, 80)
                label_text = f"{ACTIVE_USER_ID}  [ACTIVE]"
            else:
                box_color  = (0, 0, 220)
                stranger_n = idx + 1
                label_text = (f"Stranger #{stranger_n}  [ignored]"
                              if locked_encoding is not None else "Identifying...")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label background pill
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            pill_y1 = max(y1 - th - 10, 0)
            pill_y2 = max(y1 - 2, th + 4)
            cv2.rectangle(frame, (x1, pill_y1), (x1 + tw + 8, pill_y2), box_color, -1)
            cv2.putText(frame, label_text,
                        (x1 + 4, pill_y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

            # ── Draw minimal face mesh dots on active face only ─
            if is_active:
                # Draw only the lip contour landmarks for a clean look
                lip_ids = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                           375, 321, 405, 314, 17, 84, 181, 91, 146]
                for lid in lip_ids:
                    lm = face_lm.landmark[lid]
                    px_x = int(lm.x * w)
                    px_y = int(lm.y * h)
                    cv2.circle(frame, (px_x, px_y), 1, (0, 220, 100), -1)

                # ── GESTURE 1: SMILE → Play YouTube ───────────
                mar = mouth_aspect_ratio(face_lm, w, h)
                active_mar = mar

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

                active_box_drawn = True

        # HUD — active user status
        if active_box_drawn:
            put_label(frame, f"{ACTIVE_USER_ID} — controlling", (20, 40), (0, 220, 80))
        else:
            smile_counter = 0
            put_label(frame, f"{ACTIVE_USER_ID} not in frame", (20, 40), (0, 180, 255))

        # MAR debug readout (only if active face was found)
        if active_mar is not None:
            put_label(frame,
                      f"MAR: {active_mar:.2f}  (threshold > {MAR_THRESHOLD})",
                      (20, h - 15), (255, 230, 0), 0.45, 1)

    else:
        smile_counter = 0
        put_label(frame, "No face detected", (20, 40), (0, 0, 255))

    # ══════════════════════════════════════════════════════════
    #  SECTION B — HAND GESTURES (MediaPipe Hands)
    # ══════════════════════════════════════════════════════════
    hand_result = hands_model.process(rgb)

    now = time.time()

    if hand_result.multi_hand_landmarks:
        hand_lm    = hand_result.multi_hand_landmarks[0]
        handedness = hand_result.multi_handedness[0].classification[0].label

        mp_drawing.draw_landmarks(
            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 200, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(200, 80, 255), thickness=2)
        )

        finger_count, fingers = count_fingers(hand_lm, handedness)

        # ── GESTURE 2: OPEN PALM → Volume Up ──────────────────
        if is_open_palm(finger_count, fingers):
            draw_progress_bar(frame, 1, 1, 20, 97, 220, 13, (0, 210, 255))
            put_label(frame, "Open palm — Volume UP", (20, 93), (0, 210, 255), 0.55)

            if now - last_volume_time > VOLUME_COOLDOWN:
                print("[ACTION] Open palm — Volume UP")
                pyautogui.press('volumeup')
                pyautogui.press('volumeup')
                last_volume_time = now

            vsign_counter = 0

        # ── GESTURE 3: CLOSED FIST → Volume Down ──────────────
        elif is_closed_fist(finger_count, fingers):
            draw_progress_bar(frame, 1, 1, 20, 127, 220, 13, (255, 140, 0))
            put_label(frame, "Closed fist — Volume DOWN", (20, 123), (255, 140, 0), 0.55)

            if now - last_volume_time > VOLUME_COOLDOWN:
                print("[ACTION] Closed fist — Volume DOWN")
                pyautogui.press('volumedown')
                pyautogui.press('volumedown')
                last_volume_time = now

            vsign_counter = 0

        # ── GESTURE 4: V-SIGN → Close tab + Exit ──────────────
        elif is_v_sign(finger_count, fingers):
            vsign_counter += 1
            draw_progress_bar(frame, vsign_counter,
                              VSIGN_FRAMES_REQUIRED,
                              20, 157, 220, 13, (80, 80, 255))
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
                face_mesh.close()
                exit()

        else:
            vsign_counter = 0

        put_label(frame,
                  f"Fingers: {finger_count}  Hand: {handedness}",
                  (20, h - 35), (200, 200, 255), 0.45, 1)

    else:
        vsign_counter = 0
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
face_mesh.close()
print("[INFO] Camera released. Goodbye.")
