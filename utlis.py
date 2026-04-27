import cv2

def put_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(frame, text, position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)


def draw_progress_bar(frame, value, max_value, x, y, w, h, color):
    progress = int((value / max_value) * w)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.rectangle(frame, (x, y), (x + progress, y + h), color, -1)
