import time

def run_inference(action_name):
    start = time.time()

    print(f"[ACTION TRIGGERED]: {action_name}")

    end = time.time()
    inference_time = end - start

    print(f"Inference Time: {inference_time:.4f} sec")

    return inference_time


def log_fps(fps):
    print(f"[FPS]: {fps}")
