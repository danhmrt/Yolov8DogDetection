import time
import cv2
import requests
from ultralytics import YOLO

# ---------------------------
# Relay configuration
# ---------------------------
RELAY_BASE = "http://10.0.0.5:5090"
RELAY_ON_URL = f"{RELAY_BASE}/relay/on"
RELAY_OFF_URL = f"{RELAY_BASE}/relay/off"
RELAY_STATUS_URL = f"{RELAY_BASE}/status"
HTTP_TIMEOUT_S = 1.5

# ---------------------------
# Detection / behavior tuning
# ---------------------------
MODEL_WEIGHTS = "yolov8n.pt"   # small/fast. Try yolov8s.pt for better accuracy
CONF_THRESH = 0.35            # detection confidence threshold
DOG_HOLD_SECONDS = 2.0        # keep relay ON for this long after last dog seen
MIN_RELAY_TOGGLE_INTERVAL = 0.5  # avoid spamming relay

# ---------------------------
# Camera configuration
# ---------------------------
CAMERA_INDEX = 0              # 0 is usually the first USB camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def relay_get_status():
    try:
        r = requests.get(RELAY_STATUS_URL, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        return r.text.strip()
    except Exception:
        return None

def relay_set(on: bool):
    url = RELAY_ON_URL if on else RELAY_OFF_URL
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[relay] ERROR calling {url}: {e}")
        return False

def main():
    print("[info] Loading YOLOv8 model...")
    model = YOLO(MODEL_WEIGHTS)

    # Find the numeric id for "dog" in the model's names map
    # model.names is typically like {0:'person', 1:'bicycle', ..., 16:'dog', ...}
    dog_class_ids = [cid for cid, name in model.names.items() if name == "dog"]
    if not dog_class_ids:
        raise RuntimeError("Could not find 'dog' class in model.names. Are you using a COCO model?")
    DOG_ID = dog_class_ids[0]
    print(f"[info] Dog class id: {DOG_ID}")

    print("[info] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    relay_state = False
    last_dog_seen_t = 0.0
    last_toggle_t = 0.0

    print("[info] Starting loop. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[camera] Failed to read frame")
            time.sleep(0.1)
            continue

        # Run inference
        results = model.predict(frame, conf=CONF_THRESH, verbose=False)

        dog_seen = False

        # results is a list; we have one image -> results[0]
        r0 = results[0]
        if r0.boxes is not None and len(r0.boxes) > 0:
            # boxes.cls is a tensor of class ids, boxes.conf is conf, boxes.xyxy are boxes
            for box in r0.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id == DOG_ID and conf >= CONF_THRESH:
                    dog_seen = True

        now = time.time()
        if dog_seen:
            last_dog_seen_t = now

        # Determine desired relay state
        should_be_on = (now - last_dog_seen_t) <= DOG_HOLD_SECONDS

        # Toggle relay only if needed + rate-limited
        if should_be_on != relay_state and (now - last_toggle_t) >= MIN_RELAY_TOGGLE_INTERVAL:
            if relay_set(should_be_on):
                relay_state = should_be_on
                last_toggle_t = now
                print(f"[relay] set to {'ON' if relay_state else 'OFF'}")

        # Draw overlay (using Ultralytics built-in plotting)
        annotated = r0.plot()
        status_text = f"dog_seen={dog_seen} relay={'ON' if relay_state else 'OFF'}"
        cv2.putText(annotated, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Dog Relay", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print("[info] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

    # Best effort to turn off relay on exit
    relay_set(False)

if __name__ == "__main__":
    main()
