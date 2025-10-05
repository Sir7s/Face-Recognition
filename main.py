# --- YOLO + OpenCV + Serial: Authorized/Unauthorized -> Arduino Uno ---

import cv2
import time
import math
import serial
from ultralytics import YOLO

# ========= USER SETTINGS =========
MODEL_PATH = r"/Users/jingyu/Documents/NIC_CV/F25-NuclearIC/Loading_Unloading_Subproblem/Loading_Unloading_Training_Files/train6/weights/best.pt"
TARGET_CLASS = "your_class_name_here"   # <-- change to the class *name* used in your model (exact spelling)
CONF_THRESH  = 0.40                     # detection confidence threshold
PORT         = "COM3"                   # Windows: COM3/COM4... | macOS/Linux: /dev/ttyACM0 or /dev/tty.usbmodemXXXX
BAUD         = 9600                     # must match Arduino
USE_ROI      = False                    # set True to enable a region-of-interest gate
ROI_FRACTION = (0.00, 1.00, 0.00, 1.00) # (x1, x2, y1, y2) as fractions of frame size if USE_ROI=True
SEND_ON_CHANGE_ONLY = True              # send 'U'/'A' only when state changes
RESEND_EVERY_SEC     = 3.0              # if not on-change, resend at this interval
# =================================

def open_camera():
    cap = cv2.VideoCapture(0)  # mac users: try cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) if needed
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera 0. Grant camera access and try again.")
    return cap

def open_serial():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Uno auto-resets on open; give it a moment
    return ser

def box_center(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return ( (x1 + x2) // 2, (y1 + y2) // 2 )

def point_in_roi(cx, cy, frame_w, frame_h, roi_frac):
    x1f, x2f, y1f, y2f = roi_frac
    x1, x2 = int(x1f * frame_w), int(x2f * frame_w)
    y1, y2 = int(y1f * frame_h), int(y2f * frame_h)
    return (x1 <= cx <= x2) and (y1 <= cy <= y2), (x1, y1, x2, y2)

def get_target_id(names_dict, target_name):
    # names_dict: {id: 'classname', ...}
    for cid, cname in names_dict.items():
        if cname == target_name:
            return int(cid)
    return None

def decide_status(result, target_class, conf_thresh, frame_shape):
    """
    Rule: authorized if at least one box of TARGET_CLASS (above conf) is present
    (and inside ROI if enabled). Otherwise unauthorized.
    """
    h, w = frame_shape[:2]
    names = result.names
    target_id = get_target_id(names, target_class)
    if target_id is None:
        # Class not found in label map — treat as unauthorized until you set the right name.
        return "unauthorized", None

    count_in_scope = 0
    roi_px = None
    for box in result.boxes:
        if int(box.cls[0]) != target_id:
            continue
        if float(box.conf[0]) < conf_thresh:
            continue

        cx, cy = box_center(box)
        if USE_ROI:
            inside, roi_px = point_in_roi(cx, cy, w, h, ROI_FRACTION)
            if inside:
                count_in_scope += 1
        else:
            count_in_scope += 1

    return ("authorized" if count_in_scope > 0 else "unauthorized"), roi_px

def main():
    # Load model, serial, and camera
    yolo = YOLO(MODEL_PATH)
    ser = open_serial()
    cap = open_camera()

    prev_status = None
    last_send   = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Run inference on the frame
            # result is a single Results object
            result = yolo(frame)[0]

            # Draw detections
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = result.names.get(cls_id, str(cls_id))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Decide authorized/unauthorized
            status, roi_px = decide_status(result, TARGET_CLASS, CONF_THRESH, frame.shape)

            # Draw ROI if enabled
            if USE_ROI and roi_px is not None:
                x1, y1, x2, y2 = roi_px
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Put big status label
            label = "AUTHORIZED" if status == "authorized" else "UNAUTHORIZED"
            color = (0, 200, 0) if status == "authorized" else (0, 0, 255)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            # Send to Arduino if needed
            now = time.time()
            should_send = False
            if SEND_ON_CHANGE_ONLY:
                if status != prev_status:
                    should_send = True
            else:
                if (now - last_send) >= RESEND_EVERY_SEC:
                    should_send = True

            if should_send:
                ser.write(b'A' if status == "authorized" else b'U')
                prev_status = status
                last_send = now

            # Show frame
            cv2.imshow("YOLO + Arduino", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Make sure we turn things off before exiting
        try:
            ser.write(b'A')  # silence/authorized on exit
        except Exception:
            pass
        ser.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
