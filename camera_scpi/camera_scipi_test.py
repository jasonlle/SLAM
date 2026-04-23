import os
import csv
import time
from datetime import datetime

import cv2
import pyvisa
import numpy as np
import joblib

import support_functions as sf

# =========================
# User settings
# =========================
ADDR = 'TCPIP0::192.168.1.10::inst0::INSTR'
VNA_MODEL = 'E5071C'

LOW_FREQ = 902e6
UPPER_FREQ = 928e6
S_PARAMS_LIST = ['21']
MANUAL_STEP_SIZE = True
STEP_FREQ = 1e6
RESET_TO_PRESET = False

# Camera index 0 = default webcam
CAMERA_INDEX = 0

# Box / ROI drawn on the camera image
ROI_X1, ROI_Y1 = 100, 100
ROI_X2, ROI_Y2 = 400, 400

# Motion / occupancy tuning
MIN_CONTOUR_AREA = 3000
OCCUPIED_CONFIRM_FRAMES = 4
EMPTY_CONFIRM_FRAMES = 4
THRESHOLD_VALUE = 30

# Capture timing
CAPTURE_INTERVAL_SEC = 1.0
CAPTURE_ONLY_ON_STATE_CHANGE = False

# Save folders
BASE_SAVE_DIR = r"C:\Users\Student\Documents\6G_VIP_SLAM_MAIN\dataset"
EMPTY_DIR = os.path.join(BASE_SAVE_DIR, "empty")
OCCUPIED_DIR = os.path.join(BASE_SAVE_DIR, "occupied")
LOG_CSV = os.path.join(BASE_SAVE_DIR, "capture_log.csv")
MODEL_PATH = os.path.join(BASE_SAVE_DIR, "room_presence_rf.joblib")
ENABLE_RF_INFERENCE = True


class BoxOccupancyMonitor:
    def __init__(self):
        self.static_back = None
        self.occupied_streak = 0
        self.empty_streak = 0
        self.verified_inside = 0

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.static_back is None:
            self.static_back = gray
            return {
                "inside_raw": 0,
                "inside_verified": self.verified_inside,
                "largest_area": 0,
                "gray": gray,
                "diff_frame": gray,
                "thresh_frame": gray,
            }

        diff_frame = cv2.absdiff(self.static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cnts, _ = cv2.findContours(
            thresh_frame.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        inside_raw = 0
        largest_area = 0

        for contour in cnts:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            largest_area = max(largest_area, area)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            overlaps_roi = (
                x < ROI_X2 and x + w > ROI_X1 and
                y < ROI_Y2 and y + h > ROI_Y1
            )
            if overlaps_roi:
                inside_raw = 1

        # Debounce so the label is more stable for dataset collection.
        if inside_raw:
            self.occupied_streak += 1
            self.empty_streak = 0
            if self.occupied_streak >= OCCUPIED_CONFIRM_FRAMES:
                self.verified_inside = 1
        else:
            self.empty_streak += 1
            self.occupied_streak = 0
            if self.empty_streak >= EMPTY_CONFIRM_FRAMES:
                self.verified_inside = 0

        return {
            "inside_raw": inside_raw,
            "inside_verified": self.verified_inside,
            "largest_area": largest_area,
            "gray": gray,
            "diff_frame": diff_frame,
            "thresh_frame": thresh_frame,
        }


def ensure_dirs():
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    os.makedirs(EMPTY_DIR, exist_ok=True)
    os.makedirs(OCCUPIED_DIR, exist_ok=True)

    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "camera_label_int",
                "camera_label_text",
                "largest_motion_area",
                "saved_csv",
                "rf_pred_int",
                "rf_pred_text",
                "rf_prob_empty",
                "rf_prob_occupied",
            ])


def connect_and_configure_vna():
    print('\nAttempting to connect to: ' + sf.get_yellow(ADDR))

    rm = pyvisa.ResourceManager()
    vna = rm.open_resource(ADDR)

    vna.write_termination = "\n"
    vna.read_termination = None
    vna.timeout = 60000

    resp = sf.query_text(vna, "*IDN?")
    if VNA_MODEL in resp:
        sf.print_green('Successfully connected to Keysight VNA!\n')
    else:
        raise RuntimeError(f"Unexpected instrument: {resp}")

    if RESET_TO_PRESET:
        sf.toggle_preset(vna)
        vna.write("*CLS")

    vna.write(f"CALC1:PAR:COUN {len(S_PARAMS_LIST)}")
    vna.write(f"SENS1:FREQ:STAR {LOW_FREQ}")
    vna.write(f"SENS1:FREQ:STOP {UPPER_FREQ}")

    if MANUAL_STEP_SIZE:
        npts = int(round((UPPER_FREQ - LOW_FREQ) / STEP_FREQ)) + 1
        if npts < 2:
            npts = 2
        vna.write(f"SENS1:SWE:POIN {npts}")

    vna.write("*CLS")
    vna.write("INIT1:CONT OFF")

    for trace_idx, s_param in enumerate(S_PARAMS_LIST, start=1):
        meas = f"S{s_param}"
        vna.write(f"CALC1:PAR{trace_idx}:DEF {meas}")
        vna.write(f"DISP:WIND1:TRAC{trace_idx}:FEED '{meas}'")

    return rm, vna


def capture_one_trace(vna, label_int):
    label_text = "occupied" if label_int == 1 else "empty"
    save_dir = OCCUPIED_DIR if label_int == 1 else EMPTY_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    saved_files = []
    feature_vector = None

    for trace_idx, s_param in enumerate(S_PARAMS_LIST, start=1):
        meas = f"S{s_param}"
        vna.write("DISP:WIND1:ACT")
        vna.write(f"CALC1:PAR{trace_idx}:SEL")

        vna.write("INIT1:IMM")
        sf.query_text(vna, "*OPC?")

        freqs = sf.query_csv_numbers(vna, "SENS1:FREQ:DATA?")
        sdata = sf.query_csv_numbers(vna, "CALC1:DATA:SDAT?")

        if len(sdata) != 2 * len(freqs):
            raise RuntimeError(f"Mismatch: freqs={len(freqs)} sdata={len(sdata)}")

        if feature_vector is None:
            fv = []
            for i in range(len(freqs)):
                fv.append(sdata[2 * i])
                fv.append(sdata[2 * i + 1])
            feature_vector = np.array(fv, dtype=float)

        local_csv = os.path.join(save_dir, f"{meas}_{label_text}_{timestamp}.csv")

        with open(local_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frequency_Hz", "Re", "Im"])
            for i, f_hz in enumerate(freqs):
                writer.writerow([f_hz, sdata[2 * i], sdata[2 * i + 1]])

        saved_files.append(local_csv)

    return saved_files, feature_vector


def predict_rf_label(model, feature_vector):
    if model is None or feature_vector is None:
        return None, None

    x = feature_vector.reshape(1, -1)
    pred = int(model.predict(x)[0])

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x)[0]
        except Exception:
            probs = None

    return pred, probs


def append_log(camera_label_int, largest_area, saved_csvs, rf_pred=None, rf_probs=None):
    label_text = "occupied" if camera_label_int == 1 else "empty"
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        rf_pred_text = "" if rf_pred is None else ("occupied" if rf_pred == 1 else "empty")
        rf_prob_empty = ""
        rf_prob_occupied = ""
        if rf_probs is not None and len(rf_probs) >= 2:
            rf_prob_empty = float(rf_probs[0])
            rf_prob_occupied = float(rf_probs[1])

        for path in saved_csvs:
            writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                camera_label_int,
                label_text,
                int(largest_area),
                path,
                "" if rf_pred is None else rf_pred,
                rf_pred_text,
                rf_prob_empty,
                rf_prob_occupied,
            ])


def main():
    ensure_dirs()

    rm, vna = connect_and_configure_vna()
    video = cv2.VideoCapture(CAMERA_INDEX)

    if not video.isOpened():
        raise RuntimeError("Could not open camera.")

    monitor = BoxOccupancyMonitor()
    last_capture_time = 0.0
    previous_verified_state = None
    last_saved_text = "none"
    last_rf_text = "model off"
    last_match_text = ""

    model = None
    if ENABLE_RF_INFERENCE and os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Loaded RF model: {MODEL_PATH}")
        except Exception as e:
            print(f"Could not load RF model: {e}")

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                print("Could not read camera frame.")
                break

            result = monitor.process_frame(frame)
            verified_state = result["inside_verified"]
            label_text = "INSIDE BOX" if verified_state == 1 else "OUTSIDE BOX"

            cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 0, 255), 2)
            cv2.putText(frame, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if verified_state == 1 else (255, 255, 255), 2)
            cv2.putText(frame, f"Last saved: {last_saved_text}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Area: {int(result['largest_area'])}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"RF: {last_rf_text}", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(frame, f"Check: {last_match_text}", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            now = time.time()
            should_capture = False

            if CAPTURE_ONLY_ON_STATE_CHANGE:
                if previous_verified_state is None or verified_state != previous_verified_state:
                    should_capture = True
            else:
                if now - last_capture_time >= CAPTURE_INTERVAL_SEC:
                    should_capture = True

            if should_capture:
                saved_csvs, feature_vector = capture_one_trace(vna, verified_state)
                rf_pred, rf_probs = predict_rf_label(model, feature_vector)
                append_log(verified_state, result["largest_area"], saved_csvs, rf_pred, rf_probs)
                last_capture_time = now
                last_saved_text = "occupied" if verified_state == 1 else "empty"

                if rf_pred is None:
                    last_rf_text = "model off"
                    last_match_text = "camera only"
                else:
                    last_rf_text = "occupied" if rf_pred == 1 else "empty"
                    last_match_text = "MATCH" if rf_pred == verified_state else "MISMATCH"

                print(f"Saved {len(saved_csvs)} file(s) with camera label = {last_saved_text}; RF = {last_rf_text}; {last_match_text}")

            previous_verified_state = verified_state

            cv2.imshow("Color Frame", frame)
            cv2.imshow("Gray Frame", result["gray"])
            cv2.imshow("Difference Frame", result["diff_frame"])
            cv2.imshow("Threshold Frame", result["thresh_frame"])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print("Final err:", sf.query_text(vna, "SYST:ERR?"))
        video.release()
        cv2.destroyAllWindows()
        vna.close()
        rm.close()


if __name__ == "__main__":
    main()