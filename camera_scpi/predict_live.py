"""
predict_live.py - the trained model guesses where the person is from the VNA
sweeps alone and shows it on the map.

Two modes:
  default        camera OFF: only the map. The model's cell lights up blue;
                 nothing lights up when it thinks the square is empty.
  --with-camera  camera ON as a check: the map also shows the camera's dot and
                 cell (green), the model's guess is a purple outline, and a
                 running "agrees with the camera" score is shown.

Start with the square EMPTY: the first REF seconds measure today's empty room
(the model works on "sweep minus empty room", exactly like in training).

Needs live_map.py, collect_live.py and train_model.py in the same folder, and a
model made by train_model.py ON THIS COMPUTER (<data folder>/models/model_latest.joblib).

Usage:
    python predict_live.py                    # camera off, real VNA
    python predict_live.py --with-camera      # compare with the camera
    python predict_live.py --model data/models/model_20261022_143000.joblib
Keys (click on the map window first):
    q  quit
    r  measure the empty room again (square must be empty)
Each run writes predictions.csv (+ the camera's answer when the camera is on)
to <data folder>/predict_<date_time>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import collect_live as cl
import live_map as lm
import train_model as tm

# =========================== settings ======================================
REF_SECONDS_DEFAULT = None    # None = use what the model was trained with (30 s)
AGREE_WINDOW_S = 30.0         # "last 30 s" agreement score
MODEL_FILL = (170, 90, 40)    # blue fill: the model's cell (camera off)
MODEL_EDGE = (220, 80, 220)   # purple outline: the model's cell (camera on)
# ===========================================================================


def cell_rect(r: int):
    cw, ch = lm.GX / lm.N_CELLS, lm.GY / lm.N_CELLS
    i, j = (r - 1) % lm.N_CELLS, (r - 1) // lm.N_CELLS
    return lm.map_px(i * cw, (j + 1) * ch), lm.map_px((i + 1) * cw, j * ch)


def draw_predict_map(tracker, guess: str | None, conf: float, camera_on: bool,
                     lines: list[tuple[str, tuple]]) -> np.ndarray:
    img = lm.draw_map(True, tracker if camera_on else lm.Tracker())
    h = int((lm.GY + 2 * lm.MAP_MARGIN_M) * lm.MAP_PX_PER_M)
    img[h:] = 35                                   # replace live_map's status line
    font = cv2.FONT_HERSHEY_SIMPLEX
    if guess and guess not in ("empty", "outside") and guess.isdigit():
        p0, p1 = cell_rect(int(guess))
        if camera_on:
            cv2.rectangle(img, (p0[0] + 3, p0[1] + 3), (p1[0] - 3, p1[1] - 3), MODEL_EDGE, 5)
        else:
            cv2.rectangle(img, p0, p1, MODEL_FILL, -1)
            cv2.rectangle(img, p0, p1, (200, 200, 200), 1)
            cv2.putText(img, guess, (p0[0] + 6, p0[1] + 22), font, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            c = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            cv2.circle(img, c, 14, (255, 230, 150), -1, cv2.LINE_AA)
            cv2.circle(img, c, 14, (0, 0, 0), 2, cv2.LINE_AA)
    elif guess == "outside":                       # band around the square
        a, b = lm.map_px(-0.25, lm.GY + 0.25), lm.map_px(lm.GX + 0.25, -0.25)
        cv2.rectangle(img, a, b, MODEL_EDGE if camera_on else MODEL_FILL, 6)
    if guess is None:
        big = "measuring the empty room..."
    elif guess == "empty":
        big = f"MODEL: EMPTY  ({conf:.0%})"
    elif guess == "outside":
        big = f"MODEL: OUTSIDE the square  ({conf:.0%})"
    else:
        big = f"MODEL: CELL {guess}  ({conf:.0%})"
    cv2.putText(img, big, (10, h + 30), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    bar = np.full((30 * len(lines) + 10, img.shape[1], 3), 25, np.uint8)
    for n, (text, color) in enumerate(lines):
        cv2.putText(bar, text, (10, 26 + 30 * n), font, 0.52, color, 1, cv2.LINE_AA)
    return np.vstack([img, bar])


class Agreement:
    """Running score of the model's (smoothed) answer vs the camera's label."""

    def __init__(self):
        self.all = Counter()
        self.recent: deque = deque()

    def add(self, t: float, guess: str, truth: str) -> None:
        ok, pres = guess == truth, (guess != "empty") == (truth != "empty")
        occ = truth != "empty"
        self.all.update(n=1, ok=ok, pres=pres, occ=occ, cell_ok=ok and occ)
        self.recent.append((t, ok))
        while self.recent and t - self.recent[0][0] > AGREE_WINDOW_S:
            self.recent.popleft()

    def text(self) -> str:
        a = self.all
        if not a["n"]:
            return "agrees with camera: waiting for usable sweeps..."
        rec = np.mean([ok for _, ok in self.recent]) if self.recent else float("nan")
        cell = f"{a['cell_ok'] / a['occ']:.0%}" if a["occ"] else "-"
        return (f"agrees with camera: {a['ok'] / a['n']:.0%}   anyone there? {a['pres'] / a['n']:.0%}   "
                f"which cell {cell}   last {AGREE_WINDOW_S:.0f} s: {rec:.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Live prediction from VNA sweeps")
    ap.add_argument("--model", default=str(Path(lm.OUTPUT_DIR) / "models" / "model_latest.joblib"))
    ap.add_argument("--with-camera", action="store_true", help="also run the camera and compare")
    ap.add_argument("--camera", default=lm.CAMERA, help="camera index, or a video file")
    ap.add_argument("--out", default=lm.OUTPUT_DIR)
    ap.add_argument("--vna-address", default=cl.VNA_ADDRESS)
    ap.add_argument("--fake-vna", action="store_true", help="simulated VNA (only for testing)")
    ap.add_argument("--ref-seconds", type=float, default=REF_SECONDS_DEFAULT,
                    help="seconds of empty room measured at the start")
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until q")
    ap.add_argument("--no-camera-settings", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    # --- model
    import joblib
    import sklearn
    mp = Path(args.model)
    if not mp.exists():
        raise SystemExit(f"no model at {mp}\nRun train_model.py first.")
    bundle = joblib.load(mp)
    model, s_params = bundle["model"], bundle["s_params"]
    n_smooth = int(bundle.get("smooth_sweeps", 30))
    ref_s = args.ref_seconds or float(bundle.get("ref_seconds", 30.0))
    if bundle.get("sklearn_version") != sklearn.__version__:
        print(f"WARNING: model made with scikit-learn {bundle.get('sklearn_version')}, this computer "
              f"has {sklearn.__version__}. If predictions look wrong, run train_model.py here.")
    scores = bundle.get("test_scores", {}).get(bundle["model_type"], {})
    print(f"model: {bundle['model_type']} trained on {', '.join(bundle['sessions'])} "
          f"(tested: which cell {scores.get('cell', float('nan')):.0%})")

    # --- VNA (must match what the model was trained on)
    person_now = {"xy": None}
    vna = cl.FakeVNA(lambda: person_now["xy"]) if args.fake_vna else cl.RealVNA(args.vna_address)
    try:
        info = vna.setup()
    except Exception as e:
        raise SystemExit(f"VNA connection failed: {type(e).__name__}: {e}")
    freqs = [f"{f / 1e6:.3f}MHz" for f in info["freqs_hz"]]
    if freqs != list(bundle["freqs_mhz"]) or not set(s_params) <= set(cl.S_PARAMS):
        vna.close()
        raise SystemExit("the VNA settings (frequencies or S-parameters) differ from the ones the "
                         "model was trained on - collect and train with the same settings")
    print(f"VNA: {info['idn']}")

    # --- camera (optional)
    cam = None
    if args.with_camera:
        settings = None
        sp = lm.settings_path(out, args.camera)
        if args.camera.isdigit() and not args.no_camera_settings and sp.exists():
            settings = json.loads(sp.read_text())["settings"]
        raw = lm.open_camera(args.camera, lm.REQUEST_SIZE, warmup_s=lm.WARMUP_S, settings=settings)
        is_file = os.path.isfile(args.camera)
        cap = lm.Grabber(raw, threaded=not is_file)
        ok, first = cap.read()
        if not ok:
            raise SystemExit("camera opened but gave no frame")
        cam_size = (first.shape[1], first.shape[0])
        proc_size = lm.to_proc(first).shape[1::-1]
        cv2.namedWindow(lm.CAM_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(lm.CAM_WIN, *proc_size)
        corners = lm.load_calibration(out, args.camera, cam_size)
        if corners is None:
            corners = lm.run_calibration(cap, out, args.camera, cam_size)
        if corners is None:
            raise SystemExit("the camera check needs the floor calibration")
        H = lm.homography(corners, proc_size)
        area, outline = lm.detection_area(H, proc_size)
        cam = {"raw": raw, "cap": cap, "is_file": is_file, "H": H, "area": area, "outline": outline,
               "limb_k": lm.odd_kernel(lm.LIMB_KERNEL_FRAC, lm.DETECT_WIDTH),
               "min_area": lm.MIN_AREA_FRAC * area.shape[0] * area.shape[1],
               "up": proc_size[0] / lm.DETECT_WIDTH,
               "file_dt": 1.0 / (raw.get(cv2.CAP_PROP_FPS) or 30.0)}
        cam["bg"] = lm.capture_background(cap, lm.BACKGROUND_FRAMES, area)
    cv2.namedWindow(lm.MAP_WIN, cv2.WINDOW_AUTOSIZE)

    # --- output
    rdir = out / datetime.now().strftime("predict_%Y%m%d_%H%M%S")
    rdir.mkdir(parents=True, exist_ok=True)
    fp = open(rdir / "predictions.csv", "w", newline="")
    wp = csv.writer(fp)
    wp.writerow(["t0_s", "t1_s", "model_raw", "model_shown", "votes"] +
                (["camera_label", "camera_usable", "camera_x_m", "camera_y_m"] if cam else []))

    # --- go
    sweeps_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    vstat = {"n": 0, "sweep_ms": 0.0, "times": deque(maxlen=60), "error": None}
    vth = threading.Thread(target=cl.vna_loop, args=(vna, sweeps_q, stop, vstat), daemon=True)
    vth.start()
    t_start = time.perf_counter()
    phase_ref, ref_sweeps, ref_t0, ref_msg = True, [], None, ""
    ref = None
    votes: deque = deque(maxlen=n_smooth)
    guess, conf = None, 0.0
    tracker, history = lm.Tracker(), deque()
    pending_cmp: list = []
    agree = Agreement()
    t_prev = time.perf_counter()

    def write_row(t0, t1, raw_p, shown, v, camlab=None) -> None:
        row = [f"{t0 - t_start:.4f}", f"{t1 - t_start:.4f}", raw_p, shown, v]
        if cam:
            row += [camlab["label"], camlab["usable"], camlab["x_m"], camlab["y_m"]]
        wp.writerow(row)

    try:
        while True:
            # ---- camera (only in the check mode)
            det, jump = None, False
            if cam:
                ok, frame = cam["cap"].read()
                if not ok:
                    print("no more frames" if cam["is_file"] else "camera read failed")
                    break
                if cam["is_file"]:
                    time.sleep(max(0.0, cam["file_dt"] - (time.perf_counter() - t_prev)))
                    t = time.perf_counter()
                else:
                    t = cam["cap"].last_t
                t_prev = time.perf_counter()
                proc = lm.to_proc(frame)
                gray = cam["bg"].compensate(lm.small_gray(proc))
                det, _, jump = lm.detect(gray, cam["bg"], cam["area"], cv2.countNonZero(cam["area"]),
                                         cam["limb_k"], cam["min_area"])
                if not jump:
                    cam["bg"].update(gray, lm.grow_box(det["box"], 0.3, gray.shape) if det else None,
                                     cam["file_dt"])
                xy = None
                if det is not None:
                    u = cam["up"]
                    cx, cy = det["centre_px"]
                    x, y, w, h = det["box"]
                    det = {"centre_px": (cx * u, cy * u), "box": (int(x * u), int(y * u), int(w * u), int(h * u))}
                    xy = lm.to_floor(cam["H"], [det["centre_px"]])[0]
                tracker.update(xy, t)
                person_now["xy"] = None if xy is None else (float(xy[0]), float(xy[1]))
                history.append((t - cl.CAMERA_LAG_S, det is not None,
                                *(xy if xy is not None else (np.nan, np.nan)), jump))
                while history and t - history[0][0] > 15.0:
                    history.popleft()
            else:
                t = time.perf_counter()

            # ---- new sweeps
            batch = []
            while True:
                try:
                    batch.append(sweeps_q.get_nowait())
                except queue.Empty:
                    break
            if phase_ref and batch:
                if ref_t0 is None:
                    ref_t0 = batch[0][0]
                seen_person = cam is not None and any(h[1] for h in list(history)[-10:])
                if seen_person:                         # restart: the reference must be empty
                    ref_sweeps, ref_t0 = [], None
                    ref_msg = "someone is in view - empty reference restarted"
                else:
                    ref_sweeps += [d for _, _, d in batch]
                    if batch[-1][1] - ref_t0 >= ref_s and len(ref_sweeps) >= 10:
                        ref = {sp: np.mean([d[sp] for d in ref_sweeps], axis=0) for sp in s_params}
                        phase_ref, ref_sweeps, ref_msg = False, [], ""
                        print(f"empty-room reference measured ({vstat['n']} sweeps so far); predicting")
                batch = []
            if batch:
                S = {sp: np.vstack([d[sp] for _, _, d in batch]) for sp in s_params}
                raw_preds = model.predict(tm.make_features(S, ref, s_params))
                for (t0, t1, _), p in zip(batch, raw_preds):
                    votes.append(str(p))
                    guess, nv = Counter(votes).most_common(1)[0]
                    conf = nv / len(votes)
                    if cam:
                        pending_cmp.append((t0, t1, str(p), guess, nv))
                    else:
                        write_row(t0, t1, str(p), guess, nv)
            # ---- compare with the camera once it has seen past each sweep
            if cam:
                keep = []
                for t0, t1, rp, g, nv in pending_cmp:
                    if t - cl.CAMERA_LAG_S > t1 + cl.EMPTY_GUARD_S:
                        lab = cl.label_sweep(t0, t1, history)
                        if lab["usable"]:
                            agree.add(t1, g, lab["label"])
                        write_row(t0, t1, rp, g, nv, lab)
                    else:
                        keep.append((t0, t1, rp, g, nv))
                pending_cmp = keep

            # ---- display
            vt = list(vstat["times"])
            sps = (len(vt) - 1) / (vt[-1] - vt[0]) if len(vt) > 2 and vt[-1] > vt[0] else 0.0
            lines = []
            if phase_ref:
                left = ref_s - ((vt[-1] - ref_t0) if (vt and ref_t0) else 0.0)
                lines.append((f"KEEP THE SQUARE EMPTY - measuring the empty room: {max(left, 0):.0f} s left",
                              (80, 200, 255)))
                if ref_msg:
                    lines.append((ref_msg, (80, 80, 255)))
            if cam:
                lines.append(("green cell + yellow dot = camera     purple outline = model", (200, 200, 200)))
                if not phase_ref:
                    lines.append((agree.text(), (120, 255, 120)))
            lines.append((f"VNA {sps:4.1f} sweeps/s   model {bundle['model_type']}   "
                          f"vote over {n_smooth} sweeps   r = new empty reference   q = quit",
                          (170, 170, 170)))
            if vstat["error"]:
                lines.append((f"VNA STOPPED: {vstat['error'][:70]}", (80, 80, 255)))
            cv2.imshow(lm.MAP_WIN, draw_predict_map(tracker, None if phase_ref else guess, conf,
                                                    cam is not None, lines))
            if cam:
                timing = f"camera {cam['cap'].camera_fps():4.1f} fps | VNA {sps:4.1f} sweeps/s"
                cv2.imshow(lm.CAM_WIN, lm.draw_camera(proc, cam["H"], cam["outline"], det, tracker,
                                                      jump, timing))

            if args.duration and t - t_start >= args.duration:
                break
            k = cv2.waitKey(1 if cam else 15) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("r"):
                phase_ref, ref_sweeps, ref_t0, ref_msg = True, [], None, ""
                votes.clear()
                guess = None
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        vth.join(timeout=5)
        vna.close()
        if cam:
            cam["cap"].close()
        fp.close()
        cv2.destroyAllWindows()
        summary = {"model": str(mp), "model_type": bundle["model_type"],
                   "trained_on": bundle["sessions"], "vna_error": vstat["error"]}
        if cam and agree.all["n"]:
            a = agree.all
            summary["agreement_with_camera"] = {
                "all": a["ok"] / a["n"], "presence": a["pres"] / a["n"],
                "cell": a["cell_ok"] / a["occ"] if a["occ"] else None, "usable_sweeps": a["n"]}
        (rdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if cam and agree.all["n"]:
        print(agree.text())
    print(f"saved {rdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
