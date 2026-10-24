"""
collect_live.py - live map + VNA sweeping: every sweep is saved together with
where the camera saw the person while that sweep ran.

Builds on live_map.py (keep both files in the same folder): same camera,
calibration, saved camera settings, keys and map. On top of that:
  * the VNA sweeps continuously in the background (S11 + S21, 902-928 MHz)
  * every sweep gets a label from the camera:
        empty        nobody detected in or near the square (for +-1 s around it)
        1 .. 9       the person stayed inside that cell, at least 15 cm from its lines
        outside      the person stayed just outside the square
        transition   crossing a line, entering/leaving -> saved, but marked unusable
        no_camera    no camera frames for that sweep -> unusable
  * the map shows how many usable sweeps each cell has, so you know where to walk

Output: <data folder>/session_<date_time>/   (data folder: see OUTPUT_DIR in live_map.py)
    sweeps.csv          one row per sweep: time, label, usable, x/y, then S11/S21 re/im
    camera_track.csv    one row per camera frame: time, detected, x/y (raw)
    camera.mp4          the camera video (lets labels be redone later)
    session_info.json   all settings, calibration, VNA identity and frequencies

Usage:
    python collect_live.py                     # real VNA + camera 0
    python collect_live.py --fake-vna          # no VNA needed (test the whole thing)
    python collect_live.py --duration 600      # stop by itself after 10 minutes
Keys: q quit, p pause/resume saving, b new background (square empty),
      c corners, s/k camera settings, m mask

Start every session with the square EMPTY: the first seconds give the camera
background and the empty-room sweeps.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import live_map as lm

# =========================== settings ======================================
VNA_ADDRESS = "TCPIP0::192.168.1.10::inst0::INSTR"
VNA_MODEL = "E5071C"
F_START_HZ = 902e6
F_STOP_HZ = 928e6
N_POINTS = 53            # 500 kHz steps
IF_BW_HZ = 10e3          # wider = faster sweeps, slightly more noise
POWER_DBM = 0.0          # None = leave the instrument's setting alone
S_PARAMS = ["S11", "S21"]  # both come from the same forward sweep
VNA_DISPLAY_OFF = True   # faster; switched back on at the end

LABEL_MARGIN_M = 0.15    # usable only if this far inside the cell for the whole sweep
FRAME_PAD_S = 0.05       # camera frames this close to a sweep also count for it
EMPTY_GUARD_S = 1.0      # "empty" needs nobody detected for this long around the sweep
CAMERA_LAG_S = 0.0       # camera delay to correct for (measure once, later)
SAVE_VIDEO = True
# ===========================================================================

CELL_LABELS = [str(r) for r in range(1, lm.N_CELLS * lm.N_CELLS + 1)]


# =========================== VNA ===========================================
class RealVNA:
    """E5071C over pyvisa: explicit settings, one bus-triggered sweep at a time."""

    def __init__(self, address: str):
        self.address = address

    def setup(self) -> dict:
        import pyvisa
        self.rm = pyvisa.ResourceManager()
        self.inst = inst = self.rm.open_resource(self.address)
        inst.write_termination = "\n"
        inst.read_termination = "\n"
        inst.timeout = 20000
        idn = inst.query("*IDN?").strip()
        if VNA_MODEL not in idn:
            raise RuntimeError(f"expected an {VNA_MODEL}, got: {idn}")
        w = inst.write
        w("*CLS")
        w(f"SENS1:FREQ:STAR {F_START_HZ:.0f}")
        w(f"SENS1:FREQ:STOP {F_STOP_HZ:.0f}")
        w(f"SENS1:SWE:POIN {N_POINTS}")
        w(f"SENS1:BWID {IF_BW_HZ:.0f}")
        w("SENS1:AVER OFF")
        w("TRIG:AVER OFF")
        if POWER_DBM is not None:
            w(f"SOUR1:POW {POWER_DBM:.2f}")
        w(f"CALC1:PAR:COUN {len(S_PARAMS)}")
        for i, sp in enumerate(S_PARAMS, start=1):
            w(f"CALC1:PAR{i}:DEF {sp}")
        w("FORM:DATA ASC")
        w(":INIT1:CONT ON")
        w(":TRIG:SOUR BUS")            # sweep only when we say so
        if VNA_DISPLAY_OFF:
            w(":DISP:ENAB OFF")
        self.freqs = np.array(inst.query_ascii_values("SENS1:FREQ:DATA?"))
        info = {"idn": idn, "calibration_on": inst.query("SENS1:CORR:STAT?").strip(),
                "freqs_hz": self.freqs.tolist(), "setup_errors": self.read_errors()}
        for e in info["setup_errors"]:
            print(f"[vna] SCPI error during setup: {e}")
        return info

    def sweep(self) -> None:
        self.inst.write(":TRIG:SING")
        self.inst.query("*OPC?")       # returns when this sweep is finished

    def read(self) -> dict:
        out = {}
        for i, sp in enumerate(S_PARAMS, start=1):
            self.inst.write(f"CALC1:PAR{i}:SEL")
            v = np.array(self.inst.query_ascii_values("CALC1:DATA:SDAT?"))
            if v.size != 2 * self.freqs.size:
                raise RuntimeError(f"{sp}: got {v.size} values, expected {2 * self.freqs.size}")
            out[sp] = v[0::2] + 1j * v[1::2]
        return out

    def read_errors(self) -> list[str]:
        errors = []
        for _ in range(20):
            e = self.inst.query("SYST:ERR?").strip()
            if e.startswith(("+0", "0")):
                break
            errors.append(e)
        return errors

    def close(self) -> None:
        try:
            self.inst.write(":TRIG:SOUR INT")   # front panel back to free-running
            if VNA_DISPLAY_OFF:
                self.inst.write(":DISP:ENAB ON")
            for e in self.read_errors():
                print(f"[vna] SCPI error at end: {e}")
            self.inst.close()
            self.rm.close()
        except Exception:
            pass


class FakeVNA:
    """No instrument needed. The 'channel' changes with where the camera sees the
    person, so the whole pipeline (and later the training) can be tried at home."""

    def __init__(self, person_xy):
        self.person_xy = person_xy          # function -> (x, y) in metres or None
        self.rng = np.random.default_rng()

    def setup(self) -> dict:
        self.freqs = np.linspace(F_START_HZ, F_STOP_HZ, N_POINTS)
        k = 2 * np.pi * self.freqs / 3e8
        self.base = {"S11": 0.08 * np.exp(-1j * k * 6.0), "S21": 0.006 * np.exp(-1j * k * 6.1)}
        self.k = k
        return {"idn": "FAKE VNA", "calibration_on": "0", "freqs_hz": self.freqs.tolist(),
                "setup_errors": []}

    def sweep(self) -> None:
        time.sleep(0.03)

    def read(self) -> dict:
        out = {sp: b.copy() for sp, b in self.base.items()}
        xy = self.person_xy()
        if xy is not None:
            p = np.array([xy[0], xy[1], 1.0])
            a1, a2 = np.array([-0.5, -0.5, 2.7]), np.array([4.5, 4.5, 2.7])
            excess = np.linalg.norm(p - a1) + np.linalg.norm(p - a2) - np.linalg.norm(a2 - a1)
            out["S21"] = out["S21"] + 0.002 * np.exp(-1j * self.k * excess)
            out["S11"] = out["S11"] + 0.004 / np.linalg.norm(p - a1) * np.exp(-2j * self.k * np.linalg.norm(p - a1))
        for sp in out:
            out[sp] = out[sp] + 2e-4 * (self.rng.standard_normal(out[sp].size)
                                        + 1j * self.rng.standard_normal(out[sp].size))
        return out

    def close(self) -> None:
        pass


def vna_loop(vna, out_q: queue.Queue, stop: threading.Event, status: dict) -> None:
    """Background thread: sweep, timestamp, hand over, repeat."""
    try:
        while not stop.is_set():
            t0 = time.perf_counter()
            vna.sweep()
            t1 = time.perf_counter()
            data = vna.read()
            out_q.put((t0, t1, data))
            status["n"] += 1
            status["sweep_ms"] = (t1 - t0) * 1e3
            status["times"].append(t1)
    except Exception as e:
        traceback.print_exc()
        status["error"] = f"{type(e).__name__}: {e}"


# =========================== labelling =====================================
def label_sweep(t0: float, t1: float, history) -> dict:
    """Decide what the camera saw while one sweep ran.
    history: (t, detected, x, y, lighting_jump) per camera frame, oldest first."""
    a, b = t0 - FRAME_PAD_S, t1 + FRAME_PAD_S
    during = [h for h in history if a <= h[0] <= b]
    out = {"label": "no_camera", "usable": 0, "n_frames": len(during),
           "x_m": "", "y_m": "", "moved_m": ""}
    if not during or any(h[4] for h in during):
        return out
    if not any(h[1] for h in during):
        around = [h for h in history if t0 - EMPTY_GUARD_S <= h[0] <= t1 + EMPTY_GUARD_S]
        if not any(h[1] for h in around):
            out.update(label="empty", usable=1)
        else:
            out["label"] = "transition"       # someone near in time: not safely empty
        return out
    if not all(h[1] for h in during):
        out["label"] = "transition"
        return out
    pts = np.array([(h[2], h[3]) for h in during])
    out.update(x_m=f"{pts[:, 0].mean():.3f}", y_m=f"{pts[:, 1].mean():.3f}",
               moved_m=f"{np.linalg.norm(pts.max(0) - pts.min(0)):.3f}")
    regions = {lm.region_of(x, y) for x, y in pts}
    if len(regions) != 1:
        out["label"] = "transition"
        return out
    r = regions.pop()
    if min(lm.region_depth(r, x, y) for x, y in pts) < LABEL_MARGIN_M:
        out["label"] = "transition"           # too close to a line to be sure
        return out
    out.update(label="outside" if r == 0 else str(r), usable=1)
    return out


# =========================== drawing =======================================
def draw_collect_map(H, tracker, counts: Counter, lines: list[tuple[str, tuple]]) -> np.ndarray:
    img = lm.draw_map(H, tracker)
    cw, ch = lm.GX / lm.N_CELLS, lm.GY / lm.N_CELLS
    for r in range(1, lm.N_CELLS * lm.N_CELLS + 1):      # usable sweeps per cell
        i, j = (r - 1) % lm.N_CELLS, (r - 1) // lm.N_CELLS
        x1, y1 = lm.map_px((i + 1) * cw, j * ch)
        text = str(counts[str(r)])
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(img, text, (x1 - tw - 8, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (150, 220, 255), 1, cv2.LINE_AA)
    bar = np.full((30 * len(lines) + 10, img.shape[1], 3), 25, np.uint8)
    for n, (text, color) in enumerate(lines):
        cv2.putText(bar, text, (10, 26 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                    cv2.LINE_AA)
    return np.vstack([img, bar])


# =========================== main ==========================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Live map + VNA data collection")
    ap.add_argument("--camera", default=lm.CAMERA, help="camera index, or a video file")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--calibrate", action="store_true", help="click the tape corners again")
    ap.add_argument("--out", default=lm.OUTPUT_DIR, help="parent folder for sessions")
    ap.add_argument("--session", help="session folder name (default session_<date_time>)")
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until q")
    ap.add_argument("--vna-address", default=VNA_ADDRESS)
    ap.add_argument("--fake-vna", action="store_true", help="simulated VNA, no instrument")
    ap.add_argument("--no-video", action="store_true", help="don't save camera.mp4")
    ap.add_argument("--no-camera-settings", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- VNA first: if it can't connect there is no point opening the camera
    person_now = {"xy": None}
    vna = FakeVNA(lambda: person_now["xy"]) if args.fake_vna else RealVNA(args.vna_address)
    try:
        vna_info = vna.setup()
    except Exception as e:
        raise SystemExit(f"VNA connection failed: {type(e).__name__}: {e}\n"
                         "Run vna_connect_check.py, or use --fake-vna to test without it.")
    freqs = vna_info["freqs_hz"]
    print(f"VNA: {vna_info['idn']} | {len(freqs)} points | calibration on: "
          f"{vna_info['calibration_on']}")

    # --- camera, exactly as in live_map.py
    settings = None
    sp = lm.settings_path(out, args.camera)
    if args.camera.isdigit() and not args.no_camera_settings and sp.exists():
        settings = json.loads(sp.read_text())["settings"]
    size = (args.width, args.height) if args.width and args.height else lm.REQUEST_SIZE
    print("opening camera ...")
    raw = lm.open_camera(args.camera, size, warmup_s=lm.WARMUP_S, settings=settings)
    is_file = os.path.isfile(args.camera)
    cap = lm.Grabber(raw, threaded=not is_file)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("camera opened but gave no frame")
    cam_size = (first.shape[1], first.shape[0])
    proc_size = lm.to_proc(first).shape[1::-1]
    print(f"camera {args.camera}: {cam_size[0]}x{cam_size[1]} {lm.fourcc_of(raw)}")

    cv2.namedWindow(lm.CAM_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lm.CAM_WIN, *proc_size)
    cv2.namedWindow(lm.MAP_WIN, cv2.WINDOW_AUTOSIZE)
    corners = None if args.calibrate else lm.load_calibration(out, args.camera, cam_size)
    if corners is None:
        corners = lm.run_calibration(cap, out, args.camera, cam_size)
    if corners is None:
        vna.close()
        raise SystemExit("collection needs the floor calibration: run again and click the corners")
    H = lm.homography(corners, proc_size)
    area, outline = lm.detection_area(H, proc_size)
    limb_k = lm.odd_kernel(lm.LIMB_KERNEL_FRAC, lm.DETECT_WIDTH)
    min_area = lm.MIN_AREA_FRAC * area.shape[0] * area.shape[1]
    up = proc_size[0] / lm.DETECT_WIDTH
    bg = lm.capture_background(cap, lm.BACKGROUND_FRAMES, area)

    # --- session folder and files
    session = args.session or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    sdir = out / session
    sdir.mkdir(parents=True, exist_ok=False)
    t_start = time.perf_counter()
    info = {"session": session, "started": datetime.now().isoformat(timespec="seconds"),
            "vna": {**vna_info, "address": "fake" if args.fake_vna else args.vna_address,
                    "f_start_hz": F_START_HZ, "f_stop_hz": F_STOP_HZ, "n_points": N_POINTS,
                    "if_bw_hz": IF_BW_HZ, "power_dbm": POWER_DBM, "s_params": S_PARAMS},
            "camera": {"source": args.camera, "capture_size": list(cam_size),
                       "settings": settings, "corners_norm": corners},
            "grid": {"size_m": list(lm.GRID_SIZE_M), "cells": lm.N_CELLS,
                     "cell_1": "corner next to antenna 1"},
            "labelling": {"label_margin_m": LABEL_MARGIN_M, "frame_pad_s": FRAME_PAD_S,
                          "empty_guard_s": EMPTY_GUARD_S, "camera_lag_s": CAMERA_LAG_S},
            "events": []}
    f_sw = open(sdir / "sweeps.csv", "w", newline="")
    w_sw = csv.writer(f_sw)
    w_sw.writerow(["sweep", "t0_s", "t1_s", "label", "usable", "x_m", "y_m", "moved_m",
                   "n_frames"] +
                  [f"{s}_{part}_{fr / 1e6:.3f}MHz" for s in S_PARAMS for part in ("re", "im")
                   for fr in freqs])
    f_cam = open(sdir / "camera_track.csv", "w", newline="")
    w_cam = csv.writer(f_cam)
    w_cam.writerow(["frame", "t_s", "detected", "x_m", "y_m", "lighting_jump", "cell_shown"])
    video = None
    print(f"session folder: {sdir}")

    # --- start sweeping
    sweeps_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    vstat = {"n": 0, "sweep_ms": 0.0, "times": deque(maxlen=50), "error": None}
    vna_thread = threading.Thread(target=vna_loop, args=(vna, sweeps_q, stop, vstat), daemon=True)
    vna_thread.start()

    tracker = lm.Tracker()
    history: deque = deque()
    pending: list = []
    counts: Counter = Counter()
    n_saved = n_frames = 0
    paused = show_mask = False
    file_dt = 1.0 / (raw.get(cv2.CAP_PROP_FPS) or 30.0)
    t_prev = time.perf_counter()
    t_last_frame = None

    def rel(t: float) -> str:
        return f"{t - t_start:.4f}"

    def write_sweep(t0, t1, data, lab) -> None:
        """Paused = the sweep is dropped, not written."""
        nonlocal n_saved
        if paused:
            return
        row = [n_saved, rel(t0), rel(t1), lab["label"], lab["usable"], lab["x_m"],
               lab["y_m"], lab["moved_m"], lab["n_frames"]]
        for s in S_PARAMS:
            row += [f"{v:.7g}" for v in data[s].real] + [f"{v:.7g}" for v in data[s].imag]
        w_sw.writerow(row)
        n_saved += 1
        if lab["usable"]:
            counts[lab["label"]] += 1

    last_label = ""
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("no more frames" if is_file else "camera read failed")
                break
            if is_file:                               # play test videos at real speed
                time.sleep(max(0.0, file_dt - (time.perf_counter() - t_prev)))
                t = time.perf_counter()
            else:
                t = cap.last_t                        # when the frame was grabbed
            t_cam = t - CAMERA_LAG_S
            t_prev = time.perf_counter()
            dt = file_dt if t_last_frame is None else max(t - t_last_frame, 1e-3)
            t_last_frame = t

            proc = lm.to_proc(frame)
            gray = bg.compensate(lm.small_gray(proc))
            det, mask, jump = lm.detect(gray, bg, area, cv2.countNonZero(area), limb_k, min_area)
            if not jump:
                bg.update(gray, lm.grow_box(det["box"], 0.3, gray.shape) if det else None, dt)
            xy = None
            if det is not None:
                cx, cy = det["centre_px"]
                x, y, w, h = det["box"]
                det = {"centre_px": (cx * up, cy * up),
                       "box": (int(x * up), int(y * up), int(w * up), int(h * up))}
                xy = lm.to_floor(H, [det["centre_px"]])[0]
            tracker.update(xy, t)
            person_now["xy"] = None if xy is None else (float(xy[0]), float(xy[1]))

            # camera record
            history.append((t_cam, det is not None, *(xy if xy is not None else (np.nan, np.nan)), jump))
            while history and t_cam - history[0][0] > 15.0:
                history.popleft()
            w_cam.writerow([n_frames, rel(t_cam), int(det is not None),
                            "" if xy is None else f"{xy[0]:.3f}", "" if xy is None else f"{xy[1]:.3f}",
                            int(jump), "" if tracker.region is None else tracker.region])
            if SAVE_VIDEO and not args.no_video:
                if video is None:
                    video = cv2.VideoWriter(str(sdir / "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            30.0, proc_size)
                video.write(proc)
            n_frames += 1

            # label and save sweeps once the camera has seen past them
            while True:
                try:
                    pending.append(sweeps_q.get_nowait())
                except queue.Empty:
                    break
            still_pending = []
            for t0, t1, data in pending:
                if t_cam > t1 + EMPTY_GUARD_S:
                    lab = label_sweep(t0, t1, history)
                    write_sweep(t0, t1, data, lab)
                    last_label = lab["label"] + ("" if lab["usable"] else " (not used)")
                else:
                    still_pending.append((t0, t1, data))
            pending = still_pending
            if n_frames % 30 == 0:
                f_sw.flush()
                f_cam.flush()

            # display
            vt = list(vstat["times"])
            sps = (len(vt) - 1) / (vt[-1] - vt[0]) if len(vt) > 2 and vt[-1] > vt[0] else 0.0
            timing = f"camera {cap.camera_fps():4.1f} fps | VNA {sps:4.1f} sweeps/s"
            cv2.imshow(lm.CAM_WIN, lm.draw_camera(proc, H, outline, det, tracker, jump, timing))
            usable = sum(counts.values())
            lines = [(f"{'PAUSED - press p to resume' if paused else 'RECORDING'}   "
                      f"{t - t_start:5.0f} s   sweeps {n_saved}   usable {usable}   "
                      f"empty {counts['empty']}   outside {counts['outside']}",
                      (80, 200, 255) if paused else (120, 255, 120)),
                     (f"VNA {sps:4.1f} sweeps/s, {vstat['sweep_ms']:.0f} ms each   last: {last_label}",
                      (220, 220, 220))]
            if vstat["error"]:
                lines.append((f"VNA STOPPED: {vstat['error'][:70]}", (80, 80, 255)))
            cv2.imshow(lm.MAP_WIN, draw_collect_map(H, tracker, counts, lines))
            if show_mask:
                cv2.imshow(lm.MASK_WIN, mask)

            if args.duration and t - t_start >= args.duration:
                break
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("p"):
                paused = not paused
                info["events"].append([rel(t), "paused" if paused else "resumed"])
            elif k == ord("b"):
                bg = lm.capture_background(cap, lm.BACKGROUND_FRAMES, area)
                tracker.reset()
                info["events"].append([rel(t), "new background"])
            elif k == ord("c"):
                new = lm.run_calibration(cap, out, args.camera, cam_size)
                if new is not None:
                    corners, H = new, lm.homography(new, proc_size)
                    area, outline = lm.detection_area(H, proc_size)
                    bg.set_area(area)
                    info["events"].append([rel(t), "recalibrated", new])
                tracker.reset()
            elif k == ord("s"):
                raw.set(cv2.CAP_PROP_SETTINGS, 1)
            elif k == ord("k") and not is_file:
                lm.save_camera_settings(raw, out, args.camera)
            elif k == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow(lm.MASK_WIN)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        vna_thread.join(timeout=5)                    # let the last sweep finish
        while True:                                   # label what is left
            try:
                pending.append(sweeps_q.get_nowait())
            except queue.Empty:
                break
        for t0, t1, data in pending:
            write_sweep(t0, t1, data, label_sweep(t0, t1, history))
        vna.close()
        cap.close()
        if video is not None:
            video.release()
        f_sw.close()
        f_cam.close()
        cv2.destroyAllWindows()
        info["ended"] = datetime.now().isoformat(timespec="seconds")
        info["summary"] = {"sweeps": n_saved, "camera_frames": n_frames,
                           "usable_by_label": dict(counts), "vna_error": vstat["error"]}
        (sdir / "session_info.json").write_text(json.dumps(info, indent=2))

    dur = time.perf_counter() - t_start
    print(f"\n=== {session}: {n_saved} sweeps in {dur:.0f} s ({n_saved / max(dur, 1e-9):.1f}/s), "
          f"{n_frames} camera frames ===")
    order = ["empty"] + CELL_LABELS + ["outside"]
    print("usable sweeps: " + "  ".join(f"{k}:{counts[k]}" for k in order))
    print(f"saved in {sdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
