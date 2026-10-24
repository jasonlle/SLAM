"""
live_map.py - live ceiling-camera view + a top-down map with the person's dot.

No recording, no VNA: this is for checking the camera and the tracking.

Two windows:
  camera  the live image with the taped 3x3 grid drawn on top, a green box
          around the person and a yellow dot at their centre
  map     the taped square from above with cells 1-9, the antennas, and the
          person as a dot with a short trail

Usage (terminal in the folder containing this file):
    python live_map.py                    # camera 0 at full field of view
    python live_map.py --calibrate        # click the four tape corners again
    python live_map.py --probe            # find the resolution that is NOT cropped
    python live_map.py --width 1280 --height 720
    python live_map.py --no-camera-settings   # don't apply the saved settings

Keys (click on either window first):
    q  quit
    b  new background - the square must be EMPTY while it captures
    c  click the four tape corners again
    s  open the camera's own settings (Windows): turn OFF auto exposure
    k  keep: save the camera's current settings (exposure, gain, focus, ...);
       they are applied automatically every time the script starts
    m  show / hide the detection mask (black = background, white = "person")

Bottom bar: "camera" = frames per second the camera delivers, "shown" = frames
per second actually processed, "work" = processing time per frame. If camera
fps is low, it is the camera settings (exposure, Low Light Compensation, or an
uncompressed video mode), not the computer.

Calibration: click the four corners of the taped square, starting at the
corner next to antenna 1 and going around the square in either direction.
It is saved in the data folder (OUTPUT_DIR) and reused on the next run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# =========================== settings ======================================
# Data folder (sessions, calibration, models). Set the SLAM_DATA_DIR environment
# variable to use your own folder; otherwise a "data" folder next to these scripts.
OUTPUT_DIR = os.environ.get("SLAM_DATA_DIR") or str(Path(__file__).resolve().parent / "data")
CAMERA = "0"                 # camera index, or a video file for testing
REQUEST_SIZE = (1920, 1080)  # asked from the camera; it picks its nearest mode
PROC_WIDTH = 960             # frames are shown at this width (full field of view)
DETECT_WIDTH = 480           # ... and searched for people at this smaller width (faster)

GRID_SIZE_M = (4.0, 4.0)     # taped square
N_CELLS = 3                  # 3 x 3 cells
ANTENNAS_M = {"ANT1": (-0.5, -0.5), "ANT2": (4.5, 4.5)}   # for the map only

WARMUP_S = 2.0               # let the camera settle its exposure before the background
BACKGROUND_FRAMES = 30       # empty-room frames averaged into the background
DIFF_THRESHOLD = 30          # brightness change that counts as "something there"
DETECT_MARGIN_M = 1.0        # only look for people in the square + this margin
MIN_AREA_FRAC = 0.004        # smallest person blob, as a fraction of the frame
LIMB_KERNEL_FRAC = 0.03      # removes thin parts (arms) so the dot stays on the body
BG_ADAPT_S = 30.0            # background slowly follows lighting drift (not under the person)
MAX_CHANGED_FRAC = 0.30      # more than this much "changed" at once = lighting, not a person
SMOOTH_S = 0.3               # dot smoothing time constant (seconds)
LOST_RESET_S = 0.5           # forget the person after this long undetected
CELL_HYSTERESIS_M = 0.15     # must be this far into a new cell before switching
TRAIL_S = 3.0                # length of the trail on the map (seconds)
MAP_PX_PER_M = 100           # map scale
MAP_MARGIN_M = 1.0           # map border around the square
# ===========================================================================

GX, GY = GRID_SIZE_M
FLOOR_CORNERS_M = [(0.0, 0.0), (GX, 0.0), (GX, GY), (0.0, GY)]
CORNER_PROMPTS = ["corner 1 of 4: the corner next to ANTENNA 1",
                  "corner 2 of 4: the next corner, going around the square",
                  "corner 3 of 4: the corner next to ANTENNA 2",
                  "corner 4 of 4: the last corner"]
CAL_VERSION = 2
PROBE_SIZES = [(640, 480), (800, 600), (1280, 720), (1280, 960),
               (1600, 1200), (1920, 1080), (2560, 1440), (3840, 2160)]
CAM_WIN, MAP_WIN, MASK_WIN, CAL_WIN = "camera", "map", "mask", "calibration"
# camera settings saved with 'k' and re-applied at start, in this order.
# Values are the driver's own numbers (not the % shown in Logi Tune).
CAMERA_PROPS = [("autofocus", cv2.CAP_PROP_AUTOFOCUS), ("focus", cv2.CAP_PROP_FOCUS),
                ("auto_white_balance", cv2.CAP_PROP_AUTO_WB),
                ("white_balance", cv2.CAP_PROP_WB_TEMPERATURE),
                ("exposure", cv2.CAP_PROP_EXPOSURE), ("gain", cv2.CAP_PROP_GAIN),
                ("brightness", cv2.CAP_PROP_BRIGHTNESS), ("contrast", cv2.CAP_PROP_CONTRAST),
                ("saturation", cv2.CAP_PROP_SATURATION), ("sharpness", cv2.CAP_PROP_SHARPNESS)]


# =========================== camera ========================================
def camera_tag(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_")[-40:]


def settings_path(out_dir: Path, source: str) -> Path:
    return out_dir / f"camera_settings_camera_{camera_tag(source)}.json"


def save_camera_settings(cap, out_dir: Path, source: str) -> None:
    """Read what the camera is set to right now and keep it for next time."""
    vals = {name: cap.get(prop) for name, prop in CAMERA_PROPS}
    vals["autofocus"] = 0.0             # always re-apply as manual focus ...
    vals["auto_white_balance"] = 0.0    # ... and fixed white balance
    p = settings_path(out_dir, source)
    p.write_text(json.dumps({"saved": datetime.now().isoformat(timespec="seconds"),
                             "settings": vals}, indent=2))
    print(f"saved camera settings -> {p}")
    print("  " + ", ".join(f"{k}={v:g}" for k, v in vals.items()))


def apply_camera_settings(cap, settings: dict) -> None:
    rows = []
    for name, prop in CAMERA_PROPS:
        if name in settings:
            cap.set(prop, settings[name])
            rows.append(f"{name} {settings[name]:g}->{cap.get(prop):g}")
    print("camera settings applied (saved -> camera reports): " + ", ".join(rows))


def open_camera(source: str, size: tuple[int, int] | None, warmup_s: float = 0.0,
                settings: dict | None = None):
    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        if size:
            # MJPG lets USB cameras send large frames at full frame rate
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            if fourcc_of(cap) != "MJPG":          # some drivers want it after the size
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if settings and cap.isOpened():
            apply_camera_settings(cap, settings)
    else:
        cap = cv2.VideoCapture(source)            # video file, for testing
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {source!r} "
                         "(wrong index, or another app such as Teams/Camera is using it)")
    if source.isdigit():
        t_end = time.time() + warmup_s
        cap.read()
        while time.time() < t_end:                # auto exposure settles here
            cap.read()
    return cap


def fourcc_of(cap) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or "?"


class Grabber:
    """Reads a live camera in a background thread and keeps only the NEWEST frame.
    Without this, frames queue up while the previous one is processed and the
    picture falls further and further behind (lag). Video files are read in order."""

    def __init__(self, cap, threaded: bool):
        self.cap, self.threaded = cap, threaded
        self.lock = threading.Lock()
        self.frame, self.seq, self.used_seq, self.alive = None, 0, 0, True
        self.frame_t = self.last_t = 0.0      # perf_counter() time each frame was grabbed
        self.grab_times: deque = deque(maxlen=60)
        if threaded:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                self.alive = False
                break
            now = time.perf_counter()
            with self.lock:
                self.frame, self.frame_t, self.seq = f, now, self.seq + 1
                self.grab_times.append(now)

    def read(self):
        """Newest frame not handed out yet; waits for it if necessary."""
        if not self.threaded:
            ok, f = self.cap.read()
            if ok:
                self.last_t = time.perf_counter()
                self.grab_times.append(self.last_t)
            return ok, f
        t_end = time.perf_counter() + 5.0
        while time.perf_counter() < t_end:
            with self.lock:
                if self.frame is not None and self.seq != self.used_seq:
                    self.used_seq, self.last_t = self.seq, self.frame_t
                    return True, self.frame
            if not self.alive:
                return False, None
            time.sleep(0.002)
        return False, None

    def camera_fps(self) -> float:
        g = list(self.grab_times)
        return (len(g) - 1) / (g[-1] - g[0]) if len(g) > 2 and g[-1] > g[0] else 0.0

    def close(self) -> None:
        self.alive = False
        time.sleep(0.05)
        self.cap.release()


def to_proc(frame: np.ndarray) -> np.ndarray:
    """Shrink to PROC_WIDTH (keeps the full field of view, just fewer pixels)."""
    h, w = frame.shape[:2]
    if w <= PROC_WIDTH:
        return frame
    return cv2.resize(frame, (PROC_WIDTH, round(h * PROC_WIDTH / w)), interpolation=cv2.INTER_AREA)


def probe(source: str, out_dir: Path) -> None:
    """Ask for each common resolution and save one snapshot per size the camera gives."""
    if not source.isdigit():
        raise SystemExit("--probe needs a camera index, e.g. --camera 0")
    seen = set()
    for size in PROBE_SIZES:
        cap = open_camera(source, size, warmup_s=0.5)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"asked {size[0]}x{size[1]}: no frame")
            continue
        got = (frame.shape[1], frame.shape[0])
        note = ""
        if got not in seen:
            seen.add(got)
            path = out_dir / f"probe_camera{source}_{got[0]}x{got[1]}.png"
            cv2.imwrite(str(path), frame)
            note = f"  snapshot -> {path.name}"
        print(f"asked {size[0]}x{size[1]}: camera gave {got[0]}x{got[1]}{note}")
    print("\nOpen the snapshots and pick the one with the WIDEST view (compare with the "
          "Windows Camera app).\nThen run:  python live_map.py --width W --height H   "
          "(or set REQUEST_SIZE at the top of this file)")


# =========================== calibration ===================================
def calibration_path(out_dir: Path, source: str) -> Path:
    return out_dir / f"live_calibration_camera_{camera_tag(source)}.json"


def load_calibration(out_dir: Path, source: str, cam_size: tuple[int, int]):
    p = calibration_path(out_dir, source)
    if not p.exists():
        return None
    cal = json.loads(p.read_text())
    if cal.get("version") != CAL_VERSION:
        return None
    if tuple(cal["capture_size"]) != tuple(cam_size):
        print(f"calibration was made at {cal['capture_size']}, camera now gives "
              f"{list(cam_size)}: click the corners again")
        return None
    return cal["corners_norm"]


def orient_corners(clicks):
    """Clicks went around the square in either direction from the antenna-1 corner.
    Reorder so the map is a rotated (never mirrored) copy of the camera view."""
    pts = np.float32(clicks)
    signed_area = 0.5 * sum(pts[i, 0] * pts[(i + 1) % 4, 1] - pts[(i + 1) % 4, 0] * pts[i, 1]
                            for i in range(4))
    if signed_area > 0:              # same turning direction as the map's corner order
        return [clicks[0], clicks[3], clicks[2], clicks[1]]
    return list(clicks)


def run_calibration(cap, out_dir: Path, source: str, cam_size: tuple[int, int]):
    """Click the four tape corners. Returns corners as fractions of the frame, or None."""
    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    cv2.namedWindow(CAL_WIN, cv2.WINDOW_AUTOSIZE)   # 1:1 pixels, so clicks are exact
    cv2.setMouseCallback(CAL_WIN, on_mouse)
    frame = None
    while True:
        ok, new = cap.read()
        if ok:
            frame = to_proc(new)
        if frame is None:
            continue
        vis = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(vis, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2, cv2.LINE_AA)
        crossed = False
        if len(clicks) == 4:
            crossed = not cv2.isContourConvex(np.int32(clicks).reshape(-1, 1, 2))
            cv2.polylines(vis, [np.int32(clicks)], True, (0, 0, 255) if crossed else (0, 255, 0), 2)
            msg = ("lines cross: press r and click going AROUND the square" if crossed
                   else "ENTER = save    r = start over    ESC = skip")
        else:
            msg = f"click {CORNER_PROMPTS[len(clicks)]}    (r = start over)"
        banner(vis, msg, top=True, color=(0, 255, 255))
        cv2.imshow(CAL_WIN, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            clicks.clear()
        elif k in (13, 10) and len(clicks) == 4 and not crossed:
            break
        elif k == 27:
            cv2.destroyWindow(CAL_WIN)
            print("calibration skipped: the map stays empty until you press c")
            return None
    cv2.destroyWindow(CAL_WIN)
    h, w = frame.shape[:2]
    corners_norm = [[x / w, y / h] for x, y in orient_corners(clicks)]
    p = calibration_path(out_dir, source)
    p.write_text(json.dumps({"version": CAL_VERSION,
                             "created": datetime.now().isoformat(timespec="seconds"),
                             "capture_size": list(cam_size), "corners_norm": corners_norm,
                             "grid_size_m": list(GRID_SIZE_M)}, indent=2))
    print(f"saved {p}")
    return corners_norm


def homography(corners_norm, proc_size: tuple[int, int]) -> np.ndarray:
    px = np.float32([[u * proc_size[0], v * proc_size[1]] for u, v in corners_norm])
    return cv2.getPerspectiveTransform(px, np.float32(FLOOR_CORNERS_M))


def to_floor(H: np.ndarray, pts_px) -> np.ndarray:
    return cv2.perspectiveTransform(np.float32(pts_px).reshape(-1, 1, 2), H).reshape(-1, 2)


def to_image(H: np.ndarray, pts_m) -> np.ndarray:
    return cv2.perspectiveTransform(np.float32(pts_m).reshape(-1, 1, 2),
                                    np.linalg.inv(H)).reshape(-1, 2)


def detection_area(H, proc_size: tuple[int, int]):
    """Mask at detection size (255 = look here) and outline (display size) of the
    square plus DETECT_MARGIN_M."""
    w, h = proc_size
    dw, dh = DETECT_WIDTH, round(h * DETECT_WIDTH / w)
    if H is None:
        return np.full((dh, dw), 255, np.uint8), None
    m = DETECT_MARGIN_M
    outline = np.int32(np.round(to_image(H, [(-m, -m), (GX + m, -m), (GX + m, GY + m), (-m, GY + m)])))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [outline], 255)
    return cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST), outline


# =========================== detection =====================================
def small_gray(proc: np.ndarray) -> np.ndarray:
    """Shrink to DETECT_WIDTH, grey, blur: what the detector compares."""
    h, w = proc.shape[:2]
    small = cv2.resize(proc, (DETECT_WIDTH, round(h * DETECT_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (11, 11), 0).astype(np.float32)


class Background:
    """Empty-room image. Copes with the camera changing its overall brightness, and
    slowly follows lighting drift everywhere except where the person is."""

    def __init__(self, frames: list[np.ndarray], area: np.ndarray):
        self.bg = np.median(np.stack(frames), axis=0).astype(np.float32)
        self.set_area(area)

    def set_area(self, area: np.ndarray) -> None:
        self.idx = np.flatnonzero(area)[::7]          # sample of pixels used for brightness
        self.ref = float(np.median(self.bg.ravel()[self.idx]))

    def compensate(self, gray: np.ndarray) -> np.ndarray:
        gain = self.ref / max(float(np.median(gray.ravel()[self.idx])), 1.0)
        return np.clip(gray * gain, 0, 255)

    def update(self, gray: np.ndarray, keep_out_box, dt: float) -> None:
        a = min(1.0, dt / BG_ADAPT_S)
        new = self.bg + a * (gray - self.bg)
        if keep_out_box is not None:                   # leave the person out of it
            x, y, w, h = keep_out_box
            new[y:y + h, x:x + w] = self.bg[y:y + h, x:x + w]
        self.bg = new


def capture_background(cap, n: int, area: np.ndarray) -> Background:
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera stopped while capturing the background")
        p = to_proc(frame)
        frames.append(small_gray(p))
        vis = p.copy()
        banner(vis, f"BACKGROUND {len(frames)}/{n}: keep the square EMPTY", top=True,
               color=(80, 80, 255))
        cv2.imshow(CAM_WIN, vis)
        cv2.waitKey(1)
    return Background(frames, area)


def odd_kernel(frac: float, width: int) -> np.ndarray:
    k = max(3, int(frac * width) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def detect(gray: np.ndarray, bg: Background, area: np.ndarray, area_px: int,
           limb_k, min_area: float):
    """Returns (person or None, mask, lighting_jump)."""
    diff = cv2.absdiff(gray, bg.bg)
    mask = np.where(diff > DIFF_THRESHOLD, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(mask, area)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, limb_k)   # fill holes in the body
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, limb_k)    # drop arms and specks
    if cv2.countNonZero(mask) > MAX_CHANGED_FRAC * area_px:
        return None, mask, True
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, False
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, mask, False
    m = cv2.moments(c)
    return {"centre_px": (m["m10"] / m["m00"], m["m01"] / m["m00"]),
            "box": cv2.boundingRect(c)}, mask, False


def grow_box(box, frac: float, shape) -> tuple[int, int, int, int]:
    x, y, w, h = box
    dx, dy = int(w * frac), int(h * frac)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(shape[1], x + w + dx), min(shape[0], y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


# =========================== tracking ======================================
def region_of(x: float, y: float) -> int:
    """0 = outside the square, 1..9 = cell (1 at the antenna-1 corner)."""
    if not (0.0 <= x <= GX and 0.0 <= y <= GY):
        return 0
    i = min(int(x / (GX / N_CELLS)), N_CELLS - 1)
    j = min(int(y / (GY / N_CELLS)), N_CELLS - 1)
    return j * N_CELLS + i + 1


def region_depth(region: int, x: float, y: float) -> float:
    """How far (m) the point is inside that region."""
    if region == 0:
        return max(-x, x - GX, -y, y - GY)
    cw, ch = GX / N_CELLS, GY / N_CELLS
    i, j = (region - 1) % N_CELLS, (region - 1) // N_CELLS
    return min(x - i * cw, (i + 1) * cw - x, y - j * ch, (j + 1) * ch - y)


class Tracker:
    """Smooths the dot and only changes cell once the person is clearly inside the new one."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.xy = None
        self.t_seen = None
        self.region = None
        self.trail: deque = deque()

    def update(self, xy, t: float) -> None:
        if xy is None:
            if self.t_seen is not None and t - self.t_seen > LOST_RESET_S:
                self.reset()
            return
        xy = np.asarray(xy, float)
        if self.xy is None:
            self.xy = xy.copy()
        else:
            a = 1.0 - math.exp(-(t - self.t_seen) / SMOOTH_S)
            self.xy += a * (xy - self.xy)
        self.t_seen = t
        cand = region_of(*self.xy)
        if self.region is None or (cand != self.region and
                                   region_depth(cand, *self.xy) >= CELL_HYSTERESIS_M):
            self.region = cand
        self.trail.append((t, float(self.xy[0]), float(self.xy[1])))
        while self.trail and t - self.trail[0][0] > TRAIL_S:
            self.trail.popleft()

    def status(self) -> str:
        if self.xy is None:
            return "empty"
        x, y = self.xy
        where = "outside the square" if self.region == 0 else f"CELL {self.region}"
        return f"{where}   x={x:.2f} m  y={y:.2f} m"


# =========================== drawing =======================================
def banner(img, text: str, top: bool, color=(255, 255, 255)) -> None:
    """Readable text on a dark strip across the top or bottom of the image."""
    h = img.shape[0]
    y0, y1 = (0, 38) if top else (h - 32, h)
    img[y0:y1] = (img[y0:y1] * 0.35).astype(np.uint8)
    cv2.putText(img, text, (10, y0 + (27 if top else 22)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7 if top else 0.55, color, 2 if top else 1, cv2.LINE_AA)


def grid_segments_m():
    """(start, end, is_outer) for every cell line of the taped square, in metres."""
    segs = []
    for k in range(N_CELLS + 1):
        x, y = k * GX / N_CELLS, k * GY / N_CELLS
        outer = k in (0, N_CELLS)
        segs.append(((x, 0.0), (x, GY), outer))
        segs.append(((0.0, y), (GX, y), outer))
    return segs


def draw_camera(proc, H, outline, det, tracker: Tracker, lighting_jump: bool, timing: str):
    vis = proc.copy()
    if outline is not None:     # where the detector looks
        cv2.polylines(vis, [outline], True, (150, 150, 150), 1, cv2.LINE_AA)
    if H is not None:
        for a, b, outer in grid_segments_m():
            p = np.int32(np.round(to_image(H, [a, b])))
            cv2.line(vis, tuple(p[0]), tuple(p[1]), (255, 160, 0), 3 if outer else 1)
        for r in range(1, N_CELLS * N_CELLS + 1):
            i, j = (r - 1) % N_CELLS, (r - 1) // N_CELLS
            p = to_image(H, [((i + 0.5) * GX / N_CELLS, (j + 0.5) * GY / N_CELLS)])[0]
            cv2.putText(vis, str(r), (int(p[0]) - 7, int(p[1]) + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 0), 2, cv2.LINE_AA)
    if det is not None:
        x, y, w, h = det["box"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = det["centre_px"]
        cv2.circle(vis, (int(cx), int(cy)), 7, (0, 255, 255), -1)
        cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 0), 1)
    if lighting_jump:
        banner(vis, "big change across the image (lighting?) - press b with the square empty",
               top=True, color=(80, 80, 255))
    elif H is not None:
        banner(vis, tracker.status(), top=True)
    else:
        banner(vis, ("person detected" if det else "empty") + "   (not calibrated: press c)", top=True)
    banner(vis, f"{timing}    q quit  b backgr.  c corners  s/k settings  m mask",
           top=False, color=(200, 200, 200))
    return vis


def map_px(x: float, y: float) -> tuple[int, int]:
    """Metres -> map pixels (y drawn upward)."""
    px = (MAP_MARGIN_M + x) * MAP_PX_PER_M
    py = (MAP_MARGIN_M + GY - y) * MAP_PX_PER_M
    return int(round(np.clip(px, -10000, 10000))), int(round(np.clip(py, -10000, 10000)))


def draw_map(H, tracker: Tracker) -> np.ndarray:
    w = int((GX + 2 * MAP_MARGIN_M) * MAP_PX_PER_M)
    h = int((GY + 2 * MAP_MARGIN_M) * MAP_PX_PER_M)
    img = np.full((h + 44, w, 3), 35, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cw, ch = GX / N_CELLS, GY / N_CELLS
    for r in range(1, N_CELLS * N_CELLS + 1):
        i, j = (r - 1) % N_CELLS, (r - 1) // N_CELLS
        p0, p1 = map_px(i * cw, (j + 1) * ch), map_px((i + 1) * cw, j * ch)
        active = tracker.xy is not None and tracker.region == r
        cv2.rectangle(img, p0, p1, (40, 130, 40) if active else (70, 70, 70), -1)
        cv2.rectangle(img, p0, p1, (200, 200, 200), 1)
        cv2.putText(img, str(r), (p0[0] + 6, p0[1] + 22), font, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.rectangle(img, map_px(0, GY), map_px(GX, 0), (230, 230, 230), 2)

    for name, (ax, ay) in ANTENNAS_M.items():
        p = map_px(ax, ay)
        cv2.drawMarker(img, p, (0, 140, 255), cv2.MARKER_TRIANGLE_UP, 18, 2)
        cv2.putText(img, name, (p[0] - 18, p[1] + 22), font, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

    if len(tracker.trail) > 1:
        pts = np.int32([map_px(x, y) for _, x, y in tracker.trail])
        cv2.polylines(img, [pts], False, (200, 150, 60), 2, cv2.LINE_AA)
    if tracker.xy is not None:
        p = map_px(*tracker.xy)
        cv2.circle(img, p, 11, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 11, (0, 0, 0), 2, cv2.LINE_AA)

    status = tracker.status() if H is not None else "not calibrated: press c in the camera window"
    cv2.putText(img, status, (10, h + 29), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# =========================== main ==========================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Live ceiling-camera tracking with a top-down map")
    ap.add_argument("--camera", default=CAMERA, help="camera index, or a video file")
    ap.add_argument("--width", type=int, help="ask the camera for this frame width")
    ap.add_argument("--height", type=int, help="ask the camera for this frame height")
    ap.add_argument("--probe", action="store_true", help="snapshot each resolution, then exit")
    ap.add_argument("--calibrate", action="store_true", help="click the tape corners again")
    ap.add_argument("--out", default=OUTPUT_DIR, help="folder for calibration and snapshots")
    ap.add_argument("--no-camera-settings", action="store_true",
                    help="don't apply the camera settings saved with k")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe(args.camera, out)
        return 0

    size = (args.width, args.height) if args.width and args.height else REQUEST_SIZE
    print("opening camera ...")
    settings = None
    sp = settings_path(out, args.camera)
    if args.camera.isdigit() and not args.no_camera_settings and sp.exists():
        settings = json.loads(sp.read_text())["settings"]
    raw = open_camera(args.camera, size, warmup_s=WARMUP_S, settings=settings)
    is_file = os.path.isfile(args.camera)
    fmt = fourcc_of(raw)
    cap = Grabber(raw, threaded=not is_file)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("camera opened but gave no frame")
    cam_size = (first.shape[1], first.shape[0])
    p0 = to_proc(first)
    proc_size = (p0.shape[1], p0.shape[0])
    print(f"camera {args.camera}: {cam_size[0]}x{cam_size[1]} {fmt}  "
          f"(shown at {proc_size[0]}x{proc_size[1]}, searched at {DETECT_WIDTH} px wide)")
    if not is_file and fmt != "MJPG" and cam_size[0] >= 1280:
        print("  note: the camera is not sending compressed (MJPG) frames; at this size many "
              "USB cameras then manage only 5-10 fps. If 'camera fps' is low, try "
              "--width 1280 --height 720 or --width 640 --height 480 (check it is not cropped).")

    cv2.namedWindow(CAM_WIN, cv2.WINDOW_NORMAL)      # resizable: drag to make it bigger
    cv2.resizeWindow(CAM_WIN, *proc_size)
    cv2.namedWindow(MAP_WIN, cv2.WINDOW_AUTOSIZE)

    corners = None if args.calibrate else load_calibration(out, args.camera, cam_size)
    if corners is None:
        corners = run_calibration(cap, out, args.camera, cam_size)
    H = homography(corners, proc_size) if corners else None
    area, outline = detection_area(H, proc_size)

    limb_k = odd_kernel(LIMB_KERNEL_FRAC, DETECT_WIDTH)
    min_area = MIN_AREA_FRAC * area.shape[0] * area.shape[1]
    up = proc_size[0] / DETECT_WIDTH        # detection pixels -> display pixels
    bg = capture_background(cap, BACKGROUND_FRAMES, area)
    tracker = Tracker()
    show_mask = False
    file_dt = 1.0 / (raw.get(cv2.CAP_PROP_FPS) or 30.0)
    fps, proc_ms, t_prev = 0.0, 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no more frames" if is_file else "camera read failed")
            break
        t = time.perf_counter()
        if is_file:                                   # play test videos at real speed
            time.sleep(max(0.0, file_dt - (t - t_prev)))
            t = time.perf_counter()
        dt = max(t - t_prev, 1e-6)
        fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
        t_prev = t

        t_work = time.perf_counter()
        proc = to_proc(frame)
        gray = bg.compensate(small_gray(proc))
        det, mask, jump = detect(gray, bg, area, cv2.countNonZero(area), limb_k, min_area)
        if not jump:
            bg.update(gray, grow_box(det["box"], 0.3, gray.shape) if det else None, dt)
        if det is not None:                  # back to display pixels
            cx, cy = det["centre_px"]
            x, y, w, h = det["box"]
            det = {"centre_px": (cx * up, cy * up),
                   "box": (int(x * up), int(y * up), int(w * up), int(h * up))}
        xy = to_floor(H, [det["centre_px"]])[0] if (det is not None and H is not None) else None
        tracker.update(xy, t)
        work = (time.perf_counter() - t_work) * 1e3
        proc_ms = 0.9 * proc_ms + 0.1 * work if proc_ms else work
        timing = (f"camera {cap.camera_fps():4.1f} fps | shown {fps:4.1f} fps | "
                  f"work {proc_ms:3.0f} ms")

        cv2.imshow(CAM_WIN, draw_camera(proc, H, outline, det, tracker, jump, timing))
        cv2.imshow(MAP_WIN, draw_map(H, tracker))
        if show_mask:
            cv2.imshow(MASK_WIN, mask)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("b"):
            bg = capture_background(cap, BACKGROUND_FRAMES, area)
            tracker.reset()
        elif k == ord("c"):
            new = run_calibration(cap, out, args.camera, cam_size)
            if new is not None:
                H = homography(new, proc_size)
                area, outline = detection_area(H, proc_size)
                bg.set_area(area)
            tracker.reset()
        elif k == ord("s"):
            if not raw.set(cv2.CAP_PROP_SETTINGS, 1):
                print("this camera/driver has no settings dialog")
        elif k == ord("k"):
            if is_file:
                print("k only works with a live camera")
            else:
                save_camera_settings(raw, out, args.camera)
        elif k == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow(MASK_WIN)

    cap.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
