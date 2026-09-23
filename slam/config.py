"""
slam.config -- single source of truth for the SLAM RF-localization project.

Everything else in the ``slam`` package imports its constants from here
(``from slam import config``).  This module is pure standard library: no
numpy, no pyvisa, so it is safe to import anywhere (including on a laptop
that has never seen the VNA).

Sections
--------
1. VNA connection and sweep settings
2. On-disk layout (data / figures directories, CSV column names)
3. Floor grid geometry and label -> (x, y) mapping
4. Antenna placement
5. Room description (used by the digital twin)

Values tagged ``# TODO_MEASURE`` are placeholders that the team must confirm
with a tape measure / by reading equipment labels in the lab.  Everything
else is a deliberate design choice agreed in docs/CONTRACT.md.

Coordinate conventions (used by every value below)
--------------------------------------------------
* Origin (0, 0) is one corner of the blue-taped floor square.
* +x runs "to the right" along one taped edge, +y runs "away from the door"
  along the perpendicular edge, +z is up from the floor.  All lengths in
  metres.
* Azimuth is measured in the floor plane, in degrees counter-clockwise from
  the +x axis (0 deg = looking along +x, 90 deg = looking along +y).
* Tilt is the elevation of the antenna boresight in degrees; negative means
  pointing DOWN toward the floor.

Run ``python -m slam.config`` to print a summary as a quick sanity check.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# ============================================================================
# 1. VNA connection and sweep settings
# ============================================================================

# VISA resource string of the Keysight E5071C ENA on the lab network.
# The pyvisa-py backend (in requirements.txt) talks TCPIP without NI-VISA.
VNA_ADDRESS = "TCPIP0::192.168.1.10::inst0::INSTR"

# Substring that must appear in the *IDN? reply; used to refuse the wrong box.
VNA_MODEL = "E5071C"

# VISA I/O timeout in milliseconds.  Generous because a 201-point sweep with
# 1 kHz IF bandwidth and 4x averaging can take several seconds.
VNA_TIMEOUT_MS = 60000

# Sweep span: the 902-928 MHz ISM band the L-com panel antennas are built for.
FREQ_START_HZ = 902e6        # Hz
FREQ_STOP_HZ = 928e6         # Hz

# Number of frequency points per sweep.  201 gives ~130 kHz spacing, fine
# enough to resolve multipath ripple across the 26 MHz span without making
# each sweep painfully slow.
NUM_POINTS = 201

# Receiver IF bandwidth in Hz.  Narrower = lower noise floor but slower
# sweeps.  1 kHz is a reasonable trade-off for a static-scene fingerprint.
IF_BANDWIDTH_HZ = 1000

# Sweep-to-sweep averaging on the instrument.  0 or 1 = averaging off.
# Person-induced changes are only fractions of a dB, so averaging matters.
AVERAGING_COUNT = 4

# Source power in dBm.  None = leave whatever the instrument is set to.
POWER_DBM = 0.0

# S-parameters captured in ONE full 2-port sweep.  The order is fixed: trace 1
# is S11, trace 2 is S21, trace 3 is S22, and the CSV columns below follow it.
S_PARAMS = ("S11", "S21", "S22")

# ============================================================================
# 2. On-disk layout
# ============================================================================

# Repo root = the directory that contains the ``slam`` package.  Resolved at
# import time so the defaults work no matter which directory you run from.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where collect.py / digital_twin.py write sessions (<DATA_DIR>/<session>/...)
# and where analyze_sensitivity.py reads them from.  Override with the
# SLAM_DATA_DIR environment variable (absolute or relative to the cwd).
DATA_DIR = Path(os.environ.get("SLAM_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()

# Where analysis PNGs land (<FIGURES_DIR>/<session>/...).  Override with
# SLAM_FIGURES_DIR.
FIGURES_DIR = Path(os.environ.get("SLAM_FIGURES_DIR", REPO_ROOT / "figures")).expanduser().resolve()

# Per-session index file: one row per saved sweep.
MANIFEST_NAME = "manifest.csv"

# Column order of every per-sweep CSV.  Complex S-parameters are stored as
# separate real / imaginary columns (linear, not dB) so nothing is lost.
SWEEP_CSV_COLUMNS = [
    "Frequency_Hz",
    "S11_re", "S11_im",
    "S21_re", "S21_im",
    "S22_re", "S22_im",
]

# Column order of manifest.csv.
#   timestamp     ISO 8601 local time of the sweep
#   session       session name (also the sub-directory name)
#   label         grid label ("C3") or special label ("empty", "center", ...)
#   x_m, y_m      person position in metres, blank for empty/outside
#   person        who was standing there (body size matters for RF)
#   notes         free text
#   filename      the per-sweep CSV, relative to the session directory
#   sweep_time_s  wall-clock seconds for trigger + read-out
MANIFEST_COLUMNS = [
    "timestamp", "session", "label", "x_m", "y_m",
    "person", "notes", "filename", "sweep_time_s",
]

# ============================================================================
# 3. Floor grid geometry
# ============================================================================

# Size of the taped square (x extent, y extent) in metres.
GRID_SIZE_M = (4.0, 4.0)     # TODO_MEASURE: confirm the tape square is 4.0 x 4.0 m

# Spacing between standing positions in metres.  With a 4 m square this
# gives a 5 x 5 lattice including both edges.
GRID_SPACING_M = 1.0


def _build_grid_points(size_m: tuple[float, float],
                       spacing_m: float) -> dict[str, tuple[float, float]]:
    """Return {label: (x, y)} for a lattice covering the square.

    Rows are lettered A, B, C, ... along +y (A is the y = 0 edge) and
    columns are numbered 1, 2, 3, ... along +x (1 is the x = 0 edge), so
    A1 = (0, 0), A2 = (1, 0), B1 = (0, 1), ..., E5 = (4, 4) for the default
    4 m / 1 m grid.  Labels are row-major so iteration order is A1..A5, B1..
    """
    n_cols = int(round(size_m[0] / spacing_m)) + 1
    n_rows = int(round(size_m[1] / spacing_m)) + 1
    if n_rows > 26:
        raise ValueError("more than 26 rows: extend the row-letter scheme")
    points: dict[str, tuple[float, float]] = {}
    for r in range(n_rows):
        row_letter = chr(ord("A") + r)
        for c in range(n_cols):
            points[f"{row_letter}{c + 1}"] = (round(c * spacing_m, 3),
                                              round(r * spacing_m, 3))
    return points


# Standing positions inside the square, label -> (x_m, y_m).
GRID_POINTS: dict[str, tuple[float, float]] = _build_grid_points(GRID_SIZE_M, GRID_SPACING_M)

# Distance (metres) from an antenna's corner, measured along the square's
# diagonal, to the "near_antN" special position.
NEAR_ANT_OFFSET_M = 0.5

# Unit vector along the square's diagonal (used for near_ant positions).
_DIAG = 1.0 / math.sqrt(2.0)

# Non-lattice labels.  (None, None) means "no meaningful position".
#   empty      nobody in the room (the reference for complex-difference)
#   center     geometric centre of the square
#   outside    a person present but outside the taped square (negative case)
#   near_ant1  0.5 m in from ant1's corner (0, 0) along the diagonal
#   near_ant2  0.5 m in from ant2's corner (4, 4) along the diagonal
SPECIAL_LABELS: dict[str, tuple[float | None, float | None]] = {
    "empty": (None, None),
    "center": (GRID_SIZE_M[0] / 2.0, GRID_SIZE_M[1] / 2.0),
    "outside": (None, None),
    "near_ant1": (round(NEAR_ANT_OFFSET_M * _DIAG, 3),
                  round(NEAR_ANT_OFFSET_M * _DIAG, 3)),
    "near_ant2": (round(GRID_SIZE_M[0] - NEAR_ANT_OFFSET_M * _DIAG, 3),
                  round(GRID_SIZE_M[1] - NEAR_ANT_OFFSET_M * _DIAG, 3)),
}


def label_to_xy(label: str) -> tuple[float | None, float | None]:
    """Map a position label to (x_m, y_m).

    Looks in GRID_POINTS first, then SPECIAL_LABELS, and raises ValueError
    for anything else.  Matching is exact after stripping whitespace, with a
    case-insensitive fallback ("c3" -> "C3", "Empty" -> "empty") so typos at
    the collection prompt do not abort a session.
    """
    key = label.strip()
    if key in GRID_POINTS:
        return GRID_POINTS[key]
    if key in SPECIAL_LABELS:
        return SPECIAL_LABELS[key]
    if key.upper() in GRID_POINTS:
        return GRID_POINTS[key.upper()]
    if key.lower() in SPECIAL_LABELS:
        return SPECIAL_LABELS[key.lower()]
    known = ", ".join(list(GRID_POINTS) + list(SPECIAL_LABELS))
    raise ValueError(f"unknown position label {label!r}; known labels: {known}")


# ============================================================================
# 4. Antenna placement
# ============================================================================

# Both antennas sit on tripods on top of metal filing cabinets just OUTSIDE
# the taped square, at diagonally opposite corners, tilted down toward the
# square.  ant1 is by the origin corner, ant2 by the far (4, 4) corner.
#
# Keys per antenna:
#   xyz_m         phase-centre position in the grid frame (metres)
#   tilt_deg      boresight elevation, negative = down
#   azimuth_deg   boresight heading in the floor plane (deg CCW from +x);
#                 computed so each antenna looks at the square's centre
#   polarization  "vertical" / "horizontal" as mounted
#   model         antenna model string (read it off the label)
#   gain_dbi      nominal boresight gain from the datasheet
#   cable_m       coax length from VNA port to antenna (affects loss/phase)
#   vna_port      which VNA port it is connected to (1 -> S11, 2 -> S22)

_SQUARE_CENTER = (GRID_SIZE_M[0] / 2.0, GRID_SIZE_M[1] / 2.0)


def _azimuth_to_center_deg(xyz_m: tuple[float, float, float]) -> float:
    """Azimuth (deg CCW from +x) from a point to the centre of the square."""
    dx = _SQUARE_CENTER[0] - xyz_m[0]
    dy = _SQUARE_CENTER[1] - xyz_m[1]
    return round(math.degrees(math.atan2(dy, dx)) % 360.0, 1)


_ANT1_XYZ = (-0.5, -0.5, 2.7)   # TODO_MEASURE: tape from the square corner to the antenna, and floor to phase centre
_ANT2_XYZ = (4.5, 4.5, 2.7)     # TODO_MEASURE: same for ant2

ANTENNAS: dict[str, dict] = {
    "ant1": {
        "xyz_m": _ANT1_XYZ,
        "tilt_deg": -30.0,                       # TODO_MEASURE: inclinometer / phone level on the panel
        "azimuth_deg": _azimuth_to_center_deg(_ANT1_XYZ),   # 45.0 for the default placement
        "polarization": "vertical",              # TODO_MEASURE: check how the panel is mounted
        "model": "L-com 900 MHz directional panel (TODO_MEASURE: read model from label)",
        "gain_dbi": 8.0,                         # TODO_MEASURE: from the datasheet for the actual model
        "cable_m": 3.0,                          # TODO_MEASURE: coax length VNA port 1 -> ant1
        "vna_port": 1,
    },
    "ant2": {
        "xyz_m": _ANT2_XYZ,
        "tilt_deg": -30.0,                       # TODO_MEASURE
        "azimuth_deg": _azimuth_to_center_deg(_ANT2_XYZ),   # 225.0 for the default placement
        "polarization": "vertical",              # TODO_MEASURE
        "model": "L-com 900 MHz directional panel (TODO_MEASURE: read model from label)",
        "gain_dbi": 8.0,                         # TODO_MEASURE
        "cable_m": 3.0,                          # TODO_MEASURE: coax length VNA port 2 -> ant2
        "vna_port": 2,
    },
}

# ============================================================================
# 5. Room description (consumed by slam/sim/digital_twin.py)
# ============================================================================

# Rough room envelope.  The digital twin uses these for image-method
# reflections, so even +/-0.5 m accuracy is a big improvement on guesses.
ROOM: dict = {
    # (length along x, width along y, ceiling height) in metres.
    "size_m": (8.0, 6.0, 2.8),                   # TODO_MEASURE: laser measure the room
    # Position of the grid origin (taped corner) inside the room, in room
    # coordinates whose origin is the room corner nearest the grid origin.
    # Lets the simulator know how far each wall is from the square.
    "grid_origin_in_room_m": (2.0, 1.0, 0.0),    # TODO_MEASURE
    "wall_material": "drywall",                  # TODO_MEASURE: confirm (drywall vs cinder block changes reflectivity)
    "ceiling": "drop ceiling (acoustic tile over metal grid)",
    "floor": "commercial carpet tile over concrete slab",   # TODO_MEASURE
    "notes": (
        "Two metal filing cabinets (antenna stands) at diagonally opposite "
        "corners of the taped square, ~1.5 m tall, strong reflectors. "
        "Lab benches with instruments along one wall. Door on the y = 0 side "
        "of the room. Update this text when the layout changes so the "
        "session_info.json snapshot stays meaningful."
    ),
}


# ============================================================================
# Sanity-check entry point: python -m slam.config
# ============================================================================

def _summary() -> str:
    lines = [
        "SLAM configuration summary",
        "==========================",
        f"VNA            : {VNA_MODEL} at {VNA_ADDRESS} (timeout {VNA_TIMEOUT_MS} ms)",
        f"Sweep          : {FREQ_START_HZ / 1e6:.1f}-{FREQ_STOP_HZ / 1e6:.1f} MHz, "
        f"{NUM_POINTS} pts, IF BW {IF_BANDWIDTH_HZ} Hz, avg {AVERAGING_COUNT}, "
        f"power {POWER_DBM} dBm, traces {', '.join(S_PARAMS)}",
        f"Data dir       : {DATA_DIR}",
        f"Figures dir    : {FIGURES_DIR}",
        f"Grid           : {GRID_SIZE_M[0]} x {GRID_SIZE_M[1]} m, {GRID_SPACING_M} m spacing, "
        f"{len(GRID_POINTS)} lattice points ({next(iter(GRID_POINTS))} .. {list(GRID_POINTS)[-1]})",
        "Special labels : " + ", ".join(f"{k}={v}" for k, v in SPECIAL_LABELS.items()),
        "Antennas       :",
    ]
    for name, a in ANTENNAS.items():
        lines.append(
            f"  {name}: xyz={a['xyz_m']} m, tilt={a['tilt_deg']} deg, "
            f"az={a['azimuth_deg']} deg, {a['polarization']}, {a['gain_dbi']} dBi, "
            f"cable {a['cable_m']} m, VNA port {a['vna_port']}"
        )
    lines.append(f"Room           : {ROOM['size_m']} m, walls {ROOM['wall_material']}")
    lines.append("Remember to replace every TODO_MEASURE placeholder before publishing data.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_summary())
