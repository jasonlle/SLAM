# SLAM P0 module contract (shared by all agents — do not deviate)

Repo root: /home/claude/SLAM. New code lives in a top-level package `slam/`
(with `slam/__init__.py`). Existing folders (`scpiworkbench/`, `camera_scpi/`,
`themachinelearns/`, `termcolor/`) are legacy and must NOT be modified.
Everything runs as `python -m slam.<module>` from the repo root. Python 3.10+.
Use pathlib, no hard-coded Windows paths. No git commits.

Hardware context: Keysight E5071C ENA at TCPIP0::192.168.1.10::inst0::INSTR,
two L-com 900 MHz directional panel antennas on tripods atop metal cabinets at
two corners of a ~4 x 4 m blue-taped floor square, ~2.5-3 m high, tilted down
toward the square. Band 902-928 MHz. Person-induced changes are small (a few
dB / fraction of dB), so complex-difference-from-empty is the key quantity.

## slam/config.py  (Agent 1)
Plain module-level constants (importable as `from slam import config`):

    VNA_ADDRESS = "TCPIP0::192.168.1.10::inst0::INSTR"
    VNA_MODEL = "E5071C"
    VNA_TIMEOUT_MS = 60000
    FREQ_START_HZ = 902e6
    FREQ_STOP_HZ = 928e6
    NUM_POINTS = 201
    IF_BANDWIDTH_HZ = 1000
    AVERAGING_COUNT = 4          # 0 or 1 = averaging off
    POWER_DBM = 0.0              # None = leave instrument setting
    S_PARAMS = ("S11", "S21", "S22")   # one full 2-port sweep; trace order fixed
    DATA_DIR: pathlib.Path        # default <repo_root>/data, override via env SLAM_DATA_DIR
    FIGURES_DIR: pathlib.Path     # default <repo_root>/figures
    MANIFEST_NAME = "manifest.csv"
    SWEEP_CSV_COLUMNS = ["Frequency_Hz", "S11_re", "S11_im", "S21_re", "S21_im", "S22_re", "S22_im"]
    MANIFEST_COLUMNS = ["timestamp", "session", "label", "x_m", "y_m", "person", "notes", "filename", "sweep_time_s"]
    GRID_SIZE_M = (4.0, 4.0)      # taped square, origin (0,0) at one corner, x to the right, y away from the door
    GRID_SPACING_M = 1.0
    GRID_POINTS: dict[str, tuple[float, float]]   # e.g. "A1": (0.0, 0.0) ... "E5": (4.0, 4.0); rows A-E along y, cols 1-5 along x
    SPECIAL_LABELS: dict[str, tuple[float|None, float|None]]  # "empty": (None, None), "center": (2.0, 2.0), "outside": (None, None), "near_ant1": (...), "near_ant2": (...)
    ANTENNAS: dict  # {"ant1": {"xyz_m": (x, y, z), "tilt_deg": -30.0, "azimuth_deg": ..., "polarization": "vertical", "model": "L-com 900 MHz panel (TODO: read label)", "gain_dbi": 8.0, "cable_m": ...}, "ant2": {...}}
                    # values are placeholders clearly marked TODO_MEASURE, with ant1 at one corner and ant2 at the diagonally opposite corner of the square, z ~2.7 m
    ROOM = {"size_m": (approx L, W, H) placeholders, "wall_material": "drywall", "notes": "..."}
    def label_to_xy(label: str) -> tuple[float|None, float|None]   # GRID_POINTS then SPECIAL_LABELS else raises ValueError

## slam/vna.py  (Agent 2)
    @dataclass
    class Sweep:
        freq_hz: np.ndarray                 # shape (N,)
        s: dict[str, np.ndarray]            # {"S11": complex (N,), "S21": ..., "S22": ...}
        timestamp: str                      # ISO 8601 local, e.g. "2026-09-27T14:03:22.123456"
        sweep_time_s: float                 # wall time of trigger+readout

    class VNA:
        def __init__(self, address=config.VNA_ADDRESS, timeout_ms=config.VNA_TIMEOUT_MS)
        def connect(self) -> str              # opens pyvisa, checks *IDN? contains config.VNA_MODEL, returns IDN
        def configure(self, f_start=..., f_stop=..., n_points=..., if_bw=..., averaging=..., power_dbm=..., s_params=config.S_PARAMS) -> None
            # sets up channel 1 with len(s_params) traces (CALC1:PAR:COUN, CALC1:PARn:DEF Sxx), INIT1:CONT OFF, FORM:DATA ASC
        def sweep(self) -> Sweep              # INIT1:IMM, *OPC?, then for each trace: select, CALC1:DATA:SDAT? -> complex; SENS1:FREQ:DATA? once
        def measure_sweep_time(self, n=5) -> float
        def close(self)
        __enter__/__exit__
    class MockVNA:   # identical public interface, no pyvisa; returns synthetic but physically plausible data
        # S11/S22 ~ -20 dB with slow ripple; S21 ~ -45 dB with multipath ripple; small Gaussian noise (~-70 dB);
        # optional attribute `perturbation: complex | np.ndarray` added to S21 so tests can fake a person
    def sweep_to_csv(sweep: Sweep, path: Path) -> None     # columns exactly config.SWEEP_CSV_COLUMNS
    def load_sweep_csv(path: Path) -> Sweep               # inverse; timestamp/sweep_time may be parsed from filename or left ""
    def open_vna(mock: bool = False) -> VNA | MockVNA     # convenience factory
Reuse ideas from scpiworkbench/support_functions.py (query_text/query_csv_numbers drain logic) but
vna.py must be self-contained (copy the needed helpers in; do not import from the legacy folders).

## slam/collect.py  (Agent 3)
CLI:  python -m slam.collect --session <name> --label <label> [--x X --y Y] --person <name> [--n 10]
                             [--interval 0.0] [--notes "..."] [--mock] [--data-dir PATH]
 - label in config.GRID_POINTS / SPECIAL_LABELS auto-fills x,y (explicit --x/--y override).
 - Interactive mode when --label omitted: loop prompting "label [n] [notes]" until 'q'; ENTER = repeat last.
 - Files: <DATA_DIR>/<session>/<label>_<YYYYmmdd_HHMMSS_ffffff>.csv via vna.sweep_to_csv.
 - Appends one row per sweep to <DATA_DIR>/<session>/manifest.csv (config.MANIFEST_COLUMNS, header if new).
 - Also writes <DATA_DIR>/<session>/session_info.json once: VNA IDN, sweep settings, config.ANTENNAS, config.ROOM, start time.
 - Countdown (3 s, configurable --countdown) before first sweep of each label so the person can settle.
 - Ctrl-C safe; prints sweep time and running count. Uses vna.open_vna(mock).

## slam/analyze_sensitivity.py  (Agent 4)
CLI:  python -m slam.analyze_sensitivity --session <name> [--data-dir PATH] [--out PATH] [--reference-label empty]
 - Loads manifest + sweeps via vna.load_sweep_csv.
 - Reference = mean complex sweep of reference-label sweeps, per S-param. Empty spread = per-frequency std of |S - ref| over reference sweeps (complex residual), also expressed as a scalar RMS.
 - For every other label and each of S11,S21,S22: delta = S - ref (complex, per sweep); report mean |delta| (linear and dB), and ratio mean|delta| / empty-RMS ("detectability ratio"). Also plain dB-magnitude change for S21.
 - Figures (matplotlib, Agg backend, PNG to --out, default config.FIGURES_DIR/<session>/):
   1. |S21| dB vs frequency, one line per label (mean) with reference sweeps as thin grey lines.
   2. |delta| dB vs frequency per label, one subplot per S-param.
   3. Bar chart: detectability ratio per label per S-param.
   4. Scatter of mean delta in the complex plane per label (S21), points = individual sweeps.
 - Prints a summary table and writes summary.csv (label, s_param, n, mean_abs_delta, mean_abs_delta_db, empty_rms, ratio, mean_db_change).
 - Must run headless and must not require the VNA.

## slam/sim/digital_twin.py + docs/digital_twin.md  (Agent 5)
Simulator generates sessions in exactly the collect.py format (same CSV columns, manifest, session_info.json),
so analyze_sensitivity and future training code run unchanged on simulated data.
CLI: python -m slam.sim.digital_twin --session sim_<name> [--labels all|empty,center,...] [--n 10] [--seed 0] [--data-dir PATH]
Physics (baseline, all in numpy, no external EM package required): free-space direct path with antenna gain pattern
(cos^n panel model, tilt/azimuth from config.ANTENNAS), image-method reflections from floor/ceiling/4 walls (config.ROOM,
Fresnel-ish constant reflection coefficient), person = vertical dielectric cylinder / point scatterer with RCS ~1 m^2
and small random per-sweep jitter (breathing/sway), S11/S22 = antenna mismatch + back-scatter, S21 = coherent sum,
plus thermal noise at ~-70 dB and slow drift. Keep it modular so a better EM engine can replace the channel model.
docs/digital_twin.md: research write-up on options (Sionna RT, MATLAB ray tracing, Wireless InSite, CST/HFSS, FDTD/gprMax,
MoM), what fidelity is needed for this project, validation plan against real sessions, and how sim data feeds the paper.

## Reviewer (Agent 6)
pip install -r requirements.txt; pyflakes all new files; run collect --mock, digital_twin, analyze_sensitivity on both;
open the PNGs; verify formats match this contract; fix bugs directly; report what changed.
