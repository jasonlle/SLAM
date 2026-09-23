"""Keysight E5071C ENA driver for the SLAM antenna-localization project.

This module is self-contained (no imports from the legacy ``scpiworkbench``
folder) and exposes the interface fixed in ``docs/CONTRACT.md``:

* :class:`Sweep`      -- one full 2-port sweep (frequency axis + complex S-params)
* :class:`VNA`        -- real instrument over pyvisa (imported lazily in ``connect``)
* :class:`MockVNA`    -- same public interface, synthetic but physically plausible data
* :func:`sweep_to_csv` / :func:`load_sweep_csv` -- on-disk format (``config.SWEEP_CSV_COLUMNS``)
* :func:`open_vna`    -- convenience factory returning a connected + configured instrument

The SCPI sequences are the ones lab-tested in ``scpiworkbench/datagatheringscript.py``
(``INIT1:CONT OFF``, ``CALC1:PAR:COUN``, ``CALC1:PARn:DEF Sxx``, ``CALC1:PARn:SEL``,
``INIT1:IMM`` + ``*OPC?``, ``SENS1:FREQ:DATA?``, ``CALC1:DATA:SDAT?``) together with the
"drain quickly-available bytes" read logic that avoids ``-420 Query UNTERMINATED``
errors on this instrument.

Usage::

    from pathlib import Path
    from slam import vna

    with vna.open_vna(mock=False) as inst:        # connect() + configure() with config defaults
        print(inst.idn)
        print("sweep time", inst.measure_sweep_time(n=3), "s")
        sw = inst.sweep()                         # -> Sweep
        vna.sweep_to_csv(sw, Path("data/test/empty_20260927_140322_123456.csv"))

    sw2 = vna.load_sweep_csv(Path("data/test/empty_20260927_140322_123456.csv"))
    import numpy as np
    print(20 * np.log10(np.abs(sw2.s["S21"])).mean())

    # Offline development / tests:
    m = vna.MockVNA(seed=0)
    m.connect(); m.configure()
    m.perturbation = 1e-3 * np.exp(1j * 0.7)      # fake a person in the S21 path
    sw = m.sweep()

Command line::

    python -m slam.vna --mock        # one mock sweep, prints shapes and mean |S21| dB
    python -m slam.vna               # real instrument: prints *IDN? and measured sweep time
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from slam import config

__all__ = [
    "Sweep",
    "VNA",
    "MockVNA",
    "sweep_to_csv",
    "load_sweep_csv",
    "open_vna",
]

_C0 = 299_792_458.0  # speed of light, m/s

# Filename convention written by slam.collect: <label>_YYYYmmdd_HHMMSS_ffffff.csv
_FILENAME_TS_RE = re.compile(r"_(\d{8})_(\d{6})_(\d{6})\.csv$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #
@dataclass
class Sweep:
    """One VNA sweep: frequency axis plus complex S-parameters.

    ``s`` maps S-parameter names (``"S11"``, ``"S21"``, ``"S22"``) to complex
    arrays of shape ``(N,)`` in linear units (not dB). ``timestamp`` is ISO 8601
    local time with microseconds; ``sweep_time_s`` is the wall time of
    trigger + readout (``nan`` when unknown, e.g. when loaded from CSV).
    """

    freq_hz: np.ndarray
    s: dict[str, np.ndarray] = field(default_factory=dict)
    timestamp: str = ""
    sweep_time_s: float = float("nan")

    @property
    def n_points(self) -> int:
        return int(self.freq_hz.shape[0])

    def mag_db(self, s_param: str = "S21") -> np.ndarray:
        """|S| in dB for one S-parameter (floor at -300 dB to avoid log(0))."""
        mag = np.abs(self.s[s_param])
        return 20.0 * np.log10(np.maximum(mag, 1e-15))


# --------------------------------------------------------------------------- #
# Low-level SCPI helpers (copied from the lab-tested legacy scripts)
# --------------------------------------------------------------------------- #
def _drain(inst: Any, data: bytes, drain_timeout_ms: int = 200) -> bytes:
    """Append any bytes that arrive quickly after the first chunk.

    The E5071C often delivers a long ASCII response in several chunks; leaving
    bytes behind in the buffer causes ``-420 Query UNTERMINATED`` / garbled next
    replies. A short temporary timeout is used to detect "no more data".
    """
    old_timeout = inst.timeout
    try:
        inst.timeout = drain_timeout_ms
        while True:
            try:
                more = inst.read_raw()
            except Exception:
                break
            if not more:
                break
            data += more
    finally:
        inst.timeout = old_timeout
    return data


def _query_text(inst: Any, cmd: str) -> str:
    """Send a query and return the decoded, stripped text reply."""
    inst.write(cmd)
    data = inst.read_raw()
    data = _drain(inst, data)
    return data.decode("ascii", errors="replace").strip()


def _query_csv_numbers(inst: Any, cmd: str) -> np.ndarray:
    """Send a query whose reply is comma-separated ASCII floats; return float array."""
    inst.write(cmd)
    data = inst.read_raw()
    data = _drain(inst, data)
    text = data.decode("ascii", errors="replace").strip()
    return np.array([float(x) for x in text.split(",") if x.strip()], dtype=float)


def _interleaved_to_complex(vals: np.ndarray) -> np.ndarray:
    """[re0, im0, re1, im1, ...] -> complex array."""
    if vals.size % 2 != 0:
        raise RuntimeError(f"Interleaved re/im data has odd length {vals.size}")
    return vals[0::2] + 1j * vals[1::2]


# --------------------------------------------------------------------------- #
# Real instrument
# --------------------------------------------------------------------------- #
class VNA:
    """Keysight E5071C ENA over pyvisa (TCPIP/SOCKET/USB VISA address).

    pyvisa is imported inside :meth:`connect` so this module (and
    :class:`MockVNA`) can be imported on machines without VISA installed.
    """

    def __init__(
        self,
        address: str = config.VNA_ADDRESS,
        timeout_ms: int = config.VNA_TIMEOUT_MS,
    ) -> None:
        self.address = address
        self.timeout_ms = int(timeout_ms)
        self.idn: str = ""
        self._rm: Any = None
        self._inst: Any = None
        # Populated by configure(); also used by sweep() for validation.
        self.settings: dict[str, Any] = {}
        self.s_params: tuple[str, ...] = tuple(config.S_PARAMS)
        self._averaging_on: bool = False

    # -- lifecycle ---------------------------------------------------------- #
    def connect(self) -> str:
        """Open the VISA session, verify ``*IDN?`` contains ``config.VNA_MODEL``.

        Returns the IDN string. Raises RuntimeError on model mismatch.
        """
        import pyvisa  # lazy: keep MockVNA usable without pyvisa

        if self._inst is not None:      # already connected: do not leak a second VISA session
            return self.idn
        self._rm = pyvisa.ResourceManager()
        inst = self._rm.open_resource(self.address)
        inst.write_termination = "\n"
        inst.read_termination = None
        inst.timeout = self.timeout_ms
        self._inst = inst

        inst.write("*CLS")
        self.idn = _query_text(inst, "*IDN?")
        if config.VNA_MODEL not in self.idn:
            self.close()
            raise RuntimeError(
                f"Connected to {self.address!r} but *IDN? = {self.idn!r} does not "
                f"contain expected model {config.VNA_MODEL!r}"
            )
        return self.idn

    def close(self) -> None:
        for obj in (self._inst, self._rm):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._inst = None
        self._rm = None

    def __enter__(self) -> "VNA":
        if self._inst is None:
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- helpers ------------------------------------------------------------ #
    @property
    def inst(self) -> Any:
        if self._inst is None:
            raise RuntimeError("VNA is not connected; call connect() first")
        return self._inst

    def write(self, cmd: str) -> None:
        self.inst.write(cmd)

    def query(self, cmd: str) -> str:
        return _query_text(self.inst, cmd)

    def query_numbers(self, cmd: str) -> np.ndarray:
        return _query_csv_numbers(self.inst, cmd)

    def read_errors(self, max_reads: int = 20) -> list[str]:
        """Drain ``SYST:ERR?`` and return every non-zero entry (empty list = clean)."""
        errors: list[str] = []
        for _ in range(max_reads):
            err = self.query("SYST:ERR?")
            if err.startswith("+0") or err.startswith("0,") or err == "0":
                break
            errors.append(err)
        return errors

    def check_errors(self, context: str = "") -> None:
        errors = self.read_errors()
        if errors:
            where = f" after {context}" if context else ""
            raise RuntimeError(f"E5071C reported SCPI errors{where}: " + " | ".join(errors))

    # -- configuration ------------------------------------------------------ #
    def configure(
        self,
        f_start: float = config.FREQ_START_HZ,
        f_stop: float = config.FREQ_STOP_HZ,
        n_points: int = config.NUM_POINTS,
        if_bw: float = config.IF_BANDWIDTH_HZ,
        averaging: int = config.AVERAGING_COUNT,
        power_dbm: Optional[float] = config.POWER_DBM,
        s_params: Sequence[str] = config.S_PARAMS,
    ) -> None:
        """Set up channel 1 with one trace per S-parameter, single-trigger mode.

        ``averaging`` of 0 or 1 turns averaging off. With averaging on, the
        instrument is put in ``TRIG:AVER ON`` so that a single ``INIT1:IMM``
        runs the full averaging count before ``*OPC?`` returns.
        """
        s_params = tuple(str(p).upper() for p in s_params)
        if not s_params:
            raise ValueError("s_params must contain at least one S-parameter")
        for p in s_params:
            if not re.fullmatch(r"S[12][12]", p):
                raise ValueError(f"Unsupported S-parameter for a 2-port ENA: {p!r}")

        averaging = int(averaging or 0)
        avg_on = averaging > 1
        n_points = int(n_points)

        w = self.write
        w("*CLS")

        # Frequency axis and sweep setup on channel 1
        w(f"SENS1:FREQ:STAR {float(f_start):.0f}")
        w(f"SENS1:FREQ:STOP {float(f_stop):.0f}")
        w(f"SENS1:SWE:POIN {n_points}")
        w(f"SENS1:BWID {float(if_bw):.0f}")

        # Averaging: clear + count + state; TRIG:AVER ON makes one INIT:IMM run
        # 'averaging' sweeps so a single *OPC? covers the averaged result.
        if avg_on:
            w(f"SENS1:AVER:COUN {averaging}")
            w("SENS1:AVER ON")
            w("TRIG:AVER ON")
            w("SENS1:AVER:CLE")
        else:
            w("SENS1:AVER OFF")
            w("TRIG:AVER OFF")

        if power_dbm is not None:
            w(f"SOUR1:POW {float(power_dbm):.2f}")

        # Traces: one per S-parameter, in the given order (legacy loop)
        w(f"CALC1:PAR:COUN {len(s_params)}")
        for idx, meas in enumerate(s_params, start=1):
            w(f"CALC1:PAR{idx}:DEF {meas}")

        # Single-trigger mode, ASCII data transfer, channel 1 active
        w("TRIG:SOUR INT")
        w("INIT1:CONT OFF")
        w("FORM:DATA ASC")
        w("DISP:WIND1:ACT")
        w("CALC1:PAR1:SEL")

        self.s_params = s_params
        self._averaging_on = avg_on
        self.settings = {
            "f_start_hz": float(f_start),
            "f_stop_hz": float(f_stop),
            "n_points": n_points,
            "if_bw_hz": float(if_bw),
            "averaging": averaging,
            "power_dbm": None if power_dbm is None else float(power_dbm),
            "s_params": list(s_params),
        }
        self.check_errors("configure()")

    # -- measurement -------------------------------------------------------- #
    def sweep(self) -> Sweep:
        """Trigger one sweep, wait for completion, read all traces."""
        inst = self.inst
        timestamp = datetime.now().isoformat(timespec="microseconds")
        t0 = time.perf_counter()

        if self._averaging_on:
            inst.write("SENS1:AVER:CLE")
        inst.write("INIT1:IMM")
        _query_text(inst, "*OPC?")

        freq = _query_csv_numbers(inst, "SENS1:FREQ:DATA?")
        s: dict[str, np.ndarray] = {}
        for idx, meas in enumerate(self.s_params, start=1):
            inst.write(f"CALC1:PAR{idx}:SEL")
            raw = _query_csv_numbers(inst, "CALC1:DATA:SDAT?")
            if raw.size != 2 * freq.size:
                raise RuntimeError(
                    f"Length mismatch on {meas}: freq={freq.size} sdata={raw.size} "
                    f"(expected {2 * freq.size})"
                )
            s[meas] = _interleaved_to_complex(raw)

        sweep_time = time.perf_counter() - t0

        expected = self.settings.get("n_points")
        if expected is not None and freq.size != expected:
            raise RuntimeError(
                f"Instrument returned {freq.size} points, configure() asked for {expected}"
            )
        return Sweep(freq_hz=freq, s=s, timestamp=timestamp, sweep_time_s=sweep_time)

    def measure_sweep_time(self, n: int = 5) -> float:
        """Mean wall time (s) of ``n`` complete sweep() calls."""
        n = max(1, int(n))
        return float(np.mean([self.sweep().sweep_time_s for _ in range(n)]))


# --------------------------------------------------------------------------- #
# Mock instrument
# --------------------------------------------------------------------------- #
class MockVNA:
    """Drop-in replacement for :class:`VNA` that needs no hardware or pyvisa.

    Synthetic model (deterministic for a given ``seed``):

    * S11/S22: ~-20 dB antenna mismatch with slow ripple (short cable/feed echo).
    * S21: ~-45 dB direct path between the two antennas plus a few weaker,
      longer-delay multipath components -> ripple across the band.
    * Complex Gaussian noise at ``noise_db`` (default -70 dB), reduced by
      ``sqrt(averaging)`` when averaging is on; slow, tiny drift per sweep.
      ``seed=None`` draws a fresh noise stream each run (scene stays fixed).
    * ``perturbation``: complex scalar or ``(N,)`` array added to S21 so tests
      can fake a person in the room (e.g. ``1e-3 * exp(1j*phi)``).

    ``sweep_time_s`` mimics a ~0.3 s instrument sweep (scaled by points /
    averaging) but the call only sleeps for at most 0.05 s.
    """

    MAX_SLEEP_S = 0.05

    def __init__(
        self,
        address: str = config.VNA_ADDRESS,
        timeout_ms: int = config.VNA_TIMEOUT_MS,
        seed: Optional[int] = 0,
        noise_db: float = -70.0,
    ) -> None:
        # seed=None -> fresh OS entropy for the noise stream (the scene geometry
        # stays fixed so separate processes still see the same "room").
        self.address = address
        self.timeout_ms = int(timeout_ms)
        self.seed = None if seed is None else int(seed)
        self.noise_db = float(noise_db)
        self.perturbation: Union[complex, np.ndarray] = 0.0
        self.idn: str = ""
        self.connected: bool = False
        self.settings: dict[str, Any] = {}
        self.s_params: tuple[str, ...] = tuple(config.S_PARAMS)
        self._rng = np.random.default_rng(self.seed)
        self._sweep_count = 0
        self.configure()  # usable immediately; open_vna() will configure again

    # -- lifecycle ---------------------------------------------------------- #
    def connect(self) -> str:
        self.connected = True
        self.idn = f"Keysight Technologies,{config.VNA_MODEL},MOCK00000,A.00.00 (slam.vna.MockVNA)"
        return self.idn

    def close(self) -> None:
        self.connected = False

    def __enter__(self) -> "MockVNA":
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- API parity with VNA ------------------------------------------------ #
    def write(self, cmd: str) -> None:  # noqa: D401 - no-op for parity
        return None

    def query(self, cmd: str) -> str:
        if cmd.strip().upper() == "*IDN?":
            return self.idn
        if cmd.strip().upper() == "SYST:ERR?":
            return '+0,"No error"'
        return ""

    def read_errors(self, max_reads: int = 20) -> list[str]:
        return []

    def check_errors(self, context: str = "") -> None:
        return None

    def configure(
        self,
        f_start: float = config.FREQ_START_HZ,
        f_stop: float = config.FREQ_STOP_HZ,
        n_points: int = config.NUM_POINTS,
        if_bw: float = config.IF_BANDWIDTH_HZ,
        averaging: int = config.AVERAGING_COUNT,
        power_dbm: Optional[float] = config.POWER_DBM,
        s_params: Sequence[str] = config.S_PARAMS,
    ) -> None:
        s_params = tuple(str(p).upper() for p in s_params)
        if not s_params:
            raise ValueError("s_params must contain at least one S-parameter")
        averaging = int(averaging or 0)
        self.s_params = s_params
        self.settings = {
            "f_start_hz": float(f_start),
            "f_stop_hz": float(f_stop),
            "n_points": int(n_points),
            "if_bw_hz": float(if_bw),
            "averaging": averaging,
            "power_dbm": None if power_dbm is None else float(power_dbm),
            "s_params": list(s_params),
        }
        self._build_geometry()

    def _build_geometry(self) -> None:
        """Fixed (seed-derived) scene: reflection/multipath parameters."""
        g = np.random.default_rng((0 if self.seed is None else self.seed) + 12345)
        self._geom = {
            # antenna mismatch: base |Gamma| ~ -20 dB, ripple from feed echo
            "s11_mag_db": -20.0 + g.uniform(-2, 2),
            "s22_mag_db": -20.0 + g.uniform(-2, 2),
            "s11_ripple": 0.15 + g.uniform(-0.05, 0.05),
            "s22_ripple": 0.15 + g.uniform(-0.05, 0.05),
            "s11_delay_s": 2 * (1.5 + g.uniform(-0.5, 0.5)) / (0.66 * _C0),  # cable echo
            "s22_delay_s": 2 * (1.5 + g.uniform(-0.5, 0.5)) / (0.66 * _C0),
            "s11_phase": g.uniform(0, 2 * np.pi),
            "s22_phase": g.uniform(0, 2 * np.pi),
            # direct path ~ diagonal of the 4x4 m square (~5.7 m) at -45 dB
            "s21_mag_db": -45.0 + g.uniform(-1, 1),
            "s21_direct_m": 5.7 + g.uniform(-0.3, 0.3),
            # multipath components: (relative amplitude, extra path length m, phase)
            "s21_paths": [
                (0.35 + g.uniform(-0.1, 0.1), 1.6 + g.uniform(-0.4, 0.4), g.uniform(0, 2 * np.pi)),
                (0.25 + g.uniform(-0.1, 0.1), 4.1 + g.uniform(-0.8, 0.8), g.uniform(0, 2 * np.pi)),
                (0.15 + g.uniform(-0.05, 0.05), 7.5 + g.uniform(-1.0, 1.0), g.uniform(0, 2 * np.pi)),
            ],
            "drift_period_sweeps": 60.0,
        }

    def _noise(self, n: int, scale: float) -> np.ndarray:
        return scale * (self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)) / np.sqrt(2)

    def sweep(self) -> Sweep:
        st = self.settings
        n = int(st["n_points"])
        freq = np.linspace(st["f_start_hz"], st["f_stop_hz"], n)
        g = self._geom
        s: dict[str, np.ndarray] = {}

        averaging = int(st["averaging"])
        avg_factor = np.sqrt(averaging) if averaging > 1 else 1.0
        noise_lin = 10 ** (self.noise_db / 20.0) / avg_factor
        drift = 1.0 + 0.002 * np.sin(2 * np.pi * self._sweep_count / g["drift_period_sweeps"])

        def reflection(mag_db: float, ripple: float, delay: float, phase: float) -> np.ndarray:
            base = 10 ** (mag_db / 20.0)
            return base * (1.0 + ripple * np.cos(2 * np.pi * freq * delay + phase)) * np.exp(
                -1j * (2 * np.pi * freq * delay / 2 + phase)
            )

        def transmission() -> np.ndarray:
            base = 10 ** (g["s21_mag_db"] / 20.0)
            d0 = g["s21_direct_m"]
            h = np.exp(-1j * 2 * np.pi * freq * d0 / _C0)
            for amp, extra, ph in g["s21_paths"]:
                d = d0 + extra
                h = h + amp * (d0 / d) * np.exp(-1j * (2 * np.pi * freq * d / _C0 + ph))
            return base * drift * h

        for meas in self.s_params:
            if meas == "S11":
                val = reflection(g["s11_mag_db"], g["s11_ripple"], g["s11_delay_s"], g["s11_phase"])
            elif meas == "S22":
                val = reflection(g["s22_mag_db"], g["s22_ripple"], g["s22_delay_s"], g["s22_phase"])
            elif meas in ("S21", "S12"):
                val = transmission()
                val = val + np.broadcast_to(np.asarray(self.perturbation, dtype=complex), (n,))
            else:
                val = np.zeros(n, dtype=complex)
            s[meas] = val + self._noise(n, noise_lin)

        # Simulated instrument time: ~0.3 s at the config defaults (201 pts),
        # scaled with the point count, plus a little jitter. Not slept for real.
        sweep_time = 0.3 * (n / 201.0) * (1.0 + 0.03 * self._rng.standard_normal())
        sweep_time = float(max(sweep_time, 0.01))
        time.sleep(min(self.MAX_SLEEP_S, sweep_time))

        self._sweep_count += 1
        return Sweep(
            freq_hz=freq,
            s=s,
            timestamp=datetime.now().isoformat(timespec="microseconds"),
            sweep_time_s=sweep_time,
        )

    def measure_sweep_time(self, n: int = 5) -> float:
        n = max(1, int(n))
        return float(np.mean([self.sweep().sweep_time_s for _ in range(n)]))


# --------------------------------------------------------------------------- #
# CSV I/O
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    """Shortest string that round-trips exactly (Python repr of float)."""
    return repr(float(x))


def sweep_to_csv(sweep: Sweep, path: Union[str, Path]) -> None:
    """Write a sweep with columns exactly ``config.SWEEP_CSV_COLUMNS``.

    S-parameters listed in the columns but absent from ``sweep.s`` are written
    as ``nan`` so the header is always identical across files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(config.SWEEP_CSV_COLUMNS)
    n = sweep.freq_hz.shape[0]

    columns: list[np.ndarray] = []
    for col in cols:
        if col == "Frequency_Hz":
            columns.append(np.asarray(sweep.freq_hz, dtype=float))
            continue
        m = re.fullmatch(r"(S\d\d)_(re|im)", col)
        if m is None:
            raise ValueError(f"Unrecognised column name in config.SWEEP_CSV_COLUMNS: {col!r}")
        name, part = m.group(1), m.group(2)
        if name in sweep.s:
            arr = np.asarray(sweep.s[name])
            if arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} points but freq has {n}")
            columns.append(arr.real.astype(float) if part == "re" else arr.imag.astype(float))
        else:
            columns.append(np.full(n, np.nan))

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([_fmt(c[i]) for c in columns])


def _timestamp_from_filename(path: Path) -> str:
    m = _FILENAME_TS_RE.search(path.name)
    if not m:
        return ""
    try:
        dt = datetime.strptime("_".join(m.groups()), "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return ""
    return dt.isoformat(timespec="microseconds")


def load_sweep_csv(path: Union[str, Path]) -> Sweep:
    """Inverse of :func:`sweep_to_csv`.

    Only the S-parameter columns present in the file are loaded (any
    ``Sxx_re``/``Sxx_im`` pair, not just those in the config). The timestamp is
    parsed from a filename of the form ``<label>_YYYYmmdd_HHMMSS_ffffff.csv``
    when possible, otherwise ``""``. ``sweep_time_s`` is ``nan`` (not stored
    in the CSV; see the session manifest).
    """
    path = Path(path)
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty") from None
        header = [h.strip() for h in header]
        rows = [r for r in reader if r and any(c.strip() for c in r)]

    if "Frequency_Hz" not in header:
        raise ValueError(f"{path} has no 'Frequency_Hz' column (header={header})")

    def _cell(c: str) -> float:
        c = c.strip()
        # pandas writes NaN as an empty cell (na_rep=''); treat it as nan
        return float("nan") if c == "" or c.lower() in ("nan", "na") else float(c)

    data = np.array([[_cell(c) for c in r] for r in rows], dtype=float)
    if data.ndim != 2 or data.shape[1] != len(header):
        raise ValueError(f"{path}: ragged rows or column count mismatch")

    col = {name: i for i, name in enumerate(header)}
    freq = data[:, col["Frequency_Hz"]]

    s: dict[str, np.ndarray] = {}
    for name in header:
        m = re.fullmatch(r"(S\d\d)_re", name)
        if m and f"{m.group(1)}_im" in col:
            sp = m.group(1)
            s[sp] = data[:, col[f"{sp}_re"]] + 1j * data[:, col[f"{sp}_im"]]

    return Sweep(
        freq_hz=freq,
        s=s,
        timestamp=_timestamp_from_filename(path),
        sweep_time_s=float("nan"),
    )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def open_vna(mock: bool = False, **kwargs: Any) -> Union[VNA, MockVNA]:
    """Return a connected and configured instrument (config defaults).

    ``kwargs`` are passed to the constructor (e.g. ``address=``, ``seed=`` for
    the mock).
    """
    inst: Union[VNA, MockVNA] = MockVNA(**kwargs) if mock else VNA(**kwargs)
    inst.connect()
    inst.configure()
    return inst


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Quick E5071C connectivity / sweep check.")
    ap.add_argument("--mock", action="store_true", help="use MockVNA (no hardware, no pyvisa)")
    ap.add_argument("--address", default=config.VNA_ADDRESS, help="VISA resource string")
    ap.add_argument("--n", type=int, default=5, help="sweeps for measure_sweep_time (real only)")
    args = ap.parse_args(argv)

    if args.mock:
        inst = open_vna(mock=True)
        sw = inst.sweep()
        print(f"MockVNA IDN : {inst.idn}")
        print(f"freq_hz     : shape {sw.freq_hz.shape}, {sw.freq_hz[0]/1e6:.3f}-{sw.freq_hz[-1]/1e6:.3f} MHz")
        for k, v in sw.s.items():
            print(f"{k:<12}: shape {v.shape}, dtype {v.dtype}, mean |{k}| = {sw.mag_db(k).mean():.2f} dB")
        print(f"timestamp   : {sw.timestamp}")
        print(f"sweep_time_s: {sw.sweep_time_s:.3f} (simulated)")
        inst.close()
        return 0

    print(f"Connecting to {args.address} ...")
    inst = VNA(address=args.address)
    try:
        idn = inst.connect()
        print(f"IDN         : {idn}")
        inst.configure()
        print(f"settings    : {inst.settings}")
        t = inst.measure_sweep_time(n=args.n)
        print(f"sweep time  : {t:.3f} s (mean of {args.n})")
        sw = inst.sweep()
        for k in sw.s:
            print(f"{k:<12}: mean |{k}| = {sw.mag_db(k).mean():.2f} dB over {sw.n_points} pts")
        errs = inst.read_errors()
        print(f"SYST:ERR?   : {'clean' if not errs else errs}")
    finally:
        inst.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
