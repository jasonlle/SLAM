"""Position-labelled VNA sweep collector.

Replaces the legacy ``scpiworkbench/datagatheringscript.py`` /
``camera_scpi/camera_scipi_test.py`` collectors with a session-oriented tool
that tags every sweep with a floor-grid label and (x, y) position.

Layout produced under ``<data_dir>/<session>/``::

    session_info.json                  written once per session (resumes appended)
    manifest.csv                       one row per sweep, config.MANIFEST_COLUMNS
    <label>_<YYYYmmdd_HHMMSS_ffffff>.csv   one sweep, config.SWEEP_CSV_COLUMNS

CLI (see ``python -m slam.collect --help``)::

    python -m slam.collect --session s1 --label center --person alice --n 10
    python -m slam.collect --session s1 --label doorway --x 0.5 --y -0.3 --person alice
    python -m slam.collect --session s1 --person alice          # interactive loop
    python -m slam.collect --session s1 --label empty --mock    # no instrument

The building blocks (:func:`collect_batch`, :func:`write_manifest_row`,
:func:`write_session_info`, :func:`resolve_label`) are importable so a GUI or a
camera ground-truth script can reuse them without going through ``main``.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from slam import config
from slam.vna import Sweep, open_vna, sweep_to_csv

SESSION_INFO_NAME = "session_info.json"
FILENAME_TIME_FMT = "%Y%m%d_%H%M%S_%f"
DEFAULT_COUNTDOWN_S = 3.0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
class BatchInterrupted(KeyboardInterrupt):
    """Raised by :func:`collect_batch` after a Ctrl-C has been handled cleanly.

    The sweep that was in flight (if any) has been written to disk and its
    manifest row appended before this is raised. ``paths`` holds every file
    the interrupted batch managed to write.
    """

    def __init__(self, paths: list[Path]) -> None:
        super().__init__("collection interrupted by user")
        self.paths = paths


class _DeferSigint:
    """Context manager that postpones Ctrl-C until the block has finished.

    Used around the CSV + manifest write so a sweep is never half-recorded.
    """

    def __init__(self) -> None:
        self.interrupted = False
        self._previous: Any = None

    def _handler(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self.interrupted = True

    def __enter__(self) -> "_DeferSigint":
        try:
            self._previous = signal.signal(signal.SIGINT, self._handler)
        except ValueError:  # not in main thread: no deferral possible
            self._previous = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
        if self.interrupted:
            raise KeyboardInterrupt


def _now_iso() -> str:
    """Local ISO-8601 timestamp with microseconds."""
    return datetime.now().isoformat(timespec="microseconds")


def _sweep_datetime(sweep: Sweep) -> datetime:
    """Datetime used for the filename: the sweep's own timestamp if parseable."""
    try:
        return datetime.fromisoformat(sweep.timestamp)
    except (TypeError, ValueError):
        return datetime.now()


def _fmt_coord(value: float | None) -> str:
    """Manifest representation of a coordinate (empty string for unknown)."""
    return "" if value is None else repr(float(value))


def sweep_filename(label: str, when: datetime) -> str:
    """Return ``<label>_<YYYYmmdd_HHMMSS_ffffff>.csv``."""
    return f"{label}_{when.strftime(FILENAME_TIME_FMT)}.csv"


def sweep_settings() -> dict[str, Any]:
    """The sweep configuration passed to ``VNA.configure`` (from ``slam.config``)."""
    return {
        "f_start_hz": config.FREQ_START_HZ,
        "f_stop_hz": config.FREQ_STOP_HZ,
        "n_points": config.NUM_POINTS,
        "if_bw_hz": config.IF_BANDWIDTH_HZ,
        "averaging": config.AVERAGING_COUNT,
        "power_dbm": config.POWER_DBM,
        "s_params": list(config.S_PARAMS),
    }


# --------------------------------------------------------------------------- #
# Label resolution
# --------------------------------------------------------------------------- #
def known_labels() -> list[str]:
    """All labels ``config.label_to_xy`` accepts: grid points then specials."""
    return list(config.GRID_POINTS) + [
        lbl for lbl in config.SPECIAL_LABELS if lbl not in config.GRID_POINTS
    ]


def canonical_label(label: str) -> str:
    """Spell a known label the way config does ('c3' -> 'C3', 'Empty' -> 'empty').

    Unknown labels are returned stripped, so custom labels are left alone.
    """
    key = label.strip()
    if key in config.GRID_POINTS or key in config.SPECIAL_LABELS:
        return key
    if key.upper() in config.GRID_POINTS:
        return key.upper()
    if key.lower() in config.SPECIAL_LABELS:
        return key.lower()
    return key


def resolve_label(
    label: str,
    x: float | None = None,
    y: float | None = None,
    prompt_if_unknown: bool = False,
) -> tuple[float | None, float | None]:
    """Return the (x, y) position in metres for ``label``.

    Known labels (``config.GRID_POINTS`` / ``config.SPECIAL_LABELS``) supply
    their own coordinates; explicit ``x``/``y`` override them. An unknown label
    needs both ``x`` and ``y``; with ``prompt_if_unknown`` they are read from
    stdin instead, otherwise ``ValueError`` is raised.
    """
    try:
        kx, ky = config.label_to_xy(label)
    except ValueError:
        if x is None or y is None:
            if not prompt_if_unknown:
                raise ValueError(
                    f"unknown label {label!r}: pass --x and --y (metres) or use "
                    f"one of {', '.join(known_labels())}"
                ) from None
            print(f"Label {label!r} is not in config; enter its position in metres.")
            if x is None:
                x = _prompt_float("  x_m: ")
            if y is None:
                y = _prompt_float("  y_m: ")
        return x, y
    return (kx if x is None else x, ky if y is None else y)


def _prompt_float(prompt: str) -> float:
    """Keep asking until the user types a number."""
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print(f"  not a number: {raw!r}")


# --------------------------------------------------------------------------- #
# Session files
# --------------------------------------------------------------------------- #
def write_session_info(
    session_dir: Path,
    session: str,
    idn: str,
    settings: dict[str, Any] | None = None,
    mock: bool = False,
) -> Path:
    """Write ``session_info.json`` once; on later calls append to ``resumed_at``.

    The file records the VNA identification string, the sweep settings,
    ``config.ANTENNAS``, ``config.ROOM`` and the start time.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / SESSION_INFO_NAME
    now = _now_iso()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            info = json.load(fh)
        info.setdefault("resumed_at", []).append(now)
    else:
        info = {
            "session": session,
            "start_time": now,
            "resumed_at": [],
            "mock": bool(mock),
            "vna_idn": idn,
            "vna_address": config.VNA_ADDRESS,
            "sweep_settings": settings if settings is not None else sweep_settings(),
            "sweep_csv_columns": list(config.SWEEP_CSV_COLUMNS),
            "manifest_columns": list(config.MANIFEST_COLUMNS),
            "grid_size_m": list(config.GRID_SIZE_M),
            "grid_spacing_m": config.GRID_SPACING_M,
            "antennas": config.ANTENNAS,
            "room": config.ROOM,
        }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=_json_default)
        fh.write("\n")
    return path


def _json_default(obj: Any) -> Any:
    """Make Paths, tuples-in-sets and numpy scalars JSON serialisable."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def write_manifest_row(
    session_dir: Path,
    session: str,
    label: str,
    x: float | None,
    y: float | None,
    person: str,
    notes: str,
    filename: str,
    sweep_time_s: float,
    timestamp: str | None = None,
) -> None:
    """Append one row to ``<session_dir>/manifest.csv`` (header if the file is new)."""
    path = session_dir / config.MANIFEST_NAME
    is_new = not path.exists() or path.stat().st_size == 0
    row = {
        "timestamp": timestamp if timestamp is not None else _now_iso(),
        "session": session,
        "label": label,
        "x_m": _fmt_coord(x),
        "y_m": _fmt_coord(y),
        "person": person,
        "notes": notes,
        "filename": filename,
        "sweep_time_s": f"{sweep_time_s:.3f}",
    }
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(config.MANIFEST_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in config.MANIFEST_COLUMNS})


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def countdown(seconds: float, label: str) -> None:
    """Print a settle-in countdown (whole seconds) before a label's first sweep."""
    if seconds <= 0:
        return
    print(f"  Get into position for {label!r} ...", end="", flush=True)
    remaining = seconds
    while remaining > 0:
        step = min(1.0, remaining)
        print(f" {remaining:g}", end="", flush=True)
        time.sleep(step)
        remaining -= step
    print(" go")


def collect_batch(
    vna: Any,
    session_dir: Path,
    session: str,
    label: str,
    x: float | None,
    y: float | None,
    person: str,
    n: int = 10,
    interval: float = 0.0,
    notes: str = "",
    countdown_s: float = DEFAULT_COUNTDOWN_S,
    on_sweep: Callable[[Path, Sweep], None] | None = None,
) -> list[Path]:
    """Take ``n`` sweeps for one label and record each one.

    For every sweep: ``vna.sweep()`` -> ``sweep_to_csv`` into ``session_dir``
    -> :func:`write_manifest_row`. A countdown runs once before the first
    sweep; ``interval`` seconds are slept between consecutive sweeps.

    Ctrl-C is handled so the sweep in progress is either fully recorded or
    not recorded at all; :class:`BatchInterrupted` (a ``KeyboardInterrupt``
    subclass carrying the written paths) is then raised.

    Returns the list of CSV paths written.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        countdown(countdown_s, label)
        for i in range(1, n + 1):
            if i > 1 and interval > 0:
                time.sleep(interval)
            sweep = vna.sweep()
            with _DeferSigint():
                when = _sweep_datetime(sweep)
                path = session_dir / sweep_filename(label, when)
                sweep_to_csv(sweep, path)
                write_manifest_row(
                    session_dir, session, label, x, y, person, notes,
                    path.name, float(sweep.sweep_time_s),
                    timestamp=when.isoformat(timespec="microseconds"),
                )
                written.append(path)
                print(f"  [{i}/{n}] {label}  {float(sweep.sweep_time_s):.3f} s", flush=True)
                if on_sweep is not None:
                    on_sweep(path, sweep)
    except KeyboardInterrupt:
        print(f"\n  interrupted after {len(written)} sweep(s) of {label!r}")
        raise BatchInterrupted(written) from None
    return written


class RunStats:
    """Running totals for one invocation, printed at exit."""

    def __init__(self) -> None:
        self.per_label: Counter[str] = Counter()
        self.sweep_times: list[float] = []

    def record(self, label: str, sweep_time_s: float) -> None:
        self.per_label[label] += 1
        self.sweep_times.append(float(sweep_time_s))

    @property
    def total(self) -> int:
        return len(self.sweep_times)

    @property
    def mean_sweep_time(self) -> float:
        return sum(self.sweep_times) / len(self.sweep_times) if self.sweep_times else 0.0

    def print_summary(self, session_dir: Path) -> None:
        print()
        print(f"Session folder : {session_dir}")
        print(f"Total sweeps   : {self.total}")
        if self.per_label:
            per = ", ".join(f"{lbl}={cnt}" for lbl, cnt in sorted(self.per_label.items()))
            print(f"Per label      : {per}")
            print(f"Mean sweep time: {self.mean_sweep_time:.3f} s")


# --------------------------------------------------------------------------- #
# Interactive mode
# --------------------------------------------------------------------------- #
def print_label_menu() -> None:
    """Print the known labels compactly: one line per grid row, then specials."""
    rows: dict[float, list[tuple[float, str]]] = {}
    for lbl, (gx, gy) in config.GRID_POINTS.items():
        rows.setdefault(float(gy), []).append((float(gx), lbl))
    print("Known labels (grid rows, y from low to high; x increases left to right):")
    for gy in sorted(rows):
        cells = " ".join(lbl for _, lbl in sorted(rows[gy]))
        print(f"  y={gy:g} m : {cells}")
    specials = [lbl for lbl in config.SPECIAL_LABELS if lbl not in config.GRID_POINTS]
    if specials:
        print(f"  special : {' '.join(specials)}")
    print("Any other label is accepted; you will be asked for its x/y in metres.")


def parse_interactive_line(line: str, default_n: int, default_notes: str) -> tuple[str, int, str] | None:
    """Parse ``label [n] [notes...]``; returns None for an empty line or 'q'.

    ``n`` is taken from the second token only if it is a positive integer;
    everything after it is joined as notes.
    """
    tokens = line.strip().split()
    if not tokens:
        return None
    if tokens[0].lower() in {"q", "quit", "exit"}:
        return None
    label = tokens[0]
    n = default_n
    rest = tokens[1:]
    if rest:
        try:
            cand = int(rest[0])
        except ValueError:
            cand = None
        if cand is not None and cand > 0:
            n = cand
            rest = rest[1:]
    notes = " ".join(rest) if rest else default_notes
    return label, n, notes


def interactive_loop(
    vna: Any,
    session_dir: Path,
    session: str,
    person: str,
    default_n: int,
    interval: float,
    default_notes: str,
    countdown_s: float,
    stats: RunStats,
) -> None:
    """Prompt for ``label [n] [notes]`` batches until 'q' or EOF."""
    print_label_menu()
    last: tuple[str, int, str] | None = None
    while True:
        try:
            line = input("label [n] [notes] (q to quit, ENTER repeats last): ")
        except EOFError:
            print()
            return
        stripped = line.strip()
        if stripped == "":
            if last is None:
                print("  nothing to repeat yet")
                continue
            job = last
        else:
            job = parse_interactive_line(stripped, default_n, default_notes)
            if job is None:
                return
        label, n, notes = job
        label = canonical_label(label)
        try:
            x, y = resolve_label(label, prompt_if_unknown=True)
        except EOFError:
            print()
            return
        print(f"Collecting {n} x {label!r} at ({_fmt_coord(x) or '?'}, {_fmt_coord(y) or '?'}) m")
        collect_batch(
            vna, session_dir, session, label, x, y, person, n, interval, notes, countdown_s,
            on_sweep=lambda path, sw: stats.record(label, sw.sweep_time_s),
        )
        last = (label, n, notes)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m slam.collect``."""
    p = argparse.ArgumentParser(
        prog="python -m slam.collect",
        description="Collect position-labelled VNA sweeps into a session folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--session", required=True, help="session name (folder under --data-dir)")
    p.add_argument("--label", help="position label; omit for interactive mode")
    p.add_argument("--x", type=float, default=None, help="x position in metres (overrides label)")
    p.add_argument("--y", type=float, default=None, help="y position in metres (overrides label)")
    p.add_argument("--person", help="who is standing at the label (or operator for 'empty')")
    p.add_argument("--n", type=int, default=10, help="sweeps per label")
    p.add_argument("--interval", type=float, default=0.0, help="seconds to wait between sweeps")
    p.add_argument("--notes", default="", help="free-text note stored in the manifest")
    p.add_argument("--countdown", type=float, default=DEFAULT_COUNTDOWN_S,
                   help="seconds to wait before the first sweep of each label (0 to skip)")
    p.add_argument("--mock", action="store_true", help="use the simulated VNA (no instrument)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help=f"root data folder (default config.DATA_DIR = {config.DATA_DIR})")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.n <= 0:
        print("error: --n must be positive", file=sys.stderr)
        return 2
    interactive = args.label is None

    person = args.person
    if not person:
        if not interactive:
            print("error: --person is required when --label is given", file=sys.stderr)
            return 2
        try:
            person = input("person: ").strip()
        except EOFError:
            person = ""
        if not person:
            print("error: a person name is required", file=sys.stderr)
            return 2

    x = y = None
    if not interactive:
        args.label = canonical_label(args.label)
        try:
            x, y = resolve_label(args.label, args.x, args.y)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    data_dir: Path = args.data_dir if args.data_dir is not None else config.DATA_DIR
    session_dir = data_dir / args.session
    stats = RunStats()
    exit_code = 0

    # open_vna() already connects + configures; the mock gets a fresh noise
    # stream per run so repeated sessions are not bit-identical.
    vna = open_vna(mock=args.mock, **({"seed": None} if args.mock else {}))
    try:
        idn = vna.idn
        print(f"Connected: {idn}")
        vna.configure(
            f_start=config.FREQ_START_HZ,
            f_stop=config.FREQ_STOP_HZ,
            n_points=config.NUM_POINTS,
            if_bw=config.IF_BANDWIDTH_HZ,
            averaging=config.AVERAGING_COUNT,
            power_dbm=config.POWER_DBM,
            s_params=config.S_PARAMS,
        )
        info_path = write_session_info(session_dir, args.session, idn, sweep_settings(), args.mock)
        print(f"Session: {session_dir}  ({info_path.name} ready)")

        try:
            if interactive:
                interactive_loop(
                    vna, session_dir, args.session, person, args.n, args.interval,
                    args.notes, args.countdown, stats,
                )
            else:
                print(f"Collecting {args.n} x {args.label!r} at "
                      f"({_fmt_coord(x) or '?'}, {_fmt_coord(y) or '?'}) m")
                collect_batch(
                    vna, session_dir, args.session, args.label, x, y, person,
                    args.n, args.interval, args.notes, args.countdown,
                    on_sweep=lambda path, sw: stats.record(args.label, sw.sweep_time_s),
                )
        except KeyboardInterrupt:
            print("Stopped by user (Ctrl-C).")
            exit_code = 130
    finally:
        try:
            vna.close()
        except Exception as exc:  # closing must never hide the data summary
            print(f"warning: error closing VNA: {exc}", file=sys.stderr)
        stats.print_summary(session_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
