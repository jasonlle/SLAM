"""Phase-A digital twin of the SLAM lab + VNA rig (numpy only).

Generates simulated sessions in *exactly* the collect.py format (same sweep CSV
columns, manifest.csv, session_info.json) so analyze_sensitivity and future
training code run unchanged on simulated data.

Physics (see docs/digital_twin.md for the rationale and literature values):

* direct path      : free-space Friis with a cos^n panel-antenna pattern
                     (tilt / azimuth / gain from config.ANTENNAS)
* room reflections : first-order image method for the 6 room surfaces
                     (floor, ceiling, 4 walls) with constant (or optional
                     Fresnel) reflection coefficients
* person           : vertical cylinder for *blockage* (segment-to-ray clearance
                     vs. Fresnel radius) and a point scatterer at
                     (x, y, ~1 m) with a bistatic / monostatic RCS for the
                     scattered field
* S11 / S22        : static mismatch (-20 dB, slow cable ripple)
                     + monostatic backscatter from the person
                     + first-order room reflections returning to the antenna
* S21              : coherent sum of direct + reflections + scatter
* nuisances        : per-sweep person jitter (position, RCS), slow session
                     drift, per-sweep gain jitter, complex Gaussian noise

All tunable constants live in ``TwinParams`` (top of file).  The channel model
is split into small functions so a better EM engine (Sionna RT, ...) can
replace ``direct_path`` / ``image_reflections`` / ``person_scatter`` later.

CLI examples
------------
    python -m slam.sim.digital_twin --session sim_demo --labels all --n 5 --seed 0
    python -m slam.sim.digital_twin --session sim_demo --labels empty,center,A1 --n 10
    python -m slam.sim.digital_twin --sweep-plot figures/twin_check.png
    python -m slam.sim.digital_twin --session sim_low --labels all --n 5 \
        --set antenna_z_override_m=1.8 --set antenna_tilt_override_deg=-10
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

from slam import config
from slam.vna import Sweep, sweep_to_csv

C0 = 299_792_458.0
EPS0 = 8.854_187_8e-12


# --------------------------------------------------------------------------- #
# Tunable physics / nuisance parameters
# --------------------------------------------------------------------------- #
@dataclass
class TwinParams:
    """Every knob of the Phase-A twin.  Units are SI unless the name says dB."""

    # ---- antenna pattern ---------------------------------------------------
    # cos^n power pattern about boresight.  Directivity of a cos^n hemisphere
    # pattern is 2(n+1): n=2 -> 6 = 7.8 dBi, which matches the ~8 dBi L-com
    # panel.  Separate exponents for the E-plane (vertical for V-pol) and the
    # H-plane let you mimic the panel's unequal beamwidths.
    pattern_n_e: float = 2.0
    pattern_n_h: float = 2.0
    front_to_back_db: float = 15.0        # back-lobe floor relative to peak
    gain_dbi_override: float | None = None  # None -> config.ANTENNAS[*]["gain_dbi"]

    # ---- what-if geometry overrides (None -> use config.ANTENNAS) -----------
    antenna_z_override_m: float | None = None
    antenna_tilt_override_deg: float | None = None

    # ---- rig / cabling folded into static terms ------------------------------
    # Extra two-port loss: 2 cables (~0.3 dB/m at 900 MHz), connectors,
    # adapters and the pattern/efficiency shortfall of a cheap panel.  Tune
    # this first so the simulated empty-room |S21| level matches Dataset v1.
    system_loss_db: float = 10.0
    cable_velocity_factor: float = 0.66   # RG-58 / LMR-195 class coax
    cable_default_m: float = 5.0          # used when config has no "cable_m"
    s11_static_db: float = -20.0          # antenna + cable mismatch level
    s22_static_db: float = -19.0
    s11_ripple_db: float = 1.5            # slow ripple amplitude on |S11|
    s11_ripple_cycles: float = 1.3        # ripple cycles across the band

    # ---- room ----------------------------------------------------------------
    # Room bounding box relative to the taped square: the room's min corner in
    # grid coordinates.  None -> derived from config.ROOM["grid_origin_in_room_m"]
    # (falls back to (-1, -1, 0)).  Size comes from config.ROOM["size_m"]
    # unless overridden.
    room_origin_m: tuple[float, float, float] | None = None
    room_size_override_m: tuple[float, float, float] | None = None
    room_margin_m: float = 0.5            # auto-expand room to enclose rig+grid
    reflection_model: str = "constant"    # "constant" | "fresnel"
    # constant model (Fresnel-ish magnitudes at typical incidence, sign = phase
    # reversal for E-field parallel to a dielectric surface)
    refl_wall: float = -0.6
    refl_floor: float = -0.7
    refl_ceiling: float = -0.4
    # fresnel model: (eps_r, sigma_S_per_m) at ~0.9 GHz.  ITU-R P.2040 values:
    # plasterboard (2.73, 0.008), concrete (5.24, 0.043), ceiling board
    # (1.48, 0.001).  The drop ceiling is treated as a lossy half space, which
    # over-estimates its reflection; lower eps_r or use "constant" to tame it.
    wall_material: tuple[float, float] = (2.73, 0.008)
    floor_material: tuple[float, float] = (5.24, 0.043)
    ceiling_material: tuple[float, float] = (1.48, 0.001)
    include_surfaces: tuple[str, ...] = ("floor", "ceiling", "x0", "x1", "y0", "y1")

    # ---- person --------------------------------------------------------------
    person_height_m: float = 1.75
    person_radius_m: float = 0.17         # torso cylinder radius (blockage)
    person_scatter_z_m: float = 1.0       # point-scatterer height (contract)
    rcs_bistatic_m2: float = 1.0          # ~0.5-2 m^2 at UHF in the literature
    rcs_monostatic_m2: float = 1.0
    scatter_phase_rad: float = math.pi    # phase reversal of a lossy dielectric
    # blockage: full attenuation when the ray passes inside the torso, falling
    # off as a Gaussian in units of (fresnel_fraction * Fresnel radius)
    blockage_db: float = 5.0              # "a few dB" at sub-GHz
    blockage_phase_rad: float = 0.5       # extra phase when fully blocked
    fresnel_fraction: float = 0.6
    special_person_xy: dict = field(
        default_factory=lambda: {"outside": (-0.7, 2.0)})

    # ---- per-sweep jitter / drift / noise -------------------------------------
    jitter_xy_m: float = 0.03             # sway/breathing: position std
    jitter_rcs_frac: float = 0.10         # RCS std as a fraction (+-10 %)
    sweep_gain_jitter_db: float = 0.02    # VNA trace-to-trace gain jitter
    # Slow drift of a warmed-up, calibrated VNA.  NOTE: drift multiplies the
    # -20 dB static S11/S22 term, so 0.15 dB / 6 deg per hour (cold-start
    # values) produces a -52 dB "delta" on an EMPTY room after ~8 min, which
    # swamps the person's S11/S22 signature and made `outside` look
    # S11-detectable.  Use --set drift_amp_db=0.15 to study that effect.
    drift_amp_db: float = 0.02            # slow sinusoidal gain drift amplitude
    drift_period_s: float = 1800.0
    drift_phase_deg_per_hour: float = 1.0  # cable/temperature phase drift
    noise_db: float = -70.0               # E|n|^2 per point, all S-params

    # ---- timing model used for timestamps / manifest --------------------------
    sweep_time_s: float = 1.5
    label_setup_s: float = 12.0           # walk + countdown before a label

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- #
# Small geometry helpers
# --------------------------------------------------------------------------- #
def freqs_hz() -> np.ndarray:
    return np.linspace(config.FREQ_START_HZ, config.FREQ_STOP_HZ, config.NUM_POINTS)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _grid_center() -> tuple[float, float]:
    gx, gy = config.GRID_SIZE_M
    return gx / 2.0, gy / 2.0


@dataclass
class Antenna:
    name: str
    pos: np.ndarray            # (3,)
    boresight: np.ndarray      # unit (3,)
    h_axis: np.ndarray         # unit, horizontal, perpendicular to boresight
    v_axis: np.ndarray         # unit, boresight x h_axis (E-plane for V-pol)
    gain_lin: float            # peak power gain (linear)
    cable_m: float

    def gain(self, u: np.ndarray, p: TwinParams) -> float:
        """Power gain (linear) in unit direction u from the antenna."""
        u = _unit(np.asarray(u, dtype=float))
        ct = float(np.clip(np.dot(u, self.boresight), -1.0, 1.0))
        theta = math.acos(ct)
        phi = math.atan2(float(np.dot(u, self.v_axis)), float(np.dot(u, self.h_axis)))
        return self.gain_lin * panel_gain(theta, phi, (p.pattern_n_e, p.pattern_n_h),
                                          front_to_back_db=p.front_to_back_db)


def panel_gain(theta, phi, n, front_to_back_db: float = 15.0):
    """Normalised cos^n panel pattern (peak = 1), theta/phi in radians.

    theta = angle off boresight, phi = angle around boresight measured from the
    H-plane axis (phi = 90 deg is the E-plane).  ``n`` may be a scalar or a
    (n_E, n_H) pair; the effective exponent blends the two with phi.  Behind
    the panel the gain sits at the front-to-back floor.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if np.ndim(n) == 0:
        n_eff = float(n)
    else:
        n_e, n_h = float(n[0]), float(n[1])
        n_eff = n_e * np.sin(phi) ** 2 + n_h * np.cos(phi) ** 2
    floor = 10.0 ** (-front_to_back_db / 10.0)
    ct = np.clip(np.cos(theta), 0.0, 1.0)
    return floor + (1.0 - floor) * ct ** n_eff


def build_antennas(p: TwinParams) -> dict[str, Antenna]:
    """Antenna objects from config.ANTENNAS (with optional what-if overrides)."""
    cx, cy = _grid_center()
    out: dict[str, Antenna] = {}
    for name, a in config.ANTENNAS.items():
        pos = np.array(a["xyz_m"], dtype=float)
        if p.antenna_z_override_m is not None:
            pos[2] = p.antenna_z_override_m
        az = a.get("azimuth_deg")
        if az is None:
            az = math.degrees(math.atan2(cy - pos[1], cx - pos[0]))
        tilt = a.get("tilt_deg")
        if p.antenna_tilt_override_deg is not None:
            tilt = p.antenna_tilt_override_deg
        if tilt is None:
            horiz = math.hypot(cx - pos[0], cy - pos[1])
            tilt = -math.degrees(math.atan2(pos[2], horiz))
        az_r, tilt_r = math.radians(az), math.radians(tilt)
        b = np.array([math.cos(tilt_r) * math.cos(az_r),
                      math.cos(tilt_r) * math.sin(az_r),
                      math.sin(tilt_r)])
        h = _unit(np.cross(np.array([0.0, 0.0, 1.0]), b))
        if np.linalg.norm(h) == 0:           # boresight straight up/down
            h = np.array([1.0, 0.0, 0.0])
        v = np.cross(b, h)
        gain_dbi = p.gain_dbi_override if p.gain_dbi_override is not None \
            else float(a.get("gain_dbi", 8.0))
        cable = a.get("cable_m")
        cable = p.cable_default_m if cable is None else float(cable)
        out[name] = Antenna(name, pos, b, h, v, 10.0 ** (gain_dbi / 10.0), cable)
    return out


@dataclass
class Room:
    lo: np.ndarray   # (3,) min corner, grid coords
    hi: np.ndarray   # (3,) max corner

    def surfaces(self) -> dict[str, tuple[int, float, float]]:
        """name -> (axis, plane coordinate, normal sign toward the interior)."""
        return {
            "floor": (2, float(self.lo[2]), +1.0),
            "ceiling": (2, float(self.hi[2]), -1.0),
            "x0": (0, float(self.lo[0]), +1.0),
            "x1": (0, float(self.hi[0]), -1.0),
            "y0": (1, float(self.lo[1]), +1.0),
            "y1": (1, float(self.hi[1]), -1.0),
        }


def build_room(p: TwinParams, antennas: dict[str, Antenna]) -> Room:
    size = p.room_size_override_m or tuple(config.ROOM["size_m"])
    if p.room_origin_m is not None:
        origin = p.room_origin_m
    else:
        g = config.ROOM.get("grid_origin_in_room_m")
        origin = (-g[0], -g[1], 0.0) if g else (-1.0, -1.0, 0.0)
    lo = np.array(origin, dtype=float)
    hi = lo + np.array(size, dtype=float)
    # make sure the box encloses antennas, the grid and a standing person
    pts = [a.pos for a in antennas.values()]
    gx, gy = config.GRID_SIZE_M
    pts += [np.array([0.0, 0.0, 0.0]), np.array([gx, gy, p.person_height_m])]
    for xy in p.special_person_xy.values():
        pts.append(np.array([xy[0], xy[1], 0.0]))
    pts = np.array(pts)
    need_lo = pts.min(axis=0) - p.room_margin_m
    need_hi = pts.max(axis=0) + p.room_margin_m
    need_lo[2] = 0.0                       # floor stays at z=0
    need_hi[2] = pts[:, 2].max() + 0.05    # ceiling only needs to clear the rig
    if np.any(need_lo < lo - 1e-9) or np.any(need_hi > hi + 1e-9):
        lo = np.minimum(lo, need_lo)
        hi = np.maximum(hi, need_hi)
        print(f"[twin] room box expanded to enclose rig/grid: {lo} .. {hi}",
              file=sys.stderr)
    return Room(lo, hi)


# --------------------------------------------------------------------------- #
# Reflection coefficient
# --------------------------------------------------------------------------- #
def reflection_coefficient(surface: str, cos_inc: float, f: np.ndarray,
                           p: TwinParams) -> np.ndarray | complex:
    """Complex reflection coefficient for a room surface.

    "constant": the tunable per-surface constants.  "fresnel": lossy half-space
    Fresnel coefficient with the polarisation implied by a vertically polarised
    rig (walls -> TE, floor/ceiling -> TM).
    """
    kind = "floor" if surface == "floor" else "ceiling" if surface == "ceiling" else "wall"
    if p.reflection_model == "constant":
        return {"floor": p.refl_floor, "ceiling": p.refl_ceiling, "wall": p.refl_wall}[kind]
    eps_r, sigma = {"floor": p.floor_material, "ceiling": p.ceiling_material,
                    "wall": p.wall_material}[kind]
    eps_c = eps_r - 1j * sigma / (2.0 * np.pi * f * EPS0)
    ci = float(np.clip(cos_inc, 0.0, 1.0))
    si2 = 1.0 - ci * ci
    root = np.sqrt(eps_c - si2)
    if kind == "wall":                       # TE (E perpendicular to plane of inc.)
        return (ci - root) / (ci + root)
    return (eps_c * ci - root) / (eps_c * ci + root)   # TM


# --------------------------------------------------------------------------- #
# Path building blocks.  Each returns (field over freqs, list of legs) where a
# leg is a (start_xyz, end_xyz) straight segment used later for blockage.
# --------------------------------------------------------------------------- #
def direct_path(tx: Antenna, rx: Antenna, f: np.ndarray, p: TwinParams):
    """Free-space Friis amplitude with pattern gains at both ends."""
    d_vec = rx.pos - tx.pos
    d = float(np.linalg.norm(d_vec))
    u = d_vec / d
    lam = C0 / f
    g = math.sqrt(tx.gain(u, p) * rx.gain(-u, p))
    field_ = g * lam / (4.0 * np.pi * d) * np.exp(-2j * np.pi * f * d / C0)
    return field_, [(tx.pos, rx.pos)]


def image_reflections(tx: Antenna, rx: Antenna, room: Room, f: np.ndarray,
                      p: TwinParams) -> list[tuple[str, np.ndarray, list]]:
    """First-order image-method reflections off the 6 room surfaces.

    Works for the monostatic case too (tx is rx): the image path then returns
    to the same antenna, which is what feeds S11/S22.
    """
    out = []
    for name, (axis, coord, sign) in room.surfaces().items():
        if name not in p.include_surfaces:
            continue
        img = tx.pos.copy()
        img[axis] = 2.0 * coord - tx.pos[axis]
        d_vec = rx.pos - img
        d = float(np.linalg.norm(d_vec))
        if d < 1e-6:
            continue
        # intersection of img->rx with the plane
        denom = d_vec[axis]
        if abs(denom) < 1e-9:
            continue
        t = (coord - img[axis]) / denom
        if not (0.0 <= t <= 1.0):
            continue
        refl_pt = img + t * d_vec
        u_dep = _unit(refl_pt - tx.pos)          # departure direction at tx
        u_arr = _unit(refl_pt - rx.pos)          # arrival direction (rx -> point)
        cos_inc = abs(d_vec[axis]) / d
        gamma = reflection_coefficient(name, cos_inc, f, p)
        lam = C0 / f
        g = math.sqrt(tx.gain(u_dep, p) * rx.gain(u_arr, p))
        field_ = gamma * g * lam / (4.0 * np.pi * d) * np.exp(-2j * np.pi * f * d / C0)
        out.append((name, field_, [(tx.pos, refl_pt), (refl_pt, rx.pos)]))
    return out


def person_scatter(tx: Antenna, rx: Antenna, person_xyz: np.ndarray, rcs_m2: float,
                   f: np.ndarray, p: TwinParams) -> np.ndarray:
    """Point-scatterer (bistatic radar equation) field; monostatic if tx is rx."""
    v1 = person_xyz - tx.pos
    v2 = person_xyz - rx.pos
    d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    g = math.sqrt(tx.gain(v1 / d1, p) * rx.gain(v2 / d2, p))
    lam = C0 / f
    amp = g * lam * math.sqrt(max(rcs_m2, 0.0)) / ((4.0 * np.pi) ** 1.5 * d1 * d2)
    return amp * np.exp(-2j * np.pi * f * (d1 + d2) / C0 + 1j * p.scatter_phase_rad)


def _segment_distance(p1, q1, p2, q2):
    """Closest distance between segments p1-q1 and p2-q2; also the parameter
    s in [0,1] along the first segment where it occurs (Ericson, RTCD 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f_ = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r)), 0.0
    if a <= eps:
        s, t = 0.0, min(max(f_ / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= eps:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(max((b * f_ - c * e) / denom, 0.0), 1.0) if denom > eps else 0.0
            t = (b * s + f_) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return float(np.linalg.norm(c1 - c2)), s


def blockage_factor(legs: list, person_xy, f_center: float, p: TwinParams) -> complex:
    """Complex factor (<=1) applied to a path when the person's torso cylinder
    sits inside a fraction of the first Fresnel zone of any of its legs."""
    if person_xy is None:
        return 1.0
    ax0 = np.array([person_xy[0], person_xy[1], 0.0])
    ax1 = np.array([person_xy[0], person_xy[1], p.person_height_m])
    total = sum(float(np.linalg.norm(b - a)) for a, b in legs)
    lam = C0 / f_center
    worst = 0.0
    before = 0.0
    for a, b in legs:
        dist, s = _segment_distance(a, b, ax0, ax1)
        leg_len = float(np.linalg.norm(b - a))
        d_a = before + s * leg_len
        d_b = max(total - d_a, 1e-6)
        r_f = math.sqrt(lam * max(d_a, 1e-6) * d_b / total)
        clearance = dist - p.person_radius_m
        if clearance <= 0.0:
            frac = 1.0
        else:
            frac = math.exp(-(clearance / (p.fresnel_fraction * r_f)) ** 2)
        worst = max(worst, frac)
        before += leg_len
    att_db = p.blockage_db * worst
    return 10.0 ** (-att_db / 20.0) * np.exp(-1j * p.blockage_phase_rad * worst)


# --------------------------------------------------------------------------- #
# Static / nuisance terms
# --------------------------------------------------------------------------- #
def static_mismatch(f: np.ndarray, level_db: float, cable_m: float, p: TwinParams,
                    phase0: float) -> np.ndarray:
    """Antenna/cable mismatch as seen at the VNA port: -20 dB with a slow
    ripple and a linear phase from the cable's round-trip electrical length."""
    f0 = 0.5 * (f[0] + f[-1])
    bw = f[-1] - f[0]
    mag_db = level_db + p.s11_ripple_db * np.sin(2.0 * np.pi * p.s11_ripple_cycles * (f - f0) / bw)
    tau_rt = 2.0 * cable_m / (p.cable_velocity_factor * C0)
    return 10.0 ** (mag_db / 20.0) * np.exp(-2j * np.pi * f * tau_rt + 1j * phase0)


def cable_phase(f: np.ndarray, cable_m: float, p: TwinParams) -> np.ndarray:
    return np.exp(-2j * np.pi * f * cable_m / (p.cable_velocity_factor * C0))


def drift_factor(t_s: float, p: TwinParams) -> complex:
    g_db = p.drift_amp_db * math.sin(2.0 * math.pi * t_s / p.drift_period_s)
    ph = math.radians(p.drift_phase_deg_per_hour) * t_s / 3600.0
    return 10.0 ** (g_db / 20.0) * complex(math.cos(ph), math.sin(ph))


def complex_noise(rng: np.random.Generator, n: int, p: TwinParams) -> np.ndarray:
    sigma = math.sqrt(10.0 ** (p.noise_db / 10.0) / 2.0)
    return sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


# --------------------------------------------------------------------------- #
# One full 2-port sweep
# --------------------------------------------------------------------------- #
class Twin:
    """Bundles antennas, room and params so repeated sweeps are cheap."""

    def __init__(self, params: TwinParams | None = None):
        self.p = params or TwinParams()
        self.f = freqs_hz()
        self.antennas = build_antennas(self.p)
        names = list(self.antennas)
        if len(names) < 2:
            raise ValueError("config.ANTENNAS must define at least two antennas")
        self.a1 = self.antennas[names[0]]
        self.a2 = self.antennas[names[1]]
        self.room = build_room(self.p, self.antennas)
        self.f_center = 0.5 * (self.f[0] + self.f[-1])
        self.sys_amp = 10.0 ** (-self.p.system_loss_db / 20.0)

    def sweep(self, person_xy, rng: np.random.Generator, t_s: float = 0.0,
              return_parts: bool = False):
        p, f = self.p, self.f
        a1, a2 = self.a1, self.a2
        true_xy, rcs_b, rcs_m = None, 0.0, 0.0
        if person_xy is not None:
            true_xy = (float(person_xy[0]) + p.jitter_xy_m * rng.standard_normal(),
                       float(person_xy[1]) + p.jitter_xy_m * rng.standard_normal())
            rcs_scale = max(1.0 + p.jitter_rcs_frac * rng.standard_normal(), 0.05)
            rcs_b, rcs_m = p.rcs_bistatic_m2 * rcs_scale, p.rcs_monostatic_m2 * rcs_scale
            pxyz = np.array([true_xy[0], true_xy[1], p.person_scatter_z_m])

        parts: dict[str, np.ndarray] = {}
        # ---- S21: direct + reflections (both blockable) + bistatic scatter
        s21_direct, legs = direct_path(a1, a2, f, p)
        s21_direct = s21_direct * blockage_factor(legs, true_xy, self.f_center, p)
        s21_refl = np.zeros_like(f, dtype=complex)
        for name, fld, legs in image_reflections(a1, a2, self.room, f, p):
            s21_refl += fld * blockage_factor(legs, true_xy, self.f_center, p)
        s21_scat = person_scatter(a1, a2, pxyz, rcs_b, f, p) if person_xy is not None \
            else np.zeros_like(f, dtype=complex)
        s21 = (s21_direct + s21_refl + s21_scat) * self.sys_amp \
            * cable_phase(f, a1.cable_m + a2.cable_m, p)
        parts.update(S21_direct=s21_direct, S21_refl=s21_refl, S21_scatter=s21_scat)

        # ---- S11 / S22: static + returning reflections + monostatic scatter
        def reflection_port(ant: Antenna, level_db: float, phase0: float, key: str):
            static = static_mismatch(f, level_db, ant.cable_m, p, phase0)
            refl = np.zeros_like(f, dtype=complex)
            for name, fld, legs in image_reflections(ant, ant, self.room, f, p):
                refl += fld * blockage_factor(legs, true_xy, self.f_center, p)
            scat = person_scatter(ant, ant, pxyz, rcs_m, f, p) if person_xy is not None \
                else np.zeros_like(f, dtype=complex)
            parts[key + "_refl"], parts[key + "_scatter"] = refl, scat
            return static + (refl + scat) * self.sys_amp * cable_phase(f, 2.0 * ant.cable_m, p)

        s11 = reflection_port(a1, p.s11_static_db, 0.3, "S11")
        s22 = reflection_port(a2, p.s22_static_db, 1.1, "S22")

        # ---- drift, per-sweep gain jitter, noise
        common = drift_factor(t_s, p) * 10.0 ** (p.sweep_gain_jitter_db * rng.standard_normal() / 20.0)
        n = len(f)
        out = {
            "S11": s11 * common + complex_noise(rng, n, p),
            "S21": s21 * common + complex_noise(rng, n, p),
            "S22": s22 * common + complex_noise(rng, n, p),
        }
        if return_parts:
            return out, parts, true_xy
        return out


def simulate_sweep(person_xy, rng: np.random.Generator, params: TwinParams | None = None,
                   t_s: float = 0.0, twin: Twin | None = None) -> dict[str, np.ndarray]:
    """One full 2-port sweep -> {"S11", "S21", "S22"} complex arrays over
    config freqs.  person_xy=None simulates the empty room."""
    twin = twin or Twin(params)
    return twin.sweep(person_xy, rng, t_s)


# --------------------------------------------------------------------------- #
# Session writer (collect.py format)
# --------------------------------------------------------------------------- #
def all_labels() -> list[str]:
    labels = ["empty"] + list(config.GRID_POINTS)
    labels += [k for k in config.SPECIAL_LABELS if k not in labels]
    return labels


def parse_labels(spec: str) -> list[str]:
    spec = spec.strip()
    if spec == "all":
        return all_labels()
    if spec == "grid":
        return ["empty"] + list(config.GRID_POINTS)
    return [s.strip() for s in spec.split(",") if s.strip()]


def label_person_xy(label: str, p: TwinParams):
    """Where the person stands for a label; None = nobody in the room."""
    if label == "empty":
        return None
    x, y = config.label_to_xy(label)
    if x is None or y is None:
        if label in p.special_person_xy:
            return tuple(p.special_person_xy[label])
        print(f"[twin] label {label!r} has no coordinates; simulated as empty",
              file=sys.stderr)
        return None
    return (float(x), float(y))


def _sweep_settings() -> dict:
    return {
        "f_start_hz": config.FREQ_START_HZ, "f_stop_hz": config.FREQ_STOP_HZ,
        "n_points": config.NUM_POINTS, "if_bw_hz": config.IF_BANDWIDTH_HZ,
        "averaging": config.AVERAGING_COUNT, "power_dbm": config.POWER_DBM,
        "s_params": list(config.S_PARAMS),
    }


def simulate_session(session: str, labels: Iterable[str], n: int, seed: int,
                     data_dir: Path | None = None, params: TwinParams | None = None,
                     person: str = "sim", notes: str = "", start_time: datetime | None = None,
                     verbose: bool = True) -> Path:
    """Write <data_dir>/<session>/ with sweep CSVs, manifest.csv, session_info.json."""
    p = params or TwinParams()
    twin = Twin(p)
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir) if data_dir is not None else Path(config.DATA_DIR)
    sdir = data_dir / session
    sdir.mkdir(parents=True, exist_ok=True)
    start = start_time or datetime.now()

    info_path = sdir / "session_info.json"
    if not info_path.exists():
        info = {
            "session": session,
            "start_time": start.isoformat(timespec="microseconds"),
            "resumed_at": [],
            "mock": False,
            "simulated": True,
            "seed": seed,
            "vna_idn": "SIMULATED,slam.sim.digital_twin phase A (numpy image method),0,0",
            "vna_address": config.VNA_ADDRESS,
            "sweep_settings": _sweep_settings(),
            "sweep_csv_columns": list(config.SWEEP_CSV_COLUMNS),
            "manifest_columns": list(config.MANIFEST_COLUMNS),
            "grid_size_m": list(config.GRID_SIZE_M),
            "grid_spacing_m": config.GRID_SPACING_M,
            "antennas": config.ANTENNAS,
            "room": config.ROOM,
            "twin_params": p.to_json_dict(),
            "twin_room_box_m": {"lo": twin.room.lo.tolist(), "hi": twin.room.hi.tolist()},
        }
        info_path.write_text(json.dumps(info, indent=2, default=str) + "\n")

    manifest = sdir / config.MANIFEST_NAME
    new_manifest = not manifest.exists()
    t_s = 0.0
    n_written = 0
    with manifest.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=config.MANIFEST_COLUMNS)
        if new_manifest:
            w.writeheader()
        for label in labels:
            xy = label_person_xy(label, p)
            # manifest x_m/y_m follow config exactly like collect.py does (blank
            # for empty/outside); the simulated position is kept in `notes`
            try:
                man_xy = config.label_to_xy(label)
            except ValueError:
                man_xy = (None, None)
            t_s += p.label_setup_s
            for i in range(n):
                s, _parts, true_xy = twin.sweep(xy, rng, t_s, return_parts=True)
                ts = start + timedelta(seconds=t_s)
                ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")
                fname = f"{label}_{ts.strftime('%Y%m%d_%H%M%S_%f')}.csv"
                sweep = Sweep(freq_hz=twin.f, s=s, timestamp=ts_str,
                              sweep_time_s=round(p.sweep_time_s, 3))
                sweep_to_csv(sweep, sdir / fname)
                note = f"sim seed={seed} sweep={i}"
                if true_xy is not None:
                    note += f" true_xy=({true_xy[0]:.3f},{true_xy[1]:.3f})"
                if notes:
                    note += " " + notes
                w.writerow({
                    "timestamp": ts_str, "session": session, "label": label,
                    "x_m": "" if man_xy[0] is None else repr(float(man_xy[0])),
                    "y_m": "" if man_xy[1] is None else repr(float(man_xy[1])),
                    "person": person,
                    "notes": note, "filename": fname,
                    "sweep_time_s": f"{p.sweep_time_s:.3f}",
                })
                t_s += p.sweep_time_s
                n_written += 1
            if verbose:
                s21_db = 20 * np.log10(np.abs(s["S21"]) + 1e-15)
                print(f"[twin] {label:>10s}  n={n}  mean|S21|={s21_db.mean():6.1f} dB"
                      f"  xy={xy}")
    if verbose:
        print(f"[twin] wrote {n_written} sweeps to {sdir}")
    return sdir


# --------------------------------------------------------------------------- #
# Quick visual check
# --------------------------------------------------------------------------- #
def sweep_plot(path: Path, params: TwinParams | None = None, seed: int = 0,
               n_ref: int = 5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = params or TwinParams()
    twin = Twin(p)
    rng = np.random.default_rng(seed)
    f_mhz = twin.f / 1e6
    cx, cy = _grid_center()
    empties = [twin.sweep(None, rng, 20.0 + 2 * i)["S21"] for i in range(n_ref)]
    ref = np.mean(empties, axis=0)
    center = twin.sweep((cx, cy), rng, 60.0)["S21"]
    empty2 = twin.sweep(None, rng, 80.0)["S21"]
    db = lambda z: 20 * np.log10(np.abs(z) + 1e-15)  # noqa: E731

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for e in empties:
        ax[0].plot(f_mhz, db(e), color="0.7", lw=0.8)
    ax[0].plot(f_mhz, db(ref), color="k", lw=1.5, label="empty (mean of %d)" % n_ref)
    ax[0].plot(f_mhz, db(center), color="C3", lw=1.5, label="person at center")
    ax[0].set_xlabel("frequency (MHz)")
    ax[0].set_ylabel("|S21| (dB)")
    ax[0].set_title("digital twin: |S21|")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(f_mhz, db(center - ref), color="C3", label="|S21_center - ref|")
    ax[1].plot(f_mhz, db(empty2 - ref), color="0.5", label="|S21_empty - ref| (noise+drift)")
    ax[1].axhline(p.noise_db, color="k", ls="--", lw=0.8, label="noise level")
    ax[1].set_xlabel("frequency (MHz)")
    ax[1].set_ylabel("|delta| (dB)")
    ax[1].set_title("complex difference from empty reference")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[twin] wrote {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def apply_overrides(p: TwinParams, items: Iterable[str]) -> TwinParams:
    """--set name=value (value parsed as JSON, falling back to string)."""
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not hasattr(p, k):
            raise SystemExit(f"unknown TwinParams field {k!r}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        if isinstance(val, list):
            val = tuple(val)
        setattr(p, k, val)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m slam.sim.digital_twin",
        description="Phase-A numpy digital twin: writes simulated sessions in collect.py format.")
    ap.add_argument("--session", help="session name, e.g. sim_demo (written under data dir)")
    ap.add_argument("--labels", default="all",
                    help="'all' (empty + grid + specials), 'grid', or comma list e.g. empty,center,A1")
    ap.add_argument("--n", type=int, default=10, help="sweeps per label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=None, help="default config.DATA_DIR")
    ap.add_argument("--person", default="sim", help="value for the manifest 'person' column")
    ap.add_argument("--notes", default="", help="extra text appended to manifest notes")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override a TwinParams field (JSON value), repeatable")
    ap.add_argument("--sweep-plot", type=Path, default=None,
                    help="write a PNG of |S21| empty vs. person-at-center (matplotlib Agg)")
    ap.add_argument("--print-params", action="store_true", help="dump TwinParams and exit")
    args = ap.parse_args(argv)

    params = apply_overrides(TwinParams(), args.set)
    if args.print_params:
        print(json.dumps(params.to_json_dict(), indent=2))
        return 0
    if args.sweep_plot is not None:
        sweep_plot(args.sweep_plot, params, seed=args.seed)
    if args.session:
        simulate_session(args.session, parse_labels(args.labels), args.n, args.seed,
                         data_dir=args.data_dir, params=params, person=args.person,
                         notes=args.notes)
    elif args.sweep_plot is None:
        ap.error("give --session and/or --sweep-plot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
