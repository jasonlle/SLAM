"""Sensitivity analysis for a SLAM data-collection session.

Answers the go/no-go question after the first lab session: is the change a
person induces in S11 / S21 / S22 clearly larger than the spread of repeated
empty-room sweeps?  The key quantity is the *complex* difference from the
empty-room reference, delta = S - S_ref, compared against the residual spread
of the reference sweeps themselves (the "detectability ratio").

CLI
    python -m slam.analyze_sensitivity --session <name> [--data-dir PATH]
        [--out PATH] [--reference-label empty] [--show]
    python -m slam.analyze_sensitivity --make-fixture PATH [--seed 0]

Programmatic use
    from slam.analyze_sensitivity import analyze_session
    summary_df, figures = analyze_session(session_dir, reference_label="empty")

Runs headless (Agg backend) and never touches the VNA.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from slam import config
from slam.vna import load_sweep_csv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DETECTABLE_RATIO = 3.0      # ratio >= this: clearly above empty-room spread
MARGINAL_RATIO = 1.0        # 1 <= ratio < 3: marginal; < 1: buried in noise
EPS = 1e-15                 # guards log10(0)

SUMMARY_COLUMNS = ["label", "s_param", "n", "mean_abs_delta", "mean_abs_delta_db",
                   "empty_rms", "ratio", "mean_db_change"]

# Validated categorical palette (light surface), fixed slot order - never cycled.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Ordinal blue ramp, used only when a session has more than 8 non-reference
# labels (grid order is spatial, so an ordered ramp is still meaningful).
ORDINAL_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#184f95"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
REF_GREY = "#b5b4ad"
FIG_DPI = 150


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def label_sort_key(label: str, reference_label: str) -> tuple:
    """Reference first, then config grid order, then special labels, then the rest."""
    if label == reference_label:
        return (0, 0, label)
    grid = list(config.GRID_POINTS)
    if label in grid:
        return (1, grid.index(label), label)
    special = list(config.SPECIAL_LABELS)
    if label in special:
        return (2, special.index(label), label)
    return (3, 0, label)


def load_session(session_dir: Path) -> tuple[pd.DataFrame, dict[str, list]]:
    """Read manifest.csv and every sweep it lists. Returns (manifest, {label: [Sweep]})."""
    session_dir = Path(session_dir)
    manifest_path = session_dir / config.MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {config.MANIFEST_NAME} in {session_dir}")
    manifest = pd.read_csv(manifest_path, dtype={"label": str, "filename": str})
    if manifest.empty:
        raise ValueError(f"{manifest_path} has no rows")

    sweeps: dict[str, list] = {}
    missing = []
    for row in manifest.itertuples(index=False):
        path = session_dir / str(row.filename)
        if not path.exists():
            missing.append(path.name)
            continue
        sweeps.setdefault(str(row.label), []).append(load_sweep_csv(path))
    if missing:
        print(f"warning: {len(missing)} sweep file(s) listed in the manifest are missing "
              f"(first: {missing[0]})", file=sys.stderr)
    if not sweeps:
        raise ValueError(f"no sweep files could be loaded from {session_dir}")
    return manifest, sweeps


def _stack(sweeps: list, s_param: str) -> np.ndarray | None:
    """Complex array (n_sweeps, N) for one S-parameter, or None if any sweep lacks it."""
    if not sweeps or any(s_param not in sw.s for sw in sweeps):
        return None
    arrs = [np.asarray(sw.s[s_param], dtype=complex) for sw in sweeps]
    if any(a.size == 0 or np.all(np.isnan(a)) for a in arrs):
        return None                     # sweep_to_csv writes nan for absent S-params
    n = min(a.shape[0] for a in arrs)
    if any(a.shape[0] != n for a in arrs):
        print(f"warning: {s_param}: sweeps have different lengths; truncating to {n}",
              file=sys.stderr)
        arrs = [a[:n] for a in arrs]
    return np.vstack(arrs)


def _db(x) -> np.ndarray:
    return 20.0 * np.log10(np.abs(np.asarray(x)) + EPS)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_session(session_dir, reference_label: str = "empty",
                    s_params=None) -> tuple[pd.DataFrame, dict]:
    """Full analysis of one session.

    Returns
        summary_df : one row per (label, s_param) with SUMMARY_COLUMNS
        figures    : {"s21_vs_freq": Figure, "delta_vs_freq": Figure,
                      "detectability_ratio": Figure, "delta_complex_plane": Figure}
    """
    session_dir = Path(session_dir)
    s_params = list(s_params or config.S_PARAMS)
    manifest, sweeps = load_session(session_dir)

    if reference_label not in sweeps:
        raise ValueError(f"reference label {reference_label!r} not in session "
                         f"(labels: {sorted(sweeps)})")
    labels = sorted(sweeps, key=lambda lab: label_sort_key(lab, reference_label))
    ref_sweeps = sweeps[reference_label]
    freq_hz = np.asarray(ref_sweeps[0].freq_hz, dtype=float)
    n_ref = len(ref_sweeps)
    if n_ref < 2:
        print(f"warning: only {n_ref} reference sweep(s); empty-room spread is undefined "
              f"and ratios will be NaN. Collect >= 10 '{reference_label}' sweeps.",
              file=sys.stderr)

    # --- reference and empty-room spread per S-param -----------------------
    ref: dict[str, np.ndarray] = {}            # mean complex sweep (N,)
    ref_stack: dict[str, np.ndarray] = {}      # (n_ref, N)
    spread_f: dict[str, np.ndarray] = {}       # per-frequency RMS |residual| (N,)
    empty_rms: dict[str, float] = {}           # scalar RMS over sweeps and freqs
    for sp in s_params:
        stack = _stack(ref_sweeps, sp)
        if stack is None:
            print(f"warning: {sp} missing from reference sweeps; skipping", file=sys.stderr)
            continue
        n = min(stack.shape[1], freq_hz.shape[0])
        stack = stack[:, :n]
        mean = stack.mean(axis=0)
        resid = stack - mean
        ref[sp] = mean
        ref_stack[sp] = stack
        if n_ref >= 2:
            spread_f[sp] = np.sqrt(np.mean(np.abs(resid) ** 2, axis=0))
            empty_rms[sp] = float(np.sqrt(np.mean(np.abs(resid) ** 2)))
        else:
            spread_f[sp] = np.full(n, np.nan)
            empty_rms[sp] = float("nan")
    freq_hz = freq_hz[:min(len(freq_hz), *(v.shape[0] for v in ref.values()))] \
        if ref else freq_hz

    # --- per label / per S-param deltas ---------------------------------------
    rows = []
    deltas: dict[str, dict[str, np.ndarray]] = {}   # label -> sp -> (n, N) complex
    stacks: dict[str, dict[str, np.ndarray]] = {}   # label -> sp -> (n, N) complex
    for lab in labels:
        deltas[lab] = {}
        stacks[lab] = {}
        for sp in ref:
            stack = _stack(sweeps[lab], sp)
            if stack is None:
                print(f"warning: {sp} missing for label {lab!r}; skipping", file=sys.stderr)
                continue
            stack = stack[:, :len(freq_hz)]
            delta = stack - ref[sp][None, :]
            deltas[lab][sp] = delta
            stacks[lab][sp] = stack
            mean_abs = float(np.mean(np.abs(delta)))
            e_rms = empty_rms[sp]
            ratio = mean_abs / e_rms if (np.isfinite(e_rms) and e_rms > 0) else float("nan")
            mean_db_change = float(np.mean(_db(stack)) - np.mean(_db(ref[sp])))
            rows.append({
                "label": lab, "s_param": sp, "n": stack.shape[0],
                "mean_abs_delta": mean_abs,
                "mean_abs_delta_db": float(20 * np.log10(mean_abs)) if mean_abs > 0 else float("nan"),
                "empty_rms": e_rms, "ratio": ratio,
                "mean_db_change": mean_db_change,
            })
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    figures = make_figures(
        session_name=session_dir.name, reference_label=reference_label, labels=labels,
        s_params=[sp for sp in s_params if sp in ref], freq_hz=freq_hz,
        ref=ref, ref_stack=ref_stack, spread_f=spread_f, empty_rms=empty_rms,
        stacks=stacks, deltas=deltas, summary=summary,
    )
    return summary, figures


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _label_colors(labels: list[str], reference_label: str) -> dict[str, str]:
    others = [lab for lab in labels if lab != reference_label]
    colors = {reference_label: REF_GREY}
    if len(others) <= len(CATEGORICAL):
        for lab, c in zip(others, CATEGORICAL):
            colors[lab] = c
    else:
        idx = np.linspace(0, len(ORDINAL_RAMP) - 1, len(others)).round().astype(int)
        for lab, i in zip(others, idx):
            colors[lab] = ORDINAL_RAMP[i]
    return colors


MAX_LEGEND_ROWS = 12    # longer legends go below the axes in columns


def _legend_outside(ax_or_fig, handles, labels, title=None, fig_level=False):
    kw = dict(frameon=False, fontsize=8, title_fontsize=8, labelcolor=INK_2,
              handlelength=1.6)
    below = len(labels) > MAX_LEGEND_ROWS       # a 29-row legend squashes the axes
    ncol = min(6, -(-len(labels) // MAX_LEGEND_ROWS) * 3) if below else 1
    if fig_level:
        leg = ax_or_fig.legend(handles, labels, title=title, ncol=ncol,
                               loc="outside lower center" if below else "outside right upper",
                               **kw)
    elif below:
        leg = ax_or_fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                               borderaxespad=0.0, title=title, ncol=ncol, **kw)
    else:
        leg = ax_or_fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                               borderaxespad=0.0, title=title, **kw)
    if leg.get_title():
        leg.get_title().set_color(INK_2)
    return leg


def make_figures(session_name, reference_label, labels, s_params, freq_hz, ref, ref_stack,
                 spread_f, empty_rms, stacks, deltas, summary) -> dict:
    """Build the four contract figures. Returns {name: matplotlib Figure}."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    f_mhz = freq_hz / 1e6
    colors = _label_colors(labels, reference_label)
    others = [lab for lab in labels if lab != reference_label]
    figures = {}
    line_kw = dict(linewidth=1.4, solid_joinstyle="round", solid_capstyle="round")

    # ---- Figure 1: |S21| dB vs frequency -------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=FIG_DPI, layout="constrained",
                           facecolor=SURFACE)
    _style_axes(ax)
    sp_main = "S21" if "S21" in s_params else (s_params[0] if s_params else None)
    handles, names = [], []
    if sp_main is not None:
        for row in ref_stack[sp_main]:
            ax.plot(f_mhz, _db(row), color=REF_GREY, linewidth=0.6, alpha=0.7, zorder=1)
        handles.append(Line2D([], [], color=REF_GREY, linewidth=0.8))
        names.append(f"{reference_label} sweeps (n={ref_stack[sp_main].shape[0]})")
        for lab in others:
            if sp_main not in stacks[lab]:
                continue
            mean_db = _db(stacks[lab][sp_main]).mean(axis=0)
            ax.plot(f_mhz, mean_db, color=colors[lab], zorder=3, **line_kw)
            handles.append(Line2D([], [], color=colors[lab], linewidth=1.6))
            names.append(f"{lab} (n={stacks[lab][sp_main].shape[0]})")
        ax.set_ylabel(f"|{sp_main}| (dB)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_xlim(f_mhz[0], f_mhz[-1])
    ax.set_title(f"{session_name}: |{sp_main}| vs frequency, mean per label", fontsize=10,
                 loc="left")
    _legend_outside(ax, handles, names, title="Label")
    figures["s21_vs_freq"] = fig

    # ---- Figure 2: |delta| dB vs frequency, one subplot per S-param ---------
    n_sp = max(len(s_params), 1)
    fig, axes = plt.subplots(n_sp, 1, figsize=(8.5, 2.4 * n_sp + 1.0), dpi=FIG_DPI,
                             sharex=True, layout="constrained", facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    handles = [Line2D([], [], color=INK_2, linewidth=1.2, linestyle=(0, (4, 2)))]
    names = [f"{reference_label} spread (RMS |S - ref|)"]
    for lab in others:
        handles.append(Line2D([], [], color=colors[lab], linewidth=1.6))
        names.append(lab)
    for ax, sp in zip(axes, s_params):
        _style_axes(ax)
        ax.plot(f_mhz, _db(spread_f[sp]), color=INK_2, linewidth=1.2, linestyle=(0, (4, 2)),
                zorder=2)
        for lab in others:
            if sp not in deltas[lab]:
                continue
            mean_abs_f = np.abs(deltas[lab][sp]).mean(axis=0)
            ax.plot(f_mhz, _db(mean_abs_f), color=colors[lab], zorder=3, **line_kw)
        ax.set_ylabel(f"|Δ{sp}| (dB)")
        ax.set_title(f"{sp}: mean |S − ref| per label vs empty-room spread", fontsize=9,
                     loc="left", color=INK_2)
    axes[-1].set_xlabel("Frequency (MHz)")
    axes[-1].set_xlim(f_mhz[0], f_mhz[-1])
    fig.suptitle(f"{session_name}: complex difference from '{reference_label}' reference",
                 fontsize=10, x=0.01, ha="left", color=INK)
    _legend_outside(fig, handles, names, title="Label", fig_level=True)
    figures["delta_vs_freq"] = fig

    # ---- Figure 3: detectability ratio bar chart -----------------------------
    fig, ax = plt.subplots(figsize=(min(18.0, max(6.5, 0.9 * len(others) + 3.0)), 4.4), dpi=FIG_DPI,
                           layout="constrained", facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(False, axis="x")
    sp_colors = dict(zip(s_params, CATEGORICAL))
    n_groups = len(others)
    width = min(0.26, 0.8 / max(len(s_params), 1))
    x = np.arange(n_groups)
    ratios_all = []
    for j, sp in enumerate(s_params):
        vals = []
        for lab in others:
            sel = summary[(summary.label == lab) & (summary.s_param == sp)]
            vals.append(float(sel.ratio.iloc[0]) if len(sel) else np.nan)
        vals = np.asarray(vals)
        ratios_all.extend(v for v in vals if np.isfinite(v))
        offs = (j - (len(s_params) - 1) / 2) * (width + 0.02)
        bars = ax.bar(x + offs, np.nan_to_num(vals, nan=0.0), width=width, color=sp_colors[sp],
                      label=sp, zorder=3, linewidth=0)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=6.5, color=INK_2)
    use_log = bool(ratios_all) and (max(ratios_all) / max(min(ratios_all), 1e-3) > 50)
    if use_log:
        ax.set_yscale("log")
    det_line = ax.axhline(DETECTABLE_RATIO, color=INK_2, linewidth=1.0, linestyle=(0, (4, 2)),
                          zorder=2)
    nf_line = ax.axhline(MARGINAL_RATIO, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)),
                         zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(others, rotation=30 if n_groups > 6 else 0,
                       ha="right" if n_groups > 6 else "center")
    ax.set_ylabel("Detectability ratio = mean|S − ref| / empty RMS"
                  + ("  (log)" if use_log else ""))
    ax.set_xlabel("Label (person position)")
    ax.set_title(f"{session_name}: person-induced change relative to empty-room spread",
                 fontsize=10, loc="left")
    handles = [Line2D([], [], color=sp_colors[sp], linewidth=6) for sp in s_params]
    handles += [det_line, nf_line]
    names = list(s_params) + [f"detectable (ratio ≥ {DETECTABLE_RATIO:g})",
                              f"noise floor (ratio = {MARGINAL_RATIO:g})"]
    _legend_outside(ax, handles, names, title="S-param")
    figures["detectability_ratio"] = fig

    # ---- Figure 4: complex-plane scatter of mean delta (S21) -----------------
    sp_sc = "S21" if "S21" in s_params else (s_params[0] if s_params else None)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=FIG_DPI, layout="constrained",
                           facecolor=SURFACE)
    _style_axes(ax)
    handles, names = [], []
    if sp_sc is not None:
        scale = 1e3
        resid = (ref_stack[sp_sc] - ref[sp_sc][None, :]).mean(axis=1) * scale
        ax.scatter(resid.real, resid.imag, s=28, color=REF_GREY, edgecolor=SURFACE,
                   linewidth=1.0, zorder=3)
        handles.append(Line2D([], [], marker="o", color=REF_GREY, linestyle="", markersize=6))
        names.append(f"{reference_label} residuals")
        for lab in others:
            if sp_sc not in deltas[lab]:
                continue
            pts = deltas[lab][sp_sc].mean(axis=1) * scale
            ax.scatter(pts.real, pts.imag, s=28, color=colors[lab], edgecolor=SURFACE,
                       linewidth=1.0, zorder=4)
            c = pts.mean()
            ax.scatter([c.real], [c.imag], s=110, marker="D", color=colors[lab],
                       edgecolor=SURFACE, linewidth=1.5, zorder=5)
            ax.annotate(lab, (c.real, c.imag), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=INK, zorder=6)
            handles.append(Line2D([], [], marker="o", color=colors[lab], linestyle="",
                                  markersize=6))
            names.append(lab)
        handles.append(Line2D([], [], marker="D", color=INK_2, linestyle="", markersize=7))
        names.append("label centroid")
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"Re(Δ{sp_sc})  (linear, ×10⁻³)")
        ax.set_ylabel(f"Im(Δ{sp_sc})  (linear, ×10⁻³)")
    ax.set_title(f"{session_name}: Δ{sp_sc} averaged over frequency, one point per sweep",
                 fontsize=10, loc="left")
    _legend_outside(ax, handles, names, title="Label")
    figures["delta_complex_plane"] = fig

    return figures


def save_figures(figures: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (name, fig) in enumerate(figures.items(), start=1):
        p = out_dir / f"{i}_{name}.png"
        fig.savefig(p, dpi=FIG_DPI, facecolor=SURFACE, bbox_inches="tight")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_summary(summary: pd.DataFrame) -> str:
    fmt = {
        "mean_abs_delta": "{:.3e}".format, "mean_abs_delta_db": "{:7.1f}".format,
        "empty_rms": "{:.3e}".format, "ratio": "{:6.2f}".format,
        "mean_db_change": "{:+7.2f}".format,
    }
    return summary.to_string(index=False, formatters=fmt, na_rep="nan")


def interpret(summary: pd.DataFrame, reference_label: str, s_params) -> list[str]:
    lines = []
    non_ref = summary[summary.label != reference_label]
    if non_ref.empty:
        return [f"No non-reference labels in session; nothing to compare against "
                f"'{reference_label}'."]
    if non_ref.ratio.isna().all():
        return ["Detectability ratios are undefined (fewer than 2 reference sweeps)."]
    for sp in s_params:
        d = non_ref[non_ref.s_param == sp]
        if d.empty:
            continue
        det = d[d.ratio >= DETECTABLE_RATIO].label.tolist()
        marg = d[(d.ratio >= MARGINAL_RATIO) & (d.ratio < DETECTABLE_RATIO)].label.tolist()
        bur = d[d.ratio < MARGINAL_RATIO].label.tolist()
        lines.append(f"{sp}: ratio >= {DETECTABLE_RATIO:g} (clearly detectable) for: "
                     f"{', '.join(det) or 'none'};  marginal (1-3): {', '.join(marg) or 'none'};"
                     f"  buried (< 1): {', '.join(bur) or 'none'}")
    if {"S11", "S21"} <= set(s_params):
        s11 = non_ref[non_ref.s_param == "S11"].set_index("label").ratio
        s21 = non_ref[non_ref.s_param == "S21"].set_index("label").ratio
        both = s11.index.intersection(s21.index)
        s11_only_info = [lab for lab in both if s11[lab] >= DETECTABLE_RATIO]
        s11_beats = [lab for lab in both if s11[lab] > s21[lab]]
        lines.append(f"S11 informative (ratio >= {DETECTABLE_RATIO:g}) for: "
                     f"{', '.join(s11_only_info) or 'none'};  "
                     f"S11 more sensitive than S21 for: {', '.join(s11_beats) or 'none'}")
        n_det_s21 = int((s21 >= DETECTABLE_RATIO).sum())
        verdict = ("GO: S21 separates the person from the empty room at "
                   f"{n_det_s21}/{len(s21)} positions"
                   if n_det_s21 >= max(1, len(s21) // 2) else
                   f"NO-GO / RETHINK: S21 clearly detectable at only {n_det_s21}/{len(s21)} "
                   "positions - consider antenna placement, IF bandwidth or averaging")
        lines.append(verdict)
    return lines


# ---------------------------------------------------------------------------
# Fixture generator (synthetic session in the exact collect.py format)
# ---------------------------------------------------------------------------

def make_fixture_session(path, seed: int = 0) -> Path:
    """Write a synthetic session (manifest.csv + sweep CSVs) to `path`.

    20 'empty' sweeps and 10 each of center / A1 / E5 / outside / near_ant1, with
    complex perturbations of different sizes:  near_ant1 perturbs S11 strongly,
    center perturbs S21 more than S11, outside sits below the noise floor.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    f = np.linspace(config.FREQ_START_HZ, config.FREQ_STOP_HZ, config.NUM_POINTS)
    f0 = f.mean()

    def ripple(level_db, taus_ns, amps):
        base = 10 ** (level_db / 20)
        s = np.ones_like(f, dtype=complex)
        for tau, a in zip(taus_ns, amps):
            s = s + a * np.exp(-2j * np.pi * f * tau * 1e-9)
        return base * s / np.abs(s).mean()

    base = {
        "S11": ripple(-20, [3.0, 7.5], [0.35, 0.15]) * np.exp(-2j * np.pi * (f - f0) * 4e-9),
        "S21": ripple(-45, [12.0, 21.0, 33.0], [0.5, 0.3, 0.15]),
        "S22": ripple(-21, [2.5, 6.0], [0.3, 0.2]) * np.exp(-2j * np.pi * (f - f0) * 5e-9),
    }
    noise_sigma = 10 ** (-70 / 20) / np.sqrt(2)     # per re/im component, ~-70 dB total

    # (label, n, {sp: (delta_dB_level, delay_ns, phase_rad)}); ~20% per-sweep amplitude jitter
    plan = [
        ("empty", 20, {}),
        ("center", 10, {"S11": (-66.0, 6.0, 0.7), "S21": (-54.0, 9.0, 0.7), "S22": (-67.0, 5.0, 1.2)}),
        ("A1", 10, {"S11": (-60.0, 4.0, 2.1), "S21": (-59.0, 11.0, 2.6), "S22": (-70.0, 7.0, 0.4)}),
        ("E5", 10, {"S11": (-71.0, 8.0, -0.5), "S21": (-58.5, 10.0, -1.9), "S22": (-61.0, 3.0, 2.8)}),
        ("outside", 10, {"S11": (-76.0, 5.0, 0.0), "S21": (-74.0, 12.0, 1.0), "S22": (-76.0, 6.0, 0.0)}),
        ("near_ant1", 10, {"S11": (-40.0, 2.0, -2.4), "S21": (-52.0, 8.0, -0.6), "S22": (-69.0, 6.0, 1.5)}),
    ]
    t0 = _dt.datetime(2026, 9, 27, 14, 0, 0)
    rows = []
    k = 0
    for label, n, pert in plan:
        try:
            x, y = config.label_to_xy(label)
        except ValueError:
            x, y = None, None
        for _ in range(n):
            drift = 1 + 0.002 * rng.standard_normal()        # slow gain drift, same for all
            sw = {}
            for sp in config.S_PARAMS:
                s = base[sp] * drift * np.exp(1j * 0.004 * rng.standard_normal())
                if sp in pert:
                    lvl, tau, phi = pert[sp]
                    amp = 10 ** (lvl / 20) * (1 + 0.2 * rng.standard_normal())
                    jitter = rng.uniform(-0.3, 0.3)            # breathing / sway jitter
                    s = s + amp * np.exp(-2j * np.pi * (f - f0) * tau * 1e-9 + 1j * (phi + jitter))
                s = s + noise_sigma * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
                sw[sp] = s
            ts = t0 + _dt.timedelta(seconds=8 * k)
            k += 1
            fname = f"{label}_{ts.strftime('%Y%m%d_%H%M%S_%f')}.csv"
            df = pd.DataFrame({"Frequency_Hz": f})
            for sp in config.S_PARAMS:
                df[f"{sp}_re"] = sw[sp].real
                df[f"{sp}_im"] = sw[sp].imag
            df = df[config.SWEEP_CSV_COLUMNS]
            df.to_csv(path / fname, index=False, float_format="%.10g")
            rows.append({
                "timestamp": ts.isoformat(), "session": path.name, "label": label,
                "x_m": "" if x is None else x, "y_m": "" if y is None else y,
                "person": "fixture", "notes": "synthetic", "filename": fname,
                "sweep_time_s": round(1.9 + 0.05 * rng.standard_normal(), 3),
            })
    pd.DataFrame(rows, columns=config.MANIFEST_COLUMNS).to_csv(
        path / config.MANIFEST_NAME, index=False)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m slam.analyze_sensitivity",
        description="Is the person-induced change in S11/S21/S22 bigger than the "
                    "empty-room spread? Loads one session and produces summary + figures.")
    p.add_argument("--session", help="session name (folder under --data-dir)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help=f"data root (default: config.DATA_DIR = {config.DATA_DIR})")
    p.add_argument("--out", type=Path, default=None,
                   help="output folder for PNGs + summary.csv "
                        "(default: config.FIGURES_DIR/<session>/)")
    p.add_argument("--reference-label", default="empty",
                   help="label of the empty-room sweeps (default: empty)")
    p.add_argument("--show", action="store_true",
                   help="also open the figures interactively (default: headless Agg)")
    p.add_argument("--make-fixture", type=Path, metavar="PATH",
                   help="write a synthetic session to PATH and exit")
    p.add_argument("--seed", type=int, default=0, help="seed for --make-fixture")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.make_fixture is not None:
        out = make_fixture_session(args.make_fixture, seed=args.seed)
        n = sum(1 for _ in out.glob("*.csv")) - 1
        print(f"wrote fixture session with {n} sweeps to {out}")
        print(f"analyze with: python -m slam.analyze_sensitivity --session {out.name} "
              f"--data-dir {out.parent}")
        return 0

    if not args.session:
        build_parser().error("--session is required (or use --make-fixture PATH)")
    if not args.show:
        matplotlib.use("Agg")

    data_dir = Path(args.data_dir) if args.data_dir else Path(config.DATA_DIR)
    session_dir = data_dir / args.session
    out_dir = Path(args.out) if args.out else Path(config.FIGURES_DIR) / args.session

    summary, figures = analyze_session(session_dir, reference_label=args.reference_label)
    s_params = [sp for sp in config.S_PARAMS if sp in set(summary.s_param)]

    print(f"\nSession: {session_dir}   reference label: '{args.reference_label}'")
    print(format_summary(summary))
    print()
    for line in interpret(summary, args.reference_label, s_params):
        print(line)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6g")
    paths = save_figures(figures, out_dir)
    print(f"\nwrote {summary_path}")
    for p in paths:
        print(f"wrote {p}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
