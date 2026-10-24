"""
train_model.py - learn "which cell is the person in" from the sweeps that
collect_live.py saved, test it honestly, and save the model for predict_live.py.

What it does
  1. Loads every session folder in the data folder (or the ones you name).
  2. Keeps only the usable sweeps (empty, cells 1-9, outside).
  3. Features: each sweep MINUS that session's empty room, measured from the
     empty sweeps in the session's first REF_SECONDS. That is exactly what
     predict_live.py will have: half a minute of empty room at the start.
  4. Tests several model types on data they did not train on:
       2+ sessions: train on all but one session, test on that one, repeat
       1 session:   train on the first half (in time), test on the second, and
                    the other way round
  5. Trains the best model on ALL the data and saves it.

Usage:
    python train_model.py                          # all sessions in the data folder
    python train_model.py --sessions 2026-10-22_A 2026-10-23_A
    python train_model.py --model logistic         # force a model type

Output (in <data folder>/models/):
    model_<date_time>.joblib   the trained model
    model_latest.joblib        copy of the newest one (predict_live.py uses this)
    report_<date_time>.txt     everything printed below
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =========================== settings ======================================
# same data folder as the other scripts: SLAM_DATA_DIR, or "data" next to this file
DATA_DIR = os.environ.get("SLAM_DATA_DIR") or str(Path(__file__).resolve().parent / "data")
REF_SECONDS = 30.0       # empty-room reference = empty sweeps in the first 30 s
SMOOTH_SWEEPS = 30       # live display votes over the last 30 sweeps (~0.5 s)
SEED = 0
# ===========================================================================

CLASS_ORDER = ["empty"] + [str(i) for i in range(1, 10)] + ["outside"]


# =========================== data ==========================================
def find_sessions(data_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        found = [data_dir / n for n in names]
        missing = [p.name for p in found if not (p / "sweeps.csv").exists()]
        if missing:
            sys.exit(f"no sweeps.csv in: {', '.join(missing)}")
        return found
    return sorted(p for p in data_dir.iterdir() if (p / "sweeps.csv").exists())


def load_session(sdir: Path) -> dict:
    """Read one session -> times, labels, usable flags and complex S-parameters."""
    head = pd.read_csv(sdir / "sweeps.csv", nrows=0).columns
    s_cols = [c for c in head if c[:1] == "S" and "_" in c]
    dtypes = {c: np.float64 for c in s_cols}
    dtypes["label"] = str
    df = pd.read_csv(sdir / "sweeps.csv", dtype=dtypes)
    s_params = sorted({c.split("_")[0] for c in s_cols})
    S, freqs = {}, None
    for sp in s_params:
        re_c = [c for c in s_cols if c.startswith(sp + "_re_")]
        im_c = [c for c in s_cols if c.startswith(sp + "_im_")]
        S[sp] = df[re_c].to_numpy() + 1j * df[im_c].to_numpy()
        f = [c.split("_", 2)[2] for c in re_c]
        if freqs is None:
            freqs = f
        elif f != freqs:
            raise ValueError(f"{sdir.name}: {sp} has a different frequency list")
    return {"name": sdir.name, "t": df["t0_s"].to_numpy(), "label": df["label"].to_numpy(),
            "usable": df["usable"].to_numpy() == 1, "S": S, "s_params": s_params, "freqs": freqs}


def empty_reference(sess: dict, ref_seconds: float = REF_SECONDS) -> dict:
    """Average empty room over the session's first ref_seconds of empty sweeps."""
    emp = sess["usable"] & (sess["label"] == "empty")
    if not emp.any():
        raise ValueError(f"{sess['name']}: no empty sweeps, so no reference")
    t_first = sess["t"][emp][0]
    m = emp & (sess["t"] <= t_first + ref_seconds)
    return {sp: X[m].mean(axis=0) for sp, X in sess["S"].items()}


def make_features(S: dict, ref: dict, s_params: list[str]) -> np.ndarray:
    """Sweep minus empty room, as real and imaginary parts, for each S-parameter.
    S[sp]: (n_sweeps, n_freqs) complex; ref[sp]: (n_freqs,) complex.
    predict_live.py imports this, so training and live use identical features."""
    parts = []
    for sp in s_params:
        d = np.atleast_2d(S[sp]) - ref[sp]
        parts += [d.real, d.imag]
    return np.hstack(parts)


# =========================== models ========================================
def model_zoo() -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)),
        "LDA": lambda: make_pipeline(StandardScaler(),
                                     LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                        n_jobs=-1, random_state=SEED),
    }


def balanced_index(labels: np.ndarray, rng) -> np.ndarray:
    """Keep every occupied sweep; use only as many empty sweeps as the biggest cell has."""
    idx = np.arange(len(labels))
    emp, occ = idx[labels == "empty"], idx[labels != "empty"]
    if len(occ) == 0:
        return idx
    cap = max(Counter(labels[occ]).values())
    keep_e = rng.choice(emp, min(cap, len(emp)), replace=False) if len(emp) else emp
    return np.sort(np.r_[keep_e, occ])


def smooth(pred: np.ndarray, n: int = SMOOTH_SWEEPS) -> np.ndarray:
    """Majority vote over the last n predictions, like the live display."""
    out, win = [], []
    for p in pred:
        win.append(p)
        del win[:-n]
        out.append(Counter(win).most_common(1)[0][0])
    return np.array(out)


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    occ = y != "empty"
    return {"all": float(np.mean(p == y)),
            "presence": float(np.mean((p != "empty") == occ)),
            "cell": float(np.mean(p[occ] == y[occ])) if occ.any() else float("nan"),
            "n": int(len(y))}


# =========================== evaluation ====================================
def make_folds(sessions: list[dict]) -> list[tuple[str, list, list]]:
    """(description, train parts, test parts); a part is (session index, mask)."""
    if len(sessions) >= 2:
        return [(f"test on {s['name']}",
                 [(j, sessions[j]["usable"]) for j in range(len(sessions)) if j != i],
                 [(i, np.ones(len(s["t"]), bool))]) for i, s in enumerate(sessions)]
    s = sessions[0]
    occ_t = s["t"][s["usable"] & (s["label"] != "empty")]
    split = float(np.median(occ_t))
    early, late = s["t"] < split, s["t"] >= split
    return [(f"first half -> second half (split at {split:.0f} s)",
             [(0, s["usable"] & early)], [(0, late)]),
            (f"second half -> first half (split at {split:.0f} s)",
             [(0, s["usable"] & late)], [(0, early)])]


def evaluate(sessions, feats, model_names, log) -> dict:
    zoo = model_zoo()
    rng = np.random.default_rng(SEED)
    folds = make_folds(sessions)
    results = {m: {"folds": [], "y": [], "p": []} for m in model_names}
    for desc, train_parts, test_parts in folds:
        Xtr = np.vstack([feats[i][m] for i, m in train_parts])
        ytr = np.concatenate([sessions[i]["label"][m] for i, m in train_parts])
        if len(set(ytr)) < 2:
            log(f"  {desc}: skipped (training part has fewer than 2 classes)")
            continue
        b = balanced_index(ytr, rng)
        log(f"\n  {desc}")
        log(f"    train: {len(b)} sweeps ({len(ytr) - len(b)} extra empty sweeps left out for balance)")
        for name in model_names:
            t0 = time.time()
            model = zoo[name]()
            model.fit(Xtr[b], ytr[b])
            ys, ps = [], []
            for i, m in test_parts:               # predict in time order, smooth, then score usable
                pred = smooth(model.predict(feats[i][m]))
                keep = sessions[i]["usable"][m]
                ys.append(sessions[i]["label"][m][keep])
                ps.append(pred[keep])
            y, p = np.concatenate(ys), np.concatenate(ps)
            sc = scores(y, p)
            results[name]["folds"].append(sc)
            results[name]["y"].append(y)
            results[name]["p"].append(p)
            log(f"    {name:<14s} all {sc['all']:5.1%}   anyone there? {sc['presence']:5.1%}   "
                f"which cell {sc['cell']:5.1%}   ({time.time() - t0:.1f} s)")
    return results


def confusion_text(y: np.ndarray, p: np.ndarray) -> str:
    classes = [c for c in CLASS_ORDER if c in set(y) | set(p)]
    classes += sorted((set(y) | set(p)) - set(classes))
    lines = ["rows = true position, columns = model's guess, % of each row",
             "         " + "".join(f"{c[:5]:>6s}" for c in classes)]
    for c in classes:
        row = np.array([np.sum((y == c) & (p == d)) for d in classes])
        if row.sum() == 0:
            continue
        lines.append(f"{c:>8s} " + "".join(f"{100 * v / row.sum():6.0f}" for v in row) + f"   n={row.sum()}")
    return "\n".join(lines)


# =========================== main ==========================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Train and test the cell classifier")
    ap.add_argument("--data", default=DATA_DIR, help="folder that holds the session folders")
    ap.add_argument("--sessions", nargs="*", help="session folder names (default: all)")
    ap.add_argument("--model", choices=list(model_zoo()), help="save this model type "
                    "(default: the one that tested best)")
    ap.add_argument("--quick", action="store_true", help="skip random forest and kNN (faster)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    import sklearn

    data = Path(args.data)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = data / "models"
    out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report.append(msg)

    paths = find_sessions(data, args.sessions)
    if not paths:
        sys.exit(f"no session folders with sweeps.csv in {data}")
    log(f"=== training run {stamp} ===")
    sessions, feats, refs = [], [], []
    for p in paths:
        t0 = time.time()
        try:
            s = load_session(p)
            ref = empty_reference(s)
        except ValueError as e:
            log(f"skipping {p.name}: {e}")
            continue
        if sessions and (s["freqs"] != sessions[0]["freqs"] or s["s_params"] != sessions[0]["s_params"]):
            log(f"skipping {p.name}: different frequencies or S-parameters from {sessions[0]['name']}")
            continue
        sessions.append(s)
        refs.append(ref)
        feats.append(make_features(s["S"], ref, s["s_params"]))
        c = Counter(s["label"][s["usable"]])
        log(f"{p.name}: {len(s['t'])} sweeps, {int(s['usable'].sum())} usable "
            f"({' '.join(f'{k}:{c[k]}' for k in CLASS_ORDER if c[k])})  [{time.time() - t0:.0f} s to load]")
    if not sessions:
        sys.exit("no usable sessions")
    s_params, freqs = sessions[0]["s_params"], sessions[0]["freqs"]
    log(f"features: {' + '.join(s_params)}, {len(freqs)} frequencies, minus the empty room "
        f"(first {REF_SECONDS:.0f} s of empty sweeps in each session)")

    names = ["logistic", "LDA"] if args.quick else list(model_zoo())
    log("\n--- honest test: models never see the part they are tested on ---")
    log(f"(chance level: about {1 / 11:.0%} for 11 classes; scores use a "
        f"{SMOOTH_SWEEPS}-sweep majority vote like the live display)")
    res = evaluate(sessions, feats, names, log)

    log("\n--- average over the tests ---")
    summary = {}
    for name in names:
        f = res[name]["folds"]
        if not f:
            continue
        summary[name] = {k: float(np.nanmean([x[k] for x in f])) for k in ("all", "presence", "cell")}
        m = summary[name]
        log(f"  {name:<14s} all {m['all']:5.1%}   anyone there? {m['presence']:5.1%}   which cell {m['cell']:5.1%}")
    if not summary:
        sys.exit("nothing could be evaluated")
    best = args.model or max(summary, key=lambda n: summary[n]["all"])
    log(f"\nmodel to save: {best}" + ("" if args.model else " (best average score)"))
    log("\nwhere it goes wrong (" + best + ", all tests pooled):")
    log(confusion_text(np.concatenate(res[best]["y"]), np.concatenate(res[best]["p"])))

    # --- final model on everything
    X = np.vstack([f[s["usable"]] for f, s in zip(feats, sessions)])
    y = np.concatenate([s["label"][s["usable"]] for s in sessions])
    b = balanced_index(y, np.random.default_rng(SEED))
    model = model_zoo()[best]()
    model.fit(X[b], y[b])
    bundle = {"model": model, "model_type": best, "classes": [str(c) for c in model.classes_],
              "s_params": s_params, "freqs_mhz": freqs, "ref_seconds": REF_SECONDS,
              "smooth_sweeps": SMOOTH_SWEEPS, "sessions": [s["name"] for s in sessions],
              "test_scores": summary, "sklearn_version": sklearn.__version__, "created": stamp}
    import joblib
    mpath = out / f"model_{stamp}.joblib"
    joblib.dump(bundle, mpath)
    shutil.copyfile(mpath, out / "model_latest.joblib")
    log(f"\nsaved {mpath}\n  and {out / 'model_latest.joblib'} (used by predict_live.py)")
    (out / f"report_{stamp}.txt").write_text("\n".join(report), encoding="utf-8")
    (out / f"report_{stamp}.json").write_text(json.dumps(
        {"sessions": bundle["sessions"], "model_type": best, "test_scores": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
