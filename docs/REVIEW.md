# P0 code review (2026-09-23)

Scope: `requirements.txt`, `slam/config.py`, `slam/vna.py`, `slam/collect.py`,
`slam/analyze_sensitivity.py`, `slam/sim/digital_twin.py`, `docs/digital_twin.md`,
checked against `docs/CONTRACT.md`. Legacy folders untouched. No commits.

## What was run (fresh venv, Python 3.11) — all PASS

| Step | Result |
|---|---|
| `pip install -r requirements.txt` in a fresh venv | all pins install (numpy 2.2.6, pandas 2.3.2, matplotlib 3.10.9, pyvisa 1.16.2, ...) |
| pyflakes on every new `.py`; import of all five modules | clean |
| `python -m slam.config`, `python -m slam.vna --mock` | 201 pts, S11/S22 ≈ −22 dB, S21 ≈ −45 dB |
| `collect --mock`: 20 empty + 8 × {center, A1, E5, outside, near_ant1} + interactive `C3 3`, ENTER, `q` | 66 CSVs == 66 manifest rows, header == `MANIFEST_COLUMNS`, `session_info.json` valid, filenames `<label>_YYYYmmdd_HHMMSS_ffffff.csv` |
| `analyze_sensitivity --session mock_sens` | runs; every ratio 0.88–0.95 (≈ sqrt(π/4) for pure noise) — correct, MockVNA has no person |
| `digital_twin --session sim_all --labels all --n 5 --seed 0` | 150 sweeps in **0.7 s**; 201 rows/CSV; manifest/session_info in collect format |
| `analyze_sensitivity --session sim_all`, `--sweep-plot` | GO verdict, 15/29 positions S21-detectable |
| `--make-fixture` + `analyze_sensitivity --session fixture` | GO 4/5; near_ant1 S11 ratio 19, outside ≈ 1 |
| All 13 PNGs opened | axes labelled with units, legends outside the data, nothing empty |
| Physics spot-checks (sim) | empty \|S21\| −43…−47 dB; noise −70.0 dB (empty RMS −70…−72 dB); center \|ΔS21\| −54.7 dB > \|ΔS11\| −60.3 dB; near_ant1 \|ΔS11\| −52 dB (strongest); E5 \|ΔS22\| −51 dB; `NUM_POINTS = 201` |
| `docs/digital_twin.md` | table/headers render; 3 URLs spot-checked (arXiv 2507.19653, ITU-R P.2040-3 PDF, Sionna RT intro) — all resolve and say what the doc claims |

Contract conformance: public names, CLI flags, CSV/manifest columns, session_info keys,
grid convention (A1 = (0,0), rows A–E along y, cols 1–5 along x) and the ANTENNAS/ROOM dict
keys all match the contract and each other (`collect` → `configure()` kwargs, twin → `analyze`,
`sweep_to_csv`/pandas → `load_sweep_csv`).

## Bugs found and fixed

| File | Symptom | Fix |
|---|---|---|
| `slam/vna.py` `load_sweep_csv` | `float('')` crash on an empty cell (pandas `na_rep=''`, as `--make-fixture` writes) | `_cell()` maps `''`/`nan`/`NA` → `nan`; tested with a pandas-written file |
| `slam/vna.py` `VNA.connect` | `collect.py` called `open_vna()` (connects) and then `connect()` again → on real hardware a second VISA session is opened and the first leaks | `connect()` is now idempotent; `collect.py` uses `vna.idn` instead of reconnecting |
| `slam/vna.py` `MockVNA` + `slam/collect.py` | `seed=0` default meant every `collect --mock` process produced bit-identical sweeps, so A1/E5/center/outside/near_ant1 had identical data and overlaid exactly in the figures | `MockVNA(seed=None)` draws fresh noise (scene geometry stays fixed across processes); `collect` passes `seed=None` for the mock |
| `slam/collect.py` | `--label c3` resolved to C3's coordinates but was stored as label `c3` / file `c3_…csv`, splitting one position into two labels in analysis | `canonical_label()` normalises case for known labels (CLI and interactive) |
| `slam/sim/digital_twin.py` `TwinParams` | Default drift (0.15 dB, 6°/h) multiplies the −20 dB static S11/S22 → −52 dB "delta" on an *empty* room after ~8 min; swamped the person's S11/S22 signal, made `outside` S11-"detectable" (ratio 7.4) and contradicted the doc's S11 claims | Defaults lowered to 0.02 dB / 1°/h (warmed-up VNA); comment explains and gives the `--set` to restore |
| `slam/sim/digital_twin.py` manifest | `outside` rows had `x_m=-0.7,y_m=2.0` whereas `collect.py`/config write blanks | manifest x/y now come from `config.label_to_xy`; true simulated xy stays in `notes` |
| `slam/analyze_sensitivity.py` figures | with 29 labels the right-hand legend squashed fig 1 to a strip; bar chart was 30 in wide | legends with > 12 entries go below the axes in columns; bar-chart width capped at 18 in |
| `docs/digital_twin.md` | tilt described as "aims at the centre" (only azimuth does; tilt is fixed −30°); "center delta grows by ~1 dB" at 1.6 m (twin gives < 1 dB); no mention of drift/noise defaults; Phase-A numbers | corrected; added a Nuisances bullet and the measured Phase-A numbers |

## Residual concerns — must be verified on real hardware

SCPI paths in `vna.py` could not be exercised here (only MockVNA ran). Check on the E5071C:
- `configure()`: `SENS1:FREQ:STAR/STOP`, `SENS1:SWE:POIN`, `SENS1:BWID`, `SENS1:AVER:COUN`/`SENS1:AVER ON|OFF`,
  `TRIG:AVER ON|OFF`, `SENS1:AVER:CLE`, `SOUR1:POW`, `CALC1:PAR:COUN`, `CALC1:PARn:DEF Sxx`, `TRIG:SOUR INT`,
  `INIT1:CONT OFF`, `FORM:DATA ASC`, `DISP:WIND1:ACT`, `CALC1:PAR1:SEL`, then `SYST:ERR?` drain.
- `sweep()`: `INIT1:IMM` + `*OPC?` (with `TRIG:AVER ON` one trigger should run all 4 averages — confirm
  `*OPC?` really waits for the averaged result and the 60 s timeout is enough), `SENS1:FREQ:DATA?`,
  `CALC1:PARn:SEL` + `CALC1:DATA:SDAT?` (expects 2N interleaved re/im). `read_termination=None` + the
  `_drain` logic is copied from the legacy scripts; verify no `-420` errors and that sweep time is sane.
- `pyvisa-py` needs no extra packages for TCPIP INSTR, but if the ENA only accepts SOCKET use `--address`.
- Real VNA drift is unknown: interleave `empty` sweeps through each session (the analysis assumes the
  reference is stationary). The twin's S11/S22 conclusions depend on this.
- Everything tagged `TODO_MEASURE` in `config.py` (antenna xyz/tilt/model/gain/cable, room size,
  grid origin in room, materials) is a placeholder and is snapshotted into every `session_info.json`.
- Figures 1/2/4 with all 29 grid labels use an 8-step blue ramp and are not label-distinguishable;
  they are designed for the 5–6 label sensitivity session. The bar chart is the useful view at 29 labels.
- `analyze_sensitivity` reads the manifest with pandas: a label literally spelled `NA`/`nan` would be lost.

## How to run

```bash
cd /home/claude/SLAM
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export SLAM_DATA_DIR=/path/to/data SLAM_FIGURES_DIR=/path/to/figures   # optional; default <repo>/data, <repo>/figures
python -m slam.config                                   # print settings, check TODO_MEASURE values
python -m slam.vna                                      # real ENA: IDN, settings, sweep time   (--mock for offline)
python -m slam.collect --session s1 --label empty --person alice --n 20          # add --mock offline
python -m slam.collect --session s1 --label center --person alice --n 10
python -m slam.collect --session s1 --person alice      # interactive: "C3 10 notes", ENTER repeats, q quits
python -m slam.analyze_sensitivity --session s1         # summary.csv + 4 PNGs in $SLAM_FIGURES_DIR/s1/
python -m slam.sim.digital_twin --session sim_all --labels all --n 5 --seed 0   # ~1 s
python -m slam.sim.digital_twin --sweep-plot figures/twin_check.png
python -m slam.sim.digital_twin --session sim_cold --labels empty,center,outside --set drift_amp_db=0.15
python -m slam.analyze_sensitivity --make-fixture data/fixture && python -m slam.analyze_sensitivity --session fixture
```

`git status --short` at the end of the review: `?? docs/  ?? requirements.txt  ?? slam/`
(no data, figures or `__pycache__` written inside the repo).
