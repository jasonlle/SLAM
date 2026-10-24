# Camera-labelled RF localization

Locates a person in a 4 m × 4 m area from VNA sweeps (S11 + S21, 902–928 MHz).
A ceiling camera labels where the person is while the data is collected; the
trained model then guesses the position from the radio data alone, with the
camera off.

The area is split into a 3 × 3 grid of cells (about 1.33 m each). The model
answers one of 11 classes: `empty`, cells `1`–`9`, or `outside` (just outside
the taped square).


## The four scripts

| Script | What it does |
|---|---|
| `live_map.py` | Camera only. Live camera view with the grid drawn on it, plus a top-down map with the person's dot. Used to calibrate the floor, tune the detector and save camera settings. |
| `collect_live.py` | Camera + VNA. Sweeps continuously and saves every sweep together with the cell the camera saw the person in during that sweep. |
| `train_model.py` | Loads all collected sessions, tests several models honestly (on data they never trained on), and saves the best one. |
| `predict_live.py` | VNA + trained model. Shows the model's guess on the map. Camera off by default; `--with-camera` adds the camera as a live check with a running agreement score. |

`vna_connect_check.py` is an optional helper that diagnoses VNA connection
problems (network adapter, ping, ports, VISA backends).

The workflow is: **calibrate (`live_map.py`) → collect (`collect_live.py`) →
train (`train_model.py`) → predict (`predict_live.py`)**, then collect more and
retrain.

## Install

Python 3.10 or newer.

```
pip install -r requirements.txt
```

Use `opencv-python`, not `opencv-python-headless`; the headless version has no
windows.

Keep all the scripts in the same folder: `collect_live.py` imports
`live_map.py`, and `predict_live.py` imports all three of the others.

## Hardware setup

- **VNA:** Keysight/Agilent E5071C at `TCPIP0::192.168.1.10::inst0::INSTR`.
  The PC's Ethernet adapter needs a manual IPv4 address on the same subnet
  (e.g. 192.168.1.20, mask 255.255.255.0), not automatic DHCP. Port 1 →
  antenna 1, port 2 → antenna 2. Turn the VNA on about 30 minutes before
  collecting; it drifts while warming up.
- **Antennas:** two 900 MHz panel antennas. Don't move them, the cables or
  the cabinets between sessions; the radio "fingerprint" of the room changes
  if anything moves.
- **Camera:** a USB webcam on the ceiling looking straight down at the taped
  square. Auto exposure, auto focus and auto white balance off.
- **Floor:** a 4 m × 4 m taped square. Cell 1 is the corner next to
  antenna 1:

```
           antenna 2 (corner by cell 9)
 7 | 8 | 9
 4 | 5 | 6
 1 | 2 | 3
antenna 1 (corner by cell 1)
```

## Configure for your machine

All settings are constants at the top of each file.

**Data folder.** Sessions, calibration, camera settings and models all go in
one folder:

- default: a `data` folder next to the scripts (ignored by git);
- or set the `SLAM_DATA_DIR` environment variable. On Windows, run once and
  then open a new terminal:

  ```
  setx SLAM_DATA_DIR "D:\path\to\your\data"
  ```

- or pass `--out` (`--data` for `train_model.py`) on the command line.

**Other settings you may need to change:**

| Setting | File | Default |
|---|---|---|
| `CAMERA` | `live_map.py` | `"0"` (camera index; run `python live_map.py --probe` to compare resolutions) |
| `REQUEST_SIZE` | `live_map.py` | 1920 × 1080 |
| `GRID_SIZE_M`, `N_CELLS` | `live_map.py` | 4 m × 4 m, 3 × 3 |
| `ANTENNAS_M` | `live_map.py` | drawing only; set to your measured antenna positions |
| `VNA_ADDRESS` | `collect_live.py` | `TCPIP0::192.168.1.10::inst0::INSTR` |
| `F_START_HZ`, `F_STOP_HZ`, `N_POINTS`, `IF_BW_HZ`, `S_PARAMS` | `collect_live.py` | 902–928 MHz, 53 points, 10 kHz, S11 + S21 |
| `LABEL_MARGIN_M` | `collect_live.py` | 0.15 m (a sweep only counts if the person stayed this far inside the cell) |
| `CAMERA_LAG_S` | `collect_live.py` | 0 (camera delay; not measured yet) |

The sweep settings must be the same for collection and prediction.
`predict_live.py` refuses to run if they differ from what the model was
trained on.

## How to run

### 1. Check the camera and calibrate

```
python live_map.py
```

On the first run, click the four tape corners: start at the corner next to
antenna 1 and go around the square. Then keep the square empty for about a
second while it captures the background.

Open the camera's settings with `s`, turn off the automatic settings, close
Logi Tune if it is open, and press `k` to save them. They are applied
automatically from then on.

| Key | Action |
|---|---|
| `q` | quit |
| `b` | new background (square must be empty) |
| `c` | click the corners again |
| `s` | open the camera's settings dialog |
| `k` | keep the current camera settings |
| `m` | show the detection mask |

### 2. Collect a session

```
python collect_live.py --session 2026-10-22_A
python collect_live.py --fake-vna        # test without the VNA
```

The suggested protocol takes about 12 minutes:

1. Empty square, 60 s.
2. Round 1: cells 1 → 9, then outside. About 25 s each, moving around inside
   the cell.
3. Empty, 30 s.
4. Round 2: cells 5, 9, 1, 7, 3, 8, 2, 6, 4, then outside. About 25 s each,
   facing a different direction.
5. Empty, 30 s, then `q`.

The map shows how many usable sweeps each cell has. One person in the area at
a time; the operator sits in the same spot every session. `p` pauses saving.

Each sweep gets a label:

| Label | Meaning | Used for training |
|---|---|---|
| `empty` | nobody seen, including ±1 s around the sweep | yes |
| `1`–`9` | person stayed inside that cell, at least 15 cm from its lines | yes |
| `outside` | person stayed just outside the square | yes |
| `transition` | crossing a line, entering or leaving | no |
| `no_camera` | no camera frames for that sweep | no |

### 3. Train

```
python train_model.py            # all sessions in the data folder
python train_model.py --quick    # only the two linear models (seconds)
```

- With 2+ sessions, it trains on all but one session and tests on the
  one left out, for each session in turn.
- With 1 session, it trains on the first half and tests on the second, then
  the other way round.

It prints the scores and a confusion table, then trains the best model type on
all the data and saves it to `<data folder>/models/model_latest.joblib`.

Train on the computer that will run `predict_live.py`: a saved model only
loads reliably with the scikit-learn version that created it.

### 4. Predict

```
python predict_live.py                  # camera off: the demo
python predict_live.py --with-camera    # camera as a live check
```

Keep the square empty for the first 30 seconds while it measures the empty
room. Then:

- **Camera off:** the model's cell is filled blue.
- **With camera:** the camera's answer is the green cell with a yellow dot,
  the model's guess is a purple outline, and a line shows how often the two
  agree.

`r` measures the empty room again; `q` quits.

## Output files

```
<data folder>/
  live_calibration_camera_0.json   floor calibration (tape corners)
  camera_settings_camera_0.json    saved camera settings (k)
  <session>/
    sweeps.csv          one row per sweep: t0_s, t1_s, label, usable, x_m, y_m,
                        moved_m, n_frames, then S11/S21 real and imaginary per frequency
    camera_track.csv    one row per camera frame: time, detected, x_m, y_m
    camera.mp4          the camera video (labels can be redone later)
    session_info.json   VNA identity, sweep settings, calibration, summary
  models/
    model_latest.joblib, model_<date_time>.joblib, report_<date_time>.txt
  predict_<date_time>/
    predictions.csv, summary.json
```

All times are seconds since the start of the session, on one shared clock
for the camera and the VNA.

## How it works

- **Labels from the camera.** Background subtraction finds the person; the
  blob's center of mass is mapped to floor coordinates through the four-corner
  calibration. A sweep is labeled only if every camera frame during it puts
  the person in the same cell, away from the lines.
- **Features.** Each sweep minus the empty room (measured at the start of the
  session), as real and imaginary parts of S11 and S21.
- **Models compared.** Logistic regression, LDA, k-nearest neighbors, random
  forest. The linear models have done best so far.
- **Live display.** The shown guess is a majority vote over the last 30 sweeps
  (about half a second).

## Results so far

From session `2026-10-22_A` (one person, one day):

| Test | Anyone there? | Which cell (9 cells + outside) |
|---|---|---|
| Within the session (first half ↔ second half), LDA | 99.8% | 80% |
| Live run, new empty reference, camera as check | 91% | 74% |

Chance level is about 10%. The most common confusions are among cells 1, 4
and 7, and between 5 and 9. Using S11 and S21 together beats either one alone
(about 81% vs about 60% on the cell question), and phase matters: S21 magnitude
alone gives 52%.

## Known limitations / next steps

- Tested on one day and one person only. Next: sessions on other days, a
  second person, then leave-one-session-out results.
- The camera image ends at two of the tape lines, so the camera can miss a
  person standing just outside on those sides.
- Ceiling-camera parallax: positions near the edges of the square drift
  outward. A person-based calibration (standing on each cell center) would fix
  this.
- The camera delay (`CAMERA_LAG_S`) has not been measured yet.
