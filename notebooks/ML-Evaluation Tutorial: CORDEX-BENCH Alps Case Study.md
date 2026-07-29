# ML-Evaluation Tutorial: CORDEX-BENCH Alps Case Study

This tutorial demonstrates how to use the [ml-evaluation](https://github.com/c3s-cordex-ml/ml-evaluation) package to evaluate machine-learning RCM (Regional Climate Model) emulators, using the **CORDEX-BENCH Alps** domain as a case study. We walk through the full workflow: running the `ml-eval` command-line tracks (`fast`, `slow`, `signal`, `prob`) on a set of emulator predictions, and turning the resulting metric stores into **scorecards** with `ml_evaluation.viz`.

The emulators scored here (UNET-MULTI, UNET-SINGLE, RESDIFF, CPMGEM) are trained and run with [deep4production](https://github.com/SantanderMetGroup/deep4production) on the *same* case study, so the two packages compose end to end: `d4p-train` → `d4p-downscale` → `ml-eval` → scorecard.

> **Where the predictions come from.** This tutorial starts from NetCDF predictions that already exist on disk. To (re)produce them yourself, follow the deep4production tutorials on the same Alps case:
>
> - **UNET-MULTI / UNET-SINGLE** — the deterministic SongUNet regressor (`trainer_song_unet_det`): [ResDiff tutorial, section 6.1](https://github.com/SantanderMetGroup/deep4production/blob/master/notebooks/Deep4Production%20Tutorial%3A%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md#61-step-1--train-the-deterministic-songunet-regressor).
> - **RESDIFF** — the two-stage CorrDiff-style residual diffusion emulator: [ResDiff tutorial](https://github.com/SantanderMetGroup/deep4production/blob/master/notebooks/Deep4Production%20Tutorial%3A%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md).
> - **CPMGEM** — the single-stage (direct) diffusion emulator: [CPMGEM tutorial](https://github.com/SantanderMetGroup/deep4production/blob/master/notebooks/Deep4Production%20Tutorial%3A%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM.md).
> - The framework itself (project structure, `d4p-create`, `d4p-inspect`, `d4p-train`, `d4p-downscale`): [DeepESD tutorial](https://github.com/SantanderMetGroup/deep4production/blob/master/notebooks/Deep4Production%20Tutorial%3A%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md).

______________________________________________________________________

## 1. Introduction

### Two core concepts

- **Indices** describe *one* simulation `x` — its climatological mean, a percentile, wet-day frequency, an annual maximum, a lag-1 autocorrelation.
- **Diagnostics** compare *two* simulations: the reference `x0` (ground truth — the target RCM) against the prediction `x1` (the emulator output) — e.g. mean bias, RMSE, RALSD, Wasserstein distance.

`ml-evaluation` implements the indices and diagnostics recommended by [CORDEX-ML-Bench](https://github.com/WCRP-CORDEX/ml-benchmark) and extends them with the additional evaluation areas defined in the C3S2_384 deliverable D384.3.1.1 (climate-change signal, multivariate consistency, probabilistic skill).

### Installation

Either environment manager works; both give you the `ml-eval` executable.

```bash
# Option A — pixi
pixi install
pixi shell -e dev

# Option B — conda
conda env create -f environment.yaml
conda activate rcm-emulation
pip install -e .
```

Check the CLI is on the PATH:

```bash
ml-eval --help
ml-eval fast --help
```

Anywhere below, `ml-eval <track>` can be replaced by `pixi run -e dev python -m ml_evaluation.cli <track>` if you prefer not to activate an environment.

______________________________________________________________________

## 2. Case study: CORDEX-BENCH Alps

The same simplified CORDEX-BENCH configuration used by the deep4production tutorials, extended to the **emulation** test periods:

- **Domain:** Central Europe (Alps), 128 × 128 grid boxes at 0.11°
- **Target RCM (ground truth):** CNRM-CM5-ALADIN-63, referred to as **RCM**
- **Predictors:** upscaled RCM (**UPSRCM**, 16 × 16, `u/v/t/q/z` at 850/700/500 hPa) for the *perfect* framework; the raw driving GCM (CNRM-CM5) for the *imperfect* framework
- **Predictands:** `pr` (precipitation, mm day⁻¹) and `tasmax` (daily maximum temperature, K)
- **Training:** 1961-1980 · **Test:** historical 1981-2000 and mid-century 2041-2060

Four emulators are scored, all trained on the same data:

| Label | What it is | deep4production recipe |
|---|---|---|
| `UNET-SINGLE` | deterministic SongUNet, **one network per predictand** | `trainer_song_unet_det`, one run per variable |
| `UNET-MULTI` | deterministic SongUNet, **one network for all predictands** | `trainer_song_unet_det`, multi-channel output |
| `RESDIFF` | two-stage residual diffusion (regressor + EDM residual) | `trainer_resdiff` |
| `CPMGEM` | single-stage (direct) diffusion | `trainer_cpmgem` |

`UNET-SINGLE` is the **baseline** the skill scorecards are referenced against; the `RCM` itself is published as a fifth "simulation" and is the reference row of the climate-change-signal scorecards.

______________________________________________________________________

## 3. What `ml-eval` expects, and where things live

### 3.1 Input files

Every track needs plain **NetCDF** files containing the variable named by `--var`:

- **`--x0`** — the reference (target RCM).
- **`--x1`** — the prediction (emulator output).

Predictions written by `d4p-downscale` **with a template** (`saving_info.template`) already have the right shape — the template carries the target grid, so prediction and reference are directly comparable:

```
Dimensions:  (member: 1, time: 7305, y: 128, x: 128)
Coordinates:
  * member   (member) int64 0
  * time     (time) datetime64[ns] 2041-01-01T12:00:00 ... 2060-12-31T12:00:00
    lon      (y, x) float64 ...
    lat      (y, x) float64 ...
Data variables:
    tasmax   (member, time, y, x) float32 ...
    pr       (member, time, y, x) float32 ...
```

Notes:

- **Spatial dims** may be `(lat, lon)` on a regular grid or `(y, x)` with 2-D `lat`/`lon` coordinates on a curvilinear one (the Alps case) — `ml-eval` detects which and reduces over the right pair.
- **`member`** is optional. A deterministic emulator has `member: 1`; a generative one has as many members as `ensemble_size` in its inference YAML. By default every member is scored separately and the metrics carry a `member` dimension; `--member-num n` scores member *n* only, and `--member-num -1` scores the ensemble mean.
- **Units** must match between `--x0` and `--x1`. Here both are `mm day-1` for `pr` and `K` for `tasmax`. Hard-coded physical thresholds (1 mm/day for wet days, 25 °C for summer days, …) are rescaled internally to the units of the data, so no manual conversion is needed. As a safety net, `fast`/`prob` divide `pr` by 86400 if the prediction looks like mm/day while the reference looks like kg m⁻² s⁻¹, and print a warning when they do.

### 3.2 Directory convention

`ml-eval` itself is **layout-agnostic** — a call needs only `--x0`, `--x1`, `--var` and `--output`. The convention below is the one used by the CORDEX-BENCH-ALPS project, and it is what `ml_evaluation.viz` assumes when you hand it a run directory (viz appends `<var>` itself):

```
CORDEX-BENCH-ALPS/
├── SOURCE_FILES/data_zenodo/ALPS_domain/
│   ├── train/                                   ESD_pseudo_reality, Emulator_hist_future
│   └── test/{historical,mid_century,end_century}/{predictors,target}/
│
├── SIMULATIONS/
│   ├── RCM/                                     the ground truth, published as a "simulation"
│   │   ├── collect_targets.sh                   copies SOURCE_FILES targets into predictions/
│   │   ├── evaluation.sh                        ml-eval signal for the RCM truth (reference row)
│   │   └── outputs/
│   │       ├── predictions/<RCM>_<GCM>_<PERIOD>.nc                        <- x0
│   │       └── evaluation/rcm/<RCM>_<GCM>_<PERIOD>/<SEASON>/<VAR>/
│   │
│   └── <EMULATOR>/                              UNET-MULTI, UNET-SINGLE, RESDIFF, CPMGEM
│       ├── train.yaml, inference.yaml           (deep4production)
│       ├── evaluation.sh                        the ml-eval driver for this emulator
│       └── outputs/
│           ├── predictions/<CKPT>_<RCM>_<TRAIN_GCM>_<DATASET>_<PERIOD>.nc  <- x1
│           └── evaluation/<CKPT>/<CKPT>_<RCM>_<TRAIN_GCM>_<DATASET>_<PERIOD>/<SEASON>/<VAR>/
│               ├── metrics_fast_*.zarr
│               ├── metrics_slow_*.zarr
│               ├── metrics_signal_*.zarr
│               └── metrics_prob_*.zarr
│
└── FIGURES/<figure>/plot.sh                     one script per figure, writes the PNG next to itself
```

Two conventions worth keeping:

1. **The evaluation run id is the prediction file's stem.** Every metric describing one prediction file sits in one directory, so a figure script can find it from the model name alone. `<DATASET>` is the 4th field: `UPS-<RCM>-<GCM>` for the perfect framework, `<GCM>` for the imperfect one.
1. **The RCM truth is a simulation like any other.** `collect_targets.sh` copies the targets into `SIMULATIONS/RCM/outputs/predictions/` under the common naming, so `--x0` is resolved exactly like `--x1`. Its name carries no `perfect`/`imperfect` field — the RCM is boundary-independent.

______________________________________________________________________

## 4. The `ml-eval` tracks at a glance

| Track | Key arguments | Metrics computed | Output stores (`.zarr`) |
|---|---|---|---|
| **`fast`** (basic) | `--x0` truth<br>`--x1` prediction<br>`--var` | RMSE (field and domain scalar); bias of climatological mean, P01, P99, annual max/min, lag-1 autocorrelation; RAPSD + RALSD.<br>`pr` also: P98wet, R01, SDII biases (+ relative-bias variants of P98wet, SDII, Rx1day) | `metrics_fast_field`<br>`metrics_fast_float`<br>`metrics_fast_psd`<br>`metrics_fast_pr_field` (`--var pr`) |
| **`slow`** (extended) | `--x0` truth<br>`--x1` prediction<br>`--var` | Bias of interannual variability and P90; Wasserstein distance (direct and histogram-based), PSS, LHD, FCSS; Pearson/Spearman correlation over time; pattern RMSE and correlation per timestep; full histograms.<br>`pr`: P90wet, R10, CDD, CWD. `tasmax`: summer/hot/ice days. `tasmin`: tropical nights, frost days | `metrics_slow_field`<br>`metrics_slow_series`<br>`metrics_slow_hist`<br>`metrics_slow_{pr,tasmax,tasmin}_field` |
| **`signal`** | `--hist` historical<br>`--fut` future<br>`--var` | Absolute (`delta`) and relative (`reldelta`) change, future − historical, of: mean, P01, P99 (`pr`: R01, SDII, Rx1day) — as maps, annual series, monthly cycle and domain scalars | `metrics_signal_field`<br>`metrics_signal_series`<br>`metrics_signal_month`<br>`metrics_signal_float` |
| **`multivar`** | `--x0`, `--x1`<br>`--var`, `--var2` | Bias of the Pearson and Spearman correlation between two predicted variables | `metrics_mv_field` |
| **`prob`** | `--x0` truth<br>`--x1` ensemble prediction<br>`--var` | CRPS, Brier score for the > P50 event, spread–error ratio (the last only when the prediction has ≥ 2 members) | `metrics_prob_field`<br>`metrics_prob_float` |

Options shared by every track: `--output` (directory, file prefix, or an explicit `.zarr` path), `--season` (`DJF`/`MAM`/`JJA`/`SON` or a month name; omit for whole-year), `--align-leap-days` (reconcile calendar mismatches), `--force-load` (pull the lazy arrays into memory — handy on small test data), `--use-dask` (parallel, out-of-core), and `--member-num` (every track except `prob`, which needs the full ensemble).

Inside every store the metrics are stacked as a single `value` variable indexed by a **`diagnostic`** coordinate, so a metric is selected with `ds["value"].sel(diagnostic="<name>")`. Stores are split by *shape*: `_field` are maps, `_series` are time axes, `_float` are domain scalars, `_hist` are histograms, `_psd` are spectra.

This tutorial covers `fast`, `slow`, `signal` and `prob`. `multivar` follows the same pattern with a second `--var2` and is only meaningful for a multivariate emulator (here it would compare, e.g., `pr` against `tasmax`).

______________________________________________________________________

## 5. The basic track: `ml-eval fast`

The cheap, single-pass statistics that feed the scorecards. One call scores one (prediction, variable, season):

All commands and scripts below are run **from the project root** (`CORDEX-BENCH-ALPS/`), so every path is relative to it.

```bash
SIM=./SIMULATIONS

ml-eval fast \
    --x0 $SIM/RCM/outputs/predictions/CNRM-ALADIN63_CNRM-CM5_2041-2060.nc \
    --x1 $SIM/UNET-MULTI/outputs/predictions/UNET-MULTI_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_2041-2060.nc \
    --var pr \
    --output $SIM/UNET-MULTI/outputs/evaluation/UNET-MULTI/UNET-MULTI_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_2041-2060/annual/pr
```

Add `--season JJA` for a summer-only version of every metric in the call (the flag is threaded into each diagnostic, so one invocation gives either the annual or the seasonal variant of the whole battery).

### What it writes

| Store | `value` dims | `diagnostic` entries |
|---|---|---|
| `metrics_fast_field.zarr` | (diagnostic, member, y, x) | `rmse_field`, `bias_mean_field`, `bias_p99_field`, `bias_p1_field`, `bias_acorr_field`, `bias_max1day_field`, `bias_min1day_field`, plus the underlying `targ_*_field` / `pred_*_field` indices |
| `metrics_fast_float.zarr` | (diagnostic, member) | `rmse_float`, `ralsd_float` |
| `metrics_fast_psd.zarr` | (diagnostic, wavenumber, member) | `psd_x0`, `psd_x1` |
| `metrics_fast_pr_field.zarr` | (diagnostic, member, y, x) | `bias_p98wet_field`, `bias_r1_field`, `bias_sdii_field`, `relbias_p98wet_field`, `relbias_sdii_field`, `relbias_rx1day_field`, plus `targ_*` / `pred_*` |

The `member` dimension is always present (length 1 for a deterministic emulator, unless `--member-num` collapsed it), so a metric of a generative emulator keeps one value per realisation. Storing the target and prediction indices next to the bias (`targ_mean_field`, `pred_mean_field`, …) means the maps behind a score can be plotted without recomputing anything.

### Reading the results back

```python
import os
import numpy as np
import xarray as xr

CELL = ("./SIMULATIONS/UNET-MULTI/outputs/evaluation/UNET-MULTI/"
        "UNET-MULTI_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_2041-2060/"
        "annual/pr")

# 1) domain scalars
floats = xr.open_zarr(os.path.join(CELL, "metrics_fast_float.zarr"))["value"]
print(floats["diagnostic"].values)                    # ['rmse_float' 'ralsd_float']
rmse = float(floats.sel(diagnostic="rmse_float"))
ralsd = float(floats.sel(diagnostic="ralsd_float"))

# 2) a bias map, and the same reduction the scorecards use: mean(|field|)
fields = xr.open_zarr(os.path.join(CELL, "metrics_fast_field.zarr"))["value"]
bias_mean_map = fields.sel(diagnostic="bias_mean_field")        # (y, x)
bias_mean = float(np.abs(bias_mean_map).mean())

# 3) precip-only diagnostics (present because --var pr)
pr = xr.open_zarr(os.path.join(CELL, "metrics_fast_pr_field.zarr"))["value"]
sdii_bias = float(np.abs(pr.sel(diagnostic="bias_sdii_field")).mean())

print(f"RMSE={rmse:.3g}  RALSD={ralsd:.3g}  |bias mean|={bias_mean:.3g}  |bias SDII|={sdii_bias:.3g}")
```

Every scorecard metric is an **error** (lower is better): spatial biases are reduced to a domain scalar as `mean(|field|)`, RMSE and RALSD are already scalars.

______________________________________________________________________

## 6. The extended track: `ml-eval slow`

The distributional and spectral diagnostics, computed per grid point. Considerably more expensive, so in this case study it is run for precipitation only:

```bash
ml-eval slow \
    --x0 $SIM/RCM/outputs/predictions/CNRM-ALADIN63_CNRM-CM5_2041-2060.nc \
    --x1 $SIM/RESDIFF/outputs/predictions/RESDIFF600_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_2041-2060.nc \
    --var pr \
    --output $SIM/RESDIFF/outputs/evaluation/RESDIFF600/RESDIFF600_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_2041-2060/annual/pr \
    --use-dask
```

Write the output to the **same leaf directory** as the `fast` call for that prediction and variable: the per-variable scorecard reads fast- and slow-track columns from one directory.

| Store | `value` dims | `diagnostic` entries |
|---|---|---|
| `metrics_slow_field.zarr` | (diagnostic, member, y, x) | `bias_annstd_field`, `bias_p90_field`, `wd_field`, `wdh_field`, `fcss_field`, `pss_field`, `lhd_field`, `pcorr_field`, `scorr_field` (+ `targ_*` / `pred_*`) |
| `metrics_slow_series.zarr` | (diagnostic, member, time) | `rmse_series`, `pcorr_series`, `scorr_series` — spatial statistics per timestep |
| `metrics_slow_hist.zarr` | (diagnostic, bins, member) | `targ_hist_series`, `pred_hist_series`, `bins_hist_series` + the domain-wide scores `wdh_float`, `fcss_float`, `pss_float`, `lhd_float` |
| `metrics_slow_pr_field.zarr` | (diagnostic, member, y, x) | `bias_p90wet_field`, `bias_r10_field`, `bias_cdd_field`, `bias_cwd_field`, `relbias_p90wet_field` (+ `targ_*` / `pred_*`) |
| `metrics_slow_tasmax_field.zarr` | (diagnostic, member, y, x) | `bias_su_field` (≥ 25 °C), `bias_hot_field` (≥ 35 °C), `bias_ice_field` (< 0 °C) |
| `metrics_slow_tasmin_field.zarr` | (diagnostic, member, y, x) | `bias_tropnight_field` (≥ 20 °C), `bias_frost_field` (< 0 °C) |

The `_hist` store is what you plot to compare *whole distributions* rather than selected quantiles:

```python
h = xr.open_zarr(os.path.join(CELL, "metrics_slow_hist.zarr"))["value"]
bins = h.sel(diagnostic="bins_hist_series")
truth = h.sel(diagnostic="targ_hist_series")
pred = h.sel(diagnostic="pred_hist_series")
wd = float(h.sel(diagnostic="wdh_float"))          # Wasserstein distance, data units
```

______________________________________________________________________

## 7. The climate-change signal: `ml-eval signal`

`signal` is deliberately **source-agnostic**: it takes a historical and a future file *for one source* and computes the change between them. It never sees the ground truth. The signal *bias* is obtained downstream by comparing two such bundles (in the scorecard), which keeps every model's signal referenced to its own historical baseline — and thereby removes any bias inherited from the driving GCM.

For one emulator:

```bash
CKPT_DIR=$SIM/RESDIFF/outputs/evaluation/RESDIFF600/RESDIFF600_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_2041-2060/annual/pr

ml-eval signal \
    --hist $SIM/RESDIFF/outputs/predictions/RESDIFF600_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_1981-2000.nc \
    --fut  $SIM/RESDIFF/outputs/predictions/RESDIFF600_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_2041-2060.nc \
    --var pr \
    --output $CKPT_DIR
```

And **once** for the RCM truth, which becomes the reference row of the signal scorecard:

```bash
ml-eval signal \
    --hist $SIM/RCM/outputs/predictions/CNRM-ALADIN63_CNRM-CM5_1981-2000.nc \
    --fut  $SIM/RCM/outputs/predictions/CNRM-ALADIN63_CNRM-CM5_2041-2060.nc \
    --var pr \
    --output $SIM/RCM/outputs/evaluation/rcm/CNRM-ALADIN63_CNRM-CM5_2041-2060/annual/pr
```

The index set is `mean`/`p01`/`p99` for most variables and `r01`/`sdii`/`rx1day` for precipitation. Each index is stored twice — an absolute `delta` and a relative `reldelta` — across four stores:

| Store | `value` dims | `diagnostic` entries |
|---|---|---|
| `metrics_signal_field.zarr` | (diagnostic, y, x) | `delta_<index>_field`, `reldelta_<index>_field` |
| `metrics_signal_series.zarr` | (diagnostic, year) | `<index>_series` — the raw annual domain-mean over hist+fut on one `year` axis |
| `metrics_signal_month.zarr` | (diagnostic, month) | `delta_<index>_month`, `reldelta_<index>_month` — the seasonally-resolved signal, always all 12 months |
| `metrics_signal_float.zarr` | (diagnostic,) | `delta_<index>_float`, `reldelta_<index>_float` — signed domain means |

Note the sign convention: unlike the skill tracks, signal diagnostics are **signed** — a delta is a change, not an error, so it is never reduced with an absolute value.

______________________________________________________________________

## 8. Probabilistic skill: `ml-eval prob`

`prob` assesses an **ensemble** prediction: whether it captures the truth (CRPS), whether it gets threshold events right (Brier score for the > P50 event), and whether its spread reflects its error (spread–error ratio, SERAT). It is a separate track because the spread statistics are undefined for a single member.

Generative emulators produce an ensemble by drawing several samples per date. In deep4production this is one line of the inference YAML:

```yaml
##### ENSEMBLE SIZE #####
# Each reverse-diffusion trajectory draws fresh noise, so ensemble_size > 1
# produces independent realisations of the same day.
ensemble_size: 10
```

Re-run `d4p-downscale ./resdiff/inference.yaml` and the prediction comes back with `member: 10`. Then:

```bash
ml-eval prob \
    --x0 $SIM/RCM/outputs/predictions/CNRM-ALADIN63_CNRM-CM5_2041-2060.nc \
    --x1 $SIM/RESDIFF/outputs/predictions/RESDIFF600_CNRM-ALADIN63_CNRM-CM5_UPS-CNRM-ALADIN63-CNRM-CM5_2041-2060.nc \
    --var pr \
    --output $CKPT_DIR
```

| Store | `value` dims | `diagnostic` entries |
|---|---|---|
| `metrics_prob_field.zarr` | (diagnostic, y, x) | `crps_mean_field`, `bs50p_field`, `se_ssrat_field` |
| `metrics_prob_float.zarr` | (diagnostic,) | `crps_mean_float`, `bs50p_float`, `se_ssrat_float` |

Reading the interpretation right:

- **CRPS** and **BS** are errors — lower is better. On a **single-member** prediction (the deterministic UNETs) they still compute, degenerating to the mean absolute error and the 0/1 indicator loss respectively, which is what makes deterministic and generative emulators comparable on these two columns. The CLI prints a warning and drops the `se_ssrat_*` diagnostics in that case.
- **SERAT** is not an error: it is best at **1**. Under-dispersive ensembles score below 1, over-dispersive above. When ranking models on it, transform first — `|log10(SERAT)|` turns it into a lower-is-better quantity.

```python
p = xr.open_zarr(os.path.join(CELL, "metrics_prob_float.zarr"))["value"]
print(p["diagnostic"].values)
crps = float(p.sel(diagnostic="crps_mean_float"))
serat = float(p.sel(diagnostic="se_ssrat_float"))
print(f"CRPS={crps:.3g} mm/day   SERAT={serat:.2f} "
      f"({'under' if serat < 1 else 'over'}-dispersive)")
```

______________________________________________________________________

## 9. Scoring a whole emulator in one script

In practice you never call `ml-eval` by hand: each emulator carries an `evaluation.sh` that loops the tracks over its checkpoints and variables and writes everything into its own `outputs/evaluation/` tree. Condensed, that driver is:

```bash
#!/bin/bash
# SIMULATIONS/RESDIFF/evaluation.sh — score one emulator on the reference case.
# Run from the project root: bash SIMULATIONS/RESDIFF/evaluation.sh
SIM_DIR=./SIMULATIONS

# ---------------------------- PARAMETERS ------------------------------------
EMULATOR="RESDIFF"
CHECKPOINTS=("RESDIFF600")     # prediction-file prefixes, one per checkpoint to score
RCM="CNRM-ALADIN63"            # target RCM
TRAIN_GCM="CNRM-CM5"           # GCM the emulator was trained on
DRIVING_GCM="CNRM-CM5"         # GCM driving the case being scored
CASE="perfect"                 # perfect = upscaled RCM input, imperfect = raw GCM
PERIOD="2041-2060"
HIST_PERIOD="1981-2000"
VARS=("pr" "tasmax")
SLOW_VARS=("pr")               # the extended track only where it is worth the cost
SEASON="annual"

# ---------------------------- RESOLVE PATHS ---------------------------------
DATASET="$([ "$CASE" = perfect ] && echo "UPS-${RCM}-${DRIVING_GCM}" || echo "${DRIVING_GCM}")"
[ "$SEASON" = "annual" ] && SEASON_FLAG="" || SEASON_FLAG="--season $SEASON"

rcm_file()  { echo "$SIM_DIR/RCM/outputs/predictions/${RCM}_${DRIVING_GCM}_${1}.nc"; }
pred_file() { echo "$SIM_DIR/$EMULATOR/outputs/predictions/${1}_${RCM}_${TRAIN_GCM}_${DATASET}_${2}.nc"; }
eval_dir()  { echo "$SIM_DIR/$EMULATOR/outputs/evaluation/$1/${1}_${RCM}_${TRAIN_GCM}_${DATASET}_${PERIOD}/$SEASON/$2"; }

REF="$(rcm_file "$PERIOD")"

# ---------------------------- RUN -------------------------------------------
for CKPT in "${CHECKPOINTS[@]}"; do
    PRED="$(pred_file "$CKPT" "$PERIOD")"
    PRED_HIST="$(pred_file "$CKPT" "$HIST_PERIOD")"

    for var in "${VARS[@]}"; do
        OUT="$(eval_dir "$CKPT" "$var")"; mkdir -p "$OUT"

        # basic track — the six scorecard metrics
        ml-eval fast --x0 "$REF" --x1 "$PRED" --var "$var" --output "$OUT" $SEASON_FLAG

        # extended track — precipitation only
        if [[ " ${SLOW_VARS[*]} " == *" $var "* ]]; then
            ml-eval slow --x0 "$REF" --x1 "$PRED" --var "$var" --output "$OUT" $SEASON_FLAG
        fi

        # climate-change signal — this emulator against its OWN historical run
        [ -f "$PRED_HIST" ] && \
            ml-eval signal --hist "$PRED_HIST" --fut "$PRED" --var "$var" --output "$OUT" $SEASON_FLAG

        # probabilistic skill — only meaningful for an ensemble prediction
        # ml-eval prob --x0 "$REF" --x1 "$PRED" --var "$var" --output "$OUT" $SEASON_FLAG
    done
done
```

All four tracks land in the **same leaf directory** per (checkpoint, variable), which is exactly what the visualisation entry points expect.

______________________________________________________________________

## 10. Scorecards

`ml_evaluation.viz` reads the Zarr stores back and renders the figures — metrics are computed once and re-visualised many times. The scorecards form a 2 × 2: **skill** (error against the RCM truth) or **signal** (climate-change change), for **one variable** (a models × quantities heatmap) or **across variables** (a models × variables grid of hexagons).

|  | one variable | across variables |
|---|---|---|
| **skill** | `plot_var_skill_scorecard` | `plot_multivar_skill_scorecard` |
| **signal** | `plot_var_signal_scorecard` | `plot_multivar_signal_scorecard` |

All of them locate data the same way: you pass an explicit **`{label: path}`** mapping, where `path` is one run's `<SEASON>` directory and viz appends `<var>` itself. Nothing is searched and no layout is assumed — the label is what appears in the figure.

Both examples below share this preamble, which turns the naming convention of §3.2 into paths:

```python
import os

# run from the project root: every path below is relative to it
SIM = os.path.join(".", "SIMULATIONS")
FIGURES = os.path.join(".", "FIGURES")

RCM, TRAIN_GCM, TEST_GCM = "CNRM-ALADIN63", "CNRM-CM5", "CNRM-CM5"
CASE, PERIOD, SEASON = "perfect", "2041-2060", "annual"

# the input dataset is the 4th field of every prediction / run name
DATASET = f"UPS-{RCM}-{TEST_GCM}" if CASE == "perfect" else TEST_GCM


def run_dir(sim, ckpt, period=PERIOD):
    """One run's season dir; viz appends <var>."""
    stem = f"{ckpt}_{RCM}_{TRAIN_GCM}_{DATASET}_{period}"
    return os.path.join(SIM, sim, "outputs", "evaluation", ckpt, stem, SEASON)


# label -> (simulation dir, checkpoint tag). The tag is the prediction-file
# prefix, which is not always the dir name: for the diffusion emulators it
# carries the checkpoint picked by the tuning scorecards.
MODELS = [
    ("UNET-SINGLE", "UNET-SINGLE", "UNET-SINGLE"),
    ("UNET-MULTI",  "UNET-MULTI",  "UNET-MULTI"),
    ("RESDIFF",     "RESDIFF",     "RESDIFF600"),
    ("CPMGEM",      "CPMGEM",      "CPMGEM600"),
]
paths = {label: run_dir(sim, ckpt) for label, sim, ckpt in MODELS}

# drop sources with no metrics on disk, saying which
for label, path in list(paths.items()):
    if not os.path.isdir(path):
        print(f"[plot] skip {label}: {path} not found")
        del paths[label]
```

### 10.1 Multivariate skill scorecard — `plot_multivar_skill_scorecard`

One figure: **rows are models, columns are variables**, and each cell is a hexagon whose six wedges are the six `fast`-track metrics. Each wedge is coloured by the percentage improvement over a reference model,

```
100 × (err_reference − err_model) / err_reference
```

so blue = better than the baseline, red = worse, white = equal — and the reference row is blank by construction. This is the "at a glance" view: which model wins, on which variable, and on which aspect of performance.

```python
from ml_evaluation.viz.plot_hexagon import plot_multivar_skill_scorecard

print(plot_multivar_skill_scorecard(
    paths,
    os.path.join(FIGURES, "scorecard_hexagon_skill"),
    ["tasmax", "pr"],
    reference="UNET-SINGLE",     # the baseline row, moved to the top
    season=SEASON,               # label only: title and file name
))
```

The wedges, clockwise from the top, are `Bias lag-1 corr`, `RALSD`, `Bias P99`, `Bias mean`, `Bias P01`, `RMSE` — and for `pr` the three distribution wedges become `Bias Rx1day`, `Bias SDII`, `Bias R01`, because a zero-bounded, intermittent variable is not well described by its mean and its extreme quantiles. `hexagon_axes(var)` / `scorecard_axes(var)` return the exact wedge list for a variable.

Useful knobs: `clip` (colour saturation, default ± 100 %), `cmap`, `title`, `filename`. Order the `paths` mapping to control row order; the `reference` is always moved to the top.

> A related entry point, `plot_hexagons(paths, out, variables, reference=..., season=...)`, draws one radar hexagon **per variable** instead of the grid — the same metrics, plotted as overlaid model traces. With `reference=None` each axis is min-max normalised across the plotted models (centre = best in set) instead of showing % improvement.

### 10.2 Univariate skill scorecard — `plot_var_skill_scorecard`

Focusing on **one variable at a time** buys room for detail: every metric becomes its own column, showing its raw value rather than only a colour. We use **precipitation** as the example, the variable where the emulators differ most.

- **colour** — the per-column min-max normalised score: 1 = best (lowest error) of the plotted models, 0 = worst. Every column is an error, so scores are comparable across columns whose units are not (RMSE in mm/day, RALSD dimensionless, R01 a frequency).
- **number** — the raw metric value, in its own units, with the spatial **coefficient of variation** `CV = std(field) / mean(|field|)` printed underneath. The CV answers "is this domain scalar representative?": a small mean bias with a large CV is not a uniformly good field, it is compensating errors. It is omitted for metrics that only exist as a float (RALSD).

```python
from ml_evaluation.viz.plot_var_skill_scorecard import plot_var_skill_scorecard

METRICS = [
    "rmse", "bias_mean", "bias_p01", "bias_p99", "bias_rx1day",
    "bias_r01", "bias_sdii", "bias_p98wet", "ralsd", "bias_acorr",
]

print(plot_var_skill_scorecard(
    paths,
    os.path.join(FIGURES, "scorecard_pr_skill"),
    ["pr"],
    metrics=METRICS,             # columns, in order
    season=SEASON,
))
```

Passing `metrics=None` falls back to a sensible per-variable default (7 precipitation columns, 6 otherwise). The full menu is in `METRIC_LABELS`; besides the fast-track keys used above it exposes the **slow-track** columns `wd`, `wdh`, `fcss`, `lhd`, `bias_annstd`, `bias_p90` — available as soon as §6's `ml-eval slow` has run into the same directory. Only error-type diagnostics are exposed: `pss`, `pcorr` and `scorr` are higher-is-better and would score inverted in a scorecard that treats every column as an error. To rename a header, pass a `(metric_key, label)` pair instead of a bare key.

For finer control — a unit conversion, an extra row, a custom row order — split the wrapper into its two halves and post-process in between:

```python
from ml_evaluation.viz.plot_var_skill_scorecard import (
    read_var_skill_metrics, render_var_skill_scorecard,
)

per_model = {}
for label, base in paths.items():
    d = os.path.join(base, "pr")
    if os.path.isdir(d):
        per_model[label] = read_var_skill_metrics(d, "pr", METRICS)   # {key: (value, cv)}

# e.g. scale the precipitation-amount columns if the files are in kg m-2 s-1
# (here both truth and predictions are already mm/day, so the factor is 1.0)
AMOUNT = {"rmse", "bias_mean", "bias_p01", "bias_p99", "bias_rx1day",
          "bias_sdii", "bias_p98wet", "wd", "wdh", "bias_annstd", "bias_p90"}
scale = 1.0
per_model = {
    m: {k: ((v * scale if k in AMOUNT else v), cv) for k, (v, cv) in vals.items()}
    for m, vals in per_model.items()
}

print(render_var_skill_scorecard(
    per_model, "pr",
    os.path.join(FIGURES, "scorecard_pr_skill"),
    models=list(per_model), metrics=METRICS, season=SEASON,
))
```

`read_var_skill_metrics` (like `read_hexagon_metrics` for the hexagons) depends only on ml-eval's output format, not on any directory convention — so it is reusable from any project's own plotting code.

### 10.3 The signal scorecards, briefly

The two signal entry points consume the `metrics_signal_*.zarr` stores of §7 and follow the same `{label: path}` interface, with one addition: the **RCM truth** is passed separately, since it is the reference every model's signal is compared against.

```python
from ml_evaluation.viz.plot_multivar_signal_scorecard import plot_multivar_signal_scorecard

rcm_path = os.path.join(SIM, "RCM", "outputs", "evaluation", "rcm",
                        f"{RCM}_{TEST_GCM}_{PERIOD}", SEASON)

print(plot_multivar_signal_scorecard(
    paths,
    os.path.join(FIGURES, "scorecard_hexagon_signal"),
    ["tasmax", "pr"],
    reference="RCM", reference_path=rcm_path,
    season=SEASON,
    units={"pr": "%", "tasmax": "K"},   # precip as a relative change, temperature absolute
))
```

`plot_var_signal_scorecard` is its one-variable counterpart. And `plot_series` is the time-resolved view of the same runs — one variable, one index, one line per source, read from `metrics_signal_series.zarr` on the stored `year` axis:

```python
from ml_evaluation.viz.plot_series import plot_series   # see series_indices(var) for the index keys
```

______________________________________________________________________

## 11. Summary

You have now run the full evaluation workflow on CORDEX-BENCH Alps:

- **`ml-eval fast`** — the cheap, generally applicable metrics that feed the scorecards (RMSE, biases of mean / percentiles / extremes / autocorrelation, RALSD).
- **`ml-eval slow`** — the extended distributional and spectral diagnostics, plus the variable-specific threshold indices.
- **`ml-eval signal`** — each source's own climate-change signal, so model signals are compared free of any bias inherited from the driving GCM.
- **`ml-eval prob`** — CRPS, Brier score and spread–error ratio for ensemble (generative) predictions.
- **`ml_evaluation.viz`** — `plot_multivar_skill_scorecard` for the models × variables overview and `plot_var_skill_scorecard` for the per-variable detail, both reading the Zarr stores back without recomputing anything.

Two habits make the rest easy: write every track of one (prediction, variable, season) into **one leaf directory**, and name that directory after the prediction file's stem. Everything downstream — scorecards, time series, spatial maps — then locates its inputs from the model name alone.

______________________________________________________________________

## 12. References

- [ml-evaluation](https://github.com/c3s-cordex-ml/ml-evaluation) — this package
- [deep4production](https://github.com/SantanderMetGroup/deep4production) — training and inference of the emulators scored here
- [CORDEX-ML-Bench](https://github.com/WCRP-CORDEX/ml-benchmark) — the benchmark this evaluation framework builds on
- [CORDEX-BENCH Alps data (Zenodo)](https://zenodo.org/records/17957264)
- Rampal, N., González-Abad, J., Addison, H., et al., 2026: *CORDEX-ML-Bench: A benchmark for data-driven regional climate downscaling — Experiment design and overview*, submitted to JAMES.
- Kendon, E. J., et al., 2025: *Potential for machine learning emulators to augment regional climate simulations in provision of local climate change information*. Bull. Amer. Meteor. Soc., 106, E1175–E1203.
- C3S2_384 deliverable D384.3.1.1 — *Description of the evaluation methodology and minimum quality requirements for the ML-generated datasets* (full metric definitions).
