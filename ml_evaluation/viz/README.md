# `ml_evaluation.viz` — figures from the evaluation output

This directory turns the Zarr stores written by the `ml-eval` CLI back into
figures. Nothing here computes a metric: the tracks (`fast`, `slow`, `signal`,
`multivar`, `prob`) are run once and their output is re-visualised as often as
needed.

______________________________________________________________________

## Shared conventions

Every entry point follows the same three rules, so learning one is learning all
of them.

**1. Sources are an explicit `{label: path}` mapping.** The label is what
appears in the figure; `path` is that run's base directory, and one variable's
stores are read from `<path>/<var>/`. Nothing is searched and no directory
layout is assumed — the caller resolves paths, the package only reads them.

```python
paths = {
    "UNET-SINGLE": ".../evaluation/UNET-SINGLE/UNET-SINGLE_..._2041-2060/annual",
    "RESDIFF":     ".../evaluation/RESDIFF600/RESDIFF600_..._2041-2060/annual",
}
```

`normalize_paths` also accepts a sequence of `(label, path)` pairs or of
`{"label": ..., "path": ...}` dicts. The given order is the row / series order
in the figure.

**2. Loading is split from drawing.** Each module exposes three layers:

| layer | what it does |
|---|---|
| `read_*` | reads **one** directory into plain arrays. Depends only on ml-eval's output format — layout-agnostic and reusable from any project's own plotting code. |
| `render_*` | draws pre-read data. Take this route when you need to post-process between the two (unit conversion, an extra row, a custom order). |
| `plot_*` | the convenience wrapper over both — the usual entry point. |

**3. Small shared vocabulary.**

- `season` is a **label**: it goes in the title and the file name, and plays no
  part in locating data. Filter by season at the CLI (`ml-eval ... --season JJA`)
  and point the figure at that run's directory.
- `reference` names the row a figure is measured against: the baseline model for
  the skill scorecards, the RCM truth for the signal ones, the source supplying
  the truth panel for the maps. `reference_path` reads it from a directory
  outside `paths`.
- `plot_var_series` has no `member_num` — the choice is made at the CLI. An
  ensemble-mean `ml-eval signal` run stores `<index>_series_{min,max,std}` beside
  the mean series, so `spread="minmax"` / `"std"` draws a band with no extra runs.
  A label may also be given a **list of paths** (one per-member run each), which
  the figure reduces itself — the fallback for older runs.
- `variant_of={variant: base}` (series only, optional) draws a label in another
  label's colour, dashed — the way to overlay the same emulators under a second
  experiment (perfect vs imperfect boundaries) without doubling the palette.
  Defaults to None, so single-experiment figures are unaffected.
- `row_sel={label: {"n_members": N}}` (probskill only, optional) picks one
  configuration out of a store that holds several: `ml-eval prob --n-members
  "2,4,6,8,10"` stacks an ensemble-size sweep along an `n_members` dimension
  rather than splitting it across directories, so several rows can share one
  path and differ only by the size read out of it. Without it a swept store
  falls back to its largest size (the full ensemble) and says so — it is never
  averaged across sizes, which would describe no ensemble at all.
- `member_num` picks the realisation: `>= 0` that member, `< 0` the mean over
  members. Default `0` — the only member of a deterministic model. ml-eval
  scores every realisation separately, so averaging is a choice, never a silent
  default. (Not on the signal figures: `ml-eval signal` already collapses the
  ensemble, so those stores carry no `member` dimension.)
- Every entry point **returns the path(s) written**, and creates the output
  directory if needed.

______________________________________________________________________

## The figures

| Function | Module | Figure | Reads | Key args |
|---|---|---|---|---|
| `plot_multivar_skill_scorecard` | `plot_hexagon` | **one** figure: models × variables, each cell a six-wedge hexagon coloured by % error improvement over a baseline | `metrics_fast_field`, `metrics_fast_float`, `metrics_fast_pr_field` | `reference` (**required**), `season`, `clip`, `cmap`, `member_num` |
| `plot_var_skill_scorecard` | `plot_var_skill_scorecard` | one per variable: models × metrics heatmap, colour = per-column normalised score, text = raw value + spatial CV | `metrics_fast_*` + `metrics_slow_field` | `metrics` (columns), `season`, `member_num` |
| `plot_hexagons` | `plot_hexagon` | one per variable: radar hexagon, one trace per model | `metrics_fast_*` | `reference` (None → min–max mode), `season`, `member_num` |
| `plot_multivar_signal_scorecard` | `plot_multivar_signal_scorecard` | **one** figure: models × variables, hexagon of three index sectors per cell, coloured by the climate-change signal | `metrics_signal_float` | `reference`, `reference_path`, `units`, `season`, `annotate` |
| `plot_var_signal_scorecard` | `plot_var_signal_scorecard` | one per variable: models × signal quantities heatmap | `metrics_signal_float` | `indices`, `kinds` (`delta` / `reldelta`), `reference`, `season` |
| `plot_var_series` | `plot_var_series` | one per (variable, index): interannual line plot, one line per source | `metrics_signal_series` | `index`, `reference`/`driver`/`baseline` (+ `_path`), `years`, `scale`, `units` |
| `plot_multivar_skill_maps` | `plot_diagnostic_maps` | **one** figure: variables (columns) × models (rows), truth on top | any `*_field` store the metric lives in | `metric`, `kind`, `show_truth`, `reference`, `season`, `member_num` |
| `plot_var_skill_maps` | `plot_diagnostic_maps` | one per variable: truth at (1,1), models down column 2, error in column 3 | ″ | `metric`, `kind`, `show_truth`, `reference`, `season`, `member_num` |
| `plot_domain` | `plot_domain` | one map: the outline of each experiment grid over coastlines (documentation, not evaluation) | the grids' own **NetCDF** | `title`, `resolutions`, `margin`, `colors` |
| `plot_multivar_probskill_scorecard` | `plot_multivar_probskill_scorecard` | **one** figure: models × variables, each cell a triangle of CRPS / BS / SERAT for ensemble predictions | `metrics_prob_float` | `metrics` (wedges), `season`, `row_sel`, `cmap`, `annotate` |
| `plot_var_signal_maps` | `plot_diagnostic_maps` | one per variable: the RCM's own signal at (1,1), each model's signal down column 2, its signal error in column 3 | `metrics_signal_field` | `index`, `quantity` (`delta` / `reldelta`), `kind`, `reference_path` |
| `plot_multivar_signal_maps` | `plot_diagnostic_maps` | **one** figure: variables × sources, the RCM's signal on top | ″ | ″ |
| `plot_multivar_event_maps` | `plot_diagnostic_maps` | `plot_multivar_skill_maps` for a **single date** | prediction / target **NetCDF** | `date`, `kind`, `show_truth`, `reference_path`, `member_num` |
| `plot_var_event_maps` | `plot_diagnostic_maps` | `plot_var_skill_maps` for a **single date** | prediction / target **NetCDF** | ″ |

Names read `plot_{scope}_{topic}_{figure}`: **scope** is `var` (one figure per
variable) or `multivar` (all variables in one), **topic** is `skill` (error
against the RCM truth), `signal` (climate-change change) or `event` (one date),
and **figure** is `scorecard` or `maps`. `probskill` is a fourth topic, for the
ensemble diagnostics of `ml-eval prob`. `plot_var_series` is the one exception:
it produces skill or signal from the `baseline` argument at call time, so the
topic slot stays empty.

### Building blocks

| Catalogue | Reader | Renderer |
|---|---|---|
| `hexagon_axes(var)`, `scorecard_axes(var)` | `read_hexagon_metrics` | `plot_hexagon` |
| `var_skill_columns(var, metrics)`, `METRIC_LABELS` | `read_var_skill_metrics` | `render_var_skill_scorecard` |
| `default_indices(var)`, `var_signal_columns(...)` | `read_var_signal_metrics` | `render_var_signal_scorecard` |
| `signal_indices(var)` | *(internal)* | `plot_multivar_signal_scorecard` |
| `series_indices(var)` | `read_series` | `render_series` |
| `skill_map_indices(var)`, `METRIC_INFO` | `read_skill_map_metrics` | `render_multivar_maps`, `render_var_maps` |
| `signal_map_indices(var)`, `SIGNAL_INDEX_INFO` | `read_signal_map_metrics` | *(the same two renderers)* |
| `EVENT_VMINMAX` | `read_event_fields` | *(the same two renderers)* |
| `prob_columns(metrics)`, `PROB_SRC` | `read_prob_metrics` | `render_multivar_probskill_scorecard` |
| — | `domain_outline` | `plot_domain` |

Shared helpers live in `plot_hexagon`: `normalize_paths(paths)`,
`metrics_dir(path, var)` and `select_member(da, member_dim, member_num)`.

______________________________________________________________________

## Two things specific to the maps

**Colour scales** are resolved per variable by `resolve_styles`, in this order:

1. the caller's `vminmax` / `vminmax_bias` / `cmap` / `cmap_bias` — one value,
   or a `{var: value}` mapping when the figure spans several variables;
2. `get_example_color_scales(var, metric)`, whichever ends it pins;
3. a range derived from the plotted data and **shared by every panel of that
   variable** in the figure (symmetric about zero for the error panels).

Rule 3 is the safety net: with no catalogue entry, matplotlib would otherwise
stretch each map to its own min/max and two models could look identical while
differing by a factor. Every panel still draws its own colourbar, so a figure
mixing variables (columns of `plot_multivar_skill_maps`) or mixing a field with its
error (column 3 of `plot_var_skill_maps`) stays readable — the field and error
families are always scaled separately, and `relbias` is scaled as a percentage
rather than in the variable's units.

```python
plot_var_skill_maps(paths, out, ["pr"], metric="mean", kind="bias",
              vminmax=[0, 20], vminmax_bias=[-5, 5])          # pin both
plot_multivar_skill_maps(paths, out, ["pr", "tasmax"], metric="mean",
                   vminmax={"pr": [0, 40], "tasmax": [290, 305]})  # per variable
```

**The signal maps** differ from the skill maps in two ways that make them
separate entry points rather than a flag. The truth is the RCM's *own* signal,
which lives in its own run directory — `ml-eval signal` writes no `targ_`/`pred_`
pair inside a model's bundle, so `reference_path` is required. And the error
panel is *derived* (model delta − RCM delta) rather than read. They take
`quantity` (`delta` / `reldelta`, the latter stored as a fraction and shown as a
percentage) alongside `kind`, which is `raw` or `bias` only. Because a change
straddles zero, *every* panel is diverging and symmetric about it, not just the
error column.

**`kind`** says what a model panel shows: `"raw"` (the index itself), `"bias"`
(its error against the truth) or `"relbias"` (that error as a percentage of the
truth, derived as `100 × (raw − targ) / targ` and masked where the truth is ~0,
so it is available for every metric). It acts differently in the two layouts —
in `plot_multivar_skill_maps` it *replaces* the field in the model rows, in
`plot_var_skill_maps` it *adds* a third column:

```
plot_multivar_skill_maps(kind="raw")        plot_var_skill_maps(kind="bias")
┌──────────┬──────────┐               ┌───────┬──────────┬───────────────┐
│ Truth:pr │ Truth:tx │               │ Truth │ UNET     │ bias: UNET    │
│ UNET:pr  │ UNET:tx  │               │       │ RESDIFF  │ bias: RESDIFF │
│ RESD:pr  │ RESD:tx  │               └───────┴──────────┴───────────────┘
```

**The event pair reads NetCDF, not the stores.** `ml-eval` writes only
time-reduced diagnostics, so a single day is nowhere in a `metrics_*.zarr`.
`plot_multivar_event_maps` / `plot_var_event_maps` therefore take `paths` pointing at the
prediction and target **files** — the pre-evaluation eyeball check for wrong
units, spatial artefacts or over-smoothing, before any metric is trusted.

______________________________________________________________________

## Examples

```python
import os
from ml_evaluation.viz.plot_hexagon import plot_multivar_skill_scorecard
from ml_evaluation.viz.plot_var_skill_scorecard import plot_var_skill_scorecard
from ml_evaluation.viz.plot_diagnostic_maps import plot_var_skill_maps, plot_var_event_maps

SIM = "./SIMULATIONS"
paths = {
    "UNET-SINGLE": os.path.join(SIM, "UNET-SINGLE/outputs/evaluation/.../annual"),
    "RESDIFF":     os.path.join(SIM, "RESDIFF/outputs/evaluation/.../annual"),
}

# models x variables overview, against the baseline
plot_multivar_skill_scorecard(paths, "./FIGURES", ["tasmax", "pr"],
                              reference="UNET-SINGLE")

# one variable in detail, explicit columns
plot_var_skill_scorecard(paths, "./FIGURES", ["pr"],
                         metrics=["rmse", "bias_mean", "bias_r01", "ralsd"])

# the spatial field behind those scalars, with the bias column
plot_var_skill_maps(paths, "./FIGURES", ["pr"], metric="mean", kind="bias")

# a single day, straight from the NetCDFs
plot_var_event_maps({"RESDIFF": ".../predictions/RESDIFF600_..._2041-2060.nc"},
               "./FIGURES", ["pr"], "2050-08-15", kind="bias",
               reference="RCM", reference_path=".../predictions/RCM_..._2041-2060.nc")
```

For the full walk-through — running the `ml-eval` tracks first, then these
figures — see the tutorial in [`notebooks/`](../../notebooks/).
