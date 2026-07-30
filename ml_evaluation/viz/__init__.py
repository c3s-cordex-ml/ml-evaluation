"""Visualization helpers for ml-evaluation outputs.

Each figure entry point whose name equals its submodule is intentionally *not*
re-exported here -- re-exporting would shadow the submodule. Import those from
the submodule itself:

    from ml_evaluation.viz.plot_hexagon import plot_hexagon
    from ml_evaluation.viz.plot_var_skill_scorecard import plot_var_skill_scorecard
    from ml_evaluation.viz.plot_var_signal_scorecard import plot_var_signal_scorecard
    from ml_evaluation.viz.plot_multivar_signal_scorecard import (
        plot_multivar_signal_scorecard,
    )
    from ml_evaluation.viz.plot_var_series import plot_var_series
    from ml_evaluation.viz.plot_multivar_probskill_scorecard import (
        plot_multivar_probskill_scorecard,
    )

``plot_diagnostic_maps`` renders the spatial fields behind those scalars, on the
same *var* / *multivar* split and the same *skill* / *signal* split as the
scorecards: ``plot_var_skill_maps`` / ``plot_multivar_skill_maps`` (error against
the RCM truth) and ``plot_var_signal_maps`` / ``plot_multivar_signal_maps``
(climate-change change against the RCM's own signal), through the same
``{label: path}`` locating and ``read_*``/``render_*`` split.
``plot_var_event_maps`` / ``plot_multivar_event_maps`` are the single-date twins,
reading the prediction and target NetCDFs directly (a day never reaches the
metric stores).

``plot_multivar_probskill_scorecard`` adds the probabilistic view for ensemble
predictions: models x variables, each cell a triangle of CRPS / BS / SERAT read
from ``metrics_prob_float.zarr``.

The four skill/signal scorecards are a 2x2: *skill* (error metrics vs the RCM truth) or
*signal* (climate-change change), for one variable (a models x quantities
heatmap) or across variables (a models x variables grid of hexagons). The
scorecards collapse a run to scalars; ``plot_var_series`` is the time-resolved
view of the same signal runs -- one variable, one index, one line per source
on the stored ``year`` axis.
"""

from ml_evaluation.viz.plot_diagnostic_maps import (
    signal_map_indices,
    skill_map_indices,
    plot_multivar_event_maps,
    plot_multivar_signal_maps,
    plot_multivar_skill_maps,
    plot_var_event_maps,
    plot_var_signal_maps,
    plot_var_skill_maps,
    read_event_fields,
    read_signal_map_metrics,
    read_skill_map_metrics,
    render_multivar_maps,
    render_var_maps,
    resolve_styles,
)
from ml_evaluation.viz.plot_hexagon import (
    hexagon_axes,
    metrics_dir,
    normalize_paths,
    plot_hexagons,
    plot_multivar_skill_scorecard,
    read_hexagon_metrics,
)
from ml_evaluation.viz.plot_multivar_probskill_scorecard import (
    prob_columns,
    read_prob_metrics,
    render_multivar_probskill_scorecard,
)
from ml_evaluation.viz.plot_multivar_signal_scorecard import signal_indices
from ml_evaluation.viz.plot_var_series import (
    read_series,
    render_series,
    series_indices,
)
from ml_evaluation.viz.plot_var_signal_scorecard import (
    default_indices,
    read_var_signal_metrics,
    render_var_signal_scorecard,
    var_signal_columns,
)
from ml_evaluation.viz.plot_var_skill_scorecard import (
    read_var_skill_metrics,
    render_var_skill_scorecard,
    var_skill_columns,
)

__all__ = [
    "default_indices",
    "hexagon_axes",
    "metrics_dir",
    "normalize_paths",
    "plot_hexagons",
    "plot_multivar_event_maps",
    "plot_multivar_signal_maps",
    "plot_multivar_skill_maps",
    "plot_multivar_skill_scorecard",
    "plot_var_event_maps",
    "plot_var_signal_maps",
    "plot_var_skill_maps",
    "prob_columns",
    "read_event_fields",
    "read_hexagon_metrics",
    "read_prob_metrics",
    "read_series",
    "read_signal_map_metrics",
    "read_skill_map_metrics",
    "read_var_signal_metrics",
    "read_var_skill_metrics",
    "render_multivar_maps",
    "render_multivar_probskill_scorecard",
    "render_series",
    "render_var_maps",
    "render_var_signal_scorecard",
    "render_var_skill_scorecard",
    "resolve_styles",
    "series_indices",
    "signal_indices",
    "signal_map_indices",
    "skill_map_indices",
    "var_signal_columns",
    "var_skill_columns",
]
