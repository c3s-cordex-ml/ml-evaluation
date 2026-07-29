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
    from ml_evaluation.viz.plot_series import plot_series

The four scorecards are a 2x2: *skill* (error metrics vs the RCM truth) or
*signal* (climate-change change), for one variable (a models x quantities
heatmap) or across variables (a models x variables grid of hexagons). The
scorecards collapse a run to scalars; ``plot_series`` is the time-resolved
view of the same signal runs -- one variable, one index, one line per source
on the stored ``year`` axis.
"""

from ml_evaluation.viz.plot_hexagon import (
    hexagon_axes,
    metrics_dir,
    normalize_paths,
    plot_hexagons,
    plot_multivar_skill_scorecard,
    read_hexagon_metrics,
)
from ml_evaluation.viz.plot_multivar_signal_scorecard import signal_indices
from ml_evaluation.viz.plot_series import (
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
    "plot_multivar_skill_scorecard",
    "read_hexagon_metrics",
    "read_series",
    "read_var_signal_metrics",
    "read_var_skill_metrics",
    "render_series",
    "render_var_signal_scorecard",
    "render_var_skill_scorecard",
    "series_indices",
    "signal_indices",
    "var_signal_columns",
    "var_skill_columns",
]
