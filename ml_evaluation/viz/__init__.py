"""Visualization helpers for ml-evaluation outputs.

The single-figure renderers ``plot_hexagon``, ``plot_signal_scorecard`` and
``plot_var_scorecard`` are intentionally *not* re-exported here: each name equals
its submodule (``ml_evaluation.viz.plot_hexagon`` / ``...plot_signal_scorecard``
/ ``...plot_var_scorecard``) and re-exporting would shadow the submodule. Import
them from the submodule:

    from ml_evaluation.viz.plot_hexagon import plot_hexagon
    from ml_evaluation.viz.plot_signal_scorecard import plot_signal_scorecard
    from ml_evaluation.viz.plot_var_scorecard import plot_var_scorecard
"""

from ml_evaluation.viz.plot_hexagon import (
    hexagon_axes,
    metrics_dir,
    normalize_paths,
    plot_hexagons,
    plot_scorecard,
    read_hexagon_metrics,
)
from ml_evaluation.viz.plot_signal_scorecard import signal_scorecard_indices
from ml_evaluation.viz.plot_var_scorecard import (
    plot_var_scorecards,
    read_var_scorecard_metrics,
    var_scorecard_columns,
)

__all__ = [
    "hexagon_axes",
    "metrics_dir",
    "normalize_paths",
    "plot_hexagons",
    "plot_scorecard",
    "plot_var_scorecards",
    "read_hexagon_metrics",
    "read_var_scorecard_metrics",
    "signal_scorecard_indices",
    "var_scorecard_columns",
]
