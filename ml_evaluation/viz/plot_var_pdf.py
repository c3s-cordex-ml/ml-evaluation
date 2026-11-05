# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

"""
Per-variable *PDF histogram* figures for ML RCM emulators.

The distribution-view twin of ``plot_var_skill_scorecard``: the same
``{label: path}`` model spec and the same ``<path>/<var>/`` layout, but instead
of a models x metrics heatmap each figure overlays the stored histograms
(``metrics_slow_hist.zarr``, written by ``ml-eval slow``) of every model's
prediction on the ground-truth target. Drawing is delegated to
``plot_multivar_pdf.plot_pdf_multi_model_Mvar``; this module only translates the
path spec into that function's parallel-list arguments.

  * one line per model = that model's ``pred_hist_series``
  * one black line     = the ``targ_hist_series`` of the reference model (the
                         first entry of ``paths`` by default) -- every model is
                         scored against the same RCM truth, so one target line
                         is enough.

By default one PNG is written per variable; ``combined=True`` puts every
variable on its own panel of a single figure instead.

Locating the data
-----------------
As in ``plot_hexagon``, models come from an explicit ``{label: path}`` spec and
one variable's stores are read from ``<path>/<var>/``; ``season`` is a label for
the file name only.

Public API
----------
plot_var_pdf(paths, out, variables, *, season, member_num, combined, ...)
    One PNG per variable (or one multi-panel PNG), one line per entry of ``paths``.

Example
-------
    from ml_evaluation.viz.plot_var_pdf import plot_var_pdf
    plot_var_pdf(
        {
            "UNet-single": "/.../SINGV-RCM_EC-Earth3_2041-2060/annual",
            "ResDiff-v3":  "/.../RESDIFF-RCM_EC-Earth3_2041-2060/annual",
            "DownViT":     "/.../DOWNVIT-RCM_EC-Earth3_2041-2060/annual",
        },
        "/.../FIGURES/pdf",
        ["pr", "tasmax"],
        incl_diag="wdh",
    )
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_evaluation.viz.plot_hexagon import metrics_dir, normalize_paths
from ml_evaluation.viz.plot_multivar_pdf import HIST_STORE, plot_pdf_multi_model_Mvar

TARGET_LABEL = "Ground truth"
TARGET_COLOR = "black"


def _has_hist(base: str, var: str) -> bool:
    return os.path.isdir(os.path.join(metrics_dir(base, var), HIST_STORE))


def _default_colors(n: int) -> list:
    """``n`` model colours from the tab10 cycle (black is kept for the target)."""
    cycle = plt.get_cmap("tab10").colors
    return [cycle[i % len(cycle)] for i in range(n)]


def _render(
    by_label,
    models,
    variables,
    out,
    filename,
    *,
    reference,
    target_label,
    colors,
    member_num,
    season,
    incl_diag,
    ymin,
    log_scale,
):
    """One ``plot_pdf_multi_model_Mvar`` call: target line + one line per model.

    Each model's base directory is passed as-is; ``plot_pdf_multi_model_Mvar``
    reads ``<path>/<var>/`` for every variable itself.
    """
    entries = [(reference, "TARG", target_label, TARGET_COLOR)]
    entries += [(m, "PRED", m, colors[m]) for m in models]

    plot_pdf_multi_model_Mvar(
        input_dir_list=[by_label[m] for m, *_ in entries],
        input_target_pred=[tp for _, tp, _, _ in entries],
        input_member=[member_num] * len(entries),
        legend_list=[lbl for *_, lbl, _ in entries],
        color_list=[c for *_, c in entries],
        variable_list=list(variables),
        out_dir=out,
        out_fname=filename,
        season=season,
        incl_diag=incl_diag,
        ymin=ymin,
        log_scale=log_scale,
    )
    # plot_pdf_multi_model_Mvar appends the diagnostic tag and the extension.
    stem = filename + (f"_{incl_diag}" if incl_diag in {"wdh", "fcss", "pss", "lhd"} else "")
    path = os.path.join(out, stem + ".png")
    print(f"[var-pdf] {path}  (models: {', '.join(models)})")
    return path


def plot_var_pdf(
    paths,
    out,
    variables,
    *,
    season="annual",
    member_num=0,
    reference=None,
    target_label=TARGET_LABEL,
    colors=None,
    combined=False,
    filename=None,
    incl_diag=None,
    ymin=None,
    log_scale=True,
):
    """Plot model-vs-target PDF histograms for an explicit set of paths.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` (see ``plot_hexagon.normalize_paths``); ``path`` is the
        model's base directory and one variable's histogram store is read from
        ``<path>/<var>/metrics_slow_hist.zarr``. One entry = one line, in the
        given order.
    out : str
        Output directory for the PNGs.
    variables : iterable of str
        Variables to plot.
    season : str
        Label only -- output file name.
    member_num : int
        Realisation whose histogram is plotted (``>= 0``). Default 0 -- the only
        member of a deterministic model. The member mean is not supported: an
        averaged histogram is not the histogram of any field.
    reference : str, optional
        Label (key of ``paths``) whose ``targ_hist_series`` is drawn as the
        ground truth. Default: the first model that has the store.
    target_label : str
        Legend entry for the ground-truth line.
    colors : mapping or sequence, optional
        ``{label: colour}`` or a list in ``paths`` order. Default: tab10.
    combined : bool, default False
        False: one PNG per variable, skipping models without that variable.
        True: one PNG with a panel per variable; only models that have every
        variable are plotted.
    filename : str, optional
        Output file stem, without extension. Default
        ``var_pdf_<var>_<season>`` (per variable) or
        ``var_pdf_<n>var_<season>`` (combined). With ``combined=False`` and
        several variables, ``_<var>`` is appended so files do not collide.
    incl_diag : {"wdh", "fcss", "pss", "lhd"}, optional
        Distribution-distance score to append to each model's legend entry;
        also appended to the file name.
    ymin : float, optional
        Lower y-limit (useful with ``log_scale``).
    log_scale : bool, default True
        Log-scaled density axis.

    Returns
    -------
    list of str
        Paths of the PNGs written.
    """
    if member_num < 0:
        raise ValueError("member_num must be >= 0 for histograms.")
    variables = list(variables)
    if not variables:
        raise ValueError("variables must name at least one variable.")

    by_label = normalize_paths(paths)
    if reference is not None and reference not in by_label:
        raise KeyError(f"reference {reference!r} is not a label of paths.")

    if colors is None:
        colors = _default_colors(len(by_label))
    if not isinstance(colors, dict):
        colors = dict(zip(by_label, colors))
    missing = [m for m in by_label if m not in colors]
    if missing:
        raise ValueError(f"no colour given for: {', '.join(missing)}")

    common = dict(
        target_label=target_label,
        colors=colors,
        member_num=member_num,
        season=season,
        incl_diag=incl_diag,
        ymin=ymin,
        log_scale=log_scale,
    )

    if combined:
        groups = [variables]
    else:
        groups = [[v] for v in variables]

    written = []
    for group in groups:
        models = [m for m in by_label if all(_has_hist(by_label[m], v) for v in group)]
        tag = group[0] if len(group) == 1 else ", ".join(group)
        if not models:
            print(f"[var-pdf] {tag}: no model has {HIST_STORE} — skipping.")
            continue
        ref = reference or models[0]
        if not all(_has_hist(by_label[ref], v) for v in group):
            print(f"[var-pdf] {tag}: reference {ref!r} has no {HIST_STORE} — skipping.")
            continue

        if len(group) == 1:
            if filename is None:
                fname = f"var_pdf_{group[0]}_{season}"
            elif len(variables) > 1:
                fname = f"{filename}_{group[0]}"
            else:
                fname = filename
        else:
            fname = filename or f"var_pdf_{len(group)}var_{season}"

        written.append(
            _render(by_label, models, group, out, fname, reference=ref, **common)
        )
    return written
