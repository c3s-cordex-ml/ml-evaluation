from __future__ import annotations

import pathlib

import numpy as np
import pytest

import ml_evaluation.viz.plot_diagnostic_maps as pdm

LAT = np.linspace(44.0, 48.0, 4)
LON = np.linspace(5.0, 15.0, 5)


def _fields(kinds, value: float = 1.0) -> dict:
    """A read_skill_map_metrics-shaped bundle, so the drawing path can run offline."""
    return {
        "fields": {k: np.full((LAT.size, LON.size), value) for k in kinds},
        "lat": LAT,
        "lon": LON,
        "units": "mm/day",
        "title": "Climatology mean",
        "vminmax": [0, 10],
        "cmap": "viridis",
        "vminmax_bias": [-5, 5],
        "cmap_bias": "BrBG",
    }


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture what the matrix renderer is asked to draw, drawing nothing."""
    calls: list[dict] = []

    def fake_matrix(lon_list, lat_list, data_list, **kwargs):
        calls.append({"n": len(data_list), "data": data_list, **kwargs})
        return kwargs["filepath"]

    monkeypatch.setattr(pdm, "plot_map_matrix_pcolormesh", fake_matrix)
    return calls


@pytest.fixture
def read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve every requested kind from memory instead of a zarr store."""
    monkeypatch.setattr(
        pdm,
        "read_skill_map_metrics",
        lambda d, var, metric, *, kinds, **k: _fields(tuple(kinds)),
    )


def _tree(tmp_path: pathlib.Path, models, variables) -> dict[str, str]:
    for model in models:
        for var in variables:
            (tmp_path / model / var).mkdir(parents=True)
    return {m: str(tmp_path / m) for m in models}


# ---------------------------------------------------------------- catalogue --


def test_map_indices_filters_variable_specific() -> None:
    """The catalogue drops metrics belonging to another variable."""
    everything = pdm.skill_map_indices()
    assert "mean" in everything and "p98wet" in everything and "frost" in everything

    pr_only = pdm.skill_map_indices("pr")
    assert "p98wet" in pr_only and "sdii" in pr_only
    assert "frost" not in pr_only and "su" not in pr_only

    tasmax = pdm.skill_map_indices("tasmax")
    assert "su" in tasmax and "hot" in tasmax and "p98wet" not in tasmax
    assert "mean" in pr_only and "mean" in tasmax  # variable-agnostic survives


def test_member_and_season_suffixes() -> None:
    """Member 0 is the default and stays out of the file name."""
    assert pdm._member_tag(0) == "" and pdm._member_tag(None) == ""
    assert pdm._member_tag(3) == "_m3" and pdm._member_tag(-1) == "_em"
    assert pdm._season_suffix(None) == "" and pdm._season_suffix("JJA") == "_JJA"


# ------------------------------------------------------------------- reader --


def test_read_map_metrics_missing_store(tmp_path: pathlib.Path) -> None:
    """A missing store is a None, not an exception -- callers skip the source."""
    assert pdm.read_skill_map_metrics(str(tmp_path), "pr", "mean") is None


def test_read_map_metrics_rejects_wrong_variable(tmp_path: pathlib.Path) -> None:
    """A precipitation-only metric asked of tasmax is an error, not a NaN map."""
    with pytest.raises(ValueError, match="only suitable for variable pr"):
        pdm.read_skill_map_metrics(str(tmp_path), "tasmax", "p98wet")


def test_read_map_metrics_rejects_unknown_kind(tmp_path: pathlib.Path) -> None:
    """Only 'targ' and the panel kinds can be read."""
    with pytest.raises(ValueError, match="unknown kinds"):
        pdm.read_skill_map_metrics(str(tmp_path), "pr", "mean", kinds=("pred",))


def test_relbias_is_derived_and_masked() -> None:
    """relbias = 100 (raw - targ) / targ, NaN where the truth is ~0."""
    targ = np.array([[10.0, 0.0]])
    raw = np.array([[12.0, 5.0]])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * (raw - targ) / targ
    rel = np.where(np.abs(targ) < 1e-12, np.nan, rel)
    assert rel[0, 0] == pytest.approx(20.0)
    assert np.isnan(rel[0, 1])


def test_resolve_styles_uses_the_catalogue_first() -> None:
    """A catalogue range wins over the data, and each family keeps its own."""
    info = _fields(("targ", "raw", "bias"))
    styles = pdm.resolve_styles([info], "pr", "bias")
    assert styles["raw"][1] == [0, 10] and styles["raw"][2] == "mm/day"
    assert styles["bias"][1] == [-5, 5] and styles["bias"][2] == "mm/day"
    assert styles["targ"] == styles["raw"]  # truth shares the field scale


def test_resolve_styles_override_wins() -> None:
    """An explicit range beats the catalogue, per family and per variable."""
    info = _fields(("targ", "raw", "bias"))
    styles = pdm.resolve_styles(
        [info], "pr", "bias", vminmax=[0, 1], vminmax_bias=[-2, 2], cmap="magma"
    )
    assert styles["raw"][1] == [0, 1] and styles["bias"][1] == [-2, 2]
    assert styles["raw"][0].name == "magma"

    # a {var: value} mapping picks the entry for this variable only
    per_var = pdm.resolve_styles([info], "pr", "raw", vminmax={"tasmax": [0, 1]})
    assert per_var["raw"][1] == [0, 10]  # falls through to the catalogue


def test_resolve_styles_falls_back_to_a_shared_data_range() -> None:
    """With no catalogue entry, one range covers every panel of the variable."""
    a = _fields(("targ", "raw"), value=1.0)
    b = _fields(("targ", "raw"), value=5.0)
    for info in (a, b):  # pretend the catalogue knows nothing
        info["vminmax"] = [None, None]
        info["vminmax_bias"] = [None, None]

    styles = pdm.resolve_styles([a, b], "pr", "raw")
    assert styles["raw"][1] == [1.0, 5.0]  # shared, not per-panel autoscale


def test_resolve_styles_error_range_is_symmetric() -> None:
    """A derived error range is centred on zero, so the diverging map is honest."""
    a = _fields(("targ", "raw", "bias"), value=1.0)
    a["fields"]["bias"] = np.array([[-1.0, 4.0]])
    a["vminmax_bias"] = [None, None]

    styles = pdm.resolve_styles([a], "pr", "bias")
    assert styles["bias"][1] == [-4.0, 4.0]


def test_resolve_styles_relbias_ignores_the_unitful_range() -> None:
    """relbias is a percentage: the stored bias range must not leak into it."""
    a = _fields(("targ", "raw", "relbias"))
    a["fields"]["relbias"] = np.array([[-20.0, 10.0]])

    styles = pdm.resolve_styles([a], "pr", "relbias")
    assert styles["relbias"][1] == [-20.0, 20.0]  # symmetric, from the data
    assert styles["relbias"][2] == "%"
    # an explicit override still applies
    styles = pdm.resolve_styles([a], "pr", "relbias", vminmax_bias=[-50, 50])
    assert styles["relbias"][1] == [-50, 50]


def test_shared_range_ignores_nan_and_empty() -> None:
    """NaNs are skipped; nothing plottable gives an auto range."""
    assert pdm._shared_range([np.array([np.nan, 2.0]), None]) == [2.0, 2.0]
    assert pdm._shared_range([]) == [None, None]
    assert pdm._shared_range([np.zeros((2, 2))], symmetric=True) == [None, None]


# --------------------------------------------------------------- multivar ----


def test_multivar_layout_is_models_by_variables(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """(1 + N models) rows x len(variables) columns, truth on top."""
    paths = _tree(tmp_path, ["UNET-MULTI", "RESDIFF"], ["pr", "tasmax"])

    out = pdm.plot_multivar_skill_maps(
        paths, str(tmp_path / "fig"), ["pr", "tasmax"], metric="mean"
    )

    call = captured[0]
    assert call["ncols"] == 2 and call["nrows"] == 3  # truth + 2 models
    assert call["title_list"] == [
        "Truth: pr",
        "Truth: tasmax",
        "UNET-MULTI: pr",
        "UNET-MULTI: tasmax",
        "RESDIFF: pr",
        "RESDIFF: tasmax",
    ]
    assert out.endswith("multivar_maps_mean_raw_annual.png")


def test_multivar_bias_replaces_the_model_rows(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """kind='bias': the model rows show the error, not the field."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr"])

    out = pdm.plot_multivar_skill_maps(
        paths, str(tmp_path / "fig"), ["pr"], metric="mean", kind="bias"
    )

    call = captured[0]
    assert call["title_list"] == ["Truth: pr", "bias: RESDIFF: pr"]
    assert call["vminmax"] == [[0, 10], [-5, 5]]  # truth scale, then bias scale
    assert out.endswith("multivar_maps_mean_bias_annual.png")


def test_multivar_without_truth(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """show_truth=False drops the top row entirely."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr", "tasmax"])

    pdm.plot_multivar_skill_maps(
        paths, str(tmp_path / "fig"), ["pr", "tasmax"], show_truth=False
    )

    assert captured[0]["title_list"] == ["RESDIFF: pr", "RESDIFF: tasmax"]
    assert captured[0]["nrows"] == 1


# -------------------------------------------------------------------- var ----


def test_var_layout_truth_then_models(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """Truth at (1,1) with column 1 blank below it; models fill column 2."""
    paths = _tree(tmp_path, ["UNET-MULTI", "RESDIFF"], ["pr"])

    written = pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr"], metric="mean")

    call = captured[0]
    assert call["ncols"] == 2 and call["nrows"] == 2
    assert call["title_list"] == ["Truth", "UNET-MULTI", None, "RESDIFF"]
    assert call["data"][2] is None  # the blank under the truth
    assert len(written) == 1 and written[0].endswith("var_maps_pr_mean_raw_annual.png")


def test_var_bias_adds_a_third_column(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """kind='bias' keeps the field in column 2 and adds the error in column 3."""
    paths = _tree(tmp_path, ["UNET-MULTI", "RESDIFF"], ["pr"])

    pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr"], metric="mean", kind="bias")

    call = captured[0]
    assert call["ncols"] == 3 and call["nrows"] == 2
    assert call["title_list"] == [
        "Truth",
        "UNET-MULTI",
        "bias: UNET-MULTI",
        None,
        "RESDIFF",
        "bias: RESDIFF",
    ]


def test_var_relbias_column(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """kind='relbias' labels its column and carries the % colour bar."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr"])

    written = pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr"], kind="relbias")

    call = captured[0]
    assert call["title_list"] == ["Truth", "RESDIFF", "relbias: RESDIFF"]
    assert call["cbartitle_list"][2] == "%"
    assert written[0].endswith("var_maps_pr_mean_relbias_annual.png")


def test_var_one_figure_per_variable(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """Several variables give several PNGs, one per variable."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr", "tasmax"])

    written = pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr", "tasmax"])

    assert len(written) == 2 and len(captured) == 2
    assert "var_maps_pr_" in written[0] and "var_maps_tasmax_" in written[1]


def test_var_without_truth_has_no_truth_column(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """show_truth=False leaves the models alone in column 1."""
    paths = _tree(tmp_path, ["UNET-MULTI", "RESDIFF"], ["pr"])

    pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr"], show_truth=False)

    call = captured[0]
    assert call["ncols"] == 1
    assert call["title_list"] == ["UNET-MULTI", "RESDIFF"]


# ------------------------------------------------------------ shared rules ----


def test_unknown_kind_is_rejected(tmp_path: pathlib.Path) -> None:
    """The panel kind must be one of raw / bias / relbias."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr"])
    with pytest.raises(ValueError, match="kind must be one of"):
        pdm.render_var_maps(
            {"RESDIFF": _fields(("raw",))},
            None,
            "pr",
            str(tmp_path / "fig"),
            kind="anomaly",
        )
    with pytest.raises(ValueError, match="unknown kinds"):
        pdm.plot_multivar_skill_maps(paths, str(tmp_path / "fig"), ["pr"], kind="anomaly")


def test_unknown_reference(tmp_path: pathlib.Path, read: None) -> None:
    """Naming a reference that is not a label (and has no path) is an error."""
    paths = _tree(tmp_path, ["UNET-MULTI"], ["pr"])
    with pytest.raises(KeyError, match="not one of the labels"):
        pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr"], reference="RCM")


def test_sources_without_metrics_are_skipped(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """A model with no store on disk drops out; the figure is still drawn."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr"])
    paths["UNET-MULTI"] = str(tmp_path / "UNET-MULTI")  # never created

    pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr"])

    assert captured[0]["title_list"] == ["Truth", "RESDIFF"]


def test_member_tag_in_filename(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """A non-default member is recorded in the file name."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr"])

    written = pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr"], member_num=-1)

    assert written[0].endswith("var_maps_pr_mean_raw_em_annual.png")


# ------------------------------------------------------------------ events ---


def _nc(path: pathlib.Path, variables, values, members: int = 1) -> str:
    """A prediction/target-shaped NetCDF: var(member, time, lat, lon)."""
    import xarray as xr

    times = np.array(["1983-08-14T12", "1983-08-15T12"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            v: (
                ("member", "time", "lat", "lon"),
                np.full((members, times.size, LAT.size, LON.size), values[v]),
            )
            for v in variables
        },
        coords={"member": np.arange(members), "time": times, "lat": LAT, "lon": LON},
    )
    for v in variables:
        ds[v].attrs["units"] = "mm day-1" if v == "pr" else "K"
    ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def event_paths(tmp_path: pathlib.Path) -> dict[str, str]:
    """Truth and two model files, each holding pr and tasmax."""
    return {
        "RCM": _nc(tmp_path / "truth.nc", ["pr", "tasmax"], {"pr": 10.0, "tasmax": 300.0}),
        "UNET-MULTI": _nc(
            tmp_path / "unet.nc", ["pr", "tasmax"], {"pr": 12.0, "tasmax": 301.0}
        ),
        "RESDIFF": _nc(
            tmp_path / "resdiff.nc", ["pr", "tasmax"], {"pr": 11.0, "tasmax": 300.5}
        ),
    }


def test_read_event_fields_selects_one_day(event_paths: dict[str, str]) -> None:
    """A date that resolves to one timestep gives a 2-D field in display units."""
    info = pdm.read_event_fields(event_paths["RCM"], "pr", "1983-08-15")
    assert info["fields"]["raw"].shape == (LAT.size, LON.size)
    assert info["units"] == "mm/day" and info["title"] == "1983-08-15"
    assert info["vminmax"] == pdm.EVENT_VMINMAX["pr"]


def test_read_event_fields_missing_file_or_variable(
    tmp_path: pathlib.Path, event_paths: dict[str, str]
) -> None:
    """A missing file or variable is a None, so the source is simply skipped."""
    assert pdm.read_event_fields(str(tmp_path / "nope.nc"), "pr", "1983-08-15") is None
    assert pdm.read_event_fields(event_paths["RCM"], "hurs", "1983-08-15") is None


def test_read_event_fields_ambiguous_date(event_paths: dict[str, str]) -> None:
    """A date matching several timesteps is an error, not a silent mean."""
    with pytest.raises(ValueError, match="selects 2 timesteps"):
        pdm.read_event_fields(event_paths["RCM"], "pr", "1983-08")


def test_multivar_event_layout(
    tmp_path: pathlib.Path, event_paths: dict[str, str], captured: list[dict]
) -> None:
    """Same grid as plot_multivar_skill_maps: truth row, then one row per model."""
    models = {k: v for k, v in event_paths.items() if k != "RCM"}

    out = pdm.plot_multivar_event_maps(
        models,
        str(tmp_path / "fig"),
        ["pr", "tasmax"],
        "1983-08-15",
        reference="RCM",
        reference_path=event_paths["RCM"],
    )

    call = captured[0]
    assert call["ncols"] == 2 and call["nrows"] == 3
    assert call["title_list"] == [
        "Truth: pr",
        "Truth: tasmax",
        "UNET-MULTI: pr",
        "UNET-MULTI: tasmax",
        "RESDIFF: pr",
        "RESDIFF: tasmax",
    ]
    assert out.endswith("multivar_event_1983-08-15_raw.png")


def test_var_event_bias_is_model_minus_truth(
    tmp_path: pathlib.Path, event_paths: dict[str, str], captured: list[dict]
) -> None:
    """kind='bias' adds column 3, holding (model - truth) for that day."""
    models = {k: v for k, v in event_paths.items() if k != "RCM"}

    written = pdm.plot_var_event_maps(
        models,
        str(tmp_path / "fig"),
        ["pr"],
        "1983-08-15",
        kind="bias",
        reference="RCM",
        reference_path=event_paths["RCM"],
    )

    call = captured[0]
    assert call["ncols"] == 3
    assert call["title_list"] == [
        "Truth",
        "UNET-MULTI",
        "bias: UNET-MULTI",
        None,
        "RESDIFF",
        "bias: RESDIFF",
    ]
    assert np.allclose(call["data"][2], 2.0)  # 12 - 10 mm/day
    assert np.allclose(call["data"][5], 1.0)  # 11 - 10
    assert written[0].endswith("var_event_pr_1983-08-15_bias.png")


def test_var_event_relbias_is_a_percentage(
    tmp_path: pathlib.Path, event_paths: dict[str, str], captured: list[dict]
) -> None:
    """kind='relbias' divides by the truth and labels the bar in %."""
    pdm.plot_var_event_maps(
        {"UNET-MULTI": event_paths["UNET-MULTI"]},
        str(tmp_path / "fig"),
        ["pr"],
        "1983-08-15",
        kind="relbias",
        reference="RCM",
        reference_path=event_paths["RCM"],
    )

    call = captured[0]
    assert np.allclose(call["data"][2], 20.0)  # (12 - 10) / 10
    assert call["cbartitle_list"][2] == "%"


def test_multivar_scale_is_shared_down_a_column(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, captured: list[dict]
) -> None:
    """Every panel of one variable gets one range; columns differ from each other."""
    values = {("A", "pr"): 1.0, ("B", "pr"): 5.0, ("A", "tasmax"): 280.0,
              ("B", "tasmax"): 300.0}

    def fake_read(d, var, metric, *, kinds, **k):
        model = pathlib.Path(d).parent.name if pathlib.Path(d).name == var else "A"
        info = _fields(tuple(kinds), value=values.get((model, var), 1.0))
        info["vminmax"] = [None, None]  # catalogue knows nothing
        return info

    monkeypatch.setattr(pdm, "read_skill_map_metrics", fake_read)
    paths = _tree(tmp_path, ["A", "B"], ["pr", "tasmax"])

    pdm.plot_multivar_skill_maps(paths, str(tmp_path / "fig"), ["pr", "tasmax"],
                           show_truth=False)

    ranges = captured[0]["vminmax"]
    # column 0 (pr) and column 1 (tasmax) each share a range across the rows
    assert ranges[0] == ranges[2] and ranges[1] == ranges[3]
    assert ranges[0] != ranges[1]  # but the two columns are scaled separately


def test_var_maps_override_reaches_the_renderer(
    tmp_path: pathlib.Path, read: None, captured: list[dict]
) -> None:
    """vminmax / vminmax_bias passed to plot_* land on the right panels."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr"])

    pdm.plot_var_skill_maps(paths, str(tmp_path / "fig"), ["pr"], kind="bias",
                      vminmax=[0, 3], vminmax_bias=[-1, 1])

    ranges = captured[0]["vminmax"]
    assert ranges[0] == [0, 3]   # truth
    assert ranges[1] == [0, 3]   # model field, same scale as the truth
    assert ranges[2] == [-1, 1]  # the error column


# --------------------------------------------------------------- grid dims ---


def test_horizontal_dims_autodetects_both_grids() -> None:
    """A regular (lat, lon) grid and a curvilinear (y, x) one both work."""
    import xarray as xr

    regular = xr.DataArray(np.zeros((2, 3)), dims=("lat", "lon"))
    curvilinear = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"))
    assert pdm.horizontal_dims(regular) == ("lat", "lon")
    assert pdm.horizontal_dims(curvilinear) == ("y", "x")
    # an explicit pair still overrides
    assert pdm.horizontal_dims(curvilinear, ("Y", "X")) == ("Y", "X")


def test_horizontal_dims_rejects_an_unreduced_field() -> None:
    """A leftover member/time dimension is named in the error, not indexed into."""
    import xarray as xr

    da = xr.DataArray(np.zeros((2, 2, 3)), dims=("member", "y", "x"))
    with pytest.raises(ValueError, match="member"):
        pdm.horizontal_dims(da)
    da = xr.DataArray(np.zeros((2, 3)), dims=("time", "x"))
    with pytest.raises(ValueError, match="time"):
        pdm.horizontal_dims(da)


def test_check_latlon_field_reports_sizes_for_a_dataarray() -> None:
    """The size lookup must work on a DataArray, whose .dims is a tuple."""
    import xarray as xr

    da = xr.DataArray(np.zeros((3, 2, 2)), dims=("member", "y", "x"))
    with pytest.raises(ValueError, match=r"sizes: \{'member': 3\}"):
        pdm.check_latlon_field(da, ["y", "x"])


def test_plot_coords_prefers_2d_latlon_coordinates() -> None:
    """On a curvilinear grid the 2-D lat/lon coords are what pcolormesh needs."""
    import xarray as xr

    lat2d = np.tile(LAT[:, None], (1, LON.size))
    lon2d = np.tile(LON[None, :], (LAT.size, 1))
    ds = xr.Dataset(
        {"value": (("y", "x"), np.zeros((LAT.size, LON.size)))},
        coords={
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
            "y": np.arange(LAT.size),
            "x": np.arange(LON.size),
        },
    )
    lat, lon = pdm.plot_coords(ds, ds["value"], ("y", "x"))
    assert lat.shape == (LAT.size, LON.size) and lon.shape == (LAT.size, LON.size)

    # without lat/lon coords, the dimension coordinates are used
    plain = xr.Dataset(
        {"value": (("lat", "lon"), np.zeros((LAT.size, LON.size)))},
        coords={"lat": LAT, "lon": LON},
    )
    lat, lon = pdm.plot_coords(plain, plain["value"], ("lat", "lon"))
    assert lat.ndim == 1 and lon.ndim == 1


def test_auto_range_skips_the_catalogue() -> None:
    """"auto" forces the data-derived range even when the catalogue pins one."""
    a = _fields(("targ", "raw"), value=280.0)
    b = _fields(("targ", "raw"), value=300.0)  # catalogue says [0, 10]

    pinned = pdm.resolve_styles([a, b], "tasmax", "raw")
    assert pinned["raw"][1] == [0, 10]  # catalogue wins by default

    auto = pdm.resolve_styles([a, b], "tasmax", "raw", vminmax="auto")
    assert auto["raw"][1] == [280.0, 300.0]  # the data does

    # and per family: auto on the errors only
    c = _fields(("targ", "raw", "bias"))
    c["fields"]["bias"] = np.array([[-2.0, 6.0]])
    styles = pdm.resolve_styles([c], "pr", "bias", vminmax_bias="auto")
    assert styles["bias"][1] == [-6.0, 6.0]  # symmetric, from the data
    assert styles["raw"][1] == [0, 10]  # the field still uses the catalogue


def test_bias_colormap_defaults_to_a_diverging_one() -> None:
    """A bias straddles zero, so its default colormap must diverge."""
    _, _, _, cmap_bias = pdm.get_example_color_scales("pr", "mean")
    assert cmap_bias == "BrBG"
    _, _, _, cmap_bias = pdm.get_example_color_scales("tasmax", "mean")
    assert cmap_bias == "RdBu_r"
    # a deliberate per-metric choice is not overridden
    _, _, _, cmap_bias = pdm.get_example_color_scales("pr", "acorr")
    assert cmap_bias == "PiYG_r"


# ------------------------------------------------------------ signal maps ----


def _signal_bundle(kinds, value: float = 2.0) -> dict:
    """A read_signal_map_metrics-shaped bundle (diverging, no catalogue range)."""
    info = _fields(kinds, value=value)
    info.update(
        units="K", title="Mean delta", vminmax=[None, None],
        cmap="RdBu_r", vminmax_bias=[None, None], cmap_bias="RdBu_r",
    )
    return info


def test_signal_map_indices_match_the_signal_store() -> None:
    """The signal store spells indices differently from the fast track."""
    assert pdm.signal_map_indices("pr") == ["r01", "sdii", "rx1day"]
    assert pdm.signal_map_indices("tasmax") == ["mean", "p01", "p99"]
    # rx1day here vs max1day in the fast catalogue -- the trap this guards
    assert "rx1day" in pdm.signal_map_indices() and "rx1day" not in pdm.skill_map_indices()
    assert "max1day" in pdm.skill_map_indices()


def test_read_signal_map_metrics_validates_its_arguments(
    tmp_path: pathlib.Path,
) -> None:
    """Unknown quantity or a precip-only index on tasmax is an error."""
    with pytest.raises(ValueError, match="quantity must be one of"):
        pdm.read_signal_map_metrics(str(tmp_path), "pr", "sdii", quantity="anomaly")
    with pytest.raises(ValueError, match="only suitable for variable pr"):
        pdm.read_signal_map_metrics(str(tmp_path), "tasmax", "sdii")
    with pytest.raises(ValueError, match="unknown index"):
        pdm.read_signal_map_metrics(str(tmp_path), "pr", "max1day")
    # a missing store is a skip, not a raise
    assert pdm.read_signal_map_metrics(str(tmp_path), "pr", "sdii") is None


def test_signal_maps_layout_and_derived_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, captured: list[dict]
) -> None:
    """kind='bias' derives model-minus-RCM, since ml-eval stores no such field."""
    values = {"RCM": 3.0, "UNET-SINGLE": 1.0, "RESDIFF": 2.5}

    def fake_read(d, var, index, *, quantity="delta", key="raw", **k):
        label = pathlib.Path(d).parent.name
        return _signal_bundle((key,), value=values[label])

    monkeypatch.setattr(pdm, "read_signal_map_metrics", fake_read)
    paths = _tree(tmp_path, ["UNET-SINGLE", "RESDIFF"], ["tasmax"])
    (tmp_path / "RCM" / "tasmax").mkdir(parents=True)

    written = pdm.plot_var_signal_maps(
        paths, str(tmp_path / "fig"), ["tasmax"],
        index="mean", quantity="delta", kind="bias",
        reference="RCM", reference_path=str(tmp_path / "RCM"),
    )

    call = captured[0]
    assert call["ncols"] == 3
    assert call["title_list"] == [
        "Truth", "UNET-SINGLE", "bias: UNET-SINGLE",
        None, "RESDIFF", "bias: RESDIFF",
    ]
    assert np.allclose(call["data"][2], -2.0)  # 1.0 - 3.0
    assert np.allclose(call["data"][5], -0.5)  # 2.5 - 3.0
    assert written[0].endswith("var_signal_maps_tasmax_mean_delta_bias_annual.png")


def test_signal_map_ranges_are_symmetric(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, captured: list[dict]
) -> None:
    """A change straddles zero, so even the field panels centre on it."""
    def fake_read(d, var, index, *, quantity="delta", key="raw", **k):
        return _signal_bundle((key,), value=4.0)

    monkeypatch.setattr(pdm, "read_signal_map_metrics", fake_read)
    paths = _tree(tmp_path, ["RESDIFF"], ["pr"])
    (tmp_path / "RCM" / "pr").mkdir(parents=True)

    pdm.plot_var_signal_maps(
        paths, str(tmp_path / "fig"), ["pr"], index="rx1day",
        reference="RCM", reference_path=str(tmp_path / "RCM"),
    )

    assert captured[0]["vminmax"][0] == [-4.0, 4.0]  # not [4.0, 4.0]


def test_multivar_signal_maps_one_figure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, captured: list[dict]
) -> None:
    """Variables across, sources down, one PNG, per-variable index selection."""
    seen = []

    def fake_read(d, var, index, *, quantity="delta", key="raw", **k):
        seen.append((var, index))
        return _signal_bundle((key,))

    monkeypatch.setattr(pdm, "read_signal_map_metrics", fake_read)
    paths = _tree(tmp_path, ["RESDIFF"], ["pr", "tasmax"])
    (tmp_path / "RCM" / "pr").mkdir(parents=True)
    (tmp_path / "RCM" / "tasmax").mkdir(parents=True)

    out = pdm.plot_multivar_signal_maps(
        paths, str(tmp_path / "fig"), ["pr", "tasmax"],
        index={"pr": "rx1day", "tasmax": "mean"},
        reference="RCM", reference_path=str(tmp_path / "RCM"),
    )

    assert ("pr", "rx1day") in seen and ("tasmax", "mean") in seen
    assert captured[0]["title_list"] == [
        "Truth: pr", "Truth: tasmax", "RESDIFF: pr", "RESDIFF: tasmax",
    ]
    assert out.endswith("multivar_signal_maps_rx1day-mean_delta_raw_annual.png")


def test_signal_maps_reject_relbias_kind(tmp_path: pathlib.Path) -> None:
    """`quantity` already covers the relative view, so kind stays raw|bias."""
    paths = _tree(tmp_path, ["RESDIFF"], ["pr"])
    with pytest.raises(ValueError, match="kind must be 'raw' or 'bias'"):
        pdm.plot_var_signal_maps(paths, str(tmp_path / "fig"), ["pr"], kind="relbias")
