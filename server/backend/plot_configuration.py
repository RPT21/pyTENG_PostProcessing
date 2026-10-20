import json
import logging
import math
import pickle
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import render_template, request, redirect, url_for, session, flash, jsonify
from scipy.optimize import curve_fit

from server.backend.experiments_preview import (
    ExperimentFileStatus,
    LoadsDescriptionRepository,
    PostprocessingMetadataStore,
    resolve_experiment_identity,
)
from server.utils import output_storage
from server.utils.experiment_path_resolver import InvalidExperimentFormatError

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
KIND_STANDARD = "standard"
KIND_OC = "oc"
KIND_SC = "sc"

MODE_STANDARD = "standard"
MODE_OC = "oc"
MODE_SC = "sc"

POWER_METRIC_AVERAGE = "average"
POWER_METRIC_PEAK = "peak"

MAX_CUSTOM_PLOT_POINTS = 5000
MIN_POINTS_FOR_FIT = 3

COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_OC_PATTERN = re.compile(r"^(oc|open[\s_-]*circuit)$", re.IGNORECASE)
_SC_PATTERN = re.compile(r"^(sc|short[\s_-]*circuit)$", re.IGNORECASE)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Numeric-column lists keyed by (file path, mtime): reading a CleanData
# pickle just to list its columns is expensive, and the file rarely changes.
_COLUMN_CACHE = {}


class PlotConfigurationError(Exception):
    """A user-facing error raised while validating a request or building a figure."""


# ----------------------------------------------------------------------
# OC / SC classification and selection validation
# ----------------------------------------------------------------------
def classify_rload(rload_id):
    """
    Classifies an RloadId as Open Circuit ('oc', R = inf), Short Circuit
    ('sc', R = 0) or a regular load ('standard').
    """
    text = str(rload_id).strip() if rload_id is not None else ""
    if _OC_PATTERN.match(text):
        return KIND_OC
    if _SC_PATTERN.match(text):
        return KIND_SC
    return KIND_STANDARD


def validate_selection(entries):
    """
    Validates a selection of experiments for the figures of merit.

    Power cannot be calculated for OC (R = inf) or SC (R = 0), so the only
    valid, mutually exclusive selections are: only standard loads, only OC
    (any number of TribuIds) or only SC (any number of TribuIds).

    Returns:
        dict: {"ok", "mode" ('standard'|'oc'|'sc'|None), "error", "tribu_ids",
        "rload_ids", "single_tribu"}.
    """
    result = {"ok": False, "mode": None, "error": None, "tribu_ids": [], "rload_ids": [], "single_tribu": False}
    if not entries:
        result["error"] = "Select at least one experiment."
        return result

    result["tribu_ids"] = sorted({str(e["tribu_id"]) for e in entries})
    result["rload_ids"] = sorted({str(e["rload_id"]) for e in entries})
    result["single_tribu"] = len(result["tribu_ids"]) == 1

    kinds = {e["kind"] for e in entries}
    has_oc, has_sc, has_standard = KIND_OC in kinds, KIND_SC in kinds, KIND_STANDARD in kinds

    if has_oc and has_sc:
        result["error"] = (
            "Open Circuit (OC) and Short Circuit (SC) experiments cannot be combined. "
            "Select either only OC or only SC experiments."
        )
    elif (has_oc or has_sc) and has_standard:
        special = "OC" if has_oc else "SC"
        result["error"] = (
            f"Power cannot be calculated for {special} (R = {'∞' if has_oc else '0'}), so {special} cannot be "
            f"combined with VR RLoadIds. Select either only VR RLoadIds or only {special} experiments."
        )
    else:
        result["ok"] = True
        result["mode"] = MODE_OC if has_oc else MODE_SC if has_sc else MODE_STANDARD
    return result


# ----------------------------------------------------------------------
# Experiment catalogue
# ----------------------------------------------------------------------
def _experiment_label(entry):
    return (
        f"{entry['tribu_id']} | {entry['pos']}-{entry['neg']} | "
        f"R={entry['rload_id']} | {entry['date']}"
    )


def build_catalog(root_dir, experiments):
    """
    Builds one catalogue entry per experiment of the session: identity,
    effective RloadId (the one saved in the postprocessing metadata wins
    over the scanned one), Req, OC/SC kind and on-disk data availability.
    """
    inspector = ExperimentFileStatus(root_dir)
    req_map = LoadsDescriptionRepository(root_dir).load_req()
    entries = []
    for idx, experiment in enumerate(experiments):
        status = inspector.inspect(experiment)
        try:
            identity = resolve_experiment_identity(experiment)
        except InvalidExperimentFormatError:
            identity = None

        rload_id = experiment.get("RloadId")
        if identity is not None:
            meta = PostprocessingMetadataStore(root_dir, identity).load() or {}
            rload_id = meta.get("RloadId", rload_id)
        rload_key = str(rload_id).strip() if rload_id not in (None, "") else ""

        entry = {
            "id": idx,
            "tribu_id": str(experiment.get("TribuId", "")),
            "pos": str(experiment.get("SampleIdTriboPos", "")),
            "neg": str(experiment.get("SampleIdTriboNeg", "")),
            "date": str(experiment.get("Date", "")),
            "rload_id": rload_key,
            "req": req_map.get(rload_key),
            "kind": classify_rload(rload_key),
            "clean_data_exists": bool(status["clean_data_exists"]),
            "cycle_data_exists": bool(status["cycle_data_exists"]),
            "identity": identity,
        }
        entry["label"] = _experiment_label(entry)
        entries.append(entry)
    return entries


def _public_entry(entry):
    return {k: v for k, v in entry.items() if k != "identity"}


def _context():
    root_dir = session.get("root_dir")
    experiments = session.get("experiments")
    if not root_dir or not experiments:
        raise PlotConfigurationError("No experiments loaded yet. Please select a root folder first.")
    return root_dir, experiments


def _select_entries(root_dir, experiments, ids, require_cycle_data):
    if not isinstance(ids, list) or not ids:
        raise PlotConfigurationError("Select at least one experiment.")
    catalog = build_catalog(root_dir, experiments)
    selected, missing = [], []
    for raw_id in ids:
        try:
            entry = catalog[int(raw_id)]
        except (TypeError, ValueError, IndexError):
            raise PlotConfigurationError(f"Unknown experiment id: {raw_id!r}.")
        if entry["identity"] is None or not entry["clean_data_exists"]:
            missing.append(f"{entry['label']} (no CleanData)")
        elif require_cycle_data and not entry["cycle_data_exists"]:
            missing.append(f"{entry['label']} (no CycleData)")
        selected.append(entry)
    if missing:
        raise PlotConfigurationError("Missing required data for: " + "; ".join(missing) + ".")
    return selected


# ----------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------
def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_clean_df(root_dir, entry):
    path = output_storage.clean_data_path(root_dir, entry["identity"])
    df = _load_pickle(path).get("data")
    if not isinstance(df, pd.DataFrame):
        raise PlotConfigurationError(f"CleanData of '{entry['label']}' has no data table.")
    return df


def _load_cycle_container(root_dir, entry):
    path = output_storage.cycle_data_path(root_dir, entry["identity"])
    container = _load_pickle(path)
    cycles = container.get("cycles")
    if not isinstance(cycles, pd.DataFrame) or cycles.empty:
        raise PlotConfigurationError(f"CycleData of '{entry['label']}' contains no cycles.")
    return container


def _numeric_columns(root_dir, entry):
    path = output_storage.clean_data_path(root_dir, entry["identity"])
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _COLUMN_CACHE:
        df = _load_clean_df(root_dir, entry)
        _COLUMN_CACHE[key] = [
            str(c) for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
        ]
    return _COLUMN_CACHE[key]


def list_columns(root_dir, entries):
    """
    Lists the numeric CleanData columns of EACH selected experiment (column
    names are namespaced per source file, so they usually differ between
    experiments and cannot be merged into one common list).
    """
    result, unreadable = [], []
    for entry in entries:
        try:
            columns = _numeric_columns(root_dir, entry)
        except Exception as exc:
            logger.warning("Could not read columns of '%s': %s", entry["label"], exc)
            unreadable.append(entry["label"])
            continue
        result.append({"id": entry["id"], "label": entry["label"], "kind": entry["kind"], "columns": columns})
    return {"experiments": result, "unreadable": unreadable}


# ----------------------------------------------------------------------
# Custom (signal vs signal) plot
# ----------------------------------------------------------------------
def build_custom_plot(root_dir, entries, series, options):
    """
    Builds one figure from several series; each series picks its own
    experiment and its own X/Y signals: [{"experiment_id", "x_column", "y_column"}].
    """
    if not isinstance(series, list) or not series:
        raise PlotConfigurationError("Add at least one series (experiment + X and Y signals).")

    by_id = {e["id"]: e for e in entries}
    mode = options.get("mode") if options.get("mode") in ("lines", "markers", "lines+markers") else "lines"
    fig = go.Figure()
    warnings_ = []
    frames, x_names, y_names = {}, [], []
    for idx, item in enumerate(series):
        try:
            entry = by_id[int(item.get("experiment_id"))]
        except (TypeError, ValueError, KeyError, AttributeError):
            raise PlotConfigurationError(f"Series {idx + 1}: unknown experiment.")
        x_column, y_column = item.get("x_column"), item.get("y_column")
        if not x_column or not y_column:
            raise PlotConfigurationError(f"Series {idx + 1}: choose both an X and a Y signal.")

        if entry["id"] not in frames:
            frames[entry["id"]] = _load_clean_df(root_dir, entry)
        df = frames[entry["id"]]
        missing = [c for c in (x_column, y_column) if c not in df.columns]
        if missing:
            warnings_.append(f"{entry['label']}: signal(s) not found: {', '.join(missing)}.")
            continue
        x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)
        if len(x) > MAX_CUSTOM_PLOT_POINTS:
            step = int(math.ceil(len(x) / MAX_CUSTOM_PLOT_POINTS))
            x, y = x[::step], y[::step]
            warnings_.append(f"{entry['label']}: displayed every {step}-th sample ({len(x)} points).")

        color = COLOR_PALETTE[len(fig.data) % len(COLOR_PALETTE)]
        fig.add_trace(go.Scattergl(
            x=x.tolist(), y=y.tolist(), mode=mode, name=f"{entry['label']} · {_short(y_column)} vs {_short(x_column)}",
            line={"color": color}, marker={"color": color, "size": 4},
        ))
        if _short(x_column) not in x_names:
            x_names.append(_short(x_column))
        if _short(y_column) not in y_names:
            y_names.append(_short(y_column))

    if not fig.data:
        raise PlotConfigurationError("No series could be plotted: " + " ".join(warnings_))

    x_title, y_title = " / ".join(x_names), " / ".join(y_names)
    title = (options.get("title") or "").strip() or f"{y_title} vs {x_title}"
    fig.update_layout(
        title=title,
        xaxis={"title": x_title, "type": "log" if options.get("x_log") else "linear"},
        yaxis={"title": y_title, "type": "log" if options.get("y_log") else "linear"},
        template="plotly_white", hovermode="closest", legend={"orientation": "h", "y": -0.2},
        margin={"l": 70, "r": 20, "t": 60, "b": 60},
    )
    return {"title": title, "figure": _fig_json(fig), "warnings": warnings_}


def _short(column):
    return str(column).split("::")[-1]


# ----------------------------------------------------------------------
# Per-experiment metrics
# ----------------------------------------------------------------------
def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mean_std(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None, None
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def compute_experiment_metrics(root_dir, entry, voltage_column, current_column, need_power):
    """
    Computes the figures-of-merit inputs of ONE experiment from its
    CleanData signals and CycleData cycle boundaries:

    - per-cycle peak-to-peak of the voltage (and current) signal, averaged;
    - with `need_power` (requires a valid Req): P = V^2 / Req, the energy of
      every cycle E = integral of P dt (trapezoidal rule, over the cycle's
      full period), its average, the average power (total energy / total
      time of the cycles) and the peak power.
    """
    warnings_ = []
    metrics = {
        "id": entry["id"], "tribu_id": entry["tribu_id"], "rload_id": entry["rload_id"],
        "req": entry["req"], "kind": entry["kind"], "label": entry["label"], "n_cycles": 0,
        "v_ptp": None, "v_ptp_std": None, "i_ptp": None, "i_ptp_std": None,
        "energy": None, "energy_std": None, "p_avg": None, "p_peak": None,
        "warnings": warnings_,
    }

    df = _load_clean_df(root_dir, entry)
    container = _load_cycle_container(root_dir, entry)
    cycles = container["cycles"]
    time_column = (container.get("config") or {}).get("time_column")
    n_samples = len(df)

    def column_values(name):
        if not name:
            return None
        if name not in df.columns:
            warnings_.append(f"Signal '{_short(name)}' not found in CleanData.")
            return None
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

    voltage = column_values(voltage_column)
    current = column_values(current_column)
    time = column_values(time_column) if time_column else None

    bounds = []
    for start, end in zip(cycles["start_idx"], cycles["end_idx"]):
        start, end = int(start), int(end)
        if 0 <= start < end < n_samples:
            bounds.append((start, end))
    metrics["n_cycles"] = len(bounds)
    if not bounds:
        warnings_.append("No valid cycle boundaries in CycleData.")
        return metrics

    def peak_to_peak(values):
        per_cycle = []
        for start, end in bounds:
            segment = values[start:end + 1]
            segment = segment[np.isfinite(segment)]
            if segment.size:
                per_cycle.append(float(segment.max() - segment.min()))
        return _mean_std(per_cycle)

    if voltage is not None:
        metrics["v_ptp"], metrics["v_ptp_std"] = peak_to_peak(voltage)
    if current is not None:
        metrics["i_ptp"], metrics["i_ptp_std"] = peak_to_peak(current)

    if need_power and voltage is not None:
        if not entry["req"]:
            warnings_.append(f"No valid Req for RloadId '{entry['rload_id']}' in LoadsDescription.ods.")
        elif time is None:
            warnings_.append("CycleData has no time column, so energy and average power cannot be integrated.")
        else:
            power = voltage ** 2 / float(entry["req"])
            energies, duration, peak = [], 0.0, 0.0
            for start, end in bounds:
                stop = min(end + 1, n_samples - 1)
                t, p = time[start:stop + 1], power[start:stop + 1]
                ok = np.isfinite(t) & np.isfinite(p)
                if ok.sum() < 2:
                    continue
                t, p = t[ok], p[ok]
                energies.append(float(_trapezoid(p, t)))
                duration += float(t[-1] - t[0])
                peak = max(peak, float(p.max()))
            if energies:
                metrics["energy"], metrics["energy_std"] = _mean_std(energies)
                metrics["p_avg"] = float(sum(energies) / duration) if duration > 0 else None
                metrics["p_peak"] = peak
    return metrics


def _aggregate(rows, key, std_key=None):
    """
    Combines repeated experiments of the same TribuId/RloadId: mean of their
    values; the error is the spread between experiments (or, for a single
    experiment, the mean per-cycle standard deviation when available).
    """
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None, None
    mean, spread = _mean_std(values)
    if spread is None and std_key:
        inner = [r[std_key] for r in rows if r.get(std_key) is not None]
        spread = float(np.mean(inner)) if inner else None
    return mean, spread


def _group_points(metrics_rows):
    """Groups per-experiment metrics into {tribu_id: {rload_id: [rows]}}."""
    grouped = {}
    for row in metrics_rows:
        grouped.setdefault(row["tribu_id"], {}).setdefault(row["rload_id"], []).append(row)
    return grouped


# ----------------------------------------------------------------------
# Power vs R fit
# ----------------------------------------------------------------------
def _power_model(log_r, log_a, log_b):
    """P(R) = a * R / (R + b)^2: matched-load model with its maximum at R = b."""
    r = 10.0 ** log_r
    return 10.0 ** log_a * r / (r + 10.0 ** log_b) ** 2


def fit_power_curve(resistances, powers):
    """
    Fits P(R) = a*R/(R + b)^2 (fitted in log10-space of both the parameters
    and R, so loads spanning several decades are weighted evenly) with
    `scipy.optimize.curve_fit`. The fitted curve peaks at R = b with
    P = a / (4 b).

    Returns:
        dict: {"source": 'fit'|'measured', "r_opt", "p_max", "r2",
        "curve_r", "curve_p", "message"}. If there are too few points or the
        fit fails, the measured maximum is returned instead (source
        'measured', no curve).
    """
    r = np.asarray(resistances, dtype=float)
    p = np.asarray(powers, dtype=float)
    order = np.argsort(r)
    r, p = r[order], p[order]
    imax = int(np.argmax(p))
    result = {
        "source": "measured", "r_opt": float(r[imax]), "p_max": float(p[imax]), "r2": None,
        "curve_r": None, "curve_p": None, "message": None,
    }
    if len(r) < MIN_POINTS_FOR_FIT or p[imax] <= 0:
        result["message"] = f"Fewer than {MIN_POINTS_FOR_FIT} loads: the measured maximum is used instead of a fit."
        return result

    scale = float(p[imax])
    x, y = np.log10(r), p / scale
    try:
        (log_a, log_b), _ = curve_fit(
            _power_model, x, y,
            p0=[math.log10(4.0 * r[imax]), math.log10(r[imax])],
            bounds=([-30.0, x.min() - 2.0], [30.0, x.max() + 2.0]),
            maxfev=20000,
        )
    except Exception as exc:
        result["message"] = f"Curve fit failed ({exc}): the measured maximum is used instead."
        return result

    residual = float(np.sum((y - _power_model(x, log_a, log_b)) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_opt = 10.0 ** log_b
    curve_x = np.linspace(x.min() - 0.3, x.max() + 0.3, 300)
    result.update({
        "source": "fit", "r_opt": float(r_opt),
        "p_max": float(10.0 ** log_a / (4.0 * r_opt) * scale),
        "r2": (1.0 - residual / total) if total > 0 else None,
        "curve_r": (10.0 ** curve_x).tolist(),
        "curve_p": (_power_model(curve_x, log_a, log_b) * scale).tolist(),
    })
    if not (r.min() <= r_opt <= r.max()):
        result["message"] = (
            f"The fitted power peak (R = {r_opt:.3g} Ω) lies outside the measured range "
            f"({r.min():.3g}–{r.max():.3g} Ω): it is an extrapolation."
        )
    return result


# ----------------------------------------------------------------------
# Plotly figure builders
# ----------------------------------------------------------------------
def _fig_json(fig):
    """Plotly figure -> plain JSON dict (lists, NaN -> null) for Plotly.js."""
    return json.loads(fig.to_json())


def _color(tribu_id, cache):
    if tribu_id not in cache:
        cache[tribu_id] = COLOR_PALETTE[len(cache) % len(COLOR_PALETTE)]
    return cache[tribu_id]


def _layout(fig, title, x_title, y_title, log_x=False, category_x=False):
    fig.update_layout(
        title=title,
        xaxis={"title": x_title, "type": "category" if category_x else ("log" if log_x else "linear")},
        yaxis={"title": y_title},
        template="plotly_white", hovermode="closest",
        legend={"orientation": "h", "y": -0.25}, margin={"l": 70, "r": 20, "t": 60, "b": 70},
    )
    return fig


def _error_array(errors):
    return {"type": "data", "array": [e or 0.0 for e in errors], "visible": any(errors)}


def _load_points(grouped_tribu, value_key, std_key=None):
    """[(Req, RloadId, value, error)] sorted by Req for one TribuId (rows without Req/value dropped)."""
    points = []
    for rload_id, rows in grouped_tribu.items():
        req = rows[0]["req"]
        value, err = _aggregate(rows, value_key, std_key)
        if req and value is not None:
            points.append((float(req), rload_id, value, err))
    return sorted(points)


def power_value_key(power_metric):
    return "p_peak" if power_metric == POWER_METRIC_PEAK else "p_avg"


def figure_vs_r(grouped_tribu, value_key, std_key, title, y_title):
    """Single-TribuId scatter of `value_key` vs R (log axis)."""
    points = _load_points(grouped_tribu, value_key, std_key)
    if not points:
        return None
    fig = go.Figure(go.Scatter(
        x=[p[0] for p in points], y=[p[2] for p in points], mode="lines+markers",
        text=[p[1] for p in points], error_y=_error_array([p[3] for p in points]),
        hovertemplate="RLoadId %{text}<br>R = %{x:.4g} Ω<br>%{y:.4g}<extra></extra>",
        marker={"size": 9, "color": COLOR_PALETTE[0]}, name=y_title,
    ))
    return _layout(fig, title, "Load resistance R (Ω)", y_title, log_x=True)


def figure_power_fits(grouped, power_key, y_title):
    """
    Power vs R scatter of every TribuId with its fitted curve and marked
    peak. Returns (figure | None, {tribu_id: fit}, warnings).
    """
    fig = go.Figure()
    fits, warnings_, colors = {}, [], {}
    for tribu_id in sorted(grouped):
        points = _load_points(grouped[tribu_id], power_key)
        if not points:
            warnings_.append(f"TribuId {tribu_id}: no power data (is Req defined in LoadsDescription.ods?).")
            continue
        color = _color(tribu_id, colors)
        fit = fit_power_curve([p[0] for p in points], [p[2] for p in points])
        fits[tribu_id] = fit
        if fit["message"]:
            warnings_.append(f"TribuId {tribu_id}: {fit['message']}")

        fig.add_trace(go.Scatter(
            x=[p[0] for p in points], y=[p[2] for p in points], mode="markers", name=f"{tribu_id} (data)",
            legendgroup=str(tribu_id), text=[p[1] for p in points], marker={"size": 9, "color": color},
            hovertemplate="RLoadId %{text}<br>R = %{x:.4g} Ω<br>P = %{y:.4g} W<extra></extra>",
        ))
        if fit["curve_r"]:
            r2 = f", R²={fit['r2']:.3f}" if fit["r2"] is not None else ""
            fig.add_trace(go.Scatter(
                x=fit["curve_r"], y=fit["curve_p"], mode="lines", name=f"{tribu_id} (fit{r2})",
                legendgroup=str(tribu_id), line={"color": color, "dash": "dash"}, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[fit["r_opt"]], y=[fit["p_max"]], mode="markers", legendgroup=str(tribu_id), showlegend=False,
            marker={"symbol": "star", "size": 15, "color": color, "line": {"color": "black", "width": 1}},
            hovertemplate=(f"{tribu_id} {'fitted' if fit['source'] == 'fit' else 'measured'} peak"
                           "<br>R = %{x:.4g} Ω<br>P = %{y:.4g} W<extra></extra>"),
        ))
    if not fig.data:
        return None, fits, warnings_
    _layout(fig, "Power vs load resistance", "Load resistance R (Ω)", y_title, log_x=True)
    return fig, fits, warnings_


def figure_bar(categories, values, errors, title, y_title, colors=None):
    """Categorical bar chart (one bar per TribuId)."""
    colors = colors if colors is not None else {}
    fig = go.Figure(go.Bar(
        x=[str(c) for c in categories], y=values, error_y=_error_array(errors),
        marker={"color": [_color(c, colors) for c in categories]},
        text=[f"{v:.3g}" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="TribuId %{x}<br>%{y:.4g}<extra></extra>",
    ))
    return _layout(fig, title, "TribuId", y_title, category_x=True)


def figure_indicator(value, title, suffix):
    fig = go.Figure(go.Indicator(
        mode="number", value=value, title={"text": title},
        number={"valueformat": ".4g", "suffix": f" {suffix}"},
    ))
    fig.update_layout(template="plotly_white", height=260, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    return fig


# ----------------------------------------------------------------------
# Figures of merit orchestration
# ----------------------------------------------------------------------
def build_figures_of_merit(root_dir, entries, params):
    """
    Validates the selection and builds the figures of merit for it.

    Raises:
        PlotConfigurationError: for validation errors / missing parameters.

    Returns:
        dict: {"mode", "figures": [{"id","title","figure"}], "warnings", "table"}.
    """
    validation = validate_selection(entries)
    if not validation["ok"]:
        raise PlotConfigurationError(validation["error"])

    missing = [e["label"] for e in entries if not e["cycle_data_exists"]]
    if missing:
        raise PlotConfigurationError("Missing CycleData for: " + "; ".join(missing) + ".")

    mode = validation["mode"]
    power_metric = POWER_METRIC_PEAK if params.get("power_metric") == POWER_METRIC_PEAK else POWER_METRIC_AVERAGE
    fixed_rload = params.get("fixed_rload_id")
    fixed_rload = str(fixed_rload).strip() if fixed_rload not in (None, "") else None

    # Every experiment has its own Voltage (and, for OC/SC, Current) signal.
    signals = params.get("signals") if isinstance(params.get("signals"), dict) else {}

    def signal_of(entry, kind):
        chosen = signals.get(str(entry["id"])) or {}
        return (chosen.get(kind) or "").strip() or None

    for entry in entries:
        if mode == MODE_STANDARD and not signal_of(entry, "voltage"):
            raise PlotConfigurationError(f"Choose the voltage signal of: {entry['label']}.")
        if mode == MODE_OC and not signal_of(entry, "voltage"):
            raise PlotConfigurationError(f"Choose the voltage signal of: {entry['label']}.")
        if mode == MODE_SC and not signal_of(entry, "current"):
            raise PlotConfigurationError(f"Choose the current signal of: {entry['label']}.")
    if mode == MODE_STANDARD and not validation["single_tribu"] and fixed_rload is not None \
            and fixed_rload not in validation["rload_ids"]:
        raise PlotConfigurationError(f"RLoadId '{fixed_rload}' is not among the selected experiments.")

    warnings_, rows = [], []
    for entry in entries:
        try:
            metrics = compute_experiment_metrics(
                root_dir, entry, signal_of(entry, "voltage"), signal_of(entry, "current"),
                need_power=(mode == MODE_STANDARD),
            )
        except PlotConfigurationError as exc:
            warnings_.append(str(exc))
            continue
        warnings_.extend(f"{entry['label']}: {w}" for w in metrics.pop("warnings"))
        rows.append(metrics)
    if not rows:
        raise PlotConfigurationError("No data could be processed for the selected experiments.")

    grouped = _group_points(rows)
    figures = []

    def add(fig_id, title, fig):
        if fig is not None:
            figures.append({"id": fig_id, "title": title, "figure": _fig_json(fig)})

    power_key = power_value_key(power_metric)
    power_title = "Peak power (W)" if power_metric == POWER_METRIC_PEAK else "Average power (W)"

    if mode == MODE_STANDARD and validation["single_tribu"]:
        tribu_id = validation["tribu_ids"][0]
        fig, _fits, fit_warnings = figure_power_fits(grouped, power_key, power_title)
        warnings_.extend(fit_warnings)
        add("power_vs_r", f"Power vs RLoadId — TribuId {tribu_id}", fig)
        add("vpp_vs_r", f"Average peak-to-peak voltage vs RLoadId — TribuId {tribu_id}",
            figure_vs_r(grouped[tribu_id], "v_ptp", "v_ptp_std",
                        f"Average peak-to-peak voltage — TribuId {tribu_id}", "Peak-to-peak voltage (V)"))
        add("energy_vs_r", f"Average energy per cycle vs RLoadId — TribuId {tribu_id}",
            figure_vs_r(grouped[tribu_id], "energy", "energy_std",
                        f"Average energy per cycle — TribuId {tribu_id}", "Energy per cycle (J)"))

    elif mode == MODE_STANDARD:
        fig, fits, fit_warnings = figure_power_fits(grouped, power_key, power_title)
        warnings_.extend(fit_warnings)
        add("power_curves", "Power vs RLoadId — all TribuIds", fig)

        if fits:
            tribus = sorted(fits)
            add("pmax_bar", "Pmax comparison (peak of the fitted power curve)",
                figure_bar(tribus, [fits[t]["p_max"] for t in tribus], [None] * len(tribus),
                           "Pmax comparison", "Pmax (W)"))
        if fixed_rload is None:
            warnings_.append("Choose a RLoadId to compare the average peak-to-peak voltage between TribuIds.")
        else:
            tribus, values, errors = [], [], []
            for tribu_id in sorted(grouped):
                value, err = _aggregate(grouped[tribu_id].get(fixed_rload, []), "v_ptp", "v_ptp_std")
                if value is None:
                    warnings_.append(f"TribuId {tribu_id} has no experiment/voltage data for RLoadId '{fixed_rload}'.")
                    continue
                tribus.append(tribu_id)
                values.append(value)
                errors.append(err)
            if tribus:
                add("vpp_bar", f"Average peak-to-peak voltage at RLoadId {fixed_rload}",
                    figure_bar(tribus, values, errors, f"Average peak-to-peak voltage at RLoadId {fixed_rload}",
                               "Peak-to-peak voltage (V)"))
    else:
        key, std_key = ("v_ptp", "v_ptp_std") if mode == MODE_OC else ("i_ptp", "i_ptp_std")
        quantity = "Average peak-to-peak open-circuit voltage Voc" if mode == MODE_OC \
            else "Average peak-to-peak short-circuit current Isc"
        y_title, unit = ("Voc (V)", "V") if mode == MODE_OC else ("Isc (A)", "A")

        tribus, values, errors = [], [], []
        for tribu_id in sorted(grouped):
            pooled = [r for group in grouped[tribu_id].values() for r in group]
            value, err = _aggregate(pooled, key, std_key)
            if value is None:
                warnings_.append(f"TribuId {tribu_id}: no {y_title.split()[0]} could be computed.")
                continue
            tribus.append(tribu_id)
            values.append(value)
            errors.append(err)
        if len(tribus) == 1:
            add(f"{mode}_indicator", f"{quantity} — TribuId {tribus[0]}",
                figure_indicator(values[0], f"{quantity}<br>TribuId {tribus[0]}", unit))
        elif tribus:
            add(f"{mode}_bar", f"{quantity} comparison",
                figure_bar(tribus, values, errors, quantity, y_title))

    if not figures:
        raise PlotConfigurationError("No figure could be generated: " + (" ".join(warnings_) or "no valid data."))

    table = []
    for tribu_id in sorted(grouped):
        for rload_id, group in sorted(grouped[tribu_id].items()):
            row = {"tribu_id": tribu_id, "rload_id": rload_id, "req": group[0]["req"], "n_experiments": len(group)}
            for key in ("v_ptp", "i_ptp", "energy", "p_avg", "p_peak"):
                row[key] = _aggregate(group, key)[0]
            table.append({k: (_finite(v) if k not in ("tribu_id", "rload_id", "n_experiments") else v)
                          for k, v in row.items()})

    return {"mode": mode, "figures": figures, "warnings": warnings_, "table": table}


# ----------------------------------------------------------------------
# Flask views
# ----------------------------------------------------------------------
def _json_error(message, status=400):
    return jsonify({"ok": False, "error": message}), status


def _request_entries(require_cycle_data):
    payload = request.get_json(silent=True) or {}
    root_dir, experiments = _context()
    entries = _select_entries(root_dir, experiments, payload.get("experiment_ids"), require_cycle_data)
    return root_dir, entries, payload


def plot_configuration():
    """Renders the Plot Configuration page (experiment selection, custom plots, figures of merit)."""
    root_dir = session.get("root_dir")
    experiments = session.get("experiments")
    if not root_dir or not experiments:
        flash("No experiments loaded yet. Please select a root folder first.", "error")
        return redirect(url_for("render_data_loading"))

    catalog = build_catalog(root_dir, experiments)
    return render_template(
        "plot_configuration.html",
        root_dir=root_dir,
        experiments=[_public_entry(e) for e in catalog],
    )


def plot_configuration_validate():
    """AJAX: validates the OC/SC rules for the selected experiments."""
    payload = request.get_json(silent=True) or {}
    try:
        root_dir, experiments = _context()
        catalog = build_catalog(root_dir, experiments)
        ids = payload.get("experiment_ids") or []
        entries = [catalog[int(i)] for i in ids]
    except PlotConfigurationError as exc:
        return _json_error(str(exc))
    except (TypeError, ValueError, IndexError):
        return _json_error("Invalid experiment selection.")
    return jsonify(validate_selection(entries))


def plot_configuration_columns():
    """AJAX: numeric signals available in the CleanData of the selected experiments."""
    try:
        root_dir, entries, _ = _request_entries(require_cycle_data=False)
        return jsonify({"ok": True, **list_columns(root_dir, entries)})
    except PlotConfigurationError as exc:
        return _json_error(str(exc))


def plot_configuration_custom_plot():
    """AJAX: builds an individual signal-vs-signal plot for the selected experiments."""
    try:
        series = (request.get_json(silent=True) or {}).get("series")
        if not isinstance(series, list) or not series:
            raise PlotConfigurationError("Add at least one series (experiment + X and Y signals).")
        payload = request.get_json(silent=True) or {}
        payload["experiment_ids"] = sorted({s.get("experiment_id") for s in series if isinstance(s, dict)})
        root_dir, experiments = _context()
        entries = _select_entries(root_dir, experiments, payload["experiment_ids"], False)
        plot = build_custom_plot(root_dir, entries, series, payload)
        return jsonify({"ok": True, **plot})
    except PlotConfigurationError as exc:
        return _json_error(str(exc))


def plot_configuration_figures_of_merit():
    """AJAX: validates the selection (OC/SC rules) and builds the figures of merit."""
    try:
        root_dir, entries, payload = _request_entries(require_cycle_data=False)
        result = build_figures_of_merit(root_dir, entries, payload)
        return jsonify({"ok": True, **result})
    except PlotConfigurationError as exc:
        return _json_error(str(exc))
