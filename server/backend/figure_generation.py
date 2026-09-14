import io
import json
import logging
import pickle
from pathlib import Path

import pandas as pd
from flask import render_template, request, redirect, url_for, session, flash, jsonify, send_file

from server.backend.cycle_peak_extraction import CyclePeakExtractor

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
CLEAN_DATA_FOLDER_NAME = "CleanData"
CYCLE_DATA_FOLDER_NAME = "CycleData"
CLEAN_DATA_EXTENSIONS = (".pkl", ".csv")
CYCLE_DATA_EXTENSIONS = (".pkl", ".csv")

MAX_TRACE_POINTS = 3000

PLOT_TYPE_TIME_SERIES = "Time Series"
PLOT_TYPE_CYCLE_EVOLUTION = "Cycle Evolution"
PLOT_TYPE_LOAD_CURVE = "Load Curve"
PLOT_TYPE_CYCLE_WAVEFORM = "Cycle Waveform Overlay"

# Best-effort mapping from the generic plot-config "variable" names (offered
# in the Experiment Preview plot box) to the actual columns produced by the
# CycleData per-cycle metrics table (see cycle_peak_extraction.py).
CYCLE_METRIC_ALIASES = {
    "peakamplitude": "peak_to_peak",
    "voltage": "peak_value",
    "current": "peak_value",
    "charge": "peak_to_peak",
    "cycle": "cycle_number",
    "time": "duration",
}

# Qualitative color palette (Plotly default) used to consistently color
# traces/groups by TribuId or RloadId.
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _color_for(key, palette_cache):
    """Deterministically assigns a stable color to `key`, memoized in `palette_cache`."""
    if key not in palette_cache:
        palette_cache[key] = COLOR_PALETTE[len(palette_cache) % len(COLOR_PALETTE)]
    return palette_cache[key]


def _experiment_label(experiment):
    """Builds a short, human-readable trace/legend label for one experiment."""
    return (
        f"{experiment.get('TribuId', '')} "
        f"{experiment.get('SampleIdTriboPos', '')}-{experiment.get('SampleIdTriboNeg', '')} "
        f"R{experiment.get('RloadId', '')}"
    ).strip()


def _resolve_column(df, requested_name):
    """
    Resolves a user-facing variable name (e.g. 'Voltage') to an actual
    DataFrame column, trying an exact match first, then case-insensitive,
    then substring matching. Returns None if nothing reasonable is found.
    """
    if requested_name is None:
        return None
    if requested_name in df.columns:
        return requested_name

    lowered = str(requested_name).strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == lowered:
            return col
    for col in df.columns:
        if lowered in str(col).strip().lower():
            return col
    return None


def _downsample_xy(x, y, max_points=MAX_TRACE_POINTS):
    """Uniformly strides `x`/`y` to at most `max_points` samples, to keep Plotly responsive."""
    n = len(x)
    if n <= max_points:
        return list(x), list(y)
    step = max(1, n // max_points)
    return list(x[::step]), list(y[::step])


def _numeric_or_none(value):
    """Best-effort float conversion; returns None (rather than raising) on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ExperimentDataStore:
    """
    Loads and caches, per experiment, the CleanData signal table and the
    CycleData per-cycle metrics table, so repeated plot configurations
    referencing the same experiment don't re-read the same files from disk.
    """

    def __init__(self):
        self._clean_cache = {}
        self._cycle_cache = {}

    @staticmethod
    def _find_file(folder, subfolder_name, extensions):
        subfolder = Path(folder) / subfolder_name
        if not subfolder.is_dir():
            return None
        try:
            candidates = sorted(
                p for p in subfolder.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            )
        except OSError as exc:
            logger.warning("Could not read directory %s: %s", subfolder, exc)
            return None
        return candidates[0] if candidates else None

    def load_clean_data(self, experiment):
        """Returns the CleanData DataFrame for `experiment`, or None (with a logged warning) if unavailable."""
        folder = experiment.get("folder_path")
        if folder in self._clean_cache:
            return self._clean_cache[folder]

        path = self._find_file(folder, CLEAN_DATA_FOLDER_NAME, CLEAN_DATA_EXTENSIONS)
        df = None
        if path is None:
            logger.warning("No CleanData file found for experiment at %s", folder)
        elif path.suffix.lower() != ".pkl":
            df = pd.read_csv(path)
        else:
            try:
                with open(path, "rb") as f:
                    container = pickle.load(f)
                df = container.get("data")
            except Exception as exc:  # pragma: no cover - defensive catch-all
                logger.warning("Could not read CleanData file %s: %s", path, exc)

        self._clean_cache[folder] = df
        return df

    def load_cycle_data(self, experiment):
        """Returns the CycleData cycles-metrics DataFrame for `experiment`, or None if unavailable."""
        folder = experiment.get("folder_path")
        if folder in self._cycle_cache:
            return self._cycle_cache[folder]

        path = self._find_file(folder, CYCLE_DATA_FOLDER_NAME, CYCLE_DATA_EXTENSIONS)
        df = None
        if path is None:
            logger.warning("No CycleData file found for experiment at %s", folder)
        elif path.suffix.lower() != ".pkl":
            df = pd.read_csv(path)
        else:
            try:
                with open(path, "rb") as f:
                    container = pickle.load(f)
                df = container.get("cycles")
            except Exception as exc:  # pragma: no cover - defensive catch-all
                logger.warning("Could not read CycleData file %s: %s", path, exc)

        self._cycle_cache[folder] = df
        return df


class FigureBuilder:
    """
    Prepares Plotly-compatible {"data": [...traces], "layout": {...}} figures
    for each of the four supported multi-experiment plot types, given the
    experiments selected in the Experiment Preview page.
    """

    def __init__(self, experiments, data_store=None):
        self.experiments = experiments
        self.store = data_store or ExperimentDataStore()

    def build_plots(self, plot_config, plot_id_prefix):
        """
        Dispatches `plot_config` to the appropriate builder.

        Returns:
            list[dict]: One or more ready-to-render plot payloads (more
            than one only for "Cycle Waveform Overlay", which renders one
            card per experiment).
        """
        plot_type = plot_config.get("plotType", PLOT_TYPE_TIME_SERIES)
        builders = {
            PLOT_TYPE_TIME_SERIES: self._build_time_series,
            PLOT_TYPE_CYCLE_EVOLUTION: self._build_cycle_evolution,
            PLOT_TYPE_LOAD_CURVE: self._build_load_curve,
            PLOT_TYPE_CYCLE_WAVEFORM: self._build_cycle_waveform,
        }
        builder = builders.get(plot_type, self._build_time_series)

        try:
            plots = builder(plot_config)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception("Failed to build plot for config %s", plot_config)
            plots = [{
                "title": plot_config.get("title") or plot_type,
                "plot_type": plot_type,
                "data": [],
                "layout": {"title": plot_config.get("title") or plot_type},
                "warnings": [f"Could not build this plot: {exc}"],
            }]

        for i, plot in enumerate(plots):
            plot["plot_id"] = f"{plot_id_prefix}_{i}"
        return plots

    # ------------------------------------------------------------------
    # Type A: Multi-Trace Time Series
    # ------------------------------------------------------------------
    def _build_time_series(self, plot_config):
        y_var = plot_config.get("yVar", "Voltage")
        title = plot_config.get("title") or f"{y_var} vs Time"
        palette_cache = {}
        traces = []
        warnings_ = []

        for experiment in self.experiments:
            df = self.store.load_clean_data(experiment)
            if df is None:
                warnings_.append(f"{_experiment_label(experiment)}: CleanData not available, skipped.")
                continue

            time_col = _resolve_column(df, "Time") or CyclePeakExtractor.guess_time_column(df)
            y_col = _resolve_column(df, y_var)
            if time_col is None or y_col is None:
                warnings_.append(f"{_experiment_label(experiment)}: column '{y_var}' or time column not found, skipped.")
                continue

            t = df[time_col].to_numpy()
            y = df[y_col].to_numpy()
            t_relative = t - t[0] if len(t) else t

            x_ds, y_ds = _downsample_xy(t_relative, y)
            x_abs_ds, _ = _downsample_xy(t, y)
            y_max = max((abs(v) for v in y_ds if v is not None), default=0) or 1.0
            y_normalized = [v / y_max for v in y_ds]

            tribu_id = str(experiment.get("TribuId", ""))
            rload_id = str(experiment.get("RloadId", ""))

            traces.append({
                "x": x_ds,
                "y": y_ds,
                "type": "scattergl",
                "mode": "lines",
                "name": _experiment_label(experiment),
                "legendgroup": tribu_id,
                "line": {"color": _color_for(tribu_id, palette_cache)},
                "meta": {
                    "tribuId": tribu_id,
                    "rloadId": rload_id,
                    "experimentId": experiment.get("experiment_id"),
                    "colorByTribu": _color_for(tribu_id, palette_cache),
                    "colorByRload": _color_for(rload_id, palette_cache),
                    "xAbsolute": x_abs_ds,
                    "xRelative": x_ds,
                    "yRaw": y_ds,
                    "yNormalized": y_normalized,
                },
            })

        layout = {
            "title": title,
            "xaxis": {"title": "Time (relative, s)"},
            "yaxis": {"title": y_var},
            "legend": {"orientation": "h"},
        }
        return [{"title": title, "plot_type": PLOT_TYPE_TIME_SERIES, "data": traces, "layout": layout, "warnings": warnings_}]

    # ------------------------------------------------------------------
    # Type B: Cycle Evolution
    # ------------------------------------------------------------------
    def _build_cycle_evolution(self, plot_config):
        y_var = plot_config.get("yVar", "PeakAmplitude")
        title = plot_config.get("title") or f"{y_var} vs Cycle"
        y_metric_col = CYCLE_METRIC_ALIASES.get(str(y_var).strip().lower(), str(y_var))

        palette_cache = {}
        traces = []
        warnings_ = []

        for experiment in self.experiments:
            cycles_df = self.store.load_cycle_data(experiment)
            if cycles_df is None or cycles_df.empty:
                warnings_.append(f"{_experiment_label(experiment)}: CycleData not available, skipped.")
                continue

            resolved_col = _resolve_column(cycles_df, y_metric_col)
            if resolved_col is None:
                warnings_.append(f"{_experiment_label(experiment)}: metric '{y_var}' not found in CycleData, skipped.")
                continue

            x = cycles_df["cycle_number"].tolist() if "cycle_number" in cycles_df.columns else list(range(1, len(cycles_df) + 1))
            y = cycles_df[resolved_col].tolist()

            tribu_id = str(experiment.get("TribuId", ""))
            traces.append({
                "x": x,
                "y": y,
                "type": "scatter",
                "mode": "lines+markers",
                "name": _experiment_label(experiment),
                "legendgroup": tribu_id,
                "line": {"color": _color_for(tribu_id, palette_cache)},
                "meta": {
                    "tribuId": tribu_id,
                    "rloadId": str(experiment.get("RloadId", "")),
                    "experimentId": experiment.get("experiment_id"),
                    "colorByTribu": _color_for(tribu_id, palette_cache),
                    "colorByRload": _color_for(str(experiment.get("RloadId", "")), palette_cache),
                    "yRaw": y,
                    "yNormalized": [v / (max((abs(u) for u in y), default=0) or 1.0) for v in y],
                },
            })

        layout = {
            "title": title,
            "xaxis": {"title": "Cycle number"},
            "yaxis": {"title": y_var},
            "legend": {"orientation": "h"},
        }
        return [{"title": title, "plot_type": PLOT_TYPE_CYCLE_EVOLUTION, "data": traces, "layout": layout, "warnings": warnings_}]

    # ------------------------------------------------------------------
    # Type C: Parametric / Load Curves
    # ------------------------------------------------------------------
    def _build_load_curve(self, plot_config):
        y_var = plot_config.get("yVar", "PeakAmplitude")
        title = plot_config.get("title") or f"{y_var} vs Rload"
        y_metric_col = CYCLE_METRIC_ALIASES.get(str(y_var).strip().lower(), str(y_var))

        palette_cache = {}
        warnings_ = []
        groups = {}  # tribu_id -> {"x": [...], "y": [...], "labels": [...]}

        for experiment in self.experiments:
            cycles_df = self.store.load_cycle_data(experiment)
            if cycles_df is None or cycles_df.empty:
                warnings_.append(f"{_experiment_label(experiment)}: CycleData not available, skipped.")
                continue

            resolved_col = _resolve_column(cycles_df, y_metric_col)
            if resolved_col is None:
                warnings_.append(f"{_experiment_label(experiment)}: metric '{y_var}' not found in CycleData, skipped.")
                continue

            rload_value = _numeric_or_none(experiment.get("RloadId"))
            if rload_value is None:
                warnings_.append(f"{_experiment_label(experiment)}: RloadId is not numeric, skipped from load curve.")
                continue

            metric_value = float(pd.Series(cycles_df[resolved_col]).max())
            tribu_id = str(experiment.get("TribuId", ""))
            group = groups.setdefault(tribu_id, {"x": [], "y": [], "labels": []})
            group["x"].append(rload_value)
            group["y"].append(metric_value)
            group["labels"].append(_experiment_label(experiment))

        traces = []
        for tribu_id, group in groups.items():
            order = sorted(range(len(group["x"])), key=lambda i: group["x"][i])
            x_sorted = [group["x"][i] for i in order]
            y_sorted = [group["y"][i] for i in order]
            labels_sorted = [group["labels"][i] for i in order]

            traces.append({
                "x": x_sorted,
                "y": y_sorted,
                "type": "scatter",
                "mode": "lines+markers",
                "name": tribu_id,
                "legendgroup": tribu_id,
                "line": {"color": _color_for(tribu_id, palette_cache)},
                "text": labels_sorted,
                "meta": {
                    "tribuId": tribu_id,
                    "colorByTribu": _color_for(tribu_id, palette_cache),
                    "yRaw": y_sorted,
                    "yNormalized": [v / (max((abs(u) for u in y_sorted), default=0) or 1.0) for v in y_sorted],
                },
            })

        layout = {
            "title": title,
            "xaxis": {"title": "Rload", "type": "log"},
            "yaxis": {"title": y_var},
            "legend": {"orientation": "h"},
        }
        return [{"title": title, "plot_type": PLOT_TYPE_LOAD_CURVE, "data": traces, "layout": layout, "warnings": warnings_}]

    # ------------------------------------------------------------------
    # Type D: Cycle Waveform Overlay (one card per experiment)
    # ------------------------------------------------------------------
    def _build_cycle_waveform(self, plot_config):
        y_var = plot_config.get("yVar", "Voltage")
        plots = []

        for experiment in self.experiments:
            warnings_ = []
            clean_df = self.store.load_clean_data(experiment)
            cycles_df = self.store.load_cycle_data(experiment)
            title = plot_config.get("title") or f"{y_var} — Cycle overlay ({_experiment_label(experiment)})"

            if clean_df is None or cycles_df is None or cycles_df.empty:
                plots.append({
                    "title": title,
                    "plot_type": PLOT_TYPE_CYCLE_WAVEFORM,
                    "data": [],
                    "layout": {"title": title},
                    "warnings": [f"{_experiment_label(experiment)}: CleanData/CycleData not available, skipped."],
                })
                continue

            time_col = _resolve_column(clean_df, "Time") or CyclePeakExtractor.guess_time_column(clean_df)
            y_col = _resolve_column(clean_df, y_var)
            if time_col is None or y_col is None:
                plots.append({
                    "title": title,
                    "plot_type": PLOT_TYPE_CYCLE_WAVEFORM,
                    "data": [],
                    "layout": {"title": title},
                    "warnings": [f"{_experiment_label(experiment)}: column '{y_var}' or time column not found, skipped."],
                })
                continue

            t = clean_df[time_col].to_numpy()
            y = clean_df[y_col].to_numpy()
            palette_cache = {}
            traces = []

            for _, cycle_row in cycles_df.iterrows():
                start_idx = int(cycle_row.get("start_idx", 0))
                end_idx = int(cycle_row.get("end_idx", 0))
                if end_idx <= start_idx or end_idx >= len(t):
                    continue

                segment_t = t[start_idx:end_idx + 1]
                segment_y = y[start_idx:end_idx + 1]
                relative_t = segment_t - segment_t[0] if len(segment_t) else segment_t

                cycle_number = cycle_row.get("cycle_number", len(traces) + 1)
                x_ds, y_ds = _downsample_xy(relative_t, segment_y)

                traces.append({
                    "x": x_ds,
                    "y": y_ds,
                    "type": "scattergl",
                    "mode": "lines",
                    "name": f"Cycle {cycle_number}",
                    "line": {"color": _color_for(cycle_number, palette_cache)},
                    "meta": {
                        "cycleNumber": cycle_number,
                        "yRaw": y_ds,
                        "yNormalized": [v / (max((abs(u) for u in y_ds), default=0) or 1.0) for v in y_ds],
                    },
                })

            if not traces:
                warnings_.append(f"{_experiment_label(experiment)}: no valid cycle segments to overlay.")

            layout = {
                "title": title,
                "xaxis": {"title": "Time within cycle (relative, s)"},
                "yaxis": {"title": y_var},
                "legend": {"orientation": "h"},
            }
            plots.append({"title": title, "plot_type": PLOT_TYPE_CYCLE_WAVEFORM, "data": traces, "layout": layout, "warnings": warnings_})

        return plots


# ----------------------------------------------------------------------
# Flask views
# ----------------------------------------------------------------------
def figure_generation():
    """
    Renders the multi-plot dashboard for the experiments/plot configurations
    selected in the Experiment Preview page (stored in the session by
    `experiments_preview.generate_plots`).

    Returns:
        str: Rendered HTML template with the generated Plotly figures, or a
        redirect back to the experiments preview page if nothing was
        selected.
    """
    selected_experiments = session.get("selected_experiments")
    plot_configs = session.get("plot_configs")

    if not selected_experiments or not plot_configs:
        flash("No experiments/plots selected yet. Please configure plots from the experiments preview page.", "error")
        return redirect(url_for("render_experiments_preview"))

    plots, warnings_ = _build_all_plots(selected_experiments, plot_configs)

    for warning in warnings_:
        flash(warning, "warning")

    return render_template(
        "figure_generation.html",
        plots_json=json.dumps(plots),
        experiment_count=len(selected_experiments),
        plot_count=len(plots),
    )


def _build_all_plots(selected_experiments, plot_configs):
    """Builds every plot payload for the given experiments/configs, collecting warnings along the way."""
    store = ExperimentDataStore()
    builder = FigureBuilder(selected_experiments, store)

    plots = []
    warnings_ = []
    for idx, plot_config in enumerate(plot_configs):
        for plot in builder.build_plots(plot_config, plot_id_prefix=f"plot{idx}"):
            warnings_.extend(f"{plot['title']}: {w}" for w in plot.pop("warnings", []))
            plots.append(plot)
    return plots, warnings_


def export_plot_data():
    """
    Rebuilds the currently selected plots' underlying numerical data and
    returns it as a multi-sheet .xlsx workbook (one sheet per plot), for
    users who want the consolidated raw numbers behind the charts.

    Returns:
        flask.Response: An .xlsx file download, or a JSON error if no
        plots are currently configured in the session.
    """
    selected_experiments = session.get("selected_experiments")
    plot_configs = session.get("plot_configs")

    if not selected_experiments or not plot_configs:
        return jsonify({"error": "No experiments/plots selected."}), 400

    plots, _ = _build_all_plots(selected_experiments, plot_configs)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        used_sheet_names = set()
        for plot in plots:
            sheet_name = (plot.get("title") or plot["plot_id"])[:31] or plot["plot_id"]
            base_name, suffix = sheet_name, 1
            while sheet_name in used_sheet_names:
                suffix += 1
                sheet_name = f"{base_name[:28]}_{suffix}"
            used_sheet_names.add(sheet_name)

            rows = []
            for trace in plot.get("data", []):
                name = trace.get("name", "")
                for x_value, y_value in zip(trace.get("x", []), trace.get("y", [])):
                    rows.append({"trace": name, "x": x_value, "y": y_value})

            sheet_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["trace", "x", "y"])
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="figure_generation_data.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )