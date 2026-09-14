import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from flask import render_template, request, redirect, url_for, session, flash, jsonify
from scipy.signal import butter, filtfilt, find_peaks

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
MAX_CHART_POINTS = 5000

CLEAN_DATA_FOLDER_NAME = "CleanData"
CLEAN_DATA_EXTENSIONS = (".pkl", ".csv")

CYCLE_DATA_FOLDER_NAME = "CycleData"
CYCLE_DATA_EXTENSIONS = (".pkl", ".csv")
CYCLE_DATA_SUFFIX = "_cycle.pkl"

MODE_THRESHOLD = "threshold"
MODE_MOTOR_BOOLEAN = "motor_boolean"
MODE_PEAK_DISTANCE = "peak_distance"
SUPPORTED_MODES = (MODE_THRESHOLD, MODE_MOTOR_BOOLEAN, MODE_PEAK_DISTANCE)

STRATEGY_CYCLE_FIRST = "cycle_first"
STRATEGY_FIND_PEAKS = "find_peaks"


class CycleExtractionError(Exception):
    """Raised when a cycle/peak extraction configuration is invalid or cannot be executed."""


# ----------------------------------------------------------------------
# Configuration data model
# ----------------------------------------------------------------------
@dataclass
class CycleExtractionConfig:
    """
    Serializable configuration for one cycle/peak extraction run.

    `params` holds all mode/strategy-specific numeric knobs so the schema
    can grow without changing the dataclass shape:
      - threshold mode: threshold, edge ('rising'/'falling'/'both'), hysteresis
      - motor_boolean mode: transition ('rising'/'falling'/'both')
      - peak_distance mode & find_peaks strategy: height, distance,
        prominence, width, prefilter_cutoff, fs
    """

    mode: str
    signal_column: str
    time_column: Optional[str] = None
    motor_column: Optional[str] = None
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        if not data.get("mode") or not data.get("signal_column"):
            raise CycleExtractionError(f"Malformed cycle extraction config: {data}")
        if data["mode"] not in SUPPORTED_MODES:
            raise CycleExtractionError(f"Unsupported segmentation mode: '{data['mode']}'")
        return cls(
            mode=data["mode"],
            signal_column=data["signal_column"],
            time_column=data.get("time_column") or None,
            motor_column=data.get("motor_column") or None,
            params=data.get("params") or {},
        )


# ----------------------------------------------------------------------
# Extraction service
# ----------------------------------------------------------------------
class CyclePeakExtractor:
    """
    Loads a CleanData container and performs cycle segmentation plus
    peak/trough extraction according to a `CycleExtractionConfig`, using
    one of three selectable segmentation modes and one of two peak/trough
    extraction strategies (cycle-first for motor-driven segmentation,
    independent `scipy.signal.find_peaks` otherwise).
    """

    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.clean_data_dir = self.folder_path / CLEAN_DATA_FOLDER_NAME
        self.cycle_data_dir = self.folder_path / CYCLE_DATA_FOLDER_NAME

    @property
    def cycle_data_path(self):
        """Default target path for the saved CycleData container."""
        return self.cycle_data_dir / f"{self.folder_path.name}{CYCLE_DATA_SUFFIX}"

    # ------------------------------------------------------------------
    # CleanData loading
    # ------------------------------------------------------------------
    def _find_clean_data_file(self):
        if not self.clean_data_dir.is_dir():
            raise CycleExtractionError(f"CleanData folder not found: {self.clean_data_dir}")
        try:
            candidates = sorted(
                p for p in self.clean_data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in CLEAN_DATA_EXTENSIONS
            )
        except OSError as exc:
            raise CycleExtractionError(f"Could not read CleanData folder: {exc}")
        if not candidates:
            raise CycleExtractionError(f"No CleanData file found in {self.clean_data_dir}")
        return candidates[0]

    def load_clean_data(self):
        """Loads the processed CleanData DataFrame (raw + filtered columns)."""
        path = self._find_clean_data_file()
        if path.suffix.lower() != ".pkl":
            return pd.read_csv(path)

        with open(path, "rb") as f:
            container = pickle.load(f)

        df = container.get("data")
        if not isinstance(df, pd.DataFrame):
            raise CycleExtractionError(f"CleanData file is missing its processed data table: {path}")
        return df

    @staticmethod
    def numeric_columns(df):
        """Returns the names of all numerical columns in `df`."""
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    @staticmethod
    def boolean_like_columns(df):
        """Columns that are boolean dtype, or numeric with only {0, 1} values (candidate motor signals)."""
        columns = []
        for c in df.columns:
            series = df[c]
            if pd.api.types.is_bool_dtype(series):
                columns.append(c)
                continue
            if pd.api.types.is_numeric_dtype(series):
                uniques = set(pd.unique(series.dropna()))
                if uniques and uniques.issubset({0, 1, 0.0, 1.0}):
                    columns.append(c)
        return columns

    @staticmethod
    def guess_time_column(df):
        """Best-effort guess of which column represents time."""
        for c in df.columns:
            if str(c).strip().lower() in ("time", "t", "time (s)", "timestamp"):
                return c
        numeric = CyclePeakExtractor.numeric_columns(df)
        return numeric[0] if numeric else None

    # ------------------------------------------------------------------
    # Segmentation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hysteresis_crossings(values, threshold, hysteresis):
        """
        Schmitt-trigger style threshold crossing detector: returns
        (rising_indices, falling_indices), using `[threshold - hysteresis,
        threshold + hysteresis]` as the dead-band to avoid chatter.
        """
        low = threshold - abs(hysteresis)
        high = threshold + abs(hysteresis)
        rising, falling = [], []
        state_above = values[0] >= high
        for i in range(1, len(values)):
            if not state_above and values[i] >= high:
                rising.append(i)
                state_above = True
            elif state_above and values[i] <= low:
                falling.append(i)
                state_above = False
        return rising, falling

    @staticmethod
    def _markers_to_cycles(markers, n_samples):
        """Converts a sorted list of segmentation marker indices into (start, end) cycle bounds."""
        markers = sorted({int(m) for m in markers if 0 <= m < n_samples})
        if not markers:
            return []
        cycles = [(markers[i], markers[i + 1] - 1) for i in range(len(markers) - 1)]
        cycles.append((markers[-1], n_samples - 1))
        return [c for c in cycles if c[1] > c[0]]

    def _segment_threshold(self, values, params):
        threshold = params.get("threshold")
        if threshold is None or threshold == "":
            raise CycleExtractionError("A threshold value is required for threshold-based segmentation.")
        hysteresis = float(params.get("hysteresis") or 0.0)
        edge = params.get("edge", "rising")

        rising, falling = self._hysteresis_crossings(values, float(threshold), hysteresis)
        markers = sorted(rising + falling) if edge == "both" else (rising if edge == "rising" else falling)

        cycles = self._markers_to_cycles(markers, len(values))
        if not cycles:
            raise CycleExtractionError("No threshold crossings found; try a different threshold/hysteresis.")
        return cycles

    def _segment_motor_boolean(self, df, motor_column, params, n_samples):
        if not motor_column or motor_column not in df.columns:
            raise CycleExtractionError("A motor boolean column must be selected for motor-based segmentation.")

        motor_values = df[motor_column].to_numpy(dtype=float)
        transition = params.get("transition", "rising")
        rising, falling = self._hysteresis_crossings(motor_values, 0.5, 0.0)
        markers = sorted(rising + falling) if transition == "both" else (rising if transition == "rising" else falling)

        if not markers:
            raise CycleExtractionError(f"Motor column '{motor_column}' never transitions; no cycles found.")

        cycles = self._markers_to_cycles(markers, n_samples)
        if not cycles:
            raise CycleExtractionError(f"Motor column '{motor_column}' produced no usable cycles.")
        return cycles

    def _segment_peak_distance(self, values, params, warnings_):
        detection_signal = self._prepare_detection_signal(values, params)
        kwargs = self._find_peaks_kwargs(params)

        candidate_peaks, _ = find_peaks(detection_signal, **kwargs)
        if len(candidate_peaks) < 2:
            raise CycleExtractionError(
                "Not enough peaks were detected to estimate a cycle period. Adjust find_peaks parameters."
            )

        avg_period = int(round(float(np.mean(np.diff(candidate_peaks)))))
        if avg_period <= 0:
            raise CycleExtractionError("Estimated cycle period is invalid (<= 0 samples).")

        warnings_.append(f"Estimated cycle period: {avg_period} sample(s) from {len(candidate_peaks)} candidate peak(s).")

        boundaries = list(range(int(candidate_peaks[0]), len(values), avg_period))
        if boundaries[-1] != len(values) - 1:
            boundaries.append(len(values) - 1)

        cycles = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        cycles = [c for c in cycles if c[1] > c[0]]
        if not cycles:
            raise CycleExtractionError("Peak-distance segmentation produced no usable cycles.")
        return cycles

    # ------------------------------------------------------------------
    # Peak/trough extraction strategies
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_detection_signal(values, params):
        """
        Optionally applies an aggressive low-pass filter to a *copy* of the
        signal, used only to stabilize peak detection; the underlying
        waveform values reported for peaks/troughs always come from the
        original (unfiltered) array.
        """
        cutoff = params.get("prefilter_cutoff")
        fs = params.get("fs")
        if not cutoff or not fs:
            return values
        try:
            nyquist = 0.5 * float(fs)
            normalized_cutoff = min(max(float(cutoff) / nyquist, 1e-6), 0.999999)
            b, a = butter(4, normalized_cutoff, btype="low", analog=False)
            return filtfilt(b, a, values)
        except Exception as exc:
            logger.warning("Could not apply peak-detection pre-filter: %s", exc)
            return values

    @staticmethod
    def _find_peaks_kwargs(params):
        kwargs = {}
        for key in ("height", "prominence", "width"):
            value = params.get(key)
            if value not in (None, ""):
                kwargs[key] = float(value)
        distance = params.get("distance")
        if distance not in (None, ""):
            kwargs["distance"] = max(1, int(distance))
        return kwargs

    def _extract_find_peaks(self, values, params):
        detection_signal = self._prepare_detection_signal(values, params)
        kwargs = self._find_peaks_kwargs(params)
        peak_indices, _ = find_peaks(detection_signal, **kwargs)
        trough_indices, _ = find_peaks(-detection_signal, **kwargs)
        return peak_indices, trough_indices

    @staticmethod
    def _extract_cycle_first_peaks(values, cycles):
        peak_indices, trough_indices = [], []
        for start, end in cycles:
            segment = values[start:end + 1]
            peak_indices.append(start + int(np.argmax(segment)))
            trough_indices.append(start + int(np.argmin(segment)))
        return np.array(peak_indices, dtype=int), np.array(trough_indices, dtype=int)

    # ------------------------------------------------------------------
    # Cycle table assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _build_cycle_table(df, time_col, signal_col, cycles, peak_indices, trough_indices, warnings_):
        peak_indices = np.asarray(peak_indices, dtype=int)
        trough_indices = np.asarray(trough_indices, dtype=int)
        signal_values = df[signal_col].to_numpy(dtype=float)

        rows = []
        for i, (start, end) in enumerate(cycles):
            row = {"cycle_number": i + 1, "start_idx": int(start), "end_idx": int(end)}

            if time_col:
                start_time = df[time_col].iloc[start]
                end_time = df[time_col].iloc[end]
                row["start_time"] = start_time
                row["end_time"] = end_time
                row["duration"] = end_time - start_time
            else:
                row["start_time"] = None
                row["end_time"] = None
                row["duration"] = end - start

            cycle_peaks = peak_indices[(peak_indices >= start) & (peak_indices <= end)]
            cycle_troughs = trough_indices[(trough_indices >= start) & (trough_indices <= end)]

            if len(cycle_peaks) > 0:
                best_peak = int(cycle_peaks[np.argmax(signal_values[cycle_peaks])])
                row["peak_idx"] = best_peak
                row["peak_value"] = float(signal_values[best_peak])
                row["peak_time"] = df[time_col].iloc[best_peak] if time_col else None
            else:
                row["peak_idx"] = None
                row["peak_value"] = float("nan")
                row["peak_time"] = None
                warnings_.append(f"No peak detected in cycle {i + 1}.")

            if len(cycle_troughs) > 0:
                best_trough = int(cycle_troughs[np.argmin(signal_values[cycle_troughs])])
                row["trough_idx"] = best_trough
                row["trough_value"] = float(signal_values[best_trough])
                row["trough_time"] = df[time_col].iloc[best_trough] if time_col else None
            else:
                row["trough_idx"] = None
                row["trough_value"] = float("nan")
                row["trough_time"] = None
                warnings_.append(f"No trough detected in cycle {i + 1}.")

            row["peak_to_peak"] = row["peak_value"] - row["trough_value"]
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # High-level pipeline entry points
    # ------------------------------------------------------------------
    def run(self, config):
        """
        Executes segmentation + peak/trough extraction for `config` against
        a freshly loaded CleanData table.

        Returns:
            dict: {"raw_df", "cycles_df", "peak_indices", "trough_indices",
            "warnings"}.
        """
        df = self.load_clean_data()
        warnings_ = []

        if config.signal_column not in df.columns:
            raise CycleExtractionError(f"Signal column '{config.signal_column}' not found in CleanData.")

        values = df[config.signal_column].to_numpy(dtype=float)
        time_col = config.time_column if config.time_column in df.columns else None

        if config.mode == MODE_THRESHOLD:
            cycles = self._segment_threshold(values, config.params)
            strategy = STRATEGY_FIND_PEAKS
        elif config.mode == MODE_MOTOR_BOOLEAN:
            cycles = self._segment_motor_boolean(df, config.motor_column, config.params, len(values))
            strategy = STRATEGY_CYCLE_FIRST
        else:  # MODE_PEAK_DISTANCE
            cycles = self._segment_peak_distance(values, config.params, warnings_)
            strategy = STRATEGY_FIND_PEAKS

        if strategy == STRATEGY_CYCLE_FIRST:
            peak_indices, trough_indices = self._extract_cycle_first_peaks(values, cycles)
        else:
            peak_indices, trough_indices = self._extract_find_peaks(values, config.params)
            if len(peak_indices) == 0:
                warnings_.append("No peaks were detected with the current find_peaks parameters.")
            if len(trough_indices) == 0:
                warnings_.append("No troughs were detected with the current find_peaks parameters.")

        cycles_df = self._build_cycle_table(df, time_col, config.signal_column, cycles, peak_indices, trough_indices, warnings_)

        return {
            "raw_df": df,
            "cycles_df": cycles_df,
            "peak_indices": peak_indices,
            "trough_indices": trough_indices,
            "warnings": warnings_,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, config):
        """
        Runs the pipeline and writes the result to
        `CycleData/<experiment>_cycle.pkl`, alongside the serialized
        configuration metadata, so it can be re-opened for editing later.

        Returns:
            tuple[pathlib.Path, dict]: The saved file path and the `run()` result.
        """
        result = self.run(config)

        self.cycle_data_dir.mkdir(parents=True, exist_ok=True)

        container = {
            "cycles": result["cycles_df"],
            "peak_indices": [int(i) for i in result["peak_indices"]],
            "trough_indices": [int(i) for i in result["trough_indices"]],
            "config": config.to_dict(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

        with open(self.cycle_data_path, "wb") as f:
            pickle.dump(container, f)

        return self.cycle_data_path, result

    def find_existing_cycle_file(self):
        """Returns the first existing CycleData file for this experiment, or None."""
        if not self.cycle_data_dir.is_dir():
            return None
        try:
            candidates = sorted(
                p for p in self.cycle_data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in CYCLE_DATA_EXTENSIONS
            )
        except OSError as exc:
            logger.warning("Could not read CycleData directory %s: %s", self.cycle_data_dir, exc)
            return None
        return candidates[0] if candidates else None

    def load_existing(self):
        """
        Deserializes a previously saved CycleData container (config +
        cycles/peaks), if one exists, for edit mode.

        Returns:
            dict | None: {"path", "cycles", "peak_indices",
            "trough_indices", "config", "saved_at"} or None if no CycleData
            file exists yet.
        """
        path = self.find_existing_cycle_file()
        if path is None or path.suffix.lower() != ".pkl":
            return None

        with open(path, "rb") as f:
            container = pickle.load(f)

        return {
            "path": path,
            "cycles": container.get("cycles"),
            "peak_indices": container.get("peak_indices", []),
            "trough_indices": container.get("trough_indices", []),
            "config": CycleExtractionConfig.from_dict(container.get("config")),
            "saved_at": container.get("saved_at"),
        }


# ----------------------------------------------------------------------
# Flask views
# ----------------------------------------------------------------------
def cycle_peak_extraction():
    """
    Render the CycleData editing page for the experiment currently selected
    in the session (set by `experiments_preview.open_cycle_data`).

    On GET, loads the CleanData table (for column discovery and the initial
    chart) and, if a CycleData file already exists, deserializes its
    configuration so the UI can be re-populated for editing.

    Returns:
        str: Rendered HTML template for the cycle/peak extraction page, or a
        redirect back to the experiments preview page if no experiment is
        selected or the CleanData table cannot be read.
    """
    experiment = session.get("current_experiment")
    if not experiment or session.get("editor_mode") != "cycle":
        flash("No experiment selected for CycleData. Please pick one from the experiments preview.", "error")
        return redirect(url_for("render_experiments_preview"))

    extractor = CyclePeakExtractor(experiment["folder_path"])

    try:
        df = extractor.load_clean_data()
    except CycleExtractionError as exc:
        flash(str(exc), "error")
        return redirect(url_for("render_experiments_preview"))

    numeric_columns = extractor.numeric_columns(df)
    boolean_columns = extractor.boolean_like_columns(df)
    default_time_column = extractor.guess_time_column(df)

    existing = None
    try:
        existing = extractor.load_existing()
    except Exception:  # pragma: no cover - defensive catch-all
        logger.exception("Could not load existing CycleData for %s", experiment["folder_path"])
        flash("Existing CycleData file could not be read; starting a fresh configuration.", "warning")

    if existing:
        config = existing["config"]
        if existing["saved_at"]:
            flash(f"Loaded existing CycleData configuration (saved at {existing['saved_at']}).", "success")
    else:
        config = CycleExtractionConfig(
            mode=MODE_THRESHOLD,
            signal_column=numeric_columns[0] if numeric_columns else "",
            time_column=default_time_column,
            motor_column=None,
            params={},
        )

    return render_template(
        "cycle_peak_extraction.html",
        experiment=experiment,
        numeric_columns_json=json.dumps(numeric_columns),
        boolean_columns_json=json.dumps(boolean_columns),
        default_time_column_json=json.dumps(default_time_column),
        config_json=json.dumps(config.to_dict()),
        chart_data_json=json.dumps(_dataframe_to_chart_json(df)),
        has_existing=existing is not None,
    )


def preview_cycles():
    """
    Recomputes cycle segmentation and peak/trough extraction for a
    candidate configuration against the CleanData table, so the front-end
    can update its chart preview.

    Expects a JSON body matching `CycleExtractionConfig.to_dict()`.

    Returns:
        flask.Response: JSON payload with the chart data, cycle table,
        peak/trough indices, and any warnings; or a JSON error.
    """
    experiment = session.get("current_experiment")
    if not experiment:
        return jsonify({"error": "No experiment selected."}), 400

    try:
        config = CycleExtractionConfig.from_dict(request.get_json(force=True, silent=True) or {})
    except CycleExtractionError as exc:
        return jsonify({"error": str(exc)}), 400

    extractor = CyclePeakExtractor(experiment["folder_path"])
    try:
        result = extractor.run(config)
    except CycleExtractionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while detecting cycles/peaks")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    return jsonify(
        {
            "chart_data": _dataframe_to_chart_json(result["raw_df"]),
            "cycles": _cycles_to_json(result["cycles_df"]),
            "peak_indices": [int(i) for i in result["peak_indices"]],
            "trough_indices": [int(i) for i in result["trough_indices"]],
            "warnings": result["warnings"],
        }
    )


def save_cycle_data():
    """
    Executes the submitted configuration end-to-end, writes it to
    `CycleData/<experiment>_cycle.pkl` and returns a redirect URL back to
    the experiments preview page for the front-end to navigate to.

    Expects a JSON body matching `CycleExtractionConfig.to_dict()`.

    Returns:
        flask.Response: JSON payload {"message": str, "redirect_url": str}
        on success, or a JSON error otherwise.
    """
    experiment = session.get("current_experiment")
    if not experiment:
        return jsonify({"error": "No experiment selected."}), 400

    try:
        config = CycleExtractionConfig.from_dict(request.get_json(force=True, silent=True) or {})
    except CycleExtractionError as exc:
        return jsonify({"error": str(exc)}), 400

    extractor = CyclePeakExtractor(experiment["folder_path"])
    try:
        cycle_path, result = extractor.save(config)
    except CycleExtractionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while saving CycleData")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    flash(f"CycleData saved to {cycle_path} ({len(result['cycles_df'])} cycle(s)).", "success")
    for warn in result["warnings"]:
        flash(warn, "warning")

    return jsonify(
        {
            "message": "CycleData saved successfully.",
            "redirect_url": url_for("render_experiments_preview"),
        }
    )


def _dataframe_to_chart_json(df, max_points=MAX_CHART_POINTS):
    """
    Converts a DataFrame's numeric columns into a JSON-friendly dict,
    downsampling uniformly if it has more than `max_points` rows (keeps
    the browser/Plotly responsive for large CleanData files).

    Note: downsampling is only safe for the raw chart preview; peak/trough
    sample indices returned separately always refer to the *full* array.
    """
    n = len(df)
    step = max(1, n // max_points) if n > max_points else 1
    sampled = df.iloc[::step]

    return {
        "columns": {
            col: sampled[col].tolist()
            for col in sampled.columns
            if pd.api.types.is_numeric_dtype(sampled[col])
        },
        "n_samples": n,
        "sampled_n": len(sampled),
        "sample_step": step,
    }


def _cycles_to_json(cycles_df):
    """Converts the per-cycle metrics DataFrame into a JSON-safe list of dicts."""
    if cycles_df is None or cycles_df.empty:
        return []

    records = cycles_df.to_dict(orient="records")
    safe_records = []
    for row in records:
        safe_row = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            if isinstance(value, float) and np.isnan(value):
                value = None
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            safe_row[key] = value
        safe_records.append(safe_row)
    return safe_records
