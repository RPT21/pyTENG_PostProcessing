import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from flask import render_template, request, redirect, url_for, session, flash, jsonify
from scipy.signal import butter, filtfilt, iirnotch, medfilt

from server.backend.experiments_preview import (
    LoadsDescriptionRepository,
    _resolve_postprocessing_metadata,
    record_postprocessing_metadata,
)

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
MAX_CHART_POINTS = 5000

RAW_DATA_EXTENSIONS = (".pkl", ".csv", ".xlsx", ".tdms")
CLEAN_DATA_FOLDER_NAME = "CleanData"
CLEAN_DATA_EXTENSIONS = (".pkl", ".csv")
CLEAN_DATA_SUFFIX = "_clean.pkl"

STEP_LOWPASS = "lowpass"
STEP_MEDIAN = "median"
STEP_NOTCH = "notch"
SUPPORTED_STEP_TYPES = (STEP_LOWPASS, STEP_MEDIAN, STEP_NOTCH)


class RecipeError(Exception):
    """Raised when a recipe (or one of its steps) cannot be validated or executed."""


# ----------------------------------------------------------------------
# Recipe data model
# ----------------------------------------------------------------------
@dataclass
class RecipeStep:
    """A single, order-dependent transformation step in a cleaning recipe."""

    step_type: str
    input_column: str
    output_column: str
    params: dict = field(default_factory=dict)
    step_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        if "step_type" not in data or "input_column" not in data:
            raise RecipeError(f"Malformed recipe step: {data}")
        return cls(
            step_type=data["step_type"],
            input_column=data["input_column"],
            output_column=data.get("output_column") or f"{data['input_column']}_{data['step_type']}",
            params=data.get("params") or {},
            step_id=data.get("step_id"),
        )


@dataclass
class ClipConfig:
    """Uniform temporal clipping window, applied after all filter steps."""

    time_column: Optional[str] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            time_column=data.get("time_column"),
            t_start=data.get("t_start"),
            t_end=data.get("t_end"),
        )


@dataclass
class Recipe:
    """An ordered list of filter steps plus a single uniform clipping step."""

    steps: list
    clip: ClipConfig

    def to_dict(self):
        return {"steps": [s.to_dict() for s in self.steps], "clip": self.clip.to_dict()}

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        steps = [RecipeStep.from_dict(s) for s in data.get("steps", [])]
        clip = ClipConfig.from_dict(data.get("clip"))
        return cls(steps=steps, clip=clip)


# ----------------------------------------------------------------------
# Processing service
# ----------------------------------------------------------------------
class CleanDataProcessor:
    """
    Non-destructive, recipe-based signal processing pipeline for a single
    experiment's raw data.

    The original raw file is always re-read from disk and never modified;
    filters append new columns, and the (optional) uniform clipping step is
    always applied last, regardless of its position in the recipe.
    """

    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.clean_data_dir = self.folder_path / CLEAN_DATA_FOLDER_NAME

    @property
    def clean_data_path(self):
        """Default target path for the saved CleanData container."""
        return self.clean_data_dir / f"{self.folder_path.name}{CLEAN_DATA_SUFFIX}"

    # ------------------------------------------------------------------
    # Raw data loading
    # ------------------------------------------------------------------
    def find_raw_data_file(self):
        """Finds the first raw data file directly inside the experiment folder."""
        try:
            candidates = sorted(
                p for p in self.folder_path.iterdir()
                if p.is_file() and p.suffix.lower() in RAW_DATA_EXTENSIONS
            )
        except OSError as exc:
            raise RecipeError(f"Could not read experiment folder {self.folder_path}: {exc}")

        if not candidates:
            raise RecipeError(f"No raw data file found in {self.folder_path}")
        return candidates[0]

    def load_raw_data(self):
        """Loads the untouched, original raw data as a pandas DataFrame."""
        raw_file = self.find_raw_data_file()
        suffix = raw_file.suffix.lower()

        if suffix == ".pkl":
            df = pd.read_pickle(raw_file)
        elif suffix == ".csv":
            df = pd.read_csv(raw_file)
        elif suffix == ".xlsx":
            df = pd.read_excel(raw_file)
        elif suffix == ".tdms":
            df = self._load_tdms(raw_file)
        else:  # pragma: no cover - guarded by RAW_DATA_EXTENSIONS
            raise RecipeError(f"Unsupported raw data extension: {suffix}")

        if not isinstance(df, pd.DataFrame):
            raise RecipeError(f"Raw data file did not yield a DataFrame: {raw_file}")

        return df

    @staticmethod
    def _load_tdms(path):
        try:
            from nptdms import TdmsFile
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RecipeError("The 'nptdms' package is required to read .tdms files.") from exc

        tdms_file = TdmsFile.read(str(path))
        target_channel = None
        for group in tdms_file.groups():
            for channel in group.channels():
                if channel.name == 'Input 0':
                    target_channel = channel
                    break
            if target_channel:
                break

        if not target_channel:
            raise ValueError('TDMS file does not contain "Input 0" channel')

        data = target_channel[:]
        dt = target_channel.properties.get('wf_increment')
        if dt is None:
            fs = target_channel.properties.get('sampling_rate', 1000.0)
            dt = 1.0 / fs

        length = len(data)
        time_s = np.arange(length) * dt
        df = pd.DataFrame({'Input 0': data, 'Time (s)': time_s})
        return df

    @staticmethod
    def numeric_columns(df):
        """Returns the names of all numerical columns in `df`."""
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    @staticmethod
    def guess_time_column(df):
        """Best-effort guess of which column represents time."""
        for c in df.columns:
            if str(c).strip().lower() in ("time", "t", "time (s)", "time(s)", "timestamp"):
                return c
        numeric = CleanDataProcessor.numeric_columns(df)
        return numeric[0] if numeric else None

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_lowpass(series, params):
        fs = float(params.get("fs") or 1000.0)
        cutoff = float(params.get("cutoff") or 10.0)
        order = int(params.get("order") or 4)

        nyquist = 0.5 * fs
        normalized_cutoff = min(max(cutoff / nyquist, 1e-6), 0.999999)
        b, a = butter(order, normalized_cutoff, btype="low", analog=False)
        return filtfilt(b, a, series.to_numpy(dtype=float))

    @staticmethod
    def _apply_median(series, params):
        kernel_size = int(params.get("kernel_size") or 5)
        if kernel_size % 2 == 0:
            kernel_size += 1  # scipy.signal.medfilt requires an odd kernel size
        return medfilt(series.to_numpy(dtype=float), kernel_size=kernel_size)

    @staticmethod
    def _apply_notch(series, params):
        fs = float(params.get("fs") or 1000.0)
        freq = float(params.get("freq") or 50.0)
        quality = float(params.get("quality") or 30.0)

        b, a = iirnotch(freq, quality, fs)
        return filtfilt(b, a, series.to_numpy(dtype=float))

    _STEP_HANDLERS = {
        STEP_LOWPASS: "_apply_lowpass",
        STEP_MEDIAN: "_apply_median",
        STEP_NOTCH: "_apply_notch",
    }

    def apply_step(self, df, step):
        """Applies a single RecipeStep to `df` in-place, adding its output column."""
        if step.step_type not in SUPPORTED_STEP_TYPES:
            raise RecipeError(f"Unsupported step type: '{step.step_type}'")
        if step.input_column not in df.columns:
            raise RecipeError(f"Input column '{step.input_column}' not found in raw data.")

        handler = getattr(self, self._STEP_HANDLERS[step.step_type])
        df[step.output_column] = handler(df[step.input_column], step.params)
        return step.output_column

    def apply_recipe_filters(self, raw_df, recipe):
        """
        Applies all filter steps in order on a copy of `raw_df`, leaving the
        original raw columns untouched. Steps that fail to apply are skipped
        (with a logged warning) instead of aborting the whole recipe.

        Returns:
            tuple[pandas.DataFrame, list[str]]: The augmented DataFrame and
            any warning messages collected along the way.
        """
        df = raw_df.copy()
        warnings_ = []
        for step in recipe.steps:
            try:
                self.apply_step(df, step)
            except RecipeError as exc:
                logger.warning(str(exc))
                warnings_.append(str(exc))
        return df, warnings_

    @staticmethod
    def apply_clip(df, clip):
        """
        Uniformly slices `df` to the `[t_start, t_end]` window of `clip.time_column`.
        Always meant to run after all filter steps. No-op if clip is unset.
        """
        if clip is None or not clip.time_column or clip.time_column not in df.columns:
            return df
        if clip.t_start is None and clip.t_end is None:
            return df

        time_values = df[clip.time_column]
        mask = pd.Series(True, index=df.index)
        if clip.t_start is not None:
            mask &= time_values >= clip.t_start
        if clip.t_end is not None:
            mask &= time_values <= clip.t_end

        return df.loc[mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # High-level pipeline entry points
    # ------------------------------------------------------------------
    def preview(self, recipe):
        """
        Loads fresh raw data and applies only the filter steps (no clipping),
        so the caller can render the full signal and overlay the configured
        clip region for visual inspection before committing to it.

        Returns:
            tuple[pandas.DataFrame, list[str], list[str]]: The filtered
            DataFrame, the original raw column names, and any warnings.
        """
        raw_df = self.load_raw_data()
        filtered_df, warnings_ = self.apply_recipe_filters(raw_df, recipe)
        return filtered_df, list(raw_df.columns), warnings_

    def run(self, recipe):
        """
        Loads fresh raw data, applies all filter steps, then applies the
        uniform clip strictly last.

        Returns:
            tuple[pandas.DataFrame, list[str], list[str]]: The final
            processed DataFrame, the original raw column names, and any
            warnings.
        """
        raw_df = self.load_raw_data()
        filtered_df, warnings_ = self.apply_recipe_filters(raw_df, recipe)
        clipped_df = self.apply_clip(filtered_df, recipe.clip)
        return clipped_df, list(raw_df.columns), warnings_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, recipe):
        """
        Executes the recipe end-to-end and writes the result to
        `CleanData/<experiment>_clean.pkl`, alongside the serialized recipe
        metadata, so it can be re-opened for editing later.

        Returns:
            tuple[pathlib.Path, pandas.DataFrame, list[str]]: The saved
            file path, the processed DataFrame, and any warnings.
        """
        processed_df, raw_columns, warnings_ = self.run(recipe)

        self.clean_data_dir.mkdir(parents=True, exist_ok=True)

        container = {
            "data": processed_df,
            "raw_columns": raw_columns,
            "recipe": recipe.to_dict(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

        with open(self.clean_data_path, "wb") as f:
            pickle.dump(container, f)

        return self.clean_data_path, processed_df, warnings_

    def find_existing_clean_file(self):
        """Returns the first existing CleanData file for this experiment, or None."""
        if not self.clean_data_dir.is_dir():
            return None
        try:
            candidates = sorted(
                p for p in self.clean_data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in CLEAN_DATA_EXTENSIONS
            )
        except OSError as exc:
            logger.warning("Could not read CleanData directory %s: %s", self.clean_data_dir, exc)
            return None
        return candidates[0] if candidates else None

    def load_existing(self):
        """
        Deserializes a previously saved CleanData container (recipe +
        processed data), if one exists, for edit mode.

        Returns:
            dict | None: {"path", "data", "raw_columns", "recipe", "saved_at"}
            or None if no CleanData file exists yet.
        """
        path = self.find_existing_clean_file()
        if path is None:
            return None

        if path.suffix.lower() != ".pkl":
            logger.warning("Existing CleanData file %s is not a recipe-aware pickle; ignoring recipe.", path)
            return {
                "path": path,
                "data": pd.read_csv(path),
                "raw_columns": [],
                "recipe": Recipe(steps=[], clip=ClipConfig()),
                "saved_at": None,
            }

        with open(path, "rb") as f:
            container = pickle.load(f)

        return {
            "path": path,
            "data": container.get("data"),
            "raw_columns": container.get("raw_columns", []),
            "recipe": Recipe.from_dict(container.get("recipe")),
            "saved_at": container.get("saved_at"),
        }


# ----------------------------------------------------------------------
# Flask views
# ----------------------------------------------------------------------
def cleaning_filtering_data():
    """
    Render the CleanData editing page for the experiment currently selected
    in the session (set by `experiments_preview.open_clean_data`).

    On GET, loads the untouched raw data (for column discovery and the
    initial chart) and, if a CleanData file already exists, deserializes
    its recipe metadata so the UI can be re-populated for editing.

    Returns:
        str: Rendered HTML template for the cleaning/filtering page, or a
        redirect back to the experiments preview page if no experiment is
        selected or the raw data cannot be read.
    """
    experiment = session.get("current_experiment")
    if not experiment or session.get("editor_mode") != "clean":
        flash("No experiment selected for CleanData. Please pick one from the experiments preview.", "error")
        return redirect(url_for("render_experiments_preview"))

    processor = CleanDataProcessor(experiment["folder_path"])

    try:
        raw_df = processor.load_raw_data()
    except RecipeError as exc:
        flash(str(exc), "error")
        return redirect(url_for("render_experiments_preview"))

    numeric_columns = processor.numeric_columns(raw_df)
    default_time_column = processor.guess_time_column(raw_df)

    existing = None
    try:
        existing = processor.load_existing()
    except Exception:  # pragma: no cover - defensive catch-all
        logger.exception("Could not load existing CleanData for %s", experiment["folder_path"])
        flash("Existing CleanData file could not be read; starting a fresh recipe.", "warning")

    if existing:
        recipe = existing["recipe"]
        chart_df = existing["data"] if isinstance(existing["data"], pd.DataFrame) else raw_df
        if existing["saved_at"]:
            flash(f"Loaded existing CleanData recipe (saved at {existing['saved_at']}).", "success")
    else:
        recipe = Recipe(steps=[], clip=ClipConfig(time_column=default_time_column))
        chart_df = raw_df

    root_dir = session.get("root_dir")
    loads_map = LoadsDescriptionRepository(root_dir).load() if root_dir else {}
    rload_ids = sorted(loads_map.keys())

    # postprocessing_metadata.json (in the experiment's raw-data folder) is
    # the persistent source of truth for RloadId/Gain; resolve it here so
    # the CleanData panel always reflects the same value as the
    # experiments table, regardless of session state.
    _resolve_postprocessing_metadata(experiment, experiment["folder_path"], loads_map)
    session["current_experiment"] = experiment

    experiments_list = session.get("experiments") or []
    exp_id = experiment.get("experiment_id")
    if isinstance(exp_id, int) and 0 <= exp_id < len(experiments_list):
        stored = dict(experiments_list[exp_id])
        stored["RloadId"] = experiment.get("RloadId")
        stored["Gain"] = experiment.get("Gain")
        experiments_list[exp_id] = stored
        session["experiments"] = experiments_list
    session.modified = True

    rload_key = str(experiment.get("RloadId", "")).strip()
    rload_valid = rload_key in loads_map
    default_gain = loads_map.get(rload_key)
    default_rload_id = experiment.get("_original_rload_id", experiment.get("RloadId"))

    rload_context = {
        "RloadId": experiment.get("RloadId"),
        "Gain": experiment.get("Gain"),
        "rload_valid": rload_valid,
        "default_gain": default_gain,
        "default_rload_id": default_rload_id,
    }

    return render_template(
        "cleaning_filtering_data.html",
        experiment=experiment,
        numeric_columns_json=json.dumps(numeric_columns),
        default_time_column_json=json.dumps(default_time_column),
        recipe_json=json.dumps(recipe.to_dict()),
        chart_data_json=json.dumps(_dataframe_to_chart_json(chart_df)),
        has_existing=existing is not None,
        rload_ids=rload_ids,
        loads_map_json=json.dumps(loads_map),
        rload_context_json=json.dumps(rload_context),
        update_row_url=url_for("render_update_experiment_row", experiment_id=experiment["experiment_id"]),
    )


def preview_recipe():
    """
    Recomputes a candidate recipe's filter steps (without clipping) against
    fresh raw data, so the front-end can update its chart preview.

    Expects a JSON body: {"steps": [...], "clip": {...}}.

    Returns:
        flask.Response: JSON payload with the recomputed chart data, the
        raw/computed column names, and any step warnings; or a JSON error.
    """
    experiment = session.get("current_experiment")
    if not experiment:
        return jsonify({"error": "No experiment selected."}), 400

    try:
        recipe = Recipe.from_dict(request.get_json(force=True, silent=True) or {})
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400

    processor = CleanDataProcessor(experiment["folder_path"])
    try:
        filtered_df, raw_columns, warnings_ = processor.preview(recipe)
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while previewing recipe")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    return jsonify(
        {
            "chart_data": _dataframe_to_chart_json(filtered_df),
            "raw_columns": raw_columns,
            "computed_columns": [c for c in filtered_df.columns if c not in raw_columns],
            "warnings": warnings_,
        }
    )


def save_clean_data():
    """
    Executes the submitted recipe end-to-end (filters, then clipping),
    writes it to `CleanData/<experiment>_clean.pkl` and returns a redirect
    URL back to the experiments preview page for the front-end to navigate to.

    Expects a JSON body: {"steps": [...], "clip": {...}}.

    Returns:
        flask.Response: JSON payload {"message": str, "redirect_url": str}
        on success, or a JSON error otherwise.
    """
    experiment = session.get("current_experiment")
    if not experiment:
        return jsonify({"error": "No experiment selected."}), 400

    try:
        recipe = Recipe.from_dict(request.get_json(force=True, silent=True) or {})
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400

    processor = CleanDataProcessor(experiment["folder_path"])
    try:
        clean_path, processed_df, warnings_ = processor.save(recipe)
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while saving CleanData")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    saved_at = datetime.now().isoformat(timespec="seconds")
    record_postprocessing_metadata(
        experiment["folder_path"],
        "clean_data",
        {
            "recipe": recipe.to_dict(),
            "saved_at": saved_at,
            "n_samples": len(processed_df),
            "warnings": warnings_,
        },
    )

    flash(f"CleanData saved to {clean_path} ({len(processed_df)} sample(s)).", "success")
    for warn in warnings_:
        flash(warn, "warning")

    return jsonify(
        {
            "message": "CleanData saved successfully.",
            "redirect_url": url_for("render_experiments_preview"),
        }
    )


def _dataframe_to_chart_json(df, max_points=MAX_CHART_POINTS):
    """
    Converts a DataFrame's numeric columns into a JSON-friendly dict,
    downsampling uniformly if it has more than `max_points` rows (keeps
    the browser/Plotly responsive for large raw signal files).
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
    }
