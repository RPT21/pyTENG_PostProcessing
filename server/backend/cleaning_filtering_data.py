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
from scipy.signal import butter, filtfilt, iirnotch, medfilt, get_window
from scipy.signal import detrend as scipy_detrend

from server.backend.experiments_preview import (
    LoadsDescriptionRepository,
    resolve_experiment_identity,
    _resolve_postprocessing_metadata,
    record_postprocessing_metadata,
)
from server.utils.experiment_path_resolver import InvalidExperimentFormatError
from server.utils import output_storage

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
MAX_CHART_POINTS = 5000

STEP_LOWPASS = "lowpass"
STEP_MEDIAN = "median"
STEP_NOTCH = "notch"
SUPPORTED_STEP_TYPES = (STEP_LOWPASS, STEP_MEDIAN, STEP_NOTCH)


class RecipeError(Exception):
    """Raised when a recipe (or one of its steps) cannot be validated or executed."""


class FourierError(Exception):
    """Raised when a Fourier Transform request cannot be validated or computed."""


# --- Fourier Transform constants ---
FOURIER_WINDOWS = ("none", "hann", "hamming", "blackman")
FOURIER_DETRENDS = ("none", "mean", "linear")
FOURIER_SCALES = ("amplitude", "db")

# Default parameters for how a Fourier Transform is calculated/displayed.
# `nfft` None = use the signal length (no zero-padding); `f_min`/`f_max`
# None = automatic display range. `scale`, `f_min`, `f_max` and `log_x`
# only affect how the spectrum is displayed, not how it is computed.
FOURIER_DEFAULTS = {
    "detrend": "mean",
    "window": "hann",
    "nfft": None,
    "scale": "amplitude",
    "f_min": None,
    "f_max": None,
    "log_x": False,
    "log_y": False,
    "points_pct": 100.0,
}


def _optional_number(value, minimum=None, integer=False):
    """Coerces `value` to a float/int, or None when empty/invalid/out of range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or (minimum is not None and number < minimum):
        return None
    return int(number) if integer else number


def normalize_fourier_config(config):
    """
    Sanitizes one Fourier parameter dict (the defaults, or a single plot's
    parameters), filling missing/invalid values from FOURIER_DEFAULTS.
    """
    config = config if isinstance(config, dict) else {}
    detrend = config.get("detrend", FOURIER_DEFAULTS["detrend"])
    if detrend is True:
        detrend = "mean"
    elif detrend is False:
        detrend = "none"
    if detrend not in FOURIER_DETRENDS:
        detrend = FOURIER_DEFAULTS["detrend"]

    window = config.get("window", FOURIER_DEFAULTS["window"])
    if window not in FOURIER_WINDOWS:
        window = FOURIER_DEFAULTS["window"]

    scale = config.get("scale", FOURIER_DEFAULTS["scale"])
    if scale not in FOURIER_SCALES:
        scale = FOURIER_DEFAULTS["scale"]

    return {
        "detrend": detrend,
        "window": window,
        "nfft": _optional_number(config.get("nfft"), minimum=2, integer=True),
        "scale": scale,
        "f_min": _optional_number(config.get("f_min"), minimum=0),
        "f_max": _optional_number(config.get("f_max"), minimum=0),
        "log_x": bool(config.get("log_x", FOURIER_DEFAULTS["log_x"])),
        "log_y": bool(config.get("log_y", FOURIER_DEFAULTS["log_y"])),
        "points_pct": min(_optional_number(config.get("points_pct"), minimum=0.01) or 100.0, 100.0),
    }


def build_fourier_state(fourier, df, time_column):
    """
    Builds the persistable Fourier state from the client-submitted
    configuration: sanitized defaults plus every plot's parameters and
    series, each series carrying its spectrum computed here against `df`
    (the final, saved data) so the stored results always match the file.

    Returns:
        tuple[dict, list[str]]: (state, warnings).
    """
    fourier = fourier if isinstance(fourier, dict) else {}
    warnings_ = []
    plots = []
    for index, plot in enumerate(fourier.get("plots") or [], start=1):
        if not isinstance(plot, dict):
            continue
        config = normalize_fourier_config(plot)
        series_out = []
        for series in plot.get("series") or []:
            column = series.get("signal_column") if isinstance(series, dict) else None
            if not column:
                continue
            try:
                frequencies, magnitude = _compute_fft(
                    df, column, time_column,
                    detrend=config["detrend"], window=config["window"], nfft=config["nfft"], points_pct=config["points_pct"],
                )
            except FourierError as exc:
                warnings_.append(f"Fourier plot {index}: {exc}")
                frequencies, magnitude = [], []
            series_out.append(
                {"signal_column": column, "frequencies": frequencies, "magnitude": magnitude}
            )
        plots.append({**config, "series": series_out})

    state = {
        "defaults": normalize_fourier_config(fourier.get("defaults")),
        "time_column": time_column,
        "plots": plots,
    }
    return state, warnings_


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
class GainStep:
    """
    Scales one raw signal column before any filter step runs:

        column *= [1 / load_gain, if use_load_gain] * [conversion_factor, if use_conversion]

    `load_gain` is the voltage-divider Gain of the Load Configuration (a
    snapshot taken by the page). The Keithley `conversion_factor` is looked
    up server-side, by `conversion_channel`, in the experiment's
    `experiment_metadata.json`.
    """

    column: str
    use_load_gain: bool = False
    load_gain: Optional[float] = None
    use_conversion: bool = False
    conversion_channel: Optional[str] = None
    gain_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        if not data.get("column"):
            raise RecipeError(f"Malformed gain entry: {data}")
        return cls(
            column=data["column"],
            use_load_gain=bool(data.get("use_load_gain")),
            load_gain=_optional_number(data.get("load_gain")),
            use_conversion=bool(data.get("use_conversion")),
            conversion_channel=data.get("conversion_channel") or None,
            gain_id=data.get("gain_id"),
        )


@dataclass
class Recipe:
    """Per-signal gains, an ordered list of filter steps, and a single uniform clipping step."""

    steps: list
    clip: ClipConfig
    gains: list = field(default_factory=list)

    def to_dict(self):
        return {
            "gains": [g.to_dict() for g in self.gains],
            "steps": [s.to_dict() for s in self.steps],
            "clip": self.clip.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        gains = [GainStep.from_dict(g) for g in data.get("gains") or []]
        steps = [RecipeStep.from_dict(s) for s in data.get("steps", [])]
        clip = ClipConfig.from_dict(data.get("clip"))
        return cls(steps=steps, clip=clip, gains=gains)


# ----------------------------------------------------------------------
# Keithley conversion factors (experiment_metadata.json)
# ----------------------------------------------------------------------
KEITHLEY_METADATA_FILENAME = "experiment_metadata.json"


def read_keithley_channels(experiment_folder):
    """
    Reads `<experiment_folder>/experiment_metadata.json` and returns the
    conversion factor of every DAQ channel found in its `DAQTasks`.

    A factor that is missing, null, zero or not a number counts as "no
    conversion" (None), exactly like the legacy `if conversion_factor:` check.
    The file being absent or malformed is reported in `error`, never raised.

    Returns:
        dict: {"found": bool, "path": str | None, "error": str | None,
        "channels": [{"channel": str, "conversion_factor": float | None}]}
    """
    info = {"found": False, "path": None, "error": None, "channels": []}
    if not experiment_folder:
        info["error"] = "No experiment folder available."
        return info

    folder = Path(experiment_folder)
    if folder.is_file():
        folder = folder.parent
    json_path = folder / KEITHLEY_METADATA_FILENAME
    info["path"] = str(json_path)
    if not json_path.is_file():
        info["error"] = f"{KEITHLEY_METADATA_FILENAME} not found in the experiment folder."
        return info

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        tasks = metadata.get("DAQTasks") or []
    except (OSError, ValueError, AttributeError) as exc:
        info["error"] = f"Could not read {KEITHLEY_METADATA_FILENAME}: {exc}"
        return info

    factors = {}
    for task in tasks if isinstance(tasks, list) else []:
        channels = task.get("DAQ_CHANNELS") if isinstance(task, dict) else None
        if not isinstance(channels, dict):
            continue
        for name, channel in channels.items():
            factor = _optional_number(channel.get("conversion_factor")) if isinstance(channel, dict) else None
            if not factor:
                factor = None
            # A channel repeated in another task keeps its first real factor.
            if name not in factors or factors[name] is None:
                factors[name] = factor

    info["found"] = True
    info["channels"] = [{"channel": n, "conversion_factor": f} for n, f in factors.items()]
    return info


def resolve_gains(gains, keithley_info):
    """
    Resolves every GainStep into its numeric calculation.

    Returns:
        tuple[list[dict], list[str]]: One dict per entry
        {"column", "load_gain", "conversion_channel", "conversion_factor",
        "factor"} (load_gain / conversion_factor are None when not used),
        and any warnings.
    """
    factors = {c["channel"]: c["conversion_factor"] for c in keithley_info.get("channels", [])}
    resolved, warnings_ = [], []
    for gain in gains:
        load_gain = gain.load_gain if gain.use_load_gain and gain.load_gain else None
        if gain.use_load_gain and not load_gain:
            warnings_.append(f"Gain for '{gain.column}': no load Gain is set, so it was not applied.")

        conversion = None
        if gain.use_conversion:
            if gain.conversion_channel not in factors:
                warnings_.append(
                    f"Gain for '{gain.column}': Keithley channel '{gain.conversion_channel}' "
                    f"was not found in {KEITHLEY_METADATA_FILENAME}."
                )
            elif not factors[gain.conversion_channel]:
                warnings_.append(
                    f"Gain for '{gain.column}': Keithley channel '{gain.conversion_channel}' "
                    "has no conversion factor."
                )
            else:
                conversion = factors[gain.conversion_channel]

        factor = 1.0
        if load_gain:
            factor /= load_gain
        if conversion:
            factor *= conversion
        resolved.append({
            "column": gain.column,
            "load_gain": load_gain,
            "conversion_channel": gain.conversion_channel if gain.use_conversion else None,
            "conversion_factor": conversion,
            "factor": factor,
        })
    return resolved, warnings_


# ----------------------------------------------------------------------
# Processing service
# ----------------------------------------------------------------------
class CleanDataProcessor:
    """
    Non-destructive, recipe-based signal processing pipeline for a single
    experiment's raw data.

    The raw data itself always comes from the Phase 2/3 file-selection step
    (server/backend/file_selection.py): the file(s) the user picked there,
    merged onto a common time axis if more than one was selected, are
    cached to disk (see server.utils.output_storage.merged_cache_path) and
    re-read fresh here on every recipe run/preview - never modified in
    place. The (optional) uniform clipping step is always applied last,
    regardless of its position in the recipe.

    CleanData output is written to the centralized `CleanData/` directory
    under the experiments root_dir (Phase 5), named deterministically from
    the experiment's identity (see output_storage.py) so it never collides
    with another experiment's output, even though they now share a
    directory.
    """

    def __init__(self, root_dir, identity, raw_data_path, experiment_folder=None):
        self.root_dir = root_dir
        self.identity = identity
        self.raw_data_path = Path(raw_data_path)
        self.experiment_folder = experiment_folder

    @property
    def clean_data_path(self):
        """Deterministic target path for the saved CleanData container."""
        return output_storage.clean_data_path(self.root_dir, self.identity)

    # ------------------------------------------------------------------
    # Raw data loading
    # ------------------------------------------------------------------
    def load_raw_data(self):
        """Loads the Phase 3 merged/selected raw data cache as a pandas DataFrame."""
        if not self.raw_data_path.is_file():
            raise RecipeError(
                f"Raw data cache not found: {self.raw_data_path}. "
                "Please redo the file selection step."
            )

        df = pd.read_pickle(self.raw_data_path)
        if not isinstance(df, pd.DataFrame):
            raise RecipeError(f"Cached raw data is not a DataFrame: {self.raw_data_path}")
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

    def keithley_channels(self):
        """Keithley channels/conversion factors of this experiment's metadata JSON."""
        return read_keithley_channels(self.experiment_folder)

    def apply_gains(self, df, gains):
        """
        Scales the gain-configured columns of `df` in place (before any
        filter step), merging the legacy `apply_gain_to_dataframe` logic:
        voltage-divider load gain (divide) and Keithley conversion factor
        (multiply), each optional and per signal.

        Returns:
            tuple[list[dict], list[str]]: The resolved gains (see
            `resolve_gains`) that were applied, and any warnings.
        """
        if not gains:
            return [], []

        resolved, warnings_ = resolve_gains(gains, self.keithley_channels())
        applied = []
        for entry in resolved:
            column = entry["column"]
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                warnings_.append(f"Gain for '{column}': column not found or not numeric, skipped.")
                continue
            if entry["factor"] != 1.0:
                df[column] = df[column].astype(float) * entry["factor"]
            applied.append(entry)
        return applied, warnings_

    def apply_recipe_filters(self, raw_df, recipe):
        """
        Applies the per-signal gains, then all filter steps in order, on a
        copy of `raw_df`, leaving the original raw data untouched. Steps
        that fail to apply are skipped (with a logged warning) instead of
        aborting the whole recipe.

        Returns:
            tuple[pandas.DataFrame, list[str]]: The augmented DataFrame and
            any warning messages collected along the way.
        """
        df = raw_df.copy()
        _applied, warnings_ = self.apply_gains(df, recipe.gains)
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
    def save(self, recipe, fourier=None):
        """
        Executes the recipe end-to-end and writes the result to
        `CleanData/<experiment>_clean.pkl`, alongside the serialized recipe
        metadata and the Fourier plots (configuration + computed spectra),
        so it can be re-opened for editing later.

        Returns:
            tuple[pathlib.Path, pandas.DataFrame, list[str]]: The saved
            file path, the processed DataFrame, and any warnings.
        """
        processed_df, raw_columns, warnings_ = self.run(recipe)
        fourier_state, fourier_warnings = build_fourier_state(
            fourier, processed_df, recipe.clip.time_column
        )
        warnings_ = list(warnings_) + fourier_warnings
        gains_applied, _gain_warnings = resolve_gains(recipe.gains, self.keithley_channels())

        self.clean_data_path.parent.mkdir(parents=True, exist_ok=True)

        container = {
            "data": processed_df,
            "raw_columns": raw_columns,
            "recipe": recipe.to_dict(),
            "gains_applied": gains_applied,
            "fourier": fourier_state,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

        with open(self.clean_data_path, "wb") as f:
            pickle.dump(container, f)

        return self.clean_data_path, processed_df, warnings_

    def find_existing_clean_file(self):
        """Returns this experiment's CleanData file if it already exists, or None.

        Since the output filename is now deterministic (derived from the
        experiment's identity, see output_storage.py), no directory
        scanning is needed even though CleanData/ is shared by every
        experiment.
        """
        return self.clean_data_path if self.clean_data_path.is_file() else None

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

        with open(path, "rb") as f:
            container = pickle.load(f)

        return {
            "path": path,
            "data": container.get("data"),
            "raw_columns": container.get("raw_columns", []),
            "recipe": Recipe.from_dict(container.get("recipe")),
            "fourier": container.get("fourier"),
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

    root_dir = session.get("root_dir")
    merged_data_path = session.get("merged_data_path")
    if not merged_data_path or not root_dir:
        flash("Select and load raw data file(s) before editing CleanData.", "error")
        return redirect(url_for("render_file_selection"))

    try:
        identity = resolve_experiment_identity(experiment)
    except InvalidExperimentFormatError as exc:
        flash(str(exc), "error")
        return redirect(url_for("render_experiments_preview"))

    processor = CleanDataProcessor(root_dir, identity, merged_data_path, experiment.get("folder_path"))

    try:
        raw_df = processor.load_raw_data()
    except RecipeError as exc:
        flash(str(exc), "error")
        return redirect(url_for("render_file_selection"))

    numeric_columns = processor.numeric_columns(raw_df)
    default_time_column = processor.guess_time_column(raw_df)

    existing = None
    try:
        existing = processor.load_existing()
    except Exception:  # pragma: no cover - defensive catch-all
        logger.exception("Could not load existing CleanData for %s", identity)
        flash("Existing CleanData file could not be read; starting a fresh recipe.", "warning")

    if existing:
        recipe = existing["recipe"]
        fourier_state = existing.get("fourier")
        chart_df = existing["data"] if isinstance(existing["data"], pd.DataFrame) else raw_df
        if existing["saved_at"]:
            flash(f"Loaded existing CleanData recipe (saved at {existing['saved_at']}).", "success")
    else:
        fourier_state = None
        recipe = Recipe(steps=[], clip=ClipConfig(time_column=default_time_column))
        chart_df = raw_df

    loads_map = LoadsDescriptionRepository(root_dir).load() if root_dir else {}
    rload_ids = sorted(loads_map.keys())

    # The centralized postprocessing metadata file is the persistent
    # source of truth for RloadId/Gain; resolve it here so the CleanData
    # panel always reflects the same value as the experiments table,
    # regardless of session state.
    _resolve_postprocessing_metadata(experiment, root_dir, identity, loads_map)
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
        keithley_json=json.dumps(processor.keithley_channels()),
        fourier_json=json.dumps(_fourier_for_client(fourier_state)),
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
    root_dir = session.get("root_dir")
    merged_data_path = session.get("merged_data_path")
    if not experiment or not merged_data_path or not root_dir:
        return jsonify({"error": "No experiment/raw data selected."}), 400

    try:
        recipe = Recipe.from_dict(request.get_json(force=True, silent=True) or {})
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        identity = resolve_experiment_identity(experiment)
    except InvalidExperimentFormatError as exc:
        return jsonify({"error": str(exc)}), 400

    processor = CleanDataProcessor(root_dir, identity, merged_data_path, experiment.get("folder_path"))
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


def _fourier_for_client(state):
    """Returns the saved Fourier configuration (defaults + plots/series
    selections) without the bulky stored spectra; the page recomputes them."""
    defaults = normalize_fourier_config((state or {}).get("defaults"))
    plots = []
    for plot in (state or {}).get("plots") or []:
        plots.append({
            **normalize_fourier_config(plot),
            "series": [{"signal_column": s.get("signal_column")} for s in plot.get("series") or []],
        })
    return {"defaults": defaults, "plots": plots}


def preview_fourier():
    """
    Computes the Fourier Transform (magnitude spectrum) of a single signal
    column, against the FULL-resolution recipe-filtered data (never the
    downsampled chart preview - see _dataframe_to_chart_json), so the
    independent "Fourier Transform" plots stay numerically correct
    regardless of how large the raw signal is.

    Expects a JSON body: {"recipe": {...}, "signal_column": str,
    "time_column": str | None, "detrend": bool, "window": str}.
    `recipe` mirrors the same payload sent to /preview (current, possibly
    unsaved, filter steps), so a Fourier plot can compare the FFT of a raw
    signal against one of its filtered variants without requiring the
    recipe to be saved first.

    Returns:
        flask.Response: JSON payload {"frequencies", "magnitude",
        "signal_column", "warnings"}, or a JSON error.
    """
    experiment = session.get("current_experiment")
    root_dir = session.get("root_dir")
    merged_data_path = session.get("merged_data_path")
    if not experiment or not merged_data_path or not root_dir:
        return jsonify({"error": "No experiment/raw data selected."}), 400

    body = request.get_json(force=True, silent=True) or {}

    try:
        recipe = Recipe.from_dict(body.get("recipe") or {})
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400

    signal_column = body.get("signal_column")
    if not signal_column:
        return jsonify({"error": "signal_column is required."}), 400
    time_column = body.get("time_column") or None
    config = normalize_fourier_config(body)

    try:
        identity = resolve_experiment_identity(experiment)
    except InvalidExperimentFormatError as exc:
        return jsonify({"error": str(exc)}), 400

    processor = CleanDataProcessor(root_dir, identity, merged_data_path, experiment.get("folder_path"))
    try:
        filtered_df, _raw_columns, warnings_ = processor.preview(recipe)
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while previewing recipe for Fourier Transform")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    try:
        frequencies, magnitude = _compute_fft(
            filtered_df, signal_column, time_column,
            detrend=config["detrend"], window=config["window"], nfft=config["nfft"], points_pct=config["points_pct"],
        )
    except FourierError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while computing Fourier Transform")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    return jsonify(
        {
            "frequencies": frequencies,
            "magnitude": magnitude,
            "signal_column": signal_column,
            "warnings": warnings_,
        }
    )


def save_clean_data():
    """
    Executes the submitted recipe end-to-end (filters, then clipping),
    writes it to the centralized `CleanData/` directory (Phase 5) and
    returns a redirect URL back to the experiments preview page for the
    front-end to navigate to.

    Expects a JSON body: {"steps": [...], "clip": {...}, "fourier": {...}},
    where the optional "fourier" entry holds the Fourier plots
    configuration (defaults + plots) to persist with the CleanData file.

    Returns:
        flask.Response: JSON payload {"message": str, "redirect_url": str}
        on success, or a JSON error otherwise.
    """
    experiment = session.get("current_experiment")
    root_dir = session.get("root_dir")
    merged_data_path = session.get("merged_data_path")
    if not experiment or not merged_data_path or not root_dir:
        return jsonify({"error": "No experiment/raw data selected."}), 400

    body = request.get_json(force=True, silent=True) or {}
    fourier = body.get("fourier")

    try:
        recipe = Recipe.from_dict(body)
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        identity = resolve_experiment_identity(experiment)
    except InvalidExperimentFormatError as exc:
        return jsonify({"error": str(exc)}), 400

    processor = CleanDataProcessor(root_dir, identity, merged_data_path, experiment.get("folder_path"))
    try:
        clean_path, processed_df, warnings_ = processor.save(recipe, fourier)
    except RecipeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while saving CleanData")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    saved_at = datetime.now().isoformat(timespec="seconds")
    record_postprocessing_metadata(
        root_dir,
        identity,
        "clean_data",
        {
            "recipe": recipe.to_dict(),
            "gains_applied": resolve_gains(recipe.gains, processor.keithley_channels())[0],
            "fourier": _fourier_for_client(fourier) if isinstance(fourier, dict) else None,
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


def _compute_fft(df, signal_column, time_column=None, detrend="mean", window="none", nfft=None,
                 points_pct=100.0):
    """
    Computes the one-sided amplitude spectrum of `df[signal_column]` via
    `numpy.fft.rfft`, always against the FULL-resolution column (callers
    must never pass an already-downsampled chart-preview array - see
    `_dataframe_to_chart_json`'s docstring for why that would corrupt the
    frequency axis).

    Args:
        df: The (recipe-filtered, full-resolution) DataFrame.
        signal_column: Column to transform.
        time_column: Optional time column used to derive the sampling
            rate (`fs = 1 / median(diff(time))`, robust to minor jitter).
            When None/missing, falls back to a unit sample spacing and the
            returned "frequencies" are in cycles/sample instead of Hz.
        detrend: "mean" (subtract the mean, removing the huge DC spike a
            non-zero-mean signal would otherwise produce), "linear"
            (remove a least-squares line) or "none". Booleans are accepted
            for backwards compatibility (True = "mean").
        nfft: Optional FFT length; values above the signal length
            zero-pad (finer frequency grid), None uses the signal length.
        window: One of FOURIER_WINDOWS ("none"/"hann"/"hamming"/"blackman"),
            applied to reduce spectral leakage from the signal not being
            perfectly periodic within the sampled window.
        points_pct: Percentage (0-100] of the computed bins returned, by
            uniform stride (100 = every bin). Frequency resolution is
            unaffected - only how many bins are sent to the browser.

    Returns:
        tuple[list[float], list[float]]: (frequencies, magnitude).
    """
    if signal_column not in df.columns:
        raise FourierError(f"Signal column '{signal_column}' not found.")
    if window not in FOURIER_WINDOWS:
        raise FourierError(f"Unsupported window '{window}'. Expected one of {FOURIER_WINDOWS}.")

    values = pd.to_numeric(df[signal_column], errors="coerce").to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        raise FourierError(f"Signal column '{signal_column}' has too few valid samples for a Fourier Transform.")

    if time_column and time_column in df.columns:
        time_values = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
        time_values = time_values[~np.isnan(time_values)]
        if len(time_values) >= 2:
            dt = float(np.median(np.diff(time_values)))
            if dt <= 0 or not np.isfinite(dt):
                raise FourierError(f"Time column '{time_column}' does not have a valid, increasing sample spacing.")
        else:
            dt = 1.0
    else:
        dt = 1.0

    if detrend is True:
        detrend = "mean"
    elif detrend is False or detrend is None:
        detrend = "none"
    if detrend not in FOURIER_DETRENDS:
        raise FourierError(f"Unsupported detrend '{detrend}'. Expected one of {FOURIER_DETRENDS}.")

    if detrend == "mean":
        values = values - values.mean()
    elif detrend == "linear":
        values = scipy_detrend(values, type="linear")

    # Coherent gain correction keeps sine amplitudes accurate when windowing.
    gain = 1.0
    if window != "none":
        win = get_window(window, n)
        values = values * win
        gain = float(win.mean())

    n_fft = max(int(nfft), n) if nfft else n
    spectrum = np.fft.rfft(values, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, d=dt)

    # Convert to a single-sided amplitude spectrum: double every bin except
    # DC (and Nyquist, for even-length transforms), which do not have a
    # mirrored negative-frequency counterpart to combine with. Normalizing
    # by the original sample count keeps amplitudes correct under zero-padding.
    magnitude = np.abs(spectrum) * (2.0 / (n * gain))
    magnitude[0] /= 2.0
    if n_fft % 2 == 0:
        magnitude[-1] /= 2.0

    step = max(1, int(round(100.0 / points_pct))) if points_pct < 100 else 1
    return frequencies[::step].tolist(), magnitude[::step].tolist()

