import json
import logging
from pathlib import Path

import pandas as pd
from flask import render_template, request, redirect, url_for, session, flash, jsonify

from server.backend.data_loading import METADATA_LOADS_FILENAME
from server.utils.experiment_path_resolver import ExperimentPathResolver, InvalidExperimentFormatError
from server.utils import output_storage

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
# Top-level key under which CleanData/CycleData run metadata (recipe used,
# timestamp, warnings, ...) is namespaced within each experiment's
# centralized postprocessing metadata file, kept separate from the
# RloadId/Gain load configuration also stored there.
POSTPROCESSING_KEY = "postprocessing"

# Signals/metrics offered in the plot configuration box. In the absence of a
# per-experiment signal catalogue, a fixed, generic set covering the typical
# TENG post-processing quantities is used; this can later be replaced by a
# dynamic list inspected from each experiment's CleanData/CycleData files.
AVAILABLE_PLOT_VARIABLES = [
    "Time",
    "Voltage",
    "Current",
    "Charge",
    "Cycle",
    "PeakAmplitude",
    "Rload",
]
AVAILABLE_PLOT_TYPES = [
    "Time Series",             # Type A: multi-trace signal vs time, aligned across experiments
    "Cycle Evolution",         # Type B: metric vs cycle index (fatigue/degradation trends)
    "Load Curve",              # Type C: aggregated metric vs RloadId, grouped by TribuId
    "Cycle Waveform Overlay",  # Type D: individual cycles overlaid for a single experiment
]


def _resolution_row(experiment):
    """
    Builds the row used for Phase 1 path resolution / identity building,
    preferring `_original_rload_id` (the RloadId as originally scanned from
    disk) over a possibly user-edited `RloadId`, so in-table RloadId
    corrections (which only affect Gain/metadata bookkeeping) never change
    which raw-data folder is read, nor which CleanData/CycleData output
    file is looked up.
    """
    row = dict(experiment)
    original = experiment.get("_original_rload_id")
    if original not in (None, ""):
        row["RloadId"] = original
    return row


def resolve_experiment_identity(experiment):
    """Builds an experiment's naming/output identity (Phase 1 + Phase 5 key)."""
    return ExperimentPathResolver.build_identity(_resolution_row(experiment))


class ExperimentFileStatus:
    """
    Resolves the on-disk raw-data folder(s) for each experiment row (Phase 1,
    via ExperimentPathResolver - supports both the two-folder "daq"/"motor"
    layout and the single-folder TribuId/Pos-Neg/Date-RloadId layout) and
    inspects whether CleanData/CycleData outputs already exist for it in the
    centralized output directories under the experiments root_dir (Phase 5,
    via output_storage).
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.resolver = ExperimentPathResolver(root_dir)

    def resolve_location(self, experiment):
        """Resolves the ExperimentLocation (Phase 1) for a single experiment row."""
        return self.resolver.resolve(_resolution_row(experiment))

    def folder_path(self, experiment):
        """
        Best-effort, display-only description of the experiment's raw-data
        location(s): the single resolved folder for single-folder
        (Scenario B) experiments, or both resolved folders (labeled by
        role) for two-folder (Scenario A) experiments, since there is no
        single folder to point to in that case.

        Returns:
            str | None: The description, or None if the row's path could
            not be resolved at all (see InvalidExperimentFormatError).
        """
        try:
            location = self.resolve_location(experiment)
        except InvalidExperimentFormatError:
            return None

        if "data" in location.paths:
            return str(location.paths["data"])
        return " | ".join(f"{role}: {path}" for role, path in location.paths.items())

    def inspect(self, experiment):
        """
        Returns a dict describing the on-disk status of a single experiment:
            {
                "folder_path": str | None,
                "clean_data_exists": bool,
                "clean_data_path": str | None,
                "cycle_data_exists": bool,
                "cycle_data_path": str | None,
                "path_error": str | None,
            }
        `path_error` is set (and every other field left at its default)
        when the row matches neither the two-folder nor the single-folder
        experiment layout (Phase 1 exception handling).
        """
        result = {
            "folder_path": None,
            "clean_data_exists": False,
            "clean_data_path": None,
            "cycle_data_exists": False,
            "cycle_data_path": None,
            "path_error": None,
        }

        try:
            location = self.resolve_location(experiment)
        except InvalidExperimentFormatError as exc:
            result["path_error"] = str(exc)
            return result

        result["folder_path"] = self.folder_path(experiment)

        clean_path = output_storage.clean_data_path(self.root_dir, location.identity)
        if clean_path.is_file():
            result["clean_data_exists"] = True
            result["clean_data_path"] = str(clean_path)

        cycle_path = output_storage.cycle_data_path(self.root_dir, location.identity)
        if cycle_path.is_file():
            result["cycle_data_exists"] = True
            result["cycle_data_path"] = str(cycle_path)

        return result

    def augment(self, experiments):
        """
        Augments a list of experiment dict rows with a positional
        `experiment_id` and the file-status information above.

        Args:
            experiments (list[dict]): Experiment records.

        Returns:
            list[dict]: The same records, each extended with
            `experiment_id`, `folder_path`, `clean_data_exists`,
            `clean_data_path`, `cycle_data_exists`, `cycle_data_path` and
            `path_error`.
        """
        augmented = []
        for idx, experiment in enumerate(experiments):
            row = dict(experiment)
            row["experiment_id"] = idx
            row.update(self.inspect(experiment))
            augmented.append(row)
        return augmented


class PostprocessingMetadataStore:
    """
    Reads/writes a per-experiment postprocessing metadata JSON file. This
    file is owned entirely by this tool and is the single, persistent,
    on-disk source of truth for postprocessing-level configuration
    (currently `RloadId`/`Gain`, plus a `postprocessing` section with
    CleanData/CycleData run bookkeeping).

    Centralized under `output_storage.metadata_dir(root_dir)` (the
    experiments root_dir) instead of being written inside the experiment's
    raw-data folder(s): this keeps raw data folders untouched (Phase 5) and
    works uniformly for both single-folder and two-folder (Scenario A)
    experiments, which have no single natural folder to hold it. It is
    independent of the Flask session (session values are only a
    short-lived cache of the same data, used for fast rendering between
    requests).
    """

    def __init__(self, root_dir, identity):
        self.root_dir = root_dir
        self.identity = identity
        self.path = output_storage.metadata_path(root_dir, identity)

    def load(self):
        """
        Returns:
            dict | None: The deserialized metadata, or None if the file
            does not exist or could not be parsed (corrupted JSON,
            unreadable, or not a JSON object).
        """
        if not self.path.is_file():
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read postprocessing metadata '%s': %s", self.path, exc)
            return None

        if not isinstance(data, dict):
            logger.warning("Postprocessing metadata '%s' is malformed (expected a JSON object).", self.path)
            return None
        return data

    def save(self, data):
        """Writes `data` (a dict) to disk as pretty-printed JSON, creating the metadata directory if needed."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Could not write postprocessing metadata '%s': %s", self.path, exc)

    def update(self, patch):
        """
        Merges `patch` into the existing metadata (loading it first, or
        starting from `{}` if absent/corrupted) and writes the result
        back to disk. Top-level dict values (e.g. `postprocessing`) are
        merged one level deep instead of being overwritten wholesale, so
        unrelated sections (like RloadId/Gain vs. postprocessing
        metadata) never clobber each other.

        Args:
            patch (dict): Partial metadata to merge in.

        Returns:
            dict: The full, merged metadata that was written to disk.
        """
        current = self.load() or {}
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key].update(value)
            else:
                current[key] = value
        self.save(current)
        return current


def _resolve_postprocessing_metadata(experiment, root_dir, identity, loads_map):
    """
    Resolves `RloadId`/`Gain` for a single experiment using its centralized
    postprocessing metadata file (see PostprocessingMetadataStore) as the
    single source of truth:
        - If the file exists and is valid, its `RloadId`/`Gain` values
          are applied to `experiment` (overriding whatever was cached in
          the session/dataframe).
        - Otherwise, `experiment` is initialized from its as-scanned
          `RloadId` and the matching `LoadsDescription.ods` default
          `Gain` (if any), and the metadata file is written immediately
          so the choice persists across future application sessions.

    This never touches the file's `postprocessing` section (CleanData/
    CycleData run metadata), which is managed separately by
    `cleaning_filtering_data.save_clean_data` /
    `cycle_peak_extraction.save_cycle_data`.

    Note: this file is entirely separate from any raw-data experiment
    metadata (e.g. `experiments_metadata.json`), which is never read or
    modified here.

    Args:
        experiment (dict): Mutated in place with resolved `RloadId`/`Gain`.
        root_dir (str | Path): The experiments root_dir (where
            Experiments.xlsx / LoadsDescription.ods live), under which
            the centralized CleanData/.metadata directory is placed.
        identity (dict): The experiment's Phase 1 naming/output identity
            (see experiment_path_resolver.ExperimentPathResolver.build_identity),
            used as the centralized metadata file's lookup key.
        loads_map (dict[str, float | None]): RloadId -> default Gain, as
            read from LoadsDescription.ods.

    Returns:
        dict: The same `experiment` dict, for convenience.
    """
    store = PostprocessingMetadataStore(root_dir, identity)
    meta = store.load()

    if meta is not None and "RloadId" in meta:
        experiment["RloadId"] = meta.get("RloadId")
        experiment["Gain"] = meta.get("Gain")
    else:
        if experiment.get("Gain") in (None, ""):
            rload_key = str(experiment.get("RloadId", "")).strip()
            experiment["Gain"] = loads_map.get(rload_key)
        store.update({"RloadId": experiment.get("RloadId"), "Gain": experiment.get("Gain")})

    return experiment


def record_postprocessing_metadata(root_dir, identity, section, data):
    """
    Records postprocessing run metadata (e.g. the recipe/configuration
    used and a timestamp) for a single experiment, under the
    `postprocessing.<section>` key of its centralized postprocessing
    metadata file, without disturbing the `RloadId`/`Gain` load
    configuration stored alongside it.

    Called by `cleaning_filtering_data.save_clean_data` (section
    `"clean_data"`) and `cycle_peak_extraction.save_cycle_data` (section
    `"cycle_data"`), so the single metadata file also tracks what
    postprocessing has been run on each experiment and when, independent
    of the CleanData/CycleData pickle containers themselves.

    Args:
        root_dir (str | Path): The experiments root_dir.
        identity (dict): The experiment's Phase 1 naming/output identity.
        section (str): Sub-key under `postprocessing` (e.g. "clean_data").
        data (dict): Metadata to store for that section (e.g.
            `{"recipe": ..., "saved_at": ..., "warnings": [...]}`).

    Returns:
        dict: The full, merged metadata that was written to disk.
    """
    store = PostprocessingMetadataStore(root_dir, identity)
    return store.update({POSTPROCESSING_KEY: {section: data}})


class LoadsDescriptionRepository:
    """
    Reads `LoadsDescription.ods` (the source of truth for standard load
    configurations) and exposes the valid `RloadId` -> default `Gain`
    mapping used to populate the RloadId dropdown and Gain defaults in
    the experiments preview table.
    """

    def __init__(self, root_dir):
        self.path = Path(root_dir) / METADATA_LOADS_FILENAME

    def load(self):
        """
        Returns:
            dict[str, float | None]: Mapping of `RloadId` (as string) to
            its default `Gain` (float, or None if missing/invalid in the
            source file). Returns an empty dict if the file is missing,
            unreadable, or does not contain the expected columns.
        """
        if not self.path.is_file():
            logger.warning("'%s' not found; RloadId dropdown will be empty.", self.path)
            return {}

        try:
            df = pd.read_excel(self.path, engine="odf")
        except Exception as exc:
            logger.warning("Could not read '%s': %s", self.path, exc)
            return {}

        if "RloadId" not in df.columns:
            logger.warning("'%s' is missing the required 'RloadId' column.", self.path)
            return {}
        if "Gain" not in df.columns:
            logger.warning("'%s' is missing the 'Gain' column; defaults will be unavailable.", self.path)

        mapping = {}
        for _, row in df.iterrows():
            rload_id = row.get("RloadId")
            if pd.isna(rload_id):
                continue
            key = str(rload_id).strip()

            gain_value = row.get("Gain") if "Gain" in df.columns else None
            try:
                gain = None if gain_value is None or pd.isna(gain_value) else float(gain_value)
            except (TypeError, ValueError):
                logger.warning("Invalid default Gain for RloadId '%s' in '%s'.", key, self.path)
                gain = None

            mapping[key] = gain
        return mapping


# ----------------------------------------------------------------------
# Flask views
# ----------------------------------------------------------------------
def experiments_preview():
    """
    Render the experiments preview table.

    Reads the experiments loaded in the session (set by the data loading
    step), inspects the local filesystem for CleanData/CycleData outputs,
    resolves each experiment's `RloadId`/`Gain` against
    `LoadsDescription.ods`, and renders an interactive table plus a plot
    configuration box.

    Returns:
        str: Rendered HTML template for the experiments preview page.
    """
    experiments = session.get("experiments")
    root_dir = session.get("root_dir")

    if not experiments or not root_dir:
        flash("No experiments loaded yet. Please select a root folder first.", "error")
        return redirect(url_for("render_data_loading"))

    loads_map = LoadsDescriptionRepository(root_dir).load()
    if not loads_map:
        flash(
            f"'{METADATA_LOADS_FILENAME}' could not be read or has no valid RloadId/Gain "
            "entries. The RloadId dropdown will be empty and Gain defaults unavailable.",
            "warning",
        )

    inspector = ExperimentFileStatus(root_dir)

    session_dirty = False
    for experiment in experiments:
        # Preserve the RloadId as originally scanned from disk, so later
        # in-table corrections never break on-disk folder resolution, and
        # so "Reset to Default" has a stable value to fall back to.
        if "_original_rload_id" not in experiment:
            experiment["_original_rload_id"] = experiment.get("RloadId")
            session_dirty = True

        # postprocessing metadata (centralized, see PostprocessingMetadataStore)
        # is the persistent source of truth; it is created on first read if
        # missing, and takes precedence over whatever is cached here. Rows
        # with an invalid Phase 1 format are flagged but don't crash the page.
        try:
            identity = resolve_experiment_identity(experiment)
            _resolve_postprocessing_metadata(experiment, root_dir, identity, loads_map)
        except InvalidExperimentFormatError as exc:
            experiment["_path_error"] = str(exc)
        session_dirty = True

    if session_dirty:
        session["experiments"] = experiments
        session.modified = True

    augmented_experiments = inspector.augment(experiments)

    for experiment in augmented_experiments:
        rload_key = str(experiment.get("RloadId", "")).strip()
        experiment["rload_valid"] = rload_key in loads_map
        experiment["default_gain"] = loads_map.get(rload_key)
        experiment["default_rload_id"] = experiment.get("_original_rload_id")

    tribu_ids = sorted({str(exp.get("TribuId", "")) for exp in augmented_experiments})
    rload_ids = sorted(loads_map.keys())

    return render_template(
        "experiments_preview.html",
        experiments=augmented_experiments,
        tribu_ids=tribu_ids,
        root_dir=root_dir,
        plot_variables=AVAILABLE_PLOT_VARIABLES,
        plot_types=AVAILABLE_PLOT_TYPES,
        rload_ids=rload_ids,
        loads_map=loads_map,
    )


def update_experiment_row(experiment_id):
    """
    Persists an inline edit made to a single experiment row's `RloadId`
    and/or `Gain` in the experiments preview table (or the CleanData
    panel). The change is written to that experiment's on-disk
    `postprocessing_metadata.json` (the single source of truth), and the
    Flask session cache is updated to match.

    Expects a JSON body such as `{"RloadId": "R47", "Gain": 1.5}`; only
    the provided keys are updated. Does not touch `_original_rload_id`,
    which is used to resolve the on-disk raw-data folder regardless of
    later corrections made to `RloadId`.

    Args:
        experiment_id (int): Positional index of the experiment within
            the session's experiments list.

    Returns:
        flask.Response: JSON payload describing the outcome.
    """
    experiments = session.get("experiments") or []
    root_dir = session.get("root_dir")
    if not root_dir or not (0 <= experiment_id < len(experiments)):
        return jsonify({"error": "Unknown experiment."}), 404

    payload = request.get_json(silent=True) or {}
    experiment = dict(experiments[experiment_id])

    if "RloadId" in payload:
        new_rload_id = payload["RloadId"]
        if new_rload_id in (None, ""):
            return jsonify({"error": "RloadId cannot be empty."}), 400
        experiment["RloadId"] = new_rload_id

    if "Gain" in payload:
        gain_value = payload["Gain"]
        if gain_value in (None, ""):
            experiment["Gain"] = None
        else:
            try:
                experiment["Gain"] = float(gain_value)
            except (TypeError, ValueError):
                return jsonify({"error": "Gain must be a real number."}), 400

    experiments[experiment_id] = experiment
    session["experiments"] = experiments

    # Keep the CleanData/CycleData editor's snapshot (if any) in sync, so
    # edits made from either the experiments table or the CleanData panel
    # are reflected in both places without requiring a page reload.
    current_experiment = session.get("current_experiment")
    if current_experiment and current_experiment.get("experiment_id") == experiment_id:
        current_experiment["RloadId"] = experiment.get("RloadId")
        current_experiment["Gain"] = experiment.get("Gain")
        session["current_experiment"] = current_experiment

    session.modified = True

    # The centralized postprocessing metadata file is the single,
    # persistent source of truth: write it immediately so the change
    # survives session expiry, app restarts, or future application sessions.
    try:
        identity = resolve_experiment_identity(experiment)
        PostprocessingMetadataStore(root_dir, identity).update(
            {"RloadId": experiment.get("RloadId"), "Gain": experiment.get("Gain")}
        )
    except InvalidExperimentFormatError as exc:
        logger.warning("Could not persist RloadId/Gain edit for experiment %s: %s", experiment_id, exc)

    return jsonify({"success": True, "RloadId": experiment.get("RloadId"), "Gain": experiment.get("Gain")})


def resolve_entry_point(root_dir, identity):
    """
    Decides whether opening an experiment (Phase 2 entry point) should land
    on the file-selection page or skip straight to the CleanData editor
    (Requirement 2: state persistence / auto-skip).

    Deliberately called ONLY from `open_clean_data` - never from
    `file_selection.file_selection` or `cleaning_filtering_data.
    cleaning_filtering_data` themselves - so the decision is made exactly
    once per "open experiment" action and can never be re-triggered mid-flow
    (which is what would risk an infinite redirect loop).

    Args:
        root_dir (str): Experiments root directory (session["root_dir"]).
        identity (dict): Experiment identity (Phase 1 + Phase 5 key).

    Returns:
        tuple[str, dict | None]: ("clean_data", file_selection_meta) if a
        previously saved file selection is still valid (its merged cache
        file still exists on disk), else ("file_selection", None).
    """
    meta = PostprocessingMetadataStore(root_dir, identity).load() or {}
    file_selection_meta = meta.get("file_selection")
    if not file_selection_meta:
        return "file_selection", None

    cache_path = file_selection_meta.get("merged_cache_path")
    if not cache_path or not Path(cache_path).is_file():
        # Stale metadata (cache was deleted/moved): fall back to a fresh pick.
        return "file_selection", None

    return "clean_data", file_selection_meta


def open_clean_data(experiment_id):
    """
    Opens a specific experiment in CleanData mode (create or edit),
    storing the experiment context in the session.

    Requirement 2 (auto-skip): if this experiment already has a saved file
    selection whose merged cache file still exists on disk (see
    `resolve_entry_point`), the file-selection step (Phase 2/3) is bypassed
    entirely and the user is sent straight to the CleanData recipe editor
    (Phase 4), restoring the session state from the persisted metadata.
    Otherwise, it redirects to file-selection as before.

    Args:
        experiment_id (int): Positional index of the experiment within the
            session's experiments list.

    Returns:
        werkzeug.wrappers.Response: Redirect to the CleanData editor (if
        auto-skipped) or the file-selection page, or back to the preview
        page if the experiment could not be found.
    """
    experiment, root_dir = _get_experiment_or_none(experiment_id)
    if experiment is None:
        flash("Unknown experiment.", "error")
        return redirect(url_for("render_experiments_preview"))

    status = ExperimentFileStatus(root_dir).inspect(experiment)
    if status.get("path_error"):
        flash(status["path_error"], "error")
        return redirect(url_for("render_experiments_preview"))

    session["current_experiment"] = {**experiment, "experiment_id": experiment_id, **status}
    session["editor_mode"] = "clean"

    try:
        identity = resolve_experiment_identity(experiment)
        target, file_selection_meta = resolve_entry_point(root_dir, identity)
    except InvalidExperimentFormatError as exc:
        flash(str(exc), "error")
        return redirect(url_for("render_experiments_preview"))

    if target == "clean_data":
        session["merged_data_path"] = file_selection_meta["merged_cache_path"]
        session["file_selection"] = {
            "paths": [f["path"] for f in file_selection_meta.get("selected_files", [])],
            "merge_enabled": file_selection_meta.get("merge_enabled", False),
            "time_columns": {
                f["path"]: f["time_column"]
                for f in file_selection_meta.get("selected_files", [])
                if f.get("time_column")
            },
        }
        session.modified = True
        return redirect(url_for("render_cleaning_filtering_data"))

    # Selecting a (possibly different) experiment always starts a fresh
    # Phase 2/3 file selection, so Phase 4 never operates on another
    # experiment's stale merged data (see file_selection.py).
    session.pop("merged_data_path", None)
    session.pop("file_selection", None)

    return redirect(url_for("render_file_selection"))


def abort_to_menu():
    """
    "Return to Main Menu" action (Requirement 3), shared by both the
    CleanData and CycleData editors: discards any unsaved in-editor state
    and returns to the experiments preview page. Never writes anything to
    disk, so any not-yet-saved filtering/cycle-extraction progress is
    simply dropped.

    Returns:
        werkzeug.wrappers.Response: Redirect to the experiments preview page.
    """
    for key in ("current_experiment", "editor_mode", "merged_data_path", "file_selection"):
        session.pop(key, None)
    session.modified = True
    return redirect(url_for("render_experiments_preview"))


def open_cycle_data(experiment_id):
    """
    Opens a specific experiment in CycleData mode (create or edit). Only
    allowed if CleanData already exists for that experiment.

    Args:
        experiment_id (int): Positional index of the experiment within the
            session's experiments list.

    Returns:
        werkzeug.wrappers.Response: Redirect to the cycle/peak extraction
        page, or back to the preview page if not allowed / not found.
    """
    experiment, root_dir = _get_experiment_or_none(experiment_id)
    if experiment is None:
        flash("Unknown experiment.", "error")
        return redirect(url_for("render_experiments_preview"))

    status = ExperimentFileStatus(root_dir).inspect(experiment)

    if status.get("path_error"):
        flash(status["path_error"], "error")
        return redirect(url_for("render_experiments_preview"))

    if not status["clean_data_exists"]:
        flash("CleanData must be created before CycleData can be edited.", "error")
        return redirect(url_for("render_experiments_preview"))

    session["current_experiment"] = {**experiment, "experiment_id": experiment_id, **status}
    session["editor_mode"] = "cycle"

    return redirect(url_for("render_cycle_peak_extraction"))


def generate_plots():
    """
    Receives the selected experiment IDs and plot configurations submitted
    from the experiments preview page, validates them, stores them in the
    session and redirects to the multi-plot dashboard view.

    Expects a POST request with:
        - `selected_experiments`: list of experiment_id values (form field,
          repeated per checked row).
        - `plot_configs`: JSON-encoded list of plot configuration dicts,
          e.g. [{"title": "...", "xVar": "Time", "yVar": "Voltage",
          "plotType": "Line"}, ...].

    Returns:
        werkzeug.wrappers.Response: Redirect to the figure generation page,
        or back to the preview page if validation fails.
    """
    experiments = session.get("experiments") or []
    root_dir = session.get("root_dir")

    if not experiments or not root_dir:
        flash("No experiments loaded yet. Please select a root folder first.", "error")
        return redirect(url_for("render_data_loading"))

    try:
        selected_ids = [int(value) for value in request.form.getlist("selected_experiments")]
    except ValueError:
        flash("Invalid experiment selection.", "error")
        return redirect(url_for("render_experiments_preview"))

    try:
        plot_configs = json.loads(request.form.get("plot_configs", "[]"))
        if not isinstance(plot_configs, list):
            raise ValueError("plot_configs must be a list")
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid plot_configs payload: %s", exc)
        flash("Invalid plot configuration payload.", "error")
        return redirect(url_for("render_experiments_preview"))

    if not selected_ids:
        flash("Select at least one experiment with CycleData available.", "error")
        return redirect(url_for("render_experiments_preview"))

    if not plot_configs:
        flash("Add at least one plot configuration before generating plots.", "error")
        return redirect(url_for("render_experiments_preview"))

    inspector = ExperimentFileStatus(root_dir)
    selected_experiments = []
    for idx in selected_ids:
        if not (0 <= idx < len(experiments)):
            logger.warning("Skipping out-of-range experiment_id: %s", idx)
            continue
        status = inspector.inspect(experiments[idx])
        if not status["cycle_data_exists"]:
            logger.warning("Skipping experiment %s: CycleData not available.", idx)
            continue
        selected_experiments.append({**experiments[idx], "experiment_id": idx, **status})

    if not selected_experiments:
        flash("None of the selected experiments have CycleData available.", "error")
        return redirect(url_for("render_experiments_preview"))

    session["selected_experiments"] = selected_experiments
    session["plot_configs"] = plot_configs

    return redirect(url_for("render_figure_generation"))


def _get_experiment_or_none(experiment_id):
    """
    Looks up an experiment by its positional ID in the session.

    Returns:
        tuple[dict | None, str | None]: The experiment record and root_dir,
        or (None, None) if not found / session is empty.
    """
    experiments = session.get("experiments") or []
    root_dir = session.get("root_dir")

    if root_dir is None or not (0 <= experiment_id < len(experiments)):
        return None, None

    return experiments[experiment_id], root_dir
