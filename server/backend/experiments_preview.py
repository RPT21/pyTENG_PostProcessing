import json
import logging
from pathlib import Path

from flask import render_template, request, redirect, url_for, session, flash

from server.backend.data_loading import ExperimentsFolderLoader

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
CLEAN_DATA_FOLDER_NAME = "CleanData"
CYCLE_DATA_FOLDER_NAME = "CycleData"
CLEAN_DATA_EXTENSIONS = (".pkl", ".csv")
CYCLE_DATA_EXTENSIONS = (".pkl", ".csv")

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


class ExperimentFileStatus:
    """
    Resolves the on-disk raw-data folder for each experiment row and
    inspects whether CleanData/CycleData outputs already exist for it.
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

    def folder_path(self, experiment):
        """
        Rebuilds an experiment's raw-data folder path, following the
        `root_dir / TribuId / Pos-Neg / Date-RloadId` convention used by
        ExperimentsFolderLoader when the folder tree was scanned.

        Args:
            experiment (dict): A single experiment record (as produced by
                `DataFrame.to_dict(orient="records")`).

        Returns:
            pathlib.Path: The resolved folder path (it may not exist).
        """
        date_value = experiment.get("Date")
        try:
            if hasattr(date_value, "strftime"):
                # Date was parsed back into a real datetime/Timestamp by pandas.
                date_token = date_value.strftime("%d%m%Y_%H%M%S")
            else:
                date_token = ExperimentsFolderLoader._make_date_token(date_value)
        except Exception as exc:
            logger.warning("Could not build date token for experiment %s: %s", experiment, exc)
            date_token = str(date_value or "").strip()

        pos_neg = f"{experiment.get('SampleIdTriboPos', '')}-{experiment.get('SampleIdTriboNeg', '')}"
        leaf = f"{date_token}-{experiment.get('RloadId', '')}"

        return self.root_dir / str(experiment.get("TribuId", "")) / pos_neg / leaf

    @staticmethod
    def _find_existing_file(folder, subfolder_name, valid_extensions):
        """Returns the first matching file found in `folder/subfolder_name`, or None."""
        subfolder = folder / subfolder_name
        if not subfolder.is_dir():
            return None
        try:
            for entry in sorted(subfolder.iterdir()):
                if entry.is_file() and entry.suffix.lower() in valid_extensions:
                    return entry
        except OSError as exc:
            logger.warning("Could not read directory %s: %s", subfolder, exc)
        return None

    def inspect(self, experiment):
        """
        Returns a dict describing the on-disk status of a single experiment:
            {
                "folder_path": str,
                "clean_data_exists": bool,
                "clean_data_path": str | None,
                "cycle_data_exists": bool,
                "cycle_data_path": str | None,
            }
        """
        folder = self.folder_path(experiment)
        clean_file = self._find_existing_file(folder, CLEAN_DATA_FOLDER_NAME, CLEAN_DATA_EXTENSIONS)
        cycle_file = self._find_existing_file(folder, CYCLE_DATA_FOLDER_NAME, CYCLE_DATA_EXTENSIONS)

        return {
            "folder_path": str(folder),
            "clean_data_exists": clean_file is not None,
            "clean_data_path": str(clean_file) if clean_file else None,
            "cycle_data_exists": cycle_file is not None,
            "cycle_data_path": str(cycle_file) if cycle_file else None,
        }

    def augment(self, experiments):
        """
        Augments a list of experiment dict rows with a positional
        `experiment_id` and the file-status information above.

        Args:
            experiments (list[dict]): Experiment records.

        Returns:
            list[dict]: The same records, each extended with
            `experiment_id`, `folder_path`, `clean_data_exists`,
            `clean_data_path`, `cycle_data_exists` and `cycle_data_path`.
        """
        augmented = []
        for idx, experiment in enumerate(experiments):
            row = dict(experiment)
            row["experiment_id"] = idx
            row.update(self.inspect(experiment))
            augmented.append(row)
        return augmented


# ----------------------------------------------------------------------
# Flask views
# ----------------------------------------------------------------------
def experiments_preview():
    """
    Render the experiments preview table.

    Reads the experiments loaded in the session (set by the data loading
    step), inspects the local filesystem for CleanData/CycleData outputs,
    and renders an interactive table plus a plot configuration box.

    Returns:
        str: Rendered HTML template for the experiments preview page.
    """
    experiments = session.get("experiments")
    root_dir = session.get("root_dir")

    if not experiments or not root_dir:
        flash("No experiments loaded yet. Please select a root folder first.", "error")
        return redirect(url_for("render_data_loading"))

    inspector = ExperimentFileStatus(root_dir)
    augmented_experiments = inspector.augment(experiments)

    tribu_ids = sorted({str(exp.get("TribuId", "")) for exp in augmented_experiments})

    return render_template(
        "experiments_preview.html",
        experiments=augmented_experiments,
        tribu_ids=tribu_ids,
        root_dir=root_dir,
        plot_variables=AVAILABLE_PLOT_VARIABLES,
        plot_types=AVAILABLE_PLOT_TYPES,
    )


def open_clean_data(experiment_id):
    """
    Opens a specific experiment in CleanData mode (create or edit),
    storing the experiment context in the session and redirecting to the
    cleaning/filtering view.

    Args:
        experiment_id (int): Positional index of the experiment within the
            session's experiments list.

    Returns:
        werkzeug.wrappers.Response: Redirect to the cleaning/filtering page
        (or back to the preview page if the experiment could not be found).
    """
    experiment, root_dir = _get_experiment_or_none(experiment_id)
    if experiment is None:
        flash("Unknown experiment.", "error")
        return redirect(url_for("render_experiments_preview"))

    status = ExperimentFileStatus(root_dir).inspect(experiment)

    session["current_experiment"] = {**experiment, "experiment_id": experiment_id, **status}
    session["editor_mode"] = "clean"

    return redirect(url_for("render_cleaning_filtering_data"))


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
