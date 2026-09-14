import ctypes
import logging
import platform
from pathlib import Path

import pandas as pd
from flask import render_template, request, redirect, url_for, session, flash, jsonify

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
METADATA_LOADS_FILENAME = "LoadsDescription.ods"
EXPERIMENTS_FILENAME = "Experiments.xlsx"

VALID_DATA_EXTENSIONS = (".tdms", ".csv", ".xlsx", ".pkl")
EXCLUDED_FOLDER_NAMES = {"CleanData", "CycleData"}

EXPERIMENTS_COLUMNS = [
    "TribuId",
    "SampleIdTriboNeg",
    "SampleIdTriboPos",
    "Date",
    "RloadId",
    "FolderPath",
]


class ExperimentsFolderLoader:
    """
    Service class responsible for loading (or reconstructing) the
    'Experiments.xlsx' metadata table from a root experiments folder.

    Expected folder structure:
        root_dir / [TribuId] / [SampleIdTriboPos-SampleIdTriboNeg] / [Date-RloadId]

    Any folder named exactly 'CleanData' or 'CycleData' is ignored at any depth.
    """

    RAW_DATA_FOLDER_NAME = "RawData"

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.loads_description_path = self.root_dir / METADATA_LOADS_FILENAME
        self.experiments_path = self.root_dir / EXPERIMENTS_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self):
        """
        Main entry point.

        Checks whether the root folder is valid, whether the LoadsDescription
        file exists (informative only) and whether Experiments.xlsx already
        exists. If it does, it is loaded directly; otherwise the folder tree
        is scanned and Experiments.xlsx is (re)created.

        Returns:
            tuple[pandas.DataFrame, dict]: The experiments DataFrame and a
            dict with status information intended for user feedback:
                {
                    "root_dir": str,
                    "loads_description_found": bool,
                    "experiments_reconstructed": bool,
                    "n_experiments": int,
                    "warnings": list[str],
                    "messages": list[str],
                }
        """
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")

        status = {
            "root_dir": str(self.root_dir),
            "loads_description_found": False,
            "experiments_reconstructed": False,
            "n_experiments": 0,
            "warnings": [],
            "messages": [],
        }

        self._resolve_raw_data_root(status)

        status["root_dir"] = str(self.root_dir)
        status["loads_description_found"] = self.loads_description_path.is_file()

        if not status["loads_description_found"]:
            msg = f"'{METADATA_LOADS_FILENAME}' was not found in {self.root_dir}."
            logger.warning(msg)
            status["warnings"].append(msg)

        if self.experiments_path.is_file():
            df = pd.read_excel(self.experiments_path)
            status["messages"].append(
                f"Loaded existing '{EXPERIMENTS_FILENAME}' with {len(df)} experiment(s)."
            )
        else:
            df, warnings_ = self.scan_experiments()
            status["warnings"].extend(warnings_)
            status["experiments_reconstructed"] = True
            self.save_experiments(df)
            status["messages"].append(
                f"'{EXPERIMENTS_FILENAME}' not found. Reconstructed {len(df)} "
                f"experiment(s) from folder scan and saved it to {self.experiments_path}."
            )

        status["n_experiments"] = len(df)
        return df, status

    def scan_experiments(self):
        """
        Recursively scans root_dir following the expected 3-level structure
        and builds the experiments DataFrame.

        Returns:
            tuple[pandas.DataFrame, list[str]]: The reconstructed DataFrame
            and a list of warning messages collected during the scan.
        """
        rows = []
        warnings_ = []

        for tribu_dir in self._iter_subdirs(self.root_dir):
            tribu_id = tribu_dir.name

            for pos_neg_dir in self._iter_subdirs(tribu_dir):
                sample_pos, sample_neg = self._split_two(
                    pos_neg_dir.name, warnings_, pos_neg_dir
                )
                if sample_pos is None:
                    continue

                for data_dir in self._iter_subdirs(pos_neg_dir):
                    if not self._contains_valid_data(data_dir):
                        continue

                    date, rload_id = self._split_two(
                        data_dir.name, warnings_, data_dir, from_right=True
                    )
                    if date is None:
                        continue

                    rows.append(
                        {
                            "TribuId": tribu_id,
                            "SampleIdTriboPos": sample_pos,
                            "SampleIdTriboNeg": sample_neg,
                            "Date": date,
                            "RloadId": rload_id,
                            "FolderPath": str(data_dir),
                        }
                    )

        df = pd.DataFrame(rows, columns=EXPERIMENTS_COLUMNS)
        return df, warnings_

    def save_experiments(self, df):
        """Saves the experiments DataFrame to Experiments.xlsx in root_dir."""
        df.to_excel(self.experiments_path, index=False)
        return self.experiments_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_raw_data_root(self, status):
        """
        If the given root folder directly contains a subfolder named
        'RawData', switches the effective root_dir (and the derived
        LoadsDescription/Experiments paths) to that subfolder before
        looking for metadata files or scanning.
        """
        raw_data_dir = self.root_dir / self.RAW_DATA_FOLDER_NAME
        if raw_data_dir.is_dir():
            self.root_dir = raw_data_dir
            self.loads_description_path = self.root_dir / METADATA_LOADS_FILENAME
            self.experiments_path = self.root_dir / EXPERIMENTS_FILENAME
            msg = f"Found '{self.RAW_DATA_FOLDER_NAME}' folder; using it as the root: {self.root_dir}"
            logger.info(msg)
            status["messages"].append(msg)

    @staticmethod
    def _iter_subdirs(directory):
        """Yields subdirectories of `directory`, skipping excluded folder names."""
        try:
            children = sorted(directory.iterdir())
        except OSError as exc:
            logger.warning("Could not read directory %s: %s", directory, exc)
            return
        for child in children:
            if child.is_dir() and child.name not in EXCLUDED_FOLDER_NAMES:
                yield child

    @staticmethod
    def _split_two(name, warnings_, folder, from_right=False):
        """
        Safely splits a folder name in two parts separated by '-'.

        Args:
            name (str): Folder name to split.
            warnings_ (list[str]): List to append warning messages to.
            folder (Path): Folder being parsed (for logging context).
            from_right (bool): If True, splits from the right-most hyphen
                (useful for 'Date-RloadId' where Date itself may contain
                hyphens). Otherwise splits at the first hyphen.

        Returns:
            tuple[str | None, str | None]: The two parsed parts, or
            (None, None) if the naming convention does not match.
        """
        parts = name.rsplit("-", 1) if from_right else name.split("-", 1)
        if len(parts) != 2 or not all(p.strip() for p in parts):
            msg = f"Skipping folder with unexpected naming convention: {folder}"
            logger.warning(msg)
            warnings_.append(msg)
            return None, None
        return parts[0].strip(), parts[1].strip()

    @staticmethod
    def _contains_valid_data(directory):
        """
        Returns True if `directory` (recursively, excluding excluded
        subfolders) contains at least one file with a valid data extension.
        """
        try:
            entries = list(directory.iterdir())
        except OSError as exc:
            logger.warning("Could not read directory %s: %s", directory, exc)
            return False

        for entry in entries:
            if entry.is_file() and entry.suffix.lower() in VALID_DATA_EXTENSIONS:
                return True
            if entry.is_dir() and entry.name not in EXCLUDED_FOLDER_NAMES:
                if ExperimentsFolderLoader._contains_valid_data(entry):
                    return True
        return False


# ----------------------------------------------------------------------
# Flask view
# ----------------------------------------------------------------------
def data_loading():
    """
    Render the data loading page and handle folder selection/scanning.

    On POST, reads 'root_dir' from the submitted form, loads (or
    reconstructs) the experiments metadata table via
    ExperimentsFolderLoader, stores the result in the session and redirects
    to the experiments preview page.

    Returns:
        str: Rendered HTML template for the data loading page.
    """
    if request.method == "POST":
        root_dir = request.form.get("root_dir", "").strip()

        if not root_dir:
            flash("Please provide a root experiments folder path.", "error")
            return render_template("data_loading.html")

        try:
            loader = ExperimentsFolderLoader(root_dir)
            df, status = loader.load()
        except FileNotFoundError as exc:
            flash(str(exc), "error")
            return render_template("data_loading.html", root_dir=root_dir)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception("Unexpected error while loading experiments folder")
            flash(f"Unexpected error while scanning folder: {exc}", "error")
            return render_template("data_loading.html", root_dir=root_dir)

        for msg in status["messages"]:
            flash(msg, "success")
        for warn in status["warnings"]:
            flash(warn, "warning")

        session["root_dir"] = status["root_dir"]
        session["experiments"] = df.to_dict(orient="records")

        return redirect(url_for("render_experiments_preview"))

    return render_template("data_loading.html")


def _set_dpi_awareness():
    """
    Makes the current process DPI-aware on Windows so that native Tk
    dialogs (e.g. the folder picker) render sharply on HiDPI displays
    instead of being upscaled/blurred by the OS.

    Safely does nothing on non-Windows platforms or if the underlying
    Windows APIs are unavailable.
    """
    if platform.system() != "Windows":
        return

    try:
        # Per-Monitor DPI Aware (value 2) - Windows 8.1+ / 10 / 11.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            # Fallback for older Windows versions (Vista/7/8).
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            logger.warning("Could not set process DPI awareness; folder dialog may render blurry.")


def browse_folder():
    """
    Open a native OS folder-selection dialog on the server machine and
    return the chosen absolute path as JSON.

    This relies on the server running on the same machine as the browser
    (typical for this local desktop-style tool), since browsers cannot
    expose absolute filesystem paths for security reasons.

    Returns:
        flask.Response: JSON payload {"path": str} where `path` is empty
        if the user cancelled the dialog or no GUI toolkit is available.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        logger.warning("tkinter is not available; cannot open native folder dialog.")
        return jsonify({"path": "", "error": "Folder dialog is not available on this server."})

    _set_dpi_awareness()

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        folder_path = filedialog.askdirectory()
    finally:
        root.destroy()

    return jsonify({"path": folder_path or ""})