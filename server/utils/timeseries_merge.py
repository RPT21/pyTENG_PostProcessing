"""
Phase 3: turns the raw data file(s) selected in Phase 2 (file_discovery.py /
file_selection.py) into a single, time-aligned DataFrame.

This module is the orchestration layer around
synchronization_functions.synchronize_dataframes, which stays the low-level,
DAQ-agnostic numeric engine (resampling/interpolation math). Responsibilities
split as follows:
    - synchronization_functions.py: pure resampling/interpolation math,
      plus the existing DAQ/Motor-specific loaders (merge_DAQ_data,
      LoadMotorFile, LoadDAQData, ExtractCycles) used by cycle extraction.
    - timeseries_merge.py (here): generic raw-file loading + column
      namespacing + orchestration for arbitrary, user-selected files.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from server.utils.synchronization_functions import synchronize_dataframes, LTIME_to_seconds

logger = logging.getLogger(__name__)

TIME_COLUMN_CANDIDATES = ("time", "t", "time (s)", "time(s)", "timestamp", "ltime", "elapsed time")


class MergeError(Exception):
    """Raised when selected files cannot be loaded or merged."""


def detect_time_column(columns):
    """
    Guess of which column represents time: an exact (case/space-insensitive)
    name match against TIME_COLUMN_CANDIDATES.

    Used both by Phase 2/3 file-selection UI (to pre-select each file's
    time-column dropdown - see file_selection.py) and by `guess_time_column`
    below (the stricter variant used once a time column is actually
    required, e.g. before merging).

    Args:
        columns (Iterable): Column labels to consider.

    Returns:
        str | None: The matching column name, or None if none of the
        columns matches TIME_COLUMN_CANDIDATES.
    """
    for c in columns:
        if str(c).strip().lower() in TIME_COLUMN_CANDIDATES:
            return c
    return None


def guess_time_column(df: pd.DataFrame):
    """
    Best-effort guess of which column represents time.

    Falls back to the DataFrame's first column when no name in
    TIME_COLUMN_CANDIDATES is found, rather than failing outright: the user
    can always correct a wrong guess via the per-file time-column dropdown
    in file_selection.py (Requirement 1) before merging/loading actually
    happens, so a silent-but-overridable default is preferable to blocking
    the whole selection on an unrecognized column name.

    Returns:
        str | None: The guessed column name, or None if `df` has no columns
        at all.
    """
    col = detect_time_column(df.columns)
    if col is not None:
        return col
    if len(df.columns) == 0:
        return None
    logger.warning(
        "No column matching a known time-column name was found; "
        "defaulting to the first column '%s'. Verify/correct it before merging.",
        df.columns[0],
    )
    return df.columns[0]


def _normalize_time_series(series: pd.Series) -> pd.Series:
    """
    Converts a raw time column to float seconds.

    Most files already store numeric seconds and are simply cast to float.
    Some instruments instead export a custom "LTIME" string format (e.g.
    '1d2h3m4s500ms'), which synchronization_functions.LTIME_to_seconds
    knows how to parse; that conversion is applied automatically whenever
    the column isn't already numeric.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    try:
        return series.apply(LTIME_to_seconds).astype(float)
    except (TypeError, ValueError, KeyError) as exc:
        raise MergeError(f"Could not parse time column values as LTIME strings: {exc}") from exc


def prepare_time_column(df: pd.DataFrame, time_col: str = None, align_start: bool = True) -> pd.DataFrame:
    """
    Normalizes `df`'s time axis into a numeric, ascending 'Time' column
    (in seconds), so downstream steps (merging, CleanData filtering, cycle
    extraction) never have to deal with string-formatted timestamps.

    Handles both already-numeric time columns and the custom LTIME string
    format (via `_normalize_time_series`). If `align_start` (default), the
    time axis is shifted so the first sample starts at 0, i.e. the offset
    is removed - required for arbitrary user-picked files, which are not
    guaranteed to share a physical t=0 reference.
    """
    col = time_col or guess_time_column(df)
    if col is None:
        raise MergeError("File has no columns to use as a time axis.")
    df = df.rename(columns={col: "Time"}) if col != "Time" else df.copy()
    df["Time"] = _normalize_time_series(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    if align_start:
        df["Time"] = df["Time"] - df["Time"].iloc[0]
    return df


def _load_tdms(path: Path) -> pd.DataFrame:
    try:
        from nptdms import TdmsFile
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MergeError("The 'nptdms' package is required to read .tdms files.") from exc

    tdms_file = TdmsFile.read(str(path))
    target_channel = None
    for group in tdms_file.groups():
        for channel in group.channels():
            if channel.name == "Input 0":
                target_channel = channel
                break
        if target_channel:
            break

    if not target_channel:
        raise MergeError(f"TDMS file '{path}' does not contain an 'Input 0' channel")

    data = target_channel[:]
    dt = target_channel.properties.get("wf_increment")
    if dt is None:
        fs = target_channel.properties.get("sampling_rate", 1000.0)
        dt = 1.0 / fs

    time_s = np.arange(len(data)) * dt
    return pd.DataFrame({"Input 0": data, "Time (s)": time_s})


def load_raw_file(path: Path) -> pd.DataFrame:
    """
    Single dispatch point for reading one raw data file, shared by Phase 2/3
    (merging) and Phase 4 (CleanDataProcessor no longer needs its own
    reader: it always consumes the Phase-3 merged cache instead).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pkl":
        df = pd.read_pickle(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".xlsx":
        df = pd.read_excel(path)
    elif suffix == ".tdms":
        df = _load_tdms(path)
    else:
        raise MergeError(f"Unsupported file extension: {suffix}")

    if not isinstance(df, pd.DataFrame):
        raise MergeError(f"File did not yield a DataFrame: {path}")
    return df


def load_and_prepare_file(path: Path, time_col: str = None, align_start: bool = True) -> pd.DataFrame:
    """
    Loads a single raw file (Phase 2) and normalizes its time column
    (Phase 3 prerequisite), so both the single-file and multi-file
    selection paths in file_selection.py produce a consistent, numeric,
    zero-offset 'Time' column for CleanData/CycleData to consume.

    Unlike merge_selected_files, a file with no identifiable time column
    is not treated as an error here: its raw columns are returned
    untouched (with a warning logged) rather than failing the whole load,
    since a single selected file need not necessarily be a time series.
    """
    df = load_raw_file(path)
    try:
        df = prepare_time_column(df, time_col=time_col, align_start=align_start)
    except MergeError as exc:
        logger.warning("Could not normalize time column for %s: %s", path, exc)
    return df


def merge_selected_files(file_paths, time_columns=None, align_start=True, filter_time=True):
    """
    Phase 3 entry point: loads the user-selected files (Phase 2 output),
    normalizes each to a common 'Time' column, namespaces overlapping
    variable names per source file, then delegates the actual
    resampling/interpolation to synchronize_dataframes (the file with the
    highest sampling rate sets the master time axis).

    Args:
        file_paths (list[Path]): Files selected by the user in Phase 2.
        time_columns (dict[Path, str] | None): Optional explicit time
            column per file; falls back to `guess_time_column` when omitted.
        align_start (bool): If True (default), shift every file's time axis
            so it starts at 0 before synchronizing. Required here (unlike
            the DAQ-specific merge_DAQ_data) because arbitrary user-picked
            files are not guaranteed to share a physical t=0 reference.
        filter_time (bool): Passed through to synchronize_dataframes; clips
            all files to their shortest common duration.

    Returns:
        pandas.DataFrame: Single merged/interpolated DataFrame with a
        'Time' column plus every selected file's data columns, namespaced
        as '<file_stem>::<original_column>'.

    Raises:
        MergeError: If no files are given or a file can't be parsed.
    """
    if not file_paths:
        raise MergeError("No files selected to merge.")

    # Normalize keys so callers can pass either str or Path (file_selection.py
    # builds this dict from HTML form field names, which are always strings).
    time_columns = {str(Path(k)): v for k, v in (time_columns or {}).items()}
    prepared = []

    for path in file_paths:
        path = Path(path)
        df = load_raw_file(path)

        time_col = time_columns.get(str(path)) or guess_time_column(df)
        df = prepare_time_column(df, time_col=time_col, align_start=align_start)

        # Namespace non-time columns per source file to avoid silent
        # collisions (e.g. two files both having a "Force" column).
        stem = path.stem
        df = df.rename(columns={c: f"{stem}::{c}" for c in df.columns if c != "Time"})
        prepared.append(df)

    if len(prepared) == 1:
        return prepared[0]

    try:
        synced = synchronize_dataframes(
            prepared, time_col="Time", filter_time=filter_time,
            binary_cols=(), require_common_start=False,
        )
    except ValueError as exc:
        raise MergeError(str(exc)) from exc

    merged = pd.concat(synced, axis=1)
    merged.index.name = "Time"
    return merged.reset_index()
