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


# ----------------------------------------------------------------------
# Strategy B: Advanced Cluster Merge (boolean sync signals)
# ----------------------------------------------------------------------
# Used by the MergeData window (server/backend/file_selection.py) whenever
# the user splits the selected files into two clusters instead of a single
# flat "Merge selected files" list (Strategy A above). Each cluster is
# first collapsed to one DataFrame with Strategy A, then the two resulting
# DataFrames are drift-corrected and aligned against each other using a
# boolean synchronization signal present in both.

def detect_edges(series: pd.Series):
    """
    Locates the first rising edge (0->1) and last falling edge (1->0) of a
    boolean-like (0/1) series.

    Returns:
        tuple[int, int]: (first_rising_index, last_falling_index), as
        positional indices into a 0-based, reset index.

    Raises:
        MergeError: If the series never rises, or never falls back down
        (e.g. a sync column that stays constant throughout the file).
    """
    values = series.reset_index(drop=True).round().astype(int)
    diff = values.diff()

    rising = diff.index[diff == 1].tolist()
    falling = diff.index[diff == -1].tolist()

    if not rising:
        raise MergeError("No rising edge (0 -> 1) found in the sync signal column.")
    if not falling:
        raise MergeError("No falling edge (1 -> 0) found in the sync signal column.")

    return rising[0], falling[-1]


def clip_between_edges(df: pd.DataFrame, sync_column: str, time_col: str = "Time") -> pd.DataFrame:
    """
    Clips `df` to the samples between the first rising edge and the last
    falling edge of `sync_column`, then re-zeroes `time_col` so the clipped
    result starts at t=0 again (Strategy B, steps 2-3).
    """
    if sync_column not in df.columns:
        raise MergeError(f"Sync signal column '{sync_column}' not found in the merged cluster data.")

    start_idx, end_idx = detect_edges(df[sync_column])
    clipped = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    if clipped.empty:
        raise MergeError("Clipping between the sync signal's edges produced an empty DataFrame.")

    clipped[time_col] = clipped[time_col] - clipped[time_col].iloc[0]
    return clipped


def stretch_time(df: pd.DataFrame, factor: float, time_col: str = "Time") -> pd.DataFrame:
    """Scales `time_col` by `factor` (drift correction, Strategy B step 4). Returns a new DataFrame."""
    df = df.copy()
    df[time_col] = df[time_col] * factor
    return df


def interpolate_onto_time_vector(df: pd.DataFrame, master_time, time_col: str = "Time",
                                  binary_like_cols=()) -> pd.DataFrame:
    """
    Resamples every non-time column of `df` onto `master_time` using
    `numpy.interp` (Strategy B step 5, final alignment). Columns listed in
    `binary_like_cols` are rounded back to {0, 1} ints after interpolation.

    Unlike `synchronize_dataframes` (which always picks the highest-sampling-
    rate DataFrame as the master), this always targets an externally
    supplied master vector - here, the reference cluster's own time axis -
    regardless of which cluster happens to be more densely sampled.
    """
    source_time = df[time_col].to_numpy(dtype=float)
    master_time = np.asarray(master_time, dtype=float)

    result = {time_col: master_time}
    for col in df.columns:
        if col == time_col:
            continue
        values = df[col].to_numpy(dtype=float)
        interpolated = np.interp(master_time, source_time, values)
        if col in binary_like_cols:
            interpolated = np.round(interpolated).astype(int)
        result[col] = interpolated

    return pd.DataFrame(result)


def merge_clusters(cluster_paths, reference_cluster, sync_columns, time_columns=None):
    """
    Strategy B entry point: merges two clusters of files (each internally
    merged via Strategy A) using a boolean sync signal present in each
    cluster to drift-correct and align them onto a single time axis.

    Args:
        cluster_paths (dict[str, list[Path]]): Exactly two entries, e.g.
            {"A": [daq_file1, daq_file2], "B": [motor_file]}.
        reference_cluster (str): Key of `cluster_paths` whose (post-clip)
            time axis the other cluster is stretched/interpolated onto.
        sync_columns (dict[str, str]): Per-cluster boolean sync column name
            (as it appears in that cluster's Strategy-A-merged DataFrame,
            i.e. namespaced as '<file_stem>::<original_column>' whenever
            its cluster has more than one file).
        time_columns (dict[str, str] | None): Optional explicit per-file
            time column overrides, forwarded to the intra-cluster
            Strategy A merge (see `merge_selected_files`).

    Returns:
        pandas.DataFrame: A single DataFrame combining both clusters' data
        columns on the reference cluster's (clipped, zero-offset) time axis.

    Raises:
        MergeError: If the clusters are malformed, a sync column is
        missing, never transitions, or the drift-corrected durations can't
        be reconciled.
    """
    if set(cluster_paths.keys()) != {"A", "B"}:
        raise MergeError("Cluster merge requires exactly two clusters, keyed 'A' and 'B'.")
    if not cluster_paths["A"] or not cluster_paths["B"]:
        raise MergeError("Both clusters must contain at least one file.")
    if reference_cluster not in cluster_paths:
        raise MergeError(f"Unknown reference cluster: '{reference_cluster}'.")

    # Step 1: Intra-cluster merge (Strategy A), independently per cluster.
    # Always routed through merge_selected_files (even for single-file
    # clusters) so column namespacing ('<stem>::<col>') is consistent
    # regardless of how many files ended up in a given cluster - this is
    # what `sync_columns` is expected to reference.
    merged = {
        key: merge_selected_files(paths, time_columns=time_columns)
        for key, paths in cluster_paths.items()
    }

    # Steps 2-3: Sync edge detection, then clip + re-zero each cluster's time axis.
    clipped = {
        key: clip_between_edges(df, sync_columns.get(key), time_col="Time")
        for key, df in merged.items()
    }

    non_reference = next(key for key in clipped if key != reference_cluster)

    # Step 4: Time-stretching (drift correction) of the non-reference cluster.
    duration_ref = float(clipped[reference_cluster]["Time"].iloc[-1])
    duration_nonref = float(clipped[non_reference]["Time"].iloc[-1])
    if duration_ref <= 0 or duration_nonref <= 0:
        raise MergeError("A clipped cluster has zero (or negative) duration; cannot compute a stretch factor.")

    factor = duration_ref / duration_nonref
    clipped[non_reference] = stretch_time(clipped[non_reference], factor, time_col="Time")

    # Step 5: Final alignment - interpolate the non-reference cluster onto
    # the reference cluster's own (clipped, zero-offset) time vector.
    master_time = clipped[reference_cluster]["Time"].to_numpy(dtype=float)
    binary_like_cols = tuple(sync_columns.values())
    aligned_nonref = interpolate_onto_time_vector(
        clipped[non_reference], master_time, time_col="Time", binary_like_cols=binary_like_cols,
    )

    df_reference = clipped[reference_cluster].reset_index(drop=True)
    df_nonreference = aligned_nonref.drop(columns=["Time"]).reset_index(drop=True)

    return pd.concat([df_reference, df_nonreference], axis=1)
