"""
Single module for everything related to reading raw data files and
time-synchronizing them into one aligned DataFrame.

Responsibilities merged here (previously split across
synchronization_functions.py + timeseries_merge.py):
    - The single shared raw-file reader `load_raw_file` (csv/pkl/xlsx/tdms).
    - Low-level resampling/interpolation math (`synchronize_dataframes`).
    - Time-column detection/normalization helpers used by the MergeData UI.
    - Strategy A "Simple Merge" (`merge_selected_files`) and Strategy B
      "Advanced Cluster Merge" (`merge_clusters`), both driving the live
      MergeData window (server/backend/file_selection.py) for an arbitrary,
      user-selected pair of files/clusters.
    - The generalized, N-cluster synchronization/cycle-extraction pipeline
      (`synchronize_clusters`, `ExtractCycles`) kept as a standalone,
      future-use utility (not yet wired into any Flask route/UI).
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MergeError(Exception):
    """Raised when selected files cannot be loaded or merged."""


# %% --------------------------------------------------------------------------
# PARSE CUSTOM TIME TO SECONDS
# -----------------------------------------------------------------------------

def LTIME_to_seconds(LTIME):

    conversor = {"d": 86400,
                 "h": 3600,
                 "m": 60,
                 "s": 1,
                 "ms": 1e-3,
                 "us": 1e-6,
                 "ns": 1e-9}

    units = re.split(r'\d+', LTIME)[1:]
    numbers_str = re.findall(r'\d+', LTIME)
    numbers = [int(number) for number in numbers_str]

    total_time = 0

    for number, unit in zip(numbers, units):
        total_time += number * conversor[unit]

    return total_time


# %% --------------------------------------------------------------------------
# VALIDATE BINARY COLUMN FUNCTION
# -----------------------------------------------------------------------------

def validate_binary_column(df, column):
    # Check that all te values are {0, 1}
    if not df[column].isin([0, 1]).all():
        # Identify which are the values that are not 0 or 1
        wrong_values = df[~df[column].isin([0, 1])][column].unique()
        raise ValueError(f"Error: The column '{column}' contains not binary values: {wrong_values}")


def detect_binary_like_columns(df, exclude=()):
    """
    Auto-detects every boolean-like column in `df` (bool dtype, or numeric
    with only {0, 1}/{0.0, 1.0} values, ignoring NaNs), excluding `exclude`
    (typically the time column).

    This MUST be called on a DataFrame's own, not-yet-interpolated values:
    linear interpolation of a step-like 0/1 signal produces intermediate
    fractional values exactly at each transition, which would otherwise
    silently make a genuine boolean/sync signal indistinguishable from a
    regular analog one afterwards. Used by both `synchronize_dataframes`
    (Strategy A) and `merge_clusters` (Strategy B) so that ALL of a
    dataset's boolean-like signals - not just an explicitly-named handful -
    survive resampling/interpolation as exact {0, 1} values.
    """
    exclude = set(exclude)
    columns = []
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            columns.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            uniques = set(pd.unique(series.dropna()))
            if uniques and uniques.issubset({0, 1, 0.0, 1.0}):
                columns.append(col)
    return columns


# %% --------------------------------------------------------------------------
# SYNCHRONIZE DATAFRAMES FUNCTION
# -----------------------------------------------------------------------------

def synchronize_dataframes(dataframes_list, time_col='Time (s)', filter_time=True,
                            binary_cols=("LinMot_Enable", "LinMot_Up_Down"),
                            require_common_start=True):
    """
    Synchronizes a list of DataFrames to the highest sampling rate found among them.
    It does a temporal boundary alignment as well (make all data have the same physical duration).
    IMPORTANT: All dataframes have to start at the same timestep (for example, time = 0) and have different columns

    Args:
        dataframes_list (list): List of Pandas DataFrames.
        time_col (str): The name of the time column (must be present in all DFs).
        filter_time (bool): If True, clips all dataframes to their shortest common duration.
        binary_cols (tuple): Column names to round/cast to int after interpolation.
        require_common_start (bool): If True (default, preserves the original
            behavior relied upon by the legacy Motor/DAQ pipeline), raises
            if the dataframes don't all start at the exact same timestamp. Set
            to False for generic, user-selected files (see
            merge_selected_files) which are pre-aligned to t=0 by the caller
            instead and are not guaranteed to share a physical t=0 reference.

    Returns:
        List of Pandas DataFrames having the same time column as an index
    """

    if not dataframes_list:
        return []

    # =================================================================
    # STEP 0: Temporal Boundary Alignment: all data represent the same physical duration of the experiment
    # =================================================================

    # Check that timestamps are correct:
    for df in dataframes_list:
        if not (df[time_col].is_monotonic_increasing and df[time_col].is_unique):
            raise ValueError("The time column must be strictly increasing for interpolation.")

    if require_common_start:
        # Check that all dataframes start at the same time
        valor_ref = dataframes_list[0].iloc[0][time_col]
        all_equal = all(df.iloc[0][time_col] == valor_ref for df in dataframes_list)
        if not all_equal:
            raise Exception("The dataframes don't start at the same time")

    if filter_time:
        # Find the elapsed time for each dataframe and select the smallest one
        print("Analysing time duration...")
        min_end_time = float('inf')
        for i, df in enumerate(dataframes_list):
            current_time = df[time_col].iloc[-1]
            print(f"  - DataFrame {i}: Time duration = {current_time:.6f} s")
            if current_time < min_end_time:
                min_end_time = current_time
        print(f"The experiment ended at time: {min_end_time:.4f} seconds.")

        # Filter the dataframes to make them have the same time duration
        for i in range(len(dataframes_list)):
            # Filter the dataframes to the min_end_time timestamp (ignore old index [drop=True] and create a new one)
            dataframes_list[i] = \
                dataframes_list[i][dataframes_list[i][time_col] <= min_end_time].copy().reset_index(drop=True)
        print("All dataframes now have the same time duration")

    # =================================================================
    # STEP 1: Find the Highest Sampling Rate DataFrame
    # =================================================================
    min_time_step = float('inf')
    master_time_index = None

    print("Analyzing sampling rates...")

    for i, df in enumerate(dataframes_list):

        current_step = df[time_col].diff().mean()
        print(f"  - DataFrame {i}: Average timestep = {current_step:.6f} s")

        if current_step < min_time_step:
            # Save the timestamp of the highest sampling rate DataFrame
            master_time_index = df[time_col].values
            min_time_step = current_step

    print(f"Target sampling step: {min_time_step:.6f} s")

    # =================================================================
    # STEP 2: Resample and Interpolate all DataFrames
    # =================================================================
    synced_dataframes = []

    for df in dataframes_list:

        # Auto-detect this DataFrame's own boolean-like columns from its
        # ORIGINAL (pre-interpolation) values - once interpolated, a 0/1
        # step signal gets fractional values exactly at each transition,
        # so detection must happen before that, not on `df_sync` below.
        # Unioned with the caller-supplied `binary_cols` (kept for explicit/
        # legacy overrides) so every real boolean/sync signal - not just a
        # hardcoded handful of names - survives resampling as exact {0, 1}.
        binary_candidates = set(binary_cols) | set(detect_binary_like_columns(df, exclude=(time_col,)))

        # 1. Prepare the DF: remove duplicates and set the time column as the index
        df_temp = df.drop_duplicates(subset=[time_col]).set_index(time_col)

        # 2. Reindex and Interpolate
        # - union(): Merges the dataframe's timestamp with the higher sampling rate timestamp
        # - interpolate(method='index'): Uses the actual time values to calculate data in the new timesteps
        # - loc[master_time_index]: Keeps only the points belonging to the master timestamp
        df_sync = (
            df_temp
            .reindex(df_temp.index.union(master_time_index))
            .interpolate(method='index')
            .loc[master_time_index]
        )

        # 3. Ensure binary columns to be int data type
        found_binary_columns = [col for col in binary_candidates if col in df_sync.columns]
        if found_binary_columns:
            # Apply rounding and cast to integer for all matching columns simultaneously (Vectorization)
            df_sync[found_binary_columns] = df_sync[found_binary_columns].round().astype(int)

        synced_dataframes.append(df_sync)

    return synced_dataframes


# %% --------------------------------------------------------------------------
# TIME COLUMN DETECTION / NORMALIZATION (MergeData UI helpers)
# -----------------------------------------------------------------------------

TIME_COLUMN_CANDIDATES = ("time", "t", "time (s)", "time(s)", "timestamp", "ltime", "elapsed time")


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
    '1d2h3m4s500ms'), which LTIME_to_seconds knows how to parse; that
    conversion is applied automatically whenever the column isn't already
    numeric.
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


# %% --------------------------------------------------------------------------
# DYNAMIC FILE LOADING CONFIGURATION
# -----------------------------------------------------------------------------

# Default arguments used to read a raw CSV file when the caller does not provide
# a per-file override (see `file_configs` in `merge_cluster_files`/`synchronize_clusters`,
# or `csv_kwargs` in `load_raw_file`/`merge_selected_files`/`merge_clusters`).
DEFAULT_CSV_READ_KWARGS = dict(header=0, index_col=False, delimiter=',', decimal='.')


def _load_tdms(filepath):
    '''
    Reads a TDMS file's 'Input 0' channel into a two-column DataFrame (the
    channel's raw samples plus a reconstructed 'Time (s)' axis, built from
    the channel's sampling interval/rate metadata). Requires the optional
    `nptdms` package.
    '''
    try:
        from nptdms import TdmsFile
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("The 'nptdms' package is required to read .tdms files.") from e

    tdms_file = TdmsFile.read(str(filepath))
    target_channel = None
    for group in tdms_file.groups():
        for channel in group.channels():
            if channel.name == "Input 0":
                target_channel = channel
                break
        if target_channel:
            break

    if not target_channel:
        raise ValueError(f"TDMS file '{filepath}' does not contain an 'Input 0' channel.")

    data = target_channel[:]
    dt = target_channel.properties.get("wf_increment")
    if dt is None:
        fs = target_channel.properties.get("sampling_rate", 1000.0)
        dt = 1.0 / fs

    time_s = np.arange(len(data)) * dt
    return pd.DataFrame({"Input 0": data, "Time (s)": time_s})


def load_raw_file(filepath, csv_kwargs=None):
    '''
    Single, shared dispatch point for reading one raw data file - used both
    by the generalized cluster-merging pipeline below (`merge_cluster_files`)
    and by the MergeData web UI's orchestration functions
    (`merge_selected_files`/`merge_clusters`).

    The reader used is picked from the file extension: `.csv` (dynamic,
    user-overridable options - see `csv_kwargs`/`DEFAULT_CSV_READ_KWARGS`),
    `.pkl`/`.pickle`, `.xlsx`, or `.tdms` (requires the optional `nptdms`
    package; only its 'Input 0' channel is read).

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the raw data file. The file extension determines which
        reader is used.
    csv_kwargs : dict, optional
        Only relevant for '.csv' files: extra/override keyword arguments
        for `pandas.read_csv`, merged on top of `DEFAULT_CSV_READ_KWARGS`
        (header=0, index_col=False, delimiter=',', decimal='.'), so callers
        only need to specify the options that differ (e.g. a different
        `delimiter` or `decimal` separator).

    Returns
    -------
    pd.DataFrame
        The raw, unmodified contents of the file.
    '''
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == '.csv':
        kwargs = dict(DEFAULT_CSV_READ_KWARGS)
        if csv_kwargs:
            kwargs.update(csv_kwargs)
        try:
            df = pd.read_csv(filepath, **kwargs)
        except Exception as e:
            raise Exception(f"Error reading CSV file '{filepath}' with options {kwargs}: {e}.") from e

    elif ext in ('.pkl', '.pickle'):
        try:
            df = pd.read_pickle(filepath)
        except Exception as e:
            raise Exception(f"Error reading pickle file '{filepath}': {e}.") from e

    elif ext == '.xlsx':
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            raise Exception(f"Error reading Excel file '{filepath}': {e}.") from e

    elif ext == '.tdms':
        df = _load_tdms(filepath)

    else:
        raise ValueError(f"Unsupported file extension '{ext}' for file '{filepath}'.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"File did not yield a DataFrame: '{filepath}'.")
    return df


def load_and_prepare_file(path: Path, time_col: str = None, align_start: bool = True,
                           csv_kwargs: dict = None) -> pd.DataFrame:
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
    df = load_raw_file(path, csv_kwargs=csv_kwargs)
    try:
        df = prepare_time_column(df, time_col=time_col, align_start=align_start)
    except MergeError as exc:
        logger.warning("Could not normalize time column for %s: %s", path, exc)
    return df


# %% --------------------------------------------------------------------------
# STRATEGY A: SIMPLE TEMPORAL MERGE (NO SYNC SIGNALS)
# -----------------------------------------------------------------------------

def merge_selected_files(file_paths, time_columns=None, align_start=True, filter_time=True,
                          csv_kwargs=None):
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
            a fixed-source merge) because arbitrary user-picked files are
            not guaranteed to share a physical t=0 reference.
        filter_time (bool): Passed through to synchronize_dataframes; clips
            all files to their shortest common duration.
        csv_kwargs (dict[str, dict] | None): Optional per-file CSV reader
            overrides (keyed the same way as `time_columns`), merged on top
            of `DEFAULT_CSV_READ_KWARGS` for that specific file. Lets the
            user override e.g. a non-comma delimiter or a different decimal
            separator per file from the MergeData UI.

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
    csv_kwargs = {str(Path(k)): v for k, v in (csv_kwargs or {}).items()}
    prepared = []

    for path in file_paths:
        path = Path(path)
        df = load_raw_file(path, csv_kwargs=csv_kwargs.get(str(path)))

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


# %% --------------------------------------------------------------------------
# STRATEGY B: ADVANCED CLUSTER MERGE (BOOLEAN SYNC SIGNALS)
# -----------------------------------------------------------------------------
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


def merge_clusters(cluster_paths, reference_cluster, sync_columns, time_columns=None, csv_kwargs=None):
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
        csv_kwargs (dict[str, dict] | None): Optional per-file CSV reader
            overrides, forwarded to the intra-cluster Strategy A merge (see
            `merge_selected_files`/`load_raw_file`).

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
        key: merge_selected_files(paths, time_columns=time_columns, csv_kwargs=csv_kwargs)
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
    # Round-trip EVERY boolean-like signal in the non-reference cluster
    # (not just its designated sync column) back to exact {0, 1} after
    # interpolation - otherwise any other Cycle Boolean Signal in that
    # cluster would silently turn into a non-binary column here and
    # disappear from CycleData's boolean-signal selectors.
    binary_like_cols = set(sync_columns.values()) | set(
        detect_binary_like_columns(clipped[non_reference], exclude=("Time",))
    )
    aligned_nonref = interpolate_onto_time_vector(
        clipped[non_reference], master_time, time_col="Time", binary_like_cols=binary_like_cols,
    )

    df_reference = clipped[reference_cluster].reset_index(drop=True)
    df_nonreference = aligned_nonref.drop(columns=["Time"]).reset_index(drop=True)

    return pd.concat([df_reference, df_nonreference], axis=1)


# %% --------------------------------------------------------------------------
# INTRA-CLUSTER MERGE: COMBINE ALL FILES OF ONE CLUSTER INTO A SINGLE DATAFRAME
# -----------------------------------------------------------------------------
# The functions below (merge_cluster_files, _clip_to_enable_window,
# synchronize_clusters, ExtractCycles) generalize the two-cluster Strategy B
# above to an arbitrary number of user-defined clusters, with an explicit
# column->cluster provenance metadata dictionary. Kept as a standalone,
# future-use utility: not currently wired into any Flask route/UI.

def merge_cluster_files(filepaths, time_col='Time', file_configs=None, binary_cols=()):
    '''
    Loads every file belonging to a single cluster and synchronizes them onto one shared
    Time axis (the cluster's own highest sampling-rate file).

    Parameters
    ----------
    filepaths : list of str
        Paths to every raw file that belongs to this cluster.
    time_col : str
        Name of the time column, expected to be present (with the same name) in every file.
    file_configs : dict[str, dict], optional
        Per-file reader configuration: maps a filepath to the `csv_kwargs` passed to
        `load_raw_file` for that specific file (see `DEFAULT_CSV_READ_KWARGS`).
    binary_cols : tuple of str
        Column names to round/cast to int after interpolation (forwarded to
        `synchronize_dataframes`).

    Returns
    -------
    pd.DataFrame
        Single DataFrame containing the time column plus every column from every file
        in the cluster, resampled onto the cluster's own highest-rate time base.
    '''
    file_configs = file_configs or {}

    dataframes = [load_raw_file(fp, csv_kwargs=file_configs.get(fp)) for fp in filepaths]

    # `require_common_start=False`: unlike a fixed pair of sources, user-selected
    # cluster files are not guaranteed to already share a physical t=0 reference.
    synced = synchronize_dataframes(dataframes, time_col=time_col, filter_time=True,
                                     binary_cols=binary_cols, require_common_start=False)

    merged = pd.concat(synced, axis=1)
    merged.index.name = time_col
    merged = merged.reset_index()
    return merged


# %% --------------------------------------------------------------------------
# CLIP A DATAFRAME TO THE [RISING EDGE, FALLING EDGE] WINDOW OF A BOOLEAN COLUMN
# -----------------------------------------------------------------------------

def _clip_to_enable_window(df, enable_col, time_col):
    '''
    Clips `df` to the window delimited by the first rising edge (0->1) and the first
    falling edge (1->0) of `enable_col`, then resets the time column so the clipped data
    starts again at t=0. Factored out so it can be applied to any cluster's own
    "enable"-style synchronization signal.
    '''
    validate_binary_column(df, enable_col)

    diff = df[enable_col].diff()
    up_index = diff.index[diff == 1].tolist()
    down_index = diff.index[diff == -1].tolist()

    if len(up_index) != 1 or len(down_index) != 1:
        raise Exception(
            f"Error, '{enable_col}' rising/falling edge not found (expected exactly one "
            f"rising and one falling edge, found {len(up_index)} and {len(down_index)})."
        )

    up_index = up_index[0] - 1
    down_index = down_index[0]

    clipped = df.loc[up_index:down_index].reset_index(drop=True)
    clipped[time_col] = clipped[time_col] - clipped[time_col].iloc[0]
    return clipped


# %% --------------------------------------------------------------------------
# SYNCHRONIZE MULTIPLE CLUSTERS INTO A SINGLE DATAFRAME + PROVENANCE METADATA
# -----------------------------------------------------------------------------

def synchronize_clusters(clusters_files, enable_signals, time_col='Time',
                          file_configs=None, binary_cols=(), reference_cluster=None):
    '''
    Generalizes two-cluster synchronization (Strategy B) to an arbitrary number of
    user-defined clusters of files.

    Parameters
    ----------
    clusters_files : dict[str, list[str]]
        Maps a cluster name (e.g. "Cluster 1") to the list of raw file paths assigned to it.
    enable_signals : dict[str, str]
        Maps each cluster name to the column name (post-merge, inside that cluster) of its
        own "enable"-style boolean synchronization signal - i.e. a signal that is True for
        exactly one contiguous interval spanning the whole experiment. Used to clip every
        cluster to a common physical window and drift-correct clock differences between
        clusters.
    time_col : str
        Name of the time column, shared by every file/cluster.
    file_configs : dict[str, dict], optional
        Per-file reader configuration forwarded to `merge_cluster_files`/`load_raw_file`.
    binary_cols : tuple of str
        Boolean columns to keep as clean 0/1 integers through every resampling step.
    reference_cluster : str, optional
        Name of the cluster whose duration/time-base other clusters are drift-corrected
        against. Defaults to the first cluster in `clusters_files`.

    Returns
    -------
    merged_df : pd.DataFrame
        A single DataFrame containing the unified time column plus every signal from
        every cluster, resampled onto a common (highest sampling-rate) time base.
    metadata : dict[str, str or None]
        Maps every column name in `merged_df` to the name of the cluster it originated
        from. This is the key piece of information that is otherwise lost once every
        cluster's columns are flattened into one DataFrame: without it there would be no
        way to know, e.g., which cluster a given "up-down" synchronization column belongs
        to when cycles need to be extracted per cluster later on (see `ExtractCycles`).
        The unified time column itself maps to `None` since it no longer belongs to a
        single cluster after synchronization.
    '''
    file_configs = file_configs or {}
    binary_cols = tuple(binary_cols)

    # --- STEP 1: Intra-cluster merge -------------------------------------------------------
    # Combine every file inside a cluster into one per-cluster DataFrame sharing a single
    # Time axis. This is where the dynamic, per-file loading configuration is applied.
    cluster_dfs = {
        name: merge_cluster_files(paths, time_col=time_col, file_configs=file_configs,
                                   binary_cols=binary_cols)
        for name, paths in clusters_files.items()
    }

    cluster_names = list(cluster_dfs.keys())
    if not cluster_names:
        raise ValueError("clusters_files is empty; at least one cluster is required.")

    if reference_cluster is None:
        reference_cluster = cluster_names[0]
    if reference_cluster not in cluster_dfs:
        raise ValueError(f"reference_cluster '{reference_cluster}' not found among {cluster_names}.")

    # --- STEP 2: Edge-align every cluster to a common physical window ----------------------
    # Clip each cluster to its own enable-signal window, then drift-correct (time-stretch)
    # every non-reference cluster so its total duration exactly matches the reference
    # cluster's duration (compensates for independent hardware clock drift between clusters).
    ref_df = _clip_to_enable_window(cluster_dfs[reference_cluster], enable_signals[reference_cluster], time_col)
    ref_duration = ref_df[time_col].iloc[-1] - ref_df[time_col].iloc[0]

    aligned_dfs = {reference_cluster: ref_df}
    for name in cluster_names:
        if name == reference_cluster:
            continue
        df = _clip_to_enable_window(cluster_dfs[name], enable_signals[name], time_col)
        duration = df[time_col].iloc[-1] - df[time_col].iloc[0]
        df = df.copy()
        df[time_col] = df[time_col] * (ref_duration / duration)
        aligned_dfs[name] = df

    # --- STEP 3: Resample every cluster onto the highest sampling-rate cluster's time base --
    ordered_dfs = [aligned_dfs[name] for name in cluster_names]
    synced = synchronize_dataframes(ordered_dfs, time_col=time_col, filter_time=True,
                                     binary_cols=binary_cols, require_common_start=False)

    # Fail fast on cross-cluster column name collisions: pd.concat(axis=1) would silently
    # produce duplicate-named columns and the flat `metadata` dict (one cluster per column
    # *name*) would then only be able to remember the LAST cluster to claim that name,
    # silently corrupting provenance for every earlier cluster. Column names must therefore
    # be unique across clusters (e.g. rename each cluster's raw columns beforehand).
    seen_columns = {}
    for name, df in zip(cluster_names, synced):
        for col in df.columns:
            if col in seen_columns:
                raise ValueError(
                    f"Column '{col}' is present in both cluster '{seen_columns[col]}' and "
                    f"cluster '{name}'. Column names must be unique across clusters so the "
                    f"metadata dictionary can unambiguously track each column's origin."
                )
            seen_columns[col] = name

    # --- STEP 4: Concatenate side-by-side and build the provenance metadata dictionary ------
    metadata = {}
    for name, df in zip(cluster_names, synced):
        for col in df.columns:
            metadata[col] = name

    merged_df = pd.concat(synced, axis=1)
    merged_df.index.name = time_col
    merged_df = merged_df.reset_index()
    metadata[time_col] = None  # The unified time axis no longer "belongs" to one cluster.

    return merged_df, metadata


# %% --------------------------------------------------------------------------
# EXTRACT CYCLES INDEPENDENTLY PER CLUSTER AND RE-ALIGN THEM (GENERALIZED)
# -----------------------------------------------------------------------------

def ExtractCycles(merged_df, metadata, cluster_sync_columns, time_col='Time'):
    '''
    Splits the single, already-synchronized `merged_df` (as returned by
    `synchronize_clusters`) into per-cycle DataFrames, one cycle boundary set per cluster,
    and re-aligns every cluster's slice of a cycle onto a common per-cycle time base before
    stitching all cycles back together vertically.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of `synchronize_clusters`: a single dataframe with a unified time column
        plus every cluster's signals as columns.
    metadata : dict[str, str or None]
        Output of `synchronize_clusters`: maps every column name in `merged_df` back to
        the cluster it came from. Used both to validate `cluster_sync_columns` (guarding
        against a signal being attributed to the wrong cluster now that everything lives
        in one flat DataFrame) and to look up which columns belong to which cluster when
        slicing out a cycle.
    cluster_sync_columns : dict[str, str]
        Maps each cluster name to the name of THAT cluster's own "up-down" cycle-boundary
        signal. A cycle is defined as ending on a transition from 1 to 0 in this signal
        (per the caller's spec); the data is assumed to be clean, so every cycle is
        assumed complete (no partial leading/trailing cycle handling is attempted).
    time_col : str
        Name of the unified time column in `merged_df`.

    Returns
    -------
    cycles_df : pd.DataFrame
        A single DataFrame containing every extracted cycle from every cluster,
        concatenated vertically. Each cycle contributes exactly the same number of
        rows across all clusters (the cluster with fewer points in a given cycle is
        up-sampled by interpolation onto the higher sampling-rate cluster's cycle-local
        time vector - see STEP 3 below).
    cycle_start_indices : list of int
        The integer row index (into `cycles_df`) where each extracted cycle begins.
    '''
    # --- Validate provenance using the metadata dictionary ---------------------------------
    # This is precisely the check that becomes necessary once every cluster's columns have
    # been flattened into a single DataFrame: we can no longer tell just by looking at
    # `merged_df` which cluster a column came from, so every user-supplied sync column is
    # cross-checked against `metadata` before it is trusted.
    for cluster, col in cluster_sync_columns.items():
        if col not in merged_df.columns:
            raise KeyError(f"Sync column '{col}' not found in merged_df.")
        owner = metadata.get(col)
        if owner != cluster:
            raise ValueError(
                f"Sync column '{col}' belongs to cluster '{owner}' according to metadata, "
                f"not '{cluster}' as specified in cluster_sync_columns."
            )

    cluster_names = list(cluster_sync_columns.keys())

    # --- STEP 1: Find cycle boundaries independently, per cluster's own sync column --------
    # A cycle is delimited by consecutive falling edges (1 -> 0): cycle N spans from the row
    # right after cycle N-1's falling edge (or row 0, for the very first cycle) up to and
    # including cycle N's own falling edge.
    cluster_cycles = {}
    for cluster in cluster_names:
        col = cluster_sync_columns[cluster]
        validate_binary_column(merged_df, col)

        falling_edges = merged_df.index[merged_df[col].diff() == -1].tolist()
        if not falling_edges:
            raise ValueError(f"No 1->0 transition found in column '{col}' for cluster '{cluster}'.")

        boundaries = []
        prev_end = 0
        for edge in falling_edges:
            boundaries.append((prev_end, edge))
            prev_end = edge + 1
        cluster_cycles[cluster] = boundaries

    n_cycles_per_cluster = {c: len(b) for c, b in cluster_cycles.items()}
    if len(set(n_cycles_per_cluster.values())) != 1:
        raise ValueError(f"Clusters do not agree on the number of detected cycles: {n_cycles_per_cluster}.")
    n_cycles = next(iter(n_cycles_per_cluster.values()))

    cycle_frames = []
    cycle_start_indices = []
    running_row = 0

    for cycle_idx in range(n_cycles):

        # --- STEP 2: Slice out this cycle's data for every cluster, independently ----------
        per_cluster_slices = {}
        for cluster in cluster_names:
            start, end = cluster_cycles[cluster][cycle_idx]
            cluster_cols = [c for c, owner in metadata.items() if owner == cluster]
            sl = merged_df.loc[start:end, [time_col] + cluster_cols].reset_index(drop=True)
            # Reset time to a cycle-local axis starting at zero so every cluster's slice of
            # this cycle can be compared/interpolated on the same relative time base,
            # regardless of each cluster's absolute (and possibly still slightly drifted)
            # timestamp for this particular cycle.
            sl[time_col] = sl[time_col] - sl[time_col].iloc[0]
            per_cluster_slices[cluster] = sl

        # --- STEP 3: Identify the higher sampling-rate cluster for THIS cycle and use its --
        # --- cycle-local time vector as the interpolation target for every other cluster ---
        reference_cluster = max(per_cluster_slices, key=lambda c: len(per_cluster_slices[c]))
        ref_time = per_cluster_slices[reference_cluster][time_col].to_numpy()

        cycle_df = per_cluster_slices[reference_cluster].copy()

        for cluster, sl in per_cluster_slices.items():
            if cluster == reference_cluster:
                continue

            xp = sl[time_col].to_numpy()
            can_interpolate = len(xp) > 1 and np.all(np.diff(xp) > 0)

            for col in sl.columns:
                if col == time_col:
                    continue
                if can_interpolate:
                    # Up-sample the lower sampling-rate cluster onto the reference cluster's
                    # cycle-local time vector, adding points so both clusters contribute the
                    # exact same number of rows to this cycle.
                    cycle_df[col] = np.interp(ref_time, xp, sl[col].to_numpy())
                else:
                    # Degenerate cycle slice (e.g. a single sample): cannot interpolate
                    # reliably, so the column is left as NaN rather than fabricating data.
                    cycle_df[col] = np.nan

        cycle_start_indices.append(running_row)
        running_row += len(cycle_df)
        cycle_frames.append(cycle_df)

    # --- STEP 4: Concatenate every cycle vertically into one long DataFrame ----------------
    cycles_df = pd.concat(cycle_frames, axis=0, ignore_index=True)

    return cycles_df, cycle_start_indices
