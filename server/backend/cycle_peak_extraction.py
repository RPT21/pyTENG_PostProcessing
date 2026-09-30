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

from server.backend.experiments_preview import (
    resolve_experiment_identity, record_postprocessing_metadata, PostprocessingMetadataStore,
)
from server.utils.experiment_path_resolver import InvalidExperimentFormatError
from server.utils import output_storage

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
MAX_CHART_POINTS = 5000

# Cycle Extraction (independent of peak extraction - see CycleExtractionConfig).
CYCLE_MODE_THRESHOLD = "threshold"
CYCLE_MODE_CLUSTER_SYNC = "cluster_sync"
SUPPORTED_CYCLE_MODES = (CYCLE_MODE_THRESHOLD, CYCLE_MODE_CLUSTER_SYNC)

# Peak Extraction (independent of cycle extraction - see CycleExtractionConfig).
PEAK_METHOD_FIND_PEAKS = "find_peaks"
PEAK_METHOD_CYCLE_MINMAX = "cycle_minmax"
SUPPORTED_PEAK_METHODS = (PEAK_METHOD_FIND_PEAKS, PEAK_METHOD_CYCLE_MINMAX)


class CycleExtractionError(Exception):
    """Raised when a cycle/peak extraction configuration is invalid or cannot be executed."""


# ----------------------------------------------------------------------
# Configuration data model
# ----------------------------------------------------------------------
@dataclass
class CycleExtractionConfig:
    """
    Serializable configuration for one cycle/peak extraction run.

    Cycle extraction and peak extraction are two fully independent
    sub-configurations (`cycle_mode`/`cycle_params` vs.
    `peak_method`/`peak_params`): peak detection no longer depends on how
    (or whether) cycles were extracted.

    `cycle_mode` (Optional[str]):
        None: cycle extraction is skipped entirely (peak extraction can
            still run standalone via PEAK_METHOD_FIND_PEAKS).
        CYCLE_MODE_THRESHOLD: `cycle_params` = {threshold, edge
            ('rising'/'falling'/'both'), hysteresis}.
        CYCLE_MODE_CLUSTER_SYNC: `cycle_params` =
            {cluster_sync_column_a, cluster_sync_column_b}. Only valid for
            datasets merged via MergeData's Advanced Cluster Merge (2
            clusters) - enforced both in the UI and in
            CyclePeakExtractor.run().

    `peak_method` (str):
        PEAK_METHOD_FIND_PEAKS (default): `peak_params` = {height,
            distance, prominence, width, prefilter_cutoff, fs}.
        PEAK_METHOD_CYCLE_MINMAX: no params; requires `cycle_mode` to be
            set and to have produced at least one cycle.
    """

    signal_column: str
    time_column: Optional[str] = None

    cycle_mode: Optional[str] = None
    cycle_params: dict = field(default_factory=dict)

    peak_method: str = PEAK_METHOD_FIND_PEAKS
    peak_params: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        if not data.get("signal_column"):
            raise CycleExtractionError(f"Malformed cycle extraction config: {data}")
        if not data.get("time_column"):
            raise CycleExtractionError("A Time Column must be selected.")

        cycle_mode = data.get("cycle_mode") or None
        if cycle_mode is not None and cycle_mode not in SUPPORTED_CYCLE_MODES:
            raise CycleExtractionError(f"Unsupported cycle extraction mode: '{cycle_mode}'")

        peak_method = data.get("peak_method") or PEAK_METHOD_FIND_PEAKS
        if peak_method not in SUPPORTED_PEAK_METHODS:
            raise CycleExtractionError(f"Unsupported peak extraction method: '{peak_method}'")

        return cls(
            signal_column=data["signal_column"],
            time_column=data.get("time_column") or None,
            cycle_mode=cycle_mode,
            cycle_params=data.get("cycle_params") or {},
            peak_method=peak_method,
            peak_params=data.get("peak_params") or {},
        )


# ----------------------------------------------------------------------
# Extraction service
# ----------------------------------------------------------------------
class CyclePeakExtractor:
    """
    Loads a CleanData container and performs cycle segmentation and
    peak/trough extraction according to a `CycleExtractionConfig`.

    Cycle extraction (`cycle_mode`) and peak extraction (`peak_method`)
    are fully independent pipelines that only share the loaded CleanData
    table and the selected primary `signal_column`/`time_column` - neither
    depends on the other's outcome, except that PEAK_METHOD_CYCLE_MINMAX
    requires cycles to have been produced by `cycle_mode` in the same run.
    """

    def __init__(self, root_dir, identity):
        self.root_dir = root_dir
        self.identity = identity

    @property
    def clean_data_path(self):
        return output_storage.clean_data_path(self.root_dir, self.identity)

    @property
    def cycle_data_path(self):
        """Deterministic target path for the saved CycleData container."""
        return output_storage.cycle_data_path(self.root_dir, self.identity)

    # ------------------------------------------------------------------
    # CleanData loading
    # ------------------------------------------------------------------
    def load_clean_data(self):
        """Loads the processed CleanData DataFrame (raw + filtered columns)."""
        path = self.clean_data_path
        if not path.is_file():
            raise CycleExtractionError(f"CleanData file not found: {path}. Create CleanData first.")

        with open(path, "rb") as f:
            container = pickle.load(f)

        df = container.get("data")
        if not isinstance(df, pd.DataFrame):
            raise CycleExtractionError(f"CleanData file is missing its processed data table: {path}")
        return df

    def dataset_cluster_context(self, df):
        """
        Determines whether this experiment's raw data was produced by
        MergeData's Advanced Cluster Merge (2 clusters), by reading the
        persistent postprocessing metadata written by
        server.backend.file_selection at merge time.

        This is what gates CYCLE_MODE_CLUSTER_SYNC's visibility in the UI
        (and is re-checked here server-side as defense in depth): "Cluster
        Synchronization" only makes sense for a dataset that was actually
        assembled from two synchronized clusters. It also reports, per
        cluster, which of `df`'s columns actually originated from that
        cluster's file(s) - so the UI can restrict each cluster's "Cycle
        Boolean Signal" dropdown to only the signals that belong to it
        (a column from Cluster 1's own file(s) would otherwise be a
        nonsensical choice for Cluster 2's sync signal, and vice versa).

        Returns:
            dict: {"has_two_clusters": bool, "cluster_sync_column_a": str
            | None, "cluster_sync_column_b": str | None,
            "cluster_a_columns": list[str], "cluster_b_columns": list[str]}
            - the sync column names are already namespaced as
            '<file_stem>::<original_col>', matching this CleanData table's
            own column names; the two column lists contain every column
            (namespaced or not) whose source file was assigned to that
            cluster at merge time.
        """
        metadata = PostprocessingMetadataStore(self.root_dir, self.identity).load() or {}
        file_selection_meta = metadata.get("file_selection") or {}
        sync_columns = file_selection_meta.get("cluster_sync_columns") or {}
        col_a, col_b = sync_columns.get("A"), sync_columns.get("B")

        has_two_clusters = bool(
            file_selection_meta.get("merge_strategy") == "cluster"
            and col_a and col_b and col_a in df.columns and col_b in df.columns
        )

        cluster_a_columns, cluster_b_columns = [], []
        if has_two_clusters:
            # "clusters" maps each selected file's path string to "A"/"B"
            # (see file_selection.py); columns are namespaced by their
            # source file's stem ('<file_stem>::<original_col>'), so a
            # stem -> cluster lookup is enough to classify every column.
            clusters = file_selection_meta.get("clusters") or {}
            stem_to_cluster = {Path(p).stem: cluster for p, cluster in clusters.items()}
            for col in df.columns:
                col_str = str(col)
                stem = col_str.split("::", 1)[0] if "::" in col_str else None
                cluster = stem_to_cluster.get(stem)
                if cluster == "A":
                    cluster_a_columns.append(col)
                elif cluster == "B":
                    cluster_b_columns.append(col)

        return {
            "has_two_clusters": has_two_clusters,
            "cluster_sync_column_a": col_a if has_two_clusters else None,
            "cluster_sync_column_b": col_b if has_two_clusters else None,
            "cluster_a_columns": cluster_a_columns,
            "cluster_b_columns": cluster_b_columns,
        }

    @staticmethod
    def numeric_columns(df):
        """Returns the names of all numerical columns in `df`."""
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    @staticmethod
    def boolean_like_columns(df):
        """Columns that are boolean dtype, or numeric with only {0, 1} values (candidate Cycle Boolean Signals)."""
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
    # Cycle Extraction (independent of Peak Extraction)
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

    def _extract_cycles_threshold(self, values, params):
        """
        Method A: Threshold-based. Finds the intersection points (cut-offs)
        between `values` and a user-defined Y-threshold; each intersection
        marks the boundary between two consecutive cycles. `edge` selects
        which crossing direction(s) count as a boundary, and `hysteresis`
        adds a dead-band around the threshold to avoid chatter from noisy
        signals sitting right at the threshold value.
        """
        threshold = params.get("threshold")
        if threshold is None or threshold == "":
            raise CycleExtractionError("A threshold value is required for threshold-based cycle extraction.")
        hysteresis = float(params.get("hysteresis") or 0.0)
        edge = params.get("edge", "rising")

        rising, falling = self._hysteresis_crossings(values, float(threshold), hysteresis)
        markers = sorted(rising + falling) if edge == "both" else (rising if edge == "rising" else falling)

        cycles = self._markers_to_cycles(markers, len(values))
        if not cycles:
            raise CycleExtractionError("No threshold crossings found; try a different threshold/hysteresis.")
        return cycles

    def _extract_cycles_cluster_sync(self, df, params, context, n_samples):
        """
        Method B: Cluster Synchronization. Mirrors
        synchronization_functions.ExtractCycles: a cycle ends on a
        transition from 1 to 0 in each cluster's own boolean "Cycle
        Boolean Signal", and cycles from both clusters are extracted
        independently before being reconciled into one boundary set.

        Unlike the standalone (pre-synchronization) ExtractCycles utility,
        both boolean columns here already live on the SAME sample grid -
        MergeData's Advanced Cluster Merge already resampled Cluster 1 and
        Cluster 2 onto one common time axis before this data ever reached
        CleanData. So each cluster's independently-detected falling edge
        should already coincide almost exactly with the other's; any
        residual 1-2 sample disagreement (e.g. from interpolation
        rounding) is reconciled by simply averaging the two edge estimates
        - a lightweight, single-time-axis stand-in for the fuller
        per-column interpolation the standalone utility has to perform.
        """
        col_a = params.get("cluster_sync_column_a") or context.get("cluster_sync_column_a")
        col_b = params.get("cluster_sync_column_b") or context.get("cluster_sync_column_b")
        if not col_a or not col_b:
            raise CycleExtractionError("Select a Cycle Boolean Signal for both Cluster 1 and Cluster 2.")
        if col_a not in df.columns:
            raise CycleExtractionError(f"Cluster 1's Cycle Boolean Signal '{col_a}' was not found in CleanData.")
        if col_b not in df.columns:
            raise CycleExtractionError(f"Cluster 2's Cycle Boolean Signal '{col_b}' was not found in CleanData.")
        if col_a == col_b:
            raise CycleExtractionError("Cluster 1 and Cluster 2 must use different Cycle Boolean Signal columns.")

        # Defense in depth: the UI already restricts each dropdown to only
        # that cluster's own columns, but a stale/hand-crafted request
        # could still swap them - reject it rather than silently comparing
        # a cluster to itself or to the wrong cluster's signal.
        cluster_a_columns = context.get("cluster_a_columns") or []
        cluster_b_columns = context.get("cluster_b_columns") or []
        if cluster_a_columns and col_a not in cluster_a_columns:
            raise CycleExtractionError(f"'{col_a}' does not belong to Cluster 1's source file(s).")
        if cluster_b_columns and col_b not in cluster_b_columns:
            raise CycleExtractionError(f"'{col_b}' does not belong to Cluster 2's source file(s).")

        values_a = df[col_a].to_numpy(dtype=float)
        values_b = df[col_b].to_numpy(dtype=float)

        _, falling_a = self._hysteresis_crossings(values_a, 0.5, 0.0)
        _, falling_b = self._hysteresis_crossings(values_b, 0.5, 0.0)

        if not falling_a:
            raise CycleExtractionError(f"Cluster 1's Cycle Boolean Signal '{col_a}' never falls (1 -> 0); no cycles found.")
        if not falling_b:
            raise CycleExtractionError(f"Cluster 2's Cycle Boolean Signal '{col_b}' never falls (1 -> 0); no cycles found.")
        if len(falling_a) != len(falling_b):
            raise CycleExtractionError(
                f"Cluster 1 and Cluster 2 do not agree on the number of detected cycles "
                f"({len(falling_a)} vs {len(falling_b)}). Both Cycle Boolean Signals are expected "
                "to already be time-aligned (via MergeData's Advanced Cluster Merge)."
            )

        edges = sorted({int(round((a + b) / 2)) for a, b in zip(falling_a, falling_b)})
        edges = [e for e in edges if 0 <= e < n_samples]

        cycles = []
        prev_end = 0
        for edge in edges:
            if edge > prev_end:
                cycles.append((prev_end, edge))
            prev_end = edge + 1

        if not cycles:
            raise CycleExtractionError("Cluster synchronization segmentation produced no usable cycles.")
        return cycles

    # ------------------------------------------------------------------
    # Peak / Trough Extraction (independent of Cycle Extraction)
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
    def _extract_cycle_minmax_peaks(values, cycles):
        """
        Method A: Cycle Min/Max. Finds the maximum and minimum values
        strictly within the boundaries of each already-extracted cycle.
        Requires `cycles` to be non-empty (enforced by the caller).
        """
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
        Executes cycle extraction and peak/trough extraction for `config`
        against a freshly loaded CleanData table. The two are independent:
        `config.cycle_mode` controls cycle extraction (or skips it
        entirely when None) and `config.peak_method` controls peak
        extraction, regardless of whether cycles were produced - except
        PEAK_METHOD_CYCLE_MINMAX, which requires cycles to exist.

        Returns:
            dict: {"raw_df", "cycles_df", "peak_indices", "trough_indices",
            "warnings"}.
        """
        df = self.load_clean_data()
        warnings_ = []

        if config.signal_column not in df.columns:
            raise CycleExtractionError(f"Signal column '{config.signal_column}' not found in CleanData.")

        values = df[config.signal_column].to_numpy(dtype=float)
        n_samples = len(values)

        time_col = config.time_column
        if time_col and time_col not in df.columns:
            warnings_.append(f"Time column '{time_col}' not found in CleanData; falling back to sample index.")
            time_col = None

        # --- Cycle Extraction (independent of Peak Extraction) ---
        cycles = []
        if config.cycle_mode == CYCLE_MODE_THRESHOLD:
            cycles = self._extract_cycles_threshold(values, config.cycle_params)
        elif config.cycle_mode == CYCLE_MODE_CLUSTER_SYNC:
            context = self.dataset_cluster_context(df)
            if not context["has_two_clusters"]:
                raise CycleExtractionError(
                    "Cluster Synchronization is only available for datasets merged with MergeData's "
                    "Advanced Cluster Merge (2 clusters)."
                )
            cycles = self._extract_cycles_cluster_sync(df, config.cycle_params, context, n_samples)
        # config.cycle_mode is None: cycle extraction skipped entirely.

        # --- Peak/Trough Extraction (independent of Cycle Extraction) ---
        if config.peak_method == PEAK_METHOD_CYCLE_MINMAX:
            if not cycles:
                raise CycleExtractionError(
                    "Cycle Min/Max peak extraction requires cycles to be extracted first."
                )
            peak_indices, trough_indices = self._extract_cycle_minmax_peaks(values, cycles)
        else:  # PEAK_METHOD_FIND_PEAKS
            peak_indices, trough_indices = self._extract_find_peaks(values, config.peak_params)
            if len(peak_indices) == 0:
                warnings_.append("No peaks were detected with the current find_peaks parameters.")
            if len(trough_indices) == 0:
                warnings_.append("No troughs were detected with the current find_peaks parameters.")

        if cycles:
            cycles_df = self._build_cycle_table(
                df, time_col, config.signal_column, cycles, peak_indices, trough_indices, warnings_
            )
        else:
            cycles_df = pd.DataFrame(columns=[
                "cycle_number", "start_idx", "end_idx", "start_time", "end_time", "duration",
                "peak_idx", "peak_value", "peak_time", "trough_idx", "trough_value", "trough_time",
                "peak_to_peak",
            ])

        # Peak/trough (and cycle boundary) markers must be positioned using
        # this FULL-resolution table's own time/value columns, never by
        # indexing into the chart preview's data (see
        # _dataframe_to_chart_json): that payload is uniformly downsampled
        # for large signals, so a raw sample index from here would land on
        # the wrong point (or out of range) once looked up in a shorter,
        # downsampled array - visually appearing as if markers/lines were
        # placed "by index" instead of at their real time position.
        peak_times, peak_values = self._times_and_values(df, config.signal_column, time_col, peak_indices)
        trough_times, trough_values = self._times_and_values(df, config.signal_column, time_col, trough_indices)

        return {
            "raw_df": df,
            "cycles_df": cycles_df,
            "peak_indices": peak_indices,
            "trough_indices": trough_indices,
            "peak_times": peak_times,
            "peak_values": peak_values,
            "trough_times": trough_times,
            "trough_values": trough_values,
            "warnings": warnings_,
        }

    @staticmethod
    def _times_and_values(df, signal_col, time_col, indices):
        """
        Resolves (time, value) pairs for a set of full-resolution sample
        indices directly from `df`, so callers never need to look them up
        in a possibly-downsampled chart payload. Falls back to the raw
        sample index itself as the "time" value when no time column is
        selected (consistent with getTimeAxis's own index-based fallback
        on the front-end).
        """
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return [], []
        values = df[signal_col].to_numpy(dtype=float)[indices].tolist()
        if time_col:
            times = df[time_col].to_numpy(dtype=float)[indices].tolist()
        else:
            times = indices.tolist()
        return times, values

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, config):
        """
        Runs the pipeline and writes the result to the centralized
        `CycleData/` directory (Phase 5), alongside the serialized
        configuration metadata, so it can be re-opened for editing later.

        Returns:
            tuple[pathlib.Path, dict]: The saved file path and the `run()` result.
        """
        result = self.run(config)

        self.cycle_data_path.parent.mkdir(parents=True, exist_ok=True)

        container = {
            "cycles": result["cycles_df"],
            "peak_indices": [int(i) for i in result["peak_indices"]],
            "trough_indices": [int(i) for i in result["trough_indices"]],
            "peak_times": result["peak_times"],
            "peak_values": result["peak_values"],
            "trough_times": result["trough_times"],
            "trough_values": result["trough_values"],
            "statistics": _build_statistics(
                result["cycles_df"], result["peak_values"], result["trough_values"]
            ),
            "config": config.to_dict(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

        with open(self.cycle_data_path, "wb") as f:
            pickle.dump(container, f)

        return self.cycle_data_path, result

    def find_existing_cycle_file(self):
        """Returns this experiment's CycleData file if it already exists, or None.

        The filename is deterministic (derived from the experiment's
        identity), so no directory scanning is needed even though
        CycleData/ is shared by every experiment.
        """
        return self.cycle_data_path if self.cycle_data_path.is_file() else None

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
        if path is None:
            return None

        with open(path, "rb") as f:
            container = pickle.load(f)

        return {
            "path": path,
            "cycles": container.get("cycles"),
            "peak_indices": container.get("peak_indices", []),
            "trough_indices": container.get("trough_indices", []),
            # Older CycleData files (saved before time-based markers were
            # introduced) won't have these keys; default to [] so the
            # front-end simply skips drawing peak/trough markers for them
            # instead of failing to load.
            "peak_times": container.get("peak_times", []),
            "peak_values": container.get("peak_values", []),
            "trough_times": container.get("trough_times", []),
            "trough_values": container.get("trough_values", []),
            # Older CycleData files (saved before statistics persistence
            # was added) won't have this key; recompute it on the fly so
            # the UI's Statistics box still reflects reality.
            "statistics": container.get("statistics") or _build_statistics(
                container.get("cycles"), container.get("peak_values", []), container.get("trough_values", [])
            ),
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

    root_dir = session.get("root_dir")
    if not root_dir:
        flash("No experiments folder selected. Please select a root folder first.", "error")
        return redirect(url_for("render_data_loading"))

    try:
        identity = resolve_experiment_identity(experiment)
    except InvalidExperimentFormatError as exc:
        flash(str(exc), "error")
        return redirect(url_for("render_experiments_preview"))

    extractor = CyclePeakExtractor(root_dir, identity)

    try:
        df = extractor.load_clean_data()
    except CycleExtractionError as exc:
        flash(str(exc), "error")
        return redirect(url_for("render_experiments_preview"))

    numeric_columns = extractor.numeric_columns(df)
    boolean_columns = extractor.boolean_like_columns(df)
    default_time_column = extractor.guess_time_column(df)
    cluster_context = extractor.dataset_cluster_context(df)

    existing = None
    try:
        existing = extractor.load_existing()
    except Exception:  # pragma: no cover - defensive catch-all
        logger.exception("Could not load existing CycleData for %s", identity)
        flash("Existing CycleData file could not be read; starting a fresh configuration.", "warning")

    cycles_json = []
    peak_indices, trough_indices = [], []
    peak_times, peak_values, trough_times, trough_values = [], [], [], []
    extra_indices = set()

    if existing:
        config = existing["config"]
        if existing["saved_at"]:
            flash(f"Loaded existing CycleData configuration (saved at {existing['saved_at']}).", "success")

        # Re-hydrate the saved cycles/peaks so the chart is drawn exactly as
        # it was left, without requiring the user to press "Detect" again.
        cycles_json = _cycles_to_json(existing["cycles"])
        peak_indices = [int(i) for i in existing["peak_indices"]]
        trough_indices = [int(i) for i in existing["trough_indices"]]
        peak_times = existing["peak_times"]
        peak_values = existing["peak_values"]
        trough_times = existing["trough_times"]
        trough_values = existing["trough_values"]

        extra_indices = set(peak_indices) | set(trough_indices)
        for cycle in cycles_json:
            extra_indices.add(cycle["start_idx"])
            extra_indices.add(cycle["end_idx"])
    else:
        config = CycleExtractionConfig(
            signal_column=numeric_columns[0] if numeric_columns else "",
            time_column=default_time_column,
            cycle_mode=None,
            cycle_params={},
            peak_method=PEAK_METHOD_FIND_PEAKS,
            peak_params={},
        )

    return render_template(
        "cycle_peak_extraction.html",
        experiment=experiment,
        numeric_columns_json=json.dumps(numeric_columns),
        boolean_columns_json=json.dumps(boolean_columns),
        default_time_column_json=json.dumps(default_time_column),
        config_json=json.dumps(config.to_dict()),
        chart_data_json=json.dumps(_dataframe_to_chart_json(df, extra_indices=extra_indices)),
        cycles_json=json.dumps(cycles_json),
        peak_indices_json=json.dumps(peak_indices),
        trough_indices_json=json.dumps(trough_indices),
        peak_times_json=json.dumps(peak_times),
        peak_values_json=json.dumps(peak_values),
        trough_times_json=json.dumps(trough_times),
        trough_values_json=json.dumps(trough_values),
        has_existing=existing is not None,
        cluster_context_json=json.dumps(cluster_context),
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
    root_dir = session.get("root_dir")
    if not experiment or not root_dir:
        return jsonify({"error": "No experiment selected."}), 400

    try:
        config = CycleExtractionConfig.from_dict(request.get_json(force=True, silent=True) or {})
    except CycleExtractionError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        identity = resolve_experiment_identity(experiment)
    except InvalidExperimentFormatError as exc:
        return jsonify({"error": str(exc)}), 400

    extractor = CyclePeakExtractor(root_dir, identity)
    try:
        result = extractor.run(config)
    except CycleExtractionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while detecting cycles/peaks")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    cycle_boundary_indices = []
    if result["cycles_df"] is not None and not result["cycles_df"].empty:
        cycle_boundary_indices = (
            result["cycles_df"]["start_idx"].tolist() + result["cycles_df"]["end_idx"].tolist()
        )
    extra_indices = set(result["peak_indices"]) | set(result["trough_indices"]) | set(cycle_boundary_indices)

    return jsonify(
        {
            "chart_data": _dataframe_to_chart_json(result["raw_df"], extra_indices=extra_indices),
            "cycles": _cycles_to_json(result["cycles_df"]),
            "peak_indices": [int(i) for i in result["peak_indices"]],
            "trough_indices": [int(i) for i in result["trough_indices"]],
            "peak_times": result["peak_times"],
            "peak_values": result["peak_values"],
            "trough_times": result["trough_times"],
            "trough_values": result["trough_values"],
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
    root_dir = session.get("root_dir")
    if not experiment or not root_dir:
        return jsonify({"error": "No experiment selected."}), 400

    try:
        config = CycleExtractionConfig.from_dict(request.get_json(force=True, silent=True) or {})
    except CycleExtractionError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        identity = resolve_experiment_identity(experiment)
    except InvalidExperimentFormatError as exc:
        return jsonify({"error": str(exc)}), 400

    extractor = CyclePeakExtractor(root_dir, identity)
    try:
        cycle_path, result = extractor.save(config)
    except CycleExtractionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error while saving CycleData")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    record_postprocessing_metadata(
        root_dir,
        identity,
        "cycle_data",
        {
            "config": config.to_dict(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "n_cycles": len(result["cycles_df"]),
            "warnings": result["warnings"],
        },
    )

    flash(f"CycleData saved to {cycle_path} ({len(result['cycles_df'])} cycle(s)).", "success")
    for warn in result["warnings"]:
        flash(warn, "warning")

    return jsonify(
        {
            "message": "CycleData saved successfully.",
            "redirect_url": url_for("render_experiments_preview"),
        }
    )


def _dataframe_to_chart_json(df, max_points=MAX_CHART_POINTS, extra_indices=None):
    """
    Converts a DataFrame's numeric columns into a JSON-friendly dict,
    downsampling uniformly if it has more than `max_points` rows (keeps
    the browser/Plotly responsive for large CleanData files).

    `extra_indices` (optional iterable of full-resolution row positions,
    e.g. peak/trough/cycle-boundary indices) are always kept in the
    sampled output even if they fall between the uniform stride, so those
    markers/lines land exactly ON the plotted signal line instead of
    appearing to "float" near an interpolated straight segment between two
    downsampled points.
    """
    n = len(df)
    step = max(1, n // max_points) if n > max_points else 1
    idx = np.arange(0, n, step)

    if extra_indices:
        extra = np.fromiter(
            (int(i) for i in extra_indices if 0 <= int(i) < n), dtype=int
        )
        if extra.size:
            idx = np.union1d(idx, extra)

    sampled = df.iloc[idx]

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


def _summary_stats(values):
    """
    Computes {count, mean, std, min, max} for a list of numeric values,
    ignoring None/NaN entries. Mirrors the front-end's computeStats() so
    the "Statistics" box always reflects exactly what gets persisted.

    Returns:
        dict | None: None when there is no valid numeric data.
    """
    arr = np.array([v for v in values if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _build_statistics(cycles_df, peak_values, trough_values):
    """
    Builds the same 4 statistics groups shown in the UI's "Statistics" box
    (Cycles/Peaks/Troughs/Peak-to-peak), so they are persisted alongside
    the CycleData file instead of only existing as a client-side
    recomputation.
    """
    if cycles_df is not None and not cycles_df.empty:
        durations = cycles_df["duration"].tolist()
        peak_to_peaks = cycles_df["peak_to_peak"].tolist()
    else:
        durations = []
        peak_to_peaks = []

    return {
        "cycles": _summary_stats(durations),
        "peaks": _summary_stats(peak_values),
        "troughs": _summary_stats(trough_values),
        "peak_to_peak": _summary_stats(peak_to_peaks),
    }

