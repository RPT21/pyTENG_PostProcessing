"""
MergeData window: Phase 2 (file discovery + user selection), Phase 3
(merge trigger) and Phase 3-cluster (Strategy B) view.

Sits between `experiments_preview.open_clean_data` / `open_merge_data` and
`cleaning_filtering_data.cleaning_filtering_data` in the navigation flow:
the user must pick which discovered raw data file(s) to load - and
optionally merge them onto a common time axis, either via a flat "Simple
Merge" (Strategy A) or by splitting them into two synchronized clusters
("Advanced Cluster Merge", Strategy B) - before the CleanData recipe editor
can open, since the filterable columns are only known once this step has
run (Phase 4's "crucial state dependency").
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import render_template, request, redirect, url_for, session, flash

from server.utils.experiment_path_resolver import InvalidExperimentFormatError
from server.utils.file_discovery import discover_files
from server.utils.synchronization_functions import (
    merge_selected_files,
    merge_clusters,
    load_and_prepare_file,
    load_raw_file,
    guess_time_column,
    MergeError,
    DEFAULT_CSV_READ_KWARGS,
)
from server.utils import output_storage
from server.backend.experiments_preview import (
    ExperimentFileStatus,
    PostprocessingMetadataStore,
    reset_downstream_outputs,
)

logger = logging.getLogger(__name__)

MERGE_STRATEGY_SIMPLE = "simple"
MERGE_STRATEGY_CLUSTER = "cluster"



def _parse_csv_kwargs_from_form(path_str, form):
    """
    Reads the per-file "CSV read options" fields the MergeData UI exposes
    for `.csv` files (form field names suffixed "::<path>") and turns them
    back into `pandas.read_csv` keyword arguments.

    Every field defaults to `DEFAULT_CSV_READ_KWARGS`'s value (blank input
    -> default) so the user only needs to type something when a file
    actually deviates from the standard layout (header=0, index_col=False,
    delimiter=',', decimal='.').

    Returns:
        dict: Only the keys that differ from `DEFAULT_CSV_READ_KWARGS` are
        included (matches how `time_columns`/`clusters` are built above),
        so non-CSV files or untouched rows contribute nothing.
    """
    kwargs = {}

    header_raw = (form.get(f"csv_header::{path_str}") or "").strip()
    if header_raw:
        if header_raw.lower() == "none":
            kwargs["header"] = None
        else:
            try:
                kwargs["header"] = int(header_raw)
            except ValueError:
                logger.warning("Ignoring invalid CSV header row '%s' for %s", header_raw, path_str)

    index_col_raw = (form.get(f"csv_index_col::{path_str}") or "").strip()
    if index_col_raw:
        if index_col_raw.lower() == "false":
            kwargs["index_col"] = False
        else:
            try:
                kwargs["index_col"] = int(index_col_raw)
            except ValueError:
                logger.warning("Ignoring invalid CSV index_col '%s' for %s", index_col_raw, path_str)

    delimiter_raw = form.get(f"csv_delimiter::{path_str}")
    if delimiter_raw and delimiter_raw != DEFAULT_CSV_READ_KWARGS["delimiter"]:
        kwargs["delimiter"] = delimiter_raw

    decimal_raw = form.get(f"csv_decimal::{path_str}")
    if decimal_raw and decimal_raw != DEFAULT_CSV_READ_KWARGS["decimal"]:
        kwargs["decimal"] = decimal_raw

    return kwargs


def _peek_columns(path, csv_kwargs=None):
    """
    Best-effort read of a file's column names + auto-detected time column,
    used to pre-populate each file row's time-column dropdown (Requirement 1).

    The detected column defaults to the first column when no name matches a
    known time-column alias (see `guess_time_column`) - the user remains
    responsible for picking the right one via the dropdown if that default
    is wrong.

    Returns (columns, detected_time_column); (None, None) if the file can't
    be read here (e.g. corrupted/locked, or a CSV whose current
    header/delimiter/decimal overrides don't actually match the file) - the
    selection form still renders, it just won't have a dropdown for that
    particular row and falls back to auto-detection at submit time instead.
    """
    try:
        df = load_raw_file(path, csv_kwargs=csv_kwargs)
    except Exception as exc:  # pragma: no cover - defensive, keeps page usable
        logger.warning("Could not peek columns for %s: %s", path, exc)
        return None, None
    columns = [str(c) for c in df.columns]
    detected = guess_time_column(df)
    return columns, (str(detected) if detected is not None else None)


def _render_selection_form(experiment, available_files, scenario, previous=None, http_status=200):
    previous = previous or {}
    previous_time_columns = previous.get("time_columns", {})
    previous_clusters = previous.get("clusters", {})
    previous_csv_kwargs = previous.get("csv_kwargs", {})

    files_context = []
    for f in available_files:
        path_str = str(f.path)
        file_csv_kwargs = previous_csv_kwargs.get(path_str, {})
        columns, detected = _peek_columns(f.path, csv_kwargs=file_csv_kwargs)
        files_context.append({
            **f.to_dict(),
            "folder": str(f.path.parent),
            "filename": f.path.name,
            "is_csv": f.path.suffix.lower() == ".csv",
            "columns": columns or [],
            "detected_time_column": previous_time_columns.get(path_str, detected),
            "cluster": previous_clusters.get(path_str, ""),
            "csv_header": file_csv_kwargs.get("header", DEFAULT_CSV_READ_KWARGS["header"]),
            "csv_index_col": file_csv_kwargs.get("index_col", DEFAULT_CSV_READ_KWARGS["index_col"]),
            "csv_delimiter": file_csv_kwargs.get("delimiter", DEFAULT_CSV_READ_KWARGS["delimiter"]),
            "csv_decimal": file_csv_kwargs.get("decimal", DEFAULT_CSV_READ_KWARGS["decimal"]),
        })

    return render_template(
        "file_selection.html",
        experiment=experiment,
        available_files=files_context,
        scenario=scenario,
        previous_paths=previous.get("paths", []),
        previous_merge=previous.get("merge_enabled", False),
        previous_strategy=previous.get("merge_strategy", MERGE_STRATEGY_SIMPLE),
        previous_reference_cluster=previous.get("reference_cluster", "A"),
        previous_sync_column_a=previous.get("sync_columns", {}).get("A", ""),
        previous_sync_column_b=previous.get("sync_columns", {}).get("B", ""),
    ), http_status


def file_selection():
    """
    GET: discovers files in the experiment's resolved folder(s) and renders
    a selection form (multi-select + "merge" checkbox).

    POST: loads the chosen file(s) - merging them (Phase 3) if more than
    one is selected and merging is enabled - caches the result to disk,
    stores its path in the session, and redirects to the CleanData editor.

    Returns:
        str | werkzeug.wrappers.Response: Rendered selection page, or a
        redirect to the CleanData editor (on success) / experiments
        preview page (if no experiment is selected or its path is invalid).
    """
    experiment = session.get("current_experiment")
    root_dir = session.get("root_dir")
    if not experiment or session.get("editor_mode") != "clean" or not root_dir:
        flash("No experiment selected. Please pick one from the experiments preview.", "error")
        return redirect(url_for("render_experiments_preview"))

    try:
        location = ExperimentFileStatus(root_dir).resolve_location(experiment)
    except InvalidExperimentFormatError as exc:
        flash(str(exc), "error")
        return redirect(url_for("render_experiments_preview"))

    try:
        available_files = discover_files(location)
    except OSError as exc:
        flash(f"Could not scan experiment folder(s): {exc}", "error")
        return redirect(url_for("render_experiments_preview"))

    if not available_files:
        flash("No data files found in the resolved experiment folder(s).", "error")
        return redirect(url_for("render_experiments_preview"))

    if request.method == "POST":
        selected = request.form.getlist("selected_files")
        merge_enabled = request.form.get("merge_enabled") == "on"
        merge_strategy = request.form.get("merge_strategy") or MERGE_STRATEGY_SIMPLE
        reference_cluster = request.form.get("reference_cluster") or "A"

        # Per-file time column overrides (Requirement 1): each row's <select>
        # is named "time_column::<path>" so it can be read back per-file
        # regardless of submission order.
        time_columns = {}
        clusters = {}
        csv_kwargs = {}
        for p in selected:
            chosen = request.form.get(f"time_column::{p}")
            if chosen:
                time_columns[p] = chosen
            cluster = request.form.get(f"cluster::{p}")
            if cluster in ("A", "B"):
                clusters[p] = cluster
            # Per-file CSV read options (header/index_col/delimiter/decimal):
            # defaults to DEFAULT_CSV_READ_KWARGS, user-overridable per file.
            file_csv_kwargs = _parse_csv_kwargs_from_form(p, request.form)
            if file_csv_kwargs:
                csv_kwargs[p] = file_csv_kwargs

        sync_columns = {
            "A": request.form.get("sync_column_A") or "",
            "B": request.form.get("sync_column_B") or "",
        }

        previous_state = {
            "paths": selected, "merge_enabled": merge_enabled, "time_columns": time_columns,
            "merge_strategy": merge_strategy, "clusters": clusters,
            "reference_cluster": reference_cluster, "sync_columns": sync_columns,
            "csv_kwargs": csv_kwargs,
        }

        if not selected:
            flash("Select at least one file to continue.", "error")
            return _render_selection_form(experiment, available_files, location.scenario)

        use_cluster_strategy = len(selected) > 1 and merge_strategy == MERGE_STRATEGY_CLUSTER

        if len(selected) > 1 and not merge_enabled and not use_cluster_strategy:
            flash(
                "Multiple files selected: enable 'Merge selected files' (Simple Merge) or choose "
                "'Advanced Cluster Merge' to combine them onto a common time axis, or select only one file.",
                "error",
            )
            return _render_selection_form(experiment, available_files, location.scenario, previous=previous_state)

        selected_paths = [Path(p) for p in selected]

        try:
            if len(selected_paths) == 1:
                merged_df = load_and_prepare_file(
                    selected_paths[0], time_col=time_columns.get(selected[0]),
                    csv_kwargs=csv_kwargs.get(selected[0]),
                )
            elif use_cluster_strategy:
                cluster_paths = {"A": [], "B": []}
                for p in selected:
                    cluster_key = clusters.get(p)
                    if cluster_key not in ("A", "B"):
                        raise MergeError(
                            f"File '{Path(p).name}' was not assigned to Cluster 1 or Cluster 2. "
                            "Assign every selected file to one of the two clusters."
                        )
                    cluster_paths[cluster_key].append(Path(p))

                if not sync_columns["A"] or not sync_columns["B"]:
                    raise MergeError("Select a boolean sync signal column for both Cluster 1 and Cluster 2.")

                merged_df = merge_clusters(
                    cluster_paths, reference_cluster=reference_cluster,
                    sync_columns=sync_columns, time_columns=time_columns,
                    csv_kwargs=csv_kwargs,
                )
            else:
                merged_df = merge_selected_files(selected_paths, time_columns=time_columns, csv_kwargs=csv_kwargs)
        except MergeError as exc:
            flash(str(exc), "error")
            return _render_selection_form(experiment, available_files, location.scenario, previous=previous_state)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception("Failed to load/merge selected files")
            flash(f"Could not load/merge the selected files: {exc}", "error")
            return _render_selection_form(experiment, available_files, location.scenario, previous=previous_state)

        output_storage.ensure_dirs(root_dir)
        cache_path = output_storage.merged_cache_path(root_dir, location.identity)
        merged_df.to_pickle(cache_path)

        # Regenerating the merged raw data invalidates any CleanData/
        # CycleData previously computed from the old merge: cleaning must
        # always restart from scratch against the newly (re)created merge
        # (see reset_downstream_outputs), rather than silently leaving
        # stale downstream outputs that no longer match the raw data.
        reset_outputs = reset_downstream_outputs(root_dir, location.identity)

        # Phase 4 depends entirely on this: whenever the selection changes,
        # the merged cache (and thus the columns available for filtering)
        # is regenerated from scratch here, and re-pointed to in the
        # session, so cleaning_filtering_data.py never sees stale columns.
        session["merged_data_path"] = str(cache_path)
        session["file_selection"] = {
            "paths": selected, "merge_enabled": merge_enabled, "time_columns": time_columns,
            "merge_strategy": merge_strategy, "clusters": clusters,
            "reference_cluster": reference_cluster, "sync_columns": sync_columns,
            "csv_kwargs": csv_kwargs,
        }
        session.modified = True

        # Persist the same choice to the centralized, on-disk metadata store
        # (Requirement 2: state persistence) so re-opening this experiment
        # later - even in a new session - can auto-skip straight back to
        # CleanData (see experiments_preview.resolve_entry_point) as long as
        # the merged cache file this points to still exists on disk.
        role_by_path = {str(f.path): f.role for f in available_files}
        auto_detected_by_path = {}
        for f in available_files:
            path_str = str(f.path)
            if path_str in time_columns:
                continue  # already have an explicit user choice, no need to re-detect
            _, detected = _peek_columns(f.path, csv_kwargs=csv_kwargs.get(path_str))
            auto_detected_by_path[path_str] = detected

        selected_files_meta = [
            {
                "path": p,
                "role": role_by_path.get(p),
                "time_column": time_columns.get(p) or auto_detected_by_path.get(p),
                "auto_detected": p not in time_columns,
                "csv_kwargs": csv_kwargs.get(p, {}),
            }
            for p in selected
        ]
        PostprocessingMetadataStore(root_dir, location.identity).update({
            "file_selection": {
                "selected_files": selected_files_meta,
                "merge_enabled": merge_enabled,
                "align_start": True,
                "merged_cache_path": str(cache_path),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                # Always written (even when empty) so a later re-merge with a
                # different strategy can't leave a stale "cluster" merge_strategy/
                # cluster_sync_columns behind for CycleData's "is this a 2-cluster
                # dataset?" check (see CyclePeakExtractor.dataset_cluster_context).
                "merge_strategy": merge_strategy if use_cluster_strategy else MERGE_STRATEGY_SIMPLE,
                "cluster_sync_columns": sync_columns if use_cluster_strategy else {},
                # File path (str) -> "A"/"B", so CycleData can classify
                # which of its columns belong to which cluster (see
                # CyclePeakExtractor.dataset_cluster_context).
                "clusters": clusters if use_cluster_strategy else {},
            }
        })

        merged_note = ""
        if len(selected_paths) > 1:
            merged_note = " using Advanced Cluster Merge" if use_cluster_strategy else " and merged them"
        flash(f"Loaded {len(selected_paths)} file(s){merged_note}.", "success")
        if reset_outputs:
            flash(
                f"Existing {' and '.join(reset_outputs)} for this experiment were reset "
                "since the raw data was just re-merged; please redo cleaning from scratch.",
                "warning",
            )
        return redirect(url_for("render_cleaning_filtering_data"))

    previous = session.get("file_selection") or {}
    return _render_selection_form(experiment, available_files, location.scenario, previous=previous)[0]


def reset_file_selection():
    """
    "Change Selected Files" action (Requirement 3): clears the active
    file-selection state and sends the user back to this same page to pick
    new files/time columns.

    Deliberately does NOT go through `experiments_preview.open_clean_data`
    (the only place auto-skip is decided): it redirects straight to
    `render_file_selection`, so the freshly-cleared session state is
    rendered as-is with no risk of being immediately re-skipped back to
    CleanData by stale on-disk metadata.

    Returns:
        werkzeug.wrappers.Response: Redirect to the file-selection page, or
        back to the experiments preview page if no experiment is active.
    """
    if not session.get("current_experiment") or session.get("editor_mode") != "clean":
        flash("No experiment selected.", "error")
        return redirect(url_for("render_experiments_preview"))

    session.pop("merged_data_path", None)
    session.pop("file_selection", None)
    session.modified = True
    return redirect(url_for("render_file_selection"))
