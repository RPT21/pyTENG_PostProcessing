"""
Phase 2 (file discovery + user selection) and Phase 3 (merge trigger) view.

Sits between `experiments_preview.open_clean_data` and
`cleaning_filtering_data.cleaning_filtering_data` in the navigation flow:
the user must pick which discovered raw data file(s) to load - and
optionally merge them onto a common time axis - before the CleanData
recipe editor can open, since the filterable columns are only known once
this step has run (Phase 4's "crucial state dependency").
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import render_template, request, redirect, url_for, session, flash

from server.utils.experiment_path_resolver import InvalidExperimentFormatError
from server.utils.file_discovery import discover_files
from server.utils.timeseries_merge import (
    merge_selected_files,
    load_and_prepare_file,
    load_raw_file,
    guess_time_column,
    MergeError,
)
from server.utils import output_storage
from server.backend.experiments_preview import ExperimentFileStatus, PostprocessingMetadataStore

logger = logging.getLogger(__name__)


def _peek_columns(path):
    """
    Best-effort read of a file's column names + auto-detected time column,
    used to pre-populate each file row's time-column dropdown (Requirement 1).

    The detected column defaults to the first column when no name matches a
    known time-column alias (see `guess_time_column`) - the user remains
    responsible for picking the right one via the dropdown if that default
    is wrong.

    Returns (columns, detected_time_column); (None, None) if the file can't
    be read here (e.g. corrupted/locked) - the selection form still renders,
    it just won't have a dropdown for that particular row and falls back to
    auto-detection at submit time instead.
    """
    try:
        df = load_raw_file(path)
    except Exception as exc:  # pragma: no cover - defensive, keeps page usable
        logger.warning("Could not peek columns for %s: %s", path, exc)
        return None, None
    columns = [str(c) for c in df.columns]
    detected = guess_time_column(df)
    return columns, (str(detected) if detected is not None else None)


def _render_selection_form(experiment, available_files, scenario, previous=None, http_status=200):
    previous = previous or {}
    previous_time_columns = previous.get("time_columns", {})

    files_context = []
    for f in available_files:
        path_str = str(f.path)
        columns, detected = _peek_columns(f.path)
        files_context.append({
            **f.to_dict(),
            "folder": str(f.path.parent),
            "filename": f.path.name,
            "columns": columns or [],
            "detected_time_column": previous_time_columns.get(path_str, detected),
        })

    return render_template(
        "file_selection.html",
        experiment=experiment,
        available_files=files_context,
        scenario=scenario,
        previous_paths=previous.get("paths", []),
        previous_merge=previous.get("merge_enabled", False),
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

        # Per-file time column overrides (Requirement 1): each row's <select>
        # is named "time_column::<path>" so it can be read back per-file
        # regardless of submission order.
        time_columns = {}
        for p in selected:
            chosen = request.form.get(f"time_column::{p}")
            if chosen:
                time_columns[p] = chosen

        if not selected:
            flash("Select at least one file to continue.", "error")
            return _render_selection_form(experiment, available_files, location.scenario)

        if len(selected) > 1 and not merge_enabled:
            flash(
                "Multiple files selected: enable 'Merge selected files' to combine "
                "them onto a common time axis, or select only one file.",
                "error",
            )
            return _render_selection_form(
                experiment, available_files, location.scenario,
                previous={"paths": selected, "merge_enabled": merge_enabled, "time_columns": time_columns},
            )

        selected_paths = [Path(p) for p in selected]

        try:
            if len(selected_paths) == 1:
                merged_df = load_and_prepare_file(
                    selected_paths[0], time_col=time_columns.get(selected[0])
                )
            else:
                merged_df = merge_selected_files(selected_paths, time_columns=time_columns)
        except MergeError as exc:
            flash(str(exc), "error")
            return _render_selection_form(
                experiment, available_files, location.scenario,
                previous={"paths": selected, "merge_enabled": merge_enabled, "time_columns": time_columns},
            )
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception("Failed to load/merge selected files")
            flash(f"Could not load/merge the selected files: {exc}", "error")
            return _render_selection_form(
                experiment, available_files, location.scenario,
                previous={"paths": selected, "merge_enabled": merge_enabled, "time_columns": time_columns},
            )

        output_storage.ensure_dirs(root_dir)
        cache_path = output_storage.merged_cache_path(root_dir, location.identity)
        merged_df.to_pickle(cache_path)

        # Phase 4 depends entirely on this: whenever the selection changes,
        # the merged cache (and thus the columns available for filtering)
        # is regenerated from scratch here, and re-pointed to in the
        # session, so cleaning_filtering_data.py never sees stale columns.
        session["merged_data_path"] = str(cache_path)
        session["file_selection"] = {
            "paths": selected, "merge_enabled": merge_enabled, "time_columns": time_columns,
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
            _, detected = _peek_columns(f.path)
            auto_detected_by_path[path_str] = detected

        selected_files_meta = [
            {
                "path": p,
                "role": role_by_path.get(p),
                "time_column": time_columns.get(p) or auto_detected_by_path.get(p),
                "auto_detected": p not in time_columns,
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
            }
        })

        merged_note = " and merged them" if merge_enabled and len(selected_paths) > 1 else ""
        flash(f"Loaded {len(selected_paths)} file(s){merged_note}.", "success")
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
