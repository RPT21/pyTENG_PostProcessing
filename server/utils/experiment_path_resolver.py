"""
Phase 1: resolves an Experiments.xlsx row back to its on-disk raw-data
folder(s), and derives the naming "identity" used for centralized output
storage (see output_storage.py).

Two mutually exclusive experiment layouts are supported:

  Scenario A (two folders): columns "daq", "motor" and "RloadId" are
      present and non-empty. The experiment's data lives in two separate
      folders, named after the "daq"/"motor" column values, directly under
      `root_dir`. "RloadId" does not affect either folder path, but (like
      Scenario B) is required and is folded into the identity/output-naming
      key, since it is needed for postprocessing (Gain lookup) and to keep
      output files unique when the same daq/motor folder pair is reused
      across different load resistors.

  Scenario B (single folder): "daq"/"motor" are absent, but
      ["TribuId", "SampleIdTriboNeg", "SampleIdTriboPos", "Date", "RloadId"]
      are all present and non-empty. The data lives in a single folder:
          root_dir / TribuId / SampleIdTriboPos-SampleIdTriboNeg / Date-RloadId
      (Pos-Neg order matches the convention already used by
      data_loading.ExperimentsFolderLoader / experiments_preview.py.)

Any row matching neither layout raises InvalidExperimentFormatError.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd

from server.utils.date_tokens import make_date_token

SCENARIO_TWO_FOLDER = "two_folder"
SCENARIO_SINGLE_FOLDER = "single_folder"

TWO_FOLDER_COLUMNS = ("daq", "motor", "RloadId")
SINGLE_FOLDER_COLUMNS = ("TribuId", "SampleIdTriboNeg", "SampleIdTriboPos", "Date", "RloadId")


class InvalidExperimentFormatError(Exception):
    """Raised when an experiment row matches neither the two-folder nor the single-folder layout."""

    def __init__(self, row_index, row):
        present = [c for c in _row_keys(row) if _has_value(row, c)]
        row_desc = f"row {row_index}" if row_index is not None else "row"
        super().__init__(
            f"Invalid experiment format at {row_desc}: expected either "
            f"{TWO_FOLDER_COLUMNS} (two-folder layout) or {SINGLE_FOLDER_COLUMNS} "
            f"(single-folder layout) to be present and non-empty. "
            f"Columns with values found: {present}"
        )
        self.row_index = row_index


@dataclass
class ExperimentLocation:
    """Resolved on-disk location(s) for a single experiment row."""

    scenario: str
    paths: Dict[str, Path]  # {"daq": Path, "motor": Path} or {"data": Path}
    identity: Dict[str, str] = field(default_factory=dict)


def _row_keys(row):
    if isinstance(row, pd.Series):
        return list(row.index)
    if isinstance(row, dict):
        return list(row.keys())
    raise TypeError(f"Unsupported row type: {type(row)!r}")


def _has_value(row, column):
    if hasattr(row, "get"):
        value = row.get(column)
    else:
        try:
            value = row[column]
        except (KeyError, IndexError):
            return False
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _has_all(row, columns):
    return all(_has_value(row, c) for c in columns)


class ExperimentPathResolver:
    """Resolves the raw-data folder(s) for Experiments.xlsx rows under `root_dir`."""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

    def resolve(self, row, row_index=None) -> ExperimentLocation:
        """
        Args:
            row (dict | pandas.Series): A single experiment record.
            row_index: Optional row identifier, used only for error messages.

        Returns:
            ExperimentLocation

        Raises:
            InvalidExperimentFormatError: If `row` matches neither layout.
        """
        if _has_all(row, TWO_FOLDER_COLUMNS):
            return self._resolve_two_folder(row)
        if _has_all(row, SINGLE_FOLDER_COLUMNS):
            return self._resolve_single_folder(row)
        raise InvalidExperimentFormatError(row_index, row)

    def _resolve_two_folder(self, row) -> ExperimentLocation:
        identity = self.build_identity(row)
        paths = {
            "daq": self.root_dir / identity["daq"],
            "motor": self.root_dir / identity["motor"],
        }
        return ExperimentLocation(scenario=SCENARIO_TWO_FOLDER, paths=paths, identity=identity)

    def _resolve_single_folder(self, row) -> ExperimentLocation:
        identity = self.build_identity(row)
        pos_neg = f"{identity['SampleIdTriboPos']}-{identity['SampleIdTriboNeg']}"
        leaf = f"{identity['Date']}-{identity['RloadId']}"
        folder = self.root_dir / identity["TribuId"] / pos_neg / leaf
        return ExperimentLocation(scenario=SCENARIO_SINGLE_FOLDER, paths={"data": folder}, identity=identity)

    @staticmethod
    def build_identity(row, row_index=None) -> Dict[str, str]:
        """
        Extracts the naming/identity fields from a row, independent of
        resolving actual folder paths, so callers that only need a stable,
        collision-resistant key (output_storage naming, metadata lookup,
        figure_generation caching) don't need a `root_dir` or touch the
        filesystem at all.

        Returns:
            dict[str, str]: Either {"daq": ..., "motor": ..., "RloadId": ...}
            or {"TribuId": ..., "SampleIdTriboPos": ..., "SampleIdTriboNeg": ...,
            "Date": ... (as a 'DDMMYYYY_HHMMSS' token), "RloadId": ...}.

        Raises:
            InvalidExperimentFormatError: If `row` matches neither layout.
        """
        if _has_all(row, TWO_FOLDER_COLUMNS):
            def get(col):
                return row.get(col) if hasattr(row, "get") else row[col]

            return {
                "daq": str(get("daq")).strip(),
                "motor": str(get("motor")).strip(),
                "RloadId": str(get("RloadId")).strip(),
            }
        if _has_all(row, SINGLE_FOLDER_COLUMNS):
            def get(col):
                return row.get(col) if hasattr(row, "get") else row[col]

            return {
                "TribuId": str(get("TribuId")).strip(),
                "SampleIdTriboPos": str(get("SampleIdTriboPos")).strip(),
                "SampleIdTriboNeg": str(get("SampleIdTriboNeg")).strip(),
                "Date": make_date_token(get("Date")),
                "RloadId": str(get("RloadId")).strip(),
            }
        raise InvalidExperimentFormatError(row_index, row)
