"""
Phase 2: discovers candidate raw data files inside a resolved
ExperimentLocation (see experiment_path_resolver.py), for the user to pick
from in the file-selection UI (server/backend/file_selection.py).
"""
from pathlib import Path

# Mirrors data_loading.VALID_DATA_EXTENSIONS / EXCLUDED_FOLDER_NAMES.
# Duplicated (rather than imported) so this module stays framework-agnostic
# and independent of server.backend (which pulls in flask/tkinter).
VALID_DATA_EXTENSIONS = (".tdms", ".csv", ".xlsx", ".pkl")
EXCLUDED_FOLDER_NAMES = {"CleanData", "CycleData"}


class DiscoveredFile:
    """One selectable data file, tagged with the resolved role it came from
    ('daq'/'motor' for two-folder experiments, 'data' for single-folder)."""

    def __init__(self, path: Path, role: str):
        self.path = Path(path)
        self.role = role

    @property
    def label(self):
        return f"[{self.role}] {self.path.name}"

    def to_dict(self):
        return {"path": str(self.path), "role": self.role, "label": self.label}

    def __repr__(self):
        return f"DiscoveredFile(role={self.role!r}, path={self.path!r})"


def discover_files(location):
    """
    Scans every folder in `location.paths` for files with a valid data
    extension: directly inside the folder, and one level into any
    subfolder (skipping CleanData/CycleData), which covers cases like a
    "daq" folder containing several per-task .pkl files.

    Args:
        location (experiment_path_resolver.ExperimentLocation): Resolved
            folder(s) for one experiment.

    Returns:
        list[DiscoveredFile]: Sorted by role, then filename.

    Raises:
        OSError: If a resolved folder exists but cannot be listed
            (permissions, disconnected network drive, etc.). Folders that
            simply don't exist are silently skipped (surfaced instead as
            "no files found" by the caller).
    """
    discovered = []
    for role, folder in location.paths.items():
        folder = Path(folder)
        if not folder.is_dir():
            continue

        for entry in sorted(folder.iterdir()):
            if entry.is_file() and entry.suffix.lower() in VALID_DATA_EXTENSIONS:
                discovered.append(DiscoveredFile(entry, role))
            elif entry.is_dir() and entry.name not in EXCLUDED_FOLDER_NAMES:
                for sub_entry in sorted(entry.iterdir()):
                    if sub_entry.is_file() and sub_entry.suffix.lower() in VALID_DATA_EXTENSIONS:
                        discovered.append(DiscoveredFile(sub_entry, f"{role}/{entry.name}"))

    return sorted(discovered, key=lambda d: (d.role, d.path.name))
