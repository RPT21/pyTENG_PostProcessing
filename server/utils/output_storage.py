"""
Phase 5: centralized, collision-free output storage for CleanData/CycleData.

Historically, per-experiment CleanData/CycleData pickles were written
inside each experiment's own raw-data folder. That clutters the raw
dataset and breaks down for Scenario A (two-folder) experiments, where
there is no single natural folder to hold the result. This module
centralizes both output kinds (plus an internal Phase-3 merge cache and
the postprocessing metadata file) under dedicated directories placed
alongside the experiment set itself - i.e. in the *experiments* root_dir
(the same directory that holds Experiments.xlsx / LoadsDescription.ods,
see data_loading.ExperimentsFolderLoader), not the software's own
installation directory. This keeps every experiment's outputs next to its
source data/metadata regardless of which machine/checkout runs the tool,
and derives a unique, human-readable, collision-free filename per
experiment straight from its resolved identity (see
experiment_path_resolver.ExperimentPathResolver.build_identity).
"""
import hashlib
from pathlib import Path

CLEAN_DATA_DIRNAME = "CleanData"
CYCLE_DATA_DIRNAME = "CycleData"
MERGE_CACHE_SUBDIRNAME = ".merge_cache"
METADATA_SUBDIRNAME = ".metadata"


def _slug(value) -> str:
    """Filesystem-safe fragment: keep alphanumerics/-/_, replace everything else."""
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    text = text.strip("_")
    return text or "unknown"


def build_experiment_slug(identity: dict) -> str:
    """
    Builds a filesystem-safe, human-readable, collision-resistant slug
    from an ExperimentLocation.identity dict.

    A short hash of the *entire* identity is always appended, so that even
    if two experiments happen to share the same readable fields (e.g. two
    Scenario A rows re-using a "daq"/"motor" folder name pair), their
    outputs never overwrite each other in the shared CleanData/CycleData
    directories.
    """
    if not identity:
        raise ValueError("Cannot build an output filename from an empty identity.")

    if "daq" in identity and "motor" in identity:
        readable = f"{_slug(identity['daq'])}__{_slug(identity['motor'])}__{_slug(identity.get('RloadId', ''))}"
    else:
        order = ("TribuId", "SampleIdTriboPos", "SampleIdTriboNeg", "Date", "RloadId")
        readable = "_".join(_slug(identity[k]) for k in order if k in identity)

    digest = hashlib.md5(repr(sorted(identity.items())).encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def clean_data_dir(root_dir) -> Path:
    """The centralized CleanData directory, directly under the experiments root_dir."""
    return Path(root_dir) / CLEAN_DATA_DIRNAME


def cycle_data_dir(root_dir) -> Path:
    """The centralized CycleData directory, directly under the experiments root_dir."""
    return Path(root_dir) / CYCLE_DATA_DIRNAME


def merge_cache_dir(root_dir) -> Path:
    """Internal Phase-3 merge cache, hidden inside CleanData/ so it isn't mistaken for user output."""
    return clean_data_dir(root_dir) / MERGE_CACHE_SUBDIRNAME


def metadata_dir(root_dir) -> Path:
    """Internal per-experiment postprocessing metadata directory, hidden inside CleanData/."""
    return clean_data_dir(root_dir) / METADATA_SUBDIRNAME


def clean_data_path(root_dir, identity: dict) -> Path:
    """Deterministic path for this experiment's CleanData container (Phase 4 output)."""
    return clean_data_dir(root_dir) / f"{build_experiment_slug(identity)}_clean.pkl"


def cycle_data_path(root_dir, identity: dict) -> Path:
    """Deterministic path for this experiment's CycleData container."""
    return cycle_data_dir(root_dir) / f"{build_experiment_slug(identity)}_cycle.pkl"


def merged_cache_path(root_dir, identity: dict) -> Path:
    """
    Phase 3 output cache: the merged/selected raw DataFrame. Regenerated
    whenever the user changes the file selection (file_selection.py), and
    read by CleanDataProcessor as its raw data source, so the raw data is
    still always re-derived from the original files and never hand-edited.
    """
    return merge_cache_dir(root_dir) / f"{build_experiment_slug(identity)}_merged.pkl"


def metadata_path(root_dir, identity: dict) -> Path:
    """
    Per-experiment postprocessing_metadata.json equivalent, centralized
    instead of written inside the (possibly two-folder) raw-data location.
    """
    return metadata_dir(root_dir) / f"{build_experiment_slug(identity)}.json"


def ensure_dirs(root_dir):
    """Creates all centralized output directories for `root_dir` (idempotent)."""
    for directory in (
        clean_data_dir(root_dir),
        cycle_data_dir(root_dir),
        merge_cache_dir(root_dir),
        metadata_dir(root_dir),
    ):
        directory.mkdir(parents=True, exist_ok=True)
