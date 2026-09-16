"""
Shared date <-> folder-token conversion helpers.

Both directions of experiment path handling need the exact same
'DD/MM/YYYY HH:MM:SS' <-> 'DDMMYYYY_HHMMSS' conversion:
  - Folder scanning (data_loading.ExperimentsFolderLoader) reconstructs
    Experiments.xlsx rows from on-disk folder names.
  - Path resolution (experiment_path_resolver.ExperimentPathResolver)
    rebuilds the on-disk folder name from an Experiments.xlsx row.

Keeping a single implementation here (framework-agnostic, no flask/tkinter
imports) avoids the two directions silently drifting apart.
"""
from datetime import datetime


def make_date_token(value) -> str:
    """
    Converts a date value into its 'DDMMYYYY_HHMMSS' folder-name token.

    Args:
        value: Either a 'DD/MM/YYYY HH:MM:SS' string, or any object
            exposing `strftime` (e.g. `datetime`/`pandas.Timestamp`, as
            produced when pandas parses an Excel date column).

    Returns:
        str: The 'DDMMYYYY_HHMMSS' token.

    Raises:
        ValueError: If `value` is empty or cannot be parsed.
    """
    if value is None:
        raise ValueError("Empty date value")

    if hasattr(value, "strftime"):
        return value.strftime("%d%m%Y_%H%M%S")

    normalized = " ".join(str(value).split())
    try:
        dt = datetime.strptime(normalized, "%d/%m/%Y %H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"Unable to parse date value: {value!r}") from exc
    return dt.strftime("%d%m%Y_%H%M%S")


def parse_date_token(token) -> str:
    """Converts a 'DDMMYYYY_HHMMSS' folder-name token back to 'DD/MM/YYYY HH:MM:SS'."""
    if token is None:
        raise ValueError("Empty date token")

    normalized = " ".join(str(token).split())
    try:
        dt = datetime.strptime(normalized, "%d%m%Y_%H%M%S")
    except ValueError as exc:
        raise ValueError(f"Unable to parse date token: {token!r}") from exc
    return dt.strftime("%d/%m/%Y %H:%M:%S")
