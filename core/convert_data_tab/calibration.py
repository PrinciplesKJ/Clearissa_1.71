# core/convert_data_tab/load_calib.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import json
import logging
import difflib
import sys
import os

# Add resource_utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from resource_utils import get_data_path, get_config_dir

logger = logging.getLogger(__name__)


# - - - Errors - - -
class CalibrationError(Exception):
    """Base error for calibration handling."""


class CalibrationNotFound(CalibrationError):
    """Raised when a species cannot be found in the calibration records."""


# - - - Paths - - -
def default_calibration_path() -> Path:
    """
    Return the default path to the calibration JSON.

    Uses the user data directory for proper file storage:
    - Windows: C:\\Users\\<username>\\AppData\\Roaming\\Clearissa\\config\\calibration_data.json
    - Linux/macOS: ~/.clearissa/config/calibration_data.json
    """
    return Path(get_data_path("config/calibration_data.json"))


# - - - Loading and validation - - -
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_records(
    records: Any,
    *,
    required_keys: Iterable[str] = ("species",),
) -> List[Dict[str, Any]]:
    """
    Minimal schema check:
    - Top level must be a list
    - Each item must be a dict
    - Each item must contain the required keys (default: 'species')
    """
    if not isinstance(records, list):
        raise CalibrationError("Calibration JSON must be a list of objects")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(records):
        if not isinstance(item, dict):
            raise CalibrationError(f"Record {i} is not an object")
        missing = [k for k in required_keys if k not in item]
        if missing:
            raise CalibrationError(f"Record {i} missing keys: {', '.join(missing)}")
        out.append(item)
    return out


def _normalise_species(s: str) -> str:
    return (s or "").strip().lower()


@lru_cache(maxsize=1)
def load_cached(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Cached loader suitable for repeated access from the GUI.
    Clears automatically if path changes.
    """
    recs = load(path)
    return recs


def ensure_calibration_file_exists(path: Optional[Path] = None) -> Path:
    """
    Ensure calibration file exists, creating an empty template if needed.

    Parameters
    - path: explicit path to the JSON file. If None, uses default_calibration_path().

    Returns
    - Path to the calibration file (created if it didn't exist)

    Notes
    -----
    Creates an empty calibration template on first run with example structure.
    This prevents errors when the application is run for the first time.
    """
    p = Path(path) if path is not None else default_calibration_path()

    if not p.exists():
        _ensure_parent(p)
        # Create empty calibration file with example template
        template = [
            {
                "species": "EXAMPLE_SPECIES",
                "slope": 1.0,
                "intercept": 0.0,
                "notes": "This is an example calibration entry. Replace with your actual calibration data."
            }
        ]
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            logger.info(f"Created calibration template at {p}")
        except Exception as e:
            logger.error(f"Failed to create calibration template: {e}")
            raise CalibrationError(f"Failed to create calibration file: {e}") from e

    return p


def load(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load calibration records from JSON.

    Parameters
    - path: explicit path to the JSON file. If None, uses default_calibration_path().

    Returns
    - List of dict records with at least the 'species' key.

    Notes
    -----
    Automatically creates a template file if it doesn't exist.
    """
    p = Path(path) if path is not None else default_calibration_path()

    # Ensure file exists (creates template if missing)
    ensure_calibration_file_exists(p)

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise CalibrationError(f"Failed to read calibration JSON: {e}") from e

    records = _validate_records(data, required_keys=("species",))
    logger.debug("Loaded %d calibration records from %s", len(records), p)
    return records


# - - - Indexing and lookup - - -
@dataclass(frozen=True)
class CalibrationIndex:
    """Case-insensitive index of calibration records by species name."""
    by_species: Mapping[str, Dict[str, Any]]
    original_names: Mapping[str, str]  # normalized -> original name mapping

    @classmethod
    def build(cls, records: Iterable[Dict[str, Any]]) -> "CalibrationIndex":
        index: Dict[str, Dict[str, Any]] = {}
        originals: Dict[str, str] = {}
        for rec in records:
            original = str(rec.get("species", "")).strip()
            key = _normalise_species(original)
            if not key:
                # Skip nameless entries to avoid collisions
                continue
            # Last-one-wins if duplicates appear
            index[key] = rec
            originals[key] = original
        return cls(by_species=index, original_names=originals)

    def get(self, species: str) -> Optional[Dict[str, Any]]:
        return self.by_species.get(_normalise_species(species))

    def must_get(self, species: str) -> Dict[str, Any]:
        rec = self.get(species)
        if rec is None:
            raise CalibrationNotFound(f"No calibration entry for species '{species}'")
        return rec

    def suggestions(self, species: str, n: int = 5, cutoff: float = 0.6) -> List[str]:
        key = _normalise_species(species)
        choices = list(self.by_species.keys())
        return difflib.get_close_matches(key, choices, n=n, cutoff=cutoff)

    def species_list(self) -> List[str]:
        """Return list of original species names (preserving case), sorted alphabetically."""
        return sorted(self.original_names.values(), key=lambda s: s.lower())


def build_index(records: Iterable[Dict[str, Any]]) -> CalibrationIndex:
    return CalibrationIndex.build(records)


def lookup(
    records_or_index: Iterable[Dict[str, Any]] | CalibrationIndex,
    species_name: str,
    *,
    raise_if_missing: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Look up a calibration record by species name (case-insensitive).

    Parameters
    - records_or_index: either a list of records or a pre-built CalibrationIndex
    - species_name: target species name
    - raise_if_missing: if True, raise CalibrationNotFound on miss;
      otherwise return None

    Returns
    - The record dict or None
    """
    idx = records_or_index if isinstance(records_or_index, CalibrationIndex) else build_index(records_or_index)
    rec = idx.get(species_name)
    if rec is None and raise_if_missing:
        suggestions = idx.suggestions(species_name)
        hint = f" Did you mean: {', '.join(suggestions)}" if suggestions else ""
        raise CalibrationNotFound(f"No calibration entry for species '{species_name}'.{hint}")
    return rec


# - - - Utility - - -
def list_species(records_or_index: Iterable[Dict[str, Any]] | CalibrationIndex) -> List[str]:
    idx = records_or_index if isinstance(records_or_index, CalibrationIndex) else build_index(records_or_index)
    return idx.species_list()
