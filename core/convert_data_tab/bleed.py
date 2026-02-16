# core/convert_data_tab/bleedmatrix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


# - - - Modes and channel model - - -
# For 3x3 we assume channels in order: [D, A, F]
# Profile flags allow you to tailor which cross-terms exist in each mode.
# True means that column contributes to the row (measured row depends on true column).
_MODE_PROFILES: Dict[str, Dict[str, bool]] = {
    # Basic HMSD: only FRET channel has bleed from donor and acceptor
    # F_meas = F_true + beta_DF*D_true + beta_AF*A_true
    "hmsd": {"D_from_A": False, "D_from_F": False,
             "A_from_D": False, "A_from_F": False,
             "F_from_D": True,  "F_from_A": True},

    # Generic 3x3 default identical to HMSD unless you change it
    "default3x3": {"D_from_A": False, "D_from_F": False,
                   "A_from_D": False, "A_from_F": False,
                   "F_from_D": True,  "F_from_A": True},
}

def _mode_profile(mode: str) -> Dict[str, bool]:
    mode = (mode or "").lower()
    if mode in _MODE_PROFILES:
        return _MODE_PROFILES[mode]
    # Fallback to default 3x3 behaviour
    return _MODE_PROFILES["default3x3"]


# - - - Data model - - -
@dataclass(frozen=True)
class BetaFactors:
    # Cross-bleed into F from D and A. Extend if you later enable other cross-terms.
    beta_DF: float = 0.0
    beta_AF: float = 0.0
    # Optional additive offset applied to F channel prior to solving
    beta0: float = 0.0


# - - - Core builders - - -
def build_coeff_matrix(
    *,
    mode: str,
    betas: BetaFactors,
) -> np.ndarray:
    """
    Build a 3x3 coefficient matrix C that maps true signals -> measured signals:

        y_meas = C @ x_true

    Channel order is [D, A, F]. Diagonal entries are 1 by definition.
    Off-diagonals are filled according to the mode profile.
    """
    prof = _mode_profile(mode)
    C = np.eye(3, dtype=float)

    if prof["F_from_D"]:
        C[2, 0] = float(betas.beta_DF)
    if prof["F_from_A"]:
        C[2, 1] = float(betas.beta_AF)

    # If you ever enable other cross-terms, add them here:
    # if prof["D_from_A"]: C[0, 1] = ...
    # if prof["D_from_F"]: C[0, 2] = ...
    # if prof["A_from_D"]: C[1, 0] = ...
    # if prof["A_from_F"]: C[1, 2] = ...

    return C


def build_from_manual_or_calib(
    *,
    mode: str,
    manual_beta_DF: Optional[float],
    manual_beta_AF: Optional[float],
    calib_record: Optional[Mapping[str, Any]] = None,
    defaults: Tuple[float, float] = (0.0, 0.0),
    additive_offset: float = 0.0,
) -> Tuple[np.ndarray, BetaFactors]:
    """
    Determine betas using manual overrides first, then calibration record keys
    'beta_DF' and 'beta_AF' if present, finally fall back to defaults.

    Returns (C, betas) where C is 3x3 and respects the selected mode.
    """
    def _pick(key: str, manual: Optional[float], default_val: float) -> float:
        if manual is not None:
            return float(manual)
        if calib_record is not None and key in calib_record and calib_record[key] is not None:
            try:
                return float(calib_record[key])
            except Exception:
                pass
        return float(default_val)

    beta_DF_default, beta_AF_default = defaults
    betas = BetaFactors(
        beta_DF=_pick("beta_DF", manual_beta_DF, beta_DF_default),
        beta_AF=_pick("beta_AF", manual_beta_AF, beta_AF_default),
        beta0=float(additive_offset),
    )
    C = build_coeff_matrix(mode=mode, betas=betas)
    return C, betas


# - - - Normalisation and table helpers - - -
def normalise_columns(C: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Return a copy of C with each column scaled so that its diagonal element becomes 1.
    For the default 3x3 HMSD profile this is already the case.
    """
    C = np.asarray(C, dtype=float).copy()
    for i in range(min(C.shape)):
        d = C[i, i]
        if abs(d) > eps:
            C[:, i] /= d
    return C


def as_display_grid(C: np.ndarray, *, fmt: str = ".4f") -> List[List[str]]:
    """
    Convert a numeric matrix to a 2D list of strings for QTableWidget population.
    """
    C = np.asarray(C, dtype=float)
    return [[format(x, fmt) for x in row] for row in C.tolist()]


# - - - Correction utilities - - -
def correct_signals(
    measured: np.ndarray,
    C: np.ndarray,
    *,
    subtract_offset_on_F: float = 0.0,
    rcond: float = 1e-12,
) -> np.ndarray:
    """
    Solve x_true from y_meas = C @ x_true for last-dimension 3 arrays.

    - Subtracts a constant offset from F channel if provided.
    - Uses a stable pseudo-inverse for the tiny 3x3 system.

    Shapes
      measured: (..., 3)
      C: (3, 3)

    Returns
      x_true with shape (..., 3)
    """
    Y = np.asarray(measured, dtype=float).copy()
    if Y.shape[-1] != 3 or C.shape != (3, 3):
        raise ValueError("Shapes must be measured[...,3] and C[3,3]")

    if subtract_offset_on_F:
        Y[..., 2] = Y[..., 2] - float(subtract_offset_on_F)

    C_pinv = np.linalg.pinv(C, rcond=rcond)
    X = np.tensordot(Y, C_pinv.T, axes=1)  # apply per row
    return X


# - - - High-level convenience - - -
def compute_bleed_for_mode(
    *,
    mode: str,
    manual_beta_DF: Optional[float],
    manual_beta_AF: Optional[float],
    calib_record: Optional[Mapping[str, Any]],
    defaults: Tuple[float, float] = (0.0, 0.0),
    additive_offset: float = 0.0,
) -> Dict[str, Any]:
    """
    One-shot builder used by the GUI:
      - Picks betas from manual/calibration/defaults
      - Builds the 3x3 coefficient matrix for the given mode
      - Produces a display grid

    Returns
      {
        "betas": BetaFactors,
        "C": np.ndarray shape (3,3),
        "C_norm": np.ndarray shape (3,3),
        "grid": List[List[str]]  # formatted for table display
      }
    """
    C, betas = build_from_manual_or_calib(
        mode=mode,
        manual_beta_DF=manual_beta_DF,
        manual_beta_AF=manual_beta_AF,
        calib_record=calib_record,
        defaults=defaults,
        additive_offset=additive_offset,
    )
    Cn = normalise_columns(C)
    grid = as_display_grid(Cn)
    return {"betas": betas, "C": C, "C_norm": Cn, "grid": grid}
