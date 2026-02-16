# core/convert_data_tab/params.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Iterable, List, Optional


# - - - Errors - - -
class ParamsError(ValueError):
    """Raised when user parameters are invalid."""


# - - - Model - - -
@dataclass(frozen=True)
class ConvertParams:
    # mode of conversion
    mode: str  # "tmsd" | "hmsd" | "mass" | "internal_tmsd" | "normalised_sd" | "hp_quenching"

    # Common timepoints for all modes
    init_start: float  # Start of initialisation window (minutes)
    init_end: float    # End of initialisation window (minutes)
    trigger: float     # Reaction trigger timepoint (minutes)

    # Negative control timepoints (internal_tmsd mode only)
    neg_ctrl_start: Optional[float]  # Start of negative control window (minutes)
    neg_ctrl_end: Optional[float]    # End of negative control window (minutes)

    # Mode-specific injections (optional, validated per mode)
    injection_primary: Optional[float]    # Primary injection time (TMSD: nuking)
    injection_secondary: Optional[float]  # Secondary injection time (reserved for future use)

    # Manual bleed-through factors (None means auto-calculate)
    beta_DF: Optional[float]  # Donor->FRET bleed
    beta_AF: Optional[float]  # Acceptor->FRET bleed
    beta_BF: Optional[float]  # Background->FRET bleed

    # Control concentrations (nM)
    pos_ctrl_concentrations: List[float]      # Positive control concentrations
    donor_ctrl_concentrations: List[float]    # Donor control concentrations
    acceptor_ctrl_concentrations: List[float] # Acceptor control concentrations
    neg_ctrl_concentrations: List[float]      # Negative control concentrations (hp_quenching: HI product)

    # flags
    start_with_donor: bool  # For HMSD/catalytic modes
    c_ref: float = 10.0  # Positive control reference concentration (nM) for normalisation

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


# - - - Parsing helpers - - -
_ALLOWED_MODES = {"tmsd", "hmsd", "mass", "internal_tmsd", "normalised_sd", "hp_quenching"}

# Define which parameters each mode requires
_MODE_REQUIREMENTS = {
    "tmsd": {
        "needs_injection_primary": True,
        "needs_injection_secondary": False,
        "needs_pos_ctrl": True,
        "needs_donor_ctrl": False,
        "needs_acceptor_ctrl": False,
        "needs_neg_ctrl": False,
    },
    "internal_tmsd": {
        "needs_injection_primary": True,
        "needs_injection_secondary": False,
        "needs_pos_ctrl": True,
        "needs_donor_ctrl": False,
        "needs_acceptor_ctrl": False,
        "needs_neg_ctrl": True,
    },
    "hmsd": {
        "needs_injection_primary": False,
        "needs_injection_secondary": False,
        "needs_pos_ctrl": True,
        "needs_donor_ctrl": True,  # Changed: HMSD requires donor control
        "needs_acceptor_ctrl": False,  # Changed: Acceptor control no longer required
        "needs_neg_ctrl": False,
    },
    "mass": {
        "needs_injection_primary": False,
        "needs_injection_secondary": False,
        "needs_pos_ctrl": True,
        "needs_donor_ctrl": True,  # Changed: Mass-action requires donor control
        "needs_acceptor_ctrl": False,  # Acceptor control not required (only for validation, not used in computation)
        "needs_neg_ctrl": False,
    },
    "normalised_sd": {
        "needs_injection_primary": False,
        "needs_injection_secondary": False,
        "needs_pos_ctrl": True,
        "needs_donor_ctrl": False,
        "needs_acceptor_ctrl": False,
        "needs_neg_ctrl": True,
    },
    "hp_quenching": {
        "needs_injection_primary": False,
        "needs_injection_secondary": False,
        "needs_pos_ctrl": True,
        "needs_donor_ctrl": False,
        "needs_acceptor_ctrl": False,
        "needs_neg_ctrl": True,
        "needs_neg_ctrl_conc": True,
    },
}


def _require_keys(d: Dict[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ParamsError(f"Missing fields: {', '.join(missing)}")


# - - - Public parser - - -
def parse_params(raw_dict: Dict[str, Any]) -> ConvertParams:
    """
    Convert a dictionary of already-parsed values into a validated ConvertParams instance.

    All numeric values should already be floats (converted by load_parameters).
    Strings will cause validation errors.

    Expected keys in raw_dict:
      mode, init_start, init_end, trigger,
      injection_primary, injection_secondary,
      beta_DF, beta_AF, beta_BF,
      pos_ctrl_concentrations, donor_ctrl_concentrations, acceptor_ctrl_concentrations,
      start_with_donor

    Mode-specific validation:
    - TMSD: requires injection_primary (nuking correction)
    - HMSD, Mass: no injections required
    """
    _require_keys(
        raw_dict,
        [
            "mode",
            "init_start",
            "init_end",
            "trigger",
            "injection_primary",
            "injection_secondary",
            "neg_ctrl_start",
            "neg_ctrl_end",
            "pos_ctrl_concentrations",
            "donor_ctrl_concentrations",
            "acceptor_ctrl_concentrations",
            "neg_ctrl_concentrations",
            "start_with_donor",
        ],
    )

    # Mode validation
    mode = str(raw_dict["mode"]).strip().lower()
    if mode not in _ALLOWED_MODES:
        allowed = ", ".join(sorted(_ALLOWED_MODES))
        raise ParamsError(f"Unsupported mode '{mode}'. Allowed modes: {allowed}")

    # Get mode requirements
    reqs = _MODE_REQUIREMENTS[mode]

    # Extract already-converted values (should be floats from load_parameters)
    init_start = raw_dict["init_start"]
    init_end = raw_dict["init_end"]
    trigger = raw_dict["trigger"]
    injection_primary = raw_dict["injection_primary"]
    injection_secondary = raw_dict["injection_secondary"]
    neg_ctrl_start = raw_dict["neg_ctrl_start"]
    neg_ctrl_end = raw_dict["neg_ctrl_end"]

    # Validate types
    if not isinstance(init_start, (int, float)):
        raise ParamsError(f"init_start must be a number, got {type(init_start).__name__}")
    if not isinstance(init_end, (int, float)):
        raise ParamsError(f"init_end must be a number, got {type(init_end).__name__}")
    if not isinstance(trigger, (int, float)):
        raise ParamsError(f"trigger must be a number, got {type(trigger).__name__}")

    if injection_primary is not None and not isinstance(injection_primary, (int, float)):
        raise ParamsError(f"injection_primary must be a number or None, got {type(injection_primary).__name__}")
    if injection_secondary is not None and not isinstance(injection_secondary, (int, float)):
        raise ParamsError(f"injection_secondary must be a number or None, got {type(injection_secondary).__name__}")
    if neg_ctrl_start is not None and not isinstance(neg_ctrl_start, (int, float)):
        raise ParamsError(f"neg_ctrl_start must be a number or None, got {type(neg_ctrl_start).__name__}")
    if neg_ctrl_end is not None and not isinstance(neg_ctrl_end, (int, float)):
        raise ParamsError(f"neg_ctrl_end must be a number or None, got {type(neg_ctrl_end).__name__}")

    # Validate mode-specific injection requirements
    if reqs["needs_injection_primary"] and injection_primary is None:
        # Provide mode-specific error messages
        if mode in ("tmsd", "internal_tmsd"):
            injection_label = "nuking timepoint"
            description = "when the nuking injection occurs"
        else:
            injection_label = "primary injection time"
            description = "when the injection occurs"

        raise ParamsError(
            f"Mode '{mode.upper()}' requires {injection_label}. "
            f"Please specify {description}."
        )

    if reqs["needs_injection_secondary"] and injection_secondary is None:
        raise ParamsError(
            f"Mode '{mode.upper()}' requires secondary injection time (Injection 2). "
            f"Please specify when the second injection occurs."
        )

    # Extract bleed-through factors (optional)
    beta_DF = raw_dict.get("beta_DF")
    beta_AF = raw_dict.get("beta_AF")
    beta_BF = raw_dict.get("beta_BF")

    # Extract control concentrations (should already be lists of floats)
    pos_ctrl = raw_dict["pos_ctrl_concentrations"]
    donor_ctrl = raw_dict["donor_ctrl_concentrations"]
    acceptor_ctrl = raw_dict["acceptor_ctrl_concentrations"]
    neg_ctrl_conc = raw_dict["neg_ctrl_concentrations"]

    # Validate control types
    if not isinstance(pos_ctrl, list):
        raise ParamsError(f"pos_ctrl_concentrations must be a list, got {type(pos_ctrl).__name__}")
    if not isinstance(donor_ctrl, list):
        raise ParamsError(f"donor_ctrl_concentrations must be a list, got {type(donor_ctrl).__name__}")
    if not isinstance(acceptor_ctrl, list):
        raise ParamsError(f"acceptor_ctrl_concentrations must be a list, got {type(acceptor_ctrl).__name__}")
    if not isinstance(neg_ctrl_conc, list):
        raise ParamsError(f"neg_ctrl_concentrations must be a list, got {type(neg_ctrl_conc).__name__}")

    # Validate all items in lists are numbers
    for i, val in enumerate(pos_ctrl):
        if not isinstance(val, (int, float)):
            raise ParamsError(f"pos_ctrl_concentrations[{i}] must be a number, got {type(val).__name__}")
    for i, val in enumerate(donor_ctrl):
        if not isinstance(val, (int, float)):
            raise ParamsError(f"donor_ctrl_concentrations[{i}] must be a number, got {type(val).__name__}")
    for i, val in enumerate(acceptor_ctrl):
        if not isinstance(val, (int, float)):
            raise ParamsError(f"acceptor_ctrl_concentrations[{i}] must be a number, got {type(val).__name__}")
    for i, val in enumerate(neg_ctrl_conc):
        if not isinstance(val, (int, float)):
            raise ParamsError(f"neg_ctrl_concentrations[{i}] must be a number, got {type(val).__name__}")

    # Validate mode-specific control requirements
    if reqs["needs_pos_ctrl"] and not pos_ctrl:
        raise ParamsError(
            f"Mode '{mode.upper()}' requires at least one positive control concentration. "
            f"Please enter positive control concentrations."
        )

    if reqs["needs_donor_ctrl"] and not donor_ctrl:
        raise ParamsError(
            f"Mode '{mode.upper()}' requires donor control concentration. "
            f"Please enter donor control concentration."
        )

    if reqs["needs_acceptor_ctrl"] and not acceptor_ctrl:
        raise ParamsError(
            f"Mode '{mode.upper()}' requires acceptor control concentration. "
            f"Please enter acceptor control concentration."
        )

    if reqs.get("needs_neg_ctrl_conc", False) and not neg_ctrl_conc:
        raise ParamsError(
            f"Mode '{mode.upper()}' requires negative control concentrations. "
            f"Please enter negative control (HI product) concentrations."
        )

    # Validate negative control timepoints for internal_tmsd mode
    if reqs.get("needs_neg_ctrl", False):
        if neg_ctrl_start is None or neg_ctrl_end is None:
            raise ParamsError(
                f"Mode '{mode.upper()}' requires negative control timepoints. "
                f"Please specify negative control start and end times."
            )
        if neg_ctrl_end <= neg_ctrl_start:
            raise ParamsError(
                f"Negative control end time ({neg_ctrl_end}) must be greater than start time ({neg_ctrl_start})"
            )
        if neg_ctrl_start < 0:
            raise ParamsError("Negative control start time must be non-negative")

    # Extract start_with_donor flag
    start_with_donor = raw_dict["start_with_donor"]
    if not isinstance(start_with_donor, bool):
        raise ParamsError(f"start_with_donor must be a boolean, got {type(start_with_donor).__name__}")

    # Common validation checks
    if init_end <= init_start:
        raise ParamsError(
            f"Initialisation end time ({init_end}) must be greater than start time ({init_start})"
        )
    if trigger < 0:
        raise ParamsError("Reaction trigger time must be non-negative")

    # Validate injection times if provided
    if injection_primary is not None and injection_primary < 0:
        raise ParamsError("Primary injection time must be non-negative")
    if injection_secondary is not None and injection_secondary < 0:
        raise ParamsError("Secondary injection time must be non-negative")

    # Validate negative control times if provided
    if neg_ctrl_start is not None and neg_ctrl_start < 0:
        raise ParamsError("Negative control start time must be non-negative")
    if neg_ctrl_end is not None and neg_ctrl_end < 0:
        raise ParamsError("Negative control end time must be non-negative")

    return ConvertParams(
        mode=mode,
        init_start=float(init_start),
        init_end=float(init_end),
        trigger=float(trigger),
        neg_ctrl_start=float(neg_ctrl_start) if neg_ctrl_start is not None else None,
        neg_ctrl_end=float(neg_ctrl_end) if neg_ctrl_end is not None else None,
        injection_primary=float(injection_primary) if injection_primary is not None else None,
        injection_secondary=float(injection_secondary) if injection_secondary is not None else None,
        beta_DF=float(beta_DF) if beta_DF is not None else None,
        beta_AF=float(beta_AF) if beta_AF is not None else None,
        beta_BF=float(beta_BF) if beta_BF is not None else None,
        pos_ctrl_concentrations=[float(x) for x in pos_ctrl],
        donor_ctrl_concentrations=[float(x) for x in donor_ctrl],
        acceptor_ctrl_concentrations=[float(x) for x in acceptor_ctrl],
        neg_ctrl_concentrations=[float(x) for x in neg_ctrl_conc],
        start_with_donor=start_with_donor,
    )
