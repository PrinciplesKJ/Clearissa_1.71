# core/convert_data_tab/conversion_helpers.py
"""
Shared helper functions for data conversion operations.

Nomenclature (aligned with methodology documentation):
    - pos_q(t): Raw positive control fluorescence for well q
    - blank(t): Blank fluorescence
    - pos_q_corr(t): Blank-corrected positive control = pos_q(t) - blank(t)
    - pos_q_init: Mean of pos_q_corr in initialisation window
    - c_io: Calibration slope [AFU/nM] from forced-origin fit
    - r_q(t): Ratio = pos_q_corr(t) / pos_q_init
    - r_mean(t): Mean ratio across all positive control wells
    - C_ref: Reference concentration [nM]
    - p_corr(t): Reference trajectory = c_io * C_ref * r_mean(t) [AFU]
    - f_corr(t): Blank-corrected experimental fluorescence [AFU]
    - g(t): Normalised ratio = f_corr(t) / p_corr(t) [dimensionless]
    - h(t): Concentration-referenced signal = g(t) * C_ref [nM]
"""
from typing import Any, Dict, Optional
import logging
import numpy as np
import pandas as pd

from .params import ConvertParams

logger = logging.getLogger(__name__)


# ==================== VALIDATION ====================

def _require_df(df: Any, name: str) -> pd.DataFrame:
    """Validate that input is a non-empty DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if df.empty:
        raise ValueError(f"{name} is empty")
    return df


def get_channel_tag(selected_channels: dict, channel_detection: dict, key: str) -> tuple[str, int]:
    """
    Extract channel tag from metadata and return match count.

    Parameters
    ----------
    selected_channels : dict
        User-selected channel mappings (e.g. {"donor": "488-14/535-30"})
    channel_detection : dict
        Auto-detected channel information with match counts
    key : str
        Channel key to look up (e.g. "donor", "acceptor", "fret")

    Returns
    -------
    tuple[str, int]
        (channel_tag, match_count) where match_count is the number of wells
        that matched this channel. A match_count of 0 indicates the channel
        was not found in the data. A match_count of -1 indicates the channel
        was manually specified but match count is unknown.
    """
    tag = selected_channels.get(key)
    matches = 0

    if not tag:
        detection = channel_detection.get(key.capitalize()) or {}
        tag = detection.get("selected", "")
        matches = detection.get("matches", 0)
    else:
        # If tag was manually selected, check if detection has match info
        detection = channel_detection.get(key.capitalize()) or {}
        if detection.get("selected") == tag:
            matches = detection.get("matches", 0)
        else:
            # Assume manual selection is valid
            matches = -1  # Unknown, but user-specified

    return tag, matches


def validate_channel_found(
    channel_name: str,
    channel_tag: str,
    match_count: int,
    df: pd.DataFrame,
) -> None:
    """
    Validate that a channel was found in the data.

    Parameters
    ----------
    channel_name : str
        Human-readable channel name for error messages (e.g. "Donor")
    channel_tag : str
        The channel wavelength tag being searched for
    match_count : int
        Number of wells matching this channel (0 means not found, -1 means unknown)
    df : pd.DataFrame
        DataFrame to extract sample labels from for error message

    Raises
    ------
    ValueError
        If channel is not configured or was not found in the data.
        The error message guides the user to update channel settings.
    """
    # Collect unique Well labels for diagnostic output
    available_labels = []
    if df is not None and "Well" in getattr(df, "columns", []):
        available_labels = sorted(df["Well"].astype(str).unique().tolist())

    if not channel_tag:
        hint = (
            "%s channel is not configured.\n\n"
            "Go to the 'Options' tab and set the wavelengths for "
            "Donor, Acceptor, and FRET so that each tag is a substring "
            "of the corresponding channel label in your data."
            % channel_name
        )
        if available_labels:
            hint += (
                "\n\nChannel labels found in your data:\n  %s"
                % "\n  ".join(available_labels)
            )
        raise ValueError(hint)

    if match_count == 0:
        hint = (
            "%s channel '%s' was not found in the data.\n\n"
            "Go to the 'Options' tab and set the wavelengths for "
            "Donor, Acceptor, and FRET so that each tag is a substring "
            "of the corresponding channel label in your data."
            % (channel_name, channel_tag)
        )
        if available_labels:
            hint += (
                "\n\nChannel labels found in your data:\n  %s"
                % "\n  ".join(available_labels)
            )
        raise ValueError(hint)


# ==================== TIME UTILITIES ====================

_TIME_COL_CANDIDATES = ("time", "Time", "t", "seconds", "Time [min]")


def _find_time_col(df: pd.DataFrame, *, preferred: str = "Time [min]") -> str:
    """Find the time column in a DataFrame."""
    if preferred in df.columns:
        return preferred
    for c in _TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Could not infer a time column")


def find_closest_timepoint(df: pd.DataFrame, target_time: float) -> float:
    """Find the closest actual timepoint to a target time."""
    time_col = _find_time_col(df)
    times = pd.to_numeric(df[time_col], errors='coerce').dropna()
    if times.empty:
        raise ValueError("No valid time values found")
    idx = (times - target_time).abs().idxmin()
    return float(times.loc[idx])


def mean_trace_in_window(trace: np.ndarray, time_vec: np.ndarray,
                         t_start: float, t_end: float) -> float:
    """Calculate mean of a trace within a time window [t_start, t_end]."""
    mask = (time_vec >= t_start) & (time_vec <= t_end)
    if np.sum(mask) == 0:
        raise ValueError(f"No data in window [{t_start}, {t_end}]")
    values = trace[mask]
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        raise ValueError("No valid data in window")
    return float(np.mean(valid))


# ==================== DATA PROCESSING ====================

def ensure_float_block(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame data columns to float, preserving first two metadata columns."""
    if df.empty:
        return df
    left = df.iloc[:, :2].copy()
    num = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    return pd.concat([left, num], axis=1)


def filter_by_tag(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Filter DataFrame rows by tag in 'Well' column."""
    try:
        filtered = df[df["Well"].astype(str).str.contains(str(tag), na=False)].copy()
        return ensure_float_block(filtered)
    except Exception:
        return pd.DataFrame(columns=df.columns)


def bg_subtract(df: pd.DataFrame, blank_df: pd.DataFrame) -> pd.DataFrame:
    """Background subtraction: f_corr = f_raw - blank."""
    df = ensure_float_block(df)
    blank_df = ensure_float_block(blank_df)
    if blank_df.shape[0] == 0 or df.shape[0] == 0:
        return df
    blank_vec = blank_df.iloc[:, 2].to_numpy(dtype=np.float64)[:, None]
    result = df.copy()
    result.iloc[:, 2:] = df.iloc[:, 2:].to_numpy(dtype=np.float64) - blank_vec
    return result


def divide_by_vector(df: pd.DataFrame, denom_df: pd.DataFrame) -> pd.DataFrame:
    """Divide data by a time-series vector (for normalisation)."""
    df = ensure_float_block(df)
    denom_df = ensure_float_block(denom_df)
    if denom_df.shape[0] == 0 or df.shape[0] == 0:
        return df

    denom = denom_df.iloc[:, 2].to_numpy(dtype=np.float64)[:, None]
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)

    result = df.copy()
    result.iloc[:, 2:] = df.iloc[:, 2:].to_numpy(dtype=np.float64) / denom
    return result


# ==================== SIGNAL AVERAGING ====================

def mean_signal_in_window(df: pd.DataFrame, time_col: str,
                          t_start: float, t_end: float) -> float:
    """Calculate mean signal across all data columns within a time window."""
    time_vec = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = (time_vec >= t_start) & (time_vec <= t_end)

    if np.sum(mask) == 0:
        # Provide diagnostic information
        finite_times = time_vec[np.isfinite(time_vec)]
        if len(finite_times) > 0:
            actual_min = float(np.min(finite_times))
            actual_max = float(np.max(finite_times))
            logger.error(
                "No data in window [%.2f, %.2f] min. "
                "Data time range is [%.2f, %.2f] min with %d valid timepoints. "
                "This may indicate an empty DataFrame after channel filtering.",
                t_start, t_end, actual_min, actual_max, len(finite_times)
            )
        else:
            logger.error(
                "No data in window [%.2f, %.2f] min. "
                "DataFrame has %d rows but no valid numeric time values. "
                "This may indicate channel mismatch or empty data.",
                t_start, t_end, len(time_vec)
            )
        raise ValueError(f"No data in window [{t_start}, {t_end}]")

    data_cols = [c for c in df.columns if c not in ['Well', time_col]]
    signals = []
    for col in data_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        valid = values[mask][np.isfinite(values[mask])]
        if len(valid) > 0:
            signals.append(np.mean(valid))

    if not signals:
        raise ValueError("No valid signals in window")
    return float(np.mean(signals))


# ==================== CALIBRATION ====================

def extract_calibration_slope(species_map: Dict[str, Any], species_name: str, channel: str) -> float:
    """Extract and validate calibration slope from species map."""
    key = f"{species_name}-{channel}"
    if key not in species_map:
        raise KeyError(f"Missing calibration for {key}")

    rec = species_map[key]
    if not isinstance(rec, dict) or "slope" not in rec:
        raise ValueError(f"Invalid calibration record for {key}")

    slope = float(rec["slope"])
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        raise ValueError(f"Invalid slope for {key}: {slope}")
    return abs(slope)


def validate_calibration_slopes(species_map: Dict[str, Any],
                                 required_pairs: list[tuple[str, str]]) -> Dict[tuple[str, str], float]:
    """Validate and extract multiple calibration slopes."""
    slopes = {}
    missing = []

    for species, channel in required_pairs:
        try:
            slopes[(species, channel)] = extract_calibration_slope(species_map, species, channel)
        except (KeyError, ValueError) as e:
            missing.append(f"{species}-{channel}: {e}")

    if missing:
        raise ValueError("Missing calibrations:\n" + "\n".join(f"  - {m}" for m in missing))
    return slopes


def estimate_runtime_slopes(
        b1l_ctrl: pd.DataFrame,
        b1aa_ctrl: pd.DataFrame,
        b2_ctrl: pd.DataFrame,
        b1b2_ctrl_donor: pd.DataFrame,
        b1b2_ctrl_fret: pd.DataFrame,
        blank_data: pd.DataFrame,
        donor_tag: str,
        fret_tag: str,
        time_col: str,
        init_start: float,
        init_end: float,
        b1l_conc: float,
        b1aa_conc: float,
        b2_conc: float,
        b1b2_conc: float,
        noise_threshold: float = 1e-3,
) -> Dict[tuple[str, str], float]:
    """
    Estimate calibration slopes from plate-matched controls at known concentrations.

    This function provides a runtime alternative to pre-calibrated slopes stored in
    params.species. For each control species, it computes the slope m(species, channel)
    in units of AFU/nM by:
        1. Filtering the control data by channel tag
        2. Background-subtracting the blank signal
        3. Computing the mean signal in the initialisation window
        4. Dividing by the known concentration

    For B1B2, the positive control has multiple concentrations (e.g. 5, 10, 15, 20 nM)
    which are pre-normalized to C_ref using compute_normalised_posctrl(). The normalized
    trajectories are passed separately for each channel.

    Notes
    -----

    """
    # Standard controls: need channel filtering and background subtraction
    controls = {
        "B1L": (b1l_ctrl, b1l_conc),
        "B1AA": (b1aa_ctrl, b1aa_conc),
        "B2": (b2_ctrl, b2_conc),
    }

    channels = {
        "Donor": donor_tag,
        "FRET": fret_tag,
    }

    slopes = {}
    warnings_list = []

    logger.info("Estimating runtime calibration slopes from plate-matched controls")
    logger.info("Initialisation window: [%.2f, %.2f] min", init_start, init_end)

    # ========== Process B1B2 separately (pre-normalized data) ==========
    b1b2_channels = {
        "Donor": b1b2_ctrl_donor,
        "FRET": b1b2_ctrl_fret,
    }

    for channel_name, b1b2_ctrl in b1b2_channels.items():
        try:
            if b1b2_ctrl is None or b1b2_ctrl.empty:
                msg = f"Pre-normalized B1B2 control for {channel_name} channel is missing or empty"
                logger.warning(msg)
                warnings_list.append(msg)
                continue

            # B1B2 p_corr is already background-subtracted and normalized
            # It represents signal at C_ref concentration
            # The data column is typically the 3rd column (index 2)
            data_cols = [c for c in b1b2_ctrl.columns if c not in ['Well', time_col]]
            if not data_cols:
                msg = f"No data columns in B1B2 {channel_name} control"
                logger.warning(msg)
                warnings_list.append(msg)
                continue

            # Compute mean signal in initialisation window
            mean_signal = mean_signal_in_window(b1b2_ctrl, time_col, init_start, init_end)

            # Compute slope [AFU/nM]
            slope = mean_signal / b1b2_conc

            # Validate slope
            if not np.isfinite(slope):
                msg = f"Non-finite slope for B1B2-{channel_name}: {slope}"
                logger.warning(msg)
                warnings_list.append(msg)
                slope = 0.0
            elif slope < 0:
                msg = f"Negative slope for B1B2-{channel_name}: {slope:.4f} AFU/nM (control issue?)"
                logger.warning(msg)
                warnings_list.append(msg)
                slope = abs(slope)
            elif abs(slope) < noise_threshold:
                msg = (f"Near-zero slope for B1B2-{channel_name}: {slope:.6f} AFU/nM "
                       f"(below threshold {noise_threshold})")
                logger.warning(msg)
                warnings_list.append(msg)

            slopes[("B1B2", channel_name)] = abs(slope)
            logger.info("  B1B2-%s: %.4f AFU/nM (from %.1f nM normalized control, mean signal %.2f AFU)",
                        channel_name, slope, b1b2_conc, mean_signal)

        except Exception as e:
            msg = f"Failed to estimate slope for B1B2-{channel_name}: {e}"
            logger.warning(msg)
            warnings_list.append(msg)

    # ========== Process other species (standard workflow) ==========
    for species_name, (ctrl_df, concentration) in controls.items():
        # Validate control DataFrame
        if ctrl_df is None or ctrl_df.empty:
            msg = f"Control DataFrame for {species_name} is missing or empty"
            logger.warning(msg)
            warnings_list.append(msg)
            continue

        if concentration is None or concentration <= 0:
            msg = f"Invalid concentration for {species_name}: {concentration} nM"
            logger.warning(msg)
            warnings_list.append(msg)
            continue

        for channel_name, channel_tag in channels.items():
            try:
                # Filter control by channel
                ctrl_channel = filter_by_tag(ctrl_df, channel_tag)
                if ctrl_channel.empty:
                    msg = f"No data for {species_name} in {channel_name} channel (tag: {channel_tag})"
                    logger.warning(msg)
                    warnings_list.append(msg)
                    continue

                # Filter blank by channel
                blank_channel = filter_by_tag(blank_data, channel_tag)
                if blank_channel.empty:
                    msg = f"No blank data for {channel_name} channel (tag: {channel_tag})"
                    logger.warning(msg)
                    warnings_list.append(msg)
                    continue

                # Background subtract
                ctrl_bg = bg_subtract(ctrl_channel, blank_channel)

                # Compute mean signal in initialisation window
                mean_signal = mean_signal_in_window(ctrl_bg, time_col, init_start, init_end)

                # Compute slope [AFU/nM]
                slope = mean_signal / concentration

                # Validate slope
                if not np.isfinite(slope):
                    msg = f"Non-finite slope for {species_name}-{channel_name}: {slope}"
                    logger.warning(msg)
                    warnings_list.append(msg)
                    slope = 0.0
                elif slope < 0:
                    msg = f"Negative slope for {species_name}-{channel_name}: {slope:.4f} AFU/nM (control issue?)"
                    logger.warning(msg)
                    warnings_list.append(msg)
                    slope = abs(slope)  # Use absolute value but flag the issue
                elif abs(slope) < noise_threshold:
                    msg = (f"Near-zero slope for {species_name}-{channel_name}: {slope:.6f} AFU/nM "
                           f"(below threshold {noise_threshold})")
                    logger.warning(msg)
                    warnings_list.append(msg)
                    # Keep the small value but flag it

                slopes[(species_name, channel_name)] = abs(slope)
                logger.info("  %s-%s: %.4f AFU/nM (from %.1f nM control, mean signal %.2f AFU)",
                            species_name, channel_name, slope, concentration, mean_signal)

            except Exception as e:
                msg = f"Failed to estimate slope for {species_name}-{channel_name}: {e}"
                logger.warning(msg)
                warnings_list.append(msg)

    # Check that we have all required slopes
    required_pairs = [
        ("B1B2", "Donor"), ("B2", "Donor"), ("B1AA", "Donor"), ("B1L", "Donor"),
        ("B1B2", "FRET"), ("B2", "FRET"), ("B1AA", "FRET"), ("B1L", "FRET"),
    ]

    missing = [f"{s}-{c}" for s, c in required_pairs if (s, c) not in slopes]
    if missing:
        raise ValueError(
            f"Failed to estimate {len(missing)} required runtime slopes: {', '.join(missing)}. "
            f"Warnings: {'; '.join(warnings_list)}"
        )

    logger.info("Successfully estimated %d runtime calibration slopes", len(slopes))
    return slopes


def compute_brightness_coefficients(
        slopes: Dict[tuple[str, str], float],
) -> Dict[str, Dict[str, float]]:
    """
    Compute brightness coefficients from calibration slopes for the 2x2 solver.

    This function converts raw slopes m(species, channel) [AFU/nM] into the
    dimensionless brightness coefficients (alpha, beta, gamma, kappa) and their
    modified forms (kappa_prime, gamma_prime) used in the conservation-substituted
    signal equations.

    Physical interpretation:
    - alpha_c = m(B1L, c) / m(B1B2, c): Quenched acceptor brightness relative to product
    - beta_c  = m(B2, c) / m(B1B2, c):  Free donor brightness relative to product
    - gamma_c = m(B1AA, c) / m(B1B2, c): Activated acceptor brightness relative to product
    - kappa_c = 1.0: Product reference (B1B2 basis, normalised to unity)

    Modified coefficients after conservation substitution:
    - kappa_prime_c = kappa_c - beta_c - alpha_c  (effective B1B2 coefficient)
    - gamma_prime_c = gamma_c - alpha_c           (effective B1AA coefficient)

    Parameters
    ----------
    slopes : Dict[tuple[str, str], float]
        Dictionary mapping (species_name, channel_name) to slope [AFU/nM].
        Must contain all 8 required pairs: 4 species x 2 channels (Donor, FRET).

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with structure:
        {
            "Donor": {
                "alpha": float, "beta": float, "gamma": float, "kappa": float,
                "kappa_prime": float, "gamma_prime": float,
                "m_B1B2": float, "m_B2": float, "m_B1AA": float, "m_B1L": float
            },
            "FRET": { ... same structure ... }
        }

    Raises
    ------
    KeyError
        If any required slope is missing from the input dictionary.
    ValueError
        If any slope is non-finite or effectively zero.
    """
    def get_slope(species: str, channel: str) -> float:
        key = (species, channel)
        if key not in slopes:
            raise KeyError(f"Missing slope for {species}-{channel}")
        val = float(slopes[key])
        if not np.isfinite(val) or abs(val) < 1e-12:
            raise ValueError(f"Invalid slope for {species}-{channel}: {val}")
        return abs(val)

    coeffs = {}

    for channel in ["Donor", "FRET"]:
        # Extract raw slopes
        m_B1B2 = get_slope("B1B2", channel)
        m_B2 = get_slope("B2", channel)
        m_B1AA = get_slope("B1AA", channel)
        m_B1L = get_slope("B1L", channel)

        # Compute brightness coefficients (relative to B1B2)
        alpha = m_B1L / m_B1B2   # B1L brightness
        beta = m_B2 / m_B1B2     # B2 brightness
        gamma = m_B1AA / m_B1B2  # B1AA brightness
        kappa = 1.0              # B1B2 reference (normalised to 1)

        # Modified coefficients for 2x2 system after conservation substitution
        kappa_prime = kappa - beta - alpha
        gamma_prime = gamma - alpha

        coeffs[channel] = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "kappa": kappa,
            "kappa_prime": kappa_prime,
            "gamma_prime": gamma_prime,
            # Also store raw slopes for diagnostics
            "m_B1B2": m_B1B2,
            "m_B2": m_B2,
            "m_B1AA": m_B1AA,
            "m_B1L": m_B1L,
        }

    logger.debug("Brightness coefficients computed:")
    for ch, c in coeffs.items():
        logger.debug("  %s: alpha=%.4f, beta=%.4f, gamma=%.4f, kappa_prime=%.4f, gamma_prime=%.4f",
                     ch, c["alpha"], c["beta"], c["gamma"], c["kappa_prime"], c["gamma_prime"])

    return coeffs


def compute_calibration_comparison(
        precal_slopes: Dict[tuple[str, str], float],
        runtime_slopes: Dict[tuple[str, str], float],
        precal_coeffs: Dict[str, Dict[str, float]],
        runtime_coeffs: Dict[str, Dict[str, float]],
        precal_results: Dict[str, pd.DataFrame],
        runtime_results: Dict[str, pd.DataFrame],
        time_col: str,
        trigger_time: float,
        agreement_threshold: float = 0.5,
) -> Dict[str, any]:
    """
    Compute comparison metrics between pre-calibrated and runtime calibration paths.

    This function generates a comprehensive comparison of slopes, coefficients,
    and resulting concentration traces between the two calibration methods,
    enabling users to assess calibration quality and method agreement.

    Parameters
    ----------
    precal_slopes : Dict[tuple[str, str], float]
        Pre-calibrated slopes {(species, channel): slope_value}.
    runtime_slopes : Dict[tuple[str, str], float]
        Runtime-estimated slopes {(species, channel): slope_value}.
    precal_coeffs : Dict[str, Dict[str, float]]
        Brightness coefficients from pre-calibrated slopes.
    runtime_coeffs : Dict[str, Dict[str, float]]
        Brightness coefficients from runtime slopes.
    precal_results : Dict[str, pd.DataFrame]
        Per-well concentration results from pre-calibrated path.
    runtime_results : Dict[str, pd.DataFrame]
        Per-well concentration results from runtime path.
    time_col : str
        Name of the time column in result DataFrames.
    trigger_time : float
        Reaction trigger time [minutes] - comparison metrics computed post-trigger.
    agreement_threshold : float, optional
        Mean absolute difference threshold [nM] for flagging agreement.
        Default is 0.5 nM.

    Returns
    -------
    Dict[str, any]
        Comparison summary containing:
        - "slope_comparison": {(species, channel): {"precalibrated": v, "runtime": v, "ratio": r, "abs_diff": d}}
        - "coefficient_comparison": {channel: {coeff_name: {"precalibrated": v, "runtime": v, ...}}}
        - "per_well_comparison": {well: {"B1B2_mad": f, "B1AA_mad": f, "B1B2_corr": f, "B1AA_corr": f, "agrees": bool}}
        - "overall_agreement": bool - True if all wells agree within threshold
        - "summary_stats": {"mean_B1B2_mad": f, "mean_B1AA_mad": f, "n_agreeing_wells": int, "n_total_wells": int}
    """
    comparison = {
        "slope_comparison": {},
        "coefficient_comparison": {},
        "per_well_comparison": {},
        "overall_agreement": True,
        "summary_stats": {},
    }

    # ========== Slope comparison ==========
    all_keys = set(precal_slopes.keys()) | set(runtime_slopes.keys())
    for key in all_keys:
        precal_val = precal_slopes.get(key, np.nan)
        runtime_val = runtime_slopes.get(key, np.nan)
        if precal_val > 0 and runtime_val > 0:
            ratio = runtime_val / precal_val
        else:
            ratio = np.nan
        abs_diff = abs(runtime_val - precal_val) if np.isfinite(precal_val) and np.isfinite(runtime_val) else np.nan

        comparison["slope_comparison"][f"{key[0]}-{key[1]}"] = {
            "precalibrated": float(precal_val) if np.isfinite(precal_val) else None,
            "runtime": float(runtime_val) if np.isfinite(runtime_val) else None,
            "ratio": float(ratio) if np.isfinite(ratio) else None,
            "abs_diff": float(abs_diff) if np.isfinite(abs_diff) else None,
        }

    # ========== Coefficient comparison ==========
    for channel in ["Donor", "FRET"]:
        comparison["coefficient_comparison"][channel] = {}
        precal_ch = precal_coeffs.get(channel, {})
        runtime_ch = runtime_coeffs.get(channel, {})

        for coeff_name in ["alpha", "beta", "gamma", "kappa_prime", "gamma_prime"]:
            precal_val = precal_ch.get(coeff_name, np.nan)
            runtime_val = runtime_ch.get(coeff_name, np.nan)
            if abs(precal_val) > 1e-12:
                ratio = runtime_val / precal_val
            else:
                ratio = np.nan
            abs_diff = abs(runtime_val - precal_val)

            comparison["coefficient_comparison"][channel][coeff_name] = {
                "precalibrated": float(precal_val),
                "runtime": float(runtime_val),
                "ratio": float(ratio) if np.isfinite(ratio) else None,
                "abs_diff": float(abs_diff),
            }

    # ========== Per-well trace comparison ==========
    # Compare B1B2 and B1AA traces post-trigger
    common_wells = set(precal_results.keys()) & set(runtime_results.keys())
    b1b2_mads = []
    b1aa_mads = []
    n_agreeing = 0

    for well in common_wells:
        precal_df = precal_results[well]
        runtime_df = runtime_results[well]

        time_vec = precal_df[time_col].to_numpy(dtype=np.float64)
        post_mask = time_vec >= trigger_time

        if not np.any(post_mask):
            # No post-trigger data
            comparison["per_well_comparison"][well] = {
                "B1B2_mad": np.nan,
                "B1AA_mad": np.nan,
                "B1B2_corr": np.nan,
                "B1AA_corr": np.nan,
                "agrees": False,
            }
            continue

        # Extract post-trigger traces
        precal_B1B2 = precal_df["B1B2"].to_numpy(dtype=np.float64)[post_mask]
        runtime_B1B2 = runtime_df["B1B2"].to_numpy(dtype=np.float64)[post_mask]
        precal_B1AA = precal_df["B1AA"].to_numpy(dtype=np.float64)[post_mask]
        runtime_B1AA = runtime_df["B1AA"].to_numpy(dtype=np.float64)[post_mask]

        # Mean absolute difference
        b1b2_mad = float(np.nanmean(np.abs(precal_B1B2 - runtime_B1B2)))
        b1aa_mad = float(np.nanmean(np.abs(precal_B1AA - runtime_B1AA)))

        # Correlation coefficient
        if np.std(precal_B1B2) > 1e-9 and np.std(runtime_B1B2) > 1e-9:
            b1b2_corr = float(np.corrcoef(precal_B1B2, runtime_B1B2)[0, 1])
        else:
            b1b2_corr = np.nan

        if np.std(precal_B1AA) > 1e-9 and np.std(runtime_B1AA) > 1e-9:
            b1aa_corr = float(np.corrcoef(precal_B1AA, runtime_B1AA)[0, 1])
        else:
            b1aa_corr = np.nan

        # Agreement check: both MADs below threshold
        agrees = b1b2_mad < agreement_threshold and b1aa_mad < agreement_threshold
        if agrees:
            n_agreeing += 1
        else:
            comparison["overall_agreement"] = False

        b1b2_mads.append(b1b2_mad)
        b1aa_mads.append(b1aa_mad)

        comparison["per_well_comparison"][well] = {
            "B1B2_mad": b1b2_mad,
            "B1AA_mad": b1aa_mad,
            "B1B2_corr": b1b2_corr if np.isfinite(b1b2_corr) else None,
            "B1AA_corr": b1aa_corr if np.isfinite(b1aa_corr) else None,
            "agrees": agrees,
        }

    # ========== Summary statistics ==========
    comparison["summary_stats"] = {
        "mean_B1B2_mad": float(np.nanmean(b1b2_mads)) if b1b2_mads else np.nan,
        "mean_B1AA_mad": float(np.nanmean(b1aa_mads)) if b1aa_mads else np.nan,
        "n_agreeing_wells": n_agreeing,
        "n_total_wells": len(common_wells),
        "agreement_threshold_nM": agreement_threshold,
    }

    logger.info("Calibration comparison: %d/%d wells agree within %.2f nM threshold",
                n_agreeing, len(common_wells), agreement_threshold)

    return comparison


# ==================== TABLE ASSEMBLY ====================

def assemble_species_tables(results_dict: Dict[str, pd.DataFrame],
                            species_list: list[str],
                            time_column: str) -> Dict[str, pd.DataFrame]:
    """Assemble per-species concentration tables from per-well results."""
    if not results_dict:
        return {}

    first = next(iter(results_dict.values()))
    time_vec = first[time_column].to_numpy(dtype=np.float64)
    n_times = len(time_vec)

    out = {}
    for species in species_list:
        tbl = pd.DataFrame({"Well": [species] * n_times, time_column: time_vec})
        for well, df in results_dict.items():
            if species in df.columns:
                tbl[well] = df[species].to_numpy(dtype=np.float64)
        out[species] = tbl
    return out


def estimate_initial_concentration(
        ratio_df: pd.DataFrame,
        well_col: str,
        time_vec: np.ndarray,
        t_start: float,
        t_end: float
) -> float:
    """
    Estimate initial concentration from mean signal in initialisation window.

    Used for estimating B2_0, Ax_0, etc. from the init phase plateau.

    """
    trace = pd.to_numeric(ratio_df[well_col], errors="coerce").to_numpy(dtype=np.float64)
    return mean_trace_in_window(trace, time_vec, t_start, t_end)


def create_initial_value_table(
        results_dict: Dict[str, pd.DataFrame],
        initial_param_name: str,
        time_column: str
) -> pd.DataFrame:
    """
    Create a table of initial values (constants) repeated for all timepoints.

    Used for B2_0, B1AA_0, etc. which are constant per well but need to be
    in the same time-series format for export.

    """
    if not results_dict:
        return pd.DataFrame()

    first_result = next(iter(results_dict.values()))
    time_vec = first_result[time_column].to_numpy(dtype=np.float64)
    n_times = len(time_vec)

    tbl = pd.DataFrame({"Well": [initial_param_name] * n_times, time_column: time_vec})

    for well_name, well_df in results_dict.items():
        if initial_param_name in well_df.columns:
            initial_value = float(well_df[initial_param_name].iloc[0])
            tbl[well_name] = np.full(n_times, initial_value, dtype=np.float64)

    return tbl


# ==================== POSITIVE CONTROL NORMALISATION ====================

def compute_initialisation_slope(
        pos_ctrl_data: pd.DataFrame,
        blank_data: pd.DataFrame,
        params: ConvertParams,
) -> tuple[float, float]:
    """
    Calculate calibration slope c_io from positive control initialisation phase.

    Uses forced-through-origin linear regression:
        pos_q_init = c_io * [C]_q

    Algorithm:
        1. Background subtract: pos_q_corr = pos_q - blank
        2. Calculate mean in init window: pos_q_init = mean(pos_q_corr) for t in [t_init_start, t_init_end]
        3. Multi-point: c_io = sum(C_q * pos_q_init) / sum(C_q^2)
        4. Single-point: c_io = pos_q_init / C_q

    Returns:
        (c_io, 0.0) - slope and intercept (intercept always 0 for forced-origin)
    """
    logger.info("Computing calibration slope c_io from initialisation phase")

    concentrations = params.pos_ctrl_concentrations
    if not concentrations:
        raise ValueError("No positive control concentrations provided")

    logger.info("Init window: [%.1f, %.1f] min, concentrations: %s nM",
                params.init_start, params.init_end, concentrations)

    # Background subtract
    pos_corr = pos_ctrl_data.copy()
    pos_corr[pos_corr.columns[2:]] = pos_corr.iloc[:, 2:].astype(np.float64)
    pos_corr.iloc[:, 2:] = pos_corr.iloc[:, 2:].subtract(blank_data.iloc[:, 2].astype(np.float64), axis=0)

    # Extract init window means
    time_col = _find_time_col(pos_corr)
    mask_init = (pos_corr[time_col] >= params.init_start) & (pos_corr[time_col] <= params.init_end)

    pos_init_means = [pos_corr.loc[mask_init, col].mean() for col in pos_corr.columns[2:]]

    # Filter invalid points
    valid_pairs = [(c, f) for c, f in zip(concentrations, pos_init_means)
                   if f is not None and pd.notna(f) and np.isfinite(f)]

    if not valid_pairs:
        raise ValueError("No valid data for calibration")

    conc_valid, fluor_valid = zip(*valid_pairs)

    # Forced-origin fit: c_io = sum(C * F) / sum(C^2)
    if len(conc_valid) >= 2:
        C = np.array(conc_valid, dtype=np.float64)
        F = np.array(fluor_valid, dtype=np.float64)
        c_io = np.sum(C * F) / np.sum(C * C)

        # R^2 for forced-origin
        F_pred = c_io * C
        r_squared = 1.0 - np.sum((F - F_pred)**2) / np.sum(F**2)

        logger.info("Multi-point calibration: c_io=%.4f AFU/nM, R^2=%.4f (%d points)",
                    c_io, r_squared, len(conc_valid))
    else:
        c_io = fluor_valid[0] / conc_valid[0]
        logger.warning("Single-point calibration: c_io=%.4f AFU/nM from %.1f nM",
                       c_io, conc_valid[0])

    return c_io, 0.0


def compute_normalised_posctrl(
        pos_ctrl_data: pd.DataFrame,
        blank_data: pd.DataFrame,
        params: ConvertParams,
        calibration: tuple[float, float],
        plot_callback: Optional[callable] = None
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Compute normalised positive control trajectory p_corr(t).

    Algorithm (from methodology):
        1. Blank subtraction: pos_q_corr(t) = pos_q(t) - blank(t)
        2. Init mean: pos_q_init = mean(pos_q_corr) for t in init window
        3. Ratio: r_q(t) = pos_q_corr(t) / pos_q_init
        4. Mean ratio: r_mean(t) = mean(r_q(t)) across all q
        5. Reference trajectory: p_corr(t) = c_io * C_ref * r_mean

    Returns:
        (p_corr DataFrame, warning message or None)
    """
    logger.info("Computing p_corr(t) reference trajectory")

    c_io, _ = calibration
    C_ref = params.c_ref

    # Check for single concentration warning
    n_conc = len(params.pos_ctrl_concentrations) if params.pos_ctrl_concentrations else 0
    warning = None
    if n_conc == 1:
        warning = (f"Single positive control concentration ({params.pos_ctrl_concentrations[0]:.1f} nM). "
                   f"Calibration assumes F(0)=0. Use ≥2 concentrations for robust calibration.")
        logger.warning(warning)

    # Step 1: pos_q_corr = pos_q - blank
    pos_corr = pos_ctrl_data.copy()
    # Cast to float64 before arithmetic to avoid pandas dtype incompatibility warnings
    pos_corr[pos_corr.columns[2:]] = pos_corr.iloc[:, 2:].astype(np.float64)
    pos_corr.iloc[:, 2:] = pos_corr.iloc[:, 2:].values - blank_data.iloc[:, 2].astype(np.float64).values[:, None]

    # Step 2: pos_q_init = mean in init window
    time_col = _find_time_col(pos_corr)
    mask_init = (pos_corr[time_col] >= params.init_start) & (pos_corr[time_col] <= params.init_end)
    pos_init = pos_corr.loc[mask_init, pos_corr.columns[2:]].mean()

    # Step 3: r_q(t) = pos_q_corr(t) / pos_q_init
    r_q = pos_corr.copy()
    r_q.iloc[:, 2:] = pos_corr.iloc[:, 2:].div(pos_init.values, axis=1)

    # Step 4: r_mean(t) = mean across wells
    r_mean = r_q.iloc[:, 2:].mean(axis=1)

    # Step 5: p_corr(t) = c_io * C_ref * r_mean(t)
    p_corr_values = c_io * C_ref * r_mean

    logger.info("p_corr(t): c_io=%.4f, C_ref=%.1f nM, %d wells averaged",
                c_io, C_ref, len(pos_corr.columns) - 2)

    # Build output DataFrame
    time_values = pos_corr[time_col].values
    p_corr_df = pd.DataFrame({
        'Well': ['Positive Control'] * len(time_values),
        time_col: time_values,
        'Positive Control': p_corr_values.values
    })

    if plot_callback:
        plot_callback(p_corr_df, titles="Positive Control p_corr(t)")

    return p_corr_df, warning

