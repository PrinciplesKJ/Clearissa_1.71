# core/convert_data_tab/conversion_normalised_sd.py
"""
Control-Normalised Readout Module (Relative Fluorescence)
=========================================================

Reports a dimensionless normalised readout (relative fluorescence, RF) rather
than concentrations, following the methodology in the supplementary material.

Experimental Context
--------------------
This mode is used when a fluorescence-to-concentration mapping is not justified
(for example, complex stoichiometry, multiple fluorescent species, or unknown
quenching fractions). The essential principle is to report the fluorescence on
a scale normalised with respect to two controls - typically a negative control
assigned the value 0 and a positive control assigned the value 1.

Note that the system being reported need not contain only the fluorescent
species that define these controls.

Signal Model
------------
For each well, we consider:
    - f_corr(t): blank-corrected experimental fluorescence
    - p_corr(t): averaged, blank-corrected positive control trajectory
    - f_neg_corr(t): blank-corrected negative control

All quantities are defined for the time period corresponding to the main
measurement interval t in [t_meas_start, t_meas_end].

Negative-Control Baseline
-------------------------
A baseline window (typically 30 min) t in [t_nc_start, t_nc_end] within the
main measurement interval is selected after any initial transients in the
negative control due to the reaction initiation procedure have subsided.
This window defines a fixed negative control baseline:

    f_neg_base = mean(f_neg_corr(t)) for t in [t_nc_start, t_nc_end]

Relative Fluorescence Calculation
---------------------------------
We then calculate the relative fluorescence as:

    RF(t) = (f_corr(t) - f_neg_base) / (p_corr(t) - f_neg_base)

where f_neg_base is a fixed baseline value (mean of f_neg_corr in the baseline
window), ignoring (typically very small) changes in the negative control signal
over the main measurement window.

Output scale: 0 = negative control baseline, 1 = positive control level.
"""
from typing import Dict
import logging
import numpy as np
import pandas as pd

from .params import ConvertParams
from .conversion_helpers import (
    _require_df,
    _find_time_col,
    compute_initialisation_slope,
    compute_normalised_posctrl,
)

logger = logging.getLogger(__name__)


def convert_normalised_sd(
        selected_data: pd.DataFrame,
        pos_ctrl_data: pd.DataFrame,
        blank_ctrl_data: pd.DataFrame,
        neg_ctrl_data: pd.DataFrame,
        params: ConvertParams,
) -> Dict[str, pd.DataFrame]:
    """
    Convert fluorescence to relative fluorescence (RF).

    Computes RF(t) = (f_corr(t) - f_neg_base) / (p_corr(t) - f_neg_base)

    The negative control baseline f_neg_base is calculated as the mean of the
    blank-corrected negative control signal over a user-specified baseline window.
    This window should be selected after initial transients from reaction
    initiation have subsided (typically a 30 min window, starting after the
    signal has stabilised).

    Algorithm
    ---------
    1. Compute positive control trajectory p_corr(t) from positive controls
       using the calibration slope c_io from the initialisation window
    2. Background subtract: f_corr(t) = f_raw(t) - f_blank(t)
    3. Calculate f_neg_base from the negative control baseline window:
       f_neg_base = mean(f_neg_corr(t)) for t in [t_nc_start, t_nc_end]
    4. Calculate relative fluorescence:
       RF(t) = (f_corr(t) - f_neg_base) / (p_corr(t) - f_neg_base)

    Parameters
    ----------
    selected_data : pd.DataFrame
        Experimental fluorescence traces.
    pos_ctrl_data : pd.DataFrame
        Positive control traces (defines RF = 1 level).
    blank_ctrl_data : pd.DataFrame
        Blank for background subtraction.
    neg_ctrl_data : pd.DataFrame
        Negative control traces (defines RF = 0 baseline).
    params : ConvertParams
        Conversion parameters:
        - neg_ctrl_start, neg_ctrl_end: Baseline window for f_neg_base
          calculation. Should be selected after initial transients have
          subsided (typically 30 min window).
        - pos_ctrl_concentrations: Known concentrations in positive controls
        - init_start, init_end: Initialisation window for calibration slope

    Returns
    -------
    dict
        Contains:
        - 'Relative Fluorescence RF(t)': Normalised readout (0 = neg ctrl, 1 = pos ctrl)
        - 'Positive Control p_corr(t) [AFU]': Reference trajectory
        - 'Background-Corrected f_corr(t) [AFU]': Blank-subtracted experimental data
        - 'summary': Statistics including f_neg_base, conversion quality
    """
    logger.info("Normalised SD conversion: RF(t) = (f_corr - f_neg_base) / (p_corr - f_neg_base)")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    _require_df(selected_data, "selected_data")
    _require_df(pos_ctrl_data, "pos_ctrl_data")
    _require_df(blank_ctrl_data, "blank_ctrl_data")
    _require_df(neg_ctrl_data, "neg_ctrl_data")

    if params.neg_ctrl_start is None or params.neg_ctrl_end is None:
        raise ValueError("Negative control window required")
    if not params.pos_ctrl_concentrations:
        raise ValueError("Positive control concentrations required")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    time_col = _find_time_col(selected_data)
    time_vec = pd.to_numeric(selected_data[time_col], errors="coerce").to_numpy(dtype=np.float64)
    n_times = len(time_vec)
    wells = list(selected_data.columns[2:])
    neg_wells = list(neg_ctrl_data.columns[2:])

    logger.info("Neg ctrl window: [%.1f, %.1f] min", params.neg_ctrl_start, params.neg_ctrl_end)

    # -------------------------------------------------------------------------
    # Compute positive control trajectory p_corr(t)
    # -------------------------------------------------------------------------
    init_slope = compute_initialisation_slope(pos_ctrl_data, blank_ctrl_data, params)
    p_corr, posctrl_warning = compute_normalised_posctrl(pos_ctrl_data, blank_ctrl_data, params, init_slope)
    p_corr_vec = p_corr.iloc[:, 2].values

    # -------------------------------------------------------------------------
    # Background subtraction: f_corr = f_raw - f_blank
    # -------------------------------------------------------------------------
    blank_vec = blank_ctrl_data.iloc[:, 2].values

    f_corr = selected_data.copy()
    f_corr.iloc[:, 2:] = selected_data.iloc[:, 2:].values - blank_vec[:, None]

    f_neg = neg_ctrl_data.copy()
    f_neg.iloc[:, 2:] = neg_ctrl_data.iloc[:, 2:].values - blank_vec[:, None]

    # -------------------------------------------------------------------------
    # Negative control baseline: f_neg_base = mean(f_neg) in window
    # -------------------------------------------------------------------------
    mask_nc = (time_vec >= params.neg_ctrl_start) & (time_vec <= params.neg_ctrl_end)
    f_neg_mean = f_neg.iloc[:, 2:].mean(axis=1).values
    f_neg_base = float(np.nanmean(f_neg_mean[mask_nc]))

    if not np.isfinite(f_neg_base):
        raise ValueError("No valid data in negative control window")

    logger.info("f_neg_base = %.2f AFU", f_neg_base)

    # -------------------------------------------------------------------------
    # Dynamic range and relative fluorescence
    # -------------------------------------------------------------------------
    dyn_range = p_corr_vec - f_neg_base

    if np.any(dyn_range <= 0):
        n_invalid = np.sum(dyn_range <= 0)
        logger.warning("Positive control below baseline at %d timepoints", n_invalid)
        dyn_range = np.where(dyn_range <= 0, np.nan, dyn_range)

    # RF(t) = (f_corr - f_neg_base) / (p_corr - f_neg_base)
    RF = f_corr.copy()
    RF.iloc[:, 2:] = (f_corr.iloc[:, 2:].values - f_neg_base) / dyn_range[:, None]

    # -------------------------------------------------------------------------
    # Build output DataFrame
    # -------------------------------------------------------------------------
    relative_fluorescence = pd.DataFrame({'Well': ['RF'] * n_times, time_col: time_vec})

    failed_wells = []
    RF_values = []
    wells_with_high_negatives = []

    for well in wells:
        try:
            RF_well = pd.to_numeric(RF[well], errors="coerce").to_numpy(dtype=np.float64)

            if not np.any(np.isfinite(RF_well)):
                raise ValueError("No valid data")

            relative_fluorescence[well] = RF_well
            RF_values.extend(RF_well[np.isfinite(RF_well)])

            # Check for over-correction (>20% negative values)
            valid = RF_well[np.isfinite(RF_well)]
            frac_neg = np.sum(valid < 0) / len(valid) if len(valid) > 0 else 0
            if frac_neg > 0.2:
                wells_with_high_negatives.append(well)
                logger.warning("Well %s: %.1f%% negative values", well, frac_neg * 100)

        except Exception as e:
            logger.error("Well %s: %s", well, e)
            failed_wells.append(well)
            relative_fluorescence[well] = np.nan

    # Process negative control wells (should yield ~0)
    RF_neg = (f_neg.iloc[:, 2:].values - f_neg_base) / dyn_range[:, None]
    for i, neg_well in enumerate(neg_wells):
        relative_fluorescence[f"NegCtrl_{neg_well}"] = RF_neg[:, i]

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    n_success = len(wells) - len(failed_wells)
    RF_array = np.array(RF_values) if RF_values else np.array([0.0])

    logger.info("Complete: %d/%d wells, RF range [%.3f, %.3f]",
                n_success, len(wells), np.min(RF_array), np.max(RF_array))

    summary = {
        'RF_min': float(np.min(RF_array)) if RF_values else 0.0,
        'RF_mean': float(np.mean(RF_array)) if RF_values else 0.0,
        'RF_max': float(np.max(RF_array)) if RF_values else 0.0,
        'f_neg_base_AFU': f_neg_base,
        'neg_ctrl_window': [params.neg_ctrl_start, params.neg_ctrl_end],
        'n_successful': n_success,
        'n_failed': len(failed_wells),
        'failed_wells': failed_wells,
        'wells_with_high_negatives': wells_with_high_negatives,
        'posctrl_warning': posctrl_warning,
        'fraction_negative_values': float(np.sum(RF_array < 0) / len(RF_array)) if len(RF_array) > 0 else 0.0,
    }

    return {
        'Relative Fluorescence RF(t)': relative_fluorescence,
        'Positive Control p_corr(t) [AFU]': p_corr,
        'Background-Corrected f_corr(t) [AFU]': f_corr,
        'summary': summary,
    }