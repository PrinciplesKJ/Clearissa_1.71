"""
Internal TMSD Conversion Module (Step 3)
========================================

Converts fluorescence (AFU) to species concentrations for internal
toehold-mediated strand displacement (Internal TMSD) reactions, following
the methodology in the supplementary material.

Experimental Context
--------------------
Internal TMSD (step 3) follows the reaction:

    F + ABT -> ABF + T

where:
    - F:   Fuel strand. Pre-incubated in the wells before the reaction starts.
    - ABT: Substrate complex (quenched). Injected at time t_inj to initiate
           the reaction. Exhibits residual fluorescence due to incomplete
           quenching.
    - ABF: Fully fluorescent product.
    - T:   Displaced strand.

The injection of ABT produces an instantaneous fluorescence jump due to
residual ABT fluorescence. This injection offset is corrected using a
negative control well containing ABT only (no Fuel).

Positive-control wells contain the fully fluorescent product ABF at known
concentrations, used to construct the reference trajectory p_corr(t).

Signal Model
------------
After preprocessing (blank subtraction and positive control normalisation),
the concentration-referenced signal h(t) in nM is given by:

    h(t) = [ABF](t) + beta * [ABT](t)

where beta is the fractional fluorescence of ABT relative to ABF (dimensionless).

Applying mass conservation [ABT]_0 = [ABT](t) + [ABF](t), we obtain:

    [ABF](t) = (h(t) - beta * [ABT]_0) / (1 - beta)

Conversion Procedure
--------------------
Unlike standard TMSD, the initial timepoints cannot be used to estimate beta
because the ABT complex is not in solution prior to injection. Instead, a
negative control well (containing ABT but no Fuel) is used, in which no product
forms and thus h_neg(t) = beta * [ABT]_0.

1. Determine [ABT]_0 per well from the oversaturation plateau:
   [ABT]_0 can be inferred from the signal after oversaturation with excess F,
   which drives complete conversion of ABT to ABF:
       [ABT]_0 = mean(h_exp(t)) for t in [t_over_start, t_over_end]

2. Estimate a single beta from the negative control:
   Define:
       h_neg_base = mean(h_neg(t)) for t in [t_nc_start, t_nc_end]

   where [t_nc_start, t_nc_end] refers to a 30 min interval selected after
   signal stabilisation in the negative-control trace.

   To obtain a single beta for the experiment, use a robust plate-level
   estimate of the initial substrate concentration:
       [ABT]_0,plate = median([ABT]_0,well) across experimental wells
       beta = h_neg_base / [ABT]_0,plate

   This differs from standard TMSD analysis, where beta is estimated
   separately for each reaction well.

3. Reconstruct concentrations using the global beta:
   Having identified [ABT]_0 per well and a global beta, apply the conversion
   equation to infer [ABF](t) over time for each well:
       [ABF](t) = (h_exp(t) - beta * [ABT]_0,well) / (1 - beta)
       [ABT](t) = [ABT]_0,well - [ABF](t)

The returned "Normalised Signal h(t)" corresponds to the per-well corrected
signal:
    h_corr(t) = h_exp(t) - beta * [ABT]_0,well
"""

from typing import Dict
import logging
import numpy as np
import pandas as pd

from .params import ConvertParams
from .conversion_helpers import (
    _require_df,
    _find_time_col,
    find_closest_timepoint,
    compute_initialisation_slope,
    compute_normalised_posctrl,
    mean_trace_in_window,
)

logger = logging.getLogger(__name__)


def convert_internal_tmsd_to_conc(
        selected_data: pd.DataFrame,
        pos_ctrl_data: pd.DataFrame,
        blank_ctrl_data: pd.DataFrame,
        neg_ctrl_data: pd.DataFrame,
        params: ConvertParams,
) -> Dict[str, pd.DataFrame]:
    """
    Convert fluorescence to concentration for Internal TMSD reactions.

    Implements the Internal TMSD conversion methodology (step 3) for the
    reaction:

        F + ABT -> ABF + T

    where F (Fuel) is pre-incubated and ABT is injected to initiate the
    reaction. ABF is the fully fluorescent product.

    Algorithm
    ---------
    1. Compute positive control trajectory p_corr(t) from ABF controls using
       the calibration slope c_io from the initialisation window
    2. Background subtract and normalise all data to obtain h(t) in nM:
       h(t) = (f_raw(t) - f_blank(t)) / p_corr(t) * C_ref
    3. Calculate h_neg_base from a negative control window (ABT only, no Fuel).
       The window [t_nc_start, t_nc_end] is typically 30 min after signal
       stabilisation in the negative-control trace.
    4. Estimate [ABT]_0 per experimental well from its oversaturation plateau:
       [ABT]_0 = mean(h_exp(t)) for t in [t_over_start, t_over_end]
    5. Compute a single beta for the plate using the median [ABT]_0:
       [ABT]_0,plate = median([ABT]_0,well) across experimental wells
       beta = h_neg_base / [ABT]_0,plate
    6. Convert each experimental well using the global beta and per-well [ABT]_0:
       [ABF](t) = (h_exp(t) - beta * [ABT]_0,well) / (1 - beta)
       [ABT](t) = [ABT]_0,well - [ABF](t)

    Parameters
    ----------
    selected_data : pd.DataFrame
        Experimental fluorescence traces (Fuel pre-incubated, ABT injected).
    pos_ctrl_data : pd.DataFrame
        Positive control wells containing ABF at known concentrations.
        Used to construct the reference trajectory p_corr(t).
    blank_ctrl_data : pd.DataFrame
        Blank wells for background subtraction.
    neg_ctrl_data : pd.DataFrame
        Negative control wells (ABT injected, no Fuel) for beta estimation.
        In these wells, no product forms, so h_neg(t) = beta * [ABT]_0.
    params : ConvertParams
        Conversion parameters:
        - trigger: Reaction trigger timepoint (ABT injection)
        - injection_primary: Time marking end of oversaturation window
        - neg_ctrl_start, neg_ctrl_end: 30 min window for h_neg_base calculation
          (selected after signal stabilisation)
        - pos_ctrl_concentrations: Known ABF concentrations in positive controls
        - c_ref: Reference concentration C_ref for scaling (default 10 nM)

    Returns
    -------
    dict
        Contains:
        - 'Product [ABF] Concentration [nM]': Product concentration traces
        - 'Substrate [ABT] Concentration [nM]': Remaining substrate traces
        - 'Initial Substrate [ABT]_0 [nM]': Per-well initial substrate values
        - 'Positive Control p(t) [AFU]': Reference trajectory p_corr(t)
        - 'Normalised Signal h(t) [nM]': Per-well corrected signal h_corr(t)
        - 'summary': Statistics including beta, [ABT]_0,plate, h_neg_base

    Raises
    ------
    ValueError
        If required parameters are missing or data validation fails.
    """
    logger.info("Internal TMSD conversion: F + ABT -> ABF + T")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    _require_df(selected_data, "selected_data")
    _require_df(pos_ctrl_data, "pos_ctrl_data (ABF positive controls)")
    _require_df(blank_ctrl_data, "blank_ctrl_data")
    _require_df(neg_ctrl_data, "neg_ctrl_data (ABT only, no Fuel)")

    if params.injection_primary is None:
        raise ValueError("Oversaturation time (injection_primary) required for [ABT]_0 estimation")
    if params.neg_ctrl_start is None or params.neg_ctrl_end is None:
        raise ValueError("Negative control window (neg_ctrl_start, neg_ctrl_end) required for beta estimation")
    if params.trigger is None:
        raise ValueError("Trigger timepoint required for concentration conversion")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    time_col = _find_time_col(selected_data)
    time_vec = pd.to_numeric(selected_data[time_col], errors="coerce").to_numpy(dtype=np.float64)
    n_times = len(time_vec)
    wells = list(selected_data.columns[2:])
    neg_wells = list(neg_ctrl_data.columns[2:])

    C_ref = params.c_ref if params.c_ref else 10.0

    # Time windows
    t_over_end = find_closest_timepoint(selected_data, params.injection_primary)
    t_over_start = find_closest_timepoint(selected_data, t_over_end - 15.0)

    # Trigger timepoint: concentration conversion only valid for t >= trigger
    t_trigger = find_closest_timepoint(selected_data, params.trigger)
    trigger_idx = int(np.argmin(np.abs(time_vec - t_trigger)))

    logger.info("Trigger: %.1f min, Neg ctrl window: [%.1f, %.1f] min, Oversat window: [%.1f, %.1f] min",
                t_trigger, params.neg_ctrl_start, params.neg_ctrl_end, t_over_start, t_over_end)

    # -------------------------------------------------------------------------
    # Compute positive control trajectory p_corr(t)
    # -------------------------------------------------------------------------
    init_slope = compute_initialisation_slope(pos_ctrl_data, blank_ctrl_data, params)
    p_of_t, posctrl_warning = compute_normalised_posctrl(pos_ctrl_data, blank_ctrl_data, params, init_slope)
    p_vec = p_of_t.iloc[:, 2].replace(0, np.nan).values

    # -------------------------------------------------------------------------
    # Background subtraction and normalisation
    # -------------------------------------------------------------------------
    blank_vec = blank_ctrl_data.iloc[:, 2].values

    def to_normalised(df: pd.DataFrame) -> pd.DataFrame:
        """Background subtract and normalise to concentration units."""
        result = df.copy()
        result.iloc[:, 2:] = (df.iloc[:, 2:].values - blank_vec[:, None]) / p_vec[:, None] * C_ref
        return result

    h_exp = to_normalised(selected_data)
    h_neg = to_normalised(neg_ctrl_data)

    # -------------------------------------------------------------------------
    # Negative control baseline
    # -------------------------------------------------------------------------
    h_neg_mean = h_neg.iloc[:, 2:].mean(axis=1).values
    mask_nc = (time_vec >= params.neg_ctrl_start) & (time_vec <= params.neg_ctrl_end)
    h_neg_base = float(np.nanmean(h_neg_mean[mask_nc]))

    # -------------------------------------------------------------------------
    # Per-well [ABT]_0 estimation and beta calculation
    # -------------------------------------------------------------------------
    conc_ABF = pd.DataFrame({'Well': ['[ABF]'] * n_times, time_col: time_vec})
    conc_ABT = pd.DataFrame({'Well': ['[ABT]'] * n_times, time_col: time_vec})
    conc_ABT0 = pd.DataFrame({'Well': ['[ABT]_0'] * n_times, time_col: time_vec})

    failed_wells = []
    ABT0_values = []
    ABT0_by_well: Dict[str, float] = {}

    # Pass 1: estimate [ABT]_0 per experimental well
    for well in wells:
        try:
            h_exp_well = pd.to_numeric(h_exp[well], errors="coerce").to_numpy(dtype=np.float64)
            ABT0 = mean_trace_in_window(h_exp_well, time_vec, t_over_start, t_over_end)

            if not np.isfinite(ABT0) or ABT0 <= 0:
                raise ValueError("Invalid [ABT]_0 = %.4f" % ABT0)

            ABT0_by_well[well] = float(ABT0)
            ABT0_values.append(float(ABT0))

        except Exception as e:
            logger.error("Well %s failed [ABT]_0 estimation: %s", well, e)
            failed_wells.append(well)
            ABT0_by_well[well] = np.nan

    if not ABT0_values:
        raise ValueError("No valid [ABT]_0 estimates available")

    # Common beta from robust global [ABT]_0 estimate
    ABT0_global = float(np.nanmedian(np.array(ABT0_values, dtype=np.float64)))
    beta_global = float(h_neg_base / ABT0_global)

    if not np.isfinite(beta_global):
        raise ValueError("Invalid beta_global = %.4f" % beta_global)
    if abs(1.0 - beta_global) < 1e-3:
        raise ValueError("beta_global = %.4f too close to 1.0" % beta_global)

    if beta_global < 0.05 or beta_global > 0.5:
        logger.warning("beta_global = %.3f outside typical range [0.05, 0.5]", beta_global)

    logger.info("beta_global = %.4f, [ABT]_0 median = %.2f nM, h_neg_base = %.2f nM",
                beta_global, ABT0_global, h_neg_base)

    # Pass 2: convert each experimental well
    for well in wells:
        if well in failed_wells:
            conc_ABF[well] = np.nan
            conc_ABT[well] = np.nan
            conc_ABT0[well] = np.nan
            continue

        try:
            ABT0 = float(ABT0_by_well[well])
            h_exp_well = pd.to_numeric(h_exp[well], errors="coerce").to_numpy(dtype=np.float64)

            offset_term = beta_global * ABT0

            # [ABF](t) = (h_exp(t) - beta*[ABT]_0,well) / (1 - beta)
            ABF_t = (h_exp_well - offset_term) / (1.0 - beta_global)

            # [ABT](t) = [ABT]_0 - [ABF](t)
            ABT_t = ABT0 - ABF_t

            # Mask pre-trigger values to NaN (conversion not valid before ABT injection)
            ABF_t[:trigger_idx] = np.nan
            ABT_t[:trigger_idx] = np.nan

            conc_ABF[well] = ABF_t
            conc_ABT[well] = ABT_t
            conc_ABT0[well] = ABT0

        except Exception as e:
            logger.error("Well %s failed conversion: %s", well, e)
            failed_wells.append(well)
            conc_ABF[well] = np.nan
            conc_ABT[well] = np.nan
            conc_ABT0[well] = np.nan

    # Construct per-well corrected signal h_corr(t)
    h_corr = h_exp.copy()
    for well in wells:
        if well in failed_wells:
            h_corr[well] = np.nan
        else:
            h_corr[well] = pd.to_numeric(h_exp[well], errors="coerce") - beta_global * float(ABT0_by_well[well])

    # Process negative control wells
    for neg_well in neg_wells:
        h_neg_w = pd.to_numeric(h_neg[neg_well], errors="coerce").to_numpy(dtype=np.float64)
        ABF_neg = (h_neg_w - beta_global * ABT0_global) / (1.0 - beta_global)
        conc_ABF["NegCtrl_%s" % neg_well] = ABF_neg

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    n_success = len([w for w in wells if w not in failed_wells])

    logger.info("Conversion complete: %d/%d wells", n_success, len(wells))

    ABF_all = conc_ABF.iloc[:, 2:].values.flatten()
    ABF_finite = ABF_all[np.isfinite(ABF_all)]

    summary = {
        'ABF_min_nM': float(np.min(ABF_finite)) if len(ABF_finite) > 0 else 0.0,
        'ABF_mean_nM': float(np.mean(ABF_finite)) if len(ABF_finite) > 0 else 0.0,
        'ABF_max_nM': float(np.max(ABF_finite)) if len(ABF_finite) > 0 else 0.0,
        'ABT_0_median_nM': ABT0_global,
        'ABT_0_mean_nM': float(np.mean(ABT0_values)) if ABT0_values else 0.0,
        'ABT_0_std_nM': float(np.std(ABT0_values)) if ABT0_values else 0.0,
        'beta_global': beta_global,
        'h_neg_base_nM': h_neg_base,
        'trigger_time': t_trigger,
        'oversaturation_window': [t_over_start, t_over_end],
        'neg_ctrl_window': [params.neg_ctrl_start, params.neg_ctrl_end],
        'n_successful': n_success,
        'n_failed': len(set(failed_wells)),
        'failed_wells': sorted(list(set(failed_wells))),
        'posctrl_warning': posctrl_warning,
    }

    return {
        'Product [ABF] Concentration [nM]': conc_ABF,
        'Substrate [ABT] Concentration [nM]': conc_ABT,
        'Initial Substrate [ABT]_0 [nM]': conc_ABT0,
        'Positive Control p(t) [AFU]': p_of_t,
        'Normalised Signal h(t) [nM]': h_corr,
        'summary': summary,
    }