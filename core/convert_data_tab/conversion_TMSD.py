"""
TMSD Conversion Module (Step 1)
===============================

Converts fluorescence (AFU) to species concentrations for toehold-mediated
strand displacement (TMSD) reactions, following the methodology in the
supplementary material.

Experimental Context
--------------------
The assay follows a toehold-mediated strand displacement reaction:

    Ax + B -> AB + x

where:
    - Ax: Fluorophore-quencher duplex (substrate). Exhibits residual
          fluorescence due to incomplete quenching.
    - B:  Nonlabelled invader strand. Injected to initiate the reaction.
    - AB: Displacement product. Fully fluorescent.
    - x:  Displaced strand.

Positive-control wells contain AB at known concentrations, used to construct
the reference trajectory p_corr(t) (see positive control normalisation in
conversion_helpers.py).

Signal Model
------------
After preprocessing (blank subtraction and positive control normalisation),
the concentration-referenced signal h(t) in nM is given by:

    h(t) = (f_raw(t) - f_blank(t)) / p_corr(t) * C_ref

We assume:

    h(t) = [AB](t) + beta * [Ax](t)

where beta is the fractional fluorescence of Ax relative to AB (dimensionless).

Applying mass conservation [Ax]_0 = [Ax](t) + [AB](t), we obtain:

    h(t) = [AB](t) * (1 - beta) + beta * [Ax]_0

and thus:

    [AB](t) = (h(t) - beta * [Ax]_0) / (1 - beta)

Concentration Inference Procedure (per trace)
---------------------------------------------
1. Determine [Ax]_0 by oversaturation:
   Under oversaturation by injection of excess B, full conversion Ax -> AB is
   assumed, hence h(t) = [Ax]_0 in the oversaturation plateau window:
       [Ax]_0 = mean(h(t)) for t in [t_over_start, t_over_end]

2. Estimate beta from the initialisation window:
   In the initialisation window (pre-trigger), [AB] = 0 and h(t) = beta * [Ax]_0.
   We compute:
       h_init = mean(h(t)) for t in [t_init_start, t_init_end]
       beta = h_init / [Ax]_0
   This estimate of beta is calculated separately for each reaction well.

3. Reconstruct concentrations over time:
   Having identified [Ax]_0 and beta, apply the conversion equation to infer
   [AB](t) over time.
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


def convert_tmsd_to_conc(
        selected_data: pd.DataFrame,
        pos_ctrl_data: pd.DataFrame,
        blank_ctrl_data: pd.DataFrame,
        params: ConvertParams,
) -> Dict[str, pd.DataFrame]:
    """
    Convert fluorescence to concentration for TMSD reactions.

    Implements the TMSD conversion methodology (step 1) for the reaction:

        Ax + B -> AB + x

    where Ax is the fluorophore-quencher substrate and AB is the fully
    fluorescent product.

    Algorithm
    ---------
    1. Compute positive control trajectory p_corr(t) from AB controls using
       the calibration slope c_io from the initialisation window
    2. Background subtract and normalise:
       h(t) = (f_raw(t) - f_blank(t)) / p_corr(t) * C_ref
    3. Estimate [Ax]_0 from the oversaturation plateau window (assumes full
       conversion of Ax to AB under excess B)
    4. Estimate beta = h_init / [Ax]_0 from the initialisation window
       (where [AB] = 0, so h(t) = beta * [Ax]_0)
    5. Convert: [AB](t) = (h(t) - beta * [Ax]_0) / (1 - beta)

    Parameters
    ----------
    selected_data : pd.DataFrame
        Experimental fluorescence traces (reaction wells).
    pos_ctrl_data : pd.DataFrame
        Positive control wells containing AB at known concentrations.
        Used to construct the reference trajectory p_corr(t).
    blank_ctrl_data : pd.DataFrame
        Blank wells for background subtraction.
    params : ConvertParams
        Conversion parameters:
        - injection_primary: Time marking end of oversaturation window
        - init_start, init_end: Initialisation window for beta estimation
        - pos_ctrl_concentrations: Known AB concentrations in positive controls
        - c_ref: Reference concentration C_ref for scaling (default 10 nM)

    Returns
    -------
    dict
        Contains:
        - 'Product [AB] Concentration [nM]': Product concentration traces
        - 'Initial Substrate [Ax]_0 [nM]': Per-well initial substrate values
        - 'Positive Control p(t) [AFU]': Reference trajectory p_corr(t)
        - 'Normalised Signal h(t) [nM]': Concentration-referenced signal
        - 'summary': Statistics including beta, [Ax]_0, and conversion quality

    Raises
    ------
    ValueError
        If required parameters are missing or data validation fails.
    """
    logger.info("TMSD conversion: Ax + B -> AB + x")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    _require_df(selected_data, "selected_data")
    _require_df(pos_ctrl_data, "pos_ctrl_data (AB positive controls)")
    _require_df(blank_ctrl_data, "blank_ctrl_data")

    if params.mode != "tmsd":
        raise ValueError("Requires mode='tmsd', got '%s'" % params.mode)
    if params.injection_primary is None:
        raise ValueError("Oversaturation time (injection_primary) required for [Ax]_0 estimation")
    if params.init_start is None or params.init_end is None:
        raise ValueError("Initialisation window (init_start, init_end) required for beta estimation")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    time_col = _find_time_col(selected_data)
    time_vec = pd.to_numeric(selected_data[time_col], errors="coerce").to_numpy(dtype=np.float64)
    n_times = len(time_vec)
    wells = list(selected_data.columns[2:])

    C_ref = params.c_ref if params.c_ref else 10.0

    # Time windows
    # Oversaturation window: 15 minutes before injection_primary to injection_primary
    t_over_end = find_closest_timepoint(selected_data, params.injection_primary)
    t_over_start = find_closest_timepoint(selected_data, t_over_end - 15.0)

    logger.info("Initialisation window (beta estimation): [%.1f, %.1f] min", params.init_start, params.init_end)
    logger.info("Oversaturation window ([Ax]_0 estimation): [%.1f, %.1f] min", t_over_start, t_over_end)

    # -------------------------------------------------------------------------
    # Compute positive control trajectory p_corr(t)
    # -------------------------------------------------------------------------
    logger.info("Computing positive control trajectory p_corr(t) from AB controls")
    init_slope = compute_initialisation_slope(pos_ctrl_data, blank_ctrl_data, params)
    p_of_t, posctrl_warning = compute_normalised_posctrl(pos_ctrl_data, blank_ctrl_data, params, init_slope)
    p_vec = p_of_t.iloc[:, 2].replace(0, np.nan).values

    # -------------------------------------------------------------------------
    # Background subtraction and normalisation: h(t) = (F - F_blank) / p(t) * C_ref
    # -------------------------------------------------------------------------
    blank_vec = blank_ctrl_data.iloc[:, 2].values

    h_t = selected_data.copy()
    h_t.iloc[:, 2:] = (selected_data.iloc[:, 2:].values - blank_vec[:, None]) / p_vec[:, None] * C_ref

    logger.info("Computed h(t) with C_ref = %.1f nM", C_ref)

    # -------------------------------------------------------------------------
    # Per-well concentration conversion
    # -------------------------------------------------------------------------
    conc_AB = pd.DataFrame({'Well': ['[AB]'] * n_times, time_col: time_vec})
    conc_Ax0 = pd.DataFrame({'Well': ['[Ax]_0'] * n_times, time_col: time_vec})

    failed_wells = []
    Ax0_values = []
    beta_values = []

    for well in wells:
        try:
            h_well = pd.to_numeric(h_t[well], errors="coerce").to_numpy(dtype=np.float64)

            # Step 1: Estimate [Ax]_0 from oversaturation plateau
            # Under excess B, full conversion assumed: h(t) = [Ax]_0
            Ax0 = mean_trace_in_window(h_well, time_vec, t_over_start, t_over_end)

            if not np.isfinite(Ax0) or Ax0 <= 0:
                raise ValueError("Invalid [Ax]_0 = %.4f (must be positive and finite)" % Ax0)

            # Step 2: Estimate beta from pre-trigger window
            # Before reaction: [AB] = 0, so h(t) = beta * [Ax]_0
            h_init = mean_trace_in_window(h_well, time_vec, params.init_start, params.init_end)
            beta = h_init / Ax0

            if not np.isfinite(beta):
                raise ValueError("Invalid beta = %.4f (not finite)" % beta)
            if abs(1.0 - beta) < 1e-3:
                raise ValueError("Invalid beta = %.4f (too close to 1.0, division unstable)" % beta)

            Ax0_values.append(Ax0)
            beta_values.append(beta)

            if beta < 0.05 or beta > 0.5:
                logger.warning("Well %s: beta = %.3f outside typical range [0.05, 0.5]", well, beta)

            # Step 3: Reconstruct [AB](t) = (h(t) - beta * [Ax]_0) / (1 - beta)
            AB_t = (h_well - beta * Ax0) / (1.0 - beta)

            conc_AB[well] = AB_t
            conc_Ax0[well] = Ax0

            # Warn about negative values (noise can cause small negatives)
            n_neg = np.sum(AB_t < 0)
            if n_neg > 0:
                logger.warning("Well %s: %.1f%% negative [AB] values (noise)", well, 100 * n_neg / n_times)

        except Exception as e:
            logger.error("Well %s failed: %s", well, e)
            failed_wells.append(well)
            conc_AB[well] = np.nan
            conc_Ax0[well] = np.nan

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    n_success = len(wells) - len(failed_wells)

    if beta_values:
        logger.info(
            "Conversion complete: %d/%d wells, beta = %.3f +/- %.3f, [Ax]_0 = %.1f +/- %.1f nM",
            n_success, len(wells),
            np.mean(beta_values), np.std(beta_values),
            np.mean(Ax0_values), np.std(Ax0_values)
        )
    else:
        logger.warning("Conversion complete: %d/%d wells (no successful conversions)", n_success, len(wells))

    AB_all = conc_AB.iloc[:, 2:].values.flatten()
    AB_finite = AB_all[np.isfinite(AB_all)]

    summary = {
        'AB_min_nM': float(np.min(AB_finite)) if len(AB_finite) > 0 else 0.0,
        'AB_mean_nM': float(np.mean(AB_finite)) if len(AB_finite) > 0 else 0.0,
        'AB_max_nM': float(np.max(AB_finite)) if len(AB_finite) > 0 else 0.0,
        'Ax_0_mean_nM': float(np.mean(Ax0_values)) if Ax0_values else 0.0,
        'Ax_0_std_nM': float(np.std(Ax0_values)) if Ax0_values else 0.0,
        'beta_mean': float(np.mean(beta_values)) if beta_values else None,
        'beta_std': float(np.std(beta_values)) if beta_values else None,
        'oversaturation_window': [t_over_start, t_over_end],
        'init_window': [params.init_start, params.init_end],
        'n_successful': n_success,
        'n_failed': len(failed_wells),
        'failed_wells': failed_wells,
        'posctrl_warning': posctrl_warning,
    }

    return {
        'Product [AB] Concentration [nM]': conc_AB,
        'Initial Substrate [Ax]_0 [nM]': conc_Ax0,
        'Positive Control p(t) [AFU]': p_of_t,
        'Normalised Signal h(t) [nM]': h_t,
        'summary': summary,
    }
