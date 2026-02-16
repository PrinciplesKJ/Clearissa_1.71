"""
Reverted Readout Conversion Module (Internal-Toehold Hairpin TMSD)
=================================================================

Converts fluorescence (AFU) to concentrations for an internal-toehold hairpin
TMSD system where the measured fluorescence decreases over time because a
bright fluorescent state becomes quenched upon invader binding and subsequent
branch migration.

Physical model (abstraction for bookkeeping)
--------------------------------------------
We represent the fluorescent construct mass balance as:

    H + I -> HI

where:
- H:  bright fluorescent state (pre-invasion hairpin state)
- I:  quencher-bearing invader (non-fluorescent in the readout channel)
- HI: quenched state after invasion/branch migration (dim)

Mass conservation for the fluorescent construct:
    [H]_0 = [H](t) + [HI](t)

Signal model in concentration-referenced units
----------------------------------------------
After blank subtraction and positive-control normalisation, we define:

    h(t) = (f_raw(t) - f_blank(t)) / p_corr(t) * C_ref

Interpretation: h(t) is "H-equivalent nM" provided that p_corr(t) is built
from bright H-only calibration controls.

We assume:
    h(t) = [H](t) + beta * [HI](t)

where beta is the fractional brightness of HI relative to H (0 <= beta < 1).

Using conservation:
    h(t) = [H]_0 - (1 - beta) * [HI](t)

Therefore:
    [HI](t) = ([H]_0 - h(t)) / (1 - beta)
    [H](t)  = [H]_0 - [HI](t)

UI control assignment
---------------------
- pos_ctrl_data (UI "positive"):
    Bright H-only controls in multiple concentrations.
    Used to build p_corr(t) via the positive-control normalisation pipeline.

- neg_ctrl_data (UI "negative"):
    Quenched HI product controls in multiple concentrations.
    Normalised using their own independent calibration, then used to
    estimate beta globally from the init window.

Beta estimation
---------------
The HI controls are normalised independently using their own calibration.
In the init window (equilibrium, no reaction):

    h_HI_ctrl(init) = beta * C_HI   (HI appears dim)

So:  beta = h_HI_ctrl(init) / C_HI

This is estimated globally (across all HI control wells).

Required parameters
-------------------
- init_start, init_end:
    Pre-trigger window for estimating [H]_0 per reaction well.
- pos_ctrl_concentrations:
    Concentrations (nM) corresponding to H-only control wells in pos_ctrl_data.
    Must align with the order of pos_ctrl_data.columns[2:].
- neg_ctrl_concentrations:
    Concentrations (nM) corresponding to HI control wells in neg_ctrl_data.
    Must align with neg_ctrl_data.columns[2:].
- c_ref:
    Reference concentration for scaling in h(t).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from .params import ConvertParams
from .conversion_helpers import (
    _require_df,
    _find_time_col,
    compute_initialisation_slope,
    compute_normalised_posctrl,
    mean_trace_in_window,
)

logger = logging.getLogger(__name__)


def convert_reverted_readout_to_conc(
        selected_data: pd.DataFrame,
        pos_ctrl_data: pd.DataFrame,
        neg_ctrl_data: pd.DataFrame,
        blank_ctrl_data: pd.DataFrame,
        params: ConvertParams,
) -> Dict[str, pd.DataFrame]:
    """
    Reverted readout conversion for internal-toehold hairpin TMSD.

    Inputs
    ------
    selected_data:
        Reaction wells (fluorescence traces).
    pos_ctrl_data:
        UI "positive controls" bucket. Interpreted here as H-only controls
        (bright) in known concentrations for p_corr(t).
    neg_ctrl_data:
        UI "negative controls" bucket. Interpreted here as HI product controls
        (quenched) in known concentrations for beta estimation.
    blank_ctrl_data:
        Blank wells for background subtraction.

    Returns
    -------
    dict of DataFrames including:
    - 'Product [HI] Concentration [nM]'
    - 'Reactant [H] Concentration [nM]'
    - 'Initial Reactant [H]_0 [nM]'
    - 'Normalised Signal h(t) [nM]'
    - 'Positive Control p_corr(t) [AFU]'
    - 'HI Control Signal h_HI_ctrl(t) [nM]'
    - 'summary'
    """
    logger.info("Reverted readout conversion: internal-toehold hairpin TMSD (bright -> quenched)")

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    _require_df(selected_data, "selected_data")
    _require_df(pos_ctrl_data, "pos_ctrl_data (UI positive; interpreted as H-only controls)")
    _require_df(neg_ctrl_data, "neg_ctrl_data (UI negative; interpreted as HI controls)")
    _require_df(blank_ctrl_data, "blank_ctrl_data")

    if params.init_start is None or params.init_end is None:
        raise ValueError("Initialisation window (init_start, init_end) required for [H]_0 estimation")

    if not getattr(params, "pos_ctrl_concentrations", None):
        raise ValueError("pos_ctrl_concentrations required for H-only calibration controls (p_corr)")

    hi_concs = getattr(params, "neg_ctrl_concentrations", None)
    if not hi_concs:
        raise ValueError(
            "HI control concentrations missing. Please enter negative control (HI) concentrations."
        )

    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------
    time_col = _find_time_col(selected_data)
    time_vec = pd.to_numeric(selected_data[time_col], errors="coerce").to_numpy(dtype=np.float64)
    n_times = len(time_vec)
    wells = list(selected_data.columns[2:])

    C_ref = float(params.c_ref) if getattr(params, "c_ref", None) else 10.0

    logger.info("Init window ([H]_0): [%.1f, %.1f] min", params.init_start, params.init_end)

    # ---------------------------------------------------------------------
    # Build p_corr(t) from H-only controls (pos_ctrl_data)
    # ---------------------------------------------------------------------
    logger.info("Computing p_corr(t) from H-only controls (UI positive bucket)")

    calibration = compute_initialisation_slope(pos_ctrl_data, blank_ctrl_data, params)
    p_corr_df, posctrl_warning = compute_normalised_posctrl(pos_ctrl_data, blank_ctrl_data, params, calibration)

    p_vec = pd.to_numeric(p_corr_df.iloc[:, 2], errors="coerce").to_numpy(dtype=np.float64)
    p_vec = np.where(np.abs(p_vec) < 1e-12, np.nan, p_vec)

    # ---------------------------------------------------------------------
    # Compute h(t) for reaction wells: h(t) = (F - F_blank) / p_corr(t) * C_ref
    # ---------------------------------------------------------------------
    blank_vec = blank_ctrl_data.iloc[:, 2].values

    h_t = selected_data.copy()
    h_t.iloc[:, 2:] = (
        (selected_data.iloc[:, 2:].astype(np.float64).values - blank_vec[:, None])
        / p_vec[:, None] * C_ref
    )

    logger.info("Computed h(t) for %d reaction wells with C_ref = %.1f nM", len(wells), C_ref)

    # ---------------------------------------------------------------------
    # Estimate [H]_0 per reaction well from init window
    # Before reaction: [HI] = 0, so h(t) = [H]_0
    # ---------------------------------------------------------------------
    H0_values = {}
    for well in wells:
        h_well = pd.to_numeric(h_t[well], errors="coerce").to_numpy(dtype=np.float64)
        H0_values[well] = mean_trace_in_window(h_well, time_vec, params.init_start, params.init_end)

    logger.info("Estimated [H]_0 per well: mean = %.1f nM", np.mean(list(H0_values.values())))

    # ---------------------------------------------------------------------
    # Normalise HI controls using the H-derived p_corr(t)
    # h_HI(t) = (F_HI - F_blank) / p_corr(t) * C_ref
    #
    # This uses the SAME p_corr(t) built from H-only controls, NOT a
    # separate normalisation pipeline. The HI controls are just regular
    # data that needs to be expressed in H-equivalent nM.
    # ---------------------------------------------------------------------
    logger.info("Normalising HI controls using H-derived p_corr(t)")

    hi_time_col = _find_time_col(neg_ctrl_data)
    hi_time_vec = pd.to_numeric(neg_ctrl_data[hi_time_col], errors="coerce").to_numpy(dtype=np.float64)
    neg_wells = list(neg_ctrl_data.columns[2:])

    h_hi_ctrl = neg_ctrl_data.copy()
    neg_blank_vec = blank_ctrl_data.iloc[:, 2].values
    h_hi_ctrl.iloc[:, 2:] = (
        (neg_ctrl_data.iloc[:, 2:].astype(np.float64).values - neg_blank_vec[:, None])
        / p_vec[:, None] * C_ref
    )

    # ---------------------------------------------------------------------
    # Estimate beta globally from HI controls in the init window
    # h_HI_ctrl(init) = beta * C_HI  =>  beta = h_HI_ctrl(init) / C_HI
    # ---------------------------------------------------------------------
    beta_estimates = []
    for j, well in enumerate(neg_wells):
        h_well = pd.to_numeric(h_hi_ctrl[well], errors="coerce").to_numpy(dtype=np.float64)
        h_init = mean_trace_in_window(h_well, hi_time_vec, params.init_start, params.init_end)
        conc = hi_concs[j]
        if conc > 0 and np.isfinite(h_init):
            beta_j = h_init / conc
            beta_estimates.append(beta_j)
            logger.debug("HI well %s: h_init = %.3f, C_HI = %.1f, beta = %.4f", well, h_init, conc, beta_j)

    if not beta_estimates:
        raise ValueError("No valid HI control data points for beta estimation")

    beta = float(np.mean(beta_estimates))
    beta_std = float(np.std(beta_estimates)) if len(beta_estimates) > 1 else 0.0

    logger.info("Beta (HI/H brightness ratio) = %.4f +/- %.4f (from %d wells)",
                beta, beta_std, len(beta_estimates))

    if not np.isfinite(beta):
        raise ValueError("Beta estimation failed (non-finite result)")
    if abs(1.0 - beta) < 1e-3:
        raise ValueError("Beta = %.4f is too close to 1.0 (division unstable)" % beta)
    if beta < 0.0 or beta > 0.5:
        logger.warning("Beta = %.4f outside typical range [0.0, 0.5] for quenched product", beta)

    # ---------------------------------------------------------------------
    # Per-well concentration conversion
    # [HI](t) = ([H]_0 - h(t)) / (1 - beta)
    # [H](t)  = [H]_0 - [HI](t)
    # ---------------------------------------------------------------------
    conc_HI = pd.DataFrame({'Well': ['[HI]'] * n_times, time_col: time_vec})
    conc_H = pd.DataFrame({'Well': ['[H]'] * n_times, time_col: time_vec})
    conc_H0 = pd.DataFrame({'Well': ['[H]_0'] * n_times, time_col: time_vec})

    failed_wells = []

    for well in wells:
        try:
            h_well = pd.to_numeric(h_t[well], errors="coerce").to_numpy(dtype=np.float64)
            H0 = H0_values[well]

            if not np.isfinite(H0) or H0 <= 0:
                raise ValueError("Invalid [H]_0 = %.4f (must be positive and finite)" % H0)

            HI_t = (H0 - h_well) / (1.0 - beta)
            H_t = H0 - HI_t

            conc_HI[well] = HI_t
            conc_H[well] = H_t
            conc_H0[well] = H0

            n_neg = np.sum(HI_t < 0)
            if n_neg > 0:
                logger.warning("Well %s: %.1f%% negative [HI] values (noise)", well, 100 * n_neg / n_times)

        except Exception as e:
            logger.error("Well %s failed: %s", well, e)
            failed_wells.append(well)
            conc_HI[well] = np.nan
            conc_H[well] = np.nan
            conc_H0[well] = np.nan

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    n_success = len(wells) - len(failed_wells)
    successful_H0 = [H0_values[w] for w in wells if w not in failed_wells]

    if successful_H0:
        logger.info(
            "Conversion complete: %d/%d wells, beta = %.4f, [H]_0 = %.1f +/- %.1f nM",
            n_success, len(wells), beta,
            np.mean(successful_H0), np.std(successful_H0)
        )
    else:
        logger.warning("Conversion complete: %d/%d wells (no successful conversions)", n_success, len(wells))

    HI_all = conc_HI.iloc[:, 2:].values.flatten()
    HI_finite = HI_all[np.isfinite(HI_all)]

    summary = {
        'HI_min_nM': float(np.min(HI_finite)) if len(HI_finite) > 0 else 0.0,
        'HI_mean_nM': float(np.mean(HI_finite)) if len(HI_finite) > 0 else 0.0,
        'HI_max_nM': float(np.max(HI_finite)) if len(HI_finite) > 0 else 0.0,
        'H0_mean_nM': float(np.mean(successful_H0)) if successful_H0 else 0.0,
        'H0_std_nM': float(np.std(successful_H0)) if successful_H0 else 0.0,
        'beta': float(beta),
        'beta_std': float(beta_std),
        'init_window': [params.init_start, params.init_end],
        'n_successful': n_success,
        'n_failed': len(failed_wells),
        'failed_wells': failed_wells,
        'posctrl_warning': posctrl_warning,
    }

    return {
        'Product [HI] Concentration [nM]': conc_HI,
        'Reactant [H] Concentration [nM]': conc_H,
        'Initial Reactant [H]_0 [nM]': conc_H0,
        'Normalised Signal h(t) [nM]': h_t,
        'Positive Control p_corr(t) [AFU]': p_corr_df,
        'HI Control Signal h_HI_ctrl(t) [nM]': h_hi_ctrl,
        'summary': summary,
    }