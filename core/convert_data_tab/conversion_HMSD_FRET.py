"""
HMSD/FRET Conversion Module (Step 2)
====================================

Converts donor channel fluorescence to species concentrations for HMSD
(step 2) FRET reactions, following the methodology in the supplementary
material.

Experimental Context
--------------------
HMSD (step 2) is monitored in the donor channel of a donor-acceptor FRET pair.

Reaction: S1-T + R1 -> S1-T-R1

The donor-labelled complex S1-T is initially present in solution and provides
the bright donor reference signal. Upon HMSD reaction with R1, a ternary
complex forms in which donor emission is quenched by FRET. Consequently, the
donor-channel fluorescence decreases as product accumulates.

A donor reference control containing pre-annealed S1-T at a known concentration
(typically 10 nM) is used for donor-channel normalisation. This control contains
0 nM product and serves as the normalisation reference.

Signal Model
------------
The concentration-referenced donor signal h(t) is obtained by blank subtraction
followed by pointwise normalisation to the blank-subtracted S1-T reference
trajectory and scaling by its known concentration:

    h(t) = g(t) * [S1-T]_ref

where g(t) = f_corr(t) / f_ref_corr(t) is the pointwise ratio of blank-corrected
experimental trace to the blank-corrected donor reference control.

Concentration inference uses the linear signal model:

    h(t) = [S1-T](t) + beta * [S1-T-R1](t)

where beta is the residual donor brightness of the product S1-T-R1 relative
to S1-T. Assuming conservation of the donor-labelled complex:

    [S1-T]_0 = [S1-T](t) + [S1-T-R1](t)

we obtain:

    [S1-T-R1](t) = ([S1-T]_0 - h(t)) / (1 - beta)

Beta Estimation
---------------
The residual brightness factor beta is estimated from donor-channel measurements
of pre-annealed control complexes:

    beta = mean(p_corr_prod(t)) / mean(f_ref_corr(t))  for t in init window

where p_corr_prod(t) is the drift-corrected product control trajectory and
f_ref_corr(t) is the blank-subtracted donor reference control. Because both
are expressed on the same concentration-referenced scale, the choice of
reference concentration cancels in the ratio and beta is treated as
concentration-independent.
"""

import logging
import numpy as np
import pandas as pd

from .params import ConvertParams
from .conversion_helpers import (
    _require_df,
    _find_time_col,
    compute_initialisation_slope,
    compute_normalised_posctrl,
    filter_by_tag,
    divide_by_vector,
    bg_subtract,
    ensure_float_block,
    assemble_species_tables,
    mean_signal_in_window,
    get_channel_tag,
    validate_channel_found,
)

logger = logging.getLogger(__name__)


def convert_fret_onestep_to_conc(
        selected_data: pd.DataFrame,
        pos_ctrl_data: pd.DataFrame,
        blank_ctrl_data: pd.DataFrame,
        donor_data: pd.DataFrame,
        acceptor_data: pd.DataFrame = None,
        params: ConvertParams = None,
) -> dict:
    """
    Convert donor fluorescence to species concentrations for HMSD/FRET reactions.

    Implements the HMSD conversion methodology (step 2) for the reaction:

        S1-T + R1 -> S1-T-R1

    where S1-T is the donor-labelled substrate-template complex and S1-T-R1
    is the FRET-quenched product.

    Algorithm
    ---------
    1. Background subtract donor reference (S1-T) and product (S1-T-R1) controls
    2. Normalise controls if multiple concentrations are provided
    3. Compute beta from ratio of control signals in the initialisation window:
       beta = mean(p_corr_prod) / mean(f_ref_corr) for t in init window
    4. Background subtract and normalise experimental traces to h(t):
       g(t) = f_corr(t) / f_ref_corr(t), then h(t) = g(t) * [S1-T]_ref
    5. Estimate [S1-T]_0 per well from h(t) in the initialisation window
    6. Convert: [S1-T-R1](t) = ([S1-T]_0 - h(t)) / (1 - beta)

    Parameters
    ----------
    selected_data : pd.DataFrame
        Experimental fluorescence data (donor channel).
    pos_ctrl_data : pd.DataFrame
        Product control (S1-T-R1) for beta calculation. Processed using
        positive control normalisation if multiple concentrations provided.
    blank_ctrl_data : pd.DataFrame
        Blank for background subtraction.
    donor_data : pd.DataFrame
        Donor reference control (S1-T) for beta calculation and normalisation.
        Serves the same functional role as p_corr(t) in TMSD conversion.
    acceptor_data : pd.DataFrame, optional
        Acceptor channel data (not used in current implementation as donor
        channel provides better signal-to-noise ratio).
    params : ConvertParams
        Conversion parameters including init_start, init_end, concentrations.

    Returns
    -------
    dict
        Contains:
        - 'Product [S1-T-R1] Concentration [nM]': Product concentration traces
        - 'Free Donor [S1-T] Concentration [nM]': Remaining substrate traces
        - 'Initial Donor [S1-T]_0 [nM]': Per-well initial substrate values
        - 'Donor Ratio g(t)': Pointwise ratio to donor reference
        - 'Donor Reference p(t) [AFU]': Normalised donor reference trajectory
        - 'summary': Statistics including beta, [S1-T]_0, and conversion quality
    """
    logger.info("Starting HMSD/FRET conversion (S1-T + R1 -> S1-T-R1)")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    _require_df(selected_data, "selected_data")
    _require_df(pos_ctrl_data, "pos_ctrl_data (S1-T-R1 product controls)")
    _require_df(blank_ctrl_data, "blank_ctrl_data")
    _require_df(donor_data, "donor_data (S1-T donor reference controls)")

    if not isinstance(params, ConvertParams):
        raise TypeError("params must be a ConvertParams instance")

    if params.init_start is None or params.init_end is None:
        raise ValueError("Initialisation window (init_start, init_end) is required")

    # Extract channel metadata
    extra = getattr(params, 'extra', {})
    selected_channels = extra.get('selected_channels', {})
    channel_detection = extra.get('channel_detection', {})

    donor_tag, donor_matches = get_channel_tag(selected_channels, channel_detection, "donor")

    # Validate that donor channel was found in the data
    validate_channel_found("Donor", donor_tag, donor_matches, donor_data)

    logger.info("Using donor channel: %s", donor_tag)
    logger.info("Initialisation window: [%.2f, %.2f] min", params.init_start, params.init_end)

    # -------------------------------------------------------------------------
    # Extract channel-specific data
    # -------------------------------------------------------------------------
    donor_ctrl_raw = filter_by_tag(donor_data, donor_tag)
    product_ctrl_raw = filter_by_tag(pos_ctrl_data, donor_tag)
    blank_raw = filter_by_tag(blank_ctrl_data, donor_tag)
    exp_raw = filter_by_tag(selected_data, donor_tag)

    # Validate that channel filtering returned data
    _channel_hint = (
        "Go to the 'Options' tab and set the wavelengths for "
        "Donor, Acceptor, and FRET so that each tag is a substring "
        "of the corresponding channel label in your data "
        "(current tag: '%s')." % donor_tag
    )

    def _labels(df):
        """Extract unique Well labels for diagnostic output."""
        if df is not None and "Well" in getattr(df, "columns", []):
            return sorted(df["Well"].astype(str).unique().tolist())
        return []

    if donor_ctrl_raw.empty or donor_ctrl_raw.shape[1] <= 2:
        labels = _labels(donor_data)
        raise ValueError(
            "No donor reference control data found for channel '%s'.\n%s"
            "\n\nLabels in donor data:\n  %s"
            % (donor_tag, _channel_hint, "\n  ".join(labels) if labels else "(none)")
        )
    if product_ctrl_raw.empty or product_ctrl_raw.shape[1] <= 2:
        labels = _labels(pos_ctrl_data)
        raise ValueError(
            "No product control data found for channel '%s'.\n%s"
            "\n\nLabels in positive control data:\n  %s"
            % (donor_tag, _channel_hint, "\n  ".join(labels) if labels else "(none)")
        )
    if blank_raw.empty or blank_raw.shape[1] <= 2:
        labels = _labels(blank_ctrl_data)
        raise ValueError(
            "No blank data found for channel '%s'.\n%s"
            "\n\nLabels in blank data:\n  %s"
            % (donor_tag, _channel_hint, "\n  ".join(labels) if labels else "(none)")
        )
    if exp_raw.empty or exp_raw.shape[1] <= 2:
        labels = _labels(selected_data)
        raise ValueError(
            "No experimental data found for channel '%s'.\n%s"
            "\n\nLabels in experimental data:\n  %s"
            % (donor_tag, _channel_hint, "\n  ".join(labels) if labels else "(none)")
        )

    time_col = _find_time_col(donor_ctrl_raw)

    n_donor_controls = donor_ctrl_raw.shape[1] - 2
    n_product_controls = product_ctrl_raw.shape[1] - 2

    logger.info(
        "Control counts: %d donor reference (S1-T), %d product (S1-T-R1)",
        n_donor_controls, n_product_controls
    )

    # -------------------------------------------------------------------------
    # Normalise donor reference controls (S1-T)
    # -------------------------------------------------------------------------
    donor_ctrl_bg = bg_subtract(donor_ctrl_raw, blank_raw)
    donor_warning = None

    if n_donor_controls == 1:
        logger.debug("Single donor reference (S1-T) control provided")
        donor_normalised = donor_ctrl_bg.copy()
        data_cols = [c for c in donor_normalised.columns if c not in ['Well', time_col]]
        donor_normalised.rename(columns={data_cols[0]: 'Positive Control'}, inplace=True)
    else:
        if not params.donor_ctrl_concentrations:
            raise ValueError(
                "Multiple donor reference controls provided but no concentrations specified. "
                "Please provide donor_ctrl_concentrations in params."
            )
        donor_init_slope = compute_initialisation_slope(donor_ctrl_raw, blank_raw, params)
        donor_normalised, donor_warning = compute_normalised_posctrl(
            donor_ctrl_raw, blank_raw, params, donor_init_slope
        )

    # Calculate mean donor signal in init window (from normalised trajectory)
    mean_donor_signal = mean_signal_in_window(
        donor_normalised, time_col, params.init_start, params.init_end
    )
    logger.info("Donor reference (S1-T): mean signal = %.2f AFU", mean_donor_signal)

    # -------------------------------------------------------------------------
    # Normalise product controls (S1-T-R1)
    # -------------------------------------------------------------------------
    product_ctrl_bg = bg_subtract(product_ctrl_raw, blank_raw)
    product_warning = None

    if n_product_controls == 1:
        logger.debug("Single product (S1-T-R1) control provided")
        product_normalised = product_ctrl_bg.copy()
        data_cols = [c for c in product_normalised.columns if c not in ['Well', time_col]]
        product_normalised.rename(columns={data_cols[0]: 'Positive Control'}, inplace=True)
    else:
        if not params.pos_ctrl_concentrations:
            raise ValueError(
                "Multiple product controls provided but no concentrations specified. "
                "Please provide pos_ctrl_concentrations in params."
            )
        product_init_slope = compute_initialisation_slope(product_ctrl_raw, blank_raw, params)
        product_normalised, product_warning = compute_normalised_posctrl(
            product_ctrl_raw, blank_raw, params, product_init_slope
        )

    # Calculate mean product signal in init window (from normalised trajectory)
    mean_product_signal = mean_signal_in_window(
        product_normalised, time_col, params.init_start, params.init_end
    )
    logger.info("Product control (S1-T-R1): mean signal = %.2f AFU", mean_product_signal)

    # -------------------------------------------------------------------------
    # Calculate beta from normalised control signals
    # -------------------------------------------------------------------------
    if abs(mean_donor_signal) < 1e-12:
        raise ValueError(
            "Donor reference signal is zero. Cannot calculate beta."
        )

    beta = mean_product_signal / mean_donor_signal

    if not np.isfinite(beta):
        raise ValueError(
            "Beta is not finite: donor=%.2f, product=%.2f" % (mean_donor_signal, mean_product_signal)
        )

    if beta < 0:
        raise ValueError("Beta is negative (%.4f). Check control assignments." % beta)

    logger.info("Beta (residual brightness) = %.4f", beta)

    if beta >= 1.0:
        logger.warning("Beta >= 1.0 suggests no FRET quenching. Check control assignments.")

    # -------------------------------------------------------------------------
    # Convert experimental fluorescence to concentration-equivalent signal h(t)
    # -------------------------------------------------------------------------
    exp_bg = bg_subtract(exp_raw, blank_raw)

    # Ratio to donor reference control
    donor_ratio = divide_by_vector(exp_bg, donor_normalised)
    donor_ratio = ensure_float_block(donor_ratio)

    # Scale to concentration units
    donor_ctrl_conc = float(np.nanmedian(params.donor_ctrl_concentrations)) \
        if params.donor_ctrl_concentrations else 10.0

    h_t = donor_ratio.copy()
    h_t.iloc[:, 2:] = donor_ratio.iloc[:, 2:] * donor_ctrl_conc

    logger.info("Donor reference concentration [S1-T]_ref = %.2f nM", donor_ctrl_conc)

    # -------------------------------------------------------------------------
    # Per-well concentration conversion
    # -------------------------------------------------------------------------
    time_vec = pd.to_numeric(h_t[time_col], errors="coerce").to_numpy(dtype=np.float64)
    mask_init = (time_vec >= params.init_start) & (time_vec <= params.init_end)

    if np.sum(mask_init) == 0:
        raise ValueError("No data in init window [%.2f, %.2f]" % (params.init_start, params.init_end))

    trace_cols = list(h_t.columns[2:])
    results = {}
    failed_wells = []
    S1_T_0_values = []
    clipping_stats = []

    for well_name in trace_cols:
        try:
            h_t_well = pd.to_numeric(h_t[well_name], errors="coerce").to_numpy(dtype=np.float64)

            # Estimate [S1-T]_0 from init window
            h_t_init = h_t_well[mask_init]
            h_t_init_valid = h_t_init[np.isfinite(h_t_init)]

            if len(h_t_init_valid) == 0:
                failed_wells.append(well_name)
                continue

            S1_T_0 = float(np.mean(h_t_init_valid))

            if not np.isfinite(S1_T_0) or S1_T_0 <= 0:
                failed_wells.append(well_name)
                continue

            S1_T_0_values.append(S1_T_0)

            # Convert: [S1-T-R1] = ([S1-T]_0 - h(t)) / (1 - beta)
            S1_T_R1_raw = (S1_T_0 - h_t_well) / (1.0 - beta)

            # Enforce lower bound
            n_clipped = int(np.sum(S1_T_R1_raw < -1.0))
            S1_T_R1 = np.maximum(S1_T_R1_raw, -1.0)
            S1_T = S1_T_0 - S1_T_R1

            clipping_stats.append({'well': well_name, 'n_clipped': n_clipped, 'n_total': len(S1_T_R1)})

            results[well_name] = pd.DataFrame({
                "Well": [well_name] * len(time_vec),
                time_col: time_vec,
                "S1_T_R1": S1_T_R1,
                "S1_T": S1_T,
                "S1_T_0": S1_T_0,
                "beta": beta
            })

        except Exception as e:
            logger.error("Well %s failed: %s", well_name, e)
            failed_wells.append(well_name)

    if not results:
        raise RuntimeError("All wells failed conversion")

    logger.info("Converted %d wells (%d failed)", len(results), len(failed_wells))

    # -------------------------------------------------------------------------
    # Assemble output tables
    # -------------------------------------------------------------------------
    species_tables = assemble_species_tables(results, ["S1_T_R1", "S1_T"], time_col)

    # [S1-T]_0 table
    S1_T_0_table = pd.DataFrame({"Well": ["[S1-T]_0"] * len(time_vec), time_col: time_vec})
    for w, df in results.items():
        S1_T_0_table[w] = np.full(len(time_vec), float(df["S1_T_0"].iloc[0]), dtype=np.float64)
    species_tables["S1_T_0"] = S1_T_0_table

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    total_clipped = sum(s['n_clipped'] for s in clipping_stats)
    total_points = sum(s['n_total'] for s in clipping_stats)

    # Combine warnings
    posctrl_warning = None
    warnings_list = []
    if donor_warning:
        warnings_list.append("Donor: %s" % donor_warning)
    if product_warning:
        warnings_list.append("Product: %s" % product_warning)
    if warnings_list:
        posctrl_warning = " | ".join(warnings_list)

    summary = {
        "beta": beta,
        "beta_calculation": {
            'method': 'runtime_controls',
            'mean_donor_signal_AFU': mean_donor_signal,
            'mean_product_signal_AFU': mean_product_signal,
            'n_donor_controls': n_donor_controls,
            'n_product_controls': n_product_controls,
            'init_window': [params.init_start, params.init_end],
        },
        "mean_S1_T_0_nM": float(np.mean(S1_T_0_values)),
        "std_S1_T_0_nM": float(np.std(S1_T_0_values)),
        "donor_ctrl_concentration_nM": donor_ctrl_conc,
        "failed_wells": failed_wells,
        "n_successful": len(results),
        "n_failed": len(failed_wells),
        "posctrl_warning": posctrl_warning,
        "total_points_clipped_low": total_clipped,
        "mean_clipped_low_per_well": total_clipped / len(results) if results else 0,
        "percent_clipped_low": 100 * total_clipped / total_points if total_points > 0 else 0,
    }

    logger.info("Conversion complete: beta=%.4f, mean [S1-T]_0=%.2f nM", beta, summary["mean_S1_T_0_nM"])

    return {
        "Product [S1-T-R1] Concentration [nM]": species_tables["S1_T_R1"],
        "Free Donor [S1-T] Concentration [nM]": species_tables["S1_T"],
        "Initial Donor [S1-T]_0 [nM]": species_tables["S1_T_0"],
        "Donor Ratio g(t)": donor_ratio,
        "Donor Reference p(t) [AFU]": donor_normalised,
        "summary": summary
    }
