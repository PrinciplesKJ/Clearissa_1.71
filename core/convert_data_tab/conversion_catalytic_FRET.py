import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .params import ConvertParams
from .conversion_helpers import (
    _require_df,
    _find_time_col,
    compute_initialisation_slope,
    compute_normalised_posctrl,
    estimate_runtime_slopes,
    compute_brightness_coefficients,
    filter_by_tag,
    divide_by_vector,
    bg_subtract,
    ensure_float_block,
    estimate_initial_concentration,
    create_initial_value_table,
    assemble_species_tables,
    get_channel_tag,
    validate_channel_found,
)

logger = logging.getLogger(__name__)


def convert_catalytic_fret_2x2(
        selected_data: pd.DataFrame,
        pos_ctrl_data: pd.DataFrame,
        blank_data: pd.DataFrame,
        donor_data: pd.DataFrame,
        acceptor_data: pd.DataFrame | None = None,
        params: ConvertParams | None = None,
        b1l_ctrl_data: pd.DataFrame | None = None,
        b1aa_ctrl_data: pd.DataFrame | None = None,
) -> dict:
    """
    Convert fluorescence data to species concentrations for catalytic two-step
    DNA strand displacement reactions with FRET readout, using a 2x2 constrained
    solver on the Donor and FRET channels.

    This function uses runtime calibration exclusively, estimating slopes from
    plate-matched controls provided in the input data.

    Physical Model
    --------------
    Reaction cascade:
        1. B1L + AA -> B1AA + L   (TMSD: acceptor unquenching)
        2. B1AA + B2 -> B1B2 + AA (HMSD: FRET product formation)

    Species:
        B1L:  Quenched acceptor strand (B1 blocked by L)
        AA:   Catalyst duplex (unlabelled, recovered after reaction)
        B1AA: Intermediate with activated acceptor (B1 unquenched)
        L:    Released quencher strand (unlabelled, not tracked)
        B2:   Donor strand (initially free)
        B1B2: FRET-active product complex

    Mass Conservation
    -----------------
    Two conservation laws constrain the system:

        B1 conservation:  [B1L]_0 = [B1L](t) + [B1AA](t) + [B1B2](t)
        Donor conservation: [B2]_0 = [B2](t) + [B1B2](t)

    These reduce the problem from 4 unknowns to 2: solve for [B1B2] and [B1AA],
    then recover [B1L] and [B2] algebraically.

    Signal Processing
    -----------------
    Two-stage normalisation converts raw fluorescence to concentration-equivalent
    signals:

    **Stage 1: Ratio signals (dimensionless)**
        ratio_D(t) = (F_exp_D - F_blank_D) / (F_donor_ctrl - F_blank_D)
        ratio_F(t) = (F_exp_F - F_blank_F) / (F_pos_ctrl - F_blank_F)

    These ratios are scaled by control concentrations to give signals in nM
    referenced to the donor control species.

    **Stage 2: Concentration-equivalent signals h_c(t) [nM]**
        h_D(t) = ratio_D(t) * scaling_D
        h_F(t) = ratio_F(t) * C_ref

    where scaling_D = (m_B2_D * C_ref) / m_B1B2_D converts from B2-equivalent
    to B1B2-equivalent basis.

    Signal Model
    ------------
    Each concentration-equivalent signal h_c(t) is a linear combination of all
    species weighted by their relative brightness coefficients:

        h_c(t) = alpha_c*[B1L] + gamma_c*[B1AA] + beta_c*[B2] + kappa_c*[B1B2]

    Brightness coefficients (relative to B1B2):
        alpha_c = m(B1L, c) / m(B1B2, c)   quenched acceptor brightness
        beta_c  = m(B2, c) / m(B1B2, c)    free donor brightness
        gamma_c = m(B1AA, c) / m(B1B2, c)  activated acceptor brightness
        kappa_c = 1.0                       product reference (by definition)

    where m(species, channel) are calibration slopes [AFU/nM] estimated from
    plate-matched controls at runtime.

    Solver Strategy
    ---------------
    Substituting conservation laws yields a linear system in two unknowns:

        kappa_prime_c*[B1B2] + gamma_prime_c*[B1AA] = h_c - alpha_c*[B1L]_0 - beta_c*[B2]_0

    where:
        kappa_prime_c = kappa_c - beta_c - alpha_c  (effective product coefficient)
        gamma_prime_c = gamma_c - alpha_c           (effective intermediate coefficient)

    The 2x2 system (Donor + FRET channels) is solved at each timepoint using
    bounded least squares (scipy.optimize.lsq_linear, method='bvls').

    No pre-trigger offset correction is applied: B2_0 is estimated from the
    pre-trigger donor ratio (mean(ratio_D_init) * C_ref), which already anchors
    the baseline signal to the known initial state. An additive offset would be
    redundant because the same pre-trigger window defines both B2_0 and would
    define the offset, making them algebraically coupled.

    Solver Bounds (NOT Mass Conservation Constraints)
    -------------------------------------------------
    The solver uses relaxed box bounds on the unknowns:

        -2 <= [B1B2] <= 120 nM     (small negatives allowed for noise)
        -2 <= [B1AA] <= [B1L]_0 + 10 nM

    These bounds are NOT mass conservation constraints. Mass conservation is
    already implicitly enforced by the signal model equations themselves.
    Small negative values are permitted so that measurement noise does not
    get artificially clamped to zero, which would bias time-averaged
    estimates and obscure the true noise floor.

    After solving, [B1L] and [B2] are recovered algebraically from conservation:
        [B1L](t) = [B1L]_0 - [B1AA](t) - [B1B2](t)
        [B2](t)  = [B2]_0 - [B1B2](t)

    Initial Concentrations
    ----------------------
        [B1L]_0: User-specified (typically 10 nM)
        [B2]_0:  Estimated from pre-trigger donor ratio: mean(ratio_D) * C_ref

    Algorithm
    ---------
    1. Auto-detect Donor/FRET channel tags
    2. Estimate runtime calibration slopes from plate-matched controls
    3. Background subtract all data
    4. Compute ratio signals: ratio_D, ratio_F
    5. Convert to concentration-equivalent signals: h_D, h_F
    6. Estimate [B2]_0 from pre-trigger donor signal
    7. Per-timepoint 2x2 bounded linear solve for [B1B2], [B1AA]
    8. Recover [B1L], [B2] from conservation laws

    Returns
    -------
    dict
        Species concentration tables, ratio signals, diagnostics, and summary.
        Keys include 'B1B2 Concentration', 'B1L Concentration', etc.
    """

    # =========================================================================
    # STEP 0: VALIDATE INPUTS
    # =========================================================================
    _require_df(selected_data, "selected_data")
    _require_df(pos_ctrl_data, "pos_ctrl_data")
    _require_df(blank_data, "blank_ctrl_data")
    _require_df(donor_data, "donor_data")

    if not isinstance(params, ConvertParams):
        raise TypeError("params must be a ConvertParams instance")

    extra = getattr(params, "extra", {}) or {}
    selected_channels = extra.get("selected_channels", {})
    channel_detection = extra.get("channel_detection", {})

    # =========================================================================
    # STEP 1: GET PARAMETERS
    # =========================================================================
    B1L_0 = extra.get("b1l_initial_conc", 10.0)
    logger.info("Mass-action catalytic conversion with B1-L = %.2f nM", B1L_0)

    C_ref = 10.0
    b1l_ctrl_conc = extra.get("b1l_ctrl_concentration", C_ref)
    b1aa_ctrl_conc = extra.get("b1aa_ctrl_concentration", C_ref)

    b2_ctrl_conc = (
        float(np.nanmedian(params.donor_ctrl_concentrations))
        if params.donor_ctrl_concentrations
        else C_ref
    )

    # =========================================================================
    # STEP 2: CHANNEL SELECTION
    # =========================================================================
    donor_tag, donor_matches = get_channel_tag(selected_channels, channel_detection, "donor")
    fret_tag, fret_matches = get_channel_tag(selected_channels, channel_detection, "fret")

    validate_channel_found("Donor", donor_tag, donor_matches, selected_data)
    validate_channel_found("FRET", fret_tag, fret_matches, selected_data)

    logger.info("Using Donor='%s', FRET='%s'", donor_tag, fret_tag)

    # =========================================================================
    # STEP 3: SPLIT DATA BY CHANNEL
    # =========================================================================
    time_col = _find_time_col(selected_data)

    exp_D = filter_by_tag(selected_data, donor_tag)
    exp_F = filter_by_tag(selected_data, fret_tag)
    blank_D = filter_by_tag(blank_data, donor_tag)
    blank_F = filter_by_tag(blank_data, fret_tag)
    pos_D = filter_by_tag(pos_ctrl_data, donor_tag)
    pos_F = filter_by_tag(pos_ctrl_data, fret_tag)
    donor_only = filter_by_tag(donor_data, donor_tag)

    # =========================================================================
    # STEP 4: COMPUTE B1B2 POSITIVE-CONTROL NORMALISERS
    # =========================================================================
    init_slope_fret = compute_initialisation_slope(pos_F, blank_F, params)
    p_fret_ratio, posctrl_warning = compute_normalised_posctrl(
        pos_F, blank_F, params, init_slope_fret
    )

    init_slope_donor = compute_initialisation_slope(pos_D, blank_D, params)
    p_donor_ratio, _ = compute_normalised_posctrl(pos_D, blank_D, params, init_slope_donor)

    # =========================================================================
    # STEP 5: ESTIMATE RUNTIME CALIBRATION SLOPES
    # =========================================================================
    missing_ctrls = []
    if b1l_ctrl_data is None or (isinstance(b1l_ctrl_data, pd.DataFrame) and b1l_ctrl_data.empty):
        missing_ctrls.append("b1l_ctrl_data (B1L blocked acceptor control)")
    if b1aa_ctrl_data is None or (isinstance(b1aa_ctrl_data, pd.DataFrame) and b1aa_ctrl_data.empty):
        missing_ctrls.append("b1aa_ctrl_data (B1AA activated acceptor control)")

    if missing_ctrls:
        raise ValueError(
            "Runtime calibration requires control DataFrames: "
            f"{', '.join(missing_ctrls)}. "
            "Please provide the required controls in the Data Frame Processor."
        )

    logger.info("Estimating runtime calibration slopes from plate-matched controls")
    runtime_slopes = estimate_runtime_slopes(
        b1l_ctrl=b1l_ctrl_data,
        b1aa_ctrl=b1aa_ctrl_data,
        b2_ctrl=donor_data,
        b1b2_ctrl_donor=p_donor_ratio,
        b1b2_ctrl_fret=p_fret_ratio,
        blank_data=blank_data,
        donor_tag=donor_tag,
        fret_tag=fret_tag,
        time_col=time_col,
        init_start=params.init_start,
        init_end=params.init_end,
        b1l_conc=b1l_ctrl_conc,
        b1aa_conc=b1aa_ctrl_conc,
        b2_conc=b2_ctrl_conc,
        b1b2_conc=C_ref,
    )
    logger.info("Successfully estimated %d runtime slopes", len(runtime_slopes))

    # =========================================================================
    # STEP 6: BACKGROUND SUBTRACTION
    # =========================================================================
    bg_D_exp = bg_subtract(exp_D, blank_D)
    bg_F_exp = bg_subtract(exp_F, blank_F)
    bg_D_ctrl = bg_subtract(donor_only, blank_D)

    # =========================================================================
    # STEP 7: COMPUTE RATIO SIGNALS
    # =========================================================================
    donor_ctrl_conc = (
        float(np.nanmedian(params.donor_ctrl_concentrations))
        if params.donor_ctrl_concentrations
        else C_ref
    )
    factor_D = donor_ctrl_conc / C_ref

    donor_ratio = divide_by_vector(bg_D_exp, bg_D_ctrl)
    fret_ratio = divide_by_vector(bg_F_exp, p_fret_ratio)

    ratio_D = ensure_float_block(donor_ratio).copy()
    ratio_F = ensure_float_block(fret_ratio).copy()

    ratio_D.iloc[:, 2:] *= factor_D

    extraresults = {
        "ratio_D": ratio_D.copy(),
        "ratio_F": ratio_F.copy(),
        "pos_ctrl_ratio": p_fret_ratio.copy(),
    }

    # =========================================================================
    # STEP 8: COMPUTE BRIGHTNESS COEFFICIENTS
    # =========================================================================
    coeffs = compute_brightness_coefficients(runtime_slopes)

    alpha_D = coeffs["Donor"]["alpha"]
    beta_D = coeffs["Donor"]["beta"]
    gamma_D = coeffs["Donor"]["gamma"]
    kappa_D = coeffs["Donor"]["kappa"]
    kappa_prime_D = coeffs["Donor"]["kappa_prime"]
    gamma_prime_D = coeffs["Donor"]["gamma_prime"]

    alpha_F = coeffs["FRET"]["alpha"]
    beta_F = coeffs["FRET"]["beta"]
    gamma_F = coeffs["FRET"]["gamma"]
    kappa_F = coeffs["FRET"]["kappa"]
    kappa_prime_F = coeffs["FRET"]["kappa_prime"]
    gamma_prime_F = coeffs["FRET"]["gamma_prime"]

    logger.info(
        "Brightness coefficients - Donor: alpha=%.4f, beta=%.4f, gamma=%.4f",
        alpha_D, beta_D, gamma_D,
    )
    logger.info(
        "Brightness coefficients - FRET:  alpha=%.4f, beta=%.4f, gamma=%.4f",
        alpha_F, beta_F, gamma_F,
    )

    # =========================================================================
    # STEP 9: PREPARE TIME VECTOR AND TRIGGER
    # =========================================================================
    tcol = _find_time_col(ratio_D)  # single call, reused throughout
    time_vec = pd.to_numeric(ratio_D[tcol], errors="coerce").to_numpy(dtype=np.float64)
    n_times = len(time_vec)
    trace_cols = list(ratio_D.columns[2:])

    t_trigger = float(params.trigger) if params.trigger is not None else None

    # Pre-compute trigger mask once (reused for every well)
    if t_trigger is not None:
        post_trigger_mask = np.isfinite(time_vec) & (time_vec >= t_trigger)
        post_trigger_idx = np.where(post_trigger_mask)[0]
    else:
        post_trigger_idx = np.arange(n_times)

    # =========================================================================
    # STEP 10: COMPUTE SCALING FACTORS FOR h_c CONVERSION
    # =========================================================================
    m_B2_D = coeffs["Donor"]["m_B2"]
    m_B1B2_D = coeffs["Donor"]["m_B1B2"]
    s_h_D = (m_B2_D * C_ref) / m_B1B2_D
    s_h_F = C_ref

    # =========================================================================
    # STEP 11: PROCESS ALL WELLS
    # =========================================================================
    logger.info("Processing %d wells using 2x2 Donor+FRET solver", len(trace_cols))

    results: Dict[str, pd.DataFrame] = {}
    failed_wells: list[str] = []

    M_2x2 = np.array(
        [[kappa_prime_D, gamma_prime_D],
         [kappa_prime_F, gamma_prime_F]],
        dtype=np.float64,
    )

    # Solver bounds: relaxed box bounds for numerical stability.
    # Small negatives allowed so noise isn't clamped to zero (avoids bias).
    lb = np.array([-2.0, -2.0], dtype=np.float64)
    ub = np.array([120.0, B1L_0 + 10.0], dtype=np.float64)

    # Pre-compute the B1L_0-dependent part of the RHS baseline (constant across wells)
    baseline_D_B1L = alpha_D * B1L_0
    baseline_F_B1L = alpha_F * B1L_0

    for well_idx, col in enumerate(trace_cols):
        try:
            logger.debug("Processing well %d/%d: %s", well_idx + 1, len(trace_cols), col)

            # Estimate B2_0 from donor ratio in initialisation window
            B2_0 = estimate_initial_concentration(
                ratio_D, col, time_vec, params.init_start, params.init_end
            ) * C_ref

            # Convert ratios to concentration-equivalent signals h_c [nM]
            h_D = pd.to_numeric(ratio_D[col], errors="coerce").to_numpy(dtype=np.float64) * s_h_D
            h_F = pd.to_numeric(ratio_F[col], errors="coerce").to_numpy(dtype=np.float64) * s_h_F

            if len(h_D) != n_times or len(h_F) != n_times:
                logger.warning("Trace length mismatch for %s, skipping", col)
                failed_wells.append(col)
                continue

            # Initialise output arrays with pre-trigger defaults
            B1B2_t = np.zeros(n_times, dtype=np.float64)
            B1AA_t = np.zeros(n_times, dtype=np.float64)
            B2_t = np.full(n_times, B2_0, dtype=np.float64)
            B1L_t = np.full(n_times, B1L_0, dtype=np.float64)

            # Per-well static baseline: h_c(t) - baseline = kappa'*[B1B2] + gamma'*[B1AA]
            baseline_D = baseline_D_B1L + beta_D * B2_0
            baseline_F = baseline_F_B1L + beta_F * B2_0

            # Solve only post-trigger timepoints
            for i in post_trigger_idx:
                rhs = np.array(
                    [h_D[i] - baseline_D,
                     h_F[i] - baseline_F],
                    dtype=np.float64,
                )

                res = lsq_linear(M_2x2, rhs, bounds=(lb, ub), method="bvls")
                B1B2_t[i], B1AA_t[i] = res.x

                # Recover [B1L] and [B2] from mass conservation
                B2_t[i] = B2_0 - B1B2_t[i]
                B1L_t[i] = B1L_0 - B1AA_t[i] - B1B2_t[i]

            # Store results
            results[col] = pd.DataFrame({
                "Well": col,
                tcol: time_vec,
                "B1B2": B1B2_t,
                "B1AA": B1AA_t,
                "B1L": B1L_t,
                "B2": B2_t,
                "B2_0": B2_0,
                "B1AA_0": 0.0,
                "B1L_0": B1L_0,
            })

        except Exception as e:
            logger.error("Failed to process well '%s': %s", col, e, exc_info=True)
            failed_wells.append(col)

    logger.info("Processing complete: %d/%d wells successful", len(results), len(trace_cols))

    # =========================================================================
    # STEP 12: VALIDATE RESULTS
    # =========================================================================
    if not results:
        raise RuntimeError("All wells failed processing")

    if failed_wells:
        logger.warning("Failed wells (%d): %s", len(failed_wells), failed_wells)

    # =========================================================================
    # STEP 13: ASSEMBLE OUTPUT TABLES
    # =========================================================================
    logger.info("Assembling output tables")

    species = assemble_species_tables(results, ["B1B2", "B1L", "B1AA", "B2"], tcol)
    species["B2_0"] = create_initial_value_table(results, "B2_0", tcol)
    species["B1AA_0"] = create_initial_value_table(results, "B1AA_0", tcol)
    species["B1L_0"] = create_initial_value_table(results, "B1L_0", tcol)

    # =========================================================================
    # STEP 14: CREATE DIAGNOSTICS TABLE
    # =========================================================================
    diagnostics_rows = []
    for col in trace_cols:
        if col in results:
            wdf = results[col]
            diagnostics_rows.append({
                "Well": col,
                "B2_0_nM": float(wdf["B2_0"].iloc[0]),
                "B1AA_0_nM": float(wdf["B1AA_0"].iloc[0]),
                "B1L_0_nM": float(wdf["B1L_0"].iloc[0]),
                "status": "success",
            })
        else:
            diagnostics_rows.append({
                "Well": col,
                "B2_0_nM": np.nan,
                "B1AA_0_nM": np.nan,
                "B1L_0_nM": np.nan,
                "status": "failed",
            })

    diagnostics_df = pd.DataFrame(diagnostics_rows)

    # =========================================================================
    # STEP 15: PACKAGE AND RETURN RESULTS
    # =========================================================================
    m_B1AA_D = coeffs["Donor"]["m_B1AA"]
    m_B1L_D = coeffs["Donor"]["m_B1L"]
    m_B1B2_F = coeffs["FRET"]["m_B1B2"]
    m_B2_F = coeffs["FRET"]["m_B2"]
    m_B1AA_F = coeffs["FRET"]["m_B1AA"]
    m_B1L_F = coeffs["FRET"]["m_B1L"]

    b2_0_vals = species["B2_0"].iloc[0, 2:] if species.get("B2_0") is not None else None

    summary = {
        "trigger_min": float(params.trigger) if params.trigger is not None else None,
        "B1L_0_nM": B1L_0,
        "method": "2x2_donor_fret_runtime",
        "init_window_min": [float(params.init_start), float(params.init_end)],
        "calibration_slopes": {
            "Donor": {
                "m_B1B2": float(m_B1B2_D),
                "m_B2": float(m_B2_D),
                "m_B1AA": float(m_B1AA_D),
                "m_B1L": float(m_B1L_D),
            },
            "FRET": {
                "m_B1B2": float(m_B1B2_F),
                "m_B2": float(m_B2_F),
                "m_B1AA": float(m_B1AA_F),
                "m_B1L": float(m_B1L_F),
            },
        },
        "per_well_B2_0_nM": {
            "mean": float(b2_0_vals.mean()) if b2_0_vals is not None else np.nan,
            "std": float(b2_0_vals.std()) if b2_0_vals is not None else np.nan,
            "min": float(b2_0_vals.min()) if b2_0_vals is not None else np.nan,
            "max": float(b2_0_vals.max()) if b2_0_vals is not None else np.nan,
        },
        "bounds": {
            "LOWER_B1B2_NM": -2.0,
            "UPPER_B1B2_NM": 120.0,
            "LOWER_B1AA_NM": -2.0,
            "UPPER_B1AA_NM": B1L_0 + 10.0,
        },
        "coeffs": {
            "alpha": {"Donor": float(alpha_D), "FRET": float(alpha_F)},
            "beta": {"Donor": float(beta_D), "FRET": float(beta_F)},
            "gamma": {"Donor": float(gamma_D), "FRET": float(gamma_F)},
            "kappa": {"Donor": float(kappa_D), "FRET": float(kappa_F)},
            "kappa_prime": {"Donor": float(kappa_prime_D), "FRET": float(kappa_prime_F)},
            "gamma_prime": {"Donor": float(gamma_prime_D), "FRET": float(gamma_prime_F)},
        },
        "channel_tags": {"donor": donor_tag, "fret": fret_tag},
        "n_successful": len(results),
        "n_failed": len(failed_wells),
        "posctrl_warning": posctrl_warning,
        "failed_wells": failed_wells,
    }

    out = {
        "B1B2 Concentration": species.get("B1B2"),
        "B1L Concentration": species.get("B1L"),
        "B2 Concentration": species.get("B2"),
        "B2 Initial (B2_0)": species.get("B2_0"),
        "Donor Ratio": extraresults["ratio_D"],
        "FRET Ratio": extraresults["ratio_F"],
        "Positive Control Ratio": extraresults["pos_ctrl_ratio"],
        "Diagnostics": diagnostics_df,
        "summary": summary,
    }

    logger.info("Completed catalytic conversion using 2x2 Donor+FRET solver (runtime calibration)")
    return out