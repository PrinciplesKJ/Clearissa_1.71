"""
Unified Bimolecular Reaction Model
==================================

This module implements a unified framework for single-step irreversible
bimolecular reactions, following the methodology in the supplementary material.

Kinetic Model
-------------
All single-step reactions are modelled as an irreversible bimolecular conversion:

    X + Z  -->  Y + W
          kf

Where:
- Y = measured product (the experimental trace, inferred from fluorescence)
- X and Z = reactants
- W = released byproduct (not measured, derived from Y)

The ordinary differential equation governing all three reactions is:

    dY/dt = kf * [X](t) * [Z](t)

With mass conservation:
    [X](t) = [X]_0 - [Y](t)
    [Z](t) = [Z]_0 - [Y](t)

Substituting these conservation laws yields the reduced one-dimensional form:

    dY/dt = kf * ([X]_0 - Y) * ([Z]_0 - Y)

Fitting Approach
----------------
For each experimental trace, we fit a modelled trajectory y_sim(t) to the
experimentally inferred [Y](t), using kf and [X]_0 as fitting parameters and
using the value of [Z]_0 calculated when inferring [Y](t).

Parameter estimation is performed by nonlinear least squares using
scipy.optimize.curve_fit with non-negativity bounds. The default initial guess
for the rate constant is kf = 10^5 M^-1 s^-1, and the initial guess for [X]_0
is 10 nM.

Fit quality is quantified using the coefficient of determination R^2, computed
from the residual sum of squares between [Y](t) and y_sim(t), normalised by
the total sum of squares of [Y](t) about its mean over the fitted time points.

Assay Mapping
-------------
Standard TMSD (step 1):
    Y = AB (duplex product, measured)
    Z = Ax (substrate duplex, [Z]_0 from endpoint detection)
    X = B (invader, [X]_0 fitted)
    y(t) = [AB](t)

Internal TMSD (step 3):
    Y = ABF (product, measured)
    Z = ABT (substrate, [Z]_0 from endpoint detection)
    X = Fuel (invader, [X]_0 fitted)
    y(t) = [ABF](t)

HMSD (step 2):
    Y = S1-T-R1 (complex product, measured via FRET)
    Z = S1-T (substrate-template, [Z]_0 from manual input)
    X = R1 (reporter, [X]_0 fitted)
    y(t) = [S1-T-R1](t)

Units
-----
- Concentrations: nM
- Rate constant kf: M^-1 s^-1 (converted internally to nM^-1 s^-1)
- Time: seconds

Author: Krizan Jurinovic
"""

import logging
import numpy as np
from scipy.optimize import curve_fit

from .base import calculate_r_squared

logger = logging.getLogger(__name__)


# =============================================================================
# ANALYTICAL SOLUTION
# =============================================================================

def analytical_bimolecular(t_s, kf_M_per_s, X0, Z0):
    """
    Analytical solution for irreversible bimolecular reaction product Y(t).

    Solves the reduced 1D equation:
        dY/dt = kf * (X0 - Y) * (Z0 - Y)

    General solution:
        Y(t) = X0*Z0*(exp(k*(Z0-X0)*t) - 1) / (Z0*exp(k*(Z0-X0)*t) - X0)

    Special case (X0 approximately equal to Z0):
        Y(t) = X0 * k * t / (1 + X0 * k * t)

    Parameters
    ----------
    t_s : np.ndarray
        Time points in seconds.
    kf_M_per_s : float
        Forward rate constant in M-1 s-1.
    X0 : float
        Initial concentration X0 (nM) - one reactant.
    Z0 : float
        Initial concentration Z0 (nM) - other reactant.

    Returns
    -------
    np.ndarray
        Product concentration Y(t) in nM.
    """
    t = np.asarray(t_s, dtype=float)

    # Unit conversion: M-1 s-1 -> nM-1 s-1
    k = float(kf_M_per_s) * 1e-9
    X0 = float(X0)
    Z0 = float(Z0)

    # Handle edge cases
    if X0 <= 0 or Z0 <= 0 or k <= 0:
        return np.zeros_like(t)

    # Check for special case where X0 approximately equals Z0
    if abs(X0 - Z0) < 1e-10 * max(X0, Z0):
        # Special case: Y(t) = X0 * k * t / (1 + X0 * k * t)
        kt = k * X0 * t
        return X0 * kt / (1.0 + kt)

    # General case: Y(t) = X0*Z0*(exp(k*(Z0-X0)*t) - 1) / (Z0*exp(k*(Z0-X0)*t) - X0)
    diff = Z0 - X0
    exp_term = np.exp(k * diff * t)

    numerator = X0 * Z0 * (exp_term - 1.0)
    denominator = Z0 * exp_term - X0

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.where(np.abs(denominator) > 1e-15,
                     numerator / denominator,
                     np.minimum(X0, Z0))  # At equilibrium

    return y


# =============================================================================
# UNIFIED BIMOLECULAR MODEL
# =============================================================================

class BimolecularModel:
    """
    Unified model for single-step irreversible bimolecular reactions.

    This model implements the Methods document's symmetric framework:
        dY/dt = kf * (X0 - Y) * (Z0 - Y)

    Where:
    - Y(t) is the measured product trajectory (inferred from fluorescence)
    - One of X0 or Z0 is FIXED (from endpoint detection or manual input)
    - The other initial concentration is FITTED along with kf

    The model is symmetric: it does not care about assay-specific names,
    only about which concentration is fixed vs fitted.

    Attributes
    ----------
    default_kf_guess : float
        Default initial guess for kf (M-1 s-1). Default: 1e5.
    default_conc_guess : float
        Default initial guess for fitted concentration (nM). Default: 10.0.
    """

    # Default fitting parameters from Methods document
    default_kf_guess = 1e5  # M-1 s-1
    default_conc_guess = 10.0  # nM

    def simulate(self, t_s, kf, X0, Z0):
        """
        Simulate product trajectory Y(t) using analytical solution.

        Parameters
        ----------
        t_s : np.ndarray
            Time points in seconds.
        kf : float
            Forward rate constant (M-1 s-1).
        X0 : float
            Initial concentration X0 (nM).
        Z0 : float
            Initial concentration Z0 (nM).

        Returns
        -------
        np.ndarray
            Product concentration Y(t) in nM.
        """
        return analytical_bimolecular(t_s, kf, X0, Z0)

    def fit_single_trace(self, trace_name, time_sec, Y_data, params):
        """
        Fit bimolecular model to a single experimental trace.

        This is the core fitting function used by all bimolecular assays.
        It fits the 1D ODE: dY/dt = kf * (X0 - Y) * (Z0 - Y)

        Parameters
        ----------
        trace_name : str
            Identifier for logging and results.
        time_sec : array-like
            Time array in seconds, restricted to kinetic window [t0, t_meas_end].
        Y_data : array-like
            Measured product trajectory Y(t) in nM (already converted from fluorescence).
        params : dict
            Fitting parameters:
            - fixed_initial : float - Fixed initial concentration (nM), from endpoint or manual
            - fixed_is_Z0 : bool - If True, fixed_initial is Z0; if False, it is X0
            - fitted_initial_guess : float - Initial guess for fitted concentration (nM)
            - kf_guess : float - Initial guess for kf (M-1 s-1)
            - fixed_initial_source : str - 'endpoint' or 'manual' (for metadata)

        Returns
        -------
        dict
            Fitting results:
            - success : bool
            - kf_fit : float - Fitted kf (M-1 s-1)
            - fitted_initial : float - Fitted initial concentration (nM)
            - fixed_initial : float - Fixed initial concentration (nM)
            - fixed_is_Z0 : bool - Which concentration was fixed
            - fixed_initial_source : str - How fixed concentration was obtained
            - r2 : float - Coefficient of determination
            - Y_sim : np.ndarray - Simulated Y(t) for plotting
            - model : str - Always 'bimolecular'
        """
        try:
            time_s = np.asarray(time_sec, dtype=float)
            y_data = np.asarray(Y_data, dtype=float)

            # Extract parameters
            fixed_initial = params.get('fixed_initial', self.default_conc_guess)
            fixed_is_Z0 = params.get('fixed_is_Z0', True)
            fitted_initial_guess = params.get('fitted_initial_guess', self.default_conc_guess)
            kf_guess = params.get('kf_guess', self.default_kf_guess)
            fixed_initial_source = params.get('fixed_initial_source', 'unknown')

            logger.debug(
                "Bimolecular fit for %s: fixed_initial=%.2f nM (%s), "
                "fitted_guess=%.2f nM, kf_guess=%.2e M-1 s-1",
                trace_name, fixed_initial, 'Z0' if fixed_is_Z0 else 'X0',
                fitted_initial_guess, kf_guess
            )

            # Fit both kf and the unknown initial concentration
            if fixed_is_Z0:
                def model_fit(t, kf, X0):
                    return self.simulate(t, kf, X0, fixed_initial)
            else:
                def model_fit(t, kf, Z0):
                    return self.simulate(t, fixed_initial, Z0, fixed_initial)

            popt, _ = curve_fit(
                model_fit,
                time_s,
                y_data,
                p0=[kf_guess, float(fitted_initial_guess)],
                bounds=(0.0, np.inf),
                maxfev=5000
            )

            kf_fit = float(popt[0])
            fitted_initial = float(popt[1])

            if fixed_is_Z0:
                Y_sim = self.simulate(time_s, kf_fit, fitted_initial, fixed_initial)
            else:
                Y_sim = self.simulate(time_s, kf_fit, fixed_initial, fitted_initial)

            # Calculate R2
            r2 = calculate_r_squared(y_data, Y_sim)

            logger.info(
                "%s: kf = %.3e M-1 s-1, [X]0 = %.2f nM, [Z]0 = %.2f nM, R2 = %.4f",
                trace_name, kf_fit, fitted_initial, fixed_initial, r2
            )

            return {
                "model": "bimolecular",
                "success": True,
                "kf_fit": kf_fit,
                "fitted_initial": fitted_initial,
                "fixed_initial": fixed_initial,
                "fixed_is_Z0": fixed_is_Z0,
                "fixed_initial_source": fixed_initial_source,
                "r2": r2,
                "Y_sim": Y_sim,
                # Standard key names for compatibility
                "k_f_fit_M-1_s-1": kf_fit,
                "X0_fit_nM": fitted_initial if fixed_is_Z0 else fixed_initial,
                "Z0_nM": fixed_initial if fixed_is_Z0 else fitted_initial,
                "y_fit_nM": Y_sim,
            }

        except Exception as exc:
            logger.error("Fitting failed for trace %s: %s", trace_name, exc)
            return {
                "model": "bimolecular",
                "success": False,
                "error": str(exc)
            }
