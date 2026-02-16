"""
Catalytic Turnover Model
========================

Kinetic model for template-driven catalytic turnover experiments, following
the methodology in the supplementary material.

Kinetic Model
-------------
The template reaches a quasi-steady state in which it operates in a
Michaelis-Menten-like way, with turnover per template given by:

    d[R-S-D]/dt = (k * [R-L]) / (K + [R-L]) * T

where:
    [R-L]   = available substrate pool = [R-L]_0 - [R-S-D](t)  (mass conservation)
    [R-S-D] = fluorescent product concentration (nM)
    T       = template concentration (nM, fixed per trace)
    k       = turnover rate constant (min^-1)
    K       = constant (nM)

Note: The reaction order n is fixed at 1 (n=1) even though the product has
three components, because the reaction proceeds step-by-step with each step
linear in its reactant concentration.

Fitting Procedure
-----------------
Fits are performed only after a manually chosen start timepoint t0 at which
the trace transitions from the initial burst regime into a slower regime
driven by turnover (consistent with the model above).

Global fitting across all traces at all template concentrations:
    Shared parameters: k, K
    Per-trace fitted: [R-L]_0
    Per-trace fixed: T (template concentration), X0 (initial condition from data)

Numerical integration is performed using scipy.integrate.odeint (LSODA).
Parameter estimation uses scipy.optimize.least_squares (trust region reflective)
by minimising residuals across all traces simultaneously.

Initial guesses: k = 1 min^-1, K = 10 nM, [R-L]_0 = 10 nM per trace.

Units
-----
- Concentrations: nM
- Time: minutes
- Rate constant k: min^-1

Author: Krizan Jurinovic
"""

import logging
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares

from .base import (
    convert_normalised_to_nM,
    first_finite_value,
    SECONDS_PER_MINUTE,
    ODE_RTOL,
    ODE_ATOL,
)

logger = logging.getLogger(__name__)


def ode_catalytic(X, t, k, K, S10, T):
    """
    Catalytic turnover ODE: dX/dt = (k * S) / (K + S) * T
    where S = S10 - X (mass balance).

    Uses Michaelis-Menten kinetics (n=1).
    """
    S = max(0.0, S10 - X[0])
    denom = K + S
    if denom <= 0:
        return [0.0]
    return [(k * S / denom) * T]


def ode_catalytic_simple(X, t, k, S10, T):
    """
    Simple catalytic turnover ODE without saturation: dX/dt = k * S * T
    where S = S10 - X (mass balance).

    First-order in substrate, no Michaelis constant K.
    """
    S = max(0.0, S10 - X[0])
    return [k * S * T]


class CatalyticModel:
    """Catalytic Turnover kinetic model with Michaelis-Menten kinetics (n=1)."""

    def __init__(self, fit_time_min=None, fit_time_max=None):
        self.fit_time_min = fit_time_min
        self.fit_time_max = fit_time_max

    def simulate(self, time_min, k, K, S10, T, X0):
        """
        Simulate catalytic turnover, returning X(t) in nM.

        Parameters
        ----------
        time_min : np.ndarray
            Time points in minutes.
        k : float
            Rate constant (min^-1).
        K : float
            Constant (nM).
        S10 : float
            Initial substrate concentration (nM).
        T : float
            Template concentration (nM).
        X0 : float
            Initial product concentration (nM).

        Returns
        -------
        np.ndarray
            Product concentration X(t) in nM.
        """
        X0 = max(0.0, min(X0, S10))
        try:
            sol = odeint(ode_catalytic, [X0], time_min,
                         args=(k, K, S10, T),
                         rtol=ODE_RTOL, atol=ODE_ATOL)
            return sol[:, 0]
        except Exception as e:
            logger.error("ODE integration failed: %s", e)
            return np.full_like(time_min, np.nan)

    def simulate_simple(self, time_min, k, S10, T, X0):
        """
        Simulate simple catalytic turnover (no saturation), returning X(t) in nM.

        Parameters
        ----------
        time_min : np.ndarray
            Time points in minutes.
        k : float
            Rate constant (nM-1 min-1).
        S10 : float
            Initial substrate concentration (nM).
        T : float
            Template concentration (nM).
        X0 : float
            Initial product concentration (nM).

        Returns
        -------
        np.ndarray
            Product concentration X(t) in nM.
        """
        X0 = max(0.0, min(X0, S10))
        try:
            sol = odeint(ode_catalytic_simple, [X0], time_min,
                         args=(k, S10, T),
                         rtol=ODE_RTOL, atol=ODE_ATOL)
            return sol[:, 0]
        except Exception as e:
            logger.error("ODE integration failed (simple): %s", e)
            return np.full_like(time_min, np.nan)

    def fit_simple(self, traces_data, params):
        """
        Global fit across multiple traces using the simple (no saturation) model.

        Shared: k
        Per-trace: S10_i (fitted), T_i and X0_i (fixed from data)
        """
        if not traces_data:
            return {"model": "Catalytic (Simple)", "success": False, "error": "No traces"}

        full_scale_nM = params.get('full_scale_nM', 10.0)
        T_default = params.get('T_nM', 1.0)
        k_guess = params.get('k_guess', 0.01)
        S10_guess = params.get('S10_guess', 10.0)

        bounds = {
            'k': (1e-10, 10.0),
            'S10': (0.8 * S10_guess, 1.2 * S10_guess)
        }

        traces = []
        for td in traces_data:
            t = np.asarray(td['t_min'], dtype=float)
            y_norm = np.asarray(td['y_norm'], dtype=float)
            y_nM = convert_normalised_to_nM(y_norm, full_scale_nM)
            T = td.get('T_nM', T_default)

            mask = np.isfinite(y_nM)
            if self.fit_time_min is not None:
                mask &= (t >= self.fit_time_min)
            if self.fit_time_max is not None:
                mask &= (t <= self.fit_time_max)

            if mask.sum() < 5:
                continue

            traces.append({
                'name': td['trace_name'],
                't': t[mask],
                'y': y_nM[mask],
                't_full': t,
                'T': float(T),
                'X0': first_finite_value(y_nM, default=0.0),
            })

        if not traces:
            return {"model": "Catalytic (Simple)", "success": False, "error": "No valid traces"}

        num_traces = len(traces)
        logger.info("Fitting %d traces globally (simple model)", num_traces)

        def residuals(p):
            k = p[0]
            S10_vals = p[1:]
            res = []
            for i, tr in enumerate(traces):
                X0 = max(0.0, min(tr['X0'], S10_vals[i]))
                model = self.simulate_simple(tr['t'], k, S10_vals[i], tr['T'], X0)
                res.extend(model - tr['y'])
            return np.array(res)

        p0 = [k_guess] + [S10_guess] * num_traces
        lower = [bounds['k'][0]] + [bounds['S10'][0]] * num_traces
        upper = [bounds['k'][1]] + [bounds['S10'][1]] * num_traces

        p0 = np.clip(p0, lower, upper)

        result = least_squares(residuals, p0, bounds=(lower, upper),
                               method='trf', max_nfev=20000, ftol=1e-8, xtol=1e-8)

        k_fit = result.x[0]
        S10_fits = list(result.x[1:])

        per_trace = {}
        total_ss_res, total_ss_tot = 0.0, 0.0

        for i, tr in enumerate(traces):
            S10_i = S10_fits[i]
            X0_i = max(0.0, min(tr['X0'], S10_i))

            y_fit = self.simulate_simple(tr['t'], k_fit, S10_i, tr['T'], X0_i)
            y_fit_full = self.simulate_simple(tr['t_full'], k_fit, S10_i, tr['T'], X0_i)

            ss_res = np.sum((tr['y'] - y_fit) ** 2)
            ss_tot = np.sum((tr['y'] - np.mean(tr['y'])) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            total_ss_res += ss_res
            total_ss_tot += ss_tot

            per_trace[tr['name']] = {
                "model": "Catalytic (Simple)",
                "success": True,
                "k_fit_per_min": k_fit,
                "k_fit_per_sec": k_fit / SECONDS_PER_MINUTE,
                "S10_fit_nM": S10_i,
                "T_nM": tr['T'],
                "X0_nM": X0_i,
                "r2": r2,
                "y_fit_nM": y_fit_full,
            }

            logger.info(
                "%s: [R-L]0 = %.2f nM, T = %.2f nM, R2 = %.4f",
                tr['name'], S10_i, tr['T'], r2
            )

        global_r2 = 1.0 - total_ss_res / total_ss_tot if total_ss_tot > 0 else 0.0

        logger.info(
            "Global fit (simple): k = %.8f nM-1 min-1, R2 = %.4f",
            k_fit, global_r2
        )

        return {
            "model": "Catalytic (Simple)",
            "success": True,
            "k_fit_per_min": k_fit,
            "k_fit_per_sec": k_fit / SECONDS_PER_MINUTE,
            "global_r2": global_r2,
            "num_traces": num_traces,
            "per_trace_results": per_trace,
        }

    def fit(self, traces_data, params):
        """
        Global fit across multiple traces.

        Shared: k, K
        Per-trace: S10_i (fitted), T_i and X0_i (fixed from data)
        """
        if not traces_data:
            return {"model": "Catalytic", "success": False, "error": "No traces"}

        # Extract parameters
        full_scale_nM = params.get('full_scale_nM', 10.0)
        T_default = params.get('T_nM', 1.0)
        k_guess = params.get('k_guess', 0.01)
        K_guess = params.get('K_guess', 50.0)
        S10_guess = params.get('S10_guess', 10.0)

        # Bounds
        bounds = {
            'k': (1e-10, 1),
            'K': (0.5, 500),
            'S10': (0.8 * S10_guess, 1.2 * S10_guess)
        }

        # Prepare trace data
        traces = []
        for td in traces_data:
            t = np.asarray(td['t_min'], dtype=float)
            y_norm = np.asarray(td['y_norm'], dtype=float)
            y_nM = convert_normalised_to_nM(y_norm, full_scale_nM)
            T = td.get('T_nM', T_default)

            # Apply time window filter
            mask = np.isfinite(y_nM)
            if self.fit_time_min is not None:
                mask &= (t >= self.fit_time_min)
            if self.fit_time_max is not None:
                mask &= (t <= self.fit_time_max)

            if mask.sum() < 5:
                continue

            traces.append({
                'name': td['trace_name'],
                't': t[mask],
                'y': y_nM[mask],
                't_full': t,
                'T': float(T),
                'X0': first_finite_value(y_nM, default=0.0),
            })

        if not traces:
            return {"model": "Catalytic", "success": False, "error": "No valid traces"}

        num_traces = len(traces)
        logger.info("Fitting %d traces globally", num_traces)

        # Define residuals function
        def residuals(p):
            k, K = p[0], p[1]
            S10_vals = p[2:]

            res = []
            for i, tr in enumerate(traces):
                X0 = max(0.0, min(tr['X0'], S10_vals[i]))
                model = self.simulate(tr['t'], k, K, S10_vals[i], tr['T'], X0)
                res.extend(model - tr['y'])
            return np.array(res)

        # Build initial guess and bounds
        p0 = [k_guess, K_guess] + [S10_guess] * num_traces
        lower = [bounds['k'][0], bounds['K'][0]] + [bounds['S10'][0]] * num_traces
        upper = [bounds['k'][1], bounds['K'][1]] + [bounds['S10'][1]] * num_traces

        p0 = np.clip(p0, lower, upper)

        # Run optimisation
        result = least_squares(residuals, p0, bounds=(lower, upper),
                               method='trf', max_nfev=20000, ftol=1e-8, xtol=1e-8)

        # Extract results
        k_fit, K_fit = result.x[0], result.x[1]
        S10_fits = list(result.x[2:])

        # Compute per-trace results and R-squared
        per_trace = {}
        total_ss_res, total_ss_tot = 0.0, 0.0

        for i, tr in enumerate(traces):
            S10_i = S10_fits[i]
            X0_i = max(0.0, min(tr['X0'], S10_i))

            y_fit = self.simulate(tr['t'], k_fit, K_fit, S10_i, tr['T'], X0_i)
            y_fit_full = self.simulate(tr['t_full'], k_fit, K_fit, S10_i, tr['T'], X0_i)

            ss_res = np.sum((tr['y'] - y_fit) ** 2)
            ss_tot = np.sum((tr['y'] - np.mean(tr['y'])) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            total_ss_res += ss_res
            total_ss_tot += ss_tot

            per_trace[tr['name']] = {
                "model": "Catalytic",
                "success": True,
                "k_fit_per_min": k_fit,
                "k_fit_per_sec": k_fit / SECONDS_PER_MINUTE,
                "K_fit_nM": K_fit,
                "S10_fit_nM": S10_i,
                "T_nM": tr['T'],
                "X0_nM": X0_i,
                "r2": r2,
                "y_fit_nM": y_fit_full,
            }

            logger.info(
                "%s: [R-L]0 = %.2f nM, T = %.2f nM, R2 = %.4f",
                tr['name'], S10_i, tr['T'], r2
            )

        global_r2 = 1.0 - total_ss_res / total_ss_tot if total_ss_tot > 0 else 0.0

        logger.info(
            "Global fit: k = %.8f min-1, K = %.2f nM, R2 = %.4f",
            k_fit, K_fit, global_r2
        )

        return {
            "model": "Catalytic",
            "success": True,
            "k_fit_per_min": k_fit,
            "k_fit_per_sec": k_fit / SECONDS_PER_MINUTE,
            "K_fit_nM": K_fit,
            "global_r2": global_r2,
            "num_traces": num_traces,
            "per_trace_results": per_trace,
        }
