"""
Unified Fitting Engine for Kinetics Processor
==============================================

This module provides a single, clear fitting engine that handles all model types.

The fitting engine:
- Routes to the appropriate model (Bimolecular or Catalytic)
- Uses sequential fitting (simple and reliable)
- Returns results directly (synchronous operation)

Model Types:
- bimolecular: Single-step bimolecular reactions (TMSD, internal TMSD, HMSD)
- catalytic: Template-driven catalytic turnover

Author: Krizan Jurinovic
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class FittingEngine:
    """
    Fitting engine for kinetic models.

    This engine executes fitting operations synchronously and supports
    bimolecular and catalytic models.

    Usage
    -----
        engine = FittingEngine(data_df, time_col, params, model_type, trace_settings)
        results = engine.run()
    """

    def __init__(self, data_df, time_col, params, model_type, trace_settings):
        """
        Initialise fitting engine.

        Parameters
        ----------
        data_df : pd.DataFrame
            Data to fit (already windowed to selected time range).
        time_col : str
            Name of time column.
        params : dict
            Model parameters (initial guesses, concentrations, etc.)
        model_type : str
            Model type: 'bimolecular' or 'catalytic'
        trace_settings : dict
            Trace visibility and per-trace settings.
            Keys: trace column names
            Values: dicts with 'show_trace', 'Z0_nM', 'T_nM', etc.
        """
        # Copy data for safety
        self.data_df = data_df.copy()
        self.time_col = time_col
        self.params = params.copy()
        self.model_type = model_type.lower()
        self.trace_settings = trace_settings.copy()

    def run(self):
        """
        Execute fitting.

        Routes to the appropriate model and fitting strategy:
        - catalytic: Global fitting (shared k, K across traces)
        - bimolecular: Individual trace fitting

        Returns
        -------
        dict
            Results keyed by trace name.
        """
        logger.info("Starting %s model fitting", self.model_type)

        # Get visible traces
        trace_cols = [
            c for c in self.data_df.columns
            if c != self.time_col and not c.endswith('_fitted')
        ]
        visible_traces = [
            c for c in trace_cols
            if self.trace_settings.get(c, {}).get('show_trace', True)
        ]

        # Filter out traces excluded from fitting
        excluded_traces = [
            c for c in visible_traces
            if self.trace_settings.get(c, {}).get('exclude_from_fit', False)
        ]
        traces_to_fit = [c for c in visible_traces if c not in excluded_traces]

        if excluded_traces:
            logger.info("Excluding %d traces from fitting: %s",
                       len(excluded_traces), ', '.join(excluded_traces[:5]) +
                       ('...' if len(excluded_traces) > 5 else ''))

        if not traces_to_fit:
            logger.warning("No traces to fit (all visible traces are excluded)")
            return {}

        # Prepare time arrays
        t_min_original = self.data_df[self.time_col].to_numpy(dtype=float)
        time_window_start = self.params.get('time_window_start', 0.0)
        t_min = t_min_original - time_window_start
        t_sec = t_min * 60.0

        logger.info(
            "Fitting %d traces over %.1f to %.1f min (%d data points)",
            len(traces_to_fit),
            t_min_original[0] if len(t_min_original) > 0 else 0,
            t_min_original[-1] if len(t_min_original) > 0 else 0,
            len(t_min_original)
        )

        # Route to appropriate fitting method
        if self.model_type == 'catalytic':
            results = self._fit_catalytic_global(traces_to_fit, t_min)
        elif self.model_type == 'bimolecular':
            results = self._fit_bimolecular(traces_to_fit, t_sec)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        successful = sum(1 for r in results.values() if r.get('success', False))
        logger.info("Fitting complete: %d of %d traces fitted successfully", successful, len(results))
        return results

    def _fit_catalytic_global(self, visible_traces, t_min):
        """
        Fit catalytic model globally (shared k, K across all traces).

        Parameters
        ----------
        visible_traces : list
            List of trace column names to fit.
        t_min : np.ndarray
            Time in minutes (normalised).

        Returns
        -------
        dict
            Results keyed by trace name.
        """
        from core.kinetics_processor.kinetic_models import CatalyticModel

        # Get parameters
        global_T_nM = self.params.get('catalytic_template_T', 1.0)
        full_scale_nM = self.params.get('fluorescence_full_scale_nM', 10.0)

        logger.info(
            "Catalytic params: default T=%.3f nM, full_scale=%.3f nM, "
            "S10_guess=%.3f, k_guess=%.6f, K_guess=%.3f",
            global_T_nM, full_scale_nM,
            self.params.get('catalytic_S10_guess', 10.0),
            self.params.get('catalytic_k_guess', 1.0),
            self.params.get('catalytic_K_guess', 10.0)
        )

        # Collect trace data for global fitting
        traces_data = []
        for col in visible_traces:
            y_raw = self.data_df[col].astype(float).to_numpy()

            # Get per-trace T_nM
            trace_T_nM = self.trace_settings.get(col, {}).get('T_nM', None)
            if trace_T_nM is None:
                trace_T_nM = global_T_nM
                logger.debug("Trace %s: using default T=%.3f nM", col, trace_T_nM)
            else:
                logger.info("Trace %s: using per-trace T=%.3f nM", col, trace_T_nM)

            traces_data.append({
                'trace_name': col,
                't_min': t_min.copy(),
                'y_norm': y_raw.copy(),
                'T_nM': trace_T_nM
            })

        # Log T values summary
        T_values = [td['T_nM'] for td in traces_data]
        unique_T = set(T_values)
        if len(unique_T) == 1:
            logger.info("All %d traces using T=%.3f nM", len(traces_data), list(unique_T)[0])
        else:
            logger.info(
                "Per-trace T values: min=%.3f, max=%.3f, unique=%d values",
                min(T_values), max(T_values), len(unique_T)
            )

        # Create model and fit
        model = CatalyticModel()
        catalytic_sub_model = self.params.get('catalytic_sub_model', 'full')

        fit_params = {
            'full_scale_nM': full_scale_nM,
            'T_nM': global_T_nM,
            'k_guess': self.params.get('catalytic_k_guess', 1.0),
            'K_guess': self.params.get('catalytic_K_guess', 10.0),
            'S10_guess': self.params.get('catalytic_S10_guess', 10.0),
        }

        if catalytic_sub_model == 'simple':
            logger.info("Using simple catalytic model (no saturation)")
            global_result = model.fit_simple(traces_data, fit_params)
        else:
            global_result = model.fit(traces_data, fit_params)

        if global_result.get('success', False):
            results = global_result.get('per_trace_results', {})

            if catalytic_sub_model == 'simple':
                logger.info(
                    "Global catalytic fit complete (simple): k=%.8f nM-1 min-1, global R2=%.4f",
                    global_result.get('k_fit_per_min', 0),
                    global_result.get('global_r2', 0)
                )
            else:
                logger.info(
                    "Global catalytic fit complete: k=%.8f per min, K=%.2f nM, global R2=%.4f",
                    global_result.get('k_fit_per_min', 0),
                    global_result.get('K_fit_nM', 0),
                    global_result.get('global_r2', 0)
                )

            return results
        else:
            error_msg = global_result.get('error', 'Unknown error')
            logger.error("Global catalytic fit failed: %s", error_msg)
            return {col: {'error': error_msg, 'model': 'Catalytic', 'success': False}
                    for col in visible_traces}

    def _fit_bimolecular(self, visible_traces, t_sec):
        """
        Fit bimolecular model to individual traces.

        Implements the Methods document's unified framework:
            dY/dt = kf * (X0 - Y) * (Z0 - Y)

        Where one of X0 or Z0 is fixed (from endpoint detection or manual input)
        and the other is fitted along with kf.

        The fitting procedure:
        1. Extract valid (non-NaN) data points for curve fitting
        2. Fit model parameters using only valid points
        3. Generate the fitted curve for the FULL time array

        This ensures the output Y_sim aligns with the input dataframe,
        allowing direct storage without index manipulation.

        Parameters
        ----------
        visible_traces : list
            List of trace column names to fit.
        t_sec : np.ndarray
            Full time array in seconds (normalised to start at 0).

        Returns
        -------
        dict
            Results keyed by trace name. Each result contains:
            - kf_fit: fitted rate constant (M-1 s-1)
            - fitted_initial: fitted initial concentration (nM)
            - fixed_initial: fixed initial concentration (nM)
            - r2: coefficient of determination
            - y_fit_nM: simulated product trajectory (same length as input)
        """
        from core.kinetics_processor.kinetic_models import BimolecularModel

        model = BimolecularModel()
        results = {}
        total_traces = len(visible_traces)

        # Determine input mode: 'endpoint' or 'manual'
        input_mode = self.params.get('input_mode', 'endpoint')

        for i, col in enumerate(visible_traces):
            try:
                # Get trace data
                y_raw = self.data_df[col].astype(float).to_numpy()
                valid_mask = np.isfinite(y_raw)
                n_valid = valid_mask.sum()

                if n_valid < 3:
                    results[col] = {
                        'error': 'Insufficient valid data (%d points)' % n_valid,
                        'model': 'bimolecular',
                        'success': False
                    }
                    continue

                # Extract valid points for fitting
                t_valid = t_sec[valid_mask]
                y_valid = y_raw[valid_mask]

                # Get the fixed initial concentration
                trace_Z0 = self._get_trace_Z0(col, input_mode)
                fixed_initial_source = 'endpoint' if input_mode == 'endpoint' else 'manual'

                # Fit using valid points only
                fit_params = {
                    'fixed_initial': trace_Z0,
                    'fixed_is_Z0': True,
                    'fitted_initial_guess': self.params.get('X0_guess', 10.0),
                    'kf_guess': self.params.get('kf_guess', 1e5),
                    'fixed_initial_source': fixed_initial_source,
                }

                result = model.fit_single_trace(col, t_valid, y_valid, fit_params)

                # Generate fitted curve for FULL time array (for plotting alignment)
                if result.get('success', False):
                    kf_fit = result['kf_fit']
                    fitted_init = result['fitted_initial']
                    fixed_init = result['fixed_initial']

                    # Simulate over full time range
                    y_fit_full = model.simulate(t_sec, kf_fit, fitted_init, fixed_init)
                    result['y_fit_nM'] = y_fit_full
                    result['Y_sim'] = y_fit_full

                results[col] = result
                logger.debug("Fitted %s (%d/%d)", col, i + 1, total_traces)

            except Exception as e:
                logger.error("Failed to fit %s: %s", col, e, exc_info=True)
                results[col] = {'error': str(e), 'model': 'bimolecular', 'success': False}

        return results

    def _get_trace_Z0(self, col, input_mode):
        """
        Get the fixed initial concentration [Z]_0 for a trace.

        Parameters
        ----------
        col : str
            Trace column name.
        input_mode : str
            Either 'endpoint' or 'manual'.

        Returns
        -------
        float
            The [Z]_0 concentration in nM.
        """
        trace_Z0 = self.trace_settings.get(col, {}).get('Z0_nM')

        if trace_Z0 is not None:
            trace_Z0 = float(trace_Z0)
            logger.debug("Trace %s: using %s Z0=%.2f nM", col, input_mode, trace_Z0)
        else:
            # Fallback to default values
            if input_mode == 'endpoint':
                trace_Z0 = self.params.get('Z0_manual', 10.0)
            else:
                trace_Z0 = self.params.get('Z0_default', 10.0)
            logger.debug("Trace %s: using fallback Z0=%.2f nM", col, trace_Z0)

        return trace_Z0

