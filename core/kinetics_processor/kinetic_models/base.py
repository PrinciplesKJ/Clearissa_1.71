"""
Base Model Classes and Shared Utilities
=======================================

Provides shared constants and utilities for kinetic models.

Author: Krizan Jurinovic
"""

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

SECONDS_PER_MINUTE = 60.0

# ODE solver tolerances
ODE_RTOL = 1e-6
ODE_ATOL = 1e-8


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_normalised_to_nM(y_norm, full_scale_nM):
    """Convert normalised fluorescence (0 to 1) to concentration in nM."""
    return np.asarray(y_norm, dtype=float) * float(full_scale_nM)


def calculate_r_squared(y_data, y_model):
    """Calculate R-squared (coefficient of determination)."""
    y_data = np.asarray(y_data, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    ss_res = np.sum((y_data - y_model) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def first_finite_value(arr, default=0.0):
    """Return the first finite value in an array."""
    a = np.asarray(arr, dtype=float)
    finite_mask = np.isfinite(a)
    if np.any(finite_mask):
        return float(a[np.argmax(finite_mask)])
    return float(default)
