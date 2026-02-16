"""
Kinetic Models Package
======================

This package contains individual model implementations for kinetic fitting.

Each model file contains:
- Simulation function (analytical or ODE-based)
- Single-trace fitting function
- Model-specific constants and utilities

Available Models
----------------
- BimolecularModel: Unified bimolecular reaction model (covers TMSD/internal TMSD/HMSD)
- CatalyticModel: Template-driven catalytic turnover (Hill-type kinetics)

The BimolecularModel uses the unified X, Y, Z species notation:
- X = reactant with fitted initial concentration
- Z = reactant with known initial concentration
- Y = measured product
- W = released byproduct (not measured)

Usage
-----
    from core.kinetics_processor.kinetic_models import BimolecularModel, CatalyticModel

    # Create model instance
    model = BimolecularModel()

    # Fit data
    result = model.fit_single_trace(trace_name, time_sec, y_nM, params)

Author: Krizan Jurinovic
"""

# Unified bimolecular model
from .bimolecular import BimolecularModel, analytical_bimolecular

# Catalytic model
from .catalytic import CatalyticModel

__all__ = ['BimolecularModel', 'CatalyticModel', 'analytical_bimolecular']
