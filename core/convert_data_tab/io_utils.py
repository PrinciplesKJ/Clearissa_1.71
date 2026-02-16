# core/convert_data_tab/io_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any, Iterable

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add a module-level logger to avoid NameError in frozen builds
logger = logging.getLogger(__name__)

# - - - Paths - - -
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# - - - JSON helpers - - -
def load_json(path: Path | str) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path | str, data: Any, *, indent: int = 2) -> None:
    p = Path(path)
    _ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def export_conversion_package(
        result: Any,
        params: Any,
        folder_path: Path | str,
) -> Path:
    """
    Export conversion results to a timestamped subfolder (CSVs, statistics
    JSON, parameters JSON, and a README.txt).

    Args:
        result: Conversion results (dict of DataFrames or single DataFrame)
        params: ConvertParams object with all conversion parameters
        folder_path: Base folder path for export

    Returns:
        Path to the created export subfolder
    """
    base_dir = Path(folder_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped subfolder
    export_dir = base_dir / f"conversion_export_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    exported_items = []
    missing_data = []  # Track missing/unavailable data

    # Helper to convert numpy types to native Python for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj

    def safe_get(dictionary, key, default=None, data_type=None):
        """Safely get value from dict with type conversion and logging."""
        try:
            value = dictionary.get(key, default)
            if value is None:
                return default
            if data_type is not None:
                return data_type(value)
            return value
        except (TypeError, ValueError) as e:
            missing_data.append(f"{key}: conversion error ({e})")
            return default

    # 1. Export DataFrames
    if isinstance(result, dict):
        dataframes = {
            key: value
            for key, value in result.items()
            if isinstance(value, pd.DataFrame)
        }

        for name, df in dataframes.items():
            try:
                csv_path = export_dir / f"{name}.csv"
                # Ensure pandas receives a string path in all environments
                df.to_csv(str(csv_path), index=False)
                exported_items.append(f"- {name}.csv ({len(df)} rows, {len(df.columns)} columns)")
            except Exception as e:
                missing_data.append(f"Failed to export {name}.csv: {e}")

        # Export summary if present
        if 'summary' in result and isinstance(result['summary'], dict):
            try:
                summary = convert_for_json(result['summary'])
                summary_path = export_dir / "summary.json"
                with summary_path.open('w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                exported_items.append("- summary.json")
            except Exception as e:
                missing_data.append(f"Failed to export summary.json: {e}")

    elif isinstance(result, pd.DataFrame):
        try:
            csv_path = export_dir / "results.csv"
            result.to_csv(str(csv_path), index=False)
            exported_items.append(f"- results.csv ({len(result)} rows, {len(result.columns)} columns)")
        except Exception as e:
            missing_data.append(f"Failed to export results.csv: {e}")

    # 2. Export metadata
    try:
        metadata = {
            "export_timestamp": timestamp,
            "export_datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "conversion_mode": getattr(params, 'mode', 'unknown'),
            "parameters": {
                "init_start_min": getattr(params, 'init_start', None),
                "init_end_min": getattr(params, 'init_end', None),
                "trigger_min": getattr(params, 'trigger', None),
                "injection_primary_min": getattr(params, 'injection_primary', None),
                "injection_secondary_min": getattr(params, 'injection_secondary', None),
            },
            "concentrations": {
                "positive_controls_nM": getattr(params, 'pos_ctrl_concentrations', []),
                "donor_controls_nM": getattr(params, 'donor_ctrl_concentrations', []),
                "acceptor_controls_nM": getattr(params, 'acceptor_ctrl_concentrations', []),
            },
            "species_selections": {
                key: value.get("species") if isinstance(value, dict) else str(value)
                for key, value in getattr(params, 'species', {}).items()
            },
            "species_calibrations": {
                key: {
                    "species": value.get("species"),
                    "slope": value.get("slope"),
                    "intercept": value.get("intercept")
                } if isinstance(value, dict) else {"species": str(value)}
                for key, value in getattr(params, 'species', {}).items()
            },
            "channels": getattr(params, 'extra', {}).get('selected_channels', {}),
        }

        metadata_path = export_dir / "conversion_metadata.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        exported_items.append("- conversion_metadata.json")
    except Exception as e:
        missing_data.append(f"Failed to export metadata: {e}")

    # 3. Create human-readable README
    try:
        readme_path = export_dir / "README.txt"
        with readme_path.open('w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("CLEARISSA CONVERSION EXPORT PACKAGE\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Conversion Mode: {getattr(params, 'mode', 'UNKNOWN').upper()}\n\n")

            # PARAMETERS SECTION
            f.write("PARAMETERS:\n")
            f.write("-" * 70 + "\n")

            init_start = getattr(params, 'init_start', None)
            init_end = getattr(params, 'init_end', None)
            trigger = getattr(params, 'trigger', None)
            inj_primary = getattr(params, 'injection_primary', None)
            inj_secondary = getattr(params, 'injection_secondary', None)

            if init_start is not None and init_end is not None:
                f.write(f"  Initialisation Window: {init_start:.2f} - {init_end:.2f} min\n")
            else:
                f.write("  Initialisation Window: Not specified\n")

            if trigger is not None:
                f.write(f"  Trigger Time: {trigger:.2f} min\n")
            else:
                f.write("  Trigger Time: Not specified\n")

            if inj_primary is not None:
                f.write(f"  Primary Injection: {inj_primary:.2f} min\n")
            if inj_secondary is not None:
                f.write(f"  Secondary Injection: {inj_secondary:.2f} min\n")

            # CONTROL CONCENTRATIONS SECTION
            f.write("\nCONTROL CONCENTRATIONS:\n")
            f.write("-" * 70 + "\n")

            pos_ctrl = getattr(params, 'pos_ctrl_concentrations', None)
            donor_ctrl = getattr(params, 'donor_ctrl_concentrations', None)
            acceptor_ctrl = getattr(params, 'acceptor_ctrl_concentrations', None)

            if pos_ctrl:
                for i, conc in enumerate(pos_ctrl, 1):
                    f.write(f"  Positive Control {i}: {conc:.2f} nM\n")
            else:
                f.write("  Positive Controls: Not specified\n")

            if donor_ctrl:
                for i, conc in enumerate(donor_ctrl, 1):
                    f.write(f"  Donor Control {i}: {conc:.2f} nM\n")
            else:
                f.write("  Donor Controls: Not specified\n")

            if acceptor_ctrl:
                for i, conc in enumerate(acceptor_ctrl, 1):
                    f.write(f"  Acceptor Control {i}: {conc:.2f} nM\n")
            else:
                f.write("  Acceptor Controls: Not specified\n")

            # MASS ACTION PARAMETERS (only if applicable)
            b1l_0 = getattr(params, 'extra', {}).get('b1l_initial_conc', None)
            if b1l_0 is not None:
                f.write("\nMASS ACTION PARAMETERS:\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Initial Quenched Acceptor (B1L_0): {b1l_0:.2f} nM\n")

            # SPECIES SELECTIONS & CALIBRATIONS
            f.write("\nSPECIES SELECTIONS & CALIBRATIONS:\n")
            f.write("-" * 70 + "\n")

            species_dict = getattr(params, 'species', {})
            if species_dict:
                for label, cal_data in species_dict.items():
                    if isinstance(cal_data, dict):
                        species_name = cal_data.get("species", "Unknown")
                        slope = cal_data.get("slope", "N/A")
                        intercept = cal_data.get("intercept", "N/A")
                        f.write(f"  {label}:\n")
                        f.write(f"    Species: {species_name}\n")
                        f.write(f"    Slope: {slope}\n")
                        f.write(f"    Intercept: {intercept}\n")
            else:
                f.write("  No calibration data available\n")

            # RESULTS SUMMARY (if available)
            if isinstance(result, dict) and 'summary' in result:
                f.write("\nRESULTS SUMMARY:\n")
                f.write("-" * 70 + "\n")
                summary = result['summary']

                # Concentration statistics
                conc_min = safe_get(summary, 'concentration_min_nM', data_type=float)
                conc_max = safe_get(summary, 'concentration_max_nM', data_type=float)
                conc_mean = safe_get(summary, 'concentration_mean_nM', data_type=float)

                if conc_min is not None and conc_max is not None:
                    f.write(f"  Concentration Range: {conc_min:.2f} - {conc_max:.2f} nM\n")
                if conc_mean is not None:
                    f.write(f"  Mean Concentration: {conc_mean:.2f} nM\n")

                # TMSD-specific fields
                slope_ratio = safe_get(summary, 'slope_ratio_Ax_over_AB', data_type=float)
                if slope_ratio is not None:
                    f.write(f"  Slope Ratio (Ax/AB): {slope_ratio:.4f}\n")

                product_species = safe_get(summary, 'product_species')
                if product_species:
                    f.write(f"  Product Species: {product_species}\n")

                substrate_species = safe_get(summary, 'substrate_species')
                if substrate_species:
                    f.write(f"  Substrate Species: {substrate_species}\n")

                primary_nuke = safe_get(summary, 'primary_nuking_time_min', data_type=float)
                if primary_nuke is not None:
                    f.write(f"  Primary Nuking Time: {primary_nuke:.2f} min\n")

                secondary_nuke = safe_get(summary, 'secondary_nuking_time_min', data_type=float)
                if secondary_nuke is not None:
                    f.write(f"  Secondary Nuking Time: {secondary_nuke:.2f} min\n")

                # Initial donor concentration (B2_0) - FRET-specific
                if 'per_well_B2_0_nM' in summary:
                    b2_stats = summary['per_well_B2_0_nM']
                    f.write(f"\nINITIAL DONOR STATISTICS (B2_0):\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"  Mean: {safe_get(b2_stats, 'mean', 0.0, float):.2f} nM\n")
                    f.write(f"  Std Dev: {safe_get(b2_stats, 'std', 0.0, float):.2f} nM\n")
                    f.write(f"  Range: [{safe_get(b2_stats, 'min', 0.0, float):.2f}, "
                            f"{safe_get(b2_stats, 'max', 0.0, float):.2f}] nM\n")

                # Coefficients
                coeffs = safe_get(summary, 'coeffs', {})
                if coeffs and isinstance(coeffs, dict):
                    f.write(f"\nCOMPUTED COEFFICIENTS:\n")
                    f.write("-" * 70 + "\n")
                    for coeff_name in ['alpha', 'beta', 'gamma', 'kappa']:
                        if coeff_name in coeffs and isinstance(coeffs[coeff_name], dict):
                            coeff_dict = coeffs[coeff_name]
                            parts = []
                            for channel in ['Donor', 'Acceptor', 'FRET']:
                                val = safe_get(coeff_dict, channel, data_type=float)
                                if val is not None:
                                    parts.append(f"{channel}={val:.4f}")
                            if parts:
                                f.write(f"  {coeff_name}: {', '.join(parts)}\n")

                # Failed wells
                failed_wells = safe_get(summary, 'failed_wells', [])
                if failed_wells and len(failed_wells) > 0:
                    f.write(f"\nFAILED WELLS:\n")
                    f.write("-" * 70 + "\n")
                    for well in failed_wells:
                        f.write(f"  - {well}\n")

            # EXPORTED FILES
            f.write("\nEXPORTED FILES:\n")
            f.write("=" * 70 + "\n")
            for item in exported_items:
                f.write(f"{item}\n")

            # MISSING/UNAVAILABLE DATA (if any)
            if missing_data:
                f.write("\nMISSING OR UNAVAILABLE DATA:\n")
                f.write("=" * 70 + "\n")
                for item in missing_data:
                    f.write(f"  ⚠ {item}\n")
                f.write("\nNote: Some data fields were not available or could not be exported.\n")
                f.write("This may be normal depending on the conversion mode.\n")

            # USAGE NOTES
            f.write("\n" + "=" * 70 + "\n")
            f.write("USAGE NOTES:\n")
            f.write("-" * 70 + "\n")
            f.write("- CSV files can be opened in Excel, MATLAB, Python, etc.\n")
            f.write("- JSON files contain machine-readable metadata and summaries\n")
            f.write("- All concentration values are in nanomolar (nM)\n")
            f.write("- All time values are in minutes (min)\n")
            f.write("\nFor questions or issues, contact: k.jurinovic22@ic.ac.uk\n")
            f.write("=" * 70 + "\n")

        exported_items.append("- README.txt")

    except Exception as e:
        missing_data.append(f"Failed to create README.txt: {e}")

    # Log any missing data
    if missing_data:
        # Use module-level logger
        logger.warning(f"Export completed with {len(missing_data)} warnings:")
        for item in missing_data:
            logger.warning(f"  - {item}")

    return export_dir


# - - - CSV/TSV helpers - - -
def load_table(path: Path | str, *, dtype: dict | None = None) -> pd.DataFrame:
    """
    Load a CSV or TSV based on file extension.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".tsv":
        sep = "\t"
    elif ext == ".csv":
        sep = ","
    else:
        raise ValueError(f"Unsupported table extension: {ext}")
    return pd.read_csv(p, sep=sep, dtype=dtype)


def save_dataframe(df: pd.DataFrame, path: Path | str) -> None:
    """
    Save a single DataFrame to .csv, .tsv or .xlsx.
    """
    p = Path(path)
    _ensure_parent(p)
    ext = p.suffix.lower()

    if ext == ".csv":
        df.to_csv(p, index=False)
    elif ext == ".tsv":
        df.to_csv(p, index=False, sep="\t")
    elif ext == ".xlsx":
        # Engine selected by pandas; openpyxl/xlsxwriter must be installed
        with pd.ExcelWriter(p) as xw:
            df.to_excel(xw, sheet_name="Sheet1", index=False)
    else:
        raise ValueError(f"Unsupported extension for single DataFrame: {ext}")


# - - - Multi-DataFrame saving - - -
def _sanitise_sheet_name(name: str) -> str:
    # Excel sheet name rules: max 31 chars, cannot contain []:*?/\
    invalid = set('[]:*?/\\')
    cleaned = "".join(c for c in str(name) if c not in invalid).strip()
    return (cleaned or "Sheet")[:31]


def save_frames(
    frames: Mapping[str, pd.DataFrame],
    path: Path | str,
) -> None:
    """
    Save multiple DataFrames. If the target is:
    - .xlsx: each key becomes a sheet
    - .csv/.tsv: write one file per key, using suffix _<key> before the extension
    """
    p = Path(path)
    _ensure_parent(p)
    ext = p.suffix.lower()

    if ext == ".xlsx":
        with pd.ExcelWriter(p) as xw:
            for key, df in frames.items():
                sheet = _sanitise_sheet_name(str(key))
                df.to_excel(xw, sheet_name=sheet, index=False)
        return

    if ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        stem = p.stem
        parent = p.parent
        for key, df in frames.items():
            out = parent / f"{stem}_{key}{ext}"
            df.to_csv(out, index=False, sep=sep)
        return

    raise ValueError(f"Unsupported extension for multiple DataFrames: {ext}")


# - - - Public convenience - - -
def save_result(path: Path | str, result: Any) -> None:
    """
    Save either a single DataFrame or a mapping of name -> DataFrame.
    File type is inferred from the extension of 'path'.
    """
    p = Path(path)
    if isinstance(result, pd.DataFrame):
        save_dataframe(result, p)
    elif isinstance(result, Mapping) and all(isinstance(v, pd.DataFrame) for v in result.values()):
        save_frames(result, p)
    else:
        raise TypeError(
            "save_result expects a pandas DataFrame or a mapping of str -> DataFrame"
        )
