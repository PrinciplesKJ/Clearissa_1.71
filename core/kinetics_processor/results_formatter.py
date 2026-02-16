"""
Results Formatter
-----------------
Formats fitting results as HTML for display in the kinetics processor
results browser.

Author: Krizan Jurinovic
Date: January 2026
"""

import numpy as np


# CSS for the results browser
_RESULTS_CSS = """
<style>
  body { font-size: 11pt; font-family: Arial, sans-serif; }
  .results-table {
      border-collapse: collapse;
      width: 100%;
      font-size: 11pt;
  }
  .results-table th, .results-table td {
      padding: 6px 10px;
      text-align: center;
      border: 1px solid #ddd;
  }
  .results-table th {
      background-color: #4caf50;
      color: white;
      font-weight: bold;
  }
  .global-params {
      background: #e3f2fd;
      border-left: 4px solid #2196f3;
      padding: 12px;
      margin: 10px 0;
      border-radius: 4px;
  }
  .global-params h3 { margin-top: 0; color: #1565c0; font-size: 13pt; }
  .global-params p { font-size: 11pt; margin: 4px 0; }
  .summary-box {
      background: #e8f5e9;
      border-left: 4px solid #4caf50;
      padding: 12px;
      margin: 15px 0;
      border-radius: 4px;
  }
  .summary-box h3 { margin-top: 0; color: #2e7d32; font-size: 13pt; }
  .summary-box p { font-size: 11pt; }
  h2 { font-size: 14pt; color: #2e7d32; }
  .model-info { font-size: 10pt; color: #666; margin-top: 4px; }
</style>
"""


def _row_colour(r2):
    """Return background colour based on R-squared quality."""
    if r2 >= 0.97:
        return "#d4edda"
    elif r2 >= 0.95:
        return "#fff3cd"
    else:
        return "#f8d7da"


def format_results_html(all_results, is_catalytic, r2_threshold):
    """
    Format fitting results as an HTML string for the results browser.

    Parameters
    ----------
    all_results : dict
        Mapping of trace_name -> result dict from the fitting engine.
    is_catalytic : bool
        Whether the catalytic model was used.
    r2_threshold : float
        R-squared threshold for high-quality fit summary.

    Returns
    -------
    str
        Complete HTML string for display.
    """
    rows = [_RESULTS_CSS]

    model_label = 'Catalytic Turnover' if is_catalytic else 'Bimolecular Reaction'
    rows.append(
        f'<h2>Fitting Results Summary</h2>'
        f'<p class="model-info"><strong>Model:</strong> {model_label}</p>'
    )

    if is_catalytic:
        rows.extend(_format_catalytic(all_results, r2_threshold))
    else:
        rows.extend(_format_bimolecular(all_results, r2_threshold))

    return "".join(rows)


def _format_catalytic(all_results, r2_threshold):
    """Format catalytic model results."""
    rows = []

    # Extract global parameters from first successful result
    global_k = None
    global_K = None
    all_T_values = []
    is_simple_model = False

    for result in all_results.values():
        if result.get('success', False) and result.get('model', '').startswith('Catalytic'):
            if global_k is None:
                global_k = result.get('k_fit_per_min')
                global_K = result.get('K_fit_nM')
                # Detect if simple model was used (no Michaelis constant K)
                is_simple_model = 'Simple' in result.get('model', '')
            T_val = result.get('T_nM') or result.get('T_fixed_nM')
            if T_val is not None:
                all_T_values.append(T_val)

    # Global parameters box
    if global_k is not None:
        k_str = f"{global_k:.8f}"

        if all_T_values:
            unique_T = sorted(set(all_T_values))
            if len(unique_T) == 1:
                T_str = f"{unique_T[0]:.2f} nM (all traces)"
            else:
                T_str = f"{min(unique_T):.2f} to {max(unique_T):.2f} nM (see per-trace values below)"
        else:
            T_str = "N/A"

        # Use correct units based on model type:
        # - Full model (Michaelis-Menten): k has units min^-1
        # - Simple model (no saturation): k has units nM^-1 min^-1
        if is_simple_model:
            k_units = 'nM<sup>-1</sup> min<sup>-1</sup>'
        else:
            k_units = 'min<sup>-1</sup>'

        html = (
            '<div class="global-params">'
            '<h3>Global Parameters (shared across all traces)</h3>'
            f'<p><strong>k</strong> = {k_str} {k_units} (rate constant)</p>'
        )
        if global_K is not None:
            html += f'<p><strong>K</strong> = {global_K:.2f} nM (Michaelis constant)</p>'
        html += f'<p><strong>[T]</strong> = {T_str}</p></div>'
        rows.append(html)

    # Table
    table_header = (
        '<tr><th>Trace</th><th>[T] (nM)</th>'
        '<th>[S1<sup>0</sup>] (nM)</th><th>R<sup>2</sup></th></tr>'
    )
    rows.append(f"<table class='results-table'>{table_header}")

    for trace_name, result in all_results.items():
        if 'error' in result:
            rows.append(
                f"<tr><td>{trace_name}</td>"
                f"<td colspan=3 style='color:red;'>{result['error']}</td></tr>"
            )
            continue
        if not result.get('success', False):
            rows.append(
                f"<tr><td>{trace_name}</td>"
                f"<td colspan=3 style='color:red;'>Fit failed</td></tr>"
            )
            continue

        r2 = result.get('r2', 0.0)
        S10_fit = result.get('S10_fit_nM', 0.0)
        T_trace = result.get('T_nM') or result.get('T_fixed_nM') or 0.0

        rows.append(
            f"<tr style='background:{_row_colour(r2)}'>"
            f"<td>{trace_name}</td>"
            f"<td>{T_trace:.2f}</td>"
            f"<td>{S10_fit:.2f}</td>"
            f"<td>{r2:.4f}</td>"
            f"</tr>"
        )

    rows.append("</table>")
    return rows


def _format_bimolecular(all_results, r2_threshold):
    """Format bimolecular model results."""
    rows = []
    kf_values_high_quality = []

    # Determine header from first successful result
    first_model = None
    for result in all_results.values():
        if result.get('success', False):
            first_model = result.get('model', 'Unknown')
            break

    if first_model == 'bimolecular':
        table_header = (
            '<tr><th>Trace</th>'
            '<th>k<sub>f</sub> (M<sup>-1</sup> s<sup>-1</sup>)</th>'
            '<th>[Z]<sub>0</sub> (nM)<br><span style="font-weight:normal;font-size:9px;">fixed</span></th>'
            '<th>[X]<sub>0</sub> (nM)<br><span style="font-weight:normal;font-size:9px;">fitted</span></th>'
            '<th>R<sup>2</sup></th></tr>'
        )
        colspan_error = 4
    else:
        table_header = (
            '<tr><th>Trace</th>'
            '<th>k<sub>f</sub> (M<sup>-1</sup> s<sup>-1</sup>)</th>'
            '<th>Conc. (nM)</th><th>R<sup>2</sup></th></tr>'
        )
        colspan_error = 3

    rows.append(f"<table class='results-table'>{table_header}")

    for trace_name, result in all_results.items():
        if 'error' in result:
            rows.append(
                f"<tr><td>{trace_name}</td>"
                f"<td colspan={colspan_error} style='color:red;'>{result['error']}</td></tr>"
            )
            continue
        if not result.get('success', False):
            rows.append(
                f"<tr><td>{trace_name}</td>"
                f"<td colspan={colspan_error} style='color:red;'>Fit failed</td></tr>"
            )
            continue

        r2 = result.get('r2', 0.0)
        kf = result.get('kf_fit', result.get('k_f_fit_M-1_s-1', 0.0))
        X0 = result.get('fitted_initial', result.get('X0_fit_nM', 0.0))
        Z0 = result.get('fixed_initial', result.get('Z0_nM', 0.0))

        if r2 >= r2_threshold:
            kf_values_high_quality.append(kf)

        rows.append(
            f"<tr style='background:{_row_colour(r2)}'>"
            f"<td>{trace_name}</td>"
            f"<td>{kf:.2e}</td>"
            f"<td>{Z0:.2f}</td>"
            f"<td>{X0:.2f}</td>"
            f"<td>{r2:.4f}</td>"
            f"</tr>"
        )

    rows.append("</table>")

    # Summary statistics
    if kf_values_high_quality:
        kf_mean = float(np.mean(kf_values_high_quality))
        kf_std = float(np.std(kf_values_high_quality, ddof=1)) if len(kf_values_high_quality) > 1 else 0.0
        rows.append(
            f"<div class='summary-box'>"
            f"<h3>High-Quality Fits (R<sup>2</sup> &ge; {r2_threshold:.2f})</h3>"
            f"<p><strong>Forward rate (k<sub>f</sub>):</strong> {kf_mean:.3e} &plusmn; {kf_std:.3e} "
            f"M<sup>&minus;1</sup> s<sup>&minus;1</sup> (n={len(kf_values_high_quality)})</p>"
            f"</div>"
        )

    return rows
