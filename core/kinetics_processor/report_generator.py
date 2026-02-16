"""
Report Generation Module for Kinetics Processor
------------------------------------------------
Handles HTML report generation for fitting results and quality assessments.
Also handles graph export using plotnine/ggplot.

Author: Krizan Jurinovic
Date: November 2025
"""

import logging
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QFileDialog

# Import centralised time processing utilities
from .data_processor import normalise_time_column, filter_time_window

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates HTML reports and exportable graphs for kinetics analysis.
    """

    def __init__(self, parent_widget=None):
        """
        Initialise the report generator.

        Parameters
        ----------
        parent_widget : QWidget, optional
            Parent widget for dialogues
        """
        self.parent = parent_widget

    def generate_quality_report_html(self, quality_report, has_fits=True,
                                     r2_threshold=0.10):
        """
        Generate HTML report for replicate fitting quality.

        Parameters
        ----------
        quality_report : list
            List of dictionaries containing quality metrics for each group
        has_fits : bool
            Whether fitted data is available
        r2_threshold : float
            R-squared threshold for quality classification (from user settings).
            Fits at or above this value are rated 'good'; fits below are 'poor'.

        Returns
        -------
        str
            HTML-formatted quality report
        """
        html = [
            """
            <style>
                body { font-family: Arial, sans-serif; font-size: 10pt; }
                table { border-collapse: collapse; width: 100%; margin-top: 10px; }
                th, td { padding: 6px; text-align: centre; border: 1px solid #ddd; }
                th { background-color: #4CAF50; color: white; font-weight: bold; }
                .excellent { background-color: #d4edda; }
                .good { background-color: #fff3cd; }
                .poor { background-color: #f8d7da; }
                .neutral { background-color: #e7f3ff; }
                h3 { color: #2e7d32; margin: 5px 0; }
            </style>
            """
        ]


        if has_fits:
            html.append("<h3>Replicate Fitting Quality Report (Selected Traces Only)</h3>")
            html.append("""
            <table>
                <tr>
                    <th>Group</th>
                    <th>Selected Replicates</th>
                    <th>Successfully Fitted</th>
                    <th>Mean R²</th>
                    <th>Mean k<sub>f</sub> (M<sup>−1</sup> s<sup>−1</sup>)</th>
                    <th>Std Dev k<sub>f</sub></th>
                    <th>SEM k<sub>f</sub></th>
                    <th>Individual k<sub>f</sub> Values</th>
                </tr>
            """)
        else:
            html.append("<h3>Replicate Average Report (Selected Traces Only - Data Only)</h3>")
            html.append("""
            <table>
                <tr>
                    <th>Group</th>
                    <th>Selected Replicates</th>
                </tr>
            """)

        for item in quality_report:
            if has_fits:
                # Determine row colour based on mean R-squared
                mean_r2 = item['mean_r2']
                if mean_r2 is not None:
                    if mean_r2 >= r2_threshold:
                        row_class = 'good'
                    else:
                        row_class = 'poor'
                    r2_display = f"{mean_r2:.4f}"
                else:
                    row_class = 'neutral'
                    r2_display = "N/A"

                # Format mean kf, std, and SEM
                mean_kf = item.get('mean_kf')
                std_kf = item.get('std_kf', 0.0)
                sem_kf = item.get('sem_kf', 0.0)
                kf_values = item.get('kf_values', [])

                kf_mean_str = f"{mean_kf:.2e}" if mean_kf is not None else "N/A"
                # Show std/sem as 0.00e+00 for single values, N/A if no mean_kf
                kf_std_str = f"{std_kf:.2e}" if mean_kf is not None else "N/A"
                kf_sem_str = f"{sem_kf:.2e}" if mean_kf is not None else "N/A"
                kf_individual_str = ", ".join([f"{kf:.2e}" for kf in kf_values]) if kf_values else "N/A"

                html.append(
                    f"<tr class='{row_class}'>"
                    f"<td><b>{item['group']}</b></td>"
                    f"<td>{item['n_replicates']}</td>"
                    f"<td>{item['n_fitted']}</td>"
                    f"<td>{r2_display}</td>"
                    f"<td>{kf_mean_str}</td>"
                    f"<td>{kf_std_str}</td>"
                    f"<td>{kf_sem_str}</td>"
                    f"<td>{kf_individual_str}</td>"
                    f"</tr>"
                )
            else:
                # No fits - just show group and replicate count
                html.append(
                    f"<tr class='neutral'>"
                    f"<td><b>{item['group']}</b></td>"
                    f"<td>{item['n_replicates']}</td>"
                    f"</tr>"
                )

        html.append("</table>")

        if has_fits:
            html.append("<p style='font-size: 9pt; color: #666; margin-top: 10px;'>")
            html.append(f"Colour coding: <span style='background: #d4edda; padding: 2px 6px;'>Good (R² &ge; {r2_threshold:.2f})</span> ")
            html.append(f"<span style='background: #f8d7da; padding: 2px 6px;'>Poor (R² &lt; {r2_threshold:.2f})</span>")
            html.append("</p>")
        else:
            html.append("<p style='font-size: 9pt; color: #666; margin-top: 10px;'>")
            html.append("Showing averaged data only. Run 'Simulate & Fit' to generate fitted curves.")
            html.append("</p>")

        return "".join(html)

    def generate_fitting_summary_header(self):
        """
        Generate HTML header for fitting results summary with proper special characters.

        Returns
        -------
        str
            HTML header for results table
        """
        return """
            <style>
              .results-table th, .results-table td { padding:4px 8px; text-align:centre; }
              .summary-box {
                  background: #e8f5e9;
                  border-left: 4px solid #4caf50;
                  padding: 12px;
                  margin: 15px 0;
                  border-radius: 4px;
              }
              .summary-box h3 { margin-top: 0; color: #2e7d32; }
            </style>
            <h2>Fitting Results Summary</h2>
            <table class="results-table">
              <tr><th>Trace</th><th>k<sub>f</sub> (M<sup>−1</sup> s<sup>−1</sup>)</th><th>k<sub>r</sub> (M<sup>−1</sup> s<sup>−1</sup>)</th>
                  <th>[I]<sub>0</sub> (nM)</th><th>R²</th></tr>
            """

    def generate_average_rate_constants_box(self, kf_values, kr_values):
        """
        Generate HTML box showing average rate constants with proper special characters.

        Parameters
        ----------
        kf_values : list
            List of forward rate constant values
        kr_values : list
            List of reverse rate constant values

        Returns
        -------
        str
            HTML-formatted summary box
        """
        import numpy as np

        has_kf = len(kf_values) > 0
        has_kr = len(kr_values) > 0

        if has_kf or has_kr:
            kf_line = "No k<sub>f</sub> values"
            kr_line = "No k<sub>r</sub> values"

            if has_kf:
                mean_kf = float(np.mean(kf_values))
                std_kf = float(np.std(kf_values, ddof=1)) if len(kf_values) > 1 else 0.0
                sem_kf = std_kf / np.sqrt(len(kf_values)) if len(kf_values) > 1 else 0.0
                kf_line = f"k<sub>f</sub> (forward): {mean_kf:.4e} ± {sem_kf:.4e} M<sup>−1</sup> s<sup>−1</sup> (n={len(kf_values)}, σ={std_kf:.4e})"

            if has_kr:
                mean_kr = float(np.mean(kr_values))
                std_kr = float(np.std(kr_values, ddof=1)) if len(kr_values) > 1 else 0.0
                sem_kr = std_kr / np.sqrt(len(kr_values)) if len(kr_values) > 1 else 0.0
                kr_line = f"k<sub>r</sub> (reverse): {mean_kr:.4e} ± {sem_kr:.4e} M<sup>−1</sup> s<sup>−1</sup> (n={len(kr_values)}, σ={std_kr:.4e})"

            return f"""
                <div class="summary-box">
                    <h3>Average Rate Constants (R² ≥ 0.90 only)</h3>
                    <p><strong>{kf_line}</strong></p>
                    <p><strong>{kr_line}</strong></p>
                    <p style="font-size: 11px; color: #666; margin-top: 8px;">
                       Only fits with R² ≥ 0.90 are included in the averages.
                    </p>
                </div>
                """
        else:
            return """
                <div class="summary-box" style="background: #fff3cd; border-left-color: #ffc107;">
                    <h3>No High-Quality Fits Available</h3>
                    <p>No fits with R² ≥ 0.90 found. Cannot calculate average rate constants.</p>
                    <p style="font-size: 11px; color: #666;">
                       Try adjusting initial parameter guesses or the time window.
                    </p>
                </div>
                """

    def export_publication_graph(self, data_df, fitted_df, time_col, trace_settings,
                                replicate_stats_df, replicate_info, t_start, t_end):
        """
        Export figure with scatter data and dashed fitted lines.

        Parameters
        ----------
        data_df : pd.DataFrame
            Experimental data
        fitted_df : pd.DataFrame or None
            Fitted data
        time_col : str
            Name of time column
        trace_settings : dict
            Dictionary of trace visibility settings
        replicate_stats_df : pd.DataFrame or None
            Replicate statistics dataframe
        replicate_info : dict
            Dictionary of replicate group information
        t_start : float
            Start time for export window
        t_end : float
            End time for export window
        """
        try:
            from plotnine import (ggplot, aes, geom_point, geom_line, geom_ribbon, labs,
                                  theme_bw, theme, element_text, element_line, element_rect,
                                  element_blank, scale_color_manual, scale_fill_manual,
                                  scale_linetype_manual, scale_x_continuous, scale_y_continuous,
                                  guides)
            from mizani.breaks import breaks_extended
        except ImportError:
            QMessageBox.critical(
                self.parent,
                "Missing Package",
                "plotnine package is required for ggplot export.\n\nInstall with: pip install plotnine"
            )
            logger.error("plotnine package not installed")
            return

        def lighten_hex_color(hex_color, factor=0.5):
            """Lighten a hex colour by blending towards white."""
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            r = int(r + (255 - r) * factor)
            g = int(g + (255 - g) * factor)
            b = int(b + (255 - b) * factor)
            return f'#{r:02x}{g:02x}{b:02x}'

        file_dialog = QFileDialog(self.parent)
        file_path, _ = file_dialog.getSaveFileName(
            self.parent,
            "Save Graph",
            "",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)"
        )

        if file_path:
            try:
                # Prepare data for ggplot
                if data_df is None or data_df.empty:
                    QMessageBox.warning(self.parent, "No Data", "No data available to export.")
                    return

                logger.info(f"Exporting graph with time window: {t_start:.2f} to {t_end:.2f} min")

                # Get ONLY SELECTED traces from GUI
                selected_traces = []
                for col, settings in trace_settings.items():
                    if settings.get('show_trace', False):
                        selected_traces.append(col)

                if not selected_traces:
                    QMessageBox.warning(self.parent, "No Traces Selected",
                                        "Please select at least one trace in the GUI to export.")
                    return

                logger.info(f"Exporting {len(selected_traces)} selected traces: {selected_traces}")

                # Filter data to time window using centralised utility
                data_windowed = filter_time_window(data_df, time_col, t_start, t_end)

                fitted_windowed = None
                if fitted_df is not None and not fitted_df.empty:
                    fitted_windowed = filter_time_window(fitted_df, time_col, t_start, t_end)

                # Normalise time to start from zero using FIRST DATA POINT
                # This ensures publication graph matches exported data (both start at time=0)
                first_time_point = data_windowed[time_col].min()
                normalise_time_column(data_windowed, time_col, first_time_point)
                if fitted_windowed is not None and not fitted_windowed.empty:
                    normalise_time_column(fitted_windowed, time_col, first_time_point)

                logger.info(f"Normalised time for publication graph: shifted by {first_time_point:.2f} min")

                # Build plot data - prioritise replicate means if available
                plot_data_list = []
                ribbon_data_list = []
                trace_names = []

                # Check if we have replicate statistics
                if replicate_stats_df is not None and not replicate_stats_df.empty:
                    # Use replicate means for grouped traces - filter to time window
                    stats_windowed = filter_time_window(replicate_stats_df, time_col, t_start, t_end)
                    # Normalise time for replicate stats using same first data point
                    normalise_time_column(stats_windowed, time_col, first_time_point)

                    for group_name, rep_info in replicate_info.items():
                        # Check if any member of this replicate group is selected
                        members = rep_info.get('columns', [])
                        if not any(m in selected_traces for m in members):
                            continue

                        mean_col = f"{group_name}_mean"
                        sem_col = f"{group_name}_sem"

                        if mean_col in stats_windowed.columns:
                            # Check if this trace has valid data (not all NaN or all zeros)
                            signal_data = stats_windowed[mean_col].dropna()
                            if len(signal_data) == 0 or (signal_data.abs() < 1e-10).all():
                                logger.info(f"Skipping empty replicate group: {group_name}")
                                continue

                            trace_names.append(group_name)

                            # Add mean trace as scatter
                            temp_df = pd.DataFrame({
                                'Time': stats_windowed[time_col],
                                'Signal': stats_windowed[mean_col],
                                'Trace': group_name,
                                'Type': 'Experimental',
                                'Linetype': 'Data'
                            })
                            plot_data_list.append(temp_df)

                            # Add error ribbon if SEM available
                            if sem_col in stats_windowed.columns:
                                sem = stats_windowed[sem_col].values
                                signal = stats_windowed[mean_col].values
                                ribbon_df = pd.DataFrame({
                                    'Time': stats_windowed[time_col],
                                    'ymin': signal - sem,
                                    'ymax': signal + sem,
                                    'Trace': group_name
                                })
                                ribbon_data_list.append(ribbon_df)

                            logger.info(f"Added replicate mean for group: {group_name}")

                    # Add individual traces that are NOT part of any replicate group
                    all_replicate_members = set()
                    for rep_info in replicate_info.values():
                        all_replicate_members.update(rep_info.get('columns', []))

                    for col in selected_traces:
                        if col not in all_replicate_members and col in data_windowed.columns:
                            # Check if this trace has valid data
                            signal_data = data_windowed[col].dropna()
                            if len(signal_data) == 0 or (signal_data.abs() < 1e-10).all():
                                logger.info(f"Skipping empty individual trace: {col}")
                                continue

                            trace_names.append(col)
                            temp_df = pd.DataFrame({
                                'Time': data_windowed[time_col],
                                'Signal': data_windowed[col],
                                'Trace': col,
                                'Type': 'Experimental',
                                'Linetype': 'Data'
                            })
                            plot_data_list.append(temp_df)
                            logger.info(f"Added individual trace: {col}")

                else:
                    # No replicate detection - use selected individual traces
                    for col in selected_traces:
                        if col in data_windowed.columns:
                            # Check if this trace has valid data
                            signal_data = data_windowed[col].dropna()
                            if len(signal_data) == 0 or (signal_data.abs() < 1e-10).all():
                                logger.info(f"Skipping empty trace: {col}")
                                continue

                            trace_names.append(col)
                            temp_df = pd.DataFrame({
                                'Time': data_windowed[time_col],
                                'Signal': data_windowed[col],
                                'Trace': col,
                                'Type': 'Experimental',
                                'Linetype': 'Data'
                            })
                            plot_data_list.append(temp_df)

                # Add fitted data for selected traces (only if experimental data was plotted)
                if fitted_windowed is not None and not fitted_windowed.empty:
                    for col in selected_traces:
                        # Only add fitted curve if the experimental trace was actually added
                        if col not in trace_names:
                            continue

                        fitted_col = f"{col}_fitted"
                        if fitted_col in fitted_windowed.columns:
                            # Check if fitted data has valid values
                            fitted_data = fitted_windowed[fitted_col].dropna()
                            if len(fitted_data) == 0:
                                continue

                            temp_df = pd.DataFrame({
                                'Time': fitted_windowed[time_col],
                                'Signal': fitted_windowed[fitted_col],
                                'Trace': col,
                                'Type': 'Fitted',
                                'Linetype': 'Fit'
                            })
                            plot_data_list.append(temp_df)
                            logger.info(f"Added fitted trace for: {col}")

                if not plot_data_list:
                    QMessageBox.warning(self.parent, "No Data", 
                                       "No traces available to export in selected time window.")
                    return

                plot_data = pd.concat(plot_data_list, ignore_index=True)

                # Colourblind-friendly palette (Wong 2011)
                colors = [
                    '#E69F00',  # Orange
                    '#56B4E9',  # Sky Blue
                    '#009E73',  # Bluish Green
                    '#F0E442',  # Yellow
                    '#0072B2',  # Blue
                    '#D55E00',  # Vermillion
                    '#CC79A7',  # Reddish Purple
                    '#000000',  # Black
                ]

                # Extend colours if needed
                n_traces = len(trace_names)
                if n_traces > len(colors):
                    colors = colors * ((n_traces // len(colors)) + 1)

                color_map = {trace: colors[i] for i, trace in enumerate(trace_names)}
                # Create lighter versions of colours for error bar fills
                fill_map = {trace: lighten_hex_color(color, factor=0.4) 
                           for trace, color in color_map.items()}

                # Create base plot with scatter for experimental data
                exp_data = plot_data[plot_data['Type'] == 'Experimental']
                fit_data = plot_data[plot_data['Type'] == 'Fitted']

                p = (ggplot(exp_data, aes(x='Time', y='Signal', color='Trace'))
                     + geom_point(size=1.2, alpha=0.6, stroke=0))

                # Add fitted lines (dashed for clear distinction)
                if not fit_data.empty:
                    p = p + geom_line(aes(x='Time', y='Signal', color='Trace', linetype='Linetype'),
                                      data=fit_data, size=0.6)

                # Add error ribbons if available (visible, lighter colour)
                if ribbon_data_list:
                    ribbon_data = pd.concat(ribbon_data_list, ignore_index=True)
                    p = p + geom_ribbon(aes(x='Time', ymin='ymin', ymax='ymax', fill='Trace'),
                                        data=ribbon_data, alpha=0.35, inherit_aes=False)

                # Apply colour scheme
                p = (p + scale_color_manual(values=color_map)
                     + scale_fill_manual(values=fill_map)
                     + scale_linetype_manual(values={'Data': 'solid', 'Fit': 'dashed'})
                     + guides(linetype='none'))

                # Intelligent axis breaks
                p = (p + scale_x_continuous(breaks=breaks_extended(n=6))
                     + scale_y_continuous(breaks=breaks_extended(n=6)))

                # Labels and theme for scientific publication
                p = (p + labs(
                    x='Time (min)',
                    y='Concentration (nM)',
                    color='Sample',
                    fill='Sample'
                )
                     + theme_bw()
                     + theme(
                            figure_size=(3.5, 2.625),  # Single column width for journals (~89mm)
                            dpi=300,
                            text=element_text(family='Arial', size=8),
                            axis_text=element_text(size=7, colour='black'),
                            axis_title=element_text(size=8, weight='bold'),
                            axis_line=element_line(colour='black', size=0.5),
                            axis_ticks=element_line(colour='black', size=0.5),
                            panel_grid_major=element_blank(),
                            panel_grid_minor=element_blank(),
                            panel_border=element_rect(colour='black', size=0.8, fill='none'),
                            legend_position='top',
                            legend_title=element_text(size=7, weight='bold'),
                            legend_text=element_text(size=6),
                            legend_key_size=8,
                            legend_background=element_rect(fill='white', colour='black', size=0.3),
                            plot_background=element_rect(fill='white'),
                            plot_margin=element_rect(t=5, r=5, b=5, l=5)
                        ))

                # Determine file format and DPI
                ext = file_path.split('.')[-1].lower()
                dpi = 600 if ext == 'png' else 300  # Higher DPI for publication

                # Save the plot
                p.save(filename=file_path, dpi=dpi, verbose=False)

                logger.info("Graph exported as '%s'", file_path)
                QMessageBox.information(self.parent, "Export Success",
                                        f"Graph saved to:\n{file_path}\n\n"
                                        f"Exported {len(trace_names)} trace(s)\n"
                                        f"Format: {ext.upper()}, DPI: {dpi}\n"
                                        f"Figure size: 3.5\" × 2.625\" (single column)\n"
                                        f"\n✓ Data points shown as small circles\n"
                                        f"✓ Fitted curves shown as dashed lines\n"
                                        f"✓ Error bands shown as transparent shaded regions")
            except Exception as e:
                logger.error("Graph export failed: %s", e, exc_info=True)
                QMessageBox.critical(self.parent, "Export Error", f"Graph export failed: {e}")
        else:
            logger.info("Export cancelled.")
