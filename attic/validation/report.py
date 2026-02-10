"""
Report Generator

Generates human-readable reports from validation results.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class ReportGenerator:
    """
    Generates reports from validation and correlation analysis.

    Outputs:
    - Markdown report with findings
    - JSON summary for programmatic access
    - CSV of event-by-event results
    """

    def __init__(self):
        """Initialize report generator."""
        pass

    def generate_markdown_report(
        self,
        validation_results: Dict,
        correlation_results: Dict,
        output_file: str
    ):
        """
        Generate a comprehensive markdown report.

        Args:
            validation_results: Results from Backtester.run_validation()
            correlation_results: Results from CorrelationAnalyzer
            output_file: Output markdown file path
        """
        report_lines = []

        # Header
        report_lines.extend(self._generate_header())

        # Executive Summary
        report_lines.extend(self._generate_executive_summary(
            validation_results,
            correlation_results
        ))

        # Methodology
        report_lines.extend(self._generate_methodology())

        # Validation Results
        report_lines.extend(self._generate_validation_results(validation_results))

        # Correlation Analysis
        report_lines.extend(self._generate_correlation_analysis(correlation_results))

        # Event-by-Event Results
        report_lines.extend(self._generate_event_details(validation_results))

        # Conclusions
        report_lines.extend(self._generate_conclusions(validation_results))

        # Footer
        report_lines.extend(self._generate_footer())

        # Write to file
        report = '\n'.join(report_lines)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"✓ Markdown report generated: {output_file}")

    def _generate_header(self) -> List[str]:
        """Generate report header."""
        return [
            "# Stop-Hunt Detector Validation Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            "",
            "---",
            ""
        ]

    def _generate_executive_summary(
        self,
        validation_results: Dict,
        correlation_results: Dict
    ) -> List[str]:
        """Generate executive summary."""
        lines = [
            "## Executive Summary",
            ""
        ]

        summary = validation_results.get('summary', {})
        metrics = validation_results.get('metrics', {})

        lines.extend([
            f"**Total Events Validated:** {summary.get('total_events', 0)}",
            f"**Stress Elevation Rate:** {summary.get('stress_elevation_pct', 0):.1f}%",
            f"**Average Stress Before Events:** {summary.get('avg_stress_before', 0):.3f}",
            "",
            "### Success Criteria",
            ""
        ])

        success_criteria = summary.get('success_criteria', {})
        for criteria_name, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            lines.append(f"- **{criteria_name.replace('_', ' ').title()}:** {status}")

        lines.extend([
            "",
            f"**Overall Result:** {'✅ SUCCESS' if summary.get('success') else '❌ FAILURE'}",
            ""
        ])

        return lines

    def _generate_methodology(self) -> List[str]:
        """Generate methodology section."""
        return [
            "## Methodology",
            "",
            "### Stress Signals",
            "",
            "The detector measures four market stress signals:",
            "",
            "1. **Volatility Compression** - Bollinger Band squeeze + ATR decay",
            "2. **Liquidity Fragility** - Wick-to-body ratio + volume-to-range",
            "3. **Continuation Failure** - RSI/MACD divergence + momentum deceleration",
            "4. **Speed Asymmetry** - Downward vs upward velocity bias",
            "",
            "### Stress Score",
            "",
            "Composite score (0.0 → 1.0) calculated as weighted sum of signals:",
            "",
            "- Volatility: 30%",
            "- Liquidity: 30%",
            "- Continuation: 20%",
            "- Speed: 20%",
            "",
            "### State Labels",
            "",
            "- **NORMAL** (0.0 - 0.3): Calm market",
            "- **STRESSED** (0.3 - 0.6): Elevated stress",
            "- **FRAGILE** (0.6 - 0.8): Multiple stress conditions",
            "- **IMMINENT_CLEARING** (0.8 - 1.0): High liquidation probability",
            "",
            "### Validation Approach",
            "",
            "Historical validation against tagged liquidation events:",
            "",
            "- Check stress levels 1-4 hours before each event",
            "- Measure stress elevation vs baseline",
            "- Calculate correlation between stress scores and events",
            "- Assess signal-to-noise ratio",
            ""
        ]

    def _generate_validation_results(self, validation_results: Dict) -> List[str]:
        """Generate validation results section."""
        lines = [
            "## Validation Results",
            ""
        ]

        metrics = validation_results.get('metrics', {})

        lines.extend([
            "### Event Statistics",
            "",
            f"- Total events: {metrics.get('total_events_validated', 0)}",
            f"- Events with pre-event data: {metrics.get('events_with_pre_data', 0)}",
            f"- Stress elevation rate: {metrics.get('stress_elevation_rate', 0) * 100:.1f}%",
            "",
            "### Stress Metrics",
            "",
            f"- Average stress at event time: {metrics.get('avg_event_stress', 0):.3f}",
            f"- Average pre-event stress: {metrics.get('avg_pre_event_stress', 0):.3f}",
            f"- Average pre-event max stress: {metrics.get('avg_pre_event_max_stress', 0):.3f}",
            "",
            "### State Distribution at Events",
            ""
        ])

        state_dist = metrics.get('state_distribution_at_event', {})
        for state, count in state_dist.items():
            lines.append(f"- {state}: {count} events")

        lines.append("")

        return lines

    def _generate_correlation_analysis(self, correlation_results: Dict) -> List[str]:
        """Generate correlation analysis section."""
        lines = [
            "## Correlation Analysis",
            ""
        ]

        # Event correlation
        if 'pearson_correlation' in correlation_results:
            lines.extend([
                "### Stress-Event Correlation",
                "",
                f"- Pearson: {correlation_results['pearson_correlation']:.3f} (p={correlation_results['pearson_p_value']:.4f})",
                f"- Spearman: {correlation_results['spearman_correlation']:.3f} (p={correlation_results['spearman_p_value']:.4f})",
                ""
            ])

        # Stress elevation
        if 'stress_elevation' in correlation_results:
            lines.extend([
                "### Stress Elevation",
                "",
                f"- Baseline stress: {correlation_results.get('avg_baseline_stress', 0):.3f}",
                f"- Pre-event stress: {correlation_results.get('avg_pre_event_stress', 0):.3f}",
                f"- Elevation: {correlation_results.get('stress_elevation', 0):.3f}",
                f"- Elevation ratio: {correlation_results.get('stress_elevation_ratio', 0):.2f}x",
                ""
            ])

        # Signal-to-noise
        if 'signal_to_noise_ratio' in correlation_results:
            lines.extend([
                "### Signal-to-Noise Ratio",
                "",
                f"- Signal mean (stress > 0.7): {correlation_results.get('signal_mean', 0):.3f}",
                f"- Noise mean (stress < 0.7): {correlation_results.get('noise_mean', 0):.3f}",
                f"- S/N ratio: {correlation_results.get('signal_to_noise_ratio', 0):.3f}",
                ""
            ])

        return lines

    def _generate_event_details(self, validation_results: Dict) -> List[str]:
        """Generate event-by-event details."""
        lines = [
            "## Event-by-Event Details",
            ""
        ]

        events = validation_results.get('events', [])

        lines.extend([
            "| Event | Time | Severity | Event Stress | Pre-Event Max | Elevated? |",
            "|-------|------|----------|---------------|---------------|-----------|"
        ])

        for event in events:
            time_str = event['event_time'].strftime('%Y-%m-%d %H:%M') if isinstance(event['event_time'], str) else str(event['event_time'])
            elevated = "✅" if event['stress_elevated'] else "❌"

            lines.append(
                f"| {event['description']} | {time_str} | {event['severity']} | "
                f"{event['event_stress']:.3f} | {event['pre_event_max_stress']:.3f} | {elevated} |"
            )

        lines.append("")

        return lines

    def _generate_conclusions(self, validation_results: Dict) -> List[str]:
        """Generate conclusions section."""
        lines = [
            "## Conclusions",
            ""
        ]

        summary = validation_results.get('summary', {})

        if summary.get('success'):
            lines.extend([
                "✅ **Experiment SUCCESS**",
                "",
                "The stress detector successfully meets the validation criteria:",
                "- Stress elevation correlates with liquidation events",
                "- Signal-to-noise ratio shows distinct stress elevation",
                "- The detector shows promise for early warning of stop hunts",
                ""
            ])
        else:
            lines.extend([
                "❌ **Experiment FAILURE**",
                "",
                "The stress detector does not meet the validation criteria:",
                "- Stress elevation is insufficient or inconsistent",
                "- Signal-to-noise ratio is poor",
                "- Further refinement is needed",
                ""
            ])

        lines.extend([
            "### Next Steps",
            "",
            "- Review and refine signal weights",
            "- Consider additional stress signals",
            "- Test on different market conditions",
            "- If criteria met: proceed to real-time monitoring",
            "- If criteria not met: iterate or kill experiment",
            ""
        ])

        return lines

    def _generate_footer(self) -> List[str]:
        """Generate report footer."""
        return [
            "---",
            "",
            "*Report generated by Stop-Hunt Detector Validation System*",
            ""
        ]

    def export_csv(self, validation_results: Dict, output_file: str):
        """
        Export event results to CSV.

        Args:
            validation_results: Results from Backtester.run_validation()
            output_file: Output CSV file path
        """
        import pandas as pd

        events = validation_results.get('events', [])

        if not events:
            print("No events to export")
            return

        # Convert to DataFrame
        df = pd.DataFrame(events)

        # Format timestamps
        if 'event_time' in df.columns:
            df['event_time'] = pd.to_datetime(df['event_time'])

        # Save to CSV
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        print(f"✓ CSV exported: {output_file}")

    def save_json_summary(
        self,
        validation_results: Dict,
        correlation_results: Dict,
        output_file: str
    ):
        """
        Save JSON summary for programmatic access.

        Args:
            validation_results: Results from Backtester.run_validation()
            correlation_results: Results from CorrelationAnalyzer
            output_file: Output JSON file path
        """
        summary = {
            'generated_at': datetime.now().isoformat(),
            'validation': validation_results.get('summary', {}),
            'metrics': validation_results.get('metrics', {}),
            'correlation': correlation_results,
            'success': validation_results.get('summary', {}).get('success', False)
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"✓ JSON summary saved: {output_file}")


if __name__ == "__main__":
    # Example usage
    report = ReportGenerator()

    # Create dummy results for testing
    dummy_validation = {
        'events': [],
        'metrics': {
            'total_events_validated': 0,
            'stress_elevation_rate': 0.0
        },
        'summary': {
            'total_events': 0,
            'success': False
        }
    }

    dummy_correlation = {
        'pearson_correlation': 0.0,
        'stress_elevation': 0.0
    }

    # Generate report
    report.generate_markdown_report(
        dummy_validation,
        dummy_correlation,
        '/tmp/test_report.md'
    )
    print("Report generation test complete")
