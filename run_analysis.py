"""Run the Group 10 analysis in the order of the seven outline outputs."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib


def main():
    matplotlib.use("Agg")
    from analysis_config import AnalysisConfig, AnalysisOutput
    from data_processing import describe_sector_data, load_sector_data
    from inference import evaluate_inference
    from research_findings import write_findings
    from risk_analysis import backtest_risk, forecast_risk
    from sector_interactions import analyze_interactions
    from time_series import (
        fit_mean_models, fit_volatility_models, forecast_volatility, publish_volatility_evaluation,
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    defaults = AnalysisConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--bootstrap-reps", type=int, default=399,
                        help="bootstrap draws (default: 399; length-40 persistence refits are capped at 199)")
    args = parser.parse_args()
    if args.bootstrap_reps < 19:
        parser.error("--bootstrap-reps must be at least 19")
    config = replace(defaults, raw_dir=args.raw_dir.resolve(), output_dir=args.output_dir.resolve())
    output = AnalysisOutput(config.output_dir)

    print("\nOutput 1 | Source data, synchronized returns, and quality checks")
    data = load_sector_data(config, output)
    print("\nOutput 2 | Descriptive statistics and stylized facts")
    describe_sector_data(data, config, output)
    print("\nOutput 3 | Conditional mean and ARCH diagnostics")
    sample = fit_mean_models(data, config, output)
    print("\nOutput 4 | Sector volatility, shock sensitivity, and persistence")
    fit = fit_volatility_models(sample, config, output)
    print("\nOutput 5 | VAR and CCC/DCC-GARCH sector interactions")
    interactions = analyze_interactions(data, sample, config, output)
    print("\nOutput 6 | One-session VaR and ES")
    forecast = forecast_volatility(sample, config, output, fit, save=False)
    risk = forecast_risk(forecast, config, output, data=data)
    print("\nOutput 7 | Forecast evaluation, backtesting, inference, and robustness")
    publish_volatility_evaluation(fit, forecast, output)
    backtest = backtest_risk(data, sample, fit, forecast, risk, config, output)
    inference = evaluate_inference(data, sample, fit, forecast, config, output,
                                   bootstrap_reps=args.bootstrap_reps)
    print("\nEconomic interpretation and research findings")
    print(write_findings(data, fit, interactions, risk, backtest, inference, config, output))
    print(f"\nGenerated results in {config.output_dir}")


if __name__ == "__main__":
    main()
