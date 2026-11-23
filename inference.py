"""Output 7: forecast comparison, paired sector inference and robustness."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from arch import arch_model
import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from analysis_config import AnalysisConfig, AnalysisOutput
    from data_processing import SectorData
    from time_series import ModelSample, VolatilityFit, VolatilityForecast


PRIMARY_BLOCK_LENGTH = 20
SENSITIVITY_BLOCK_LENGTH = 40
SENSITIVITY_REPS = 199
BOOTSTRAP_SEED = 104108
DM_HAC_LAGS = 5
PERIODS = (
    ("Full sample", None, None),
    ("Pre-COVID", None, "2019-12-31"),
    ("COVID (2020–2021)", "2020-01-01", "2021-12-31"),
    ("Post-COVID (2022+)", "2022-01-01", None),
)
ROBUSTNESS_SPECS = ("GARCH-n", "GARCH-t", "GJR-t", "EGARCH-t")
FIT_ERRORS = (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError)


@dataclass(frozen=True)
class InferenceResult:
    diebold_mariano: pd.DataFrame
    cross_sector_bootstrap: pd.DataFrame
    subsample_robustness: pd.DataFrame
    window_robustness: pd.DataFrame
    bootstrap_fit_log: pd.DataFrame


def _adjust_holm(frame, group_columns):
    """Holm's step-down p-values; missing results still count in their planned family."""
    frame = frame.copy()
    frame["p_holm"] = np.nan
    for _, group in frame.groupby(group_columns, sort=False):
        ordered = group["p_value"].dropna().sort_values()
        if not ordered.empty:
            scaled = ordered.to_numpy() * np.arange(len(group), len(group) - len(ordered), -1)
            frame.loc[ordered.index, "p_holm"] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return frame


def _dm_hac(difference):
    """Mean loss difference and a Student-t one-step DM test with Bartlett HAC."""
    values = np.asarray(difference, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    if n < 20 or not np.isfinite(values).all():
        return mean, np.nan, np.nan, np.nan
    centered = values - mean
    long_run_var = np.dot(centered, centered) / n
    for lag in range(1, DM_HAC_LAGS + 1):
        covariance = np.dot(centered[lag:], centered[:-lag]) / n
        long_run_var += 2 * (1 - lag / (DM_HAC_LAGS + 1)) * covariance
    if long_run_var <= np.finfo(float).eps * np.mean(values**2):
        return mean, np.nan, np.nan, np.nan
    standard_error = float(np.sqrt(long_run_var / n))
    # Harvey–Leybourne–Newbold multiplier at forecast horizon one.
    statistic = mean / standard_error * np.sqrt((n - 1) / n)
    return mean, standard_error, statistic, 2 * stats.t.sf(abs(statistic), n - 1)


def _daily_losses(frame, proxy=None):
    variance = frame["variance"]
    if proxy is None:
        proxy = frame["proxy_var"]
    return pd.DataFrame({
        "QLIKE": np.log(variance) + proxy / variance,
        "MAE": (variance - proxy).abs(),
        "MSE": (variance - proxy)**2,
    })


def _forecast_comparisons(forecast):
    rows = []
    for sector in dict.fromkeys(sector for sector, _ in forecast.forecasts):
        candidates = {
            model: _daily_losses(frame)
            for (candidate, model), frame in forecast.forecasts.items() if candidate == sector
        }
        for model_a, model_b in combinations(candidates, 2):
            a, b = candidates[model_a].align(candidates[model_b], join="inner", axis=0)
            for metric in ("QLIKE", "MAE"):
                losses = pd.concat([a[metric], b[metric]], axis=1).replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                diff = (losses.iloc[:, 0] - losses.iloc[:, 1]).to_numpy()
                mean, se, statistic, p_value = _dm_hac(diff)
                rows.append({
                    "sector": sector, "loss": metric, "model_a": model_a, "model_b": model_b,
                    "n_matched": len(diff), "loss_a_minus_b": mean, "hac_lags": DM_HAC_LAGS,
                    "hac_se": se, "dm_statistic": statistic, "p_value": p_value,
                })
    columns = [
        "sector", "loss", "model_a", "model_b", "n_matched", "loss_a_minus_b",
        "hac_lags", "hac_se", "dm_statistic", "p_value",
    ]
    return _adjust_holm(pd.DataFrame(rows, columns=columns), ["sector", "loss"])


def _fit_bootstrap_garch(residuals, starting_values=None):
    """Fit only the zero-mean GARCH-t parameters; no covariance calculation."""
    fit = arch_model(
        residuals, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist="t", rescale=False
    ).fit(
        disp="off", show_warning=False, update_freq=0,
        starting_values=starting_values, options={"maxiter": 350},
    )
    params = fit.params
    if fit.convergence_flag != 0 or not np.isfinite(params.to_numpy()).all():
        raise ValueError(f"GARCH-t optimizer flag {fit.convergence_flag}")
    persistence = float(params["alpha[1]"] + params["beta[1]"])
    return persistence, params.to_numpy()


def _circular_block_indices(rng, n, length):
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    return ((starts[:, None] + np.arange(length)) % n).reshape(-1)[:n]


def _basic_interval_and_test(point, values, alpha):
    """Basic interval and two-sided p-value from bootstrap sampling errors."""
    errors = values[np.isfinite(values)] - point
    if not np.isfinite(point) or len(errors) == 0:
        return np.nan, np.nan, np.nan, np.nan
    low_error, high_error = np.quantile(errors, [alpha / 2, 1 - alpha / 2])
    p_value = (1 + np.count_nonzero(np.abs(errors) >= abs(point))) / (len(errors) + 1)
    return point - high_error, point - low_error, p_value, float(errors.mean())


def _cross_sector_bootstrap(data, sample, config, reps):
    # Use the same training sessions for every sector and both metrics.
    aligned_residuals = pd.DataFrame(sample.mean_residuals).reindex(columns=data.active_sectors)
    aligned_residuals = aligned_residuals.sort_index().replace([np.inf, -np.inf], np.nan).dropna()
    aligned_raw = data.returns.reindex(aligned_residuals.index).reindex(columns=aligned_residuals.columns)
    complete = aligned_raw.notna().all(axis=1)
    aligned_residuals, aligned_raw = aligned_residuals.loc[complete], aligned_raw.loc[complete]
    names = list(aligned_residuals.columns)
    residuals, raw = aligned_residuals.to_numpy(dtype=float), aligned_raw.to_numpy(dtype=float)
    n, dimensionality = residuals.shape
    columns = [
        "metric", "block_sessions", "reps_requested", "reps_success", "sector_a", "sector_b",
        "n_matched_training", "training_start", "training_end", "seed", "point_a", "point_b",
        "difference_a_minus_b", "ci_level", "ci_low", "ci_high", "bootstrap_bias",
        "smallest_simulated_p", "p_value", "status",
    ]
    log_columns = [
        "block_sessions", "sector", "n_matched_training", "reps_requested",
        "reps_converged", "reps_near_boundary", "point_fit_status",
    ]
    if dimensionality < 2 or n < 2 * SENSITIVITY_BLOCK_LENGTH:
        return (
            _adjust_holm(pd.DataFrame(columns=columns), ["metric", "block_sessions"]),
            pd.DataFrame(columns=log_columns),
        )

    keys = ("annualized_volatility_pct", "garch_t_persistence")
    baseline = {
        keys[0]: np.std(raw, axis=0, ddof=1) * np.sqrt(252.0),
        keys[1]: np.full(dimensionality, np.nan),
    }
    initial_parameters = []
    fit_status = []
    for j, sector in enumerate(names):
        try:
            point, params = _fit_bootstrap_garch(residuals[:, j])
        except FIT_ERRORS as error:
            params = None
            fit_status.append(f"Unavailable: {str(error).splitlines()[0]}")
        else:
            baseline[keys[1]][j] = point
            fit_status.append("Fit near unit persistence" if point >= 0.995 else "Fit")
        initial_parameters.append(params)

    # Independent reproducible draws for each predeclared block length.
    seed_sequences = np.random.SeedSequence(BOOTSTRAP_SEED).spawn(2)
    settings = ((PRIMARY_BLOCK_LENGTH, reps), (SENSITIVITY_BLOCK_LENGTH, min(reps, SENSITIVITY_REPS)))
    rows, fit_log = [], []
    for (block_length, persistence_reps), sequence in zip(settings, seed_sequences):
        rng = np.random.default_rng(sequence)
        draws = {
            keys[0]: np.full((reps, dimensionality), np.nan),
            keys[1]: np.full((persistence_reps, dimensionality), np.nan),
        }
        for iteration in range(reps):
            # The same resampled dates are assigned to all sectors, preserving
            # observed cross-sector dependence within each contiguous block.
            positions = _circular_block_indices(rng, n, block_length)
            draws[keys[0]][iteration] = np.std(raw[positions], axis=0, ddof=1) * np.sqrt(252.0)
            if iteration >= persistence_reps:
                continue
            for j, params in enumerate(initial_parameters):
                if params is None:
                    continue
                try:
                    value, _ = _fit_bootstrap_garch(residuals[positions, j], params)
                except FIT_ERRORS:
                    continue
                draws[keys[1]][iteration, j] = value

        for j, sector in enumerate(names):
            values = draws[keys[1]][:, j]
            fit_log.append({
                "block_sessions": block_length, "sector": sector, "n_matched_training": n,
                "reps_requested": persistence_reps, "reps_converged": int(np.isfinite(values).sum()),
                "reps_near_boundary": int((values >= 0.995).sum()), "point_fit_status": fit_status[j],
            })
        for metric in keys:
            points = baseline[metric]
            count = len(draws[metric])
            for i, j in combinations(range(dimensionality), 2):
                point = points[i] - points[j]
                difference = draws[metric][:, i] - draws[metric][:, j]
                successful = int(np.isfinite(difference).sum())
                enough = successful >= max(39, int(np.ceil(0.8 * count)))
                interval = (np.nan,) * 4
                if enough and np.isfinite(point):
                    interval = _basic_interval_and_test(point, difference, config.significance)
                lo, hi, p_value, bias = interval
                rows.append({
                    "metric": metric, "block_sessions": block_length, "reps_requested": count,
                    "reps_success": successful, "sector_a": names[i], "sector_b": names[j],
                    "n_matched_training": n, "training_start": aligned_residuals.index[0],
                    "training_end": aligned_residuals.index[-1], "seed": BOOTSTRAP_SEED,
                    "point_a": points[i], "point_b": points[j], "difference_a_minus_b": point,
                    "ci_level": 1 - config.significance, "ci_low": lo, "ci_high": hi,
                    "bootstrap_bias": bias, "smallest_simulated_p": 1 / (successful + 1),
                    "p_value": p_value,
                    "status": "OK" if np.isfinite(p_value) else "Insufficient successful replicates",
                })

    return _adjust_holm(
        pd.DataFrame(rows, columns=columns), ["metric", "block_sessions"],
    ), pd.DataFrame(fit_log, columns=log_columns)


def _subsample_robustness(data, sample):
    """Retrospective refits used only to examine specification and time sensitivity."""
    from time_series import (
        _arma_residuals, _estimate_arma, _parameter_admissibility, fit_volatility,
    )

    rows = []
    for period, start, end in PERIODS:
        sector_returns = data.returns.loc[start:end, data.active_sectors].dropna()
        for sector in data.active_sectors:
            y = sector_returns[sector]
            raw_volatility = y.std(ddof=1) * np.sqrt(252.0)
            innovations, mean_status = None, ""
            if len(y) < 300:
                mean_status = "Fewer than 300 aligned returns"
            elif sector not in sample.orders:
                mean_status = "Training mean-model order unavailable"
            else:
                try:
                    order = sample.orders[sector]
                    mean_fit = _estimate_arma(y, order)
                    innovations = _arma_residuals(mean_fit, y.index, order)
                    if len(innovations) < 300:
                        innovations = None
                        mean_status = "Fewer than 300 usable innovations"
                except FIT_ERRORS as error:
                    mean_status = f"Mean fit: {str(error).splitlines()[0]}"

            for model in ROBUSTNESS_SPECS:
                row = {
                    "period": period, "sector": sector, "specification": model,
                    "start": y.index[0] if not y.empty else pd.NaT,
                    "end": y.index[-1] if not y.empty else pd.NaT,
                    "n_aligned_returns": len(y), "n_mean_innovations": 0,
                    "training_mean_order": str(sample.orders.get(sector, "")),
                    "annualized_volatility_pct": raw_volatility,
                    "mean_daily_conditional_volatility_pct": np.nan,
                    "alpha": np.nan, "beta": np.nan, "gamma": np.nan, "nu": np.nan,
                    "persistence": np.nan, "persistence_scale": "",
                    "half_life_sessions": np.nan,
                    "near_boundary_persistence": pd.NA, "parameter_admissible": pd.NA, "mean_reverting": pd.NA,
                    "status": mean_status,
                }
                if innovations is not None:
                    row["n_mean_innovations"] = len(innovations)
                    try:
                        params, h, _ = fit_volatility(innovations, model)
                    except FIT_ERRORS as error:
                        row["status"] = f"Variance fit: {str(error).splitlines()[0]}"
                    else:
                        alpha = float(params["alpha[1]"])
                        beta = float(params["beta[1]"])
                        gamma = float(params.get("gamma[1]", np.nan))
                        persistence = beta if model == "EGARCH-t" else alpha + beta
                        if model == "GJR-t":
                            # These standardized Student-t innovations are symmetric.
                            persistence += 0.5 * gamma
                        admissible, mean_reverting = _parameter_admissibility(model, params, persistence)
                        half_life = (
                            np.log(0.5) / np.log(abs(persistence))
                            if admissible and mean_reverting and 0 < abs(persistence) < 1 else np.nan
                        )
                        row.update({
                            "mean_daily_conditional_volatility_pct": np.sqrt(h).mean(),
                            "alpha": alpha, "beta": beta, "gamma": gamma,
                            "nu": params.get("nu", np.nan), "persistence": persistence,
                            "persistence_scale": "log-variance" if model == "EGARCH-t" else "variance",
                            "half_life_sessions": half_life,
                            "near_boundary_persistence": abs(persistence) >= 0.995,
                            "parameter_admissible": admissible, "mean_reverting": mean_reverting,
                            "status": (
                                "Fit with inadmissible parameters" if not admissible else
                                "Fit without mean reversion" if not mean_reverting else "Fit"
                            ),
                        })
                rows.append(row)
    return pd.DataFrame(rows)


def _window_robustness(sample, fit, rolling, config, output):
    from time_series import forecast_volatility

    expanded = forecast_volatility(
        sample, config, output, fit, expanding=True, models=("GARCH-t",), save=False
    )
    results = []
    for sector in sample.series_by_sector:
        key = (sector, "GARCH-t")
        if key not in expanded.forecasts or key not in rolling.forecasts:
            results.append({
                "sector": sector, "model": key[1], "n_matched": 0,
                "status": "A complete GARCH-t forecast path was unavailable",
            })
            continue
        aligned_index = rolling.forecasts[key].index.intersection(expanded.forecasts[key].index)
        rolling_frame = rolling.forecasts[key].loc[aligned_index]
        expanding_frame = expanded.forecasts[key].loc[aligned_index]
        # A volatility comparison requires a common realized proxy. Use the
        # rolling ARMA one-step innovation squared for both variance paths.
        proxy = rolling_frame["proxy_var"]
        a, b = _daily_losses(rolling_frame, proxy), _daily_losses(expanding_frame, proxy)
        row = {
            "sector": sector, "model": key[1], "n_matched": len(a),
            "test_start": a.index[0] if len(a) else pd.NaT, "test_end": a.index[-1] if len(a) else pd.NaT,
            "initial_rolling_train_n": min(config.max_window, sample.split_positions[sector]),
            "initial_expanding_train_n": sample.split_positions[sector],
            "refit_every_sessions": config.refit,
            "common_proxy": "Rolling one-step ARMA innovation squared",
            "status": "Fit",
        }
        for metric in ("QLIKE", "MAE", "MSE"):
            lower = metric.lower()
            row[f"rolling_{lower}"] = a[metric].mean()
            row[f"expanding_{lower}"] = b[metric].mean()
            row[f"rolling_minus_expanding_{lower}"] = (a[metric] - b[metric]).mean()
        results.append(row)
    return pd.DataFrame(results)


def evaluate_inference(
    data: SectorData,
    sample: ModelSample,
    fit: VolatilityFit,
    forecast: VolatilityForecast,
    config: AnalysisConfig,
    output: AnalysisOutput,
    *,
    bootstrap_reps: int = 399,
) -> InferenceResult:
    """Produce paired inference and robustness tables after the main risk evaluation."""
    if not isinstance(bootstrap_reps, (int, np.integer)) or bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be a positive integer")

    print("Output 7 · Aligned one-step forecast losses and Diebold–Mariano comparisons")
    dm = _forecast_comparisons(forecast)
    output.save_table(dm, "07_diebold_mariano.csv", index=False)
    if not dm.empty:
        print(
            f"DM uses Bartlett HAC({DM_HAC_LAGS}) and Holm adjustment within each sector/loss family. "
            "Negative loss differences favor model A. Heavy tails can make inference imprecise."
        )
        dm_counts = dm.assign(reject=dm["p_holm"] < config.significance).groupby(
            ["sector", "loss"], sort=False,
        ).agg(pairs=("p_value", "size"), valid_tests=("p_value", "count"), holm_rejections=("reject", "sum"))
        output.display(dm_counts.reset_index())

    print("Output 7 · Paired date-block bootstrap on the common training sample")
    bootstrap, bootstrap_log = _cross_sector_bootstrap(data, sample, config, bootstrap_reps)
    output.save_table(bootstrap, "07_cross_sector_bootstrap.csv", index=False)
    output.save_table(bootstrap_log, "07_bootstrap_fit_log.csv", index=False)
    primary = bootstrap.loc[bootstrap["block_sessions"].eq(PRIMARY_BLOCK_LENGTH)]
    if not primary.empty:
        print(
            "Primary block length 20 sessions. Length 40 checks sensitivity. "
            "Basic intervals are pointwise and Holm families are defined by metric and block length. "
            "GARCH-t inference conditions on the previously estimated ARMA mean filters. "
            "Persistent shocks spanning block boundaries may be partly broken by resampling."
        )
        output.display(primary[[
            "metric", "sector_a", "sector_b", "difference_a_minus_b",
            "ci_low", "ci_high", "p_holm", "reps_success",
        ]])
    else:
        print("At least two sectors and 80 aligned training innovations are needed for this analysis.")

    print("Output 7 · Retrospective full-sample and period-specific parameter estimates")
    subsample = _subsample_robustness(data, sample)
    output.save_table(subsample, "07_subsample_robustness.csv", index=False)
    if not subsample.empty:
        period_table = subsample.loc[subsample["specification"].eq("GARCH-t")].pivot(
            index="sector", columns="period", values="persistence"
        ).reindex(columns=[name for name, _, _ in PERIODS])
        output.display(period_table)

    print("Output 7 · Rolling versus expanding GARCH-t forecasting windows")
    windows = _window_robustness(sample, fit, forecast, config, output)
    output.save_table(windows, "07_expanding_vs_rolling.csv", index=False)
    if not windows.empty:
        kept = ["sector", "n_matched", "rolling_qlike", "expanding_qlike", "status"]
        output.display(windows.reindex(columns=kept))

    return InferenceResult(dm, bootstrap, subsample, windows, bootstrap_log)
