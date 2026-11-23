"""Outputs 6–7: one-session downside risk forecasts and their backtests."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlog1py, xlogy
from statsmodels.regression.linear_model import OLS

if TYPE_CHECKING:
    from analysis_config import AnalysisConfig, AnalysisOutput
    from data_processing import SectorData
    from time_series import ModelSample, VolatilityFit, VolatilityForecast


_RISK_GROUP_COLUMNS = ["sector", "model", "confidence_pct"]
CONFIDENCE_LEVELS = (99, 95)
FHS_MODEL = "FHS GARCH-t"
HISTORICAL_MODEL = "Historical Simulation"
VAR_COV_MODEL = "Variance-Covariance Normal"
SPELL_MODEL = "GARCH-t"
SPELL_PERCENTILE = 0.90
DQ_LAGS = 4
DQ_DRAWS = 9_999


@dataclass(frozen=True)
class RiskForecast:
    daily_risk: pd.DataFrame
    risk_summary: pd.DataFrame
    downside_ranks: pd.DataFrame
    garch_tail_effect: pd.DataFrame
    high_var_spells: pd.DataFrame
    high_var_summary: pd.DataFrame


@dataclass(frozen=True)
class RiskBacktest:
    backtests: pd.DataFrame
    comparison: pd.DataFrame
    persistence_link: pd.DataFrame


def tail_forecast(forecast: pd.DataFrame, confidence: int) -> pd.DataFrame:
    """Calculate signed loss thresholds, in the units of the input returns."""
    alpha = (100 - confidence) / 100
    sigma = np.sqrt(forecast["variance"])
    nu = forecast.get("nu", pd.Series(np.nan, index=forecast.index))
    if nu.isna().all():
        quantile = stats.norm.ppf(alpha)
        tail_mean = -stats.norm.pdf(quantile) / alpha
    elif nu.gt(2).all():
        scale = np.sqrt((nu - 2) / nu)
        t_quantile = stats.t.ppf(alpha, nu)
        quantile = scale * t_quantile
        tail_mean = -scale * stats.t.pdf(t_quantile, nu) * (
            (nu + t_quantile ** 2) / ((nu - 1) * alpha)
        )
    else:
        raise ValueError("Student-t degrees of freedom must be > 2 on every forecast date.")

    return _loss_frame(
        -forecast["actual"], -(forecast["mu"] + sigma * quantile), -(forecast["mu"] + sigma * tail_mean),
    )


def _loss_frame(actual_loss, var, es) -> pd.DataFrame:
    risk = pd.DataFrame({"loss_pct": actual_loss, "var_pct": var, "es_pct": es})
    if not np.isfinite(risk.to_numpy()).all() or (risk["es_pct"] < risk["var_pct"]).any():
        raise ValueError("Invalid VaR or ES forecast. Check the mean, variance and tail parameters.")
    risk["exceeded"] = risk["loss_pct"].gt(risk["var_pct"])
    return risk


def _empirical_lower_tail(returns: np.ndarray, tail_pct: int) -> tuple[float, float]:
    """Empirical quantile and integrated quantile using exactly tail_pct/100 mass."""
    ordered = np.sort(returns)
    whole, remainder = divmod(tail_pct * len(ordered), 100)
    cutoff = whole + bool(remainder) - 1
    fraction = remainder / 100
    return float(ordered[cutoff]), float((ordered[:whole].sum() + fraction * ordered[cutoff]) / (whole + fraction))


def _historical_risk(history: pd.Series, holdout: pd.DataFrame, max_window: int):
    """Compute rolling historical and normal benchmarks from preceding sessions only."""
    prior = history.index.get_indexer(holdout.index)
    if (prior < 2).any() or not np.allclose(history.iloc[prior], holdout["actual"]):
        raise ValueError("Benchmark history must cover the same return series as the model holdout.")
    values = history.to_numpy()
    moments = []
    historical = {confidence: [] for confidence in CONFIDENCE_LEVELS}
    for position in prior:
        window = values[max(0, position - max_window):position]
        moments.append((window.mean(), window.var(ddof=1)))
        for confidence in CONFIDENCE_LEVELS:
            historical[confidence].append(_empirical_lower_tail(window, 100 - confidence))

    moment_forecast = pd.DataFrame(moments, index=holdout.index, columns=["mu", "variance"])
    moment_forecast["actual"] = holdout["actual"]
    for confidence, left_tail in historical.items():
        quantile, tail_mean = np.asarray(left_tail).T
        yield HISTORICAL_MODEL, confidence, _loss_frame(-holdout["actual"], -quantile, -tail_mean)
        yield VAR_COV_MODEL, confidence, tail_forecast(moment_forecast, confidence)


def _filtered_risk(forecast: pd.DataFrame, confidence: int) -> pd.DataFrame:
    tail_pct = 100 - confidence
    sigma = np.sqrt(forecast["variance"])
    return _loss_frame(
        -forecast["actual"],
        -(forecast["mu"] + sigma * forecast[f"fhs_q{tail_pct:02}"]),
        -(forecast["mu"] + sigma * forecast[f"fhs_es{tail_pct:02}"]),
    )


def _risk_rows(sector, model, confidence, frame) -> pd.DataFrame:
    return frame.rename_axis("date").reset_index().assign(sector=sector, model=model, confidence_pct=confidence)


def forecast_risk(
    forecast: VolatilityForecast, config: AnalysisConfig, output: AnalysisOutput, *, data: SectorData | None = None,
) -> RiskForecast:
    """Compute conditional and benchmark VaR/ES on each sector's common holdout."""
    risk_rows = []
    for (sector, model), sector_forecast in forecast.forecasts.items():
        for confidence in CONFIDENCE_LEVELS:
            risk_rows.append(_risk_rows(sector, model, confidence, tail_forecast(sector_forecast, confidence)))
        if model == SPELL_MODEL and {"fhs_q01", "fhs_es01", "fhs_q05", "fhs_es05"}.issubset(sector_forecast):
            for confidence in CONFIDENCE_LEVELS:
                risk_rows.append(_risk_rows(sector, FHS_MODEL, confidence, _filtered_risk(sector_forecast, confidence)))

    if data is not None:
        sector_holdouts = {sector: rows for (sector, _), rows in forecast.forecasts.items()}
        for sector in config.targets:
            if sector not in sector_holdouts:
                continue
            history = data.common_returns[sector].sort_index()
            for model, confidence, rows in _historical_risk(history, sector_holdouts[sector], config.max_window):
                risk_rows.append(_risk_rows(sector, model, confidence, rows))

    daily_risk = pd.DataFrame(columns=[
        "date", "loss_pct", "var_pct", "es_pct", "exceeded", *_RISK_GROUP_COLUMNS,
    ])
    if not risk_rows:
        print("Chưa có dự báo ngoài mẫu đủ dữ liệu để tính VaR và ES.")
        empty = pd.DataFrame()
        return RiskForecast(daily_risk, empty, empty, empty, empty, empty)

    daily_risk = pd.concat(risk_rows, ignore_index=True).sort_values("date", kind="stable", ignore_index=True)
    if daily_risk.duplicated([*_RISK_GROUP_COLUMNS, "date"]).any():
        raise ValueError("Each risk method must forecast each target date at most once.")
    _check_comparable_risk_dates(daily_risk)
    output.save_table(daily_risk, "06_daily_var_es.csv", index=False)
    risk_summary = daily_risk.groupby(_RISK_GROUP_COLUMNS, sort=False).agg(
        n_test=("date", "size"), first_date=("date", "min"), last_date=("date", "max"),
        mean_var_pct=("var_pct", "mean"), mean_es_pct=("es_pct", "mean"),
        last_var_pct=("var_pct", "last"), last_es_pct=("es_pct", "last"),
    ).reset_index()
    output.save_table(risk_summary, "06_var_es_summary.csv", index=False)
    risk_numbers = ["mean_var_pct", "mean_es_pct", "last_var_pct", "last_es_pct"]
    output.display(risk_summary.query("confidence_pct == 99 and model == @SPELL_MODEL").round(dict.fromkeys(risk_numbers, 4)))

    downside_ranks, garch_tail_effect = _compare_downside_risk(risk_summary, output)
    high_var_spells, high_var_summary = _high_var_spells(daily_risk, output)
    _plot_risk_forecasts(daily_risk, output)
    return RiskForecast(daily_risk, risk_summary, downside_ranks, garch_tail_effect, high_var_spells, high_var_summary)


def _check_comparable_risk_dates(daily_risk: pd.DataFrame) -> None:
    signatures = {
        tuple(rows["date"])
        for _, rows in daily_risk.groupby(_RISK_GROUP_COLUMNS, sort=False)
    }
    if len(signatures) != 1:
        raise ValueError("Risk methods and sectors must share common holdout sessions for these comparisons.")


def _compare_downside_risk(summary: pd.DataFrame, output: AnalysisOutput):
    ranks = summary.copy()
    within = ranks.groupby(["model", "confidence_pct"], sort=False)
    ranks["rank_var_high_to_low"] = within["mean_var_pct"].rank(method="min", ascending=False).astype(int)
    ranks["rank_es_high_to_low"] = within["mean_es_pct"].rank(method="min", ascending=False).astype(int)
    ranks["es_rank_minus_var_rank"] = ranks["rank_es_high_to_low"] - ranks["rank_var_high_to_low"]
    ranks["mean_es_minus_var_pct"] = ranks["mean_es_pct"] - ranks["mean_var_pct"]
    output.save_table(ranks, "06_downside_cross_sector_ranks.csv", index=False)
    contrasts = pd.DataFrame()
    if {"GARCH-n", "GARCH-t"}.issubset(set(ranks["model"])):
        scores = ["mean_var_pct", "mean_es_pct", "rank_var_high_to_low", "rank_es_high_to_low"]
        normal = ranks.query("model == 'GARCH-n'").set_index(["sector", "confidence_pct"])[scores]
        student = ranks.query("model == 'GARCH-t'").set_index(["sector", "confidence_pct"])[scores]
        alternative = ranks.loc[ranks["model"].isin(["GARCH-t", "GJR-t", "EGARCH-t"])]
        contrasts = alternative.copy()
        for baseline, table in [("garch_normal", normal), ("garch_t", student)]:
            contrasts = contrasts.merge(table.add_suffix(f"_{baseline}"), on=["sector", "confidence_pct"], how="left")
            for metric in scores:
                contrasts[f"{metric}_delta_vs_{baseline}"] = contrasts[metric] - contrasts[f"{metric}_{baseline}"]
        output.save_table(contrasts, "06_garch_tail_specification_comparison.csv", index=False)
    output.display(ranks.loc[ranks["model"].eq(SPELL_MODEL), [
        "sector", "confidence_pct", "rank_var_high_to_low", "rank_es_high_to_low", "mean_es_minus_var_pct",
    ]].round({"mean_es_minus_var_pct": 4}))
    return ranks, contrasts


def _high_var_spells(daily_risk: pd.DataFrame, output: AnalysisOutput):
    """Describe consecutive trading sessions above sector-relative holdout VaR deciles."""
    spells, summaries = [], []
    for sector, rows in daily_risk.query("confidence_pct == 99 and model == @SPELL_MODEL").groupby("sector", sort=False):
        rows = rows.sort_values("date", ignore_index=True)
        threshold = float(rows["var_pct"].quantile(SPELL_PERCENTILE))
        high_var = rows["var_pct"].gt(threshold).to_numpy()
        changes = np.diff(np.r_[False, high_var, False].astype(int))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        lengths = ends - starts
        for start, end in zip(starts, ends):
            spells.append({
                "sector": sector, "start_date": rows["date"].iloc[start], "end_date": rows["date"].iloc[end - 1],
                "observed_duration_sessions": end - start, "peak_var_pct": rows["var_pct"].iloc[start:end].max(),
                "touches_holdout_start": start == 0, "touches_holdout_end": end == len(rows),
            })
        summaries.append({
            "sector": sector, "model": SPELL_MODEL, "confidence_pct": 99,
            "holdout_quantile": SPELL_PERCENTILE, "var_threshold_pct": threshold,
            "n_test": len(rows), "n_high_var_sessions": int(high_var.sum()), "n_spells": len(lengths),
            "mean_observed_duration_sessions": lengths.mean() if len(lengths) else np.nan,
            "median_observed_duration_sessions": np.median(lengths) if len(lengths) else np.nan,
            "p90_observed_duration_sessions": np.quantile(lengths, 0.90) if len(lengths) else np.nan,
            "max_observed_duration_sessions": lengths.max() if len(lengths) else np.nan,
        })
    detail, summary = pd.DataFrame(spells), pd.DataFrame(summaries)
    if not summary.empty:
        output.save_table(detail, "06_high_var_spell_dates.csv", index=False)
        output.save_table(summary, "06_high_var_spell_summary.csv", index=False)
        output.display(summary.round(4))
    return detail, summary


def _plot_risk_forecasts(daily_risk: pd.DataFrame, output: AnalysisOutput) -> None:

    for figure_no, (sector, rows) in enumerate(
        daily_risk.loc[daily_risk["confidence_pct"].eq(99)].groupby("sector", sort=False),
        start=1,
    ):
        model_groups = list(rows.groupby("model", sort=False))
        n_models = len(model_groups)
        n_cols = min(n_models, 2)
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6.4 * n_cols, 2.75 * n_rows + 0.45),
            sharex=True, sharey=True, squeeze=False, layout="constrained",
        )
        for ax, (model, frame) in zip(axes.flat, model_groups):
            frame = frame.set_index("date").sort_index()
            hits = frame.loc[frame["exceeded"]]
            ax.plot(frame.index, frame["loss_pct"], color="#718096", lw=0.7, label="Realized loss")
            ax.plot(frame.index, frame["var_pct"], color="#1d4ed8", lw=1.25, label="VaR 99%")
            ax.plot(frame.index, frame["es_pct"], color="#0f766e", lw=0.9, ls="--", label="ES 99%")
            ax.scatter(hits.index, hits["loss_pct"], s=19, color="#b91c1c", zorder=3, label="VaR exceedance")
            ax.set(title=f"{model} · {len(hits)} exceedances", ylabel="Log loss (%)")
            ax.grid(axis="y", alpha=0.15)
        for ax in list(axes.flat)[n_models:]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle(f"Figure 6.{figure_no} · {sector} · One-session 99% VaR and ES", fontweight="bold")
        output.show_figure(fig, f"06_{sector.replace(' ', '_')}_var_es_99.png")


def bernoulli_loglik(count, n, probability):
    """Use the continuous convention 0 log(0) = 0 at the boundaries."""
    return xlogy(count, probability) + xlog1py(n - count, -probability)


def coverage_tests(exceeded: pd.Series, alpha: float) -> dict[str, float | int]:
    """Compute unconditional and conditional coverage tests from ordered VaR hits."""
    hits = np.asarray(exceeded, dtype=bool)
    n = len(hits)
    count = int(hits.sum())
    lr_uc = max(0.0, 2 * (
        bernoulli_loglik(count, n, count / n) - bernoulli_loglik(count, n, alpha)
    ))
    previous, current = hits[:-1], hits[1:]
    n00 = int((~previous & ~current).sum())
    n01 = int((~previous & current).sum())
    n10 = int((previous & ~current).sum())
    n11 = int((previous & current).sum())
    n0, n1 = n00 + n01, n10 + n11
    lr_ind = lr_cc = np.nan
    if n0 and n1:
        rate = (n01 + n11) / (n0 + n1)
        loglik_independent = bernoulli_loglik(n01 + n11, n0 + n1, rate)
        loglik_markov = bernoulli_loglik(n01, n0, n01 / n0) + bernoulli_loglik(n11, n1, n11 / n1)
        lr_ind = max(0.0, 2 * (loglik_markov - loglik_independent))
        lr_cc = lr_uc + lr_ind
    return {
        "n_test": n, "expected_hits": n * alpha, "hits": count, "hit_rate_pct": 100 * count / n,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "uc_lr": lr_uc, "uc_p": stats.chi2.sf(lr_uc, 1),
        "ind_lr": lr_ind, "ind_p": stats.chi2.sf(lr_ind, 1),
        "cc_lr": lr_cc, "cc_p": stats.chi2.sf(lr_cc, 2),
        "exact_binomial_p": stats.binomtest(count, n, p=alpha).pvalue,
    }


def _dq_statistic(hits: np.ndarray, alpha: float, lags: int):
    """Compute the out-of-sample DQ quadratic for one or many hit sequences."""
    windows = np.lib.stride_tricks.sliding_window_view(hits, lags + 1, axis=-1)
    centered = windows[..., -1] - alpha
    leading_ones = np.ones_like(centered)[..., None]
    design = np.concatenate([leading_ones, windows[..., :-1]], axis=-1)
    transposed = np.swapaxes(design, -1, -2)
    covariance = transposed @ design
    rank = np.linalg.matrix_rank(covariance, hermitian=True)
    projection = transposed @ centered[..., None]
    metric = np.linalg.pinv(covariance, hermitian=True)
    statistic = (np.swapaxes(projection, -1, -2) @ metric @ projection)[..., 0, 0] / (alpha * (1 - alpha))
    diverse = ((windows[..., -1].sum(axis=-1) > 0) & (windows[..., -1].sum(axis=-1) < centered.shape[-1]))
    return statistic, rank, diverse


@lru_cache(maxsize=16)
def _dq_null_statistics(n: int, alpha: float, lags: int) -> np.ndarray:
    """Approximate the Bernoulli null, retaining the same nondegenerate designs as in the test."""
    generator = np.random.default_rng(408)
    stats_batches = []
    batch_size = 512
    for first in range(0, DQ_DRAWS, batch_size):
        simulated_hits = generator.random((min(batch_size, DQ_DRAWS - first), n)) < alpha
        statistic, rank, diverse = _dq_statistic(simulated_hits, alpha, lags)
        stats_batches.append(statistic[(rank == lags + 1) & diverse])
    return np.concatenate(stats_batches)


def dynamic_quantile_test(exceeded: pd.Series, alpha: float, lags: int = DQ_LAGS) -> dict[str, float | int]:
    """Test hit calibration with an intercept and lagged exceedances, using a simulated p-value."""
    hits = np.asarray(exceeded, dtype=bool)
    effective_n = max(0, len(hits) - lags)
    out = {
        "dq_lags": lags, "dq_n_effective": effective_n, "dq_design_rank": np.nan,
        "dq_stat": np.nan, "dq_mc_draws": 0, "dq_p_mc": np.nan,
    }
    if effective_n <= lags + 1:
        return out
    statistic, rank, diverse = _dq_statistic(hits, alpha, lags)
    out["dq_design_rank"] = int(rank)
    if rank < lags + 1 or not diverse:
        return out
    null = _dq_null_statistics(len(hits), alpha, lags)
    out.update(dq_stat=float(statistic), dq_mc_draws=len(null))
    if len(null) >= 100:
        close_or_larger = (null > statistic) | np.isclose(null, statistic, rtol=1e-10, atol=1e-12)
        out["dq_p_mc"] = float((1 + close_or_larger.sum()) / (1 + len(null)))
    return out


def fz0_loss(frame: pd.DataFrame, alpha: float) -> pd.Series:
    """Patton–Ziegel–Chen FZ0 score with the upper tail of signed losses."""
    var = frame["var_pct"].to_numpy()
    es = frame["es_pct"].to_numpy()
    actual = frame["loss_pct"].to_numpy()
    valid = np.isfinite(actual) & np.isfinite(var) & np.isfinite(es) & (var > 0) & (es >= var)
    score = np.full(len(frame), np.nan)
    excess = np.maximum(actual[valid] - var[valid], 0)
    score[valid] = excess / (alpha * es[valid]) + var[valid] / es[valid] + np.log(es[valid]) - 1
    return pd.Series(score, index=frame.index, name="fz0_loss")


def es_calibration(frame: pd.DataFrame, alpha: float) -> dict[str, float | int]:
    """Test the mean of the ES component of the joint VaR–ES identification function."""
    tail_excess = (frame["loss_pct"] - frame["var_pct"]).clip(lower=0)
    moment = (frame["var_pct"] - frame["es_pct"] + tail_excess / alpha).to_numpy()
    n = len(moment)
    mean = float(moment.mean())
    maxlags = min(n - 1, int(4 * (n / 100) ** (2 / 9)))
    stats_out = {"es_moment_mean_pct": mean, "es_hac_lags": maxlags, "es_hac_se": np.nan, "es_moment_p": np.nan}
    if n > 1 and frame["exceeded"].any() and not np.allclose(moment, mean, rtol=1e-10, atol=1e-12):
        fit = OLS(moment, np.ones((n, 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}, use_t=False,
        )
        se = float(fit.bse[0])
        if not np.isfinite(se) or se <= 0:
            return stats_out
        stats_out.update(es_hac_se=se, es_moment_p=2 * stats.norm.sf(abs(mean / se)))
    return stats_out


def backtest_risk(
    data: SectorData,
    sample: ModelSample,
    fit: VolatilityFit,
    forecast: VolatilityForecast,
    risk: RiskForecast,
    config: AnalysisConfig,
    output: AnalysisOutput,
) -> RiskBacktest:
    """Test VaR coverage, exceedance dependence and joint VaR–ES calibration."""
    daily_risk = risk.daily_risk
    backtest_rows = []
    for (sector, model, confidence), frame in daily_risk.groupby(_RISK_GROUP_COLUMNS, sort=False):
        frame = frame.sort_values("date")
        alpha = (100 - confidence) / 100
        exceeded = frame["exceeded"]
        hit_rows = frame.loc[exceeded]
        loss_over_var = frame["loss_pct"] - frame["var_pct"]
        pinball_loss = (1 - alpha) * loss_over_var.clip(lower=0) - alpha * loss_over_var.clip(upper=0)
        joint_score = fz0_loss(frame, alpha)

        last_250_hits = np.nan
        basel_zone = pd.NA
        if confidence == 99 and len(frame) >= 250:
            last_250_hits = int(exceeded.tail(250).sum())
            basel_zone = "Red"
            if last_250_hits <= 4:
                basel_zone = "Green"
            elif last_250_hits <= 9:
                basel_zone = "Yellow"

        mean_observed_tail_loss = hit_rows["loss_pct"].mean()
        mean_forecast_es_on_hits = hit_rows["es_pct"].mean()
        backtest_rows.append({
            "sector": sector, "model": model, "confidence_pct": confidence,
            "test_start": frame["date"].iloc[0], "test_end": frame["date"].iloc[-1],
            **coverage_tests(exceeded, alpha),
            **dynamic_quantile_test(exceeded, alpha),
            **es_calibration(frame, alpha),
            "quantile_loss": pinball_loss.mean(),
            "fz0_n_valid": int(joint_score.notna().sum()),
            "fz0_loss": joint_score.mean(skipna=False),
            "basel_250_hits": last_250_hits, "basel_250_zone": basel_zone,
            "realized_tail_loss_pct": mean_observed_tail_loss,
            "predicted_es_on_hits_pct": mean_forecast_es_on_hits,
            "es_gap_on_hits_pct": mean_observed_tail_loss - mean_forecast_es_on_hits,
        })

    backtests = pd.DataFrame(backtest_rows)
    if not backtests.empty:
        output.save_table(backtests, "07_var_es_backtests.csv", index=False)
        report_cols = [
            "sector", "model", "n_test", "expected_hits", "hits", "hit_rate_pct",
            "uc_p", "exact_binomial_p", "ind_p", "cc_p", "dq_p_mc", "quantile_loss",
        ]
        for confidence in CONFIDENCE_LEVELS:
            print(f"VaR {confidence}% · Kupiec UC, Christoffersen IND/CC và DQ Monte Carlo")
            columns = report_cols + (["basel_250_hits", "basel_250_zone"] if confidence == 99 else [])
            output.display(backtests.loc[backtests["confidence_pct"].eq(confidence), columns].round(4))
        for confidence in CONFIDENCE_LEVELS:
            print(f"VaR–ES {confidence}% · FZ0, đối chiếu các phiên vượt VaR và kiểm định hiệu chỉnh trung bình")
            output.display(backtests.loc[backtests["confidence_pct"].eq(confidence), [
                "sector", "model", "hits", "realized_tail_loss_pct", "predicted_es_on_hits_pct", "es_gap_on_hits_pct",
                "es_moment_mean_pct", "es_hac_se", "es_moment_p", "fz0_n_valid", "fz0_loss",
            ]].round(4))

    if backtests.empty:
        print("Chưa có dự báo rủi ro ngoài mẫu để thực hiện backtest.")
        return RiskBacktest(backtests, pd.DataFrame(), pd.DataFrame())

    _print_backtest_summary(backtests, forecast.variance_scores, config)
    comparison = _compare_sectors(
        data, sample.model_samples, fit.vol_diagnostics, forecast.variance_scores, backtests, output
    )
    persistence_link = _summarize_persistence_link(risk.high_var_summary, fit.vol_diagnostics, output)
    return RiskBacktest(backtests, comparison, persistence_link)


def _print_backtest_summary(
    backtests: pd.DataFrame, variance_scores: pd.DataFrame, config: AnalysisConfig
) -> None:
    """Describe the best forecasts and the rejected backtests for each sector."""
    for sector, frame in backtests.query("confidence_pct == 99").groupby("sector", sort=False):
        print(f"{sector} · VaR 99% trên {int(frame['n_test'].iloc[0]):,} phiên ngoài mẫu")
        variance_frame = variance_scores.loc[variance_scores["sector"].eq(sector)].dropna(subset=["qlike"])
        if not variance_frame.empty:
            lowest_qlike = variance_frame.loc[variance_frame["qlike"].idxmin()]
            print(f"  QLIKE thấp nhất thuộc {lowest_qlike['model']} ({lowest_qlike['qlike']:.4f}).")
        lowest_quantile_loss = frame.loc[frame["quantile_loss"].idxmin()]
        print(
            f"  Quantile loss thấp nhất thuộc {lowest_quantile_loss['model']} "
            f"({lowest_quantile_loss['quantile_loss']:.4f}), với "
            f"{int(lowest_quantile_loss['hits'])} lần vượt so với "
            f"{lowest_quantile_loss['expected_hits']:.2f} lần kỳ vọng."
        )
        valid_fz0 = frame.dropna(subset=["fz0_loss"])
        if not valid_fz0.empty:
            best = valid_fz0.loc[valid_fz0["fz0_loss"].idxmin()]
            print(f"  FZ0 thấp nhất thuộc {best['model']} ({best['fz0_loss']:.4f}).")
        for p_column, label in [
            ("uc_p", "UC"), ("ind_p", "IND"), ("cc_p", "CC"), ("dq_p_mc", "DQ"), ("es_moment_p", "ES HAC"),
        ]:
            rejected = frame.loc[frame[p_column].lt(config.significance), "model"].tolist()
            undefined = frame.loc[frame[p_column].isna(), "model"].tolist()
            if rejected:
                print(f"  {label} bác bỏ ở mức {config.significance:.0%} với {', '.join(rejected)}.")
            elif frame[p_column].notna().any():
                print(f"  {label} chưa bác bỏ mô hình nào có p-value tính được ở mức {config.significance:.0%}.")
            if undefined:
                print(f"  {label} chưa tính được cho {', '.join(undefined)}.")

    modeled_sectors = backtests["sector"].unique().tolist()
    pending = [sector for sector in config.targets if sector not in modeled_sectors]
    if pending:
        print(f"\nCòn thiếu kết quả backtest cho {', '.join(pending)}.")


def _compare_sectors(
    data: SectorData,
    model_samples: pd.DataFrame,
    vol_diagnostics: pd.DataFrame,
    variance_scores: pd.DataFrame,
    backtests: pd.DataFrame,
    output: AnalysisOutput,
) -> pd.DataFrame:
    """Compare realized volatility and model performance across the target sectors."""
    if variance_scores.empty:
        return pd.DataFrame()

    common = data.returns[list(data.active_sectors)].dropna()
    comparison = (common.std() * np.sqrt(252)).rename("Annualized vol (%)").to_frame()
    comparison.index.name = "sector"

    common_spec = "GARCH-t"
    persistence = vol_diagnostics.loc[
        vol_diagnostics["model"].eq(common_spec)
    ].set_index("sector")["persistence"]
    comparison["GARCH-t persistence"] = persistence
    best_variance = variance_scores.sort_values(["qlike", "mse"]).drop_duplicates("sector").set_index("sector")
    comparison["Lowest QLIKE"] = best_variance["model"]

    bt99 = backtests.loc[backtests["confidence_pct"].eq(99)]
    best_tail = bt99.sort_values("quantile_loss").drop_duplicates("sector").set_index("sector")
    comparison["Lowest 99% quantile loss"] = best_tail["model"]
    comparison["99% VaR hits"] = best_tail["hits"].astype("Int64")
    comparison["99% expected hits"] = best_tail["expected_hits"]
    comparison["99% CC p-value"] = best_tail["cc_p"]
    valid_fz0 = bt99.dropna(subset=["fz0_loss"])
    best_joint = valid_fz0.sort_values("fz0_loss").drop_duplicates("sector").set_index("sector")
    comparison["Lowest 99% FZ0"] = best_joint["model"]
    output.save_table(comparison, "07_sector_comparison.csv")
    output.display(comparison.round(4))

    if len(common) >= 2 and len(data.active_sectors) >= 2:
        sector_vol = comparison["Annualized vol (%)"]
        low, high = sector_vol.idxmin(), sector_vol.idxmax()
        print(
            f"Trên {len(common):,} phiên lợi suất chung, {high} có độ biến động quy năm cao nhất "
            f"({sector_vol[high]:.2f}%) và {low} thấp nhất ({sector_vol[low]:.2f}%)."
        )
    if (
        len(model_samples) >= 2
        and model_samples[["train_n", "train_start", "train_end"]].drop_duplicates().shape[0] == 1
        and persistence.notna().sum() >= 2
        and data.returns.loc[model_samples["train_start"].iloc[0]:model_samples["train_end"].iloc[0],
                        persistence.dropna().index].notna().all().all()
    ):
        high, low = persistence.idxmax(), persistence.idxmin()
        print(
            f"Trong mẫu huấn luyện chung, độ dai dẳng GARCH-t cao nhất ở {high} "
            f"({persistence[high]:.4f}) và thấp nhất ở {low} ({persistence[low]:.4f})."
        )
    return comparison


def _summarize_persistence_link(
    spell_summary: pd.DataFrame, vol_diagnostics: pd.DataFrame, output: AnalysisOutput,
) -> pd.DataFrame:
    if spell_summary.empty:
        return pd.DataFrame()
    persistence = vol_diagnostics.loc[vol_diagnostics["model"].eq(SPELL_MODEL), ["sector", "persistence"]]
    link = spell_summary.merge(persistence, on="sector", how="left", validate="one_to_one")
    link["persistence_rank_high_to_low"] = link["persistence"].rank(method="min", ascending=False)
    durations = ["mean_observed_duration_sessions", "max_observed_duration_sessions"]
    for measure in durations:
        link[f"rank_{measure}_high_to_low"] = link[measure].rank(method="min", ascending=False)
    output.save_table(link, "07_persistence_and_high_var_spells.csv", index=False)
    output.display(link[[
        "sector", "persistence", "n_high_var_sessions", "n_spells", *durations, "median_observed_duration_sessions",
    ]].round(4))

    association_rows = []
    for measure in durations:
        paired = link[["persistence", measure]].dropna()
        rho = np.nan
        if len(paired) >= 3 and paired.nunique().gt(1).all():
            rho = float(stats.spearmanr(paired["persistence"], paired[measure]).statistic)
        association_rows.append({"measure": measure, "n_sectors": len(paired), "spearman_rho_descriptive": rho})
    association = pd.DataFrame(association_rows)
    output.save_table(association, "07_persistence_high_var_association.csv", index=False)
    output.display(association.round(4))
    return link
