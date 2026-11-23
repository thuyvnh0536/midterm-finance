"""Output 5: VAR predictability and conditional correlations of sector indices."""

from dataclasses import dataclass
from itertools import combinations, permutations
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.api import VAR

from analysis_config import AnalysisConfig, AnalysisOutput
from data_processing import SectorData
from time_series import ModelSample, fit_volatility


IRF_HORIZON = 10
HAC_LAGS = 5
DCC_LIMIT = 0.999
REGIMES = (
    ("Pre-COVID", "2015-01-01", "2019-12-31"),
    ("Early COVID-19", "2020-02-01", "2020-06-30"),
    ("Late-2022 financial stress", "2022-10-01", "2022-12-31"),
)


@dataclass(frozen=True)
class SectorInteractions:
    """Key estimates and diagnostics for the synchronized training sample."""

    var_summary: pd.DataFrame
    granger: pd.DataFrame
    irf: pd.DataFrame
    fevd: pd.DataFrame
    dcc_fit: pd.DataFrame
    dcc_margins: pd.DataFrame
    correlations: pd.DataFrame
    pair_summary: pd.DataFrame
    regime_summary: pd.DataFrame


def _adjust_holm(table: pd.DataFrame, p_column: str) -> np.ndarray:
    values = table[p_column].to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if valid.any():
        adjusted[valid] = multipletests(values[valid], method="holm")[1]
    return adjusted


def _hac_wald(y: np.ndarray, design: np.ndarray, columns: list[int]) -> tuple[float, float]:
    """Test that the chosen coefficients are jointly zero with Bartlett HAC."""
    result = OLS(y, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    restrictions = np.eye(design.shape[1])[columns]
    test = result.wald_test(restrictions, scalar=True, use_f=False)
    return float(test.statistic), float(test.pvalue)


def _var_analysis(daily: pd.DataFrame, config: AnalysisConfig, output: AnalysisOutput):
    """Fit and report a training-sample VAR on synchronized returns."""
    sectors = list(daily.columns)
    max_lags = min(10, (len(daily) - len(sectors) - 1) // (len(sectors) + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValueWarning)
        # Trading dates are irregular in calendar time but consecutive in session order.
        model = VAR(daily)
    selection = model.select_order(maxlags=max_lags, trend="c")
    order_table = pd.DataFrame(selection.ics).rename_axis("lag")
    # Also report VAR(0). If BIC chooses it, condition lagged diagnostics on exploratory VAR(1).
    bic_order = selection.selected_orders["bic"]
    order = max(1, bic_order)
    fitted = model.fit(order, trend="c")
    is_stable = bool(np.min(np.abs(fitted.roots)) > 1.0)
    whiteness = fitted.test_whiteness(max(12, order + 2), adjusted=True)
    summary = pd.DataFrame([{
        "aligned_start": daily.index.min(), "aligned_end": daily.index.max(),
        "aligned_sessions": len(daily), "fit_observations": fitted.nobs,
        "lag_bic": bic_order, "lag_aic": selection.selected_orders["aic"],
        "lag_reported": order, "estimated_system_stable": is_stable,
        "companion_spectral_radius": 1.0 / np.min(np.abs(fitted.roots)),
        "portmanteau_lag": max(12, order + 2), "portmanteau_p_classical": float(whiteness.pvalue),
    }])
    output.save_table(order_table, "05_var_lag_selection.csv")
    output.save_table(summary, "05_var_summary.csv", index=False)
    output.save_table(fitted.params, "05_var_coefficients.csv")
    output.display(order_table, summary)

    granger_rows = []
    for cause, effect in permutations(sectors, 2):
        lag_columns = [fitted.params.index.get_loc(f"L{lag}.{cause}") for lag in range(1, order + 1)]
        test = fitted.test_causality(effect, cause, kind="f")
        robust_stat, robust_p = _hac_wald(
            fitted.endog[order:, sectors.index(effect)], fitted.endog_lagged, lag_columns,
        )
        granger_rows.append({
            "predictor": cause, "response": effect, "lag": order,
            "classical_F": float(test.test_statistic), "classical_p": float(test.pvalue),
            "HAC_Wald": robust_stat, "HAC_p": robust_p,
        })
    granger = pd.DataFrame(granger_rows)
    granger["HAC_Holm_p"] = _adjust_holm(granger, "HAC_p")
    granger["reject_at_configured_level"] = granger["HAC_Holm_p"].lt(config.significance)
    granger = granger.sort_values("HAC_Holm_p", ignore_index=True)
    output.save_table(granger, "05_var_granger_pairs.csv", index=False)
    output.display(granger)
    if bic_order == 0:
        print("BIC chọn VAR(0). Các kiểm định động dưới VAR(1) mang tính thăm dò.")
    significant = granger.loc[granger["reject_at_configured_level"]]
    print(f"Granger HAC–Holm: {len(significant)}/{len(granger)} hướng có p < {config.significance:g}.")
    if not is_stable:
        print("VAR thiếu ổn định; IRF và FEVD vẫn được xuất để chẩn đoán.")

    # Impulse response columns are shocks, and rows are responding variables.
    irf_obj = fitted.irf(IRF_HORIZON)
    irf = pd.DataFrame([
        {"lag": horizon, "shock_sector": shock, "response_sector": effect,
         "reduced_form_return_pp": irf_obj.irfs[horizon, j, i],
         "orthogonalized_return_pp": irf_obj.orth_irfs[horizon, j, i]}
        for horizon in range(IRF_HORIZON + 1)
        for i, shock in enumerate(sectors)
        for j, effect in enumerate(sectors)
    ])
    output.save_table(irf, "05_var_impulse_responses.csv", index=False)
    fig, axes = plt.subplots(
        len(sectors), len(sectors), figsize=(2.45 * len(sectors), 1.6 * len(sectors)),
        sharex=True, layout="constrained", squeeze=False,
    )
    for i, effect in enumerate(sectors):
        for j, shock in enumerate(sectors):
            ax = axes[i, j]
            ax.axhline(0, color="#64748b", lw=0.6)
            ax.plot(range(IRF_HORIZON + 1), irf_obj.orth_irfs[:, i, j], color="#0f6773", lw=1.15)
            if i == 0:
                ax.set_title(shock, fontsize=8)
            if j == 0:
                ax.set_ylabel(effect, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.supxlabel("Trading sessions after the innovation · columns are shocks, rows are responses")
    fig.suptitle("Figure 5.1. VAR Cholesky impulse responses (log-return percentage points)")
    output.show_figure(fig, "05_var_impulse_responses.png")

    # Ordering sensitivity uses the same fitted lag and sample with the reverse Cholesky order.
    primary = fitted.fevd(IRF_HORIZON).decomp
    reverse = fitted.reorder(sectors[::-1]).fevd(IRF_HORIZON).decomp[::-1, :, ::-1]
    fevd = pd.DataFrame([
        {"horizon": horizon, "response_sector": effect, "shock_sector": shock,
         "primary_share_pct": 100 * primary[j, horizon - 1, i],
         "reverse_order_share_pct": 100 * reverse[j, horizon - 1, i],
         "reverse_minus_primary_pp": 100 * (reverse - primary)[j, horizon - 1, i]}
        for horizon in range(1, IRF_HORIZON + 1)
        for j, effect in enumerate(sectors)
        for i, shock in enumerate(sectors)
    ])
    output.save_table(fevd, "05_var_fevd.csv", index=False)
    last = fevd.loc[fevd["horizon"].eq(IRF_HORIZON)]
    fevd_matrix = last.pivot(index="response_sector", columns="shock_sector", values="primary_share_pct")
    output.display(fevd_matrix.reindex(index=sectors, columns=sectors))
    print(f"FEVD chân trời {IRF_HORIZON} phiên: chênh lệch ô lớn nhất khi đảo thứ tự "
          f"Cholesky = {last['reverse_minus_primary_pp'].abs().max():.1f} điểm phần trăm.")
    return summary, granger, irf, fevd


def _normalize_correlation(q: np.ndarray) -> np.ndarray:
    standard_deviation = np.sqrt(np.diagonal(q, axis1=-2, axis2=-1))
    return q / (standard_deviation[..., :, None] * standard_deviation[..., None, :])


def _dcc_correlations(alpha: float, beta: float, target: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    # shocks[0] is 0; all later shocks use z at the previous trading session.
    deviations = lfilter([alpha], [1.0, -beta], shocks, axis=0)
    return _normalize_correlation(target + deviations)


def _gaussian_correlation_loglik(z: np.ndarray, correlation: np.ndarray) -> float:
    cholesky = np.linalg.cholesky(correlation)
    whitened = np.linalg.solve(cholesky, z[..., None])[..., 0]
    log_det = 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1)
    return float(-0.5 * np.sum(log_det + np.square(whitened).sum(axis=-1) - np.square(z).sum(axis=-1)))


def _fit_dcc(z: np.ndarray):
    target = z.T @ z / len(z)
    if np.min(np.linalg.eigvalsh(target)) <= 1e-10:
        raise ValueError("Standardized-residual covariance is singular or poorly conditioned.")
    shocks = np.zeros((len(z), z.shape[1], z.shape[1]))
    shocks[1:] = z[:-1, :, None] * z[:-1, None, :] - target
    ccc = _normalize_correlation(target)
    ccc_obj = _gaussian_correlation_loglik(z, ccc)

    def objective(parameters):
        alpha, beta = parameters
        if not np.isfinite(parameters).all() or alpha + beta >= 1:
            return 1e10
        try:
            correlation = _dcc_correlations(alpha, beta, target, shocks)
            return -_gaussian_correlation_loglik(z, correlation) / len(z)
        except (np.linalg.LinAlgError, FloatingPointError):
            return 1e10

    starts = ([0.03, 0.93], [0.08, 0.65], [0.01, 0.98])
    estimates = [
        minimize(
            objective, start, method="SLSQP", bounds=((0.0, DCC_LIMIT), (0.0, DCC_LIMIT)),
            constraints=({"type": "ineq", "fun": lambda x: DCC_LIMIT - x.sum()},),
            options={"ftol": 1e-10, "maxiter": 120},
        )
        for start in starts
    ]
    converged = [
        estimate for estimate in estimates
        if estimate.success and np.isfinite(estimate.fun) and estimate.x.sum() <= DCC_LIMIT + 1e-8
    ]
    if not converged:
        raise ValueError("DCC optimizer did not converge from its starting values.")
    best = min(converged, key=lambda estimate: estimate.fun)
    gain = -best.fun * len(z) - ccc_obj
    if gain <= 1e-7:
        alpha, beta = 0.0, 0.0
    else:
        alpha, beta = best.x
    return alpha, beta, ccc, ccc_obj, _dcc_correlations(alpha, beta, target, shocks)


def _pair_diagnostics(z: np.ndarray, ccc: np.ndarray, dcc: np.ndarray, sectors: list[str]):
    """Screen for lag predictability of pair products under fitted correlations."""
    columns = [
        "sector_a", "sector_b", "CCC_constant_corr", "DCC_mean_corr", "DCC_sd_corr",
        "DCC_p05_corr", "DCC_p95_corr", "CCC_pair_HAC_p", "DCC_pair_HAC_p",
    ]
    rows = []
    for i, j in combinations(range(len(sectors)), 2):
        product = z[:, i] * z[:, j]
        design = np.column_stack(
            [np.ones(len(z) - HAC_LAGS)]
            + [product[HAC_LAGS - lag:-lag] for lag in range(1, HAC_LAGS + 1)]
        )
        tested = list(range(1, HAC_LAGS + 1))
        _, ccc_p = _hac_wald(product[HAC_LAGS:] - ccc[i, j], design, tested)
        path = dcc[:, i, j]
        _, dcc_p = _hac_wald(product[HAC_LAGS:] - path[HAC_LAGS:], design, tested)
        rows.append([
            sectors[i], sectors[j], ccc[i, j], path.mean(), path.std(ddof=1),
            np.quantile(path, 0.05), np.quantile(path, 0.95), ccc_p, dcc_p,
        ])
    table = pd.DataFrame(rows, columns=columns)
    table["CCC_pair_Holm_p"] = _adjust_holm(table, "CCC_pair_HAC_p")
    table["DCC_pair_Holm_p"] = _adjust_holm(table, "DCC_pair_HAC_p")
    return table


def _dcc_analysis(sample: ModelSample, sectors: list[str], config: AnalysisConfig, output: AnalysisOutput):
    innovations = pd.concat({sector: sample.mean_residuals[sector] for sector in sectors}, axis=1).dropna()
    if len(innovations) < 100:
        raise ValueError("DCC requires at least 100 synchronized training residuals.")
    margin_rows, fitted_volatility = [], {}
    for sector in sectors:
        params, h, _ = fit_volatility(innovations[sector], "GARCH-t")
        fitted_volatility[sector] = np.sqrt(h)
        margin_rows.append({
            "sector": sector, "observations": len(h), "alpha": params["alpha[1]"],
            "beta": params["beta[1]"], "persistence": params["alpha[1]"] + params["beta[1]"],
            "student_t_df": params["nu"], "mean_daily_vol_pct": np.sqrt(h).mean(),
        })
    margins = pd.DataFrame(margin_rows)
    volatility = pd.DataFrame(fitted_volatility)
    z = (innovations / volatility).to_numpy()
    alpha, beta, ccc, ccc_ll, dcc = _fit_dcc(z)
    dcc_ll = _gaussian_correlation_loglik(z, dcc)
    comparison = pd.DataFrame([
        {"model": "CCC-GARCH-t", "observations": len(z), "alpha_dcc": 0.0, "beta_dcc": np.nan,
         "alpha_plus_beta": np.nan, "gaussian_correlation_loglik": ccc_ll},
        {"model": "DCC-GARCH-t", "observations": len(z), "alpha_dcc": alpha, "beta_dcc": beta,
         "alpha_plus_beta": alpha + beta if alpha > 0 else np.nan, "gaussian_correlation_loglik": dcc_ll},
    ])
    comparison["fit_start"] = innovations.index.min()
    comparison["fit_end"] = innovations.index.max()
    comparison["gain_from_CCC"] = comparison["gaussian_correlation_loglik"] - ccc_ll
    comparison["at_persistence_constraint"] = np.isclose(comparison["alpha_plus_beta"], DCC_LIMIT)
    output.save_table(margins, "05_dcc_garch_margins.csv", index=False)
    output.save_table(comparison, "05_ccc_dcc_fit.csv", index=False)
    output.save_table(pd.DataFrame(ccc, index=sectors, columns=sectors), "05_ccc_matrix.csv")
    pair_summary = _pair_diagnostics(z, ccc, dcc, sectors)
    output.save_table(pair_summary, "05_ccc_dcc_pair_summary.csv", index=False)

    pairs = list(combinations(range(len(sectors)), 2))
    labels = [f"{sectors[i]} / {sectors[j]}" for i, j in pairs]
    weights = volatility.to_numpy() / len(sectors)
    daily = pd.DataFrame(
        {label: dcc[:, i, j] for (i, j), label in zip(pairs, labels)}, index=innovations.index,
    ).rename_axis("Date")
    daily["Average pairwise correlation"] = daily.mean(axis=1)
    daily["Equal-weight indicative DCC volatility (%)"] = np.sqrt(np.einsum("ti,tij,tj->t", weights, dcc, weights))
    daily["Equal-weight indicative CCC volatility (%)"] = np.sqrt(np.einsum("ti,ij,tj->t", weights, ccc, weights))
    sectors_name = {sector.split(" L", 1)[0]: sector for sector in sectors}
    readout_labels = [
        label for label, (i, j) in zip(labels, pairs)
        if {sectors[i], sectors[j]} == {sectors_name.get("Banks"), sectors_name.get("Real Estate")}
    ]
    technology = sectors_name.get("Technology")
    technology_labels = [
        label for label, (i, j) in zip(labels, pairs) if technology and technology in (sectors[i], sectors[j])
    ]
    if technology_labels:
        daily["Technology average pairwise correlation"] = daily[technology_labels].mean(axis=1)
        readout_labels.append("Technology average pairwise correlation")
    shown = ["Average pairwise correlation", *readout_labels]
    regimes = []
    for label, start, end in REGIMES:
        window = daily.loc[start:end]
        if window.empty:
            continue
        regimes.append({
            "regime": label, "first_session": window.index.min(), "last_session": window.index.max(),
            "sessions": len(window), **window[shown].mean().to_dict(),
        })
    regime_summary = pd.DataFrame(regimes)
    output.save_table(regime_summary, "05_dcc_regime_summary.csv", index=False)
    output.display(comparison, pair_summary, regime_summary)
    significant_pairs = pair_summary["CCC_pair_Holm_p"].lt(config.significance).sum()
    remaining_pairs = pair_summary["DCC_pair_Holm_p"].lt(config.significance).sum()
    print(f"Kiểm tra dự báo sai lệch tích phần dư của {len(pair_summary)} cặp "
          f"(HAC–Holm): {significant_pairs} cặp dưới CCC và {remaining_pairs} cặp dưới DCC "
          f"có p < {config.significance:g}. Đây là chẩn đoán trong mẫu.")
    if alpha > 0:
        print(f"DCC tăng Gaussian correlation log-likelihood trong mẫu thêm {dcc_ll - ccc_ll:.1f}.")
    else:
        print("Tối ưu DCC quy về đường tương quan cố định trong mẫu này.")
    print("DCC mô tả đồng biến có điều kiện. Các phương trình phương sai riêng được giữ nguyên, "
          "nên kết quả không phân rã độ dai dẳng thành phần riêng và tác động lan truyền.")

    fig, (ax_corr, ax_vol) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, layout="constrained")
    for column in shown:
        ax_corr.plot(daily.index, daily[column], lw=0.95, label=column)
    ax_corr.axhline(np.mean([ccc[i, j] for i, j in pairs]), color="#64748b", lw=0.8, ls="--",
                   label="CCC average")
    ax_corr.set(ylabel="Conditional correlation", title="Conditional correlations from DCC and CCC")
    ax_corr.legend(loc="best", fontsize=8)
    for _, start, end in REGIMES[1:]:
        ax_corr.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#94a3b8", alpha=0.17)
    for column in daily.columns:
        if "volatility" in column:
            ax_vol.plot(daily.index, daily[column], lw=0.9, label=column.replace("Equal-weight indicative ", ""))
    ax_vol.set(ylabel="Daily volatility (%)", title="Model-implied volatility for equal sector weights")
    ax_vol.legend(loc="best")
    fig.suptitle("Figure 5.2. Dynamic correlations and their effect on joint volatility")
    output.show_figure(fig, "05_ccc_dcc_correlations.png")
    output.save_table(daily, "05_dcc_daily_correlations.csv")
    return comparison, margins, daily, pair_summary, regime_summary


def analyze_interactions(data: SectorData, sample: ModelSample, config: AnalysisConfig,
                         output: AnalysisOutput) -> SectorInteractions:
    """Compare short-run mean predictability and conditional sector comovement."""
    sectors = list(sample.series_by_sector)
    if len(sectors) < 2:
        raise ValueError("Output 5 requires at least two model-eligible sectors.")
    train_end = min(series.index[sample.split_positions[sector] - 1]
                    for sector, series in sample.series_by_sector.items())
    common = data.returns.loc[:train_end, sectors].dropna()
    if len(common) < 100:
        raise ValueError("Output 5 requires at least 100 common training sessions.")
    var_summary, granger, irf, fevd = _var_analysis(common, config, output)
    dcc_fit, margins, correlations, pairs, regimes = _dcc_analysis(sample, sectors, config, output)
    return SectorInteractions(var_summary, granger, irf, fevd, dcc_fit, margins, correlations, pairs, regimes)
