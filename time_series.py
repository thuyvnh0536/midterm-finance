"""Mean filtering, volatility models, and rolling one-session forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings

from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning as ArchConvergenceWarning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf

if TYPE_CHECKING:
    from analysis_config import AnalysisConfig, AnalysisOutput
    from data_processing import SectorData


@dataclass
class ModelSample:
    """Synchronized return series, chronological split, and training innovations."""

    series_by_sector: dict[str, pd.Series]
    split_positions: dict[str, int]
    model_samples: pd.DataFrame
    orders: dict[str, tuple[int, int]] = field(default_factory=dict)
    mean_residuals: dict[str, pd.Series] = field(default_factory=dict)
    mean_parameters: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class VolatilityFit:
    """Training estimates and standardized innovations."""

    vol_diagnostics: pd.DataFrame
    standardized_residuals: dict[tuple[str, str], pd.Series] = field(default_factory=dict)
    parameters: dict[tuple[str, str], pd.Series] = field(default_factory=dict)


@dataclass
class VolatilityForecast:
    """Holdout predictions and their variance scores."""

    forecasts: dict[tuple[str, str], pd.DataFrame]
    variance_scores: pd.DataFrame
    refit_log: pd.DataFrame = field(default_factory=pd.DataFrame)


EWMA_LAMBDA = 0.94
VOL_SPECS = {
    "EWMA-n": None,
    "GARCH-n": dict(vol="GARCH", p=1, o=0, q=1, dist="normal"),
    "GARCH-t": dict(vol="GARCH", p=1, o=0, q=1, dist="t"),
    "GJR-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
}
FIT_ERRORS = (ValueError, RuntimeError, OverflowError, FloatingPointError, np.linalg.LinAlgError)


def _arma_model(y, order):
    # NumPy input makes lag order follow observed sessions, not a calendar-day frequency.
    p, q = order
    return ARIMA(y.to_numpy(), order=(p, 0, q), trend="c")


def _estimate_arma(y, order):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SMConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning, message="Non-stationary starting.*")
        warnings.filterwarnings("ignore", category=UserWarning, message="Non-invertible starting.*")
        result = _arma_model(y, order).fit(method_kwargs={"maxiter": 200})
    if not result.mle_retvals["converged"] or not np.isfinite(result.params).all():
        raise ValueError("ARMA likelihood did not converge.")
    return result


def _arma_residuals(fit, index, order):
    p, q = order
    burn = max(p, q)
    return pd.Series(fit.resid[burn:], index=index[burn:], name="mean_innovation")


def _lb(values, model_df=0):
    return acorr_ljungbox(values, lags=[10], model_df=model_df).iloc[0]


def fit_mean_models(
    data: SectorData, config: AnalysisConfig, output: AnalysisOutput
) -> ModelSample:
    """Choose a parsimonious conditional mean and filter returns on the training sample."""
    panel = data.returns[list(data.active_sectors)].dropna()
    if data.active_sectors and int(len(panel) * (1 - config.test_fraction)) < 60:
        raise ValueError("The synchronized return series is too short for the selected train/test split.")
    series_by_sector = {sector: panel[sector].sort_index() for sector in data.active_sectors}
    split_positions = {
        sector: len(y) - int(np.ceil(config.test_fraction * len(y)))
        for sector, y in series_by_sector.items()
    }
    sample_rows, grid_rows, selection_rows, parameter_rows = [], [], [], []
    orders, mean_residuals, mean_parameters = {}, {}, {}
    for sector, y in series_by_sector.items():
        cut = split_positions[sector]
        train, test = y.iloc[:cut], y.iloc[cut:]
        sample_rows.append({
            "sector": sector, "train_n": len(train), "train_start": train.index[0],
            "train_end": train.index[-1], "test_n": len(test), "test_start": test.index[0],
            "test_end": test.index[-1], "rolling_window_n": min(config.max_window, len(train)),
        })
        raw_lb = float(_lb(train)["lb_pvalue"])
        raw_acf = acf(train, nlags=20, fft=True)[1:]
        significant_lags = np.flatnonzero(abs(raw_acf) > 1.96 / np.sqrt(len(train))) + 1
        candidate_orders = (
            [(p, q) for p in range(3) for q in range(3)]
            if raw_lb < config.significance else [(0, 0)]
        )
        fits = []
        for order in candidate_orders:
            p, q = order
            row = {"sector": sector, "p": p, "q": q, "AIC": np.nan, "BIC": np.nan}
            try:
                model = _estimate_arma(train, order)
                residuals = _arma_residuals(model, train.index, order)
                lb = _lb(residuals, p + q)
                row.update(AIC=model.aic, BIC=model.bic, residual_lb10_p=lb["lb_pvalue"], status="fit")
                fits.append((order, model, residuals, lb))
            except FIT_ERRORS as exc:
                row["status"] = str(exc).splitlines()[0]
            grid_rows.append(row)
        if not fits:
            raise RuntimeError(f"No converged mean model for {sector}.")

        best = min(fits, key=lambda candidate: candidate[1].bic)
        acceptable = [
            candidate for candidate in fits
            if candidate[1].bic <= best[1].bic + 2 and candidate[3]["lb_pvalue"] >= config.significance
        ]
        # A 2-point BIC neighborhood permits a simpler model with white residuals.
        chosen = min(acceptable, key=lambda item: (sum(item[0]), item[1].bic)) if acceptable else best
        order, model, residuals, lb = chosen
        p, q = order
        orders[sector] = order
        mean_residuals[sector] = residuals
        mean_parameters[sector] = model.params
        arch_test = het_arch(residuals, nlags=10, ddof=p + q, result_object=True)
        arch_lm, arch_p = arch_test.lm, arch_test.lmpval
        squared_lb = _lb(residuals.pow(2))["lb_pvalue"]
        model_name = f"ARMA({p},{q})"
        selection_rows.append({
            "sector": sector, "order": model_name, "AIC": model.aic, "BIC": model.bic,
            "raw_LB10_p": raw_lb,
            "raw_ACF_significant_lags_1_to_20": ", ".join(map(str, significant_lags)),
            "residual_LB10_stat": lb["lb_stat"], "residual_LB10_p": lb["lb_pvalue"],
            "residual_squared_LB10_p": squared_lb,
            "residual_ARCH_LM10_stat": arch_lm, "residual_ARCH_LM10_p": arch_p,
            "variance_modelling_evidence": (
                "ARCH detected" if arch_p < config.significance else "ARCH not detected"
            ),
            "selection": (
                "Constant mean: raw Ljung–Box does not reject" if len(candidate_orders) == 1
                else "Parsimonious fit within 2 BIC points with no residual rejection" if acceptable
                else "Minimum BIC; no near-BIC fit clears the residual Ljung–Box"
            ),
        })
        for name, value, se, p_value in zip(model.param_names, model.params, model.bse, model.pvalues):
            parameter_rows.append({
                "sector": sector, "model": model_name, "parameter": name, "estimate": value,
                "std_error": se, "p_value": p_value,
            })
        print(
            f"{sector} · {model_name}: Ljung–Box p={lb['lb_pvalue']:.3g} trên phần dư, "
            f"ARCH-LM p={arch_p:.3g}. "
            + (
                "Có bằng chứng về ARCH, tiếp tục mô hình hóa phương sai có điều kiện."
                if arch_p < config.significance
                else "Chưa phát hiện ARCH ở mức đã chọn, giữ mô hình phương sai để đối chiếu."
            )
        )

    model_samples = pd.DataFrame(sample_rows, columns=[
        "sector", "train_n", "train_start", "train_end", "test_n", "test_start",
        "test_end", "rolling_window_n",
    ])
    output.save_table(model_samples, "03_model_samples.csv", index=False)
    output.save_table(pd.DataFrame(grid_rows), "03_arma_grid.csv", index=False)
    output.save_table(pd.DataFrame(parameter_rows), "03_arma_parameters.csv", index=False)
    if selection_rows:
        selection = pd.DataFrame(selection_rows)
        output.save_table(selection, "03_arma_selection.csv", index=False)
        output.save_table(pd.DataFrame(mean_residuals), "03_training_mean_innovations.csv")
        output.display(model_samples, selection)
        _show_mean_acf(series_by_sector, split_positions, mean_residuals, output)
    return ModelSample(series_by_sector, split_positions, model_samples, orders, mean_residuals, mean_parameters)


def _show_mean_acf(returns, split_positions, residuals, output):
    fig, axes = plt.subplots(
        len(residuals), 4, figsize=(14, 2.55 * len(residuals)), squeeze=False,
        layout="constrained",
    )
    for (sector, e), row in zip(residuals.items(), axes):
        train = returns[sector].iloc[:split_positions[sector]]
        plot_acf(train, lags=20, zero=False, ax=row[0], title="Return ACF")
        plot_pacf(train, lags=20, zero=False, ax=row[1], title="Return PACF", method="ywm")
        plot_acf(e, lags=20, zero=False, ax=row[2], title="Innovation ACF")
        plot_acf(e.pow(2), lags=20, zero=False, ax=row[3], title="Squared innovation ACF")
        row[0].set_ylabel(sector)
    fig.suptitle("Output 3 · Training sample mean selection and diagnostics", fontweight="bold")
    output.show_figure(fig, "03_mean_and_residual_acf.png")


def fit_volatility(y, model):
    """Fit conditional variance to zero-mean innovations already filtered by ARMA."""
    if model == "EWMA-n":
        squared_errors = y.pow(2)
        seed = float(squared_errors.iloc[:20].mean())
        if not np.isfinite(seed) or seed <= 0:
            raise ValueError("The EWMA initialization variance is nonpositive.")
        seeded_errors = np.r_[seed, squared_errors.to_numpy()[:-1]]
        conditional_variance = pd.Series(seeded_errors, index=y.index).ewm(
            alpha=1 - EWMA_LAMBDA, adjust=False
        ).mean()
        return pd.Series({"lambda": EWMA_LAMBDA}), conditional_variance, None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ArchConvergenceWarning)
        fit = arch_model(y, mean="Zero", rescale=False, **VOL_SPECS[model]).fit(
            disp="off", update_freq=0, cov_type="robust", options={"maxiter": 1500}
        )
    if fit.convergence_flag != 0 or not np.isfinite(fit.params).all():
        raise ValueError(f"Optimizer flag {fit.convergence_flag}: {fit.optimization_result.message}")
    conditional_variance = fit.conditional_volatility.pow(2)
    _check_variances(conditional_variance)
    return fit.params, conditional_variance, fit


def _check_variances(variance):
    if not np.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError("The estimated conditional variance is nonpositive or nonfinite.")


def _persistence(model, params):
    if model == "EWMA-n":
        return 1.0
    if model == "EGARCH-t":
        return params["beta[1]"]
    gamma = params.get("gamma[1]", 0.0)
    return params["alpha[1]"] + params["beta[1]"] + 0.5 * gamma


def _parameter_admissibility(model, params, persistence):
    t_df_valid = params.get("nu", np.inf) > 2
    if model == "EWMA-n":
        return 0 < params["lambda"] < 1, False
    if model == "EGARCH-t":
        return t_df_valid, abs(persistence) < 1 - 1e-8
    admissible = bool(
        params["omega"] > 0 and params["alpha[1]"] >= -1e-8
        and params["beta[1]"] >= -1e-8
        and params["alpha[1]"] + params.get("gamma[1]", 0.0) >= -1e-8
        and t_df_valid
    )
    return admissible, 0 <= persistence < 1 - 1e-8


def _fit_pvalues(fit, params):
    if fit is None:
        return pd.Series(np.nan, index=params.index), pd.Series(np.nan, index=params.index)
    try:
        return fit.std_err, fit.pvalues
    except np.linalg.LinAlgError:
        # Diagnostics and fitted paths remain valid if robust covariance inversion fails.
        return pd.Series(np.nan, index=params.index), pd.Series(np.nan, index=params.index)


def volatility_diagnostics(sector, model, params, h, z, fit, pvalues):
    persistence = _persistence(model, params)
    admissible, mean_reverting = _parameter_admissibility(model, params, persistence)
    half_life = (
        np.log(0.5) / np.log(persistence)
        if mean_reverting and 0 < persistence < 1 else np.nan
    )
    arch_test = het_arch(z, nlags=10, result_object=True)
    return {
        "sector": sector, "model": model, "status": "fit",
        "AIC": fit.aic if fit is not None else np.nan,
        "BIC": fit.bic if fit is not None else np.nan,
        "mean_conditional_vol_pct": np.sqrt(h).iloc[20:].mean(),
        "omega": params.get("omega", np.nan), "alpha": params.get("alpha[1]", np.nan),
        "alpha_p": pvalues.get("alpha[1]", np.nan), "beta": params.get("beta[1]", np.nan),
        "beta_p": pvalues.get("beta[1]", np.nan), "persistence": persistence,
        "shock_half_life_sessions": half_life,
        "gamma": params.get("gamma[1]", np.nan), "gamma_p": pvalues.get("gamma[1]", np.nan),
        "nu": params.get("nu", np.nan), "admissible": admissible,
        "mean_reverting": bool(admissible and mean_reverting),
        "LB(10) z p": _lb(z)["lb_pvalue"], "LB(10) z² p": _lb(z.pow(2))["lb_pvalue"],
        "ARCH-LM(10) z stat": arch_test.lm, "ARCH-LM(10) z p": arch_test.lmpval, "diagnostic_n": len(z),
    }


def _show_conditional_volatility(train_volatility, rolling_sd, output):
    sectors = list(rolling_sd)
    fig, axes = plt.subplots(
        len(sectors), 1, figsize=(11, 2.8 * len(sectors)), squeeze=False, layout="constrained",
    )
    for sector, ax in zip(sectors, axes[:, 0]):
        benchmark = rolling_sd[sector]
        ax.plot(benchmark.index, benchmark, color="grey", lw=0.9, ls="--", label="Rolling SD (21 sessions)")
        for model in VOL_SPECS:
            key = (sector, model)
            if key in train_volatility:
                h_sqrt = train_volatility[key]
                ax.plot(h_sqrt.index, h_sqrt, lw=0.8, label=model)
        ax.set(title=sector, ylabel="Daily volatility (%)")
        ax.legend(ncol=3, fontsize=8)
    fig.suptitle("Output 4 · In-sample volatility and simple benchmarks", fontweight="bold")
    output.show_figure(fig, "04_conditional_volatility.png")


def _show_model_acf(standardized_residuals, output):
    for sector in dict.fromkeys(sector for sector, _ in standardized_residuals):
        series = {
            model: z for (model_sector, model), z in standardized_residuals.items()
            if model_sector == sector
        }
        fig, axes = plt.subplots(
            len(series), 2, figsize=(10, 2.0 * len(series)),
            squeeze=False, layout="constrained",
        )
        for (model, z), (raw_ax, sq_ax) in zip(series.items(), axes):
            plot_acf(z, lags=20, zero=False, ax=raw_ax, title=f"{model} · z")
            plot_acf(z.pow(2), lags=20, zero=False, ax=sq_ax, title=f"{model} · z²")
        fig.suptitle(f"Output 7 · Standardized innovations · {sector}", fontweight="bold")
        output.show_figure(fig, f"07_standardized_ACF_{sector.replace(' ', '_')}.png")


def fit_volatility_models(
    sample: ModelSample, config: AnalysisConfig, output: AnalysisOutput
) -> VolatilityFit:
    """Fit five variance methods to the same mean-filtered training innovations."""
    diagnostic_rows, parameter_rows, history = [], [], []
    train_volatility, standardized, fitted_parameters, rolling_sd = {}, {}, {}, {}
    for sector, innovations in sample.mean_residuals.items():
        rolling = innovations.rolling(21).std().shift(1).rename("rolling_sd_21")
        rolling_sd[sector] = rolling
        history.append(rolling.rename_axis("date").reset_index().assign(sector=sector, model="Rolling SD"))
        for model in VOL_SPECS:
            key = (sector, model)
            try:
                params, h, fit = fit_volatility(innovations, model)
                stderr, pvalues = _fit_pvalues(fit, params)
                conditional_vol = np.sqrt(h)
                z_full = (innovations / conditional_vol).rename("standardized_innovation")
                z = z_full.iloc[20:]
                diagnostic_rows.append(volatility_diagnostics(sector, model, params, h, z, fit, pvalues))
                train_volatility[key] = conditional_vol
                standardized[key] = z
                fitted_parameters[key] = params
                history.append(pd.DataFrame({
                    "innovation": innovations, "variance": h,
                    "conditional_vol_pct": conditional_vol, "standardized_innovation": z_full,
                }).rename_axis("date").reset_index().assign(sector=sector, model=model))
                for name, value in params.items():
                    parameter_rows.append({
                        "sector": sector, "model": model, "parameter": name, "estimate": value,
                        "robust_std_error": stderr.get(name, np.nan),
                        "p_value": pvalues.get(name, np.nan) if name != "nu" else np.nan,
                    })
            except FIT_ERRORS as exc:
                diagnostic_rows.append({"sector": sector, "model": model, "status": str(exc).splitlines()[0]})

    columns = [
        "sector", "model", "status", "AIC", "BIC", "mean_conditional_vol_pct", "omega",
        "alpha", "alpha_p", "beta", "beta_p", "persistence", "shock_half_life_sessions",
        "gamma", "gamma_p", "nu", "admissible", "mean_reverting",
        "LB(10) z p", "LB(10) z² p", "ARCH-LM(10) z stat", "ARCH-LM(10) z p", "diagnostic_n",
    ]
    diagnostics = pd.DataFrame(diagnostic_rows, columns=columns)
    details = columns[:18]
    output.save_table(diagnostics[details], "04_volatility_model_estimates.csv", index=False)
    output.save_table(pd.DataFrame(parameter_rows), "04_volatility_parameters.csv", index=False)
    if history:
        output.save_table(pd.concat(history, ignore_index=True), "04_training_volatility_and_residuals.csv", index=False)
        _show_conditional_volatility(train_volatility, rolling_sd, output)
    if not diagnostics.empty:
        primary = diagnostics.loc[diagnostics["model"].eq("GARCH-t"), [
            "sector", "status", "mean_conditional_vol_pct", "alpha", "alpha_p", "beta", "beta_p",
            "persistence", "shock_half_life_sessions", "nu", "admissible", "mean_reverting",
        ]].copy()
        for model, suffix in (("GJR-t", "GJR"), ("EGARCH-t", "EGARCH")):
            by_sector = diagnostics.loc[diagnostics["model"].eq(model)].set_index("sector")
            primary[f"gamma_{suffix}"] = primary["sector"].map(by_sector["gamma"])
            primary[f"gamma_p_{suffix}"] = primary["sector"].map(by_sector["gamma_p"])
        primary = primary.sort_values("persistence", ascending=False)
        output.save_table(primary, "04_sector_persistence_garch_t.csv", index=False)
        output.display(primary, diagnostics[details])
    return VolatilityFit(diagnostics, standardized, fitted_parameters)


def next_variance(model, params, residual, variance):
    """Update next session's variance using today's innovation and variance."""
    if model == "EWMA-n":
        return EWMA_LAMBDA * variance + (1 - EWMA_LAMBDA) * residual**2
    if model == "EGARCH-t":
        z = residual / np.sqrt(variance)
        log_variance = (
            params["omega"] + params["alpha[1]"] * (abs(z) - np.sqrt(2 / np.pi))
            + params["gamma[1]"] * z + params["beta[1]"] * np.log(variance)
        )
        with np.errstate(over="raise", invalid="raise"):
            return np.exp(log_variance)
    impact = params["alpha[1]"]
    if model == "GJR-t" and residual < 0:
        impact += params["gamma[1]"]
    return params["omega"] + impact * residual**2 + params["beta[1]"] * variance


def _rolling_mean(recent, targets, order, previous_params):
    status = "fit"
    try:
        fit = _estimate_arma(recent, order)
    except FIT_ERRORS as exc:
        fit = _arma_model(recent, order).filter(previous_params)
        status = f"previous mean parameters: {exc}"
    innovations = _arma_residuals(fit, recent.index, order)
    # extend filters the held-out block in sequence, without smoothing or parameter
    # estimation. forecasts[0, k] is the prediction before processing targets[k].
    predicted_mean = fit.extend(targets.to_numpy()).filter_results.forecasts[0]
    means = pd.Series(predicted_mean, index=targets.index, name="mu")
    return innovations, means, fit.params, status


def _empirical_lower_tail(values, alpha):
    """Inverse-CDF quantile and mean of the lowest exact alpha probability mass."""
    ordered = np.sort(values)
    mass = alpha * len(ordered)
    full = int(np.floor(mass))
    cutoff = int(np.ceil(mass)) - 1
    total = ordered[:full].sum()
    if full < len(ordered):
        total += (mass - full) * ordered[full]
    return float(ordered[cutoff]), float(total / mass)


def forecast_block(innovations, targets, means, model, previous_params=None):
    """Forecast each next variance, then update with the observed innovation."""
    status = "fit"
    try:
        params, h, _ = fit_volatility(innovations, model)
    except FIT_ERRORS as exc:
        if previous_params is None or model == "EWMA-n":
            raise
        fixed = arch_model(innovations, mean="Zero", rescale=False, **VOL_SPECS[model]).fix(previous_params)
        params, h = fixed.params, fixed.conditional_volatility.pow(2)
        _check_variances(h)
        status = f"previous volatility parameters: {exc}"

    realized = targets - means
    variance = float(h.iloc[-1])
    last_innovation = float(innovations.iloc[-1])
    predictions = []
    for innovation in realized:
        variance = float(next_variance(model, params, last_innovation, variance))
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("The variance forecast is nonpositive or nonfinite.")
        predictions.append(variance)
        last_innovation = innovation
    frame = pd.DataFrame({
        "actual": targets, "mu": means, "variance": predictions,
        "nu": params.get("nu", np.nan), "innovation": realized,
        "proxy_var": realized.pow(2), "std_resid": realized / np.sqrt(predictions),
    }, index=targets.index)
    if model == "GARCH-t":
        residual_sample = (innovations / np.sqrt(h)).iloc[20:].to_numpy()
        for alpha, suffix in ((0.01, "01"), (0.05, "05")):
            q, tail_mean = _empirical_lower_tail(residual_sample, alpha)
            frame[f"fhs_q{suffix}"] = q
            frame[f"fhs_es{suffix}"] = tail_mean
    return frame, params, status


def forecast_volatility(
    sample: ModelSample, config: AnalysisConfig, output: AnalysisOutput, fit: VolatilityFit | None = None,
    *, expanding: bool = False, models: tuple[str, ...] | None = None, save: bool = True,
) -> VolatilityForecast:
    """Refit a fixed ARMA order and separate variance models on the selected history."""
    selected = tuple(VOL_SPECS) if models is None else models
    forecasts, refit_rows, score_rows = {}, [], []
    for sector, y in sample.series_by_sector.items():
        cut = sample.split_positions[sector]
        window = min(cut, config.max_window)
        test = y.iloc[cut:]
        order = sample.orders[sector]
        last_mean = sample.mean_parameters[sector]
        prior_volatility = {
            model: fit.parameters.get((sector, model)) if fit is not None else None
            for model in selected
        }
        blocks = {model: [] for model in selected}
        for origin in range(cut, len(y), config.refit):
            start = 0 if expanding else origin - window
            recent = y.iloc[start:origin]
            targets = y.iloc[origin:origin + config.refit]
            innovations, means, last_mean, mean_status = _rolling_mean(recent, targets, order, last_mean)
            for model in selected:
                convergence = "fit"
                parameter_checks = {}
                try:
                    frame, params, convergence = forecast_block(
                        innovations, targets, means, model, prior_volatility[model]
                    )
                    prior_volatility[model] = params
                    blocks[model].append(frame)
                    persistence = _persistence(model, params)
                    admissible, mean_reverting = _parameter_admissibility(model, params, persistence)
                    parameter_checks = {
                        "alpha": params.get("alpha[1]", np.nan),
                        "beta": params.get("beta[1]", np.nan),
                        "gamma": params.get("gamma[1]", np.nan),
                        "nu": params.get("nu", np.nan),
                        "persistence": persistence,
                        "admissible": admissible,
                        "mean_reverting": bool(admissible and mean_reverting),
                    }
                except FIT_ERRORS as exc:
                    convergence = f"no forecast: {str(exc).splitlines()[0]}"
                refit_rows.append({
                    "sector": sector, "model": model, "first_target": y.index[origin],
                    "fit_start": recent.index[0], "fit_end": recent.index[-1],
                    "mean_status": mean_status, "status": convergence,
                    **parameter_checks,
                })

        for model, model_blocks in blocks.items():
            if not model_blocks:
                continue
            block_forecasts = pd.concat(model_blocks)
            if not block_forecasts.index.equals(test.index):
                continue
            forecasts[sector, model] = block_forecasts
            var_hat, proxy = block_forecasts["variance"], block_forecasts["proxy_var"]
            error = proxy - var_hat
            score_rows.append({
                "sector": sector, "model": model, "n_test": len(test),
                "mse": error.pow(2).mean(), "mae": error.abs().mean(),
                "qlike": (np.log(var_hat) + proxy / var_hat).mean(),
            })

    refit_log = pd.DataFrame(refit_rows, columns=[
        "sector", "model", "first_target", "fit_start", "fit_end", "mean_status", "status",
        "alpha", "beta", "gamma", "nu", "persistence", "admissible", "mean_reverting",
    ])
    variance_scores = pd.DataFrame(score_rows, columns=["sector", "model", "n_test", "mse", "mae", "qlike"])
    variance_scores = variance_scores.sort_values(["sector", "qlike", "mse"], ignore_index=True)
    result = VolatilityForecast(forecasts, variance_scores, refit_log)
    if save:
        publish_volatility_evaluation(fit, result, output)
    return result


def publish_volatility_evaluation(
    fit: VolatilityFit | None, forecast: VolatilityForecast, output: AnalysisOutput,
):
    """Export Output 7 diagnostics and holdout evaluation without re-estimating models."""
    if fit is not None:
        diagnostic_columns = [
            "sector", "model", "status", "admissible", "mean_reverting", "LB(10) z p",
            "LB(10) z² p", "ARCH-LM(10) z stat", "ARCH-LM(10) z p", "diagnostic_n",
        ]
        residual_tests = fit.vol_diagnostics[diagnostic_columns]
        output.save_table(residual_tests, "07_standardized_residual_tests.csv", index=False)
        if not residual_tests.empty:
            output.display(residual_tests)
            _show_model_acf(fit.standardized_residuals, output)
    forecasts, variance_scores, refit_log = forecast.forecasts, forecast.variance_scores, forecast.refit_log
    output.save_table(variance_scores, "07_out_of_sample_scores.csv", index=False)
    output.save_table(refit_log, "07_model_refits.csv", index=False)
    if forecasts:
        history = pd.concat([
            frame.rename_axis("date").reset_index().assign(sector=sector, model=model)
            for (sector, model), frame in forecasts.items()
        ], ignore_index=True)
        output.save_table(history, "07_daily_volatility_forecasts.csv", index=False)
        output.display(variance_scores)
        _show_variance_forecasts(forecasts, variance_scores, output)
    else:
        print("Chưa có dự báo đủ toàn bộ kỳ kiểm định.")

    if not refit_log.empty:
        exceptions = refit_log.loc[refit_log["status"].ne("fit") | refit_log["mean_status"].ne("fit")]
        if not exceptions.empty:
            print("Nhật ký ghi các lần dùng lại tham số và các khối thiếu ngày dự báo.")
            output.display(exceptions.groupby(["sector", "model", "mean_status", "status"]).size().rename("blocks"))


def _show_variance_forecasts(forecasts, scores, output):
    sectors = scores["sector"].unique()
    fig, axes = plt.subplots(
        len(sectors), 1, figsize=(11, 2.65 * len(sectors)),
        squeeze=False, layout="constrained",
    )
    for sector, ax in zip(sectors, axes[:, 0]):
        by_sector = scores.loc[scores["sector"].eq(sector)]
        winner = by_sector.iloc[0]["model"]
        mse_winner = by_sector.loc[by_sector["mse"].idxmin(), "model"]
        mae_winner = by_sector.loc[by_sector["mae"].idxmin(), "model"]
        print(f"{sector} · QLIKE thấp nhất: {winner}. MSE: {mse_winner}. MAE: {mae_winner}.")
        for model in by_sector["model"]:
            fc_variance = forecasts[sector, model]["variance"]
            ax.plot(fc_variance.index, np.sqrt(fc_variance), lw=1.35 if model == winner else 0.75, label=model)
        ax.set(title=sector, ylabel="Forecast volatility (%)")
        ax.legend(ncol=5, fontsize=8)
    fig.suptitle("Output 7 · One-session holdout volatility forecasts", fontweight="bold")
    output.show_figure(fig, "07_rolling_volatility.png")
