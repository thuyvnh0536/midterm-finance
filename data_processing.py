"""Outputs 1–2: FiinProX closing prices, daily returns and exploratory diagnostics."""

from dataclasses import dataclass
from itertools import cycle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss

from analysis_config import AnalysisConfig, AnalysisOutput


REQUIRED_HEADERS = {"Index/Sector", "Date", "Close Index (D)"}
LABEL_ALIASES = {"Consumer Good L1": "Consumer Goods L1"}
SOURCE_NOTES = """# Output 1 · FiinProX data sources and construction

Source: FiinProX Trading data – Index & Sector (Group by date), exported November 2026.
The original workbooks are under `data/raw/`. See `01_input_sources.csv` for the
workbook and worksheet list and `01_sector_provenance.csv` for the actual labels,
coverage and row counts from each source. Consumer Goods L1 is the spelling in
the FiinProX data. Categories retain their L1, L2 or L4 level as exported, so the
five indices represent sectors at different levels of breadth.

The input variable is the daily closing index level in points. Dates are sorted,
identical valid duplicates are collapsed, and conflicting valid closes are
reported for reconciliation. Missing, nonnumeric and nonpositive closes are
excluded. Returns are 100 × ln(P[t] / P[t−1]), measured in percent. They require
both prices in consecutive rows of the observed calendar. Missing prices are not
filled. The synchronized panel keeps returns observed for every selected index.
Its long-form file includes |r| in percent and r² in squared percent.

The union of selected dates serves as the observed calendar. The date-gap audit
flags weekdays between observations for calendar reconciliation. Vietnamese
exchange holidays (including Tết) and other market closures may account for these
gaps. The exports do not contain an official exchange calendar or permit a gap
common to all sectors to be distinguished from a closure. An unchanged close
alone does not establish that the index or its constituents had no trading.

The extreme-move table reports the largest absolute moves per index, while the
data audit flags any absolute simple change above 10% for inspection. This is an
index-level review threshold. The files lack constituent-level closes, exchange
codes, weights and trading status, so the price bands of constituent securities,
constituent concentration and individual nontrading sessions cannot be checked.
The files also do not establish the full HOSE/HNX universe, the ICB classification
version, how prices are adjusted for corporate actions or whether index methodology
changed. Those checks would require the FiinProX methodology, constituent/weight
histories and official exchange data. Extreme observations are kept as observed.
"""


@dataclass(frozen=True)
class SectorData:
    """Closing prices and returns aligned to the observed trading-session dates."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    available_sectors: list[str]
    active_sectors: list[str]
    coverage: pd.DataFrame
    common_returns: pd.DataFrame


def _header_name(value: object) -> str:
    return str(value).split("\n", 1)[0].strip()


def _read_workbooks(config: AnalysisConfig) -> tuple[pd.DataFrame, list[dict]]:
    """Read the requested price and date columns from the supplied workbooks."""
    excel_paths = sorted(
        path for path in config.raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".xlsx" and not path.name.startswith("~$")
    ) if config.raw_dir.is_dir() else []
    if not excel_paths:
        raise FileNotFoundError(f"Chưa tìm thấy tệp .xlsx trong {config.raw_dir}. Kiểm tra thư mục dữ liệu.")

    source_frames, source_audit = [], []
    for path in excel_paths:
        with pd.ExcelFile(path, engine="openpyxl") as workbook:
            for sheet in workbook.sheet_names:
                source = f"{path.name} / {sheet}"
                raw = pd.read_excel(
                    workbook, sheet_name=sheet,
                    usecols=lambda name: _header_name(name) in REQUIRED_HEADERS,
                )
                raw.columns = [_header_name(name) for name in raw.columns]
                if not REQUIRED_HEADERS.issubset(raw.columns):
                    source_audit.append({
                        "Source": source, "Rows read": len(raw), "Rows selected": 0,
                        "Workbook status": "Bỏ qua sheet thiếu cột bắt buộc",
                        "Source labels selected": "",
                    })
                    continue

                source_labels = raw["Index/Sector"].astype("string").str.strip()
                labels = source_labels.replace(LABEL_ALIASES)
                is_target = labels.isin(config.targets)
                selected = raw.loc[is_target, ["Date", "Close Index (D)"]].copy()
                selected["Index/Sector"] = labels.loc[selected.index]
                selected["Source label"] = source_labels.loc[selected.index]
                selected["Source"] = source
                source_frames.append(selected.rename(columns={"Close Index (D)": "Close"}))
                source_audit.append({
                    "Source": source, "Rows read": len(raw), "Rows selected": len(selected),
                    "Workbook status": "Đã đọc",
                    "Source labels selected": " · ".join(sorted(source_labels.loc[is_target].unique())),
                })

    if not source_frames:
        raise ValueError("Các sheet Excel cần có Index/Sector, Date và Close Index (D).")
    return pd.concat(source_frames, ignore_index=True), source_audit


def load_sector_data(config: AnalysisConfig, output: AnalysisOutput) -> SectorData:
    """Filter the five sector series, audit observations and export sorted prices."""
    raw_selected, source_audit = _read_workbooks(config)
    save_table, display = output.save_table, output.display
    source_report = pd.DataFrame(source_audit)
    save_table(source_report, "01_input_sources.csv", index=False)
    display(source_report)
    # Excel có thể trả ngày kiểu datetime, chuỗi hoặc số sê-ri Excel.
    serial_dates = raw_selected["Date"].map(
        lambda value: isinstance(value, (int, float, np.number)) and pd.notna(value)
    ).astype(bool)
    date_text = raw_selected["Date"].where(~serial_dates)
    date_values = pd.to_datetime(date_text, format="ISO8601", errors="coerce")
    remaining_dates = date_values.isna() & ~serial_dates
    date_values.loc[remaining_dates] = pd.to_datetime(
        date_text.loc[remaining_dates], format="mixed", dayfirst=True, errors="coerce"
    )
    if serial_dates.any():
        date_values.loc[serial_dates] = pd.to_datetime(
            pd.to_numeric(raw_selected.loc[serial_dates, "Date"]),
            unit="D", origin="1899-12-30", errors="coerce",
        )
    raw_selected["Date"] = date_values.dt.normalize()
    raw_selected["Close"] = pd.to_numeric(raw_selected["Close"], errors="coerce")
    raw_selected["Invalid date"] = raw_selected["Date"].isna()
    raw_selected["Invalid close"] = (~np.isfinite(raw_selected["Close"])) | raw_selected["Close"].le(0)
    provenance = (
        raw_selected.groupby(["Index/Sector", "Source", "Source label"], as_index=False)
        .agg(**{"Raw rows": ("Date", "size"), "First date": ("Date", "min"), "Last date": ("Date", "max")})
    )
    save_table(provenance, "01_sector_provenance.csv", index=False)
    (output.directory / "01_source_and_cleaning_notes.md").write_text(SOURCE_NOTES, encoding="utf-8")

    key = ["Index/Sector", "Date"]
    valid_rows = raw_selected.loc[~raw_selected["Invalid date"] & ~raw_selected["Invalid close"]].copy()
    conflicts = valid_rows.groupby(key)["Close"].nunique()
    if conflicts.gt(1).any():
        conflict_rows = valid_rows.merge(conflicts[conflicts.gt(1)].rename("N prices"), on=key)
        save_table(conflict_rows, "01_conflicting_prices.csv", index=False)
        display(conflict_rows.head(15))
        raise ValueError(f"{conflicts.gt(1).sum()} cặp ngành/ngày có giá xung đột. Kiểm tra 01_conflicting_prices.csv.")

    valid_rows["Redundant row"] = valid_rows.duplicated(subset=key)
    clean_prices = valid_rows.loc[~valid_rows["Redundant row"]].sort_values(["Date", "Index/Sector"])
    observed_dates = pd.DatetimeIndex(raw_selected["Date"].dropna().unique(), name="Date").sort_values()
    prices = clean_prices.pivot(index="Date", columns="Index/Sector", values="Close").reindex(
        index=observed_dates, columns=config.targets
    )
    returns = (100.0 * np.log(prices).diff()).iloc[1:]
    available_sectors = [sector for sector in config.targets if returns[sector].notna().any()]
    active_sectors = []
    coverage_rows, extreme_moves = [], []
    review_threshold = 10.0
    price_gaps = []
    preceding_date = prices.index.to_series().shift(1)
    for sector in config.targets:
        original = raw_selected.loc[raw_selected["Index/Sector"].eq(sector)]
        series = prices[sector].dropna()
        daily = returns[sector].dropna()
        first, last = series.first_valid_index(), series.last_valid_index()
        enough_history = len(series) >= 700 and last >= first + pd.DateOffset(years=3)
        enough_returns = len(daily) >= 650 and daily.nunique() > 1
        if enough_history and enough_returns:
            active_sectors.append(sector)
        level_prices = prices[sector]
        missing_dates = level_prices.loc[first:last].loc[lambda s: s.isna()].index if not series.empty else []
        price_gaps.extend({"Date": date, "Index/Sector": sector} for date in missing_dates)
        changes = 100 * level_prices.pct_change(fill_method=None).dropna()
        flagged = changes.abs().gt(review_threshold)
        extreme_dates = changes.abs().nlargest(5).index.union(changes.index[flagged], sort=False)
        preceding_close = level_prices.shift(1)
        for date in extreme_dates:
            extreme_moves.append({
                "Index/Sector": sector, "Date": date, "Previous observed date": preceding_date.at[date],
                "Previous close": preceding_close.at[date], "Close": level_prices.at[date],
                "Simple return (%)": changes.at[date], "Log return (%)": daily.at[date],
                "Exceeds review threshold": flagged.at[date],
            })
        coverage_rows.append({
            "Index/Sector": sector,
            "Raw rows": len(original),
            "Invalid date": int(original["Invalid date"].sum()),
            "Invalid close": int(original["Invalid close"].sum()),
            "Redundant valid rows": int(valid_rows.loc[valid_rows["Index/Sector"].eq(sector), "Redundant row"].sum()),
            "First date": first, "Last date": last,
            "Valid closes": len(series),
            "Missing within observed span": len(missing_dates),
            "Unchanged closes": int(changes.eq(0).sum()),
            f"|Daily simple change| > {review_threshold:g}%": int(flagged.sum()),
            "Returns": len(daily),
            "Status": (
                "Đủ dữ liệu mô hình" if sector in active_sectors else
                "Chưa có dòng khớp tên" if original.empty else
                "Chưa đủ dữ liệu hợp lệ để mô hình hóa"
            ),
        })
        if not series.empty:
            filename = sector.lower().replace(" ", "_")
            export = series.rename("Close Index (D)").to_frame().reset_index()
            export.insert(1, "Index/Sector", sector)
            save_table(export, f"01_close_{filename}.csv", index=False)

    coverage = pd.DataFrame(coverage_rows).set_index("Index/Sector")
    save_table(coverage, "01_data_coverage.csv")
    save_table(prices, "01_closes_by_date.csv")
    save_table(returns, "01_log_returns_pct.csv")
    synchronized = returns.dropna()
    save_table(synchronized, "01_synchronized_log_returns_pct.csv")
    long_returns = synchronized.reset_index().melt(
        id_vars="Date", var_name="Index/Sector", value_name="Log return (%)",
    ).sort_values(["Date", "Index/Sector"])
    long_returns["Absolute log return (%)"] = long_returns["Log return (%)"].abs()
    long_returns["Squared log return (%²)"] = long_returns["Log return (%)"] ** 2
    save_table(long_returns, "01_synchronized_return_transforms_long.csv", index=False)
    save_table(pd.DataFrame(price_gaps, columns=["Date", "Index/Sector"]), "01_known_missing_closes.csv", index=False)
    extremes = pd.DataFrame(extreme_moves, columns=[
        "Index/Sector", "Date", "Previous observed date", "Previous close", "Close",
        "Simple return (%)", "Log return (%)", "Exceeds review threshold",
    ])
    save_table(extremes, "01_extreme_price_moves.csv", index=False)
    save_table(extremes.loc[extremes["Exceeds review threshold"].astype(bool)], "01_large_moves_for_review.csv", index=False)

    each_date = observed_dates.to_numpy(dtype="datetime64[D]")
    before, after = each_date[:-1], each_date[1:]
    weekday_gaps = np.busday_count(before + np.timedelta64(1, "D"), after)
    calendar_gaps = pd.DataFrame({
        "Previous observed date": pd.to_datetime(before), "Next observed date": pd.to_datetime(after),
        "Calendar days between observations": (after - before).astype(int) - 1,
        "Unverified weekdays between observations": weekday_gaps,
    })
    save_table(calendar_gaps.loc[weekday_gaps > 0], "01_observed_calendar_gaps.csv", index=False)
    display(coverage)
    common_returns = returns[available_sectors].dropna() if available_sectors else returns.iloc[:0, :0]
    if available_sectors:
        if common_returns.empty:
            print("Các ngành có lợi suất hiện chưa có ngày trùng nhau.")
        else:
            print(f"Lợi suất có đủ ở {len(available_sectors)} ngành hiện có: {len(common_returns):,} phiên "
                  f"({common_returns.index.min():%d/%m/%Y}–{common_returns.index.max():%d/%m/%Y}).")
    if common_returns.shape[1] != len(config.targets):
        print("File lợi suất đồng bộ cần dữ liệu của mọi ngành trong cấu hình.")
    missing_sectors = [sector for sector in config.targets if coverage.at[sector, "Raw rows"] == 0]
    if missing_sectors:
        print("Chưa có dữ liệu nguồn khớp tên: " + ", ".join(missing_sectors) + ".")
    print("Đủ dữ liệu để mô hình hóa:", ", ".join(active_sectors) or "chưa có ngành nào")
    print("Biến động giá tuyệt đối trên 10% là cờ kiểm tra; dữ liệu quan sát vẫn được giữ nguyên.")
    print(f"Các khoảng có ít nhất một ngày trong tuần chưa xác minh: {int((weekday_gaps > 0).sum())}.")
    print("Xem 01_source_and_cleaning_notes.md để biết nguồn, cách đồng bộ và giới hạn về lịch, cấu phần và phương pháp chỉ số.")
    return SectorData(prices, returns, available_sectors, active_sectors, coverage, common_returns)


def _stationarity_tests(series: pd.Series) -> dict[str, float | str]:
    """Test unit root and level stationarity on a nonconstant series."""
    series = series.dropna()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="adfuller currently returns", category=FutureWarning)
        adf_result = adfuller(series, regression="c", autolag="AIC")
    with warnings.catch_warnings(record=True) as kpss_warnings:
        warnings.simplefilter("always", InterpolationWarning)
        kpss_result = kpss(series, regression="c", nlags="auto")
    bounds_reached = any(issubclass(w.category, InterpolationWarning) for w in kpss_warnings)
    display_bound = ("<" if kpss_result[1] <= .01 else ">") if bounds_reached else "="
    return {
        "N": len(series), "ADF statistic": adf_result[0], "ADF p-value": adf_result[1],
        "ADF lags": adf_result[2], "KPSS statistic": kpss_result[0], "KPSS p-value": kpss_result[1],
        "KPSS p-value bound": display_bound, "KPSS lags": kpss_result[2],
    }


def describe_sector_data(data: SectorData, config: AnalysisConfig, output: AnalysisOutput) -> None:
    """Generate distribution, stationarity and dependence tables and figures."""
    prices, returns = data.prices, data.returns
    available_sectors, common_returns = data.available_sectors, data.common_returns
    save_table, display, show_figure = output.save_table, output.display, output.show_figure

    style_colors = ("#0F6773", "#A7452E", "#644C9A", "#246128", "#8A5F00")
    plot_colors = dict(zip(config.targets, cycle(style_colors)))
    if available_sectors:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
        for sector in available_sectors:
            level = prices[sector]
            axes[0].plot(level, label=sector, color=plot_colors[sector], linewidth=0.9)
            rebased = 100 * level / level.dropna().iloc[0]
            axes[1].plot(rebased, label=sector, color=plot_colors[sector], linewidth=0.9)
        axes[0].set_ylabel("Điểm chỉ số")
        axes[1].set_ylabel("Phiên đầu = 100")
        axes[0].legend(loc="upper left", ncol=min(3, len(available_sectors)), frameon=False)
        fig.suptitle("Figure 1. Closing index levels and rebased closing indices")
        show_figure(fig, "02_figure_1_closing_levels.png")

        fig, axes = plt.subplots(
            len(available_sectors), 1, figsize=(12, 2 + 1.55 * len(available_sectors)),
            sharex=True, squeeze=False, layout="constrained",
        )
        for sector, ax in zip(available_sectors, axes[:, 0]):
            ax.plot(returns[sector], color=plot_colors[sector], linewidth=0.65)
            ax.axhline(0, color="#52606D", linewidth=0.6)
            ax.set_title(sector, loc="left")
            ax.set_ylabel("%")
        fig.suptitle("Figure 2. Daily log returns")
        show_figure(fig, "02_figure_2_log_returns.png")
    else:
        print("Chưa có cặp giá đóng cửa liên tiếp để tính lợi suất.")

    summary_rows = []
    diagnostic_sectors = []
    for sector in available_sectors:
        daily = returns[sector].dropna()
        if len(daily) >= 40 and daily.nunique() > 1:
            diagnostic_sectors.append(sector)
        jb = stats.jarque_bera(daily) if len(daily) >= 3 and daily.nunique() > 1 else None
        summary_rows.append({
            "Index/Sector": sector,
            "N": len(daily),
            "Mean daily (%)": daily.mean(),
            "SD daily (%)": daily.std(),
            "Vol annualized (%)": daily.std() * np.sqrt(252),
            "Variance daily (%²)": daily.var(),
            "Skewness": daily.skew(),
            "Excess kurtosis": daily.kurt(),
            "Minimum (%)": daily.min(),
            "Maximum (%)": daily.max(),
            **{f"P{int(q * 100):02d} (%)": daily.quantile(q) for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)},
            "JB statistic": jb.statistic if jb else np.nan,
            "JB p-value": jb.pvalue if jb else np.nan,
            "JB log10(p-value)": -jb.statistic / (2 * np.log(10)) if jb else np.nan,
        })
    if summary_rows:
        eda_summary = pd.DataFrame(summary_rows).set_index("Index/Sector")
        save_table(eda_summary, "02_descriptive_statistics.csv")
        display(eda_summary)
        _plot_return_distributions(returns, available_sectors, plot_colors, show_figure)

    if len(available_sectors) >= 2 and len(common_returns) >= 2:
        common_vol = (common_returns.std() * np.sqrt(252)).rename("Vol annualized, common sample (%)").to_frame()
        common_vol["Common observations"] = len(common_returns)
        common_corr = common_returns.corr()
        save_table(common_vol, "02_common_sample_volatility.csv")
        save_table(common_corr, "02_common_sample_correlations.csv")
        display(common_vol, common_corr)

    if diagnostic_sectors:
        fig, axes = plt.subplots(
            len(diagnostic_sectors), 2, figsize=(12, 2.5 * len(diagnostic_sectors)),
            squeeze=False, layout="constrained", gridspec_kw={"width_ratios": [1, 1.4]},
        )
        for sector, row_axes in zip(diagnostic_sectors, axes):
            ax_qq, ax_abs = row_axes
            daily = returns[sector].dropna()
            stats.probplot(daily, dist="norm", plot=ax_qq)
            ax_qq.get_lines()[0].set_color(plot_colors[sector])
            ax_qq.get_lines()[0].set_markersize(3)
            ax_qq.get_lines()[1].set_color("#52606D")
            ax_qq.set_title(sector)
            ax_qq.set_ylabel("Log return (%)")
            ax_abs.plot(returns[sector].abs(), color=plot_colors[sector], linewidth=0.65)
            ax_abs.set_title(f"{sector} · absolute returns")
            ax_abs.set_ylabel("%")
        fig.suptitle("Figure 3. Normal Q–Q plots and absolute daily returns")
        show_figure(fig, "02_figure_3_qq_and_absolute_returns.png")

    stationarity_rows, clustering_rows = [], []
    if diagnostic_sectors:
        fig, axes = plt.subplots(
            len(diagnostic_sectors), 3, figsize=(13, 2.45 * len(diagnostic_sectors)),
            squeeze=False, layout="constrained",
        )
        for sector, (ax_ret, ax_abs, ax_sq) in zip(diagnostic_sectors, axes):
            levels = np.log(prices[sector].dropna())
            daily = returns[sector].dropna()
            for label, series in (("Log price", levels), ("Log return", daily)):
                stationarity_rows.append({
                    "Index/Sector": sector, "Series": label, **_stationarity_tests(series)
                })
            centered = daily - daily.mean()
            plot_acf(daily, lags=20, ax=ax_ret, zero=False, title=f"{sector} · returns")
            plot_acf(daily.abs(), lags=20, ax=ax_abs, zero=False, title=f"{sector} · absolute returns")
            plot_acf(daily ** 2, lags=20, ax=ax_sq, zero=False, title=f"{sector} · squared returns")
            lb_return = acorr_ljungbox(daily, lags=[10], return_df=True).iloc[0]
            lb_absolute = acorr_ljungbox(daily.abs(), lags=[10], return_df=True).iloc[0]
            lb_square = acorr_ljungbox(daily ** 2, lags=[10], return_df=True).iloc[0]
            return_acf = acf(daily, nlags=10, fft=True)[1:]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="acorr_lm currently returns", category=FutureWarning)
                lm_stat, lm_p, _, _ = het_arch(centered, nlags=10)
            clustering_rows.append({
                "Index/Sector": sector,
                "ACF r lag 1": return_acf[0],
                "Maximum |ACF r|, lags 1–10": np.max(np.abs(return_acf)),
                "LB(10) r statistic": lb_return["lb_stat"], "LB(10) r p-value": lb_return["lb_pvalue"],
                "LB(10) absolute statistic": lb_absolute["lb_stat"], "LB(10) absolute p-value": lb_absolute["lb_pvalue"],
                "LB(10) squared statistic": lb_square["lb_stat"], "LB(10) squared p-value": lb_square["lb_pvalue"],
                "ARCH-LM(10) statistic": lm_stat, "ARCH-LM(10) p-value": lm_p,
            })
        fig.suptitle("Figure 4. Autocorrelations of raw, absolute and squared daily returns")
        show_figure(fig, "02_figure_4_autocorrelations.png")
        eda_stationarity = pd.DataFrame(stationarity_rows).set_index(["Index/Sector", "Series"])
        eda_clustering = pd.DataFrame(clustering_rows).set_index("Index/Sector")
        save_table(eda_stationarity, "02_stationarity_tests.csv")
        save_table(eda_clustering, "02_dependence_tests.csv")
        display(eda_stationarity, eda_clustering)
    else:
        print("ADF, KPSS, ACF, Ljung–Box và ARCH–LM cần ít nhất 40 lợi suất có biến thiên.")

    if diagnostic_sectors:
        _summarize_stylized_facts(
            returns[diagnostic_sectors], eda_summary, eda_stationarity, eda_clustering, config.significance, output,
        )


def _plot_return_distributions(returns, sectors, colors, show_figure):
    """Show sector distributions on common axes and compare their tails."""
    valid = returns[sectors].stack().to_numpy()
    lower, upper = valid.min(), valid.max()
    margin = max((upper - lower) * 0.03, 0.01)
    bins = np.linspace(lower - margin, upper + margin, 51)
    grid = np.linspace(bins[0], bins[-1], 250)
    fig, axes = plt.subplots(
        len(sectors), 1, figsize=(9, 1.9 * len(sectors) + 1.3),
        sharex=True, squeeze=False, layout="constrained"
    )
    for sector, ax in zip(sectors, axes[:, 0]):
        daily = returns[sector].dropna()
        ax.hist(daily, bins=bins, density=True, color=colors[sector], alpha=0.76)
        if len(daily) >= 2 and daily.std() > 0:
            ax.plot(grid, stats.norm.pdf(grid, daily.mean(), daily.std()), color="#34495E", linewidth=1.1)
        ax.set_title(sector, loc="left")
        ax.set_ylabel("Density")
    axes[-1, 0].set_xlabel("Daily log return (%)")
    fig.suptitle("Figure 2a. Return distributions with fitted normal reference")
    show_figure(fig, "02_fig_histograms.png")

    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    boxes = ax.boxplot(
        [returns[sector].dropna().to_numpy() for sector in sectors],
        tick_labels=sectors, patch_artist=True,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.45},
        medianprops={"color": "#17283e", "linewidth": 1.5},
    )
    for sector, box in zip(sectors, boxes["boxes"]):
        box.set_facecolor(colors[sector])
        box.set_alpha(0.52)
    ax.axhline(0, color="#52606D", linewidth=0.6)
    ax.set_ylabel("Daily log return (%)")
    ax.set_title("Figure 2b. Daily log returns by sector · Tukey boxplots")
    show_figure(fig, "02_fig_boxplots.png")


def _volatility_asymmetry_screen(series: pd.Series) -> dict:
    """Relate the next-session square to today's magnitude, with a separate negative-shock slope."""
    centered = series - series.mean()
    magnitude = centered.abs().rename("Magnitude")
    negative_term = magnitude.where(centered.lt(0), 0).rename("Negative × magnitude")
    pairs = pd.concat([centered.shift(-1).pow(2).rename("Next squared return"), magnitude, negative_term], axis=1)
    pairs = pairs.dropna()
    if len(pairs) < 20 or pairs["Negative × magnitude"].nunique() < 2:
        return {
            "Adjacent pairs": len(pairs),
            "Positive slope": np.nan,
            "Slope difference, negative minus positive": np.nan,
            "HAC p-value (two-sided)": np.nan,
        }
    model = OLS(
        pairs["Next squared return"], add_constant(pairs[["Magnitude", "Negative × magnitude"]])
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "Adjacent pairs": len(pairs),
        "Positive slope": model.params["Magnitude"],
        "Slope difference, negative minus positive": model.params["Negative × magnitude"],
        "HAC p-value (two-sided)": model.pvalues["Negative × magnitude"],
    }


def _aggregate_log_returns(series: pd.Series, horizon: int) -> pd.Series:
    """Take complete, non-overlapping blocks of adjacent observations."""
    complete_length = len(series) - len(series) % horizon
    blocks = series.iloc[:complete_length].to_numpy().reshape(-1, horizon)
    return pd.Series(blocks.sum(axis=1)).dropna()


def _summarize_stylized_facts(returns, summary, stationarity, dependence, significance, output):
    """Translate seven sample regularities into sector-specific modeling decisions."""
    interpretation, asymmetry_rows, aggregation_rows = [], [], []
    for sector in returns:
        series = returns[sector]
        daily = series.dropna()
        descriptive = summary.loc[sector]
        price_tests = stationarity.loc[(sector, "Log price")]
        return_tests = stationarity.loc[(sector, "Log return")]
        dynamics = dependence.loc[sector]
        level_unit_root = price_tests["KPSS p-value"] < significance <= price_tests["ADF p-value"]
        return_stationary = return_tests["ADF p-value"] < significance <= return_tests["KPSS p-value"]
        asymmetry = _volatility_asymmetry_screen(series)
        asymmetry_rows.append({
            "Index/Sector": sector,
            "Negative sessions (%)": 100 * daily.lt(0).mean(),
            "Mean gain given positive return (%)": daily[daily.gt(0)].mean(),
            "Mean loss magnitude given negative return (%)": -daily[daily.lt(0)].mean(),
            **asymmetry,
        })
        tail_kurtosis = {}
        for horizon in (1, 5, 20):
            sums = _aggregate_log_returns(series, horizon)
            k = sums.kurt()
            jb = stats.jarque_bera(sums) if len(sums) >= 8 and sums.nunique() > 1 else None
            tail_kurtosis[horizon] = k
            aggregation_rows.append({
                "Index/Sector": sector, "Horizon (sessions)": horizon, "Complete blocks": len(sums),
                "Skewness": sums.skew(), "Excess kurtosis": k,
                "JB p-value (exploratory)": jb.pvalue if jb else np.nan,
            })
        price_evidence = "Log prices consistent with a unit root" if level_unit_root else "Read log-price ADF and KPSS separately"
        return_evidence = "returns consistent with level stationarity" if return_stationary else "read return ADF and KPSS separately"
        low_q, high_q = descriptive["P01 (%)"], descriptive["P99 (%)"]
        skewness = descriptive["Skewness"]
        peak_acf = dynamics["Maximum |ACF r|, lags 1–10"]
        leverage_delta = asymmetry["Slope difference, negative minus positive"]
        leverage_p = asymmetry["HAC p-value (two-sided)"]
        leverage_evidence = (
            f"Negative minus positive slope {leverage_delta:+.3f}, exploratory HAC p={leverage_p:.3g}"
            if np.isfinite(leverage_p) else "Insufficient adjacent observations for the asymmetry screen"
        )
        changing_volatility = min(
            dynamics["LB(10) absolute p-value"],
            dynamics["LB(10) squared p-value"],
            dynamics["ARCH-LM(10) p-value"],
        ) < significance
        fixed_mean = dynamics["LB(10) r p-value"] >= significance
        heavy_point_estimate = descriptive["Excess kurtosis"] > 0 and descriptive["JB p-value"] < significance
        interpretations = [
            (
                "1. Price and return stationarity",
                f"{price_evidence}; {return_evidence}",
                "Use daily log returns as inputs. Review stationarity before fitting.",
            ),
            (
                "2. Heavy tails",
                f"Excess kurtosis {descriptive['Excess kurtosis']:.2f}, JB log10(p) {descriptive['JB log10(p-value)']:.1f}",
                "Compare Gaussian and Student-t innovations." if heavy_point_estimate else
                "Check Q–Q and standardized residuals before favoring heavier-tailed innovations.",
            ),
            (
                "3. Gain/loss asymmetry",
                f"Skewness {skewness:+.2f}; 1% / 99% quantiles {low_q:.2f}% / {high_q:.2f}%",
                "Compare empirical and parametric downside tails. Treat daily skewness as descriptive.",
            ),
            (
                "4. Linear dependence",
                f"Max |ACF| at lags 1–10 is {peak_acf:.3f}; Ljung–Box p={dynamics['LB(10) r p-value']:.3g}",
                "Use a constant mean as a starting point and diagnose residuals." if fixed_mean else
                "Check whether a small ARMA improves BIC and residual diagnostics.",
            ),
            (
                "5. Volatility clustering",
                f"Absolute/squared Ljung–Box p={dynamics['LB(10) absolute p-value']:.3g}/"
                f"{dynamics['LB(10) squared p-value']:.3g}; ARCH–LM p={dynamics['ARCH-LM(10) p-value']:.3g}",
                "Compare EWMA and GARCH after checking ARCH effects on mean residuals." if changing_volatility else
                "Check mean residuals for volatility clustering before selecting dynamic variance.",
            ),
            (
                "6. Asymmetric volatility response",
                leverage_evidence,
                "Compare GJR-GARCH and EGARCH, then read fitted asymmetry and fit diagnostics." if leverage_p < significance else
                "Keep symmetric GARCH as the baseline. Use asymmetric models for comparison.",
            ),
            (
                "7. Aggregational Gaussianity",
                "Excess kurtosis at 1 / 5 / 20 sessions is "
                + " / ".join(f"{tail_kurtosis[h]:.2f}" for h in (1, 5, 20)),
                "Check distributional assumptions at each horizon. Forecast VaR directly at one session.",
            ),
        ]
        interpretation.extend(
            {"Index/Sector": sector, "Stylized fact": fact, "Sample evidence": evidence, "Model decision": decision}
            for fact, evidence, decision in interpretations
        )
    readout = pd.DataFrame(interpretation).set_index(["Index/Sector", "Stylized fact"])
    output.save_table(readout, "02_stylized_facts_and_model_decisions.csv")
    output.save_table(readout, "02_diagnostics_readout.csv")
    output.save_table(pd.DataFrame(asymmetry_rows).set_index("Index/Sector"), "02_asymmetry_screen.csv")
    output.save_table(
        pd.DataFrame(aggregation_rows).set_index(["Index/Sector", "Horizon (sessions)"]),
        "02_aggregational_gaussianity.csv",
    )
    output.display(readout)
    print(f"Đọc các kiểm định mô tả ở mức ý nghĩa {significance:.0%} cho từng kiểm định.")
    print(
        "Sàng lọc bất đối xứng hồi quy lợi suất bình phương phiên sau trên độ lớn lợi suất đã trừ trung bình "
        "và tương tác với dấu âm. Sai số chuẩn HAC dùng 5 độ trễ. Đây là liên hệ mô tả trên toàn mẫu, "
        "bình phương lợi suất là đại diện nhiễu cho biến động và hệ số hồi quy không xác định quan hệ nhân quả."
    )
    print(
        "Lợi suất 5/20 phiên là tổng log return theo các khối không chồng lấn, giữ các khối đủ dữ liệu. "
        "Đọc skewness và excess kurtosis cùng cỡ mẫu. Các p-value Jarque–Bera theo kỳ hạn có công suất khác nhau."
    )
