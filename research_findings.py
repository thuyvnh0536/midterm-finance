"""Build an English interpretation from the executed Group 10 outputs."""

from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT = Path(__file__).resolve().with_name("economic_context.md")


def _name_pair(row):
    return f"{row['sector_a']} and {row['sector_b']}"


def write_findings(data, fit, interactions, risk, backtest, inference, config, output):
    """Connect the five modelled sectors, adjusted tests and economic context."""
    alpha = config.significance
    lines = [
        "# Group 10 research findings",
        "",
        "## Measurement",
        "",
    ]
    full_vol = (data.common_returns.std(ddof=1) * np.sqrt(252)).dropna()
    if len(full_vol) >= 2:
        high, low = full_vol.idxmax(), full_vol.idxmin()
        lines.append(
            f"Across {len(data.common_returns):,} matched daily returns, **{high}** had the highest "
            f"annualized sample volatility ({full_vol[high]:.2f}%) and **{low}** the lowest "
            f"({full_vol[low]:.2f}%). These estimates describe the entire observed sample."
        )
    persistence = (
        fit.vol_diagnostics.loc[fit.vol_diagnostics["model"].eq("GARCH-t")]
        .set_index("sector")["persistence"].dropna()
    )
    if len(persistence) >= 2:
        high, low = persistence.idxmax(), persistence.idxmin()
        days = [np.log(0.5) / np.log(persistence[s]) if 0 < persistence[s] < 1 else np.nan for s in (high, low)]
        lines.append(
            f"\nUnder the same training-sample GARCH-t specification, persistence was highest for **{high}** "
            f"({persistence[high]:.4f}, half-life {days[0]:.1f} sessions) and lowest for **{low}** "
            f"({persistence[low]:.4f}, half-life {days[1]:.1f} sessions). "
            "Persistence measures how slowly a variance shock decays, while sample volatility measures "
            "the overall scale of fluctuations."
        )

    lines.extend(["", "## Sector interactions", ""])
    var = interactions.var_summary
    if not var.empty:
        row = var.iloc[0]
        granger = interactions.granger
        tested = granger["HAC_Holm_p"].notna().sum()
        rejected = granger["reject_at_configured_level"].sum()
        lines.append(
            f"The BIC-selected VAR lag was {int(row['lag_bic'])}. The reported VAR({int(row['lag_reported'])}) "
            f"was {'stable' if row['estimated_system_stable'] else 'unstable'}. "
            f"The Holm-adjusted Granger tests rejected {rejected} of {tested} directed hypotheses at {alpha:.0%}. "
            "These tests concern predictive information in lagged returns."
        )
    dcc = interactions.dcc_fit
    if not dcc.empty:
        row = dcc.loc[dcc["model"].str.contains("DCC")].iloc[0]
        lines.append(
            "\nThe DCC fit reported a correlation quasi-log-likelihood improvement over CCC of "
            f"{row['gain_from_CCC']:.2f} and a correlation decay coefficient sum of "
            f"{row['alpha_plus_beta']:.4f}. The likelihood comparison is descriptive. A DCC correlation "
            "describes conditional co-movement and leaves the channels behind that co-movement unidentified."
        )
    pairs = interactions.pair_summary
    if not pairs.empty:
        pair = pairs.loc[pairs["DCC_sd_corr"].idxmax()]
        lines.append(
            "\nAmong the ten matched pairs, the largest time-series standard deviation of estimated DCC "
            f"correlation was for **{_name_pair(pair)}** ({pair['DCC_sd_corr']:.3f}). "
            "Compare the 05-series regime and daily-correlation tables before discussing any stress episode."
        )
    regime = interactions.regime_summary
    bank_property = "Banks L1 / Real Estate L2"
    if bank_property in regime and len(regime) >= 3:
        levels = ", ".join(
            f"{value:.3f} in {window}" for window, value in zip(regime["regime"], regime[bank_property])
        )
        lines.append(
            f"\nThe Bank–Real Estate conditional-correlation averages were {levels}. "
            "A shared exposure to financing conditions is one plausible channel for that movement. "
            "Investigating it requires dated credit, liquidity or property-finance indicators."
        )

    lines.extend(["", "## Downside risk and statistical significance", ""])
    ranks = risk.downside_ranks
    common_ranks = (
        ranks.loc[ranks["model"].eq("GARCH-t") & ranks["confidence_pct"].eq(99)]
        if not ranks.empty else ranks
    )
    if not common_ranks.empty:
        max_es = common_ranks.loc[common_ranks["mean_es_pct"].idxmax()]
        max_var = common_ranks.loc[common_ranks["mean_var_pct"].idxmax()]
        lines.append(
            f"On the GARCH-t holdout paths at 99%, **{max_var['sector']}** had the highest average "
            f"VaR ({max_var['mean_var_pct']:.2f}% log loss) and **{max_es['sector']}** the highest average "
            f"ES ({max_es['mean_es_pct']:.2f}% log loss). The 06-series tables give ranks by confidence "
            "level and method. Changes in rank reflect both volatility and tail shape."
        )
        gaussian = ranks.loc[ranks["model"].eq("GARCH-n") & ranks["confidence_pct"].eq(99)]
        if not gaussian.empty:
            highest_normal = gaussian.loc[gaussian["mean_es_pct"].idxmax()]
            lines.append(
                f"\nWith Gaussian GARCH, **{highest_normal['sector']}** had the highest average "
                f"99% ES ({highest_normal['mean_es_pct']:.2f}%). Comparing the two fits shows why a "
                "sector risk ranking can depend on the innovation distribution."
            )
    duration = risk.high_var_summary
    if not duration.empty:
        row = duration.loc[duration["max_observed_duration_sessions"].idxmax()]
        lines.append(
            f"\nThe longest observed high-VaR spell was {int(row['max_observed_duration_sessions'])} "
            f"trading sessions for **{row['sector']}**. The cutoffs are the sectors' own holdout 90th "
            "percentiles of GARCH-t 99% VaR. The duration comparison is retrospective and relative to each "
            "sector's usual risk level."
        )
    pair_tests = inference.cross_sector_bootstrap
    if not pair_tests.empty:
        valid = pair_tests.dropna(subset=["p_holm"])
        for metric, label in (
            ("annualized_volatility_pct", "annualized-volatility"),
            ("garch_t_persistence", "GARCH-t persistence"),
        ):
            eligible = valid.loc[valid["metric"].eq(metric)]
            evidence = []
            for block in (20, 40):
                at_block = eligible.loc[eligible["block_sessions"].eq(block)]
                n_reject = at_block["p_holm"].lt(alpha).sum()
                evidence.append(f"{n_reject} of {len(at_block)} at block length {block}")
            lines.append(
                f"\nFor {label}, the bootstrap rejected {' and '.join(evidence)} after Holm correction "
                f"at {alpha:.0%}. It uses matched training observations. Confidence intervals and "
                "the finite simulation resolution are in `07_cross_sector_bootstrap.csv`."
            )
    dm = inference.diebold_mariano
    if not dm.empty:
        qlike = dm.loc[dm["loss"].eq("QLIKE")].dropna(subset=["p_holm"])
        n_reject = qlike["p_holm"].lt(alpha).sum()
        lines.append(
            f"\nThe QLIKE Diebold–Mariano tests rejected {n_reject} of {len(qlike)} comparisons after "
            "Holm adjustment. The QLIKE score rankings summarize observed performance. "
            "Use the tests to assess the uncertainty around those rankings."
        )
    subsamples = inference.subsample_robustness
    if not subsamples.empty:
        persistent = subsamples.loc[
            subsamples["specification"].eq("GARCH-t")
            & subsamples["status"].eq("Fit without mean reversion")
        ]
        if not persistent.empty:
            labels = ", ".join(
                f"{row.sector} in {row.period}" for row in persistent.itertuples(index=False)
            )
            lines.append(
                f"\nThe retrospective GARCH-t fits flagged a lack of mean reversion for {labels}. "
                "These fits have no finite half-life under the reported specification."
            )
    all_bt = backtest.backtests
    if not all_bt.empty:
        row_count = len(all_bt)
        default99 = all_bt.loc[all_bt["confidence_pct"].eq(99)]
        rejected = default99["dq_p_mc"].lt(alpha).sum()
        available = default99["dq_p_mc"].notna().sum()
        lines.append(
            f"\nThe risk evaluation contains {row_count} sector–model–confidence results. "
            f"At 99%, the lagged-hit DQ test rejected {rejected} of {available} testable paths at "
            f"{alpha:.0%} before correction. With sparse tail hits, read the DQ, coverage and ES "
            "checks together with loss scores and their domain coverage."
        )

    lines.extend([
        "",
        "## Interpreting the findings",
        "",
        "A sector's volatility level, the persistence of a shock and the length of high-risk spells are "
        "distinct results. Use the paired bootstrap for cross-sector differences, the subperiod table for "
        "regime sensitivity and the forecast comparisons for performance in the holdout. The five baskets "
        "differ in ICB breadth and historical weights are unavailable. A mechanism remains a hypothesis "
        "until evidence tied to that mechanism is tested against dated sector results.",
        "",
        "[Economic context, practical implications and sources](../economic_context.md)",
        "",
    ])
    findings = "\n".join(lines)
    (output.directory / "research_findings.md").write_text(findings, encoding="utf-8")
    return findings
