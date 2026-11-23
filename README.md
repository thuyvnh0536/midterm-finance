# Group 10 midterm T8

## Reproduce the analysis

Extract the complete ZIP, open a terminal in that directory, and use Python 3.12.

```bash
python -m pip install -r requirements.txt
python run_analysis.py
```

The notebook `midterm_template.ipynb` uses the same Python modules as the command. It includes the seven outputs, methodology, research findings, limitations and references. Use JupyterLab or VS Code to run all cells in order. The packaged CSVs and PNGs are under `outputs_sector_volatility/`. The five original exports under `data/raw/` are the only input time series. The analysis works offline after dependency installation.

For a development smoke run, pass `--bootstrap-reps 39 --output-dir scratch_results`. For the research results, use the default 399 bootstrap draws. At block length 40 the volatility tests also use 399 draws, and the GARCH-t persistence sensitivity uses 199 refits. Allow a few minutes for all model and bootstrap fits. On systems using multithreaded OpenBLAS, running with a single BLAS thread can reduce overhead from repeatedly fitting small time-series models.

## How the files fit the outline

| Output | Module and main results |
|---|---|
| 1 | `data_processing.py` · price extraction, aligned log/absolute/squared returns, provenance, date coverage and quality audit |
| 2 | `data_processing.py` · descriptive statistics, distribution plots, stationarity, serial dependence, clustering, asymmetry and aggregation |
| 3 | `time_series.py` · conditional-mean selection and innovation diagnostics |
| 4 | `time_series.py` · rolling SD, EWMA and univariate GARCH family, persistence and half-life |
| 5 | `sector_interactions.py` · VAR, Granger, impulse responses, FEVD, CCC/DCC-GARCH |
| 6 | `risk_analysis.py` · rolling one-session VaR and ES at 95% and 99%, benchmarks, risk ranking and high-risk spells |
| 7 | `time_series.py`, `risk_analysis.py` and `inference.py` · residual diagnostics, volatility losses, DM, backtesting, cross-sector bootstrap and robustness |

`OUTLINE_AUDIT.md` compares the previous ZIP with the Group 10 outline. `economic_context.md` provides cited context. `outputs_sector_volatility/research_findings.md` connects measured results to the study questions. The analysis uses five sectors in line with the research question and supplied workbooks, including the source label `Consumer Goods L1`.
"# midterm-finance" 
"# midterm-finance" 
