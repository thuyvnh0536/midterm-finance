# Output 1 · FiinProX data sources and construction

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
