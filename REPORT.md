# Belief Dispersion Interval Forecasting (Offline Synthetic Study)

Because outbound network access is blocked in this environment, the workflow generates a synthetic S&P 500 (^GSPC) price path and VIX-like uncertainty proxy rather than downloading live market data. The code in `analysis.py` mirrors the requested end-to-end process using only the Python standard library: feature engineering on returns/volatility/VIX proxy, rolling-window training, interval estimation, and simple trading heuristics.

## Key Outputs
- Metrics and ASCII interval traces are written under `outputs/` for horizons h=1 and h=5, feature sets with/without the VIX proxy, and two model families (AR1 and bagged linear ensemble).
- `outputs/summary.txt` provides consolidated MAE/RMSE, interval coverage, interval scores, and trading statistics for point- versus interval-based strategies.

## Notable Findings from the Synthetic Experiment
- Interval coverage hovered near the 90% target for h=1 and slightly below for h=5, showing the residual-quantile construction is reasonably calibrated on the simulated series.
- Interval-aware strategies sat mostly in cash (intervals straddle zero frequently), preserving capital versus the point-only strategies that suffered drawdowns on noisy forecasts.
- Adding the VIX-like proxy had muted effects in this synthetic setting; coverage and interval scores moved only marginally relative to the base feature set.
- Bagged linear models offered small stability gains in cumulative return versus single linear fits for h=1, while performance differences were minor for h=5.

These outputs are placeholders illustrating the full workflow structure. Substituting real S&P 500 and VIX data will allow the same pipeline to answer the research questions with empirical evidence once network access is available.
