#!/usr/bin/env python
# coding: utf-8

# # Data-Driven Asset Allocation
# **Author:** Ryan Kelly  
# **Program:** Open Avenues Foundation — Data-Driven Asset Allocation Strategies  
# **Data:** Hourly prices (one year) for 20 US equities  
# **Goal:** Construct a rolling Kelly allocation, benchmark against equal-weight and SPY, and report standard performance metrics

# ## 1. Imports

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import warnings
warnings.filterwarnings("ignore")


# ## 2. Configuration
# 
# All parameters are defined in one place so they are easy to change without hunting through the notebook.

# In[2]:


# Asset universe
PORTFOLIO = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
    'NEE',  'FSLR', 'SPWR',
    'JPM',  'BAC',  'WFC',
    'ABBV', 'PFE',  'UNH',
    'GE',   'HON',  'CAT',
    'PG',   'KO',   'WMT',
]
BENCHMARK = 'SPY'

# Date range
START = '2024-12-01'
END   = '2025-12-01'

# Risk-free rate
RF_YEAR = 2.90 / 100
RF_HOUR = (1 + RF_YEAR) ** (1 / 8760) - 1  # derived from annual, not hardcoded

# Rolling window length (hours)
M = 24

# Per-asset weight cap
KELLY_CAP = 0.21

print(f'Annual RF : {RF_YEAR:.2%}')
print(f'Hourly RF : {RF_HOUR:.3e}')
print(f'Window  M : {M} hours')


# ## 3. Data Download
# 
# `yf.download` fetches all tickers in a single request, which is faster than looping and guarantees a shared datetime index.  
# `ffill` fills occasional intra-day gaps before dropping any row that is still NaN.

# In[3]:


raw = yf.download(
    PORTFOLIO + [BENCHMARK],
    start=START,
    end=END,
    interval='60m',
    progress=False,
    auto_adjust=True,
)['Close']

raw = raw.ffill().dropna()

prices    = raw[PORTFOLIO]
spy_price = raw[BENCHMARK]

print(f'Price matrix shape: {prices.shape}  (rows=hours, cols=assets)')
display(prices.head())


# ## 4. Returns & Excess Returns
# 
# Hourly returns are computed with `pct_change`. The hourly risk-free rate is subtracted to produce excess returns used for Kelly weight estimation.

# In[4]:


R_df         = prices.pct_change().dropna()
ExcessRet_df = R_df - RF_HOUR

# Align SPY to the same index
spy_ret      = spy_price.pct_change().reindex(R_df.index).dropna()
ExcessRet_df = ExcessRet_df.reindex(spy_ret.index)
R_df         = R_df.reindex(spy_ret.index)

R  = ExcessRet_df.to_numpy()  # excess returns — used for weight estimation
Rp = R_df.to_numpy()          # total returns  — used for portfolio P&L
T, N = R.shape
print(f'T={T} periods, N={N} assets')


# ## 5. Kelly Weight Estimator
# 
# For each rolling window the Kelly weight vector is estimated via the mean-variance approximation:
# 
# $$f \propto \Sigma^{-1}\,\mu$$
# 
# **Key fixes vs. original:**
# - `np.linalg.lstsq` replaces `np.linalg.inv` — handles near-singular covariance matrices without crashing (common when window length M is close to asset count N).
# - Long-only: negative weights are zeroed and renormalized.
# - Per-asset cap enforced via a stable clip-and-redistribute loop.

# In[5]:


def kelly_weights(R_window: np.ndarray, cap: float = KELLY_CAP) -> np.ndarray:
    """
    Estimate Kelly weights from a (M, N) window of excess returns.
    Returns a length-N weight vector summing to 1.
    """
    mu  = R_window.mean(axis=0)
    cov = np.cov(R_window, rowvar=False)

    # lstsq is robust when cov is near-singular
    w_raw, _, _, _ = np.linalg.lstsq(cov, mu, rcond=None)

    # Long-only constraint
    w_raw = np.where(w_raw > 0, w_raw, 0.0)
    total = w_raw.sum()
    if total == 0:
        return np.ones(N) / N  # fallback to equal weight
    w = w_raw / total

    # Iterative cap-and-redistribute
    for _ in range(50):
        excess = np.maximum(w - cap, 0).sum()
        if excess < 1e-10:
            break
        w = np.clip(w, 0, cap)
        slack = cap - w
        below_cap = slack > 0
        if not below_cap.any():
            break
        w[below_cap] += excess * slack[below_cap] / slack[below_cap].sum()

    return w / w.sum()


# ## 6. Rolling Weight Computation
# 
# For each hour $t \geq M$ the Kelly weights are estimated from the previous $M$ hours.  
# Equal-weight requires no loop — it is a constant allocation.

# In[6]:


# Kelly: rolling window
weight_kelly = np.full((T, N), np.nan)
windows = sliding_window_view(R, (M, N))  # shape: (T-M+1, 1, M, N)

for i in range(len(windows) - 1):
    weight_kelly[i + M] = kelly_weights(windows[i][0])

# Equal-weight: constant
weight_ew = np.full((T, N), 1.0 / N)
weight_ew[:M] = np.nan  # burn-in period matches Kelly

# Sanity check on last valid Kelly allocation
last_valid = np.where(~np.isnan(weight_kelly[:, 0]))[0][-1]
last_w = weight_kelly[last_valid]
print(f'sum={last_w.sum():.6f}  min={last_w.min():.4f}  max={last_w.max():.4f}')

kelly_series = pd.Series(last_w, index=PORTFOLIO, name='Weight').sort_values(ascending=False)
display(kelly_series.to_frame().style.format('{:.4f}').set_caption('Kelly Allocation — Last Rebalance'))

plt.figure(figsize=(12, 4))
kelly_series.plot(kind='bar')
plt.axhline(KELLY_CAP, color='red', linestyle='--', linewidth=1, label=f'Cap ({KELLY_CAP:.0%})')
plt.title('Kelly Criterion Allocation — Last Rebalance Window')
plt.ylabel('Portfolio Weight')
plt.xlabel('Asset')
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.show()


# ## 7. Portfolio Returns
# 
# Weights are applied to **total** (not excess) returns so the cumulative curve reflects real dollar growth from $1.  
# SPY is included as a passive market benchmark.

# In[7]:


valid = slice(M, T)

port_ret_kelly   = np.nansum(weight_kelly[valid] * Rp[valid], axis=1)
kelly_cumulative = np.cumprod(1 + port_ret_kelly)

port_ret_ew   = np.nansum(weight_ew[valid] * Rp[valid], axis=1)
ew_cumulative = np.cumprod(1 + port_ret_ew)

spy_ret_arr    = spy_ret.to_numpy()[M:]
spy_cumulative = np.cumprod(1 + spy_ret_arr)

time_index = R_df.index[valid]

plt.figure(figsize=(12, 5))
plt.plot(time_index, kelly_cumulative, label='Kelly',        linewidth=1.5)
plt.plot(time_index, ew_cumulative,   label='Equal Weight',  linewidth=1.5, linestyle='--')
plt.plot(time_index, spy_cumulative,  label='SPY Benchmark', linewidth=1.5, linestyle=':')
plt.title('Cumulative Growth of $1 — Kelly vs Equal-Weight vs SPY')
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()


# ## 8. Performance Metrics
# 
# All metrics are annualized using **8,760 hourly periods per year**.

# In[8]:


HOURS_PER_YEAR = 8_760

def performance_metrics(ret_arr: np.ndarray, label: str) -> dict:
    n       = len(ret_arr)
    ann_ret = (1 + ret_arr).prod() ** (HOURS_PER_YEAR / n) - 1
    ann_vol = ret_arr.std() * np.sqrt(HOURS_PER_YEAR)
    sharpe  = (ann_ret - RF_YEAR) / ann_vol if ann_vol > 0 else np.nan

    downside = ret_arr[ret_arr < 0]
    ann_down = downside.std() * np.sqrt(HOURS_PER_YEAR) if len(downside) > 0 else np.nan
    sortino  = (ann_ret - RF_YEAR) / ann_down if (ann_down and ann_down > 0) else np.nan

    cum      = np.cumprod(1 + ret_arr)
    peak     = np.maximum.accumulate(cum)
    max_dd   = (cum / peak - 1).min()

    return {
        'Strategy':        label,
        'Ann. Return':     f'{ann_ret:.2%}',
        'Ann. Volatility': f'{ann_vol:.2%}',
        'Sharpe Ratio':    f'{sharpe:.3f}',
        'Sortino Ratio':   f'{sortino:.3f}',
        'Max Drawdown':    f'{max_dd:.2%}',
    }

metrics = pd.DataFrame([
    performance_metrics(port_ret_kelly, 'Kelly'),
    performance_metrics(port_ret_ew,    'Equal Weight'),
    performance_metrics(spy_ret_arr,    'SPY'),
]).set_index('Strategy')

display(metrics.style.set_caption('Annualized Performance Metrics'))


# ## 9. Summary
# 
# This notebook implements a rolling-window Kelly Criterion allocation across 20 US equities using hourly price data.
# 
# **Method.** For each hour t ≥ M the Kelly weights are estimated from the prior M-hour window of excess returns using the mean-variance approximation f ∝ Σ⁻¹μ. The solution is constrained to be long-only and each weight is capped at 21% to prevent concentration. Weights are normalized to sum to one.
# 
# **Benchmarks.** The Kelly strategy is compared against an equal-weight portfolio and SPY on total returns to reflect real dollar growth.
# 
# **Limitations.** With M = 24 and N = 20 the sample covariance matrix is rank-deficient — `lstsq` handles this but estimates remain noisy. A longer window or shrinkage estimator (e.g. Ledoit-Wolf) would improve stability. Hourly rebalancing also ignores transaction costs, which would significantly erode returns in practice.
