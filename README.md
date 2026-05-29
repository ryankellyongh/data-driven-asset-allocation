# data-driven-asset-allocation

# Data-Driven Asset Allocation Strategies

Open Avenues Foundation — Quantitative Finance Track  
Mentor: Flora Li, Finance Fellow

---

## Research Question

Can Kelly Criterion-based capital allocation improve performance in a momentum-driven investment strategy compared to classical allocation methods?

Momentum investing assumes that assets with strong recent performance continue to outperform in the near term. This project tests whether applying the Kelly Criterion to weight those momentum signals — rather than allocating naively — produces a more risk-adjusted outcome.

---

## Project Structure

```
├── Build_Project_2025_Fall_Final_Project.ipynb
├── README.md
└── data/                  # Downloaded via yfinance at runtime
```

---

## Asset Universe

15 publicly traded securities spanning multiple sectors:

| Ticker | Ticker | Ticker |
|--------|--------|--------|
| SPY    | TSLA   | MRNA   |
| GOOG   | GS     | PFE    |
| AMZN   | NVDA   | XOM    |
| MSFT   | BAC    | CVX    |
| PG     | VZ     | BA     |

Data source: Yahoo Finance via `yfinance`, starting 2015-01-01. Prices are sampled at the hourly frequency. The annual risk-free rate is set to 4.10%, converted to an equivalent hourly rate.

---

## Methodology

### 1. Exploratory Data Analysis

Examined the statistical properties of hourly excess returns across all 15 assets — distributions, volatility differences, and cross-asset relationships — before constructing any strategy.

### 2. Momentum Strategy

The momentum factor is computed as the difference between each stock's current price and its 21-trading-day (one month) average price. Stocks with a positive factor are flagged as buys; stocks with a negative factor are flagged as sells. Factor performance is tracked over the full sample period.

### 3. Portfolio Allocation Methods

Six allocation methods are implemented inside a single `getweight(N, T, M, R, method)` function and evaluated side by side:

| Method | Description |
|--------|-------------|
| `ew` | Equal Weight — uniform 1/N allocation |
| `mv` | Mean-Variance — maximizes Sharpe ratio subject to long-only and full-investment constraints |
| `u-mv` | Unconstrained Mean-Variance — relaxes the long-only constraint |
| `min` | Minimum Variance — minimizes portfolio variance |
| `pt` | Parametric — shrinkage-based allocation |
| `kelly` | Kelly Criterion — maximizes expected log-utility via convex optimization (`cvxpy`) |

The Kelly Criterion is solved using `cvxpy`: weights are constrained to be non-negative and sum to at most 1 (long-only, no leverage in the base case), and the objective is to maximize the sum of log portfolio returns over the rolling window.

### 4. Strategy Evaluation

Methods are compared across in-sample and out-of-sample periods using a rolling window of 120 observations. Metrics computed for each method:

- In-Sample Sharpe Ratio (ISR) and Out-of-Sample Sharpe Ratio (OSR)
- Turnover
- Sortino Ratio
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Maximum Drawdown and Calmar Ratio
- Omega Ratio

### 5. Kelly Portfolio Deep Dive

The best-performing Kelly allocation is analyzed in detail against the S&P 500 benchmark:

| Metric | Description |
|--------|-------------|
| Annual Returns | Annualized portfolio return |
| Annual Volatility | Annualized standard deviation of daily returns |
| Sharpe Ratio | Excess return per unit of total risk |
| Sortino Ratio | Excess return per unit of downside risk |
| Beta | Sensitivity to S&P 500 |
| Treynor Ratio | Excess return per unit of systematic risk |
| Information Ratio | Active return relative to tracking error |
| Skewness | Asymmetry of the return distribution |
| Excess Kurtosis | Tail weight relative to a normal distribution |
| Maximum Drawdown | Largest peak-to-trough decline |

A 2x leveraged version of the Kelly portfolio is also plotted against the S&P 500 to explore the effect of leverage on cumulative returns.

---

## Libraries

```
yfinance
pandas
numpy
matplotlib
scipy
cvxpy
prettytable
tabulate
```

Install all dependencies:

```bash
pip install yfinance pandas numpy matplotlib scipy cvxpy prettytable tabulate
```

---

## How to Run

1. Clone the repository.
2. Install dependencies (see above).
3. Open `Build_Project_2025_Fall_Final_Project.ipynb` in Jupyter Notebook or JupyterLab.
4. Run all cells sequentially. Price data is downloaded automatically from Yahoo Finance.

---

## Acknowledgments

Built as part of the Open Avenues Foundation Data-Driven Asset Allocation project.
