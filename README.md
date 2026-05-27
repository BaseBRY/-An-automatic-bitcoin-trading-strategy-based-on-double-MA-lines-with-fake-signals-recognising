## Overview

A quantitative trading strategy for BTC/USDT that combines **dual Exponential Moving Average (EMA) crossovers** with a **two-layer fake signal filtering system**.

The core insight: standard MA crossover strategies suffer from excessive false signals in ranging (non-trending) markets. This project addresses that with two independent filters that must *both* pass before a trade is entered.

### Strategy Logic

```
Entry Condition — ALL three must be satisfied:
  ① Golden Cross  : Fast EMA crosses above Slow EMA
  ② Volume Filter : Crossover-day volume > N-day avg × threshold
                    (low-volume crossovers = unconvincing breakouts)
  ③ ATR Filter    : ATR / Close price > minimum ratio
                    (tight consolidation zones → skip)

Exit Condition — ANY one triggers:
  ① Death Cross   : Fast EMA crosses below Slow EMA
  ② Trend Break   : Close price falls below Slow EMA
```

---

## Why Two Filters?

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| MA lines weave back and forth in flat markets | Consolidation → low ATR | **ATR / Price ratio filter** rejects signals when market is not trending |
| Breakout on thin volume reverses quickly | No institutional participation | **Volume threshold filter** requires volume ≥ N× rolling average at crossover |

The combination significantly reduces the number of trades compared to a plain dual-MA strategy, while aiming to improve the quality of each trade taken.

---

## Project Structure

```
.
├── strategy.py          # Core strategy: fetch → indicators → signals → backtest → chart
├── requirements.txt     # Python dependencies
├── backtest_report.png  # Output chart (generated on run)
└── README.md
```

---

## Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/BaseBRY/-An-automatic-bitcoin-trading-strategy-based-on-double-MA-lines-with-fake-signals-recognising.git
cd 
pip install -r requirements.txt
```

### 2. (Optional) Set proxy for restricted networks

```bash
# If direct access to Binance API is unavailable in your region:
export PROXY_URL=http://127.0.0.1:7890
```

### 3. Run the backtest

```bash
python strategy.py
```

### 4. View output

- **Console**: Full performance report printed to terminal
- **Chart**: `backtest_report.png` — three-panel visualization (equity curve, drawdown, volume)

---

## Configuration

All parameters are in the `Config` dataclass at the top of `strategy.py`. No external config file needed.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_span` | `20` | Fast EMA period |
| `slow_span` | `60` | Slow EMA period |
| `volume_multiplier` | `1.5` | Volume must be ≥ this × rolling average |
| `volume_ma_period` | `20` | Rolling window for average volume |
| `atr_period` | `14` | ATR calculation period |
| `atr_min_ratio` | `0.018` | ATR/Price minimum (below = consolidation) |
| `commission` | `0.0004` | Per-trade commission (Binance taker) |
| `slippage` | `0.0005` | Estimated slippage per trade |
| `risk_free_rate` | `0.03` | Annual risk-free rate (Sharpe/Sortino) |
| `limit` | `1000` | Candles to fetch (Binance max) |

---

## Performance Metrics

The backtest outputs the following metrics, compared against a BTC buy-and-hold benchmark:

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative return over the full period |
| **Annualized Return** | CAGR equivalent |
| **Annualized Volatility** | Std dev of daily returns × √365 |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Sharpe Ratio** | Excess return per unit of total risk |
| **Sortino Ratio** | Excess return per unit of *downside* risk only |
| **Calmar Ratio** | Annualized return / Max drawdown |
| **Win Rate** | % of closed trades with positive P&L |
| **Total Trades** | Number of completed round trips |

---

## Output Chart (3 Panels)

```
┌──────────────────────────────────────────────────┐
│ Panel 1 │ Equity curve: Strategy vs BTC Benchmark │
│         │ Entry markers (▲) and blocked signals (×)│
├──────────────────────────────────────────────────┤
│ Panel 2 │ Drawdown area: Strategy vs Benchmark    │
├──────────────────────────────────────────────────┤
│ Panel 3 │ Daily volume + threshold line           │
│         │ Valid entry days and blocked days marked │
└──────────────────────────────────────────────────┘
```

---

## Limitations & Disclaimer

- **Backtesting only.** This project does not connect to any live trading API and cannot place real orders.
- Past performance does not guarantee future results.
- Cryptocurrency trading carries significant risk. This code is for educational and research purposes only.
- Backtest results are subject to look-ahead bias mitigation (signals use `shift(1)`) and include transaction costs, but real-world execution may differ.

---

## Dependencies

See `requirements.txt`. Core libraries:

- `pandas` — data manipulation and time series
- `numpy` — numerical computation
- `requests` — HTTP client for Binance API
- `matplotlib` — charting and visualization

---

## Author

**[Your Name]**  
Electrical Engineering & Automation, Shanghai Institute of Technology  
GitHub: [@BaseBRY](https://github.com/BaseBRY)

---

## License

MIT License — free to use, modify, and distribute with attribution.
