import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
matplotlib.use('Agg')


def run_all_weather_v28_enhanced():
    # --- 0. 获取数据 (Data Fetching) ---
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000}
    headers = {'User-Agent': 'Mozilla/5.0'}
    proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}  # 请根据你的本地网络情况调整

    try:
        res = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=30, verify=False)
        df = pd.DataFrame(res.json(),
                          columns=['Time', 'Open', 'High', 'Low', 'Close', 'Vol', 'T2', 'Q', 'Trades', 'TB', 'TQ', 'I'])
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        df.set_index('Time', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Vol']].astype(float)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # --- 1. 复合指标计算 (Indicator Calculation) ---
    df['EMA200'] = df['Close'].ewm(span=200).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Panic_Lower'] = df['MA20'] - (2.5 * df['Std'])

    # --- 2. 混合逻辑 (Hybrid Logic: Trend + Mean Reversion) ---
    df['Position'] = 0

    # 进场逻辑
    trend_long = (df['Close'] > df['EMA200']) & (df['Close'] > df['MA20'])
    panic_buy = (df['Close'] < df['Panic_Lower'])
    df.loc[trend_long | panic_buy, 'Position'] = 1

    # 离场逻辑
    df['Position'] = df['Position'].replace(0, np.nan).ffill().fillna(0)
    df.loc[(df['Close'] < df['EMA200']) & (df['Close'] < df['MA20']), 'Position'] = 0

    # --- 3. 真实交易绩效计算 (Realistic Performance Metrics) ---
    commission = 0.0004  # 手续费
    slippage = 0.0005  # 新增：模拟交易滑点损耗 (Slippage cost)
    total_friction = commission + slippage

    df['Market_Ret'] = df['Close'].pct_change()
    # 净收益 = 策略收益 - 调仓时的摩擦成本 (Friction costs)
    df['Net_Ret'] = (df['Position'].shift(1) * df['Market_Ret']) - (df['Position'].diff().abs() * total_friction)

    df['Cum_Strategy'] = (1 + df['Net_Ret'].fillna(0)).cumprod()
    df['Cum_Market'] = (1 + df['Market_Ret'].fillna(0)).cumprod()

    # --- 4. 深度量化指标 (Advanced Quantitative Metrics) ---
    trading_days_per_year = 365  # 加密货币全年无休
    risk_free_rate = 0.03  # 假设无风险利率为3%

    # 策略年化收益与波动率 (Annualized Return & Volatility)
    strat_ann_ret = df['Net_Ret'].mean() * trading_days_per_year
    strat_ann_vol = df['Net_Ret'].std() * np.sqrt(trading_days_per_year)

    # 基准年化收益与波动率 (Benchmark metrics)
    bench_ann_ret = df['Market_Ret'].mean() * trading_days_per_year
    bench_ann_vol = df['Market_Ret'].std() * np.sqrt(trading_days_per_year)

    # 最大回撤 (Maximum Drawdown)
    df['Drawdown_Strategy'] = (df['Cum_Strategy'] - df['Cum_Strategy'].cummax()) / df['Cum_Strategy'].cummax()
    df['Drawdown_Market'] = (df['Cum_Market'] - df['Cum_Market'].cummax()) / df['Cum_Market'].cummax()
    max_dd_strat = df['Drawdown_Strategy'].min()
    max_dd_bench = df['Drawdown_Market'].min()

    # 夏普比率 (Sharpe Ratio)
    sharpe_ratio = (strat_ann_ret - risk_free_rate) / strat_ann_vol if strat_ann_vol != 0 else 0

    # 索提诺比率 (Sortino Ratio) - 仅计算下行风险 (Downside risk)
    downside_returns = df.loc[df['Net_Ret'] < 0, 'Net_Ret']
    downside_vol = downside_returns.std() * np.sqrt(trading_days_per_year)
    sortino_ratio = (strat_ann_ret - risk_free_rate) / downside_vol if downside_vol != 0 else 0

    # 卡玛比率 (Calmar Ratio)
    calmar_ratio = strat_ann_ret / abs(max_dd_strat) if max_dd_strat != 0 else 0

    # --- 终端输出打印 ---
    print("\n" + "🌟 V28 增强版：全天候混合策略 (Enhanced Hybrid Strategy)".center(60, '='))
    print(f"【收益表现】")
    print(
        f"策略总收益 (Total Return): {(df['Cum_Strategy'].iloc[-1] - 1) * 100:>8.2f}%  |  年化 (Ann. Ret): {strat_ann_ret * 100:.2f}%")
    print(
        f"基准总收益 (Bench Return): {(df['Cum_Market'].iloc[-1] - 1) * 100:>8.2f}%  |  年化 (Ann. Ret): {bench_ann_ret * 100:.2f}%")
    print(f"\n【风险控制】")
    print(f"策略最大回撤 (Max Drawdown): {max_dd_strat * 100:>8.2f}%")
    print(f"基准最大回撤 (Bench Max DD): {max_dd_bench * 100:>8.2f}%")
    print(f"\n【风险调整后收益】 (Risk-Adjusted Metrics)")
    print(f"夏普比率 (Sharpe Ratio):    {sharpe_ratio:>8.2f}")
    print(f"索提诺比率 (Sortino Ratio):   {sortino_ratio:>8.2f}")
    print(f"卡玛比率 (Calmar Ratio):    {calmar_ratio:>8.2f}")
    print("=" * 66)

    # --- 5. 双子图可视化 (Dual-Axis Visualization) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.1)

    # 主图：净值曲线 (Equity Curve)
    ax1.plot(df['Cum_Strategy'], label=f'Hybrid Strategy (Ann. Ret: {strat_ann_ret * 100:.1f}%)', color='#2980B9', lw=2)
    ax1.plot(df['Cum_Market'], label=f'BTC Benchmark (Ann. Ret: {bench_ann_ret * 100:.1f}%)', color='gray', alpha=0.4,
             lw=1.5)
    ax1.set_title('All-Weather Hybrid Strategy V28: Performance & Drawdown Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (Multiplier)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 副图：动态回撤区 (Drawdown Area)
    ax2.fill_between(df.index, df['Drawdown_Strategy'] * 100, 0, color='#E74C3C', alpha=0.5, label='Strategy Drawdown')
    ax2.plot(df.index, df['Drawdown_Market'] * 100, color='gray', alpha=0.4, label='BTC Drawdown', lw=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.savefig('strategy_v28_enhanced_report.png', bbox_inches='tight', dpi=300)
    print("\n✅ 回测报告图表已保存为 'strategy_v28_enhanced_report.png'")


if __name__ == "__main__":
    run_all_weather_v28_enhanced()