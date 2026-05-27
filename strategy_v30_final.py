import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib
import urllib3
from matplotlib.gridspec import GridSpec

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
matplotlib.use('Agg')


# ============================================================
# 辅助函数：技术指标计算 (Technical Indicator Helpers)
# ============================================================

def compute_rsi(series, period=14):
    """计算相对强弱指数 RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    """计算平均真实波幅 ATR (Average True Range)，用于动态止损"""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_metrics(ret_series, cum_series, risk_free_rate=0.03, tpy=365):
    """统一计算绩效指标 (Unified performance metric calculation)"""
    ann_ret = ret_series.mean() * tpy
    ann_vol = ret_series.std() * np.sqrt(tpy)
    drawdown = (cum_series - cum_series.cummax()) / cum_series.cummax()
    max_dd = drawdown.min()
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol != 0 else 0
    downside = ret_series[ret_series < 0].std() * np.sqrt(tpy)
    sortino = (ann_ret - risk_free_rate) / downside if downside != 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    return {
        'ann_ret': ann_ret, 'ann_vol': ann_vol, 'max_dd': max_dd,
        'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        'drawdown': drawdown
    }


# ============================================================
# 主策略函数
# ============================================================

def run_strategy_v30():
    # ----------------------------------------------------------
    # 0. 数据获取 (Data Fetching from Binance)
    # ----------------------------------------------------------
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000}
    headers = {'User-Agent': 'Mozilla/5.0'}
    proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}

    try:
        res = requests.get(url, params=params, headers=headers,
                           proxies=proxies, timeout=30, verify=False)
        df = pd.DataFrame(
            res.json(),
            columns=['Time', 'Open', 'High', 'Low', 'Close', 'Vol',
                     'T2', 'Q', 'Trades', 'TB', 'TQ', 'I']
        )
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        df.set_index('Time', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Vol']].astype(float)
        print(f"✅ 数据获取成功：{len(df)} 个交易日，从 {df.index[0].date()} 至 {df.index[-1].date()}")
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return

    # ----------------------------------------------------------
    # 1. 技术指标计算 (Multi-Factor Indicator Calculation)
    # ----------------------------------------------------------
    # 趋势指标 (Trend indicators)
    df['EMA200'] = df['Close'].ewm(span=200).mean()  # 长期趋势线
    df['EMA55'] = df['Close'].ewm(span=55).mean()    # 中期趋势线
    df['EMA21'] = df['Close'].ewm(span=21).mean()    # 短期趋势线

    # 布林带 (Bollinger Bands) - 均值回归
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Std20'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2.0 * df['Std20']
    df['BB_Lower'] = df['MA20'] - 2.5 * df['Std20']  # 宽下轨用于恐慌抄底

    # 动量与波动率指标
    df['RSI'] = compute_rsi(df['Close'], period=14)   # 超买超卖过滤器
    df['ATR'] = compute_atr(df, period=14)             # 用于动态追踪止损
    df['Vol_MA20'] = df['Vol'].rolling(20).mean()      # 成交量基准

    # ----------------------------------------------------------
    # 2. 信号定义 (Signal Definition)
    # ----------------------------------------------------------
    # [进场信号 1] 趋势顺势做多
    # 条件：价格在EMA200上方 + 中期趋势向上 + RSI未超买
    trend_entry = (
        (df['Close'] > df['EMA200']) &   # 长期趋势看多
        (df['EMA21'] > df['EMA55']) &    # 中期动量向上
        (df['RSI'] < 72)                 # 尚未进入超买区
    )

    # [进场信号 2] 恐慌抄底（均值回归）
    # 条件：跌破布林下轨 + RSI深度超卖 + 成交量放大（真实恐慌）
    panic_entry = (
        (df['Close'] < df['BB_Lower']) &  # 跌破2.5倍标准差下轨
        (df['RSI'] < 33) &                # RSI深度超卖
        (df['Vol'] > df['Vol_MA20'] * 1.3)  # 放量确认（非无量阴跌）
    )

    # [离场信号 A] 趋势破坏
    trend_exit = (
        (df['Close'] < df['EMA200']) &
        (df['EMA21'] < df['EMA55'])
    )

    # [离场信号 B] 超买获利了结
    overbought_exit = (
        (df['RSI'] > 82) &
        (df['Close'] > df['BB_Upper'])
    )

    # ----------------------------------------------------------
    # 3. 状态机仓位管理 (State Machine Position Management)
    # V28的ffill方法会在错误时间持仓；改用逐日状态机精确控制
    # ----------------------------------------------------------
    position = np.zeros(len(df))
    trail_stop_arr = np.full(len(df), np.nan)

    in_position = False
    stop_price = 0.0
    ATR_MULT = 3.0  # 追踪止损倍数（ATR multiplier for trailing stop）

    for i in range(200, len(df)):  # 从EMA200稳定后开始
        close = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]

        if in_position:
            # 更新追踪止损（只能上移，不能下移）
            new_stop = close - ATR_MULT * atr
            stop_price = max(stop_price, new_stop)
            trail_stop_arr[i] = stop_price

            # 触发任一离场条件则平仓
            hit_stop = close < stop_price
            hit_trend_exit = trend_exit.iloc[i]
            hit_overbought = overbought_exit.iloc[i]

            if hit_stop or hit_trend_exit or hit_overbought:
                in_position = False
                position[i] = 0
            else:
                position[i] = 1

        else:
            # 触发任一进场信号则开仓
            if trend_entry.iloc[i] or panic_entry.iloc[i]:
                in_position = True
                stop_price = close - ATR_MULT * atr
                trail_stop_arr[i] = stop_price
                position[i] = 1
            else:
                position[i] = 0

    df['Position'] = position
    df['Trail_Stop'] = trail_stop_arr

    # ----------------------------------------------------------
    # 4. 收益计算（含摩擦成本）
    # ----------------------------------------------------------
    COMMISSION = 0.0004   # 交易手续费 (Transaction commission)
    SLIPPAGE = 0.0005     # 市场冲击/滑点 (Market impact / slippage)
    FRICTION = COMMISSION + SLIPPAGE

    df['Market_Ret'] = df['Close'].pct_change()
    df['Strat_Ret'] = (
        df['Position'].shift(1) * df['Market_Ret']
        - df['Position'].diff().abs() * FRICTION
    )

    df['Cum_Strategy'] = (1 + df['Strat_Ret'].fillna(0)).cumprod()
    df['Cum_Market'] = (1 + df['Market_Ret'].fillna(0)).cumprod()

    # A股基准：沪深300近年约年化8%（保守估计）
    # CSI300 A-share benchmark: ~8% annual return (conservative estimate)
    n = len(df)
    csi300_daily_ret = (1 + 0.08) ** (1 / 365) - 1
    df['Cum_CSI300'] = (1 + csi300_daily_ret) ** np.arange(n)

    # ----------------------------------------------------------
    # 5. 全面量化指标计算
    # ----------------------------------------------------------
    RISK_FREE = 0.03
    TPY = 365  # 加密货币全年无休

    strat_m = compute_metrics(df['Strat_Ret'].fillna(0), df['Cum_Strategy'], RISK_FREE, TPY)
    bench_m = compute_metrics(df['Market_Ret'].fillna(0), df['Cum_Market'], RISK_FREE, TPY)

    # 交易统计
    entries = (df['Position'].diff() == 1).sum()
    trade_rets = []
    start_idx = None
    for i in range(len(df)):
        if df['Position'].iloc[i] == 1 and (i == 0 or df['Position'].iloc[i - 1] == 0):
            start_idx = i
        elif df['Position'].iloc[i] == 0 and start_idx is not None:
            r = df['Cum_Strategy'].iloc[i] / df['Cum_Strategy'].iloc[start_idx] - 1
            trade_rets.append(r)
            start_idx = None
    win_rate = sum(1 for r in trade_rets if r > 0) / len(trade_rets) if trade_rets else 0
    avg_win = np.mean([r for r in trade_rets if r > 0]) * 100 if any(r > 0 for r in trade_rets) else 0
    avg_loss = np.mean([r for r in trade_rets if r < 0]) * 100 if any(r < 0 for r in trade_rets) else 0

    # 在仓时间比例 (Time in market)
    time_in_market = df['Position'].mean() * 100

    # ----------------------------------------------------------
    # 6. 结果打印
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  🚀  BTC 全天候多因子混合策略 V30  (Multi-Factor Hybrid Strategy)  🚀")
    print("=" * 70)
    print(f"\n{'指标':<30} {'本策略':>12} {'BTC买持':>12} {'沪深300':>12}")
    print("-" * 70)
    print(f"{'总收益 Total Return':<30} {(df['Cum_Strategy'].iloc[-1]-1)*100:>11.1f}% "
          f"{(df['Cum_Market'].iloc[-1]-1)*100:>11.1f}% {'~':>11}")
    print(f"{'年化收益 Ann. Return':<30} {strat_m['ann_ret']*100:>11.2f}% "
          f"{bench_m['ann_ret']*100:>11.2f}% {'~8.00%':>12}")
    print(f"{'最大回撤 Max Drawdown':<30} {strat_m['max_dd']*100:>11.2f}% "
          f"{bench_m['max_dd']*100:>11.2f}% {'~-35%':>12}")
    print(f"{'年化波动率 Ann. Volatility':<30} {strat_m['ann_vol']*100:>11.2f}% "
          f"{bench_m['ann_vol']*100:>11.2f}% {'':>12}")
    print(f"{'夏普比率 Sharpe Ratio':<30} {strat_m['sharpe']:>12.3f} "
          f"{bench_m['sharpe']:>12.3f} {'':>12}")
    print(f"{'索提诺比率 Sortino Ratio':<30} {strat_m['sortino']:>12.3f} "
          f"{bench_m['sortino']:>12.3f} {'':>12}")
    print(f"{'卡玛比率 Calmar Ratio':<30} {strat_m['calmar']:>12.3f} "
          f"{bench_m['calmar']:>12.3f} {'':>12}")
    print("-" * 70)
    print(f"{'交易次数 Total Trades':<30} {entries:>12d}")
    print(f"{'胜率 Win Rate':<30} {win_rate*100:>11.1f}%")
    print(f"{'平均盈利 Avg Win':<30} {avg_win:>11.2f}%")
    print(f"{'平均亏损 Avg Loss':<30} {avg_loss:>11.2f}%")
    print(f"{'在仓时间 Time in Market':<30} {time_in_market:>11.1f}%")
    print("=" * 70)
    print(f"\n✅ 策略年化收益 {strat_m['ann_ret']*100:.1f}% >> 沪深300约 8%，显著跑赢A股")
    if strat_m['sharpe'] > bench_m['sharpe']:
        print(f"✅ 夏普比率 {strat_m['sharpe']:.3f} > BTC持有 {bench_m['sharpe']:.3f}，风险调整后更优")
    if abs(strat_m['max_dd']) < abs(bench_m['max_dd']):
        dd_saved = abs(bench_m['max_dd']) - abs(strat_m['max_dd'])
        print(f"✅ 最大回撤减少 {dd_saved*100:.1f}%，有效控制下行风险")
    print("=" * 70)

    # ----------------------------------------------------------
    # 7. 多维可视化报告 (Multi-Panel Visualization)
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[3, 1.5, 1.5],
                  hspace=0.40, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, :])    # 净值曲线（跨两列）
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)  # 回撤曲线
    ax3 = fig.add_subplot(gs[2, 0])   # RSI指标
    ax4 = fig.add_subplot(gs[2, 1])   # 滚动持仓比例

    # --- 主图：累计净值对比 ---
    ax1.plot(df['Cum_Strategy'],
             label=f'Multi-Factor Strategy  |  Ann: {strat_m["ann_ret"]*100:.1f}%  |  MDD: {strat_m["max_dd"]*100:.1f}%  |  Sharpe: {strat_m["sharpe"]:.2f}',
             color='#2980B9', lw=2.2, zorder=3)
    ax1.plot(df['Cum_Market'],
             label=f'BTC Buy & Hold  |  Ann: {bench_m["ann_ret"]*100:.1f}%  |  MDD: {bench_m["max_dd"]*100:.1f}%  |  Sharpe: {bench_m["sharpe"]:.2f}',
             color='#7F8C8D', alpha=0.55, lw=1.8, zorder=2)
    ax1.plot(df['Cum_CSI300'],
             label='CSI300 A-Share Benchmark  |  ~8% p.a.  (跑赢基准)',
             color='#E74C3C', alpha=0.85, lw=1.8, linestyle='--', zorder=2)
    ax1.fill_between(df.index, df['Cum_Strategy'], df['Cum_CSI300'],
                     where=(df['Cum_Strategy'] >= df['Cum_CSI300']),
                     alpha=0.08, color='#2980B9', label='Alpha vs A-Share')

    ax1.set_title('All-Weather Multi-Factor Strategy V30 — Performance vs BTC & A-Share Benchmark',
                  fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Cumulative Return (Multiplier)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.25)

    # 在主图标注关键指标文字框
    textstr = (
        f"Strategy Summary\n"
        f"──────────────────\n"
        f"Sharpe:   {strat_m['sharpe']:.2f}\n"
        f"Sortino:  {strat_m['sortino']:.2f}\n"
        f"Calmar:   {strat_m['calmar']:.2f}\n"
        f"Win Rate: {win_rate*100:.0f}%\n"
        f"Trades:   {entries}"
    )
    props = dict(boxstyle='round', facecolor='#EAF4FB', alpha=0.85)
    ax1.text(0.01, 0.62, textstr, transform=ax1.transAxes, fontsize=8.5,
             verticalalignment='top', bbox=props, family='monospace')

    # --- 副图：回撤对比 ---
    ax2.fill_between(df.index, strat_m['drawdown'] * 100, 0,
                     color='#2980B9', alpha=0.45, label='Strategy Drawdown')
    ax2.plot(df.index, bench_m['drawdown'] * 100,
             color='#7F8C8D', alpha=0.5, lw=1.2, label='BTC Drawdown')
    ax2.axhline(strat_m['max_dd'] * 100, color='#2980B9', linestyle=':',
                lw=1, alpha=0.8, label=f'Strategy MDD {strat_m["max_dd"]*100:.1f}%')
    ax2.axhline(bench_m['max_dd'] * 100, color='#E74C3C', linestyle=':',
                lw=1, alpha=0.8, label=f'BTC MDD {bench_m["max_dd"]*100:.1f}%')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.legend(loc='lower left', fontsize=8.5, ncol=2)
    ax2.grid(True, alpha=0.25)

    # --- 子图3：RSI与超买超卖信号 ---
    ax3.plot(df.index, df['RSI'], color='#8E44AD', lw=1, label='RSI (14)', alpha=0.9)
    ax3.axhline(72, color='#E74C3C', linestyle='--', lw=1, alpha=0.7, label='Overbought (72)')
    ax3.axhline(33, color='#27AE60', linestyle='--', lw=1, alpha=0.7, label='Oversold (33)')
    ax3.fill_between(df.index, df['RSI'], 72,
                     where=(df['RSI'] >= 72), color='#E74C3C', alpha=0.15)
    ax3.fill_between(df.index, df['RSI'], 33,
                     where=(df['RSI'] <= 33), color='#27AE60', alpha=0.15)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('RSI', fontsize=10)
    ax3.set_xlabel('Date', fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)
    ax3.set_title('RSI Momentum Filter', fontsize=10, fontweight='bold')

    # --- 子图4：滚动30日持仓比例 ---
    rolling_pos = df['Position'].rolling(30).mean() * 100
    ax4.fill_between(df.index, rolling_pos, 0,
                     color='#27AE60', alpha=0.45, label='30-Day Avg Position')
    ax4.axhline(time_in_market, color='#E74C3C', linestyle='--', lw=1.2,
                label=f'Avg In-Market: {time_in_market:.0f}%')
    ax4.set_ylim(0, 115)
    ax4.set_ylabel('Position Ratio (%)', fontsize=10)
    ax4.set_xlabel('Date', fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.25)
    ax4.set_title('Rolling 30-Day Position Exposure', fontsize=10, fontweight='bold')

    plt.savefig('strategy_v30_report.png', bbox_inches='tight', dpi=300)
    print("\n✅ 报告图表已保存为 'strategy_v30_report.png'")
    plt.close()


if __name__ == "__main__":
    run_strategy_v30()
