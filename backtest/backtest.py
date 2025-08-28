import backtrader as bt
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

class PandasSignalData(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('datetime', None), # ✅ 使用 DataFrame 的 DatetimeIndex
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1), # 若无则填 -1
        ('signal', 'final_signal'),
    )

def run_backtest(strategy_class, signals_df, strategy_params,initial_cash=100000, commission=0.0005):
    # 初始化
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    # 加入数据
    data_feed = PandasSignalData(dataname=signals_df)
    cerebro.adddata(data_feed)
    # 加入策略
    cerebro.addstrategy(strategy_class, **strategy_params)
    # 加入分析
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # ✅ 新增逐bar收益（不指定 timeframe => 按数据频率）
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturns')

    # 运行回测
    results = cerebro.run()

    return results

def print_bt_evals(bt_strat):
    # 1. Sharpe Ratio
    print("Sharpe Ratio:", bt_strat.analyzers.sharpe.get_analysis()['sharperatio'])

    # 2. Drawdown
    dd = bt_strat.analyzers.drawdown.get_analysis()
    print(f"Max Drawdown: {dd['max']['drawdown']:.2f}%")

    # 3. Returns
    ret = bt_strat.analyzers.returns.get_analysis()
    print(f"Total Return: {ret['rtot']:.2%}, Annual Return: {ret['rnorm']:.2%}")

    # 4. Trade Stats
    pprint(bt_strat.analyzers.trades.get_analysis())



def extract_trades_df(bt_strat) -> pd.DataFrame:
    """
    将策略里记录的 trade_log 转为 DataFrame，并做基本清洗/排序
    """
    trades = getattr(bt_strat, 'trade_log', [])
    if not trades:
        return pd.DataFrame(columns=[
            'symbol','open_dt','close_dt','size','is_long','entry_price','exit_price',
            'barlen','pnl_gross','pnl_net','commission','broker_value'
        ])

    df = pd.DataFrame(trades)
    # 时间列转为 pandas datetime，方便后续分析
    for col in ['open_dt', 'close_dt']:
        df[col] = pd.to_datetime(df[col])
    # 排序
    df.sort_values('close_dt', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_equity_series(bt_strat, initial_cash: float) -> pd.Series:
    """
    从 TimeReturn analyzer 拿到逐bar收益，累乘得到净值曲线
    """
    ret_dict = bt_strat.analyzers.timereturns.get_analysis()  # {datetime: return}
    if isinstance(ret_dict, dict) and ret_dict:
        # 转成按时间排序的 Series
        sr = pd.Series(ret_dict)
        sr.index = pd.to_datetime(sr.index)
        sr = sr.sort_index()
        equity = (1.0 + sr).cumprod() * float(initial_cash)
        equity.name = 'equity'
        return equity
    else:
        return pd.Series(dtype=float)


def plot_pnl_curve(bt_strat, initial_cash: float, show=True, ax=None):
    """
    以净值曲线展示 PnL 趋势（净值-初始资金），并标注每笔平仓点位
    """
    equity = get_equity_series(bt_strat, initial_cash)
    if equity.empty:
        print("No equity/returns data to plot.")
        return

    pnl = equity - float(initial_cash)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    pnl.plot(ax=ax)
    ax.set_title('PnL Trend (Equity - Initial Cash)')
    ax.set_xlabel('Time')
    ax.set_ylabel('PnL')
    ax.grid(True, alpha=0.3)

    # 可选：在每笔“平仓时刻”画竖线辅助观察
    trades_df = extract_trades_df(bt_strat)
    if not trades_df.empty:
        for dt in trades_df['close_dt']:
            ax.axvline(pd.to_datetime(dt), linestyle='--', alpha=0.15)

    if show:
        plt.tight_layout()
        plt.show()
