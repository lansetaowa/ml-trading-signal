import backtrader as bt
import pandas as pd
from pprint import pprint

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