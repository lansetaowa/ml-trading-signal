import numpy as np
import pandas as pd

import mplfinance as mpf
import matplotlib.dates as mdates

from data.db_utils import get_connection
DB_PATH = '../data/crypto_data.db'

from config import target

def plot_recent_signals(limit = 24*3, strategy_name='rf-reg'):

    query = f""" 
    SELECT
    a.datetime, a.symbol, open, high, low, close, volume, final_signal
    FROM
    (SELECT * 
    FROM signals
    WHERE symbol=?
    ORDER BY datetime DESC 
    LIMIT ?) a 
    LEFT JOIN kline b
    ON a.symbol=b.symbol and a.datetime = b.datetime
    """

    conn = get_connection(DB_PATH)

    filtered_df = pd.read_sql_query(
        query,
        conn,
        params=(target, limit),
        parse_dates=['datetime']
    )

    filtered_df = filtered_df.sort_values('datetime').set_index('datetime')

    # 保证 OHLC 有统一索引（DatetimeIndex）
    ohlc = filtered_df[['open', 'high', 'low', 'close']].copy()

    # ===== ATR(12) 计算（基于 TR 的 12 期滚动均值）=====
    prev_close = ohlc['close'].shift(1)
    tr = np.maximum(
        ohlc['high'] - ohlc['low'],
        np.maximum((ohlc['high'] - prev_close).abs(), (ohlc['low'] - prev_close).abs())
    )
    atr = tr.rolling(window=12, min_periods=1).mean()

    # 构造 y 序列：跟 ohlc 一样长，其它位置设为 np.nan
    long_signal_y = np.full(len(filtered_df), np.nan)
    short_signal_y = np.full(len(filtered_df), np.nan)

    long_signal_y[filtered_df['final_signal'] == 1] = filtered_df.loc[filtered_df['final_signal'] == 1, 'low'] * 0.99
    short_signal_y[filtered_df['final_signal'] == -1] = filtered_df.loc[filtered_df['final_signal'] == -1, 'high'] * 1.01

    # 构造 addplot 列表
    apds = [
        mpf.make_addplot(long_signal_y, type='scatter', marker='^', color='blue', markersize=100),
        mpf.make_addplot(short_signal_y, type='scatter', marker='v', color='orange', markersize=100),

        # ATR 作为第2个面板（panel=1）
        mpf.make_addplot(atr, panel=1, type='line', ylabel='ATR(12)')
    ]

    # 画图并拿到轴对象，便于给 ATR 标注
    mpf.plot(
        ohlc,
        type='candle',
        style='charles',
        title=f'{target} Strategy Entry Points - {strategy_name}',
        ylabel='Price',
        addplot=apds,
        figsize=(14, 6),
        datetime_format='%b %d %H:%M',
        xrotation=20,
        panel_ratios=(3, 1),
        returnfig=True
    )

    mpf.show()

def plot_signals(df, start, end, symbol, strategy_name):

    filtered_df = df[(df.index>=start)&(df.index<=end)]

    # 保证 OHLC 有统一索引（DatetimeIndex）
    ohlc = filtered_df[['open', 'high', 'low', 'close']].copy()
    # 构造 y 序列：跟 ohlc 一样长，其它位置设为 np.nan
    long_signal_y = np.full(len(filtered_df), np.nan)
    short_signal_y = np.full(len(filtered_df), np.nan)

    long_signal_y[filtered_df['final_signal'] == 1] = filtered_df.loc[filtered_df['final_signal'] == 1, 'low'] * 0.99
    short_signal_y[filtered_df['final_signal'] == -1] = filtered_df.loc[filtered_df['final_signal'] == -1, 'high'] * 1.01

    # 构造 addplot 列表
    apds = [
        mpf.make_addplot(long_signal_y, type='scatter', marker='^', color='blue', markersize=100),
        mpf.make_addplot(short_signal_y, type='scatter', marker='v', color='orange', markersize=100),
    ]

    mpf.plot(
        ohlc,
        type='candle',
        style='charles',
        title=f'{symbol} Strategy Entry Points - {strategy_name}',
        ylabel='Price',
        addplot=apds,
        figsize=(14, 6),
        datetime_format='%b %d %H:%M',
        xrotation=20
    )

if __name__ == '__main__':
    # import sqlite3
    # import pandas as pd
    #
    # query = f"""
    #     SELECT
    #     a.datetime, a.symbol, open, high, low, close, volume, final_signal
    #     FROM
    #     (SELECT * from signals
    #     WHERE symbol='{target}'
    #     ORDER BY datetime DESC
    #     LIMIT 24*3) a
    #     LEFT JOIN
    #     kline b
    #     ON a.symbol=b.symbol and a.datetime = b.datetime
    #     """
    #
    # signals = pd.read_sql_query(
    #     query,
    #     sqlite3.connect('../data/crypto_data.db'),
    #     parse_dates=['datetime']
    # )
    #
    # print(signals.info())
    # print(signals.head())

    plot_recent_signals(limit=3*24)