# streamlit_app.py
import os, sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ===== 配置 =====
DB_PATH = "../data/crypto_data.db"
symbol = "ETHUSDT"
strategy_name = "rf-reg"

def get_connection(db_path: str):
    return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

# ===== 数据访问 =====
def list_available_days(symbol: str) -> pd.DatetimeIndex:
    """从 kline（或 signals）取该 symbol 的可用日期列表（按天去重）"""
    sql = """
    SELECT datetime
    FROM kline
    WHERE symbol=?
    ORDER BY datetime ASC
    """
    with get_connection(DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn, params=(symbol,), parse_dates=['datetime'])
    if df.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(df['datetime'].dt.normalize().unique()).sort_values()

def load_range(symbol: str, start_dt, end_dt) -> pd.DataFrame:
    """按时间范围读取 signals + kline 联表"""
    sql = """
    SELECT
      a.datetime, a.symbol, b.open, b.high, b.low, b.close, b.volume, a.final_signal
    FROM signals a
    LEFT JOIN kline b
      ON a.symbol=b.symbol AND a.datetime=b.datetime
    WHERE a.symbol = ?
      AND a.datetime >= ?
      AND a.datetime <= ?
    ORDER BY a.datetime ASC
    """

    # --- 关键：把 pandas.Timestamp -> SQLite 友好格式 ---
    def to_sqlite_ts(ts):
        ts = pd.Timestamp(ts)
        # 若是带时区，先转 UTC 再去时区
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC')
        ts = ts.tz_localize(None)
        return ts.strftime('%Y-%m-%d %H:%M:%S')

    start_s = to_sqlite_ts(start_dt)
    end_s   = to_sqlite_ts(end_dt)

    with get_connection(DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn, params=(symbol, start_s, end_s), parse_dates=['datetime'])
    return df

# ===== 技术指标 =====
def compute_atr_12(ohlc: pd.DataFrame) -> pd.Series:
    prev_close = ohlc['close'].shift(1)
    tr = np.maximum(
        ohlc['high'] - ohlc['low'],
        np.maximum((ohlc['high'] - prev_close).abs(), (ohlc['low'] - prev_close).abs())
    )
    return tr.rolling(window=12, min_periods=1).mean()

def compute_ma(ohlc: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    out = pd.DataFrame(index=ohlc.index)
    for w in windows:
        out[f"MA{w}"] = ohlc['close'].rolling(w, min_periods=1).mean()
    return out

def compute_macd(ohlc: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = ohlc['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = ohlc['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# ===== 作图 =====
def plot_chart(df: pd.DataFrame, symbol: str, strategy_name: str):
    if df.empty:
        st.warning("查询结果为空。请调整时间范围或检查数据库。")
        return

    df = df.set_index('datetime')
    ohlc = df[['open','high','low','close']].copy()

    # 信号散点（与主图同轴）
    long_y  = np.full(len(df), np.nan)
    short_y = np.full(len(df), np.nan)
    long_m  = (df['final_signal'] == 1)
    short_m = (df['final_signal'] == -1)
    long_y[long_m]   = df.loc[long_m,  'low']  * 0.99
    short_y[short_m] = df.loc[short_m, 'high'] * 1.01

    # 指标
    atr = compute_atr_12(ohlc)
    ma_df = compute_ma(ohlc, windows=(20, 50))
    macd, macd_sig, macd_hist = compute_macd(ohlc)

    apds = [
        # 信号
        mpf.make_addplot(long_y,  type='scatter', marker='^', color='blue',   markersize=100),
        mpf.make_addplot(short_y, type='scatter', marker='v', color='orange', markersize=100),

        # ATR panel=1
        mpf.make_addplot(atr, panel=1, type='line', ylabel='ATR(12)'),

        # MA panel=2
        mpf.make_addplot(ma_df['MA20'], panel=2, type='line', ylabel='MA(20/50)'),
        mpf.make_addplot(ma_df['MA50'], panel=2, type='line'),

        # MACD panel=3（线 + 柱）
        mpf.make_addplot(macd,      panel=3, type='line', ylabel='MACD'),
        mpf.make_addplot(macd_sig,  panel=3, type='line'),
        mpf.make_addplot(macd_hist, panel=3, type='bar'),
    ]

    fig, axes = mpf.plot(
        ohlc,
        type='candle',
        style='charles',
        ylabel='Price',
        addplot=apds,
        figsize=(14, 9),
        datetime_format='%b %d %H:%M',
        xrotation=15,
        panel_ratios=(3,1,1,1),
        returnfig=True,
        tight_layout=True
    )

    fig.suptitle(
        f'{symbol} Strategy Entry Points - {strategy_name}',
        fontsize=14,
        fontweight='bold',
        y=1.02  # 标题位置，>1 会往外上移
    )

    st.pyplot(fig, clear_figure=True)

# ===== UI =====
st.set_page_config(page_title="Crypto ML Signals Dashboard", layout="wide")
st.title("📈 Crypto ML Signals + ATR / MA / MACD")

# 生成日期下拉（来自数据库可用日期）
available_days = list_available_days(symbol)
cutoff = pd.Timestamp("2025-06-25")  # 要求 start_date > 2025-06-25
options = available_days[available_days > cutoff]

if len(options) < 2:
    st.warning("可用日期不足（需要至少两个日期且晚于 2025-06-25）。")
    st.stop()

# 只保留日期两个下拉框
c3, c4 = st.columns([1, 1])
with c3:
    start_date = st.selectbox(
        "Start Date (must > 2025-06-25)",
        list(options),
        index=0,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

# 计算默认 end 索引
start_pos = int(np.searchsorted(options.values, np.datetime64(start_date), side='left'))
default_end_idx = int(min(start_pos + 1, len(options) - 1))

with c4:
    end_date = st.selectbox(
        "End Date (> Start)",
        list(options),
        index=default_end_idx,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

# 校验
if start_date <= cutoff:
    st.error("Start Date 必须大于 2025-06-25。")
    st.stop()
if end_date <= start_date:
    st.error("End Date 必须大于 Start Date。")
    st.stop()

# 拼接查询的时间区间（含整天）
start_dt = pd.Timestamp(start_date)                    # 00:00:00
end_dt   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # 当天 23:59:59

df = load_range(symbol, start_dt, end_dt)
plot_chart(df, symbol, strategy_name)

st.caption("Note：ATR(12)=TR's 12-period average；MA(20/50) and MACD(12,26,9) is based on close price。Select range is based on available dates from database。")
