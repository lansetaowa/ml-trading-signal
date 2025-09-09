import warnings
from cgitb import handler

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from data.crypto_data_loader import DataHandler, load_multi_symbol_data
from data.db_utils import get_connection, upsert_df, fetch_predictions_history, get_last_timestamp, init_db

from model.feature_generator import FeatureGenerator, FeatureProcessor
from model.fit_pred import split_data, clean_xy, fit_predict_regression_model
from model.signal_generator import SignalGenerator

from sklearn.ensemble import RandomForestRegressor
from model.timeseries_cv import MultipleTimeSeriesCV

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_info_columns', 200)

# --- 配置部分 ---
from conf.settings_loader import settings

feature_config = settings.features.model_dump() # 特征工程配置
feature_process_config = settings.feature_process.model_dump() # 特征清洗配置
signal_config = settings.signal.model_dump() # 信号生成配置
target = settings.data.symbols.target # 预测目标symbol
load_symbols = settings.data.symbols.load_symbols # 伴随的作为特征的symbols
all_symbols = list(set([target]+load_symbols)) # 实际获取哪些symbol的数据（目标和伴随有可能重复，要去重）

# CV & Model
tscv = MultipleTimeSeriesCV(
    train_length=settings.cv.train_length,
    test_length=settings.cv.test_length,
    lookahead=settings.cv.lookahead,
    date_idx=settings.cv.date_idx
)
train_length = settings.cv.train_length # 预测用bar的数量
train_model = RandomForestRegressor(**settings.model.params) # 预测模型配置

# INTERVAL = settings.data.interval # 在哪个level的时间间隔上预测, 1h/30m/15m etc.
INTERVAL = '30m'
START_DELTA = timedelta(days=settings.data.start_delta_days)
DB_PATH = settings.paths.db_path
MODEL_NAME = settings.model.name
STRATEGY_NAME = settings.signal.strategy_name
target_col = 'target_1bar'

# === Step 1: 获取数据库连接和最新时间 ===
def ensure_db_initialized():
    handler = DataHandler()
    conn = get_connection(DB_PATH)
    # init_db(conn)

    return conn, handler

# === Step 2: 从给定起点拉价并做特征工程（带缓冲窗口） ===
def prepare_feature_data_from_start(handler,
                                    start_str,
                                    buffer_windows=300,
                                    train_windows=24*7*4):
    """
    从 start_str 开始做特征工程，但为满足窗口/滞后，会向前多取 buffer_hours 小时的价格。
    返回: (df_processed, price_all)；df_processed 已裁掉 warmup，只保留 start_str 及之后。
    """
    start_dt = pd.to_datetime(start_str)
    start_with_buffer = (start_dt - pd.Timedelta(
                                                hours=(buffer_windows+train_windows)*(interval_minutes(INTERVAL)/60)
                                                )
                         ).strftime('%Y-%m-%d %H:%M:%S')

    price_all = load_multi_symbol_data(handler,
                                       symbols=all_symbols,
                                       start_str=start_with_buffer,
                                       interval=INTERVAL)
    df_features = (
        FeatureGenerator(config=feature_config)
        .load_data(price_all)
        .select_symbols(target_symbol=target)
        .compute_volume_features()
        .compute_momentum_features()
        .compute_volatility_features()
        .compute_target_cols()
        .compute_time_dummies()
        .compute_tech_indicators()
        .compute_candle_patterns()
        .get_single_symbol_data(target_symbol=target)
    )
    df_processed = (
        FeatureProcessor(config=feature_process_config)
        .load_data(df_features)
        .scale_metrics()
        .drop_cols()
        .df
    )
    # 裁掉 warmup 只保留从 start_str 开始
    cut = (start_dt - pd.Timedelta(hours=train_windows)
                         ).strftime('%Y-%m-%d %H:%M:%S')
    df_processed = df_processed.loc[pd.to_datetime(cut):]
    return df_processed

# === Step 3: 一次性补全从 start_str 到最近一条的所有预测, 建模 + 预测（时间交叉验证）===
def _train_and_predict(df_processed, model, cv):
    X, y = split_data(df_processed)
    X_clean, y_clean = clean_xy(X, y[target_col])
    pred_df = fit_predict_regression_model(model=model, X=X_clean, y=y_clean, cv=cv)

    return pred_df

def backfill_predictions_from(handler,
                              start_str: str,
                              buffer_windows: int = 300,
                              train_windows: int = 24*7*4):
    """
    使用你已有的 fit_predict_regression_model + tscv(train_length/test_length/lookahead)
    将从 start_str 起能形成的所有预测，写入 predictions（仅 predicted）。
    """
    # 1) 特征工程（带 warmup）
    df_processed = prepare_feature_data_from_start(
        handler=handler,
        start_str=start_str,
        buffer_windows=buffer_windows,
        train_windows=train_windows
    )

    # 2) 建模 + 预测（滚动CV）
    pred_df = _train_and_predict(df_processed, model=train_model, cv=tscv)

    if pred_df is None or pred_df.empty:
        print("[backfill] no predictions generated")
        return None

    # 3) 仅保留 start_str 之后
    pred_df = pred_df[['predicted']].copy()
    pred_df = pred_df.loc[pd.to_datetime(start_str):]
    pred_df['symbol'] = target
    pred_df['model_name'] = MODEL_NAME

    # 5) 统一 datetime 为 string 格式
    pred_df = pred_df.reset_index().rename(columns={'index': 'datetime'})
    pred_df['datetime'] = pd.to_datetime(pred_df['datetime'], utc=True).dt.tz_convert(None).dt.strftime(
        '%Y-%m-%d %H:%M:%S')

    return pred_df

INTERVAL_MIN  = {"1h": 60,  "30m": 30,   "15m": 15}
INTERVAL_FREQ = {"1h": "H", "30m": "30T", "15m": "15T"}

def interval_minutes(interval: str) -> int:
    return INTERVAL_MIN.get(interval, 60)

def floor_to_interval(ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True).tz_convert(None).floor(INTERVAL_FREQ.get(interval, "H"))

# === Step 4: 生成交易信号 ===
def _to_utc_naive_str(ts):
    """统一把 Timestamp/str 转为 'YYYY-MM-DD HH:MM:SS'（naive UTC）的字符串"""
    return (pd.to_datetime(ts, utc=True)
            .tz_convert(None)
            .strftime('%Y-%m-%d %H:%M:%S'))

def _add_actuals_from_close(df_with_close: pd.DataFrame) -> pd.DataFrame:
    """actuals_t = close_{t+1}/close_t - 1；最后一行因无 t+1 为 NaN"""
    df = df_with_close.copy()
    df['actuals'] = df['close'].shift(-1) / df['close'] - 1
    return df

# 从 DB 的历史预测构造整段 signals（仅返回，不落库）
# 适用于“项目运行初期”需要把缺口一次性补齐的场景，也可以用于生成最近的signals，按缺口来决定
def generate_signals_from_predictions(handler,
                                      conn,
                                      start_str: str) -> pd.DataFrame:
    """
    从 predictions 表读取 [start_str, now] 的历史预测，合并价格并生成一整段 signals（含 OHLCV、actuals）。
    仅返回 DataFrame，不写库，便于 notebook 调试。
    """

    # 1) 拉历史预测
    until_dt = floor_to_interval(pd.Timestamp.now(
                                                    tz=timezone.utc
                                                ),
                                 INTERVAL)

    lookback_bars = int(np.ceil(
                        (until_dt - pd.to_datetime(start_str)
                         ).total_seconds() / (interval_minutes(INTERVAL)*60) )
                        ) + signal_config['z_window'] # 由于z-score和计算position reversal，需要往前多取

    pred_hist = fetch_predictions_history(conn,
                                          symbol=target,
                                          lookback_bars=lookback_bars)

    # 2) 拉价（含足够 buffer 以满足指标窗口/ATR）
    atr_long = signal_config['atr_windows']['long']
    price_start = (pred_hist.index.min() - pd.Timedelta(
                                                hours=(interval_minutes(INTERVAL)/60) * atr_long
                                                        )
                   )  # 给 ATR 多一点缓冲
    price_all = load_multi_symbol_data(handler,
                                       symbols=all_symbols,
                                       interval=INTERVAL,
                                       start_str=_to_utc_naive_str(price_start))

    # 3) 组装 pred_df 传入 SignalGenerator（索引为 datetime，且带 symbol）
    pred_df = pred_hist.copy()
    pred_df['symbol'] = target
    pred_df.index.name = 'datetime'

    # 4) 生成信号（这一步保留 OHLCV，不再 drop）
    df_signal = (
        SignalGenerator(config=signal_config)
        .load_data(pred_df=pred_df, price_df=price_all)
        .merge_pred_price(target_symbol=target)
        .zscore_normalize()
        .compute_raw_signal()
        .apply_atr_volatility_filter()
        .compute_positions()
        .compute_reversals()
        .apply_min_signal_spacing(min_space=2)
    )

    df_signal.dropna(inplace=True) # 去掉没有滚动zscore的记录

    # 5) 补 actuals（来自 close 列）
    df_signal = _add_actuals_from_close(df_signal)

    # 6) 整理列 & 元数据
    out = df_signal.reset_index().rename(columns={'index': 'datetime'})
    # out['symbol'] = target
    out['model_name'] = MODEL_NAME
    out['strategy_name'] = STRATEGY_NAME

    # 7) 统一 datetime 字符串格式（避免 00:00:00 被显示成纯日期）
    out['datetime'] = out['datetime'].map(_to_utc_naive_str)

    # signals 表需要的列顺序（含 OHLCV）
    cols = ['symbol', 'datetime',
            'open', 'high', 'low', 'close', 'volume',
            'actuals', 'predicted', 'zscore',
            'raw_signal', 'vol_filter', 'filtered_signal',
            'position', 'signal_reversal', 'final_signal',
            'model_name', 'strategy_name']
    out = out[cols]

    return out

# === 综合上述几步 ===
# 第一次跑，predictions/signals表还是空的，一次性补全一段时间的预测和信号
def initial_run(start_str):
    conn, handler = ensure_db_initialized()
    pred_df = backfill_predictions_from(handler=handler, start_str=start_str, buffer_windows=300,
                                        train_windows=train_length)
    upsert_df(pred_df, 'predictions', conn)

    signals_df = generate_signals_from_predictions(handler, conn, start_str)

    upsert_df(signals_df, 'signals', conn)

# predictions和signals表非空，补全缺失的
def run_pipeline():
    # 初始化数据库连接
    conn, handler = ensure_db_initialized()

    # 已有预测的最大时间
    max_dt = get_last_timestamp(conn, table='predictions')

    # 如果和目前时间差距不足2小时，则说明预测已齐全，跳过预测
    gap = pd.Timestamp.utcnow() - pd.to_datetime(max_dt, utc=True)
    if gap.total_seconds() / (interval_minutes(INTERVAL)*60) < 2:
        print("预测已是最新（<2个周期差），跳过补全与后续步骤。")
        return

    # 补全缺失的预测
    pred_fill_df = backfill_predictions_from(handler=handler,
                                              start_str=max_dt,
                                             buffer_windows=300,
                                              train_windows=train_length)

    # 新增预测入库
    upsert_df(pred_fill_df.reset_index(), 'predictions', conn)

    # 已有信号的最大时间
    start_str = get_last_timestamp(conn, table='signals')
    # 补全缺失的信号
    sig_all = generate_signals_from_predictions(
        handler=handler,
        conn=conn,
        start_str=start_str
    )
    # 新增信号入库
    upsert_df(sig_all,'signals', conn)


if __name__ == '__main__':

    # from conf.settings_loader import settings
    #
    # print(settings.binance.model_dump())

    # run_pipeline()

    # conn, handler= ensure_db_initialized()
    # max_dt = get_last_timestamp(conn, 'predictions')
    # print(max_dt)
    # gap = pd.Timestamp.utcnow() - pd.to_datetime(max_dt, utc=True)
    # print(gap)
    # print(gap.total_seconds() / (interval_minutes(INTERVAL)*60))
    #
    # pred_fill_df = backfill_predictions_from(handler=handler,
    #                                          start_str=max_dt,
    #                                          buffer_windows=300,
    #                                          train_windows=train_length)
    # print(pred_fill_df.info())
    # df_processed, price_all = prepare_feature_data(handler, symbols, target, model_start_date)
    #
    # print(df_processed.info())
    # print(price_all.info())

    # print(settings.cv.model_dump())
    # handler = DataHandler()
    #
    # latest = predict_latest(handler=handler,
    #                                  symbols=all_symbols,
    #                                  target=target,
    #                                  buffer_windows=300,
    #                                  train_windows=24 * 7 * 4)
    # print(latest)

    start_str = '2025-08-15 00:00:00'
    # pred_df = backfill_predictions_from(handler=handler, start_str=start_str, buffer_windows=300,
    #                                     train_windows=train_length)
    # print(pred_df.info())
    # print(pred_df.head())
    # print(pred_df.tail())
    # upsert_df(pred_df, 'predictions', conn)
    # signals_df = generate_signals_from_predictions(handler, conn, start_str)
    # print(signals_df.info())
    # print(signals_df.head())
    # print(signals_df.tail())
    # upsert_df(signals_df, 'signals', conn)
    initial_run(start_str=start_str)

