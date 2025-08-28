import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import datetime, timedelta, timezone

from data.crypto_data_loader import DataHandler, load_multi_symbol_data
from data.db_utils import get_connection, upsert_df, fetch_predictions_history

from model.feature_generator import FeatureGenerator, FeatureProcessor
from model.fit_pred import split_data, clean_xy, fit_predict_regression_model, fit_predict_last_line
from model.signal_generator import SignalGenerator

from sklearn.ensemble import RandomForestRegressor
from model.timeseries_cv import MultipleTimeSeriesCV

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_info_columns', 200)

# --- 配置部分 ---
from conf.settings_loader import settings

feature_config = settings.features.model_dump()
feature_process_config = settings.feature_process.model_dump()
signal_config = settings.signal.model_dump()
target = settings.data.symbols.target # 预测目标symbol
load_symbols = settings.data.symbols.load_symbols # 伴随的作为特征的symbols
symbols = list(set([target]+load_symbols)) # 实际获取哪些symbol的数据（目标和伴随有可能重复，要去重）

# CV & Model
tscv = MultipleTimeSeriesCV(
    train_length=settings.cv.train_length,
    test_length=settings.cv.test_length,
    lookahead=settings.cv.lookahead,
    date_idx=settings.cv.date_idx
)
train_model = RandomForestRegressor(**settings.model.params)

INTERVAL = settings.data.interval
START_DELTA = timedelta(days=settings.data.start_delta_days)
DB_PATH = settings.paths.db_path
MODEL_NAME = settings.model.name
STRATEGY_NAME = settings.signal.strategy_name
target_col = 'target_1h'

# === Step 1: 获取数据库连接和最新时间 ===
def ensure_db_initialized():
    handler = DataHandler()
    # symbols = list(set([target]+load_symbols)) # 获取哪些symbol的数据
    conn = get_connection(DB_PATH)
    # init_db(conn)
    # model_start_date = (datetime.utcnow() - START_DELTA).strftime('%Y-%m-%d %H:%M:%S')

    return conn, handler

# === Step 2: 回溯 N 天并写到“此刻”为止（允许未完结K线，后续覆盖） ===
def fetch_and_store_backfill_no_lag(conn, handler, symbols, backfill_hours=4):

    now_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)  # 到当前时刻（分级对齐）
    start_dt = now_utc - timedelta(hours=backfill_hours)

    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_str   = now_utc.strftime('%Y-%m-%d %H:%M:%S')

    print(f'Fetching price data backfill: [{start_str}, {end_str}] for {len(symbols)} symbols ...')

    df_price = load_multi_symbol_data(handler, symbols, start_str=start_str)
    if df_price.empty:
        print('No price data returned.')
        return

    # 规范列、去重（同一(symbol, datetime)取最后一条，便于覆盖半成品）
    df_price = (df_price
                .reset_index()
                [['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
                .sort_values(['symbol', 'datetime']))
    df_price = df_price.drop_duplicates(subset=['symbol', 'datetime'], keep='last')

    print(f'Upserting {len(df_price)} rows into kline (backfill {backfill_hours}h, no lag, allow provisional bars)...')
    upsert_df(df_price, table='kline', conn=conn)

# === Step 3-1: 读取最近 N 天的数据用于建模 ===
def prepare_feature_data(handler, symbols, target, model_start_date):

    # 数据来自从币安api实时获取
    price_all = load_multi_symbol_data(handler, symbols, start_str=model_start_date)

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
    return df_processed, price_all

# === Step 3-2: 从给定起点拉价并做特征工程（带缓冲窗口） ===
def prepare_feature_data_from_start(handler,
                                    symbols,
                                    target,
                                    start_str,
                                    buffer_windows=300,
                                    train_windows=24*7*4):
    """
    从 start_str 开始做特征工程，但为满足窗口/滞后，会向前多取 buffer_hours 小时的价格。
    返回: (df_processed, price_all)；df_processed 已裁掉 warmup，只保留 start_str 及之后。
    """
    start_dt = pd.to_datetime(start_str)
    start_with_buffer = (start_dt - pd.Timedelta(hours=buffer_windows+train_windows)
                         ).strftime('%Y-%m-%d %H:%M:%S')

    price_all = load_multi_symbol_data(handler, symbols, start_str=start_with_buffer)
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
    return df_processed, price_all

# === Step 4-1: 建模 + 预测（时间交叉验证）===
def train_and_predict(df_processed, model, cv):
    X, y = split_data(df_processed)
    X_clean, y_clean = clean_xy(X, y[target_col])
    pred_df = fit_predict_regression_model(model=model, X=X_clean, y=y_clean, cv=cv)

    return pred_df

# === Step 4-2:一次性补全从 start_str 到最近一条的所有预测 ===
def backfill_predictions_from(handler,
                              start_str: str,
                              symbols,
                              target,
                              buffer_windows: int = 300,
                              train_windows: int = 24*7*4):
    """
    使用你已有的 fit_predict_regression_model + tscv(train_length/test_length/lookahead)
    将从 start_str 起能形成的所有预测，写入 predictions（仅 predicted）。
    """
    # 1) 特征工程（带 warmup）
    df_processed, _ = prepare_feature_data_from_start(
        handler=handler,
        symbols=symbols,
        target=target,
        start_str=start_str,
        buffer_windows=buffer_windows,
        train_windows=train_windows
    )

    # 2) 建模 + 预测（滚动CV）
    pred_df = train_and_predict(df_processed, model=train_model, cv=tscv)

    if pred_df is None or pred_df.empty:
        print("[backfill] no predictions generated")
        return None

    # 3) 仅保留 start_str 之后
    pred_df = pred_df[['predicted']].copy()
    pred_df = pred_df.loc[pd.to_datetime(start_str):]
    pred_df['symbol'] = target
    pred_df['model_name'] = MODEL_NAME

    return pred_df

# === Step 4-3: 只预测最后一条 ===
def predict_latest(handler,
                   symbols,
                   target,
                   buffer_windows: int = 300,
                   train_windows: int = 24 * 7 * 4):
    """
    只预测最近 1 小时并写入 predictions。
    仍然先拉足够长的价格与特征，以保证最后一行可预测。
    """
    start_str = (datetime.utcnow().replace(minute=0, second=0, microsecond=0)
                 - timedelta(hours=2)
                 ).strftime("%Y-%m-%d %H:%M:%S")

    # 可直接复用你现有的 prepare_feature_data（按 settings.data.start_delta_days 回看）
    df_processed, _ = prepare_feature_data_from_start(handler=handler,
                                                      symbols=symbols,
                                                      target=target,
                                                      start_str=start_str,
                                                      buffer_windows=buffer_windows,
                                                      train_windows=train_windows)

    X, y = split_data(df_processed)
    X_clean, y_clean = clean_xy(X, y[target_col])

    # 只预测最后一行
    out = fit_predict_last_line(train_model, X_clean, y_clean, train_length=train_windows)

    return out

# === Step 5: 预测存入predictions表 ===
def store_predictions(df_pred, conn):
    """
    期望 df_pred: index 为 datetime 或有 datetime 列，且包含 'predicted'、以及 'symbol','model_name'
    这个函数只把 (symbol, datetime, predicted, model_name) upsert 到 predictions
    """
    if df_pred is None or len(df_pred) == 0:
        return

    out = df_pred.copy()

    cols = ['symbol', 'datetime', 'predicted', 'model_name']
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"store_predictions: missing columns {missing}")

    upsert_df(out[cols], table='predictions', conn=conn)

# === Step 6: 生成交易信号 ===
def generate_signals(pred_df, price_all, target):
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

    df_signal_final = df_signal.drop(columns=['open', 'high', 'low', 'close', 'volume']).copy()
    df_signal_final = df_signal_final.reset_index()
    df_signal_final['symbol'] = target
    df_signal_final['model_name'] = MODEL_NAME
    df_signal_final['strategy_name'] = STRATEGY_NAME

    return df_signal_final[
        ['symbol', 'datetime', 'actuals', 'predicted', 'zscore',
         'raw_signal', 'vol_filter', 'filtered_signal',
         'position', 'signal_reversal', 'final_signal',
         'model_name', 'strategy_name']
    ]

def generate_latest_signal_from_predictions(symbol: str = None, lookback_hours: int = 168):
    """
    从 predictions 读取最近 lookback_hours 小时的历史预测，交给 SignalGenerator，
    生成一段信号，但**只 upsert 最后一行**到 signals 表（其余仅作上下文）。
    """
    pass

# === Step 7: 信号结果存入数据库 signal 表 ===
def store_signals(df_signal_final, conn):
    upsert_df(df_signal_final, table='signals', conn=conn)

# === 综合上述几步 ===
def run_pipeline():
    # # 初始化数据库连接
    # conn, handler = ensure_db_initialized()
    #
    # # 下载并存储最新价格数据
    # fetch_and_store_backfill_no_lag(conn, handler, symbols, backfill_hours=4)
    #
    # # 特征工程
    # df_processed, price_all = prepare_feature_data(handler, symbols, target, model_start_date)
    #
    # # 模型训练与预测
    # pred_df = train_and_predict(df_processed, cv=tscv)
    #
    # # 生成信号
    # df_signal_final = generate_signals(pred_df, price_all, target)
    #
    # # 写入数据库
    # store_signals(df_signal_final, conn)

    pass



if __name__ == '__main__':

    # from conf.settings_loader import settings
    #
    # print(settings.binance.model_dump())

    # run_pipeline()
    # conn, handler, symbols, model_start_date = ensure_db_initialized()
    # df_processed, price_all = prepare_feature_data(handler, symbols, target, model_start_date)
    #
    # print(df_processed.info())
    # print(price_all.info())

    print(settings.cv.model_dump())


