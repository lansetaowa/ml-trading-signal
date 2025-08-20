import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import datetime, timedelta, timezone

from data.crypto_data_loader import DataHandler, load_multi_symbol_data
from data.db_utils import get_connection, upsert_df

from model.feature_generator import FeatureGenerator, FeatureProcessor
from model.fit_pred import split_data, clean_xy, fit_predict_regression_model
from model.signal_generator import SignalGenerator

from sklearn.ensemble import RandomForestRegressor
from model.timeseries_cv import MultipleTimeSeriesCV

# from config import (feature_config,
#                     feature_process_config,
#                     signal_config,
#                     target,
#                     load_symbols,
#                     tscv,
#                     reg_rf)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_info_columns', 200)

# --- 配置部分 ---
from conf.settings_loader import settings

feature_config = settings.features.model_dump()
feature_process_config = settings.feature_process.model_dump()
signal_config = settings.signal.model_dump()
target = settings.data.symbols.target
load_symbols = settings.data.symbols.load_symbols

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


# INTERVAL = '1h'
# START_DELTA = timedelta(days=60)
# DB_PATH = 'data/crypto_data.db'
# MODEL_NAME = 'rf-reg'
# STRATEGY_NAME = 'zscore_atr_v1'
# target_col = 'target_1h'
# train_model = reg_rf

# === Step 1: 获取数据库连接和最新时间 ===
def ensure_db_initialized():
    handler = DataHandler()
    symbols = list(set([target]+load_symbols)) # 获取哪些symbol的数据
    conn = get_connection(DB_PATH)
    # init_db(conn)

    model_start_date = (datetime.utcnow() - START_DELTA).strftime('%Y-%m-%d %H:%M:%S')

    return conn, handler, symbols, model_start_date

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


# === Step 4: 读取最近 N 天的数据用于建模 ===
def prepare_feature_data(conn, handler, symbols, target, model_start_date):

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

# === Step 6: 建模 + 预测（时间交叉验证）===
def train_and_predict(df_processed, cv):
    X, y = split_data(df_processed)
    X_clean, y_clean = clean_xy(X, y[target_col])
    pred_df = fit_predict_regression_model(train_model, X_clean, y_clean, cv)

    return pred_df

# === Step 7: 生成交易信号 ===
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

# === Step 8: 结果存入数据库 prediction / signal 表 ===
def store_signals(df_signal_final, conn):
    upsert_df(df_signal_final, table='signals', conn=conn)

def run_pipeline():
    # 初始化数据库连接
    conn, handler, symbols, model_start_date = ensure_db_initialized()

    # 下载并存储最新价格数据
    fetch_and_store_backfill_no_lag(conn, handler, symbols, backfill_hours=4)

    # 特征工程
    df_processed, price_all = prepare_feature_data(conn, handler, symbols, target, model_start_date)

    # 模型训练与预测
    pred_df = train_and_predict(df_processed, cv=tscv)

    # 生成信号
    df_signal_final = generate_signals(pred_df, price_all, target)

    # 写入数据库
    store_signals(df_signal_final, conn)

if __name__ == '__main__':

    # from conf.settings_loader import settings
    #
    # print(settings.binance.model_dump())

    # run_pipeline()
    conn, handler, symbols, model_start_date = ensure_db_initialized()
    df_processed, price_all = prepare_feature_data(conn, handler, symbols, target, model_start_date)

    print(df_processed.info())
    print(price_all.info())


