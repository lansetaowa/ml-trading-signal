import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import datetime, timedelta, timezone

from data.crypto_data_loader import DataHandler, load_multi_symbol_data
from data.db_utils import get_connection, init_db, get_last_timestamp, upsert_df

from model.feature_generator import FeatureGenerator, FeatureProcessor
from model.timeseries_cv import MultipleTimeSeriesCV
from model.fit_pred import split_data, clean_xy, fit_predict_regression_model
from model.signal_generator import SignalGenerator

from config import feature_config, feature_process_config, target, load_symbols, reg_model, signal_config

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_info_columns', 200)

# --- 配置部分 ---
INTERVAL = '1h'
START_DELTA = timedelta(days=90)
model_start_date = (datetime.utcnow() - START_DELTA).strftime('%Y-%m-%d %H:%M:%S')
DB_PATH = 'data/crypto_data.db'
MODEL_NAME = 'rf-reg'
STRATEGY_NAME = 'zscore_atr_v1'
symbols = list(set([target]+load_symbols)) # 获取哪些symbol的数据
handler = DataHandler()
target_col = 'target_1h'

# === Step 1: 获取数据库连接和最新时间 ===
conn = get_connection(DB_PATH)
init_db(conn)
last_dt = get_last_timestamp(conn, table='kline')
# print(last_dt)

start_str = (
    (last_dt + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S') if last_dt
    else (datetime.utcnow() - START_DELTA).strftime('%Y-%m-%d %H:%M:%S')
)
# print(start_str)

# === Step 2: 下载最新价格数据 ===
if datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
    df_price = load_multi_symbol_data(handler, symbols, start_str=start_str)
    df_price.reset_index(inplace=True)
    df_price = df_price[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']]

# print(df_price.info())
# print(df_price.head())

# === Step 3: 存入数据库 kline 表 ===
    if not df_price.empty:
        print(f'upserting {len(df_price)} lines of price data...')
        upsert_df(df_price, table='kline', conn=conn)

else:
    print('Price data is up to date.')

# === Step 4: 读取最近 N 天的数据用于建模 ===
price_all = pd.read_sql_query(
    f"SELECT * FROM kline WHERE datetime >= '{model_start_date}' ORDER BY datetime",
    conn,
    parse_dates=['datetime']
)

price_all.set_index(['symbol', 'datetime'], inplace=True)

# print(price_all.info())

# === Step 5: 特征工程 ===
df_features = (
        FeatureGenerator(feature_config)
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

# print(df_processed.info())
# print(df_processed.tail(5))

# === Step 6: 建模 + 预测（时间交叉验证）===
X, y = split_data(df_processed)
X_clean, y_clean = clean_xy(X, y[target_col])
# print(X_clean.index.get_level_values('datetime').max())
# print(y_clean.index.get_level_values('datetime').max())

# print(X_clean.shape)
# print(y_clean.shape)

cv = MultipleTimeSeriesCV(
    train_length=24 * 7 * 4,
    test_length=24,
    lookahead=1,
    date_idx='datetime'
)

pred_df = fit_predict_regression_model(reg_model, X_clean, y_clean, cv)

# === Step 7: 生成交易信号 ===
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

# print(df_signal.info())
# print(df_signal.tail())

# === Step 8: 结果存入数据库 prediction / signal 表 ===
df_signal = df_signal.reset_index()
df_signal['symbol'] = target
df_signal_pred = df_signal[['symbol', 'datetime', 'predicted']]
df_signal_pred.rename(columns={'predicted': 'predicted_return'}, inplace=True)
df_signal_pred['model_name'] = MODEL_NAME

df_signal_final = df_signal[['symbol', 'datetime', 'final_signal']]
df_signal_final['strategy_name'] = STRATEGY_NAME

print(f'upserting pred table')
upsert_df(df_signal_pred, table='prediction', conn=conn)
print(f'upserting signal table')
upsert_df(df_signal_final, table='signal', conn=conn)