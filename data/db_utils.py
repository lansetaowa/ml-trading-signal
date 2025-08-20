# db_utils.py

import sqlite3
import pandas as pd
from typing import Literal

def get_connection(db_path='crypto_data.db'):
    return sqlite3.connect(db_path)

def init_db(conn):
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS kline (
        symbol TEXT,
        datetime TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        PRIMARY KEY (symbol, datetime)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signals (
        symbol TEXT,
        datetime TEXT,
        actuals REAL,
        predicted REAL,
        zscore REAL,
        raw_signal INTEGER,
        vol_filter BOOLEAN,
        filtered_signal INTEGER,
        position INTEGER,
        signal_reversal INTEGER,
        final_signal INTEGER,
        model_name TEXT,
        strategy_name TEXT,
        PRIMARY KEY (symbol, datetime, model_name, strategy_name)
    )
    ''')

    conn.commit()

def upsert_df(df: pd.DataFrame, table: Literal['kline', 'signals'], conn):
    cursor = conn.cursor()

    if table == 'kline':
        insert_sql = '''
            INSERT INTO kline (symbol, datetime, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, datetime) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
        '''
        df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(None).astype(str)


    elif table == 'signals':
        # 仅更新actuals列，更新之前未完成的bar信息
        insert_sql = '''
            INSERT INTO signals
            (symbol, datetime, actuals, predicted, zscore,
             raw_signal, vol_filter, filtered_signal,
             position, signal_reversal, final_signal,
             model_name, strategy_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, datetime, model_name, strategy_name) DO UPDATE SET
                actuals=excluded.actuals
        '''
        df = df[['symbol', 'datetime', 'actuals', 'predicted', 'zscore',
                 'raw_signal', 'vol_filter', 'filtered_signal',
                 'position', 'signal_reversal', 'final_signal',
                 'model_name', 'strategy_name']].copy()

        # 统一成无 tz 的 UTC 字符串，避免主键冲突的“伪不同”
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(None).astype(str)

    else:
        raise ValueError(f"Unsupported table name: {table}")

    cursor.executemany(insert_sql, df.values.tolist())
    conn.commit()

def get_last_timestamp(conn, table: str) -> pd.Timestamp | None:
    query = f'''
    SELECT MAX(datetime) as max_dt FROM {table}
    '''
    df = pd.read_sql_query(query, conn)

    return pd.to_datetime(df['max_dt'].iloc[0]) if pd.notnull(df['max_dt'].iloc[0]) else None

if __name__ == '__main__':
    # from data.crypto_data_loader import load_multi_symbol_data, DataHandler
    # pd.set_option('display.max_columns', None)
    #
    # handler = DataHandler()
    # symbols = ['BTCUSDT','ETHUSDT']
    # df = load_multi_symbol_data(handler, symbols, interval='1h', start_str='2025-07-01 00:00:00')
    # df.reset_index(inplace=True)
    #
    conn = get_connection()
    print(conn)
    # init_db(conn=conn)
    # # upsert_df(df=df, table='kline', conn=conn)
    #
    # latest = read_latest_data(conn, table='kline', symbol='BTCUSDT', limit=5)
    # print(latest)

    # df1 = pd.DataFrame({
    #     'symbol': ['ETHUSDT', 'ETHUSDT'],
    #     'datetime': ['2024-08-01 10:00:00', '2024-08-01 11:00:00'],
    #     'open': [1000, 1010],
    #     'high': [1010, 1020],
    #     'low': [990, 1005],
    #     'close': [1005, 1015],
    #     'volume': [100, 150],
    # })
    #
    # upsert_df(df1, 'kline', conn)


