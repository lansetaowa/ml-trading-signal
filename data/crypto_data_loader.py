import pandas as pd
import requests
from binance.client import Client
from datetime import datetime, timezone
import numpy as np
import time
from config import binance_proxy

class DataHandler:
    def __init__(self, api_key=None, api_secret=None):
        self.client = Client(api_key, api_secret, requests_params={
                'proxies': {
                    'http': binance_proxy,
                    'https': binance_proxy,
                    }
                })

    @staticmethod
    def transform_df(df):
        df = df.iloc[:, :6]
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True) # 转为 tz-aware UTC
        df['datetime'] = df['datetime'].dt.tz_convert(None)  # 去掉 tzinfo，统一为 naive
        df.set_index('datetime', inplace=True)
        return df.astype(float)

    def get_historical_klines(self, symbol='BTCUSDT', interval='1h', start_str='30 days ago UTC', end_str=None):
        """
        获取现货 K 线，返回 naive-UTC 时间索引。
        """
        raw_data = self.client.get_historical_klines(symbol=symbol,
                                                     interval=interval,
                                                     start_str=start_str,
                                                     end_str=end_str)
        if not raw_data:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame(raw_data)
        return self.transform_df(df)

    def get_futures_klines(
                            self,
                            symbol: str = 'BTCUSDT',
                            interval: str = '1h',
                            start_str: str = '2025-05-01 00:00:00',
                            end_str: str | None = None,
                            limit: int = 1500,
                        ):
        """
        获取合约 K 线（USDT-M/COIN-M 统一用 futures_klines），输入和输出均为 UTC 时间。
        - start_str / end_str: 'YYYY-MM-DD HH:MM:SS'（UTC）
        - interval: 例如 '1m','5m','15m','1h','4h','1d' 等
        """
        if not start_str:
            raise ValueError("Must provide start time in UTC like 'YYYY-MM-DD HH:MM:SS'")

        # ---- 1) 解析为 UTC 毫秒 ----
        def to_utc_ms(dt_str: str) -> int:
            # 把 naive 字符串当作 UTC，避免被本地时区影响
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)

        start_ms = to_utc_ms(start_str)
        end_ms = to_utc_ms(end_str) if end_str else None

        # ---- 2) interval -> 毫秒，用于翻页推进 ----
        _ms = {
            '1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
            '1h': 3_600_000, '2h': 7_200_000, '4h': 14_400_000, '6h': 21_600_000,
            '8h': 28_800_000, '12h': 43_200_000, '1d': 86_400_000, '3d': 259_200_000,
            '1w': 604_800_000, '1M': 2_592_000_000,  # 1M按30天粗略估
        }
        if interval not in _ms:
            raise ValueError(f'Unsupported interval: {interval}')
        step_ms = _ms[interval]

        # ---- 3) 分页抓取 ----
        all_rows = []
        cursor = start_ms
        while True:
            batch = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=cursor,
                endTime=end_ms,
                limit=limit
            )
            if not batch:
                break

            all_rows.extend(batch)

            # Binance kline: 每条的第0个元素是 open time (ms)
            last_open_ms = batch[-1][0]
            next_cursor = last_open_ms + step_ms

            # 退出条件：已经到尽头或这批未满
            if end_ms is not None and next_cursor > end_ms:
                break
            if len(batch) < limit:
                break

            cursor = next_cursor

        # ---- 4) 组装 DataFrame & 标准化 ----
        if not all_rows:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame(all_rows)
        df = self.transform_df(df)  # 会做：ms -> UTC，重命名列，设为索引等

        # 保险起见按时间排序
        df.sort_index(inplace=True)
        return df

    def get_binance_symbols(self, asset='USDT'):
        exchange_info = self.client.get_exchange_info()

        symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['status'] == 'TRADING'
               and s['isSpotTradingAllowed']
               and s['quoteAsset'] == asset
        ]

        return symbols

def load_multi_symbol_data(handler, symbols, interval='1h', start_str='30 days ago UTC', end_str=None):
    all_data = []
    for symbol in symbols:
        # print(f"getting data for {symbol}")
        df = handler.get_futures_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str)
        df['symbol'] = symbol
        all_data.append(df)

    df_all = pd.concat(all_data)
    df_all.set_index(['symbol'], append=True, inplace=True)
    df_all = df_all.reorder_levels(['symbol', 'datetime']).sort_index()
    return df_all

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # from data.db_utils import get_connection, upsert_df

    # conn = get_connection('crypto_data.db')

    data_handler = DataHandler()
    symbols = ['BTCUSDT', 'ETHUSDT','SOLUSDT','DOGEUSDT']
    # print(symbols)
    #
    df_all = load_multi_symbol_data(data_handler, symbols, interval='1h', start_str='2025-08-10 00:00:00')
    # df_all = df_all.reset_index()
    # df_all['datetime'] = df_all['datetime'].dt.tz_convert(None)
    # df_all = df_all.set_index(['symbol', 'Date'])
    print(df_all.info())
    # upsert_df(df_all, table='kline', conn=conn)

    # df_all.to_hdf('data.h5', 'bn/price')
    #
    # df = pd.read_hdf('data.h5', 'bn/price')
    # df = df.tz_localize('UTC', level='Date')  # 恢复为带 tz 的 datetime

    # df = data_handler.get_futures_klines(
    #     symbol='ETHUSDT',
    #     interval='1h',
    #     start_str='2025-08-10 00:00:00',
    #     end_str=None
    # )
    # print(df.index.tz)  # 应是 None
    # print(df.tail())


