import pandas as pd
import requests
from binance.client import Client
from datetime import datetime
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
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        return df.astype(float)

    def get_historical_klines(self, symbol='BTCUSDT', interval='1h', start_str='30 days ago UTC', end_str=None):
        raw_data = self.client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str)
        df = pd.DataFrame(raw_data)
        return self.transform_df(df)

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
        print(f"getting data for {symbol}")
        df = handler.get_historical_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str)
        df['symbol'] = symbol
        all_data.append(df)

    df_all = pd.concat(all_data)
    df_all.set_index(['symbol'], append=True, inplace=True)
    df_all = df_all.reorder_levels(['symbol', 'datetime']).sort_index()
    return df_all

if __name__ == '__main__':

    data_handler = DataHandler()
    symbols = data_handler.get_binance_symbols()[:5]
    print(symbols)
    #
    df_all = load_multi_symbol_data(data_handler, symbols, interval='1h', start_str='2025-06-01 00:00:00')
    # df_all = df_all.reset_index()
    # df_all['Date'] = df_all['Date'].dt.tz_convert(None)
    # df_all = df_all.set_index(['symbol', 'Date'])
    print(df_all.info())
    print(df_all.head())
    # df_all.to_hdf('data.h5', 'bn/price')
    #
    # df = pd.read_hdf('data.h5', 'bn/price')
    # df = df.tz_localize('UTC', level='Date')  # 恢复为带 tz 的 datetime


