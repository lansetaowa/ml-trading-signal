import pandas as pd
from pandas import IndexSlice as idx
import numpy as np
import talib

class FeatureGenerator:

    def __init__(self, config=None):
        """
        df must contain: open/high/low/close/volume and with symbol/datetime indexes (named symbol/datetime)
        """
        self.df = None
        config = config

        self.other_symbols = config.get('other_symbols', ['BTCUSDT', 'ETHUSDT','SOLUSDT','DOGEUSDT'])
        self.lags = config.get('lags', [1, 2, 4, 12, 24, 48, 72])
        self.rt_targets = config.get('rt_targets', [1, 2, 4, 12, 24])
        self.vol_window = config.get('vol_window', 24)
        self.rsi_window = config.get('rsi_window', 12)
        self.mfi_window = config.get('mfi_window', 12)
        self.bb_window = config.get('bb_window', 24)
        self.tema_windows = config.get('tema_windows', [12,24,36])
        self.adx_window = config.get('adx_window', 12)
        self.cmo_window = config.get('cmo_window', 12)
        self.ulti_os_windows = config.get('ulti_os_windows', {
            'period1':8,
            'period2':12,
            'period3':24
        })
        self.patterns = config.get('patterns', {
            'CDLENGULFING': '吞没形态',
            'CDLHAMMER': '锤子线',
            'CDLHANGINGMAN': '吊人线',
            'CDLDOJI': '十字星',
            'CDLDRAGONFLYDOJI': '蜻蜓十字星',
            'CDLGRAVESTONEDOJI': '墓碑十字星',
            'CDLMORNINGSTAR': '晨星',
            'CDLEVENINGSTAR': '暮星',
            'CDLSHOOTINGSTAR': '流星',
            'CDLMARUBOZU': '光头光脚'
        })

    def load_data(self, df):
        self.df = df.copy()
        return self

    def select_symbols(self, target_symbol='ETHUSDT'):
        all_symbols = self.other_symbols + [target_symbol]
        all_symbols = list(set(all_symbols))
        self.df = self.df.loc[idx[all_symbols,:],:]
        return self

    def compute_volume_features(self):
        self.df['dollar_vol'] = self.df[['close', 'volume']].prod(axis=1)
        self.df['log_dollar_vol'] = np.log1p(self.df['dollar_vol'])
        self.df['log_dollar_vol_3d'] = (
            self.df.groupby(level='symbol')['dollar_vol']
            .rolling(window=3 * 24)
            .mean()  # 滚动平均
            .apply(np.log1p)  # 取log
            .reset_index(level=0, drop=True)  # 使 index 对齐原始 DataFrame
        )
        return self

    def compute_momentum_features(self, cut_q = 0.0001):
        for lag in self.lags:
            self.df[f'return_{lag}h'] = (self.df.groupby(level='symbol').close
                                        .pct_change(lag)
                                        .pipe(lambda x: x.clip(lower=x.quantile(cut_q),
                                                               upper=x.quantile(1 - cut_q)))
                                        )

        for t in [1, 2, 3]:  # 滞后倍数
            for lag in self.lags:  # 原始收益窗口
                self.df[f'return_{lag}h_lag{t}'] = (self.df.groupby(level='symbol')
                                                   [f'return_{lag}h'].shift(t * lag))

        return self

    def compute_volatility_features(self):

        self.df['std_ret'] = (
            self.df.groupby(level='symbol')['return_1h']
            .rolling(window=self.vol_window)
            .std()
            .reset_index(level=0, drop=True)
        )

        self.df['skew_ret'] = (
            self.df.groupby(level='symbol')['return_1h']
            .rolling(window=self.vol_window)
            .skew()
            .reset_index(level=0, drop=True)
        )

        self.df['kurtosis_ret'] = (
            self.df.groupby(level='symbol')['return_1h']
            .rolling(window=self.vol_window)
            .kurt()
            .reset_index(level=0, drop=True)
        )

        self.df['max_ret'] = (
            self.df.groupby(level='symbol')['return_1h']
            .rolling(window=self.vol_window)
            .max()
            .reset_index(level=0, drop=True)
        )

        self.df['std_dollar_vol'] = (
            self.df.groupby(level='symbol')['log_dollar_vol']
            .rolling(window=self.vol_window)
            .std()
            .reset_index(level=0, drop=True)
        )

        return self

    def compute_target_cols(self):
        for t in self.rt_targets:
            self.df[f'target_{t}h'] = self.df.groupby(level='symbol')[f'return_{t}h'].shift(-t)
        return self

    def compute_time_dummies(self):
        self.df['weekday'] = self.df.index.get_level_values('datetime').weekday
        self.df['hour'] = self.df.index.get_level_values('datetime').hour
        self.df['hour_bin'] = (self.df['hour'] // 4)
        hour_bin_labels = {
            0: '00-03',
            1: '04-07',
            2: '08-11',
            3: '12-15',
            4: '16-19',
            5: '20-23'
        }
        self.df['hour_bin_label'] = self.df['hour_bin'].map(hour_bin_labels)
        self.df = pd.get_dummies(self.df,
                                columns=['weekday', 'hour_bin'],
                                prefix=['weekday', 'hour_bin'], drop_first=True)
        return self

    def compute_bb(self, close):
        high, mid, low = talib.BBANDS(close, timeperiod=self.bb_window)
        return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)

    def compute_tech_indicators(self):

        self.df['rsi'] = self.df.groupby(level='symbol', group_keys=False).apply(
            lambda df: talib.RSI(df.close, timeperiod=self.rsi_window))

        self.df['mfi'] = self.df.groupby(level='symbol', group_keys=False).apply(
            lambda df: talib.MFI(df.high, df.low, df.close, df.volume, timeperiod=self.mfi_window))

        self.df['atr'] = (self.df.groupby('symbol', group_keys=False)
                         .apply(lambda df: talib.ATR(df.high, df.low, df.close, timeperiod=24)))

        self.df['atr_norm'] = self.df['atr'] / self.df['open']

        self.df['macdhist'] = self.df.groupby('symbol', group_keys=False).apply(
            lambda df: talib.MACD(real=df.close, fastperiod=12, slowperiod=26, signalperiod=9)[2])
        self.df['macdhist_norm'] = self.df['macdhist'] / self.df['open']

        bb = self.df.groupby(level='symbol')['close'].apply(self.compute_bb)
        bb.index = bb.index.droplevel(0)

        self.df = self.df.join(bb)

        self.df['bb_high'] = self.df.bb_high.sub(self.df.open).div(self.df.open).apply(np.log1p)  # 上轨与当前价格的 log 比率（上轨相对偏离度）
        self.df['bb_low'] = self.df.open.sub(self.df.bb_low).div(self.df.open).apply(np.log1p)  # 下轨与当前价格的 log 比率（下轨相对偏离度）

        lags = [12, 24, 36]
        for lag in lags:
            self.df[f"tema_{lag}"] = self.df.groupby(level='symbol', group_keys=False).apply(
                lambda df: talib.TEMA(df.close, timeperiod=lag))
            self.df[f"tema_{lag}_norm"] = self.df[f"tema_{lag}"] / self.df['open']

        self.df['adx'] = self.df.groupby(level='symbol', group_keys=False).apply(
            lambda df: talib.ADX(df.high, df.low, df.close, timeperiod=self.adx_window))

        self.df['minus_di'] = self.df.groupby(level='symbol', group_keys=False).apply(
            lambda df: talib.MINUS_DI(df.high, df.low, df.close, timeperiod=self.adx_window))

        self.df['plus_di'] = self.df.groupby(level='symbol', group_keys=False).apply(
            lambda df: talib.PLUS_DI(df.high, df.low, df.close, timeperiod=self.adx_window))

        self.df['cmo'] = self.df.groupby(level='symbol', group_keys=False).apply(
            lambda df: talib.CMO(df.close, timeperiod=self.cmo_window))

        self.df['ultosc'] = (self.df.groupby('symbol', group_keys=False)
                            .apply(
            lambda df: talib.ULTOSC(df.high, df.low, df.close,
                                    timeperiod1=self.ulti_os_windows['period1'],
                                    timeperiod2=self.ulti_os_windows['period2'],
                                    timeperiod3=self.ulti_os_windows['period3']))
        )

        self.df['bop'] = self.df.groupby(level='symbol', group_keys=False).apply(lambda df: talib.BOP(df.open,
                                                                                              df.high,
                                                                                              df.low,
                                                                                              df.close))

        return self

    def compute_candle_patterns(self):
        open_ = self.df['open']
        high_ = self.df['high']
        low_ = self.df['low']
        close_ = self.df['close']

        for pattern, desc in self.patterns.items():
            func = getattr(talib, pattern)
            self.df[pattern] = func(open_, high_, low_, close_)

        return self

    def get_single_symbol_data(self, target_symbol='ETHUSDT'):

        target_data = self.df.loc[target_symbol].copy()

        selected_features = ['log_dollar_vol', 'return_1h', 'return_2h', 'return_4h', 'std_ret', 'std_dollar_vol',
                             'rsi', 'bb_high', 'bb_low', 'adx', 'plus_di', 'minus_di', 'bop']

        other_symbols = [sym for sym in self.other_symbols if sym != target_symbol]

        for sym in other_symbols:
            df = self.df.loc[sym, selected_features].copy()
            # 重命名列，加上 symbol 前缀
            sym_name = sym.replace("USDT", "")
            df.columns = [f"{sym_name}_{col}" for col in df.columns]
            target_data = target_data.join(df, how='left')

        return target_data

class FeatureProcessor:

    def __init__(self, config=None):
        self.df = None
        config = config

        self.metrics_to_scale = config.get('metrics_to_scale', ["high", "low", "close"])
        self.cols_to_drop = config.get('cols_to_drop', ["open","high","low","close","volume","dollar_vol","hour",
                "atr","macdhist",'hour_bin_label','cmo','tema_12','tema_24','tema_36'])

    def load_data(self, df):
        self.df = df.copy()
        return self

    def scale_metrics(self):

        for col in self.metrics_to_scale:
            self.df[col+'_norm'] = self.df[col] / self.df['open']

        return self

    def drop_cols(self):
        self.df = self.df.drop([c for c in self.cols_to_drop if c in self.df.columns], axis=1)
        return self

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_info_columns', 200)

    from config import feature_config, feature_process_config, target, load_symbols
    from data.crypto_data_loader import load_multi_symbol_data, DataHandler

    handler = DataHandler()
    symbols = list(set([target]+load_symbols))
    prices = load_multi_symbol_data(handler, symbols, interval='1h', start_str='2025-07-01 00:00:00')

    df_features = (
        FeatureGenerator(feature_config)
        .load_data(prices)
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

    # print(df_features.info())
    # print(df_features.tail(5))

    df_processed = (
        FeatureProcessor(config=feature_process_config)
        .load_data(df_features)
        .scale_metrics()
        .drop_cols()
        .df
    )

    print(df_processed.info())
    print(df_processed.tail(5))


















