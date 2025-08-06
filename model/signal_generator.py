import pandas as pd
from pandas import IndexSlice as idx
import numpy as np
import talib

class SignalGenerator:

    def __init__(self, config):

        """
        pred_df must contain: actuals, predicted, with datetime index
        price_df must contain: open, high, low, close, volume, with datetime index

        they must be about the same symbol
        """

        self.pred_df = None
        self.price_df = None

        config = config

        self.z_window = config.get('z_window', 24)
        self.z_threshold = config.get('z_threshold', 1)
        self.atr_windows = config.get('atr_windows', {
            'short': 4,
            'long': 24
        })
        self.atr_threshold = config.get('atr_threshold', 0.9)


    def load_data(self, pred_df, price_df):
        self.pred_df = pred_df.copy()
        self.price_df = price_df.copy()

        return self

    # === Step 0: 合并pred和price ===
    def merge_pred_price(self, target_symbol):

        prices = self.price_df.loc[idx[target_symbol, :], :].droplevel('symbol')
        prices.columns = prices.columns.str.lower()

        self.pred_df = self.pred_df.join(prices, how='left')

        return self

    # === Step 1: 预测值 rolling z-score 标准化 ===
    def zscore_normalize(self):

        self.pred_df['zscore'] = self.pred_df['predicted'].rolling(self.z_window).apply(
            lambda x: (x[-1] - np.mean(x)) / np.std(x), raw=True
        )

        return self

    # === Step 2: 初始信号 ===
    def compute_raw_signal(self):

        self.pred_df['raw_signal'] = 0
        self.pred_df.loc[self.pred_df['zscore'] > self.z_threshold, 'raw_signal'] = 1
        self.pred_df.loc[self.pred_df['zscore'] < -self.z_threshold, 'raw_signal'] = -1

        return self

    # === Step 3: 加入波动率过滤器 ===
    def apply_atr_volatility_filter(self):

        high = self.pred_df['high']
        low = self.pred_df['low']
        close = self.pred_df['close']

        atr_short = talib.ATR(high, low, close, timeperiod=self.atr_windows['short'])
        atr_long = talib.ATR(high, low, close, timeperiod=self.atr_windows['long'])

        self.pred_df['vol_filter'] = atr_short > atr_long * self.atr_threshold

        # 应用过滤器
        self.pred_df['filtered_signal'] = self.pred_df['raw_signal']
        self.pred_df.loc[ ~ self.pred_df['vol_filter'], 'filtered_signal'] = 0

        return self

    # === Step 4: 模拟持仓，避免频繁反转 ===
    # 只有当 signal 连续保持不同方向才切换，否则维持原持仓
    def compute_positions(self):
        self.pred_df['position'] = 0
        for i in range(1, len(self.pred_df)):
            prev = self.pred_df.iloc[i - 1]['position']
            curr = self.pred_df.iloc[i]['filtered_signal']
            if curr == 0:
                self.pred_df.iloc[i, self.pred_df.columns.get_loc('position')] = prev  # 保持原持仓
            elif curr != prev:
                self.pred_df.iloc[i, self.pred_df.columns.get_loc('position')] = curr  # 方向变化才切换
            else:
                self.pred_df.iloc[i, self.pred_df.columns.get_loc('position')] = prev  # 同方向，继续持有

        return self

    # === Step 5: 找出持仓反转的点 ===
    def compute_reversals(self):

        self.pred_df['signal_reversal'] = 0
        self.pred_df.loc[(self.pred_df['position'] == 1) & (self.pred_df['position'].shift(1) != 1), 'signal_reversal'] = 1
        self.pred_df.loc[(self.pred_df['position'] == -1) & (self.pred_df['position'].shift(1) != -1), 'signal_reversal'] = -1

        return self

    # === Step 6: 最后一步，每隔几个bar才发出信号，生成final_signal ===
    def apply_min_signal_spacing(self, min_space=2):

        df = self.pred_df.copy()
        df['final_signal'] = 0

        last_signal_time = None

        for idx in df.index:
            sig = df.at[idx, 'signal_reversal']
            if sig == 0:
                continue
            if last_signal_time is None or (idx - last_signal_time).total_seconds() >= min_space * 3600:
                df.at[idx, 'final_signal'] = sig
                last_signal_time = idx

        return df

if __name__ == '__main__':
    pass
    # from config import signal_config
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_info_columns', 200)
    #



