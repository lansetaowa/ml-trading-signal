from dotenv import load_dotenv
import os
from sklearn.ensemble import RandomForestRegressor
from model.timeseries_cv import MultipleTimeSeriesCV

# 获取 config.py 的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load proxies
load_dotenv(os.path.join(BASE_DIR, 'proxy.env'))
binance_proxy = os.getenv('binance_proxy')

# load Binance api/key
load_dotenv(os.path.join(BASE_DIR, "binance_api.env"))
BINANCE_API_KEY = os.getenv('B_KEY')
BINANCE_API_SECRET = os.getenv('B_SECRET')

# data symbols
target = 'ETHUSDT'
load_symbols = ['BTCUSDT', 'ETHUSDT','SOLUSDT','DOGEUSDT']

# feature engineering params
feature_config = {
    "other_symbols": ['BTCUSDT', 'ETHUSDT','SOLUSDT','DOGEUSDT'],
    "lags": [1, 2, 4, 12, 24, 48, 72], # lags must contain rt_targets
    "rt_targets": [1, 2, 4, 12, 24],
    "vol_window": 24,
    "rsi_window": 12,
    "mfi_window": 12,
    "bb_window": 24,
    "tema_windows": [12,24,36],
    "adx_window": 12,
    "cmo_window": 12,
    "ulti_os_windows": {
            'period1':8,
            'period2':12,
            'period3':24
        },
    "patterns": {
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
        }
}

# cols to scale and drop
feature_process_config = {
    'metrics_to_scale': ["high", "low", "close"],
    'cols_to_drop': ["open","high","low","close","volume","dollar_vol","hour",
                "atr","macdhist",'hour_bin_label','cmo','tema_12','tema_24','tema_36']
}

# cv
tscv = MultipleTimeSeriesCV(
        train_length=24 * 7 * 4,
        test_length=24,
        lookahead=1,
        date_idx='datetime'
    )

# model
reg_rf = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=7,
    max_features='sqrt',
    max_depth=10,
    random_state=42,
    n_jobs=-1)

# signal filter params
signal_config = {
    'z_window': 24,
    'z_threshold': 1,
    'atr_windows': {
        'short': 4,
        'long': 24
    },
    'atr_threshold': 0.9
}


