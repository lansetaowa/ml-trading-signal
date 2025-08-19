from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# === Leaf sections ===

class AppLogging(BaseModel):
    log_dir: str = "logs"
    runner_log: str = "hourly_runner.log"

class AppConfig(BaseModel):
    timezone: str = "UTC"
    logging: AppLogging = AppLogging()

class PathsConfig(BaseModel):
    db_path: str
    log_path: str

class BinanceConfig(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    proxy_env: str = "proxy.env"
    api_env: str = "binance_api.env"
    http_proxy: Optional[str] = None

class DataSymbols(BaseModel):
    target: str
    load_symbols: List[str]

    @field_validator("load_symbols")
    def include_target(cls, v, values):
        tgt = values.get("target")
        if tgt and tgt not in v:
            return v + [tgt]
        return v

class DataConfig(BaseModel):
    interval: Literal["1m","5m","15m","1h","4h","1d"] = "1h"
    start_delta_days: int = 60
    model_start_from_now: bool = True
    symbols: DataSymbols

class FeaturesUlti(BaseModel):
    period1: int = 8
    period2: int = 12
    period3: int = 24

class FeaturesConfig(BaseModel):
    other_symbols: List[str]
    lags: List[int]
    rt_targets: List[int]
    vol_window: int
    rsi_window: int
    mfi_window: int
    bb_window: int
    tema_windows: List[int]
    adx_window: int
    cmo_window: int
    ulti_os_windows: FeaturesUlti
    patterns: Dict[str, str] = Field(default_factory=dict)

    @field_validator("lags")
    def lags_include_rt_targets(cls, lags, values):
        rts = values.get("rt_targets", [])
        missing = [t for t in rts if t not in lags]
        if missing:
            raise ValueError(f"lags 必须包含 rt_targets：缺少 {missing}")
        return lags

class FeatureProcessConfig(BaseModel):
    metrics_to_scale: List[str]
    cols_to_drop: List[str]

class CVConfig(BaseModel):
    train_length: int
    test_length: int
    lookahead: int = 1
    date_idx: str = "datetime"

class ModelConfig(BaseModel):
    name: str
    type: Literal["RandomForestRegressor"] = "RandomForestRegressor"
    params: Dict

class SignalATRWindows(BaseModel):
    short: int = 4
    long: int = 24

class SignalConfig(BaseModel):
    z_window: int = 24
    z_threshold: float = 1.0
    atr_windows: SignalATRWindows = SignalATRWindows()
    atr_threshold: float = 0.9
    strategy_name: str = "zscore_atr_v1"

class TradingExecConfig(BaseModel):
    symbol: str
    dualSidePosition: bool = True
    use_balance_ratio: float = 0.95
    atr_period: int = 12
    atr_k: float = 1.5
    slippage_ticks: int = 2

class TradingSchedule(BaseModel):
    minute_at: int = 1  # 每小时第 minute_at 分，0-59

class TradingConfig(BaseModel):
    enabled: bool = True
    exec: TradingExecConfig
    schedule: TradingSchedule

# === Root ===

class Settings(BaseModel):
    app: AppConfig
    paths: PathsConfig
    binance: BinanceConfig
    data: DataConfig
    features: FeaturesConfig
    feature_process: FeatureProcessConfig
    cv: CVConfig
    model: ModelConfig
    signal: SignalConfig
    trading: TradingConfig
