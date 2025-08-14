import schedule
import os
import time
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# ====== run_pipeline 导入路径 ======
from pipeline import run_pipeline

# ====== 交易执行：按信号下单 ======
from trade.future_trader import BinanceFutureTrader
from trade.signal_executor import SignalExecutor, DBConfig, ExecConfig
from config import BINANCE_API_KEY, BINANCE_API_SECRET

# ====== 日志配置 ======
log_file = "logs/hourly_runner.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"),  # 保存到文件，最大5MB，保留5个
    ]
)

# ====== 全局初始化（避免每次调度重复创建客户端）======
trader = BinanceFutureTrader(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

db_cfg = DBConfig(
    db_path="data/crypto_data.db",  # ← 改成你的 SQLite 路径
    signals_table="signals",
    klines_table="kline",
    signals_symbol_col="symbol",
    signals_value_col="final_signal",
    klines_symbol_col="symbol",
    k_open_col="open",
    k_high_col="high",
    k_low_col="low",
    k_close_col="close",
    k_time_col="datetime",
)

exec_cfg = ExecConfig(
    symbol="ETHUSDT",
    dualSidePosition=True,  # 当前是双向
    use_balance_ratio=0.5,  # 用多少比例的余额开仓
    atr_period=12,
    atr_k=1.5,
    slippage_ticks=2
)

executor = SignalExecutor(trader, db_cfg, exec_cfg)

# 可选：环境变量开关（例如线上想临时停掉真实下单）
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "1") == "1"


def job():
    logging.info("==== run_pipeline START ====")
    try:
        run_pipeline()
        logging.info("==== run_pipeline DONE ====")
    except Exception as e:
        logging.exception(f"run_pipeline FAILED: {e}")
        # 数据没跑通就别下单了
        return

    if not TRADING_ENABLED:
        logging.warning("TRADING_ENABLED=0 → 跳过下单执行。")
        return

    logging.info("==== signal_executor START ====")
    try:
        executor.run_once()
        logging.info("==== signal_executor DONE ====")
    except Exception as e:
        logging.exception(f"signal_executor FAILED: {e}")

job() # 启动后立即先跑一次
schedule.every().hour.at(":01").do(job) # 每个小时的第 1 分钟执行一次

logging.info("Scheduler started. Waiting for the first run...")

while True:
    schedule.run_pending()
    time.sleep(1)
