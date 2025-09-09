import schedule
import os
import time
import logging
from logging.handlers import RotatingFileHandler

# ====== run_pipeline 导入路径 ======
from pipeline import run_pipeline

# ====== 交易执行：按信号下单 ======
from trade.future_trader import BinanceFutureTrader
from trade.signal_executor import SignalExecutor, DBConfig, ExecConfig

from conf.settings_loader import settings

# ====== 日志配置 ======
log_file = settings.paths.log_path
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(),  # 输出到控制台
        RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"),  # 保存到文件，最大5MB，保留5个
    ])

# ====== 全局初始化（避免每次调度重复创建客户端）======
trader = BinanceFutureTrader(api_key=settings.binance.api_key,api_secret=settings.binance.api_secret)
db_cfg = DBConfig(db_path=settings.paths.db_path)
exec_cfg = ExecConfig(**settings.trading.exec.model_dump())

executor = SignalExecutor(trader, db_cfg, exec_cfg)

def job():
    logging.info("==== run_pipeline START (30m) ====")
    try:
        run_pipeline()
        logging.info("==== run_pipeline DONE ====")
    except Exception as e:
        logging.exception(f"run_pipeline FAILED: {e}")
        return # 数据没跑通就别下单了

    if settings.trading.enabled:
        logging.info("==== signal_executor START ====")
        try:
            executor.run_once()
            logging.info("==== signal_executor DONE ====")
        except Exception as e:
            logging.exception(f"signal_executor FAILED: {e}")
    else:
        logging.info("trading.enabled = false → 跳过下单执行。")

if __name__ == '__main__':

    logging.info("Scheduler started. First Run...")
    job() # 启动后立即先跑一次

    # 以 30 分钟节奏运行：给K线收盘留1分钟缓冲
    schedule.every().hour.at(":01").do(job)
    schedule.every().hour.at(":31").do(job)

    logging.info("Scheduler started. Running every 30 minutes at :01 and :31 ...")
    while True:
        schedule.run_pending()
        time.sleep(1)
