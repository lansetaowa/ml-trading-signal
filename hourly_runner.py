import schedule
import time
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# ====== run_pipeline 导入路径 ======
from pipeline import run_pipeline

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

def job():
    logging.info("==== run_pipeline START ====")
    try:
        run_pipeline()
        logging.info("==== run_pipeline DONE ====")
    except Exception as e:
        logging.exception(f"run_pipeline FAILED: {e}")

job() # 启动后立即先跑一次
schedule.every().hour.at(":01").do(job) # 每个小时的第 1 分钟执行一次

logging.info("Scheduler started. Waiting for the first run...")

while True:
    schedule.run_pending()
    time.sleep(1)
