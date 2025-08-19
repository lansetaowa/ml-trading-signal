import os
import yaml
from dotenv import load_dotenv
from conf.config_models import Settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
CONFIG_PATH = os.path.join(BASE_DIR, "conf", "settings.yaml")

def _expand_env(value: str) -> str:
    # 支持 ${VAR} 展开
    if isinstance(value, str) and "${" in value:
        return os.path.expandvars(value)
    return value

def _expand_env_in_dict(d):
    if isinstance(d, dict):
        return {k: _expand_env_in_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_expand_env_in_dict(v) for v in d]
    return _expand_env(d)

def load_settings() -> Settings:
    # 1) 先加载 .env（包括 proxy.env 和 binance_api.env）
    #    注意：settings.yaml 里提供了两个 env 文件名
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 先加载 proxy.env / api.env（如果存在）
    config_binance = raw.get("binance", {})
    proxy_env = config_binance.get("proxy_env", "proxy.env")
    api_env = config_binance.get("api_env", "binance_api.env")

    load_dotenv(os.path.join(BASE_DIR, "conf", proxy_env))
    load_dotenv(os.path.join(BASE_DIR, "conf", api_env))
    load_dotenv(os.path.join(BASE_DIR, ".env"))  # 允许根目录额外 .env

    # 2) 展开 ${VAR} 引用，并允许环境变量覆盖
    expanded = _expand_env_in_dict(raw)

    # 允许通过环境变量 TRADING_ENABLED 覆盖
    te = os.getenv("TRADING_ENABLED")
    if te is not None:
        expanded["trading"]["enabled"] = te == "1"

    # 3) 解析成 Pydantic Settings
    return Settings(**expanded)

# 模块级单例
settings = load_settings()

if __name__ == '__main__':
    print(settings.app)
