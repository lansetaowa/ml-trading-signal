# ML Signal Trading

> 统一的数据获取 → 特征/指标 → 建模预测 → 信号 → 回测 → 可视化 → 实盘执行 的端到端项目。  
> 代码基于 `Python 3.12`（见 `.venv/`）与本地 SQLite/文件持久化。

## 项目速览
``` text 
├─ pipeline.py # 全量/批处理流水线（离线）
├─ hourly_runner.py # 定时/增量流水线（每小时）
├─ backtest/ # 回测器与策略
├─ conf/ # 配置模型 & YAML & ENV
├─ data/ # 数据与数据库工具
├─ model/ # 特征/建模/评估/时序CV/信号
├─ streamlit_app/ # 实时可视化
├─ trade/ # 实盘交易执行（合约）
├─ notebooks/ # 研究型Notebook
└─ logs/ # 运行日志
```

## 快速开始

### 1) 环境

- **Windows（推荐路径已是 Windows）**
- 建议使用项目自带的虚拟环境 `.venv/` 或自行新建

```powershell
# 若需新建：
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt   # 若无，可后续生成 
```

### 2) 配置
- conf/settings.yaml：主配置（见下方示例） 
- conf/binance_api.env：API Key/Secret
- conf/proxy.env：HTTP(S) 代理（如需）
- config.py / conf/settings_loader.py：加载与校验配置

### 3) 运行
``` powershell
# 离线全流程（按 settings.yaml 的配置跑一遍）
python pipeline.py

# 小时级增量，滚动执行（建议先用 Testnet/干跑模式）
python hourly_runner.py

# 可视化（研究/信号/价格）
python plot_signals.py
```

## 配置说明
settings.yaml（示例草案）。下方是示例，请用你现有字段替换/对齐（最终以 config_models.py 的 Pydantic 模型为准）。

```yaml
app:
  timezone: "UTC"
  logging:
    log_dir: "logs"
    runner_log: "hourly_runner.log"

paths:
  db_path: "data/crypto_data.db"
  log_path: "logs/hourly_runner.log"

binance:
  # 实际key/secret存放在 .env 中，yaml 里只留占位；真正加载时允许 env 覆盖
  api_key: "${B_KEY}"
  api_secret: "${B_SECRET}"
  proxy_env: "proxy.env"
  api_env: "binance_api.env"
  http_proxy: "${binance_proxy}"

data:
  interval: "1h"
  start_delta_days: 60
  model_start_from_now: true  # 以 NOW-60d 作为训练数据起点
  symbols:
    target: "ETHUSDT"
    load_symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
```

### 环境文件
- conf/binance_api.env
``` ini
BINANCE_API_KEY=YOUR_KEY
BINANCE_API_SECRET=YOUR_SECRET
```
- conf/proxy.env
``` ini
HTTP_PROXY=YOUR_PROXY
HTTPS_PROXY=YOUR_PROXY
```

### 配置加载
conf/config_models.py：Pydantic 模型定义（字段、默认值、校验）
conf/settings_loader.py：合并 ENV + YAML，导出 settings（供各模块使用）

## 数据层
- SQLite：data/crypto_data.db
- 工具：data/db_utils.py、data/crypto_data_loader.py
- K线时间：项目采用naive UTC（无 tz，但含义为 UTC），与 DataHandler.transform_df 保持一致。

``` 
表结构：
Table: kline(symbol, datetime, open, high, low, close, volume)
Table: signals(symbol, datetime, actuals, predicted, zscore, raw_signal, vol_filter, filtered_signal, position, signal_reversal, final_signal, model_name, strategy_name)
```

## 模块职责地图
``` text 
pipeline.py
  └─ 调用 model/ & data/ & predict/ & signal/，完成「拉取→特征→训练→预测→信号→落库」

hourly_runner.py
  └─ 循环/定时窗口化执行（近1小时/日），包含数据pipeline以及信号实盘执行

backtest/
  ├─ strategies.py     # 根据信号回测规则/参数（TP/SL/ATR/费用等）
  └─ backtest.py       # 回测引擎（收益/回撤/Sharpe）

data/
  ├─ crypto_data_loader.py     # 从币安获取kline数据，包含合约和现货
  └─ db_utils.py       # 和本地sqlite数据库交互

model/
  ├─ feature_generator.py  # 指标与特征生成
  ├─ fit_pred.py           # 拟合/预测（封装不同模型）
  ├─ timeseries_cv.py      # 时序交叉验证，避免未来数据泄露
  ├─ signal_generator.py   # zscore阈值/atr过滤/持仓冷却/上下限等
  └─ model_evaluator.py    # 评估指标、参数选择与对比

trade/
  ├─ future_trader.py      # 合约的下单/撤单/止盈止损/仓位/查询
  └─ signal_executor.py    # 将信号转换为订单（反转、平仓、限价/市价策略）

streamlit_app/
  └─ streamlit_app.py      # （仅限本地）实时读取数据/信号/回测结果进行展示
  └─ plot_signals.py       # 可选读取数据/信号/回测结果进行展示
```

## 典型工作流
1. 离线研究（notebooks/）
   - 载入 feature_generator/fit_pred/signal_generator/backtest，做参数调整 
   - 可视化（streamlit_app/）
2. 批处理/回放（pipeline.py）
    - 一致性地复现离线最佳参数
3. 准实盘/仿真（hourly_runner.py with trade.testnet=true 或 dry-run）
4. 实盘（trade.testnet=false，谨慎切换，先小仓位）

## 日志与排障
- 日志文件：logs/hourly_runner.log
- 常见问题：
    - 时区/时间戳：确保所有 DataFrame 索引是naive UTC；插库前去 tz。 
    - API 限速/代理：检查 conf/proxy.env 与 binance_api.env 是否加载。
    - Upsert 行为：data/db_utils.py 的 upsert 逻辑是否仅更新 actuals 字段（见下方 TODO）。

## 安全与风控（实盘前必读）
- trade/future_trader.py 支持 testnet 与 主网 切换（在 settings.yaml中设置）。
- 开仓/平仓逻辑：
    - 空仓遇到 final_signal_filtered == 1 → 先平空再开多；反之亦然。
    - 止盈止损：限价优先（避免市价滑点过大），可按 ATR/Break-even 调整。
- 资金费率与多交易所价差：若参与套利，请确认资金费率采集/估价逻辑的正确性与时效性。

