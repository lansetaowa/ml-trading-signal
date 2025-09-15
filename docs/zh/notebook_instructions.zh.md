## Notebook使用指南

### 1. Notebook目标
在 Notebook 中复现“从数据 → 特征 → 训练/预测 → 生成信号 → 策略回测”的闭环，用于测试新模型和定期进行参数调优。

### 2. 步骤
#### Step1 导入依赖，初始化方法
依赖包含：
- 基础：pandas/datetime/matplotlib
- 项目路径: sys/pathlib
- 项目模块：
  - data: Binance合约数据导入
  - model: 
    - feature_generator 数据清洗/特征工程
    - fit_pred 模型训练/预测
    - model_evaluator 模型评估
    - signal_generator 信号生成
    - timeseries_cv 时序cv，确保未来数据不泄露
  - streamlit_app: plot_signals/plot_recent_signals 可视化信号
  - backtest: 
    - strategies 交易策略
    - backtest 回测执行，效果评估
#### Step2 设置相关参数
在notebook里手动设置参数，在实盘中用conf/settings.yaml统一设置。
- symbol参数
  - target 目标symbol
  - load_symbols 获取哪些symbols
- 数据拉取参数
  - interval k线interval
  - START_DELTA 数据拉取起点
  - target_col/target_cols 用于模型评估
- 特征工程参数
  - other_symbols 作为特征的其他symbols
  - lags lag窗口
  - rt_targets 未来return窗口
  - vol/rsi/mfi/bb/tema/adx/cmo/ulti_os 窗口
  - patterns k线形态
- 数据清洗参数
  - metrics_to_scale 需要缩放的特征
  - cols_to_drop 要去掉的列
- 时序cv参数
  - train_length 训练窗口长度
  - test_length 预测窗口长度，在实操中，一般为1
  - look_ahead 训练和测试之间隔开的
- 模型参数，目前是random forest
- 信号生成参数
  - z_window 生成zscore所需窗口数
  - z_threshold 过滤z值，生成原始信号
  - atr_windows 用atr-short > atr-long作为波动性过滤
  - atr_threshold 短期atr达到长期atr的比例作为过滤门槛
#### Step3 从Binance Futures API拉数
这一步，生成df_price：
- 时间范围：START_DELTA算出起始时间，至今
- interval为设置值
- symbols包括目标和伴随
- 来自Binance合约数据
#### Step4 特征工程
这一步，对上述df_price进行处理，生成特征数据df_processed
- 每个时间点有一行数据，最后一行数据是当前未完全形成的
- 包含目标symbol的特征和target值
- 可调参数包含：
  - 各技术指标的参数，如窗口值
  - drop的列
#### Step5 生成预测
这一步：
1. 将上述df_processed拆分为X和y，并去除缺失数据的行（即未形成完整k线的当前行）
2. 用tscv，滚动生成预测值，形成pred_df，包含列datetime和predicted

可调整参数：
- model：可尝试多种regression模型，以及对应参数。目前是random forest
- tscv：可尝试不同训练数据时长。目前是24*7*4
#### Step6 模型参数调整/生成预测/模型效果评估 （反复）
#### Step7 生成信号（这边要修改成滚动增加模式，而不是一次性生成）/可视化信号
这一步：对predicted列进行处理，生成最终交易信号final_signal
1. upload预测和价格数据，合并
2. 对predicted列进行zscore normalize。具体而言，计算当前predicted值相对过去24个predicted值分布中的z值。
3. 对上述z值，大于或小于阈值，赋值1或-1，生成raw_signal.
4. 进行波动率过滤。具体来说，短期atr大于长期atr。生成filtered_signal.
5. 用filtered_signal计算持仓
6. 根据上述持仓，生成持仓反转点
7. 根据上述持仓反转点，进行信号冷却，即起码过n个间隔之后才能再出现信号，避免频繁交易。目前n=2.

可调整参数（在signal_config里）：
- z值窗口
- z值过滤的阈值
- atr的长期、短期窗口
- atr短期大于长期窗口的倍数阈值
- 信号冷却窗口数

#### Step8 回测
