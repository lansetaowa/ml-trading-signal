"""
预测完数据后：
读取signals表里最后一条数据的final signal
- 如果是0，什么都不做
- 如果是1/-1：
    - 先检查目前是否有仓位
        - 如果没有仓位，但是还有止盈或止损的委托，就取消它们
        - 再计算当前的atr（period=12），用于后续设置止盈止损
            - ohlcv数据来源于klines表，排除最后一条数据，因为还不全
    - 如果是1：
        - 如果目前已有多仓，那么什么都不做
        - 如果目前已有空仓，那么先平空仓，再开多仓
        - 如果目前没有仓位，那么开多仓
    - 如果是-1：
        - 如果目前已有多仓，那么先平多仓，再开空仓
        - 如果目前已有空仓，那么什么都不做
        - 如果目前没有仓位，开空仓

每次开仓，都开市价单，然后获取entry_price，用 +/-atr*1.5 来设置止盈/止损价格，然后下限价止盈止损单

其他：
- 下单金额：用合约账户里所有的钱
- 止盈止损的quantity和开仓的quantity相同，即全部平仓
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple
import time

import numpy as np
import pandas as pd

from trade.future_trader import BinanceFutureTrader

@dataclass
class DBConfig:
    db_path: str
    signals_table: str = "signals"
    klines_table: str = "kline"
    # 列名配置（按你自己的库结构调整）
    signals_symbol_col: str = "symbol"
    signals_value_col: str = "final_signal"           # 信号列名

    klines_symbol_col: str = "symbol"
    k_open_col: str = "open"
    k_high_col: str = "high"
    k_low_col: str = "low"
    k_close_col: str = "close"
    k_time_col: Optional[str] = "datetime"

@dataclass
class ExecConfig:
    symbol: str
    dualSidePosition: bool = True           # 当前是双向
    use_balance_ratio: float = 1.0          # 用全部余额
    atr_period: int = 12
    atr_k: float = 1.5
    slippage_ticks: int = 2                 # 触发后挂单限价的 tick 偏移，增强成交概率
    max_position_close_retries: int = 5     # 下单后读取仓位重试次数
    position_retry_sleep_sec: float = 0.5  # 每次重试间隔


class SignalExecutor:
    def __init__(self, trader: BinanceFutureTrader, db_cfg: DBConfig, exec_cfg: ExecConfig):
        self.trader = trader
        self.db = db_cfg
        self.cfg = exec_cfg

    # ---------------- DB helpers ----------------
    def _get_latest_signal(self) -> Optional[int]:
        """
        读取 signals 表最后一条的信号值。优先按 symbol 过滤；若没有 symbol 列则直接取最后一行。
        返回：1 / -1 / 0 / None
        """
        conn = sqlite3.connect(self.db.db_path)
        q = f"""
            SELECT {self.db.signals_value_col}
            FROM {self.db.signals_table}
            WHERE {self.db.signals_symbol_col} = ?
            ORDER BY datetime DESC
            LIMIT 1
        """
        row = pd.read_sql_query(q, conn, params=[self.cfg.symbol])

        if row.empty:
            return None
        val = row.iloc[0, 0]
        try:
            return int(val)
        except Exception:
            # 兼容 float/bool
            return int(float(val))
        finally:
            conn.close()

    def _load_ohlcv(self) -> pd.DataFrame:
        """
        读取 klines（按 symbol 过滤、按时间升序），并丢弃最后一行（未完成 bar）。
        必要列：open/high/low/close；大小写不敏感，具体列名由 DBConfig 提供。
        """
        conn = sqlite3.connect(self.db.db_path)
        try:
            q = f"""
                SELECT *
                FROM {self.db.klines_table}
                WHERE {self.db.klines_symbol_col} = ?
                ORDER BY {self.db.k_time_col if self.db.k_time_col else 'ROWID'} ASC
            """
            df = pd.read_sql_query(q, conn, params=[self.cfg.symbol])

            if df.empty:
                raise ValueError("klines 表为空")

            # 去掉最后一条未完成 bar
            if len(df) > 1:
                df = df.iloc[:-1, :]

            # 统一列名访问
            rename_map = {
                self.db.k_open_col: "open",
                self.db.k_high_col: "high",
                self.db.k_low_col: "low",
                self.db.k_close_col: "close",
            }
            df = df.rename(columns=rename_map)
            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    raise ValueError(f"klines 缺少列: {col}")
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["open", "high", "low", "close"])

            return df
        finally:
            conn.close()

    # ---------------- Indicators ----------------
    @staticmethod
    def _atr_wilder(df: pd.DataFrame, period: int) -> pd.Series:
        """
        计算 Welles Wilder ATR。
        TR = max(H-L, |H-PrevC|, |L-PrevC|)
        ATR = EMA(alpha=1/period) of TR（Wilder 平滑）
        """
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        return atr

    def _compute_atr(self) -> float:
        df = self._load_ohlcv()
        atr = self._atr_wilder(df, self.cfg.atr_period)
        if atr.dropna().empty:
            raise ValueError("ATR 计算失败（有效数据不足）")
        return float(atr.dropna().iloc[-1])

    # ---------------- Binance helpers ----------------

    def _get_mark_price(self) -> float:
        mp = self.trader.client.futures_mark_price(symbol=self.cfg.symbol)
        return float(mp["markPrice"])

    def _get_position_qtys(self) -> Tuple[float, float]:
        """
        返回 (long_qty, short_qty)；对 one-way 模式也做了兼容（并到 long/short）
        """
        pos = self.trader.get_position(self.cfg.symbol, dualSidePosition=self.cfg.dualSidePosition)
        if self.cfg.dualSidePosition:
            long_qty = abs(float(pos["LONG"]["positionAmt"]))
            short_qty = abs(float(pos["SHORT"]["positionAmt"]))
        else:
            qty = float(pos["positionAmt"])
            long_qty = max(qty, 0.0)
            short_qty = max(-qty, 0.0)
        return long_qty, short_qty

    def _cancel_symbol_conditional_orders(self):
        """
        取消该 symbol 下的 TP/SL 条件单（TAKE_PROFIT[_MARKET] / STOP[_MARKET] / TRAILING_STOP_MARKET）。
        避免无仓位时仍残留条件单。
        """
        open_orders = self.trader.client.futures_get_open_orders(symbol=self.cfg.symbol)
        types_to_cancel = {
            "TAKE_PROFIT", "TAKE_PROFIT_MARKET",
            "STOP", "STOP_MARKET",
            "TRAILING_STOP_MARKET"
        }
        for od in open_orders:
            if od.get("type") in types_to_cancel:
                try:
                    self.trader.client.futures_cancel_order(
                        symbol=self.cfg.symbol, orderId=od["orderId"]
                    )
                    print(f"✅ 已取消遗留条件单: id={od['orderId']}, type={od['type']}")
                except Exception as e:
                    print(f"❌ 取消订单失败 id={od.get('orderId')}：{e}")

    def _close_side_if_any(self, position_side: str):
        """
        若该方向有仓位，则以市价全平。
        position_side: 'LONG' | 'SHORT'
        """
        long_qty, short_qty = self._get_position_qtys()
        qty = long_qty if position_side == "LONG" else short_qty
        if qty > 0:
            side = "SELL" if position_side == "LONG" else "BUY"
            print(f"→ 平 {position_side} 仓, qty={qty}")
            self.trader.place_market_order(
                symbol=self.cfg.symbol,
                side=side,
                quantity=self.trader.round_to_size(qty, self.trader.get_step_size(self.cfg.symbol)),
                dualSidePosition=self.cfg.dualSidePosition,
                positionSide=position_side
            )

    def _open_market_full_balance(self, position_side: str) -> float:
        """
        用全部可用 USDT 余额按市价开仓，返回开仓后的 entryPrice（从仓位信息读取）。
        position_side: 'LONG' | 'SHORT'
        """
        mark_price = self._get_mark_price()
        balance = self.trader.get_available_balance()
        if balance is None or balance <= 0:
            raise ValueError("可用余额不足")

        notional = balance * self.cfg.use_balance_ratio
        qty = self.trader.calc_order_quantity(self.cfg.symbol, price=mark_price, notional_usdt=notional)
        side = "BUY" if position_side == "LONG" else "SELL"

        print(f"→ 开 {position_side} 仓 市价, qty={qty}, notional≈{notional:.2f} USDT (mark≈{mark_price})")
        self.trader.place_market_order(
            symbol=self.cfg.symbol,
            side=side,
            quantity=qty,
            dualSidePosition=self.cfg.dualSidePosition,
            positionSide=position_side
        )

        # 读取 entryPrice（可能需要等撮合落账，做少量重试）
        entry_price = None
        for _ in range(self.cfg.max_position_close_retries):
            pos = self.trader.get_position(self.cfg.symbol, dualSidePosition=self.cfg.dualSidePosition)
            p = pos[position_side] if self.cfg.dualSidePosition else pos
            ep = float(p.get("entryPrice", 0) or 0)
            amt = abs(float(p.get("positionAmt", 0) or 0))
            if ep > 0 and amt > 0:
                entry_price = ep
                break
            time.sleep(self.cfg.position_retry_sleep_sec)

        if entry_price is None:
            # 兜底用最新标记价
            entry_price = self._get_mark_price()
            print(f"⚠️ 未及时拿到 entryPrice，用 markPrice≈{entry_price} 近似。")

        return entry_price

    def _set_tp_sl_limit(self, position_side: str, entry_price: float, atr: float):
        """
        依据 entry ± k*ATR 设置 **限价** TP/SL：
        - LONG:  TP stop=entry + k*ATR, price = stop - slippage_ticks*tick
                 SL stop=entry - k*ATR, price = stop - slippage_ticks*tick
        - SHORT: TP stop=entry - k*ATR, price = stop + slippage_ticks*tick
                 SL stop=entry + k*ATR, price = stop + slippage_ticks*tick
        数量使用当前持仓数量（全平）。
        """
        tick = self.trader.get_tick_size(self.cfg.symbol)
        step = self.trader.get_step_size(self.cfg.symbol)

        # 读取当前方向仓位数量
        pos = self.trader.get_position(self.cfg.symbol, dualSidePosition=self.cfg.dualSidePosition)
        p = pos[position_side] if self.cfg.dualSidePosition else pos
        qty = abs(float(p.get("positionAmt", 0) or 0))
        if qty <= 0:
            raise ValueError("设置 TP/SL 时未检测到持仓数量 > 0")
        qty = self.trader.round_to_size(qty, step)

        k = self.cfg.atr_k
        if position_side == "LONG":
            tp_stop = entry_price + k * atr
            tp_price = tp_stop - self.cfg.slippage_ticks * tick
            sl_stop = entry_price - k * atr
            sl_price = sl_stop - self.cfg.slippage_ticks * tick

            print(
                f"→ LONG TP @ stop≈{tp_stop:.4f}, price≈{tp_price:.4f}; SL @ stop≈{sl_stop:.4f}, price≈{sl_price:.4f}")
            self.trader.set_take_profit_limit(
                symbol=self.cfg.symbol,
                stop_price=tp_stop,
                price=tp_price,
                dualSidePosition=self.cfg.dualSidePosition,
                positionSide="LONG",
                quantity=qty
            )
            self.trader.set_stop_loss_limit(
                symbol=self.cfg.symbol,
                stop_price=sl_stop,
                price=sl_price,
                dualSidePosition=self.cfg.dualSidePosition,
                positionSide="LONG",
                quantity=qty
            )

        else:  # SHORT
            tp_stop = entry_price - k * atr
            tp_price = tp_stop + self.cfg.slippage_ticks * tick
            sl_stop = entry_price + k * atr
            sl_price = sl_stop + self.cfg.slippage_ticks * tick

            print(
                f"→ SHORT TP @ stop≈{tp_stop:.4f}, price≈{tp_price:.4f}; SL @ stop≈{sl_stop:.4f}, price≈{sl_price:.4f}")
            self.trader.set_take_profit_limit(
                symbol=self.cfg.symbol,
                stop_price=tp_stop,
                price=tp_price,
                dualSidePosition=self.cfg.dualSidePosition,
                positionSide="SHORT",
                quantity=qty
            )
            self.trader.set_stop_loss_limit(
                symbol=self.cfg.symbol,
                stop_price=sl_stop,
                price=sl_price,
                dualSidePosition=self.cfg.dualSidePosition,
                positionSide="SHORT",
                quantity=qty
            )

    # ---------------- Main orchestration ----------------

    def run_once(self):
        symbol = self.cfg.symbol

        sig = self._get_latest_signal()
        print(f"最新信号: {sig}")
        if sig is None:
            print("⚠️ 未读取到信号，退出。")
            return
        if sig == 0:
            print("信号为 0，不执行。")
            return

        # 仓位 & 遗留条件单处理
        long_qty, short_qty = self._get_position_qtys()
        print(f"当前仓位: LONG={long_qty}, SHORT={short_qty}")

        if long_qty == 0 and short_qty == 0:
            # 无仓位但可能残留 TP/SL
            self._cancel_symbol_conditional_orders()

        # 计算 ATR
        atr_val = self._compute_atr()
        print(f"ATR({self.cfg.atr_period}) = {atr_val:.6f}")

        if sig == 1:
            # 目标：持有 LONG
            if long_qty > 0 and short_qty == 0:
                print("已有多仓，保持不变。")
                return

            # 若有空仓，先平空
            if short_qty > 0:
                self._close_side_if_any("SHORT")
                # 平掉空仓的条件单
                self._cancel_symbol_conditional_orders()

            # 若未持多，则开多
            if long_qty == 0:
                entry = self._open_market_full_balance("LONG")
                self._set_tp_sl_limit("LONG", entry, atr_val)

        elif sig == -1:
            # 目标：持有 SHORT
            if short_qty > 0 and long_qty == 0:
                print("已有空仓，保持不变。")
                return

            # 若有多仓，先平多
            if long_qty > 0:
                self._close_side_if_any("LONG")
                self._cancel_symbol_conditional_orders() # 平掉多仓的条件单

            # 若未持空，则开空
            if short_qty == 0:
                entry = self._open_market_full_balance("SHORT")
                self._set_tp_sl_limit("SHORT", entry, atr_val)

if __name__ == '__main__':
    from config import BINANCE_API_KEY, BINANCE_API_SECRET
    trader = BinanceFutureTrader(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

    # 2) 配置数据库与执行参数（按需修改列名/表名/路径）
    db_cfg = DBConfig(
        db_path="../data/crypto_data.db",  # ← 改成你的 SQLite 路径
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

    exec = SignalExecutor(trader=trader, db_cfg=db_cfg, exec_cfg=exec_cfg)
    # print(exec._get_latest_signal())
    # df = exec._load_ohlcv()
    # atr = exec._atr_wilder(df, period=12)
    # print(atr.info())
    # print(atr.tail())
    # print(exec._compute_atr())
    # print(exec._get_mark_price())
    # long_qty, short_qty = exec._get_position_qtys()
    # print(long_qty, short_qty)
    # exec._close_side_if_any(position_side='SHORT')
    # exec._cancel_symbol_conditional_orders()
    # entry_price = exec._open_market_full_balance(position_side='LONG')
    # print(entry_price)
    #
    # exec._set_tp_sl_limit(position_side='LONG',entry_price=entry_price, atr=40)
    # exec.run_once()
