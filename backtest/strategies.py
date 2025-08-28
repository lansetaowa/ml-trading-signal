import backtrader as bt
from datetime import datetime

class TradeLoggerMixin:
    """
    将此 Mixin 混入你的策略类：class MyStrategy(TradeLoggerMixin, bt.Strategy):
    作用：记录每一笔完成交易（进场->出场），含方向、数量、开平价、开平时刻、持仓bar数、毛/净PnL等
    """
    def __init__(self):
        # 记录未完成的开仓（用于提取开仓价/时间）；以及已完成的完整交易列表
        self._open_position_info = None  # {'dt': datetime, 'price': float, 'size': int, 'is_long': bool, 'commission': float}
        self.trade_log = []  # list of dicts

    def notify_order(self, order):
        # 记录开仓单完成的细节（为后续平仓补全）
        if order.status in [order.Completed]:
            # buy: 正数；sell: 负数
            if order.isbuy():
                # 建多或加仓：记录最近的开仓信息
                self._open_position_info = {
                    'dt': self.data.datetime.datetime(0),
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'is_long': True,
                    'commission': order.executed.comm or 0.0,
                }
            else:
                # 建空或加仓：记录最近的开仓信息
                self._open_position_info = {
                    'dt': self.data.datetime.datetime(0),
                    'price': order.executed.price,
                    'size': -abs(order.executed.size),  # 统一用负数表示空头
                    'is_long': False,
                    'commission': order.executed.comm or 0.0,
                }

    def notify_trade(self, trade):
        # 只有交易“关闭”时才产生一条完整记录
        if not trade.isclosed:
            return

        close_dt = self.data.datetime.datetime(0)

        # trade 对象里直接给了关键字段：pnl(毛利) / pnlcomm(净利, 含手续费)
        record = {
            'symbol'      : getattr(self.data, '_name', 'data0'),
            'open_dt'     : getattr(trade, 'dtopen', None) or (self._open_position_info['dt'] if self._open_position_info else None),
            'close_dt'    : getattr(trade, 'dtclose', None) or close_dt,
            'size'        : trade.size,
            'is_long'     : trade.size > 0,
            'entry_price' : getattr(trade, 'price', None) or (self._open_position_info['price'] if self._open_position_info else None),
            'exit_price'  : self.data.close[0],
            'barlen'      : getattr(trade, 'barlen', None),   # 持仓bar数
            'pnl_gross'   : trade.pnl,
            'pnl_net'     : trade.pnlcomm,                    # 扣除手续费
            'commission'  : (self._open_position_info['commission'] if self._open_position_info else 0.0),
            'broker_value': self.broker.getvalue(),           # 记录平仓后账户权益
        }

        self.trade_log.append(record)
        # 清空开仓信息（简单场景）；如需多次加仓/分批，可扩展为 list 并匹配 size
        self._open_position_info = None

class AtrDynamicStopStrategy(TradeLoggerMixin, bt.Strategy):
    params = dict(
        atr_period=12,
        atr_tp_factor=1,
        atr_sl_factor=1.5,
        signal_column='final_signal_filtered',
    )

    def __init__(self):
        super().__init__()
        self.signal = self.datas[0].lines.signal
        self.atr = bt.ind.ATR(self.datas[0], period=self.p.atr_period)
        self.order = None
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} - {txt}')

    def next(self):
        if self.order:
            return  # 有未完成订单

        pos = self.position
        price = self.data.close[0]
        high = self.data.high[0]
        low = self.data.low[0]
        signal = self.signal[0]
        atr = self.atr[0]

        # === 无仓位，根据信号开仓 ===
        if not pos:
            size = self.broker.get_cash() // price - 1
            if signal == 1:
                self.entry_price = price
                self.tp_price = self.entry_price + self.p.atr_tp_factor * atr
                self.sl_price = self.entry_price - self.p.atr_sl_factor * atr
                self.log(f'开多仓 @ {price:.2f} (TP={self.tp_price:.2f}, SL={self.sl_price:.2f})')
                self.order = self.buy(size=size)
            elif signal == -1:
                self.entry_price = price
                self.tp_price = self.entry_price - self.p.atr_tp_factor * atr
                self.sl_price = self.entry_price + self.p.atr_sl_factor * atr
                self.log(f'开空仓 @ {price:.2f} (TP={self.tp_price:.2f}, SL={self.sl_price:.2f})')
                self.order = self.sell(size=size)
            return

        # === 多头持仓 ===
        if pos.size > 0:
            if high >= self.tp_price:
                self.log(f'多头止盈 @ {self.tp_price:.2f}')
                self.close(price=self.tp_price)
            elif low <= self.sl_price:
                self.log(f'多头止损 @ {self.sl_price:.2f}')
                self.close(price=self.sl_price)
            elif signal == -1:
                self.log(f'反转做空：平多仓 @ {price:.2f}')
                self.close()
                size = self.broker.get_cash() // price - 1
                self.entry_price = price
                self.tp_price = self.entry_price - self.p.atr_tp_factor * atr
                self.sl_price = self.entry_price + self.p.atr_sl_factor * atr
                self.order = self.sell(size=size)

        # === 空头持仓 ===
        elif pos.size < 0:
            if low <= self.tp_price:
                self.log(f'空头止盈 @ {self.tp_price:.2f}')
                self.close(price=self.tp_price)
            elif high >= self.sl_price:
                self.log(f'空头止损 @ {self.sl_price:.2f}')
                self.close(price=self.sl_price)
            elif signal == 1:
                self.log(f'反转做多：平空仓 @ {price:.2f}')
                self.close()
                size = self.broker.get_cash() // price - 1
                self.entry_price = price
                self.tp_price = self.entry_price + self.p.atr_tp_factor * atr
                self.sl_price = self.entry_price - self.p.atr_sl_factor * atr
                self.order = self.buy(size=size)

    def notify_order(self, order):
        super().notify_order(order)
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            action = '买入' if order.isbuy() else '卖出'
            # self.log(f'订单完成：{action} {order.executed.size:.4f} @ {order.executed.price:.2f}，手续费: {order.executed.comm:.4f}')
        elif order.status in [order.Canceled, order.Rejected, order.Margin]:
            self.log(f'订单失败：{order.Status[order.status]}')
        self.order = None

class AtrLongOnlyStrategy(TradeLoggerMixin, bt.Strategy):
    params = dict(
        atr_period=12,
        atr_tp_factor=1,
        atr_sl_factor=1,
        signal_column='final_signal_filtered'
    )

    def __init__(self):
        super().__init__()
        self.signal = self.datas[0].lines.signal
        self.atr = bt.ind.ATR(self.datas[0], period=self.p.atr_period)
        self.order = None
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None
        self.trade_log = []

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} - {txt}')

    def next(self):
        if self.order:
            return

        pos = self.position
        price = self.data.close[0]
        high = self.data.high[0]
        low = self.data.low[0]
        signal = self.signal[0]
        atr = self.atr[0]

        # No position, only consider long signals
        if not pos:
            size = self.broker.get_cash() // price
            if signal == 1:
                self.entry_price = price
                self.tp_price = self.entry_price + self.p.atr_tp_factor * atr
                self.sl_price = self.entry_price - self.p.atr_sl_factor * atr
                self.log(f'Opening long position @ {price:.2f} (TP={self.tp_price:.2f}, SL={self.sl_price:.2f})')
                self.order = self.buy(size=size)
            return

        # Manage long position (take profit/stop loss)
        if pos.size > 0:
            if high >= self.tp_price:
                self.log(f'Taking profit @ {self.tp_price:.2f}')
                self.close(price=self.tp_price)
            elif low <= self.sl_price:
                self.log(f'Stop loss @ {self.sl_price:.2f}')
                self.close(price=self.sl_price)
            elif signal == 1:
                pass  # Continue holding on long signal
            elif signal == -1:
                pass  # Ignore reversal short signal

    def notify_order(self, order):
        super().notify_order(order)
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            action = 'Buy' if order.isbuy() else 'Sell'
            # self.log(f'Order completed: {action} {order.executed.size:.4f} @ {order.executed.price:.2f}, Commission: {order.executed.comm:.4f}')
        elif order.status in [order.Canceled, order.Rejected, order.Margin]:
            self.log(f'Order failed: {order.Status[order.status]}')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        if trade.size == 0:
            self.log('Skipping trade with size 0 (possibly invalid)')
            return

        entry_dt = trade.open_datetime()
        exit_dt = trade.close_datetime()
        entry_price = trade.price
        exit_price = trade.price + trade.pnl / trade.size

        self.log(f'Closing position: PnL {trade.pnl:.2f} | Commission {trade.commission:.4f} | Holding time: {(exit_dt - entry_dt)}')

        self.trade_log.append({
            'Entry Time': entry_dt,
            'Exit Time': exit_dt,
            'Direction': 'Long',
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Pnl': trade.pnl,
            'Commission': trade.commission,
            'Duration': (exit_dt - entry_dt).total_seconds() / 3600
        })