from binance.client import Client
from decimal import Decimal, ROUND_DOWN
from binance.enums import (
    FUTURE_ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
    FUTURE_ORDER_TYPE_STOP_MARKET,
    FUTURE_ORDER_TYPE_TAKE_PROFIT,  # 限价止盈
    FUTURE_ORDER_TYPE_STOP,         # 限价止损
)

from config import binance_proxy

class BinanceFutureTrader:

    def __init__(self, api_key=None, api_secret=None):

        self.client = Client(
            api_key, api_secret,
            requests_params={'proxies': {'http': binance_proxy,'https': binance_proxy}}
            )

    # ---------- 基础工具 ----------

    # 某symbol合约的filter集合
    def get_symbol_filters(self, symbol: str) -> dict:
        """
        返回该合约交易对的过滤器字典，键为 filterType，值为该 filter 的字典。
        例如：filters['PRICE_FILTER']['tickSize'], filters['LOT_SIZE']['stepSize'] 等。
        """
        info = self.client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                out = {}
                for f in s["filters"]:
                    out[f["filterType"]] = f
                return out
        raise ValueError(f"Symbol {symbol} not found in futures exchange info")

    # 价格精度，tick size
    def get_tick_size(self, symbol: str) -> float:
        """
        获取价格精度（tick size），来自 PRICE_FILTER。
        """
        filters = self.get_symbol_filters(symbol)
        tick = filters["PRICE_FILTER"]["tickSize"]
        return float(tick)

    # 数量精度，step size
    def get_step_size(self, symbol: str) -> float:
        """
        获取数量精度（step size），来自 LOT_SIZE（或 MARKET_LOT_SIZE；多数情况下 LOT_SIZE 即可）。
        """
        filters = self.get_symbol_filters(symbol)
        step = filters["LOT_SIZE"]["stepSize"]
        return float(step)

    # 按step/tick size，下单数量/下单价格 向下取整
    def round_to_size(self, value, size):
        """
        按 step 向下取整（避免浮点误差）。
        """
        v = Decimal(str(value))
        s = Decimal(str(size))
        return float((v / s).quantize(Decimal('1.'), rounding=ROUND_DOWN) * s)

    # ---------- 获取账户信息 ----------
    # 合约账户仓位模式
    def get_position_mode(self) -> str:
        """
        返回 'HEDGE' 或 'ONE_WAY'
        """
        mode = self.client.futures_get_position_mode()
        return 'HEDGE' if mode.get('dualSidePosition') else 'ONE_WAY'

    # 查询合约usdt余额
    def get_available_balance(self):
        try:
            account_info = self.client.futures_account_balance()
            for asset in account_info:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
        except Exception as e:
            print(f"❌ 获取 Binance 合约账户余额出错: {e}")
            return None

    # ---------- 获取某个symbol已有仓位 ----------
    def _parse_position_row(self, p: dict) -> dict:
        """
        规范化单条 position 信息为易用字典。
        适用于 futures_position_information 返回的元素。
        """
        amt = float(p.get("positionAmt", "0"))
        side = "FLAT"
        if amt > 0:
            side = "LONG"
        elif amt < 0:
            side = "SHORT"

        return {
            'symbol': p.get('symbol', ''),
            'positionAmt': amt,
            'entryPrice': float(p.get('entryPrice', '0') or 0),
            'unrealizedProfit': float(p.get('unRealizedProfit', '0') or 0),
            'leverage': int(float(p.get('leverage', '0') or 0)),
            'liquidationPrice': float(p.get('liquidationPrice', '0') or 0),
            'marginType': p.get('marginType', ''),
            'positionSide': p.get('positionSide', ''),  # 仅在双向模式下有意义
            'side': side,  # 基于 positionAmt 推断
        }

    def get_position(self, symbol: str, *, dualSidePosition: bool = True):
        """
        获取该 symbol 的仓位信息。
        - 双向模式：返回 {'LONG': {...}, 'SHORT': {...}}，若某方向无仓位则 positionAmt=0
        - 单向模式：返回 {...} 单条记录（side=LONG/SHORT/FLAT）
        """
        rows = self.client.futures_position_information(symbol=symbol)
        if not rows:
            # 理论上不会为空，但兜底
            if dualSidePosition:
                return {'LONG': {'symbol': symbol, 'positionAmt': 0.0, 'side': 'FLAT'},
                        'SHORT': {'symbol': symbol, 'positionAmt': 0.0, 'side': 'FLAT'}}
            return {'symbol': symbol, 'positionAmt': 0.0, 'side': 'FLAT'}

        if dualSidePosition:
            long_info = None
            short_info = None
            for p in rows:
                if p.get('positionSide') == 'LONG':
                    long_info = self._parse_position_row(p)
                elif p.get('positionSide') == 'SHORT':
                    short_info = self._parse_position_row(p)

            # 若某方向不存在，填补空结构
            if long_info is None:
                long_info = {
                    'symbol': symbol, 'positionAmt': 0.0, 'entryPrice': 0.0,
                    'unrealizedProfit': 0.0, 'leverage': 0, 'liquidationPrice': 0.0,
                    'marginType': '', 'positionSide': 'LONG', 'side': 'FLAT'
                }
            if short_info is None:
                short_info = {
                    'symbol': symbol, 'positionAmt': 0.0, 'entryPrice': 0.0,
                    'unrealizedProfit': 0.0, 'leverage': 0, 'liquidationPrice': 0.0,
                    'marginType': '', 'positionSide': 'SHORT', 'side': 'FLAT'
                }
            return {'LONG': long_info, 'SHORT': short_info}

        # 单向模式：Binance 通常返回一条该 symbol 的记录（有时仍是列表，这里取第一条）
        p = rows[0]
        return self._parse_position_row(p)

    # ---------- 计算下单 quantity ----------

    def calc_order_quantity(self, symbol: str, price: float, notional_usdt: float) -> float:
        """
        根据目标名义价值（USDT）和价格，计算合约张数/数量，并按 stepSize 向下取整。
        仅适用于以 USDT 计价的线性合约（如 BTCUSDT、ETHUSDT 等）。
        例如：notional_usdt=100, price=62000 -> 原始数量=100/62000=0.0016129...
        """
        if price <= 0:
            raise ValueError("Price must be positive")

        raw_qty = notional_usdt / price
        step = self.get_step_size(symbol)
        qty = self.round_to_size(raw_qty, step)

        # Binance 还有最小下单数量 / 最小名义价值等限制，这里可选做个简单检查：
        # MIN_NOTIONAL 在合约里可能叫 NOTIONAL，但并非所有 symbol 强制相同；
        # 这里保留接口，若需要可扩展成严格校验。
        if qty <= 0:
            raise ValueError("Calculated quantity is too small after rounding. Increase notional_usdt.")

        return qty

    # 下market合约单
    def place_market_order(
            self,
            symbol: str,
            side: str,
            quantity: float,
            *,
            dualSidePosition: bool = True,
            positionSide: str | None = None,
            reduce_only: bool = False
    ) -> dict:
        """
        市价单下单（支持单向/双向）。
        - dualSidePosition=True  → 双向模式，必须传 positionSide ∈ {'LONG','SHORT'}
        - dualSidePosition=False → 单向模式，禁止传 positionSide
        - reduce_only=True       → 仅减少仓位（不增加持仓）
        """
        side = side.upper()
        if side not in ('BUY', 'SELL'):
            raise ValueError("side must be 'BUY' or 'SELL'")

        params = {
            'symbol': symbol,
            'type': FUTURE_ORDER_TYPE_MARKET,
            'quantity': quantity
        }

        if dualSidePosition:
            if positionSide not in ('LONG', 'SHORT'):
                raise ValueError("In hedge mode, positionSide must be 'LONG' or 'SHORT'")
            params['positionSide'] = positionSide
            params['side'] = side
            # 重要：Hedge 模式禁止 reduceOnly 参数
            if reduce_only:
                # 也可以选择静默忽略：不抛异常
                raise ValueError("reduce_only cannot be used in Hedge mode (Binance API).")
        else:
            if positionSide is not None:
                raise ValueError("In one-way mode, do NOT pass positionSide")
            params['side'] = side
            if reduce_only:
                params['reduceOnly'] = 'true'

        return self.client.futures_create_order(**params)

    # 设置止盈
    def set_take_profit(
            self,
            symbol: str,
            stop_price: float,
            *,
            dualSidePosition: bool = True,
            positionSide: str | None = None,
            close_position: bool = True,
            quantity: float | None = None,
            working_type: str = 'MARK_PRICE'
    ) -> dict:
        """
        TAKE_PROFIT_MARKET 止盈单（触发即以市价成交）。
        - 双向：必须给 positionSide（'LONG' 或 'SHORT'），函数会自动设置 side（LONG→SELL，SHORT→BUY）
        - 单向：不传 positionSide。建议使用 close_position=True；若需要部分平仓，需要你自己保证方向正确
        - close_position=True：仅传 closePosition，不传 reduceOnly/quantity
        - close_position=False：传 quantity，并且加 reduceOnly=true（按 stepSize 取整）
        """
        tick = self.get_tick_size(symbol)
        sp = self.round_to_size(stop_price, tick)

        params = {
            'symbol': symbol,
            'type': FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            'stopPrice': sp,
            'workingType': working_type,
        }

        if dualSidePosition:
            if positionSide not in ('LONG', 'SHORT'):
                raise ValueError("In hedge mode, positionSide must be 'LONG' or 'SHORT'")
            params['positionSide'] = positionSide
            params['side'] = 'SELL' if positionSide == 'LONG' else 'BUY'
        else:
            if positionSide is not None:
                raise ValueError("In one-way mode, do NOT pass positionSide")
            # 单向模式下，若 close_position=True 可不传 side；若部分平仓请改用 STOP/TP 并自己传正确 side 的版本

        if close_position:
            params['closePosition'] = 'true'
            # 不传 reduceOnly；否则可能触发 -1106
        else:
            if not quantity or quantity <= 0:
                raise ValueError("quantity must be provided and > 0 when close_position=False")
            step = self.get_step_size(symbol)
            params['quantity'] = self.round_to_size(quantity, step)
            params['reduceOnly'] = 'true'

        return self.client.futures_create_order(**params)

    # 设置限价止盈
    def _infer_oneway_side(self, symbol: str) -> str:
        """单向模式下，根据当前仓位推断平仓方向（平多=SELL，平空=BUY）。"""
        p = self.get_position(symbol, dualSidePosition=False)
        amt = p.get('positionAmt', 0.0)
        if amt > 0:
            return 'SELL'
        elif amt < 0:
            return 'BUY'
        raise ValueError("No position to close in one-way mode (positionAmt == 0).")

    def set_take_profit_limit(
            self,
            symbol: str,
            stop_price: float,
            price: float,
            *,
            dualSidePosition: bool = True,
            positionSide: str | None = None,
            quantity: float | None = None,
            working_type: str = 'MARK_PRICE',
            time_in_force: str = 'GTC'
    ) -> dict:
        # 限价 TP 不支持 closePosition；需要 quantity
        if not quantity or quantity <= 0:
            raise ValueError("quantity is required (>0) for TAKE_PROFIT (limit).")

        tick = self.get_tick_size(symbol)
        sp = self.round_to_size(stop_price, tick)
        lp = self.round_to_size(price, tick)

        params = {
            'symbol': symbol,
            'type': FUTURE_ORDER_TYPE_TAKE_PROFIT,
            'stopPrice': sp,
            'price': lp,
            'workingType': working_type,
            'timeInForce': time_in_force,
        }

        if dualSidePosition:
            if positionSide not in ('LONG', 'SHORT'):
                raise ValueError("In hedge mode, positionSide must be 'LONG' or 'SHORT'")
            params['positionSide'] = positionSide
            params['side'] = 'SELL' if positionSide == 'LONG' else 'BUY'
            # 重要：Hedge 模式禁止 reduceOnly 参数（不发送）
        else:
            if positionSide is not None:
                raise ValueError("In one-way mode, do NOT pass positionSide")
            # 单向模式下，限价部分平仓建议启用 reduceOnly
            params['side'] = self._infer_oneway_side(symbol)
            params['reduceOnly'] = 'true'

        step = self.get_step_size(symbol)
        params['quantity'] = self.round_to_size(quantity, step)

        return self.client.futures_create_order(**params)

    # 设置止损
    def set_stop_loss(
            self,
            symbol: str,
            stop_price: float,
            *,
            dualSidePosition: bool = True,
            positionSide: str | None = None,
            close_position: bool = True,
            quantity: float | None = None,
            working_type: str = 'MARK_PRICE'
    ) -> dict:
        """
        STOP_MARKET 止损单（触发即以市价成交）。
        - 双向：必须给 positionSide（'LONG' 或 'SHORT'），函数会自动设置 side（LONG→SELL，SHORT→BUY）
        - 单向：不传 positionSide。建议使用 close_position=True；若需要部分平仓，需要你自己保证方向正确
        - close_position=True：仅传 closePosition，不传 reduceOnly/quantity
        - close_position=False：传 quantity，并且加 reduceOnly=true（按 stepSize 取整）
        """
        tick = self.get_tick_size(symbol)
        sp = self.round_to_size(stop_price, tick)

        params = {
            'symbol': symbol,
            'type': FUTURE_ORDER_TYPE_STOP_MARKET,
            'stopPrice': sp,
            'workingType': working_type,
        }

        if dualSidePosition:
            if positionSide not in ('LONG', 'SHORT'):
                raise ValueError("In hedge mode, positionSide must be 'LONG' or 'SHORT'")
            params['positionSide'] = positionSide
            params['side'] = 'SELL' if positionSide == 'LONG' else 'BUY'
        else:
            if positionSide is not None:
                raise ValueError("In one-way mode, do NOT pass positionSide")

        if close_position:
            params['closePosition'] = 'true'
            # 不传 reduceOnly；否则可能触发 -1106
        else:
            if not quantity or quantity <= 0:
                raise ValueError("quantity must be provided and > 0 when close_position=False")
            step = self.get_step_size(symbol)
            params['quantity'] = self.round_to_size(quantity, step)
            params['reduceOnly'] = 'true'

        return self.client.futures_create_order(**params)

    # 设置限价止损
    def set_stop_loss_limit(
            self,
            symbol: str,
            stop_price: float,
            price: float,
            *,
            dualSidePosition: bool = True,
            positionSide: str | None = None,
            quantity: float | None = None,
            working_type: str = 'MARK_PRICE',
            time_in_force: str = 'GTC'
    ) -> dict:
        # 限价 SL 不支持 closePosition；需要 quantity
        if not quantity or quantity <= 0:
            raise ValueError("quantity is required (>0) for STOP (limit).")

        tick = self.get_tick_size(symbol)
        sp = self.round_to_size(stop_price, tick)
        lp = self.round_to_size(price, tick)

        params = {
            'symbol': symbol,
            'type': FUTURE_ORDER_TYPE_STOP,
            'stopPrice': sp,
            'price': lp,
            'workingType': working_type,
            'timeInForce': time_in_force,
        }

        if dualSidePosition:
            if positionSide not in ('LONG', 'SHORT'):
                raise ValueError("In hedge mode, positionSide must be 'LONG' or 'SHORT'")
            params['positionSide'] = positionSide
            params['side'] = 'SELL' if positionSide == 'LONG' else 'BUY'
            # 重要：Hedge 模式禁止 reduceOnly 参数（不发送）
        else:
            if positionSide is not None:
                raise ValueError("In one-way mode, do NOT pass positionSide")
            params['side'] = self._infer_oneway_side(symbol)
            params['reduceOnly'] = 'true'

        step = self.get_step_size(symbol)
        params['quantity'] = self.round_to_size(quantity, step)

        return self.client.futures_create_order(**params)

if __name__ == '__main__':
    from config import BINANCE_API_KEY, BINANCE_API_SECRET
    bn_future_trader = BinanceFutureTrader(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

    symbol = 'ETHUSDT'
    # print(bn_future_trader.get_available_balance())
    # print(bn_future_trader.get_symbol_filters(symbol))
    # print(bn_future_trader.get_tick_size(symbol))
    # print(bn_future_trader.get_step_size(symbol))
    # print(bn_future_trader.round_to_size(1.053234, 0.001))
    # print(bn_future_trader.client.futures_get_position_mode())
    # print(bn_future_trader.calc_order_quantity(symbol, price=4713.54, notional_usdt=1000))
    # mkt_long_order = bn_future_trader.place_market_order(symbol=symbol,
    #                                                      side='BUY',
    #                                                      quantity=0.01,
    #                                                      positionSide='LONG')
    # print(mkt_long_order)
    # print(bn_future_trader.get_position(symbol))
    # tp_order = bn_future_trader.set_take_profit(symbol=symbol,
    #                                             stop_price=4760.5555,
    #                                             positionSide='LONG',
    #                                             quantity=0.01)
    # print(tp_order)
    # sl_order = bn_future_trader.set_stop_loss(symbol=symbol,
    #                                           stop_price=4650.5555,
    #                                           positionSide='LONG',
    #                                           quantity=0.01)
    # print(sl_order)
    # tp_order = bn_future_trader.set_take_profit_limit(symbol=symbol,
    #                                             stop_price=4760.5555,
    #                                             price=4761,
    #                                             positionSide='LONG',
    #                                             quantity=0.01)
    # print(tp_order)
    # sl_order = bn_future_trader.set_stop_loss_limit(symbol=symbol,
    #                                             stop_price=4660.5555,
    #                                             price=4760,
    #                                             positionSide='LONG',
    #                                             quantity=0.01)
    # print(sl_order)

    # mkt_short_order = bn_future_trader.place_market_order(symbol=symbol,
    #                                                      side='SELL',
    #                                                      quantity=0.01,
    #                                                      positionSide='SHORT')
    # print(mkt_short_order)
    # print(bn_future_trader.get_position(symbol))
    # tp_order = bn_future_trader.set_take_profit(symbol=symbol,
    #                                             stop_price=4720.5555,
    #                                             positionSide='SHORT',
    #                                             quantity=0.01)
    # print(tp_order)
    # sl_order = bn_future_trader.set_stop_loss(symbol=symbol,
    #                                           stop_price=4750.5555,
    #                                           positionSide='SHORT',
    #                                           quantity=0.01)
    # print(sl_order)
    # tp_order = bn_future_trader.set_take_profit_limit(symbol=symbol,
    #                                             stop_price=4480,
    #                                             price=4480,
    #                                             positionSide='SHORT',
    #                                             quantity=0.01)
    # print(tp_order)
    # sl_order = bn_future_trader.set_stop_loss_limit(symbol=symbol,
    #                                             stop_price=4570,
    #                                             price=4570,
    #                                             positionSide='SHORT',
    #                                             quantity=0.01)
    # print(sl_order)