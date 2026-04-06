"""Base Neon Breakout strategy.

Momentum breakout strategy for Base that trades WETH/USDC when volatility expands
and RSI confirms trend strength.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from almanak.framework.intents import Intent
from almanak.framework.strategies import IntentStrategy, MarketSnapshot, almanak_strategy

if TYPE_CHECKING:
    from almanak.framework.teardown import TeardownMode, TeardownPositionSummary

logger = logging.getLogger(__name__)


@almanak_strategy(
    name="base_neon_breakout",
    description="Volatility breakout swap strategy on Base with RSI and Bollinger filters",
    version="1.0.0",
    author="Almanak Code",
    tags=["base", "breakout", "momentum", "rsi", "bollinger", "aerodrome"],
    supported_chains=["base"],
    supported_protocols=["aerodrome"],
    intent_types=["SWAP", "HOLD"],
    default_chain="base",
)
class BaseNeonBreakoutStrategy(IntentStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_token = self.get_config("base_token", "WETH")
        self.quote_token = self.get_config("quote_token", "USDC")
        self.dex_protocol = self.get_config("dex_protocol", "aerodrome")

        self.trade_size_usd = Decimal(str(self.get_config("trade_size_usd", "120")))
        self.min_quote_reserve_usd = Decimal(str(self.get_config("min_quote_reserve_usd", "60")))
        self.min_position_usd = Decimal(str(self.get_config("min_position_usd", "20")))
        self.max_slippage_bps = int(self.get_config("max_slippage_bps", 80))

        self.rsi_period = int(self.get_config("rsi_period", 14))
        self.breakout_rsi_min = Decimal(str(self.get_config("breakout_rsi_min", "56")))
        self.exit_rsi_max = Decimal(str(self.get_config("exit_rsi_max", "47")))
        self.profit_take_rsi = Decimal(str(self.get_config("profit_take_rsi", "72")))

        self.bb_period = int(self.get_config("bb_period", 20))
        self.bb_std_dev = float(self.get_config("bb_std_dev", 2.0))
        self.breakout_percent_b = float(self.get_config("breakout_percent_b", 0.85))
        self.breakdown_percent_b = float(self.get_config("breakdown_percent_b", 0.3))
        self.min_bandwidth = float(self.get_config("min_bandwidth", 0.018))
        self.bandwidth_expansion_factor = float(self.get_config("bandwidth_expansion_factor", 1.2))

        self.cooldown_iterations = int(self.get_config("cooldown_iterations", 1))

        self._previous_bandwidth: float | None = None
        self._cooldown_remaining = 0
        self._last_signal = "init"

    def decide(self, market: MarketSnapshot) -> Intent | None:
        try:
            rsi = market.rsi(self.base_token, period=self.rsi_period)
            bb = market.bollinger_bands(self.base_token, period=self.bb_period, std_dev=self.bb_std_dev)
            quote_balance = market.balance(self.quote_token)
            base_balance = market.balance(self.base_token)
        except ValueError as exc:
            return Intent.hold(reason=f"Market data unavailable: {exc}")

        expansion_threshold = self.min_bandwidth
        if self._previous_bandwidth is not None:
            expansion_threshold = max(expansion_threshold, self._previous_bandwidth * self.bandwidth_expansion_factor)
        volatility_expanding = bb.bandwidth >= expansion_threshold

        holding_base = base_balance.balance_usd >= self.min_position_usd
        slippage = Decimal(str(self.max_slippage_bps)) / Decimal("10000")

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._previous_bandwidth = bb.bandwidth
            return Intent.hold(reason=f"Cooldown active ({self._cooldown_remaining} iterations left)")

        breakout_signal = (
            volatility_expanding
            and bb.percent_b >= self.breakout_percent_b
            and rsi.value >= self.breakout_rsi_min
        )
        exit_signal = (
            bb.percent_b <= self.breakdown_percent_b
            or rsi.value <= self.exit_rsi_max
            or rsi.value >= self.profit_take_rsi
        )

        if not holding_base and breakout_signal:
            spendable_quote = quote_balance.balance_usd - self.min_quote_reserve_usd
            if spendable_quote >= self.trade_size_usd:
                self._last_signal = "breakout_buy"
                self._cooldown_remaining = self.cooldown_iterations
                self._previous_bandwidth = bb.bandwidth
                return Intent.swap(
                    from_token=self.quote_token,
                    to_token=self.base_token,
                    amount_usd=self.trade_size_usd,
                    max_slippage=slippage,
                    protocol=self.dex_protocol,
                )
            self._previous_bandwidth = bb.bandwidth
            return Intent.hold(
                reason=f"Breakout but low {self.quote_token} balance ({quote_balance.balance_usd:.2f} USD)"
            )

        if holding_base and exit_signal:
            sell_size = min(self.trade_size_usd, base_balance.balance_usd)
            if sell_size >= self.min_position_usd:
                self._last_signal = "risk_exit"
                self._cooldown_remaining = self.cooldown_iterations
                self._previous_bandwidth = bb.bandwidth
                return Intent.swap(
                    from_token=self.base_token,
                    to_token=self.quote_token,
                    amount_usd=sell_size,
                    max_slippage=slippage,
                    protocol=self.dex_protocol,
                )

        self._last_signal = "hold"
        self._previous_bandwidth = bb.bandwidth
        return Intent.hold(
            reason=(
                f"No action | RSI={rsi.value:.2f} %B={bb.percent_b:.3f} "
                f"BW={bb.bandwidth:.4f} expanding={volatility_expanding}"
            )
        )

    def supports_teardown(self) -> bool:
        return True

    def get_open_positions(self) -> "TeardownPositionSummary":
        from almanak.framework.teardown import PositionInfo, PositionType, TeardownPositionSummary

        positions: list[PositionInfo] = []

        try:
            market = self.create_market_snapshot()
            base_balance = market.balance(self.base_token)
            if base_balance.balance_usd >= self.min_position_usd:
                positions.append(
                    PositionInfo(
                        position_type=PositionType.TOKEN,
                        position_id="base_neon_breakout_spot",
                        chain=self.chain,
                        protocol=self.dex_protocol,
                        value_usd=base_balance.balance_usd,
                        details={
                            "asset": self.base_token,
                            "quote_token": self.quote_token,
                            "amount": str(base_balance.balance),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("Unable to inspect open positions: %s", exc)

        return TeardownPositionSummary(
            strategy_id=getattr(self, "strategy_id", self.STRATEGY_NAME),
            timestamp=datetime.now(UTC),
            positions=positions,
        )

    def generate_teardown_intents(self, mode: "TeardownMode", market=None) -> list[Intent]:
        from almanak.framework.teardown import TeardownMode

        max_slippage = Decimal("0.03") if mode == TeardownMode.HARD else Decimal(str(self.max_slippage_bps)) / Decimal("10000")
        return [
            Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount="all",
                max_slippage=max_slippage,
                protocol=self.dex_protocol,
            )
        ]

    def get_persistent_state(self) -> dict[str, Any]:
        return {
            "previous_bandwidth": self._previous_bandwidth,
            "cooldown_remaining": self._cooldown_remaining,
            "last_signal": self._last_signal,
        }

    def load_persistent_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        self._previous_bandwidth = state.get("previous_bandwidth")
        self._cooldown_remaining = int(state.get("cooldown_remaining", 0))
        self._last_signal = str(state.get("last_signal", "init"))

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": self.STRATEGY_NAME,
            "chain": self.chain,
            "pair": f"{self.base_token}/{self.quote_token}",
            "protocol": self.dex_protocol,
            "trade_size_usd": str(self.trade_size_usd),
            "cooldown_remaining": self._cooldown_remaining,
            "previous_bandwidth": self._previous_bandwidth,
            "last_signal": self._last_signal,
        }
