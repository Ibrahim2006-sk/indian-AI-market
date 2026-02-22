from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MarketState:
    step: int
    price: float
    cash: Dict[str, float]
    inventory: Dict[str, int]
    sentiment: float = 0.0
    volatility: float = 0.02
    news_shock: float = 0.0


@dataclass
class TradeIntent:
    agent: str
    quantity: int
    confidence: float
    rationale: str


class Agent:
    name: str

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        raise NotImplementedError


class FearAgent(Agent):
    name = "Fear Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        if len(history) < 3:
            return TradeIntent(self.name, 0, 0.2, "Waiting for trend clarity")
        momentum = (history[-1] - history[-3]) / history[-3]
        panic = max(0.0, -momentum + max(0.0, -state.news_shock))
        qty = -max(0, int(panic * 220))
        return TradeIntent(self.name, qty, min(1.0, 0.2 + panic), "Selling into downside pressure")


class GreedAgent(Agent):
    name = "Greed Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        if len(history) < 3:
            return TradeIntent(self.name, 4, 0.3, "Opening speculative long")
        momentum = (history[-1] - history[-3]) / history[-3]
        exuberance = max(0.0, momentum + max(0.0, state.news_shock))
        qty = max(0, int(exuberance * 200)) + (4 if momentum > 0 else 0)
        return TradeIntent(self.name, qty, min(1.0, 0.3 + exuberance), "Chasing upside breakout")


class NewsAgent(Agent):
    name = "News Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        shock_bias = state.news_shock
        qty = int(shock_bias * 180)
        tone = "positive headlines" if shock_bias >= 0 else "negative headlines"
        return TradeIntent(self.name, qty, min(1.0, 0.25 + abs(shock_bias)), f"Reacting to {tone}")


class RetailTraderAgent(Agent):
    name = "Retail Trader Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        if len(history) < 4:
            return TradeIntent(self.name, 2, 0.2, "Small starter position")
        short_ma = sum(history[-3:]) / 3
        long_ma = sum(history[-8:]) / min(8, len(history))
        fomo = (short_ma - long_ma) / long_ma
        sentiment_boost = state.sentiment * 0.4
        signal = fomo + sentiment_boost + random.uniform(-0.02, 0.02)
        qty = int(signal * 140)
        return TradeIntent(self.name, qty, min(1.0, 0.2 + abs(signal)), "Social-trend plus moving-average signal")


class InstitutionalAgent(Agent):
    name = "Institutional Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        fair_value = sum(history[-10:]) / min(10, len(history))
        mispricing = (fair_value - state.price) / state.price
        contrarian = -state.sentiment * 0.25
        signal = mispricing + contrarian
        raw_qty = int(signal * 260)
        qty = max(-70, min(70, raw_qty))
        return TradeIntent(self.name, qty, min(1.0, 0.35 + abs(signal)), "Mean-reversion with risk caps")


@dataclass
class Simulator:
    start_price: float = 100.0
    steps: int = 50
    seed: int = 42
    agents: List[Agent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.agents:
            self.agents = [
                FearAgent(),
                GreedAgent(),
                NewsAgent(),
                RetailTraderAgent(),
                InstitutionalAgent(),
            ]
        random.seed(self.seed)

    def _cash_book(self) -> Dict[str, float]:
        return {a.name: 1_000_000.0 for a in self.agents}

    def _inventory_book(self) -> Dict[str, int]:
        return {a.name: 0 for a in self.agents}

    def run(self) -> List[MarketState]:
        price = self.start_price
        history = [price]
        states: List[MarketState] = []
        cash = self._cash_book()
        inventory = self._inventory_book()

        for step in range(1, self.steps + 1):
            news_shock = random.gauss(0, 0.08)
            sentiment = max(-1.0, min(1.0, news_shock + random.gauss(0, 0.15)))

            state = MarketState(
                step=step,
                price=price,
                cash=cash,
                inventory=inventory,
                sentiment=sentiment,
                news_shock=news_shock,
                volatility=0.015 + abs(news_shock) * 0.12,
            )

            intents = [agent.decide(state, history) for agent in self.agents]
            net_demand = sum(intent.quantity for intent in intents)
            confidence = sum(abs(intent.quantity) * intent.confidence for intent in intents)
            liquidity = 600 + random.uniform(-50, 50)
            impact = net_demand / liquidity
            noise = random.gauss(0, state.volatility * 0.45)

            drift = 0.0015 * math.tanh(state.sentiment)
            price_change = drift + impact + noise
            new_price = max(5.0, price * (1 + price_change))

            for intent in intents:
                qty = intent.quantity
                trade_price = (price + new_price) / 2
                cash[intent.agent] -= qty * trade_price
                inventory[intent.agent] += qty

            history.append(new_price)
            price = new_price
            states.append(
                MarketState(
                    step=step,
                    price=price,
                    cash=dict(cash),
                    inventory=dict(inventory),
                    sentiment=sentiment,
                    volatility=state.volatility,
                    news_shock=news_shock,
                )
            )

            if step % 10 == 0:
                confidence_factor = confidence / max(1.0, abs(net_demand))
                print(
                    f"Step {step:>2} | Price {price:>7.2f} | Net demand {net_demand:>4} | "
                    f"Sentiment {sentiment:+.2f} | Confidence {confidence_factor:.2f}"
                )

        return states


def summarize(states: List[MarketState], start_price: float) -> str:
    final = states[-1]
    returns = (final.price - start_price) / start_price

    lines = [
        "\n=== Indian AI Stock Market Simulator Summary ===",
        f"Start price: {start_price:.2f}",
        f"Final price: {final.price:.2f}",
        f"Total return: {returns * 100:+.2f}%",
        f"Final sentiment: {final.sentiment:+.2f}",
        "\nAgent books:",
    ]

    for agent, cash in final.cash.items():
        shares = final.inventory[agent]
        nav = cash + shares * final.price
        lines.append(f"- {agent:<24} Cash {cash:>11,.0f} | Shares {shares:>5} | NAV {nav:>11,.0f}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indian AI stock market multi-agent simulator")
    parser.add_argument("--steps", type=int, default=60, help="Simulation steps")
    parser.add_argument("--start-price", type=float, default=120.0, help="Initial stock price")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator = Simulator(start_price=args.start_price, steps=args.steps, seed=args.seed)
    states = simulator.run()
    print(summarize(states, args.start_price))


if __name__ == "__main__":
    main()
