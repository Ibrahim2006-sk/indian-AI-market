from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
            return TradeIntent(self.name, 0, 0.2, "Waiting")
        momentum = (history[-1] - history[-3]) / history[-3]
        panic = max(0.0, -momentum + max(0.0, -state.news_shock))
        qty = -max(0, int(panic * 220))
        return TradeIntent(self.name, qty, min(1.0, 0.2 + panic), "Sell pressure")


class GreedAgent(Agent):
    name = "Greed Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        if len(history) < 3:
            return TradeIntent(self.name, 4, 0.3, "Speculative long")
        momentum = (history[-1] - history[-3]) / history[-3]
        exuberance = max(0.0, momentum + max(0.0, state.news_shock))
        qty = max(0, int(exuberance * 200)) + (4 if momentum > 0 else 0)
        return TradeIntent(self.name, qty, min(1.0, 0.3 + exuberance), "Upside chase")


class NewsAgent(Agent):
    name = "News Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        qty = int(state.news_shock * 180)
        return TradeIntent(self.name, qty, 0.3 + abs(state.news_shock), "News reaction")


class RetailTraderAgent(Agent):
    name = "Retail Trader Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        if len(history) < 4:
            return TradeIntent(self.name, 2, 0.2, "Starter position")
        short_ma = sum(history[-3:]) / 3
        long_ma = sum(history[-8:]) / min(8, len(history))
        signal = (short_ma - long_ma) / long_ma + state.sentiment * 0.4
        qty = int(signal * 140)
        return TradeIntent(self.name, qty, 0.2 + abs(signal), "MA + sentiment")


class InstitutionalAgent(Agent):
    name = "Institutional Agent"

    def decide(self, state: MarketState, history: List[float]) -> TradeIntent:
        fair_value = sum(history[-10:]) / min(10, len(history))
        signal = (fair_value - state.price) / state.price - state.sentiment * 0.25
        qty = max(-70, min(70, int(signal * 260)))
        return TradeIntent(self.name, qty, 0.35 + abs(signal), "Mean reversion")


@dataclass
class Simulator:
    start_price: float = 100.0
    steps: int = 60
    seed: int = 42
    agents: List[Agent] = field(default_factory=list)

    def __post_init__(self):
        if not self.agents:
            self.agents = [
                FearAgent(),
                GreedAgent(),
                NewsAgent(),
                RetailTraderAgent(),
                InstitutionalAgent(),
            ]
        random.seed(self.seed)

    def run(self) -> List[MarketState]:
        price = self.start_price
        history = [price]
        states = []
        cash = {a.name: 1_000_000.0 for a in self.agents}
        inventory = {a.name: 0 for a in self.agents}

        for step in range(1, self.steps + 1):
            news_shock = random.gauss(0, 0.08)
            sentiment = max(-1, min(1, news_shock + random.gauss(0, 0.15)))

            state = MarketState(step, price, cash, inventory, sentiment, news_shock=news_shock)

            intents = [a.decide(state, history) for a in self.agents]
            net_demand = sum(i.quantity for i in intents)

            liquidity = 600
            impact = net_demand / liquidity
            noise = random.gauss(0, 0.02)
            drift = 0.0015 * math.tanh(sentiment)

            new_price = max(5, price * (1 + drift + impact + noise))

            for intent in intents:
                trade_price = (price + new_price) / 2
                cash[intent.agent] -= intent.quantity * trade_price
                inventory[intent.agent] += intent.quantity

            history.append(new_price)
            price = new_price

            states.append(MarketState(step, price, dict(cash), dict(inventory), sentiment, news_shock=news_shock))

        return states


def plot_results(states: List[MarketState], start_price: float):
    steps = [s.step for s in states]
    prices = [s.price for s in states]
    sentiments = [s.sentiment for s in states]

    df = pd.DataFrame({
        "Step": steps,
        "Price": prices,
        "Sentiment": sentiments
    })

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    ax[0].plot(df["Step"], df["Price"], linewidth=2)
    ax[0].set_title("Emergent AI Market Price Movement")
    ax[0].set_xlabel("Time Step")
    ax[0].set_ylabel("Price")
    ax[0].grid(True)

    ax[1].plot(df["Step"], df["Sentiment"])
    ax[1].set_title("Market Sentiment")
    ax[1].set_xlabel("Time Step")
    ax[1].set_ylabel("Sentiment")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig("indian_ai_market_simulation.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--start-price", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    simulator = Simulator(args.start_price, args.steps, args.seed)
    states = simulator.run()

    print("\nSimulation Complete")
    print(f"Start Price: {args.start_price}")
    print(f"Final Price: {states[-1].price:.2f}")

    plot_results(states, args.start_price)


if __name__ == "__main__":
    main()
