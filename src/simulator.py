from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime, timedelta

plt.style.use("dark_background")


# =========================
# MARKET STRUCTURES
# =========================

@dataclass
class MarketState:
    step: int
    price: float
    sentiment: float
    volatility: float
    news_shock: float
    net_demand: float


# =========================
# SIMULATOR
# =========================

@dataclass
class Simulator:
    start_price: float = 100
    steps: int = 80
    seed: int = 42

    def run(self) -> List[MarketState]:
        random.seed(self.seed)

        price = self.start_price
        states = []

        for step in range(1, self.steps + 1):
            news = random.gauss(0, 0.1)
            sentiment = max(-1, min(1, news + random.gauss(0, 0.2)))
            volatility = 0.01 + abs(news) * 0.1

            # Agent forces
            fear = max(0, -sentiment) * random.uniform(0.5, 1.5)
            greed = max(0, sentiment) * random.uniform(0.5, 1.5)
            institutional = random.uniform(-0.3, 0.3)

            net_demand = greed - fear + institutional
            noise = random.gauss(0, volatility)

            price = max(5, price * (1 + net_demand * 0.02 + noise))

            states.append(
                MarketState(
                    step,
                    price,
                    sentiment,
                    volatility,
                    news,
                    net_demand,
                )
            )

        return states


# =========================
# ANIMATION
# =========================

def animate_market(states: List[MarketState]):
    fig = plt.figure(figsize=(14, 8))
    ax_price = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax_sentiment = plt.subplot2grid((3, 1), (2, 0))

    prices = []
    sentiments = []
    dates = []
    base_time = datetime.now()

    # Text panel
    info_text = ax_price.text(
        0.02,
        0.95,
        "",
        transform=ax_price.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.7),
    )

    def update(frame):
        state = states[frame]

        prices.append(state.price)
        sentiments.append(state.sentiment)
        dates.append(base_time + timedelta(minutes=frame))

        ax_price.clear()
        ax_sentiment.clear()

        # --- Candlestick ---
        for i in range(len(prices)):
            color = "lime" if i == 0 or prices[i] >= prices[i - 1] else "red"
            ax_price.plot([dates[i], dates[i]], 
                          [prices[i] * 0.995, prices[i] * 1.005], 
                          color=color)
            ax_price.scatter(dates[i], prices[i], color=color, s=15)

        # --- Agent Influence Overlay ---
        ax_price.plot(dates, prices, linewidth=1.5)

        ax_price.set_title("Indian AI Market - Live Trading Terminal")
        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.2)

        # --- Sentiment ---
        ax_sentiment.plot(dates, sentiments)
        ax_sentiment.set_ylabel("Sentiment")
        ax_sentiment.set_ylim(-1.1, 1.1)
        ax_sentiment.grid(True, alpha=0.2)

        # --- Live Info Panel ---
        info = (
            f"Step: {state.step}\n"
            f"Price: {state.price:.2f}\n"
            f"Net Demand: {state.net_demand:+.3f}\n"
            f"Sentiment: {state.sentiment:+.2f}\n"
            f"Volatility: {state.volatility:.3f}"
        )
        info_text.set_text(info)

        ax_price.add_artist(info_text)

        plt.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=200,
        repeat=False,
    )

    # --- Save MP4 ---
    ani.save("indian_ai_market_live.mp4", writer="ffmpeg", fps=5)

    plt.show()


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--start-price", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sim = Simulator(args.start_price, args.steps, args.seed)
    states = sim.run()

    animate_market(states)


if __name__ == "__main__":
    main()
