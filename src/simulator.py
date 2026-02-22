from __future__ import annotations

# ---- FIX FOR WINDOWS CMD ----
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import argparse
from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta

plt.style.use("dark_background")


# =========================
# MARKET STATE
# =========================

@dataclass
class MarketState:
    step: int
    price: float
    sentiment: float
    volatility: float
    net_demand: float


# =========================
# SIMULATOR
# =========================

class Simulator:
    def __init__(self, start_price=120, steps=80):
        self.start_price = start_price
        self.steps = steps

    def run(self) -> List[MarketState]:
        price = self.start_price
        states = []

        for step in range(1, self.steps + 1):

            # Random sentiment & volatility (NO fixed seed = new result every run)
            sentiment = random.uniform(-1, 1)
            volatility = 0.01 + abs(sentiment) * 0.05

            # Agent forces
            fear = max(0, -sentiment) * random.uniform(0.5, 1.5)
            greed = max(0, sentiment) * random.uniform(0.5, 1.5)
            institutional = random.uniform(-0.3, 0.3)

            net_demand = greed - fear + institutional
            noise = random.gauss(0, volatility)

            price = max(5, price * (1 + net_demand * 0.02 + noise))

            states.append(
                MarketState(step, price, sentiment, volatility, net_demand)
            )

        return states


# =========================
# ANIMATION
# =========================

def animate_market(states: List[MarketState]):

    fig, (ax_price, ax_sentiment) = plt.subplots(2, 1, figsize=(14, 8))

    prices = []
    sentiments = []
    steps = []

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
        steps.append(state.step)

        ax_price.clear()
        ax_sentiment.clear()

        # ---- CANDLESTICK STYLE ----
        for i in range(len(prices)):
            color = "lime" if i == 0 or prices[i] >= prices[i - 1] else "red"
            ax_price.plot([steps[i], steps[i]],
                          [prices[i] * 0.995, prices[i] * 1.005],
                          color=color)
            ax_price.scatter(steps[i], prices[i], color=color, s=20)

        ax_price.set_title("Indian AI Market - Live Trading Terminal")
        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.2)

        ax_sentiment.plot(steps, sentiments)
        ax_sentiment.set_ylim(-1.1, 1.1)
        ax_sentiment.set_ylabel("Sentiment")
        ax_sentiment.set_xlabel("Step")
        ax_sentiment.grid(True, alpha=0.2)

        # ---- LIVE INFO PANEL ----
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

    # ---- SAVE AS GIF (NO FFMPEG REQUIRED) ----
    ani.save("indian_ai_market_live.gif", writer="pillow", fps=5)

    plt.show()


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--start-price", type=float, default=120)
    args = parser.parse_args()

    sim = Simulator(args.start_price, args.steps)
    states = sim.run()

    animate_market(states)


if __name__ == "__main__":
    main()
