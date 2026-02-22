import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from dataclasses import dataclass
from typing import List
import argparse

plt.style.use("dark_background")


@dataclass
class MarketState:
    step: int
    price: float
    sentiment: float
    volatility: float
    net_demand: float


class Simulator:
    def __init__(self, start_price=120, steps=80):
        self.start_price = start_price
        self.steps = steps

    def run(self) -> List[MarketState]:
        price = self.start_price
        states = []

        for step in range(1, self.steps + 1):
            sentiment = random.uniform(-1, 1)
            volatility = 0.01 + abs(sentiment) * 0.05

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


def animate_market(states: List[MarketState]):

    fig, (ax_price, ax_sentiment) = plt.subplots(2, 1, figsize=(12, 7))

    prices = []
    sentiments = []
    steps = []

    info_text = ax_price.text(
        0.02, 0.95, "", transform=ax_price.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.7)
    )

    def update(frame):
        state = states[frame]

        prices.append(state.price)
        sentiments.append(state.sentiment)
        steps.append(state.step)

        ax_price.clear()
        ax_sentiment.clear()

        # Price line
        ax_price.plot(steps, prices)
        ax_price.set_title("Indian AI Market - Live Animation")
        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.2)

        # Sentiment line
        ax_sentiment.plot(steps, sentiments)
        ax_sentiment.set_ylim(-1.1, 1.1)
        ax_sentiment.set_ylabel("Sentiment")
        ax_sentiment.set_xlabel("Step")
        ax_sentiment.grid(True, alpha=0.2)

        info = (
            f"Step: {state.step}\n"
            f"Price: {state.price:.2f}\n"
            f"Net Demand: {state.net_demand:+.3f}\n"
            f"Sentiment: {state.sentiment:+.2f}\n"
            f"Volatility: {state.volatility:.3f}"
        )

        info_text.set_text(info)
        ax_price.add_artist(info_text)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=200,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()

    # SAVE AFTER WINDOW CLOSES
    try:
        print("Saving GIF...")
        ani.save("indian_ai_market_live.gif", writer="pillow", fps=5)
        print("Saved successfully!")
    except Exception as e:
        print("Saving failed:", e)


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
