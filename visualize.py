"""
visualize.py
------------
Plotting functions for bandit experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_noncontextual(out_eps, out_ucb, probs, T, epsilon, c, save_path=None):
    """
    Side-by-side plots for ε-greedy and UCB (Part 1).
    Shows: average reward, cumulative regret, pull counts.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Part 1: Non-Contextual Bandits", fontsize=13)

    for row, (out, label) in enumerate([(out_eps, f"ε-Greedy (ε={epsilon})"),
                                         (out_ucb,  f"UCB (c={c})")]):
        avg = np.cumsum(out["rewards"]) / np.arange(1, T + 1)
        axes[row, 0].plot(avg)
        axes[row, 0].set_title(f"{label}: Average Reward")
        axes[row, 0].set_xlabel("Step"); axes[row, 0].set_ylabel("Avg reward")
        axes[row, 0].axhline(max(probs), color="red", linestyle="--",
                              linewidth=0.8, label=f"Optimal={max(probs)}")
        axes[row, 0].legend(fontsize=8)

        axes[row, 1].plot(np.cumsum(out["regrets"]))
        axes[row, 1].set_title(f"{label}: Cumulative Regret")
        axes[row, 1].set_xlabel("Step"); axes[row, 1].set_ylabel("Regret")

        counts = np.bincount(out["actions"], minlength=len(probs))
        axes[row, 2].bar(range(len(probs)), counts,
                          color=["#c0392b" if i == np.argmax(probs) else "#2980b9"
                                 for i in range(len(probs))])
        axes[row, 2].set_title(f"{label}: Pull Counts")
        axes[row, 2].set_xlabel("Arm"); axes[row, 2].set_ylabel("# pulls")
        axes[row, 2].set_xticks(range(len(probs)))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_contextual_comparison(results, save_path=None):
    """
    Average reward and cumulative regret curves for all three contextual policies.
    """
    meta = results["_meta"]
    T = meta["T"]

    labels = {
        "random": "Random",
        "eps":    f"Contextual ε-Greedy (ε={meta['epsilon']})",
        "linucb": f"LinUCB (α={meta['alpha']}, λ={meta['lambda_']})",
    }
    colors = {"random": "#7f8c8d", "eps": "#2980b9", "linucb": "#27ae60"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Part 2: Contextual Bandits — Policy Comparison", fontsize=13)

    for key in ("random", "eps", "linucb"):
        r = results[key]["rewards"]
        reg = results[key]["regrets"]
        axes[0].plot(np.cumsum(r) / np.arange(1, T + 1),
                     label=labels[key], color=colors[key])
        axes[1].plot(np.cumsum(reg),
                     label=labels[key], color=colors[key])

    axes[0].set_title("Average Reward (running mean)")
    axes[0].set_xlabel("Time step"); axes[0].set_ylabel("Avg reward")
    axes[0].legend()

    axes[1].set_title("Cumulative Regret")
    axes[1].set_xlabel("Time step"); axes[1].set_ylabel("Cumulative regret")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_summary(results):
    """Print final totals for each policy."""
    meta = results["_meta"]
    T = meta["T"]
    print(f"\nResults over T={T:,} steps")
    print("-" * 60)
    for key, label in [("random", "Random"),
                        ("eps",    "Contextual ε-Greedy"),
                        ("linucb", "LinUCB")]:
        r = results[key]["rewards"]
        reg = results[key]["regrets"]
        print(f"  {label:<24}  reward={r.sum():5d}  "
              f"avg={np.cumsum(r)[-1]/T:.3f}  "
              f"cum_regret={np.cumsum(reg)[-1]:.1f}")
