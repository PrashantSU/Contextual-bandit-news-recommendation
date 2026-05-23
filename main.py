"""
main.py
-------
Entry point: runs both Part 1 (non-contextual) and Part 2 (contextual) experiments
and produces result plots.

Usage:
    python main.py
"""

import numpy as np
from policies import run_epsilon_greedy, run_ucb
from simulate import run_comparison
from visualize import plot_noncontextual, plot_contextual_comparison, print_summary

# ---------------------------------------------------------------------------
# Part 1: Non-contextual bandits
# ---------------------------------------------------------------------------
PROBS   = [0.2, 0.1, 0.5, 0.9]
T       = 10_000
EPSILON = 0.1
C       = 1.0
SEED    = 123

print("Running Part 1: Non-Contextual Bandits...")
out_eps = run_epsilon_greedy(PROBS, T, epsilon=EPSILON, seed=SEED)
out_ucb = run_ucb(PROBS, T, c=C, seed=SEED)

plot_noncontextual(
    out_eps, out_ucb,
    probs=PROBS, T=T, epsilon=EPSILON, c=C,
    save_path="results/part1_noncontextual.png",
)

print(f"\nPart 1 Results (T={T:,})")
print(f"  ε-Greedy: total reward={out_eps['rewards'].sum()}, "
      f"avg={out_eps['rewards'].mean():.3f}, "
      f"cum regret={out_eps['regrets'].sum():.1f}")
print(f"  UCB:      total reward={out_ucb['rewards'].sum()}, "
      f"avg={out_ucb['rewards'].mean():.3f}, "
      f"cum regret={out_ucb['regrets'].sum():.1f}")

# ---------------------------------------------------------------------------
# Part 2: Contextual bandits
# ---------------------------------------------------------------------------
print("\nRunning Part 2: Contextual Bandits...")
results = run_comparison(T=20_000, epsilon=0.1, alpha=1.0, lambda_=1.0, seed=7)

plot_contextual_comparison(
    results,
    save_path="results/part2_contextual.png",
)
print_summary(results)
