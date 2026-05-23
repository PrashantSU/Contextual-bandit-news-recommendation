"""
simulate.py
-----------
Simulation helpers for running bandit policies and collecting metrics.
"""

import numpy as np
from environment import ContextualNewsEnv
from policies import RandomPolicy, ContextualEpsGreedy, LinUCB


def run_on_contexts(policy, env_seed, contexts):
    """
    Run a contextual policy on a fixed sequence of contexts.

    Parameters
    ----------
    policy    : Policy instance
    env_seed  : int   seed for the reward environment
    contexts  : list  pre-sampled context vectors

    Returns
    -------
    rewards      : np.array[T]  observed binary rewards
    inst_regret  : np.array[T]  expected instantaneous regret at each step
    """
    env = ContextualNewsEnv(seed=env_seed)
    T = len(contexts)
    rewards = np.zeros(T, dtype=int)
    inst_regret = np.zeros(T, dtype=float)

    for t, x in enumerate(contexts):
        arm = policy.select(x)
        r, p_sel = env.click(arm, x)
        rewards[t] = r

        # Instantaneous regret = p*(x) - p_selected(x)
        true_scores = env.theta @ x
        p_star = 1.0 / (1.0 + np.exp(-np.max(true_scores)))
        inst_regret[t] = p_star - p_sel

        policy.update(arm, x, r)

    return rewards, inst_regret


def run_comparison(T=20000, epsilon=0.1, alpha=1.0, lambda_=1.0, seed=7):
    """
    Run Random, Contextual ε-Greedy, and LinUCB on the same context sequence.

    Returns
    -------
    dict with results for each algorithm:
        {
          'random':   {'rewards': ..., 'regrets': ...},
          'eps':      {'rewards': ..., 'regrets': ...},
          'linucb':   {'rewards': ..., 'regrets': ...},
        }
    and metadata: T, epsilon, alpha, lambda_, seed
    """
    env = ContextualNewsEnv(seed=seed)
    k, d = env.k, env.d
    contexts = [env.sample_context() for _ in range(T)]

    policies = {
        "random": RandomPolicy(k, seed=seed),
        "eps":    ContextualEpsGreedy(k, d, epsilon=epsilon, seed=seed),
        "linucb": LinUCB(k, d, alpha=alpha, lambda_=lambda_),
    }

    results = {}
    for name, policy in policies.items():
        rewards, regrets = run_on_contexts(policy, seed, contexts)
        results[name] = {"rewards": rewards, "regrets": regrets}

    results["_meta"] = {
        "T": T, "epsilon": epsilon, "alpha": alpha,
        "lambda_": lambda_, "seed": seed,
    }
    return results
