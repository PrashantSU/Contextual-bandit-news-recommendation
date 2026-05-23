"""
policies.py
-----------
Bandit algorithms implemented in this project:

Part 1 (non-contextual):
    EpsilonGreedy   — ε-greedy with incremental mean update
    UCB             — Upper Confidence Bound (UCB1)

Part 2 (contextual):
    RandomPolicy          — uniform random baseline
    ContextualEpsGreedy   — linear model per arm + ε-greedy exploration
    LinUCB                — ridge regression per arm + UCB bonus
"""

import numpy as np


# ---------------------------------------------------------------------------
# Part 1: Non-contextual bandits
# ---------------------------------------------------------------------------

def run_epsilon_greedy(probs, T, epsilon=0.1, seed=0):
    """
    Epsilon-greedy on a Bernoulli multi-armed bandit.

    Parameters
    ----------
    probs   : list[float]  true click probabilities for each arm
    T       : int          number of time steps
    epsilon : float        exploration probability
    seed    : int          random seed

    Returns
    -------
    dict with keys: rewards, regrets, actions, Q (value estimates), N (pull counts)
    """
    from environment import BernoulliBandit

    assert all(0.0 <= p <= 1.0 for p in probs)
    bandit = BernoulliBandit(probs)
    rng = np.random.default_rng(seed)

    k = len(probs)
    Q = np.zeros(k, dtype=float)
    N = np.zeros(k, dtype=int)
    rewards = np.zeros(T, dtype=int)
    regrets = np.zeros(T, dtype=float)
    actions = np.zeros(T, dtype=int)
    optimal_mean = float(np.max(probs))

    for t in range(T):
        if rng.random() < epsilon:
            arm = int(rng.integers(0, k))
        else:
            arm = int(np.argmax(Q))

        r = int(bandit.pull(arm))
        rewards[t] = r
        actions[t] = arm
        regrets[t] = optimal_mean - probs[arm]
        N[arm] += 1
        Q[arm] += (r - Q[arm]) / N[arm]

    return {"rewards": rewards, "regrets": regrets, "actions": actions, "Q": Q, "N": N}


def run_ucb(probs, T, c=1.0, seed=0):
    """
    UCB1 on a Bernoulli multi-armed bandit.

    UCB score: Q[a] + c * sqrt(ln(t) / N[a])
    Unvisited arms get score +inf (forced exploration).

    Parameters
    ----------
    probs : list[float]  true click probabilities
    T     : int          number of time steps
    c     : float        exploration coefficient
    seed  : int          random seed

    Returns
    -------
    dict with keys: rewards, regrets, actions, Q, N
    """
    from environment import BernoulliBandit

    assert all(0.0 <= p <= 1.0 for p in probs)
    bandit = BernoulliBandit(probs)
    rng = np.random.default_rng(seed)

    k = len(probs)
    Q = np.zeros(k, dtype=float)
    N = np.zeros(k, dtype=int)
    rewards = np.zeros(T, dtype=int)
    regrets = np.zeros(T, dtype=float)
    actions = np.zeros(T, dtype=int)
    optimal_mean = float(np.max(probs))

    for t in range(1, T + 1):
        ucb_scores = np.where(
            N == 0,
            np.inf,
            Q + c * np.sqrt(np.log(t) / N)
        )
        arm = int(np.argmax(ucb_scores))

        r = int(bandit.pull(arm))
        rewards[t - 1] = r
        actions[t - 1] = arm
        regrets[t - 1] = optimal_mean - probs[arm]
        N[arm] += 1
        Q[arm] += (r - Q[arm]) / N[arm]

    return {"rewards": rewards, "regrets": regrets, "actions": actions, "Q": Q, "N": N}


# ---------------------------------------------------------------------------
# Part 2: Contextual bandit policies
# ---------------------------------------------------------------------------

class Policy:
    """Abstract base class for contextual bandit policies."""
    def select(self, x: np.ndarray) -> int:
        raise NotImplementedError
    def update(self, arm: int, x: np.ndarray, r: int):
        pass


class RandomPolicy(Policy):
    """Uniform random baseline — ignores context entirely."""

    def __init__(self, k, seed=0):
        self.k = k
        self.rng = np.random.default_rng(seed)

    def select(self, x):
        return int(self.rng.integers(0, self.k))


class ContextualEpsGreedy(Policy):
    """
    Contextual epsilon-greedy with a linear reward model per arm.

    Each arm maintains a weight vector updated by stochastic gradient descent.
    With probability epsilon, a random arm is chosen (exploration).
    Otherwise the arm with highest predicted reward Q[a] @ x is chosen.
    """

    def __init__(self, k, d, epsilon=0.1, seed=0):
        self.eps = float(epsilon)
        self.Q = np.zeros((k, d))   # weight matrix: k arms x d features
        self.N = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng(seed)

    def select(self, x):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.Q.shape[0]))
        scores = self.Q @ x
        best = np.flatnonzero(scores == np.max(scores))
        return int(self.rng.choice(best))

    def update(self, arm, x, r):
        self.N[arm] += 1
        pred = self.Q[arm] @ x
        self.Q[arm] += (r - pred) * x / self.N[arm]


class LinUCB(Policy):
    """
    LinUCB with disjoint linear models (Li et al., 2010).

    Per arm: maintains A_a (regularized gram matrix) and b_a (reward vector).
    UCB score: theta_hat_a^T x + alpha * sqrt(x^T A_a^{-1} x)

    The bonus term (alpha * sqrt(...)) quantifies uncertainty and drives
    exploration in under-sampled regions of the feature space.
    """

    def __init__(self, k, d, alpha=1.0, lambda_=1.0):
        self.k = k
        self.d = d
        self.alpha = float(alpha)
        self.A = [lambda_ * np.eye(d) for _ in range(k)]   # regularized gram matrices
        self.b = [np.zeros(d) for _ in range(k)]            # reward vectors

    def select(self, x):
        vals = np.empty(self.k)
        for a in range(self.k):
            A_inv_x = np.linalg.solve(self.A[a], x)
            theta_hat = np.linalg.solve(self.A[a], self.b[a])
            vals[a] = theta_hat @ x + self.alpha * np.sqrt(x @ A_inv_x)
        return int(np.argmax(vals))

    def update(self, arm, x, r):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += r * x
