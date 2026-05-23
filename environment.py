"""
environment.py
--------------
Contextual bandit environment simulating a news recommendation system.

Each user arrives with a 6-dimensional context vector (preferences + age).
Each article (arm) has a true linear reward parameter theta.
Rewards are Bernoulli-distributed with click probability = sigmoid(theta_a · x).
"""

import numpy as np


class BernoulliBandit:
    """Simple multi-armed Bernoulli bandit for Part 1 (non-contextual)."""

    def __init__(self, probs):
        self.probs = np.array(probs)
        self.k = len(probs)

    def pull(self, arm):
        return int(np.random.random() < self.probs[arm])

    @property
    def best_arm(self):
        return int(np.argmax(self.probs))


class ContextualNewsEnv:
    """
    Contextual bandit for news recommendation with interpretable features.

    Features (d=6):
        likes_politics, sports_fan, techie, mobile_user, morning_reader, age_z

    Arms (k=4):
        Politics, Sports, Tech, Lifestyle

    Reward model:
        P(click | arm a, context x) = sigmoid(theta[a] @ x)
    """

    FEATURE_NAMES = [
        "likes_politics",
        "sports_fan",
        "techie",
        "mobile_user",
        "morning_reader",
        "age_z",          # standardized age: (age - 40) / 15
    ]

    ARM_NAMES = ["Politics", "Sports", "Tech", "Lifestyle"]

    # True reward parameters (arms x features)
    THETA = np.array([
        [ 1.6,  0.2,  0.1,  0.2,  0.7,  0.4],   # Politics: older, morning readers
        [ 0.1,  1.8,  0.1,  0.7,  0.2, -0.1],   # Sports: mobile, slightly younger
        [ 0.0,  0.1,  1.9, -0.1, -0.2, -0.2],   # Tech: techies, younger skew
        [ 0.3,  0.2,  0.2,  1.0,  0.8,  0.0],   # Lifestyle: device + time driven
    ], dtype=float)

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.d = len(self.FEATURE_NAMES)
        self.k = len(self.ARM_NAMES)
        self.theta = self.THETA.copy()

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sample_age_z(self):
        age = float(np.clip(self.rng.normal(40, 12), 18, 80))
        return (age - 40.0) / 15.0

    def sample_context(self):
        """Sample an interpretable user context vector of shape (d,)."""
        seg = self.rng.choice(["politics", "sports", "tech", "on_the_go", "morning_person"])
        segment_cores = {
            "politics":      np.array([1.6, 0.2, 0.2, 0.3, 0.9]),
            "sports":        np.array([0.2, 1.8, 0.2, 1.0, 0.3]),
            "tech":          np.array([0.2, 0.2, 1.9, 0.3, 0.2]),
            "on_the_go":     np.array([0.4, 0.9, 0.5, 1.8, 0.7]),
            "morning_person":np.array([0.8, 0.3, 0.2, 0.6, 1.9]),
        }
        core = segment_cores[seg] + self.rng.normal(0, 0.2, size=5)
        x = np.empty(self.d)
        x[:5] = core
        x[5] = self._sample_age_z()
        return x

    def click_prob(self, arm, x):
        """Expected click probability for arm given context x."""
        return float(self._sigmoid(self.theta[arm] @ x))

    def click(self, arm, x):
        """Sample a Bernoulli reward. Returns (reward, click_probability)."""
        p = self.click_prob(arm, x)
        r = int(self.rng.random() < p)
        return r, p

    def best_arm(self, x):
        """Return the oracle-optimal arm and its click probability for context x."""
        scores = self.theta @ x
        arm = int(np.argmax(scores))
        return arm, float(self._sigmoid(scores[arm]))
