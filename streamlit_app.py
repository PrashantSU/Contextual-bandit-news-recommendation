"""Interactive LinUCB explorer — adjust α, ε, λ and watch the trade-off."""
import time

import matplotlib
matplotlib.use("Agg")  # headless backend required on Streamlit Cloud
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from environment import ContextualNewsEnv
from policies import ContextualEpsGreedy, LinUCB, RandomPolicy

st.set_page_config(page_title="Contextual Bandit Explorer", layout="wide")
st.title("🎯 Contextual Bandit Explorer")
st.markdown(
    """
    A **LinUCB** recommender learns which news article to show each user, from a
    6-dimensional context vector (politics, sports, tech, mobile, morning, age).

    Drag **α** to see the exploration-exploitation trade-off in real time.
    Baselines **Random** and **Contextual ε-Greedy** run alongside for reference.
    """
)

with st.sidebar:
    st.header("Hyperparameters")
    alpha = st.slider("LinUCB α (exploration bonus)", 0.0, 3.0, 1.0, 0.1,
                      help="0 = pure exploitation; higher = more exploration")
    epsilon = st.slider("ε-Greedy ε", 0.0, 0.5, 0.1, 0.01)
    lambda_ = st.slider("LinUCB λ (ridge regularization)", 0.01, 5.0, 1.0, 0.01)
    T = st.slider("Horizon T (steps)", 1000, 20000, 5000, 500)
    seed = st.number_input("Random seed", value=7, step=1)
    run = st.button("Run simulation", type="primary")


@st.cache_data(show_spinner=False)
def make_contexts(seed, T):
    env = ContextualNewsEnv(seed=int(seed))
    return [env.sample_context() for _ in range(int(T))], env.theta.copy()


def run_policy(policy, contexts, theta, env_seed):
    env = ContextualNewsEnv(seed=env_seed)
    rewards = np.zeros(len(contexts), dtype=int)
    regrets = np.zeros(len(contexts), dtype=float)
    for t, x in enumerate(contexts):
        arm = policy.select(x)
        r, p_sel = env.click(arm, x)
        rewards[t] = r
        true_scores = theta @ x
        p_star = 1.0 / (1.0 + np.exp(-np.max(true_scores)))
        regrets[t] = p_star - p_sel
        policy.update(arm, x, r)
    return rewards, regrets


if run:
    contexts, theta = make_contexts(seed, T)
    env = ContextualNewsEnv(seed=seed)
    k, d = env.k, env.d

    t0 = time.time()
    r_rand, reg_rand = run_policy(RandomPolicy(k, seed), contexts, theta, seed)
    r_eps, reg_eps = run_policy(
        ContextualEpsGreedy(k, d, epsilon, seed), contexts, theta, seed)
    r_ucb, reg_ucb = run_policy(
        LinUCB(k, d, alpha=alpha, lambda_=lambda_), contexts, theta, seed)
    elapsed = time.time() - t0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LinUCB avg reward", f"{r_ucb.mean():.3f}")
    col2.metric("LinUCB regret", f"{reg_ucb.sum():,.0f}")
    col3.metric("ε-Greedy regret", f"{reg_eps.sum():,.0f}")
    col4.metric("Random regret", f"{reg_rand.sum():,.0f}")

    steps = np.arange(1, T + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.plot(steps, np.cumsum(r_rand) / steps, label="Random", lw=1.5)
    ax1.plot(steps, np.cumsum(r_eps) / steps,
             label=f"ε-Greedy (ε={epsilon:.2f})", lw=1.5)
    ax1.plot(steps, np.cumsum(r_ucb) / steps,
             label=f"LinUCB (α={alpha:.2f})", lw=2)
    ax1.set_xlabel("t"); ax1.set_ylabel("Avg reward")
    ax1.set_title("Average Reward"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(steps, np.cumsum(reg_rand), label="Random", lw=1.5)
    ax2.plot(steps, np.cumsum(reg_eps),
             label=f"ε-Greedy (ε={epsilon:.2f})", lw=1.5)
    ax2.plot(steps, np.cumsum(reg_ucb),
             label=f"LinUCB (α={alpha:.2f})", lw=2)
    ax2.set_xlabel("t"); ax2.set_ylabel("Cumulative regret")
    ax2.set_title("Cumulative Regret"); ax2.legend(); ax2.grid(alpha=0.3)

    st.pyplot(fig, use_container_width=True)
    st.caption(f"Simulated {T:,} steps × 3 policies in {elapsed:.2f}s.")

    with st.expander("What am I looking at?"):
        st.markdown(
            f"""
            - **α = {alpha:.2f}** controls LinUCB's exploration bonus.
            - **α = 0** → pure exploitation, commits early, can get stuck.
            - **High α** → over-explores, steady regret tax.
            - **Sweet spot** → the α where regret flattens fastest.
            """
        )
else:
    st.info("👈 Adjust the sliders and click **Run simulation**.")
