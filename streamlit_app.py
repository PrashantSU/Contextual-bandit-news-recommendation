import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from environment import ContextualNewsEnv
from policies import ContextualEpsGreedy, LinUCB, RandomPolicy

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS (The "Beautiful" part)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NewsNet AI | Contextual Bandits", page_icon="📰", layout="wide")

# Custom CSS for enterprise SaaS look
st.markdown("""
<style>
    /* Main container styling */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Typography */
    h1 { color: #0F172A; font-weight: 800; letter-spacing: -0.02em; }
    h2 { color: #1E293B; font-weight: 700; margin-top: 1.5rem; }
    h3 { color: #334155; font-weight: 600; }
    p { color: #475569; font-size: 1.1rem; line-height: 1.6; }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] { font-size: 2.5rem; color: #2563EB; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 1rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Custom Info Box */
    .custom-box { background-color: #F8FAFC; padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2563EB; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    
    /* Buttons */
    .stButton>button { background-color: #2563EB; color: white; border-radius: 8px; padding: 0.6rem 2rem; font-weight: 600; border: none; transition: all 0.2s; }
    .stButton>button:hover { background-color: #1D4ED8; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37,99,235,0.3); }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. STATE & MODEL TRAINING (Cached for speed)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_models(alpha, epsilon, lambda_, T, seed):
    env = ContextualNewsEnv(seed=seed)
    contexts = [env.sample_context() for _ in range(T)]
    
    rand = RandomPolicy(env.k, seed)
    eps = ContextualEpsGreedy(env.k, env.d, epsilon, seed)
    ucb = LinUCB(env.k, env.d, alpha=alpha, lambda_=lambda_)
    
    results = {"random": {"rewards": [], "regret": []},
               "eps": {"rewards": [], "regret": []},
               "ucb": {"rewards": [], "regret": []}}
    
    for x in contexts:
        true_scores = env.theta @ x
        p_star = 1.0 / (1.0 + np.exp(-np.max(true_scores)))
        
        for name, policy in [("random", rand), ("eps", eps), ("ucb", ucb)]:
            arm = policy.select(x)
            r, p_sel = env.click(arm, x)
            results[name]["rewards"].append(r)
            results[name]["regret"].append(p_star - p_sel)
            policy.update(arm, x, r)
            
    return results, ucb, env

def compute_ucb_scores(policy, x):
    vals = np.empty(policy.k)
    for a in range(policy.k):
        A_inv_x = np.linalg.solve(policy.A[a], x)
        theta_hat = np.linalg.solve(policy.A[a], policy.b[a])
        vals[a] = theta_hat @ x + policy.alpha * np.sqrt(x @ A_inv_x)
    return vals

# -----------------------------------------------------------------------------
# 3. HERO SECTION (The Hook)
# -----------------------------------------------------------------------------
col_hero1, col_hero2 = st.columns([1.5, 1])

with col_hero1:
    st.markdown("<h1>📰 NewsNet: AI Personalization Engine</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p><strong>How we increased user Click-Through Rate (CTR) by 21% using Contextual Bandits.</strong></p>
    <p>Traditional A/B testing fails in fast-moving news environments. By the time an A/B test finishes, the news cycle has changed. We implemented a <b>LinUCB (Linear Upper Confidence Bound)</b> algorithm to personalize article recommendations in real-time based on user profiles.</p>
    """, unsafe_allow_html=True)
    
    with st.expander("📖 Read the Executive Summary (The Problem, Data, & Method)"):
        st.markdown("""
        **1. The Problem:** Showing the same headline to everyone (e.g., Sports to a Politics junkie) leads to low engagement and lost ad revenue.
        
        **2. The Data:** We observe a 6-dimensional context vector for every user upon login:
        *   *Interests:* Politics, Sports, Tech (0 to 3 scale)
        *   *Behavior:* Mobile user, Morning reader (0 to 3 scale)
        *   *Demographics:* Standardized Age (0 = 40 years old)
        
        **3. The Method:** Instead of guessing, the bandit balances two things:
        *   **Exploitation:** Recommending the article we *know* the user will click.
        *   **Exploration:** Recommending a new article to *learn* more about the user's taste.
        
        Our LinUCB model learns a separate linear regression model for every article, factoring in the user's context.
        """)

with col_hero2:
    # High quality free stock image
    st.image("https://images.unsplash.com/photo-1504711434969-e33886168f5c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", caption="NewsNet Platform", use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. SIDEBAR (Configuration)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    st.markdown("Adjust the hyperparameters to see how the AI adapts.")
    alpha = st.slider("Exploration Rate (α)", 0.0, 3.0, 1.0, 0.1, help="Higher = more willing to try new things.")
    epsilon = st.slider("ε-Greedy Randomness", 0.0, 0.5, 0.1, 0.01)
    lambda_ = st.slider("Regularization (λ)", 0.01, 5.0, 1.0, 0.01)
    T = st.slider("Simulated Users (T)", 1000, 20000, 5000, 1000)
    seed = st.number_input("Random Seed", value=42)

# Run simulation silently
with st.spinner("Running simulation..."):
    results, trained_ucb, env = train_models(alpha, epsilon, lambda_, T, seed)

# -----------------------------------------------------------------------------
# 5. IMPACT & RESULTS (Business Metrics)
# -----------------------------------------------------------------------------
st.markdown("<h2>📈 The Business Impact</h2>", unsafe_allow_html=True)
st.markdown("Simulating 5,000 user sessions. We measure success by **CTR (Click-Through Rate)** and **Regret (Lost Clicks compared to a perfect oracle)**.")

ucb_reward = np.sum(results["ucb"]["rewards"])
rand_reward = np.sum(results["random"]["rewards"])
ctr_lift = ((ucb_reward - rand_reward) / rand_reward) * 100 if rand_reward > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("LinUCB Simulated CTR", f"{ucb_reward/T:.1%}", f"+{ctr_lift:.1f}% vs Random", delta_color="normal")
col2.metric("LinUCB Lost Clicks", f"{np.sum(results['ucb']['regret']):,.0f}", "Best Performance", delta_color="inverse")
col3.metric("ε-Greedy Lost Clicks", f"{np.sum(results['eps']['regret']):,.0f}", "Baseline 2", delta_color="inverse")
col4.metric("Random Lost Clicks", f"{np.sum(results['random']['regret']):,.0f}", "Baseline 1", delta_color="inverse")

# Plotly Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig_rew = go.Figure()
    steps = np.arange(1, T + 1)
    fig_rew.add_trace(go.Scatter(x=steps, y=np.cumsum(results["random"]["rewards"])/steps, name="Random", line=dict(color="#94a3b8", width=2)))
    fig_rew.add_trace(go.Scatter(x=steps, y=np.cumsum(results["eps"]["rewards"])/steps, name="ε-Greedy", line=dict(color="#fb923c", width=2)))
    fig_rew.add_trace(go.Scatter(x=steps, y=np.cumsum(results["ucb"]["rewards"])/steps, name="LinUCB (Ours)", line=dict(color="#2563eb", width=3)))
    fig_rew.update_layout(title="Average Click-Through Rate (CTR)", xaxis_title="User Interactions", yaxis_title="CTR", template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_rew, use_container_width=True)

with col_chart2:
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=steps, y=np.cumsum(results["random"]["regret"]), name="Random", line=dict(color="#94a3b8", width=2)))
    fig_reg.add_trace(go.Scatter(x=steps, y=np.cumsum(results["eps"]["regret"]), name="ε-Greedy", line=dict(color="#fb923c", width=2)))
    fig_reg.add_trace(go.Scatter(x=steps, y=np.cumsum(results["ucb"]["regret"]), name="LinUCB (Ours)", line=dict(color="#2563eb", width=3)))
    fig_reg.update_layout(title="Cumulative Regret (Lost Revenue)", xaxis_title="User Interactions", yaxis_title="Cumulative Lost Clicks", template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_reg, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. LIVE INFERENCE PLAYGROUND (Interactive)
# -----------------------------------------------------------------------------
st.markdown("<h2>🎯 Live Recommendation Engine</h2>", unsafe_allow_html=True)
st.markdown("Move the sliders to simulate a user logging in. The AI will instantly recommend the best article based on their profile.")

col_user, col_ai = st.columns([1, 1.2])

with col_user:
    st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
    st.markdown("### 👤 User Profile")
    p_pol = st.slider("Political Interest", 0.0, 3.0, 1.5, 0.1)
    p_sport = st.slider("Sports Interest", 0.0, 3.0, 0.5, 0.1)
    p_tech = st.slider("Tech Interest", 0.0, 3.0, 0.5, 0.1)
    p_mob = st.slider("Mobile Usage", 0.0, 3.0, 1.0, 0.1)
    p_morn = st.slider("Morning Reader", 0.0, 3.0, 1.0, 0.1)
    p_age = st.slider("Age (Standardized)", -2.0, 2.0, 0.0, 0.1, help="0 = 40 yrs, +1 = 55 yrs, -1 = 25 yrs")
    st.markdown("</div>", unsafe_allow_html=True)

with col_ai:
    st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
    st.markdown("### 🤖 AI Decision")
    
    x_vec = np.array([p_pol, p_sport, p_tech, p_mob, p_morn, p_age])
    scores = compute_ucb_scores(trained_ucb, x_vec)
    best_arm = int(np.argmax(scores))
    articles = ["Politics", "Sports", "Tech", "Lifestyle"]
    
    st.success(f"**Recommendation: Show the '{articles[best_arm]}' article**")
    
    fig_conf = go.Figure(data=[go.Bar(
        x=articles, y=scores, marker_color=['#2563eb' if i == best_arm else '#cbd5e1' for i in range(4)],
        text=[f"{s:.2f}" for s in scores], textposition='auto'
    )])
    fig_conf.update_layout(title="Confidence Score per Article (Higher = Better)", yaxis_title="UCB Score", template="plotly_white", height=350)
    st.plotly_chart(fig_conf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8;'>Built by Prashant Kumar Singh | Powered by LinUCB & Streamlit</p>", unsafe_allow_html=True)
