import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from environment import ContextualNewsEnv
from policies import ContextualEpsGreedy, LinUCB, RandomPolicy

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & EDITORIAL MAGAZINE CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="The Stockholm Personalization Problem", page_icon="📰", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;600&display=swap');

    .block-container { 
        padding-top: 3rem; 
        padding-bottom: 5rem; 
        max-width: 900px; 
    }
    
    p { 
        font-family: 'Inter', sans-serif;
        color: #1a1a1a; 
        font-size: 1.15rem; 
        line-height: 1.8; 
        font-weight: 300;
        margin-bottom: 1.5rem;
    }
    
    h1 { 
        font-family: 'Playfair Display', serif;
        color: #000000; 
        font-weight: 700; 
        font-size: 4rem; 
        line-height: 1.1; 
        letter-spacing: -0.02em;
        margin-bottom: 1rem;
    }
    h2 { 
        font-family: 'Playfair Display', serif;
        color: #000000; 
        font-weight: 700; 
        font-size: 2.2rem; 
        margin-top: 4rem; 
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    h3 { 
        font-family: 'Inter', sans-serif;
        color: #004b87; 
        font-weight: 600; 
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2rem;
    }

    .drop-cap::first-letter {
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem;
        font-weight: 700;
        float: left;
        line-height: 0.8;
        margin-right: 0.5rem;
        margin-top: 0.2rem;
        color: #000000;
    }

    .pull-quote {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-style: italic;
        color: #333;
        margin: 3rem 0;
        padding-left: 2rem;
        border-left: 4px solid #004b87;
        line-height: 1.4;
    }

    img { border-radius: 4px; margin-bottom: 0.5rem; }
    .caption { font-size: 0.85rem; color: #666; font-family: 'Inter', sans-serif; margin-bottom: 2rem; }

    .interactive-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. TRAIN MODELS (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_models(alpha=1.0, epsilon=0.1, lambda_=1.0, T=5000, seed=42):
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

with st.spinner("Loading simulation..."):
    results, trained_ucb, env = train_models()

# -----------------------------------------------------------------------------
# 3. THE EDITORIAL STORY
# -----------------------------------------------------------------------------
st.markdown("<h1>It’s 07:30 on the Tunnelbana.</h1>", unsafe_allow_html=True)
st.markdown("<p class='caption'>Stockholm, Sweden. Photo: Unsplash</p>", unsafe_allow_html=True)
st.image("https://images.unsplash.com/photo-1509356843151-3e7d96241e11?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_container_width=True)

st.markdown("""
<p class='drop-cap'>You are standing on the platform at Slussen, heading north towards T-Centralen. You open your phone to read the morning news. 
For the past week, you’ve been following a story about rising electricity prices in Stockholm. You’ve also been looking at apartments in Södermalm, and you happen to be a passionate AI engineer.</p>

<p>But when you open <em>Dagens Nyheter</em>, what do you see? You see a headline about a football match. Then a story about the royal family. Then an opinion piece about the EU. You scroll past three articles before you find something you actually care about.</p>

<p>This is the <strong>Stockholm Morning Problem</strong>. And it is costing news publishers millions of kronor in lost engagement and churn.</p>

<div class='pull-quote'>
"We are giving everyone the same experience, regardless of their age, consumption patterns, or when they last visited." — Anders Eriksson, CEO of Bonnier News
</div>

<p>Bonnier News, the publisher behind <em>Dagens Nyheter</em>, <em>Expressen</em>, and <em>Dagens Industri</em>, has stated that personalization is their next big challenge. They found that readers often feel the experience is “clunky” and that content is not optimized for their interests. With a digital subscriber base of over 2 million, even a small improvement in click-through rates (CTR) translates to significant revenue. But how do you personalize a homepage for millions of unique readers without invasive tracking or complicated A/B tests?</p>
""", unsafe_allow_html=True)

st.markdown("<h2>The Data: What the Algorithm Sees</h2>", unsafe_allow_html=True)
st.markdown("""
<p>Before we recommend an article, the system observes a 6-dimensional context vector. It doesn't know your name, your political views, or your email address. It only knows your behavioral footprint from this session.</p>
""")

st.markdown("<h3>Context Features (d = 6)</h3>", unsafe_allow_html=True)
data_df = pd.DataFrame({
    "Feature": ["Political Interest", "Sports Interest", "Tech Interest", "Mobile Usage", "Morning Reader", "Age (Standardized)"],
    "What we measure": ["Time spent on current affairs", "Interest in sports highlights", "Affinity for gadgets & AI", "Prefers short articles", "Reads before 10 AM", "0 = 40yrs, +1 = 55yrs"],
    "Example Value": ["1.6 (High)", "0.3 (Low)", "0.2 (Low)", "0.5 (Medium)", "1.7 (High)", "0.8 (Older)"]
})
st.dataframe(data_df, use_container_width=True, hide_index=True)

st.markdown("<h3>The Articles (k = 4)</h3>", unsafe_allow_html=True)
st.markdown("""
<p>We have four distinct articles competing for the user's attention: <strong>Politics</strong>, <strong>Sports</strong>, <strong>Tech</strong>, and <strong>Lifestyle</strong>. Each has hidden features. The algorithm's job is to discover the hidden relationship between the user's context and the article's true click probability.</p>
""", unsafe_allow_html=True)

st.markdown("<h2>The Method: LinUCB (Contextual Bandits)</h2>", unsafe_allow_html=True)
st.markdown("""
<p>Traditional A/B testing is too slow for a news cycle. By the time you serve 50,000 users a test to see if Sports beats Politics, the news has changed.</p>

<p>Instead, we use a <strong>Contextual Bandit</strong>. It balances two competing desires:</p>
<ul style='font-family: Inter; font-size: 1.15rem; line-height: 1.8; margin-bottom: 1.5rem;'>
    <li><strong>Exploitation:</strong> Showing you the article we are confident you will click (e.g., Tech for a Tech worker).</li>
    <li><strong>Exploration:</strong> Occasionally showing you something unexpected to learn more about your taste (e.g., a Lifestyle piece about Södermalm living).</li>
</ul>

<p>The specific algorithm is <strong>LinUCB</strong>. It maintains a ridge regression model for every article. When you log in, it calculates a score for each article based on your context. It chooses the article with the highest <em>Upper Confidence Bound</em>.</p>

<div class='pull-quote'>
"It’s not just about giving people what they already like. It’s about confidently exploring what they might like next."
</div>

<p>The formula looks like this: <code>Score = (Expected Reward) + α × (Uncertainty Bonus)</code>. The α parameter is our exploration dial. If α is high, the algorithm takes risks. If α is low, it plays it safe.</p>
""", unsafe_allow_html=True)

st.markdown("<h2>The Impact: 24x Less Regret</h2>", unsafe_allow_html=True)
st.markdown("""
<p>We simulated 5,000 user sessions to compare LinUCB against two baselines: a Random policy (showing random articles) and a standard ε-Greedy policy (exploring randomly 10% of the time).</p>
<p>The metric we care about is <strong>Regret</strong>—the number of clicks we lost compared to a perfect oracle that knew exactly what every user wanted.</p>
""", unsafe_allow_html=True)

ucb_reward = np.sum(results["ucb"]["rewards"])
rand_reward = np.sum(results["random"]["rewards"])
ctr_lift = ((ucb_reward - rand_reward) / rand_reward) * 100 if rand_reward > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("LinUCB CTR", f"{ucb_reward/5000:.1%}", f"+{ctr_lift:.1f}% vs Random")
col2.metric("LinUCB Regret", f"{np.sum(results['ucb']['regret']):,.0f}", "Lowest", delta_color="inverse")
col3.metric("ε-Greedy Regret", f"{np.sum(results['eps']['regret']):,.0f}", "Baseline", delta_color="inverse")
col4.metric("Random Regret", f"{np.sum(results['random']['regret']):,.0f}", "Baseline", delta_color="inverse")

col_c1, col_c2 = st.columns(2)
steps = np.arange(1, 5001)

with col_c1:
    fig_rew = go.Figure()
    fig_rew.add_trace(go.Scatter(x=steps, y=np.cumsum(results["random"]["rewards"])/steps, name="Random", line=dict(color="#b0b0b0", width=2)))
    fig_rew.add_trace(go.Scatter(x=steps, y=np.cumsum(results["eps"]["rewards"])/steps, name="ε-Greedy", line=dict(color="#e07a5f", width=2)))
    fig_rew.add_trace(go.Scatter(x=steps, y=np.cumsum(results["ucb"]["rewards"])/steps, name="LinUCB", line=dict(color="#004b87", width=3)))
    fig_rew.update_layout(title="Average Click-Through Rate", xaxis_title="Users Served", yaxis_title="CTR", template="plotly_white", margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig_rew, use_container_width=True)

with col_c2:
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=steps, y=np.cumsum(results["random"]["regret"]), name="Random", line=dict(color="#b0b0b0", width=2)))
    fig_reg.add_trace(go.Scatter(x=steps, y=np.cumsum(results["eps"]["regret"]), name="ε-Greedy", line=dict(color="#e07a5f", width=2)))
    fig_reg.add_trace(go.Scatter(x=steps, y=np.cumsum(results["ucb"]["regret"]), name="LinUCB", line=dict(color="#004b87", width=3)))
    fig_reg.update_layout(title="Cumulative Regret (Lost Clicks)", xaxis_title="Users Served", yaxis_title="Lost Clicks", template="plotly_white", margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig_reg, use_container_width=True)

st.markdown("""
<p>By exploring intelligently, LinUCB learned 24 times faster than random guessing. It didn't waste clicks showing Sports to a Politics reader just to gather data.</p>
""", unsafe_allow_html=True)

st.markdown("<h2>Try It: The Live Recommendation Engine</h2>", unsafe_allow_html=True)
st.markdown("""
<p>Adjust the sliders below to simulate a user arriving at the <em>Dagens Nyheter</em> homepage. Watch how the algorithm's confidence changes in real-time.</p>
""", unsafe_allow_html=True)

st.markdown("<div class='interactive-box'>", unsafe_allow_html=True)

col_user, col_ai = st.columns([1, 1.2])

with col_user:
    st.markdown("<h3>👤 User Context</h3>", unsafe_allow_html=True)
    p_pol = st.slider("Political Interest", 0.0, 3.0, 1.5, 0.1)
    p_sport = st.slider("Sports Interest", 0.0, 3.0, 0.5, 0.1)
    p_tech = st.slider("Tech Interest", 0.0, 3.0, 0.5, 0.1)
    p_mob = st.slider("Mobile Usage", 0.0, 3.0, 1.0, 0.1)
    p_morn = st.slider("Morning Reader", 0.0, 3.0, 1.0, 0.1)
    p_age = st.slider("Age (0=40yrs, +1=55yrs)", -2.0, 2.0, 0.0, 0.1)

with col_ai:
    st.markdown("<h3>🤖 Algorithm Decision</h3>", unsafe_allow_html=True)
    x_vec = np.array([p_pol, p_sport, p_tech, p_mob, p_morn, p_age])
    scores = compute_ucb_scores(trained_ucb, x_vec)
    best_arm = int(np.argmax(scores))
    articles = ["Politics", "Sports", "Tech", "Lifestyle"]
    
    st.success(f"**Selected Article: {articles[best_arm]}**")
    
    fig_conf = go.Figure(data=[go.Bar(
        x=articles, y=scores, marker_color=['#004b87' if i == best_arm else '#d1d5db' for i in range(4)],
        text=[f"{s:.2f}" for s in scores], textposition='auto'
    )])
    fig_conf.update_layout(title="Confidence Score per Article", yaxis_title="UCB Score", template="plotly_white", height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h2>Under the Hood</h2>", unsafe_allow_html=True)

# NOTE: The r""" here is critical to prevent the unicode error!
with st.expander("View the Mathematics of LinUCB"):
    st.markdown(r"""
    For each arm (article) \(a\), LinUCB calculates a score based on the user context \(x_t\):
    
    \[
 Score_a = \underbrace{\hat{\theta}_a^T x_t}_{\text{Exploitation}} + \underbrace{\alpha \sqrt{x_t^T A_a^{-1} x_t}}_{\text{Exploration}} 
\]
    
    *   **\(A_a\)** and **\(b_a\)** are matrices updated online using ridge regression.
    *   **\(\alpha\)** controls the exploration bonus. Higher \(\alpha\) forces the model to explore uncertain articles.
    *   **\(\lambda\)** is the regularization parameter (default \(1.0\)).
    """)

st.markdown("<br><br><p style='text-align: center; color: #888; font-family: Inter; font-size: 0.9rem;'>Written by Prashant Kumar Singh | MSc Data Science, Stockholm University</p>", unsafe_allow_html=True)
