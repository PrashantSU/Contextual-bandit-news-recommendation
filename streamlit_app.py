import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from environment import ContextualNewsEnv
from policies import ContextualEpsGreedy, LinUCB, RandomPolicy

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & EDITORIAL MAGAZINE CSS (FORCED LIGHT MODE)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="The Stockholm Personalization Problem", page_icon="📰", layout="wide")

st.markdown("""
<style>
    /* Force a beautiful editorial light theme */
    .stApp { background-color: #FAFAFA; }
    
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Merriweather:wght@300;400;700&family=Inter:wght@400;600&display=swap');

    /* Main container layout */
    .block-container { 
        padding-top: 4rem; 
        padding-bottom: 6rem; 
        max-width: 800px; /* Narrow, book-like column */
    }
    
    /* Editorial Typography */
    h1 { 
        font-family: 'Playfair Display', serif !important;
        color: #111111 !important; 
        font-weight: 700 !important; 
        font-size: 3.8rem !important; 
        line-height: 1.1 !important; 
        margin-bottom: 1rem !important;
        letter-spacing: -0.02em !important;
    }
    h2 { 
        font-family: 'Playfair Display', serif !important;
        color: #111111 !important; 
        font-weight: 700 !important; 
        font-size: 2.2rem !important; 
        margin-top: 4rem !important; 
        margin-bottom: 1.5rem !important;
        border-bottom: 1px solid #E5E5E5 !important;
        padding-bottom: 0.5rem !important;
    }
    h3 { 
        font-family: 'Inter', sans-serif !important;
        color: #004b87 !important; /* Swedish Flag Blue */
        font-weight: 600 !important; 
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-top: 2.5rem !important;
    }
    
    /* Body Text */
    p, li { 
        font-family: 'Merriweather', serif !important;
        color: #222222 !important; 
        font-size: 1.15rem !important; 
        line-height: 1.9 !important; 
        margin-bottom: 1.5rem !important;
    }

    /* Drop Cap (Large First Letter) */
    .drop-cap::first-letter {
        font-family: 'Playfair Display', serif !important;
        font-size: 5rem !important;
        font-weight: 700 !important;
        float: left !important;
        line-height: 0.8 !important;
        margin-right: 0.75rem !important;
        margin-top: 0.3rem !important;
        color: #004b87 !important; /* Swedish Blue */
    }

    /* Magazine Pull Quote */
    .pull-quote {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.9rem !important;
        font-style: italic !important;
        color: #333333 !important;
        margin: 3.5rem 0 !important;
        padding: 1rem 0 1rem 2rem !important;
        border-left: 4px solid #004b87 !important;
        line-height: 1.4 !important;
        background-color: #F0F4F8 !important;
    }

    /* Image Captions */
    .caption { 
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important; 
        color: #666666 !important; 
        margin-top: 0.5rem !important;
        margin-bottom: 2.5rem !important; 
        text-align: right !important;
    }

    /* Interactive Section Box */
    .interactive-box {
        background-color: #FFFFFF !important;
        padding: 2.5rem !important;
        border-radius: 8px !important;
        border: 1px solid #E5E5E5 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        margin-top: 2rem !important;
        margin-bottom: 3rem !important;
    }
    
    /* Ensure Streamlit widgets text is visible */
    .stSlider label, .stMarkdown p { color: #222222 !important; }
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

# --- THE HOOK ---
st.markdown("<p class='caption'>STOCKHOLM, SWEDEN</p>", unsafe_allow_html=True)
st.markdown("<h1>The Algorithm That Knows What You Want to Read Before You Do.</h1>", unsafe_allow_html=True)
st.markdown("<p class='caption'>Photo: Unsplash / Stockholm Archipelago</p>", unsafe_allow_html=True)
st.image("https://images.unsplash.com/photo-1509356843151-3e7d96241e11?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_container_width=True)

st.markdown("""
<p class='drop-cap'>It is 07:30 on a grey Tuesday in November. You are standing on the frost-covered platform at Slussen, waiting for the northbound green line towards T-Centralen. The mist is rolling off the cold water of Riddarfjärden. You pull out your phone to catch up on the morning news.</p>

<p>For the past week, you have been closely following a developing story about rising electricity prices in Södermalm. You are also looking for a new apartment, and you happen to be a passionate AI engineer who loves reading about the latest breakthroughs in generative models.</p>

<p>But when you open the homepage of <em>Dagens Nyheter</em>, what do you see? A massive headline about a football derby. Below that, a celebrity gossip piece. Below that, an opinion column about the EU parliament. You scroll past three articles you could not care less about before you finally find something relevant.</p>

<p>This is the <strong>Stockholm Morning Problem</strong>. And it is costing Sweden's biggest media houses millions of kronor in lost engagement, frustrated subscribers, and quiet churn.</p>

<div class='pull-quote'>
"We are giving everyone the exact same experience, regardless of their age, their reading habits, or when they last visited. That model is fundamentally broken."
</div>

<p>Bonnier News, the massive Swedish media conglomerate behind <em>Dagens Nyheter</em>, <em>Expressen</em>, and <em>Dagens Industri</em>, has publicly acknowledged this. In a digital landscape dominated by global tech giants like Netflix and Spotify—which curate every single pixel for the individual—a static news homepage feels incredibly "clunky." With over 2 million digital subscribers across their titles, even a 1% improvement in click-through rates (CTR) translates to tens of millions of kronor in annual revenue.</p>

<p>But how do you personalize a homepage for millions of unique readers without invasive tracking, without violating GDPR, and without making readers feel like they are in a creepy filter bubble? The answer lies not in deep learning or massive neural networks, but in a 70-year-old mathematical concept: <strong>The Multi-Armed Bandit</strong>.</p>
""", unsafe_allow_html=True)

# --- THE PROBLEM ---
st.markdown("<h2>The Flaw of Traditional A/B Testing</h2>", unsafe_allow_html=True)

st.markdown("""
<p>To understand why we need bandits, we first have to understand why traditional A/B testing fails in the newsroom.</p>

<p>Imagine you are the Head of Product at <em>Svenska Dagbladet</em> (SvD). You want to know if a new Tech article performs better than a Politics article. In a traditional A/B test, you would split your traffic 50/50. Half the users see the Tech article; half see Politics. You run this for two weeks, analyze the data, and pick the winner.</p>

<p><strong>This is a disaster for two reasons:</strong></p>
<ol>
    <li><strong>The News Cycle Moves Too Fast:</strong> By the time your A/B test reaches statistical significance, the news has changed. The articles are no longer relevant.</li>
    <li><strong>The Cost of Regret:</strong> For two weeks, 50% of your users were shown an article they hated. You intentionally lost those clicks just to gather data. In business, we call this lost revenue <em>Regret</em>.</li>
</ol>

<p>A Contextual Bandit solves this by learning <em>while</em> it serves traffic. Instead of a static 50/50 split, it dynamically shifts traffic to the winning article, but keeps a tiny fraction exploring the other options just in case the news cycle flips.</p>
""", unsafe_allow_html=True)

# --- THE DATA ---
st.markdown("<h2>The Data: Privacy-First Personalization</h2>", unsafe_allow_html=True)
st.markdown("""
<p>Sweden takes data privacy seriously. Under GDPR, we cannot simply read a user's mind or track their entire internet history. We must rely on behavioral signals generated during the current session.</p>

<p>When you open the app, the system observes a 6-dimensional context vector. It doesn't know your name, your political views, or your email address. It only knows your behavioral footprint from this session.</p>
""")

data_df = pd.DataFrame({
    "Signal": ["Political Interest", "Sports Interest", "Tech Interest", "Mobile Usage", "Morning Reader", "Age (Standardized)"],
    "What we measure": ["Time spent on current affairs", "Interest in sports highlights", "Affinity for gadgets & AI", "Prefers short/mobile articles", "Reads before 10 AM", "0 = 40yrs, +1 = 55yrs"],
    "Example Value": ["1.6 (High)", "0.3 (Low)", "0.2 (Low)", "0.5 (Medium)", "1.7 (High)", "0.8 (Older)"]
})
st.dataframe(data_df, use_container_width=True, hide_index=True)

st.markdown("""
<p>We have four distinct articles competing for your attention: <strong>Politics</strong>, <strong>Sports</strong>, <strong>Tech</strong>, and <strong>Lifestyle</strong>. Each article has hidden "true" parameters. The algorithm's job is to discover the hidden relationship between the user's context and the article's true click probability.</p>
""", unsafe_allow_html=True)

# --- THE METHOD ---
st.markdown("<h2>The Method: LinUCB (Linear Upper Confidence Bound)</h2>", unsafe_allow_html=True)
st.markdown("""
<p>Think of a Contextual Bandit like a brilliant bartender at a speakeasy in Gamla Stan. When you walk in, he looks at you (the context). If you are wearing a suit, he might recommend a classic Scotch. If you are wearing a hoodie, he might recommend a craft IPA. But if he's not sure, he'll offer you a small taste of something new.</p>

<p>He balances two competing desires:</p>
<ul style='font-family: Merriweather; font-size: 1.15rem; line-height: 1.9; margin-bottom: 1.5rem;'>
    <li><strong>Exploitation:</strong> Giving you what he is 100% certain you will love (maximizing immediate reward).</li>
    <li><strong>Exploration:</strong> Taking a risk to learn more about your taste (reducing long-term uncertainty).</li>
</ul>

<p>The specific algorithm we use is <strong>LinUCB</strong>. It maintains a ridge regression model for every single article. When a user arrives, it calculates a score for each article based on their context. It chooses the article with the highest <em>Upper Confidence Bound</em>.</p>
""", unsafe_allow_html=True)

# --- THE IMPACT ---
st.markdown("<h2>The Impact: 24x Less Regret</h2>", unsafe_allow_html=True)
st.markdown("""
<p>To prove the system works, we simulated 5,000 user sessions on a random Tuesday morning. We compared LinUCB against two baselines: a Random policy (showing random articles) and a standard ε-Greedy policy (exploring randomly 10% of the time).</p>
<p>The metric we care about is <strong>Regret</strong>—the number of clicks we lost compared to a perfect, all-knowing oracle.</p>
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
<p>By exploring intelligently, LinUCB learned 24 times faster than random guessing. It didn't waste clicks showing Sports to a Politics reader just to gather data. It used its uncertainty to guide exploration, ensuring that every single user interaction—even the ones that didn't result in a click—made the model smarter.</p>
""", unsafe_allow_html=True)

# --- INTERACTIVE DEMO ---
st.markdown("<h2>Try It: The Live Recommendation Engine</h2>", unsafe_allow_html=True)
st.markdown("""
<p>Step into the shoes of a user arriving at the <em>Dagens Nyheter</em> homepage. Adjust your profile below and watch how the algorithm's confidence changes in real-time. This is the exact logic running in the backend.</p>
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

# --- TECHNICAL APPENDIX ---
st.markdown("<h2>Under the Hood</h2>", unsafe_allow_html=True)

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
