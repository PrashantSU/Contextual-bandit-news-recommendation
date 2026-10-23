"""
The Stockholm Personalization Problem: portfolio edition
Run:  streamlit run bandit_app.py
Needs your existing environment.py and policies.py next to this file.
Tip:  add .streamlit/config.toml with
      [theme]
      base = "light"
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from environment import ContextualNewsEnv
from policies import ContextualEpsGreedy, LinUCB, RandomPolicy

# -----------------------------------------------------------------------------
# CONFIG: edit these
# -----------------------------------------------------------------------------
AUTHOR = "Prashant Kumar Singh"
TAGLINE = "MSc Data Science, Stockholm University"
GITHUB = "https://github.com/"          # replace with your repo
LINKEDIN = "https://www.linkedin.com/"  # replace with your profile

IMG_HERO = "https://images.unsplash.com/photo-1509356843151-3e7d96241e11?auto=format&fit=crop&w=1800&q=80"
IMG_NEWS = "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1400&q=80"
IMG_BAR = "https://images.unsplash.com/photo-1514933651103-005eec06c04b?auto=format&fit=crop&w=1400&q=80"

BLUE, GOLD, INK, GREY, MIST, LINE = "#006AA7", "#E3AE00", "#0E1A26", "#A7B2BD", "#EEF3F7", "#DAE2E9"
ARTICLES = ["Politics", "Sports", "Tech", "Lifestyle"]
FEATURES = ["pol", "sport", "tech", "mob", "morn", "age"]
LABELS = ["Political interest", "Sports interest", "Tech interest", "Mobile usage", "Morning reader", "Age (0 = 40 yrs, +1 = 55 yrs)"]
RANGES = [(0.0, 3.0), (0.0, 3.0), (0.0, 3.0), (0.0, 3.0), (0.0, 3.0), (-2.0, 2.0)]
PERSONAS = {
    "Custom": None,
    "Commuter who follows politics": [2.6, 0.3, 0.4, 1.8, 2.4, 0.9],
    "Weekend sports fan": [0.4, 2.7, 0.5, 1.2, 0.6, -0.8],
    "AI engineer on a lunch break": [0.6, 0.4, 2.8, 0.8, 0.5, -0.5],
    "Casual evening scroller": [0.5, 0.6, 0.4, 2.2, 0.2, 0.3],
}
T_STEPS = 5000

st.set_page_config(page_title="The Stockholm Personalization Problem", page_icon="📰", layout="centered")

# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');
:root { --blue:#006AA7; --gold:#E3AE00; --ink:#0E1A26; --mist:#EEF3F7; --line:#DAE2E9; --muted:#5E6E7D; }
.stApp { background:#fff; }
.block-container { max-width:860px; padding-top:2rem; padding-bottom:5rem; }
header[data-testid="stHeader"] { background:transparent; }
h1,h2,h3 { font-family:'Archivo',sans-serif !important; color:var(--ink) !important; }
h1 { font-weight:800 !important; font-size:3rem !important; line-height:1.08 !important; letter-spacing:-0.035em !important; }
h2 { font-weight:800 !important; font-size:2rem !important; letter-spacing:-0.025em !important; margin-top:3.6rem !important; }
h3 { font-weight:700 !important; font-size:1.2rem !important; margin-top:2rem !important; }
p, li, label, .stMarkdown { font-family:'Source Serif 4',serif !important; color:#27343F; font-size:1.13rem; line-height:1.8; }
.body { max-width:64ch; margin-bottom:1.3rem; }
.body b, .body strong { color:var(--ink); }
.ui, .ui p { font-family:'Archivo',sans-serif !important; }

.hero { position:relative; border-radius:4px; overflow:hidden; min-height:440px; display:flex; align-items:flex-end; background:var(--blue); }
.hero-img { position:absolute; inset:0; background-size:cover; background-position:center 65%; }
.hero-tint { position:absolute; inset:0; background:linear-gradient(180deg,rgba(0,106,167,.25) 0%,rgba(8,42,66,.92) 100%); }
.hero-body { position:relative; padding:2.6rem 2.8rem; }
.hero h1 { color:#fff !important; margin:0 0 1rem 0; max-width:15ch; }
.hero p { font-family:'Archivo',sans-serif !important; color:#D6E6F1 !important; font-size:1.1rem; line-height:1.6; max-width:50ch; margin:0; }
.credit { font-family:'Archivo',sans-serif !important; font-size:.8rem !important; color:#7A8996 !important; margin:.5rem 0 2rem 0; }

.stat { background:#fff; border:1px solid var(--line); border-top:4px solid var(--blue); border-radius:3px; padding:1.1rem 1.2rem; height:100%; }
.stat.gold { border-top-color:var(--gold); }
.stat .v { font-family:'Archivo',sans-serif; font-weight:800; font-size:2.2rem; color:var(--ink); line-height:1.1; letter-spacing:-0.03em; }
.stat .l { font-family:'Archivo',sans-serif; font-weight:700; font-size:.95rem; color:var(--ink); margin-top:.3rem; }
.stat .n { font-family:'Archivo',sans-serif; font-size:.85rem; color:var(--muted); line-height:1.45; margin-top:.2rem; }

.drop::first-letter { font-family:'Archivo',sans-serif; font-weight:800; font-size:4.6rem; float:left; line-height:.82; margin:.35rem .7rem 0 0; color:var(--blue); }
.pull { font-family:'Source Serif 4',serif; font-style:italic; font-size:1.7rem; line-height:1.45; color:var(--ink); margin:2.8rem 0; padding-left:1.6rem; border-left:5px solid var(--gold); max-width:30ch; }

.photo { border-radius:3px; height:260px; background-size:cover; background-position:center; background-color:var(--line); }
.cap { font-family:'Archivo',sans-serif !important; font-size:.82rem !important; color:var(--muted) !important; margin:.5rem 0 2rem 0 !important; line-height:1.4 !important; }

.loop { display:flex; gap:.6rem; align-items:stretch; margin:1.6rem 0 2rem 0; flex-wrap:wrap; }
.loop .step { flex:1 1 180px; background:var(--mist); border-radius:3px; padding:1rem 1.1rem; font-family:'Archivo',sans-serif; }
.loop .step b { display:block; color:var(--ink); font-size:1rem; margin-bottom:.25rem; }
.loop .step span { font-size:.9rem; color:#3B4A57; line-height:1.5; }
.loop .arrow { align-self:center; color:var(--blue); font-weight:800; font-size:1.4rem; }

.card { border:1px solid var(--line); border-radius:3px; padding:1.2rem 1.3rem; height:100%; background:#fff; }
.card b { font-family:'Archivo',sans-serif; color:var(--ink); font-size:1.05rem; }
.card p { font-family:'Archivo',sans-serif !important; font-size:.95rem !important; line-height:1.6 !important; margin:.4rem 0 0 0 !important; }
.callout { background:var(--mist); border-left:5px solid var(--blue); border-radius:2px 10px 10px 2px; padding:1rem 1.3rem; margin:1.2rem 0; font-family:'Archivo',sans-serif; font-size:.98rem; line-height:1.65; color:#27343F; }
.callout.gold { border-left-color:var(--gold); }
.verdict { background:var(--ink); color:#fff; border-radius:4px; padding:1.3rem 1.6rem; margin:1rem 0; font-family:'Archivo',sans-serif; }
.verdict .k { color:#9FB3C4; font-size:.9rem; }
.verdict .v { font-weight:800; font-size:1.9rem; letter-spacing:-0.02em; }
.verdict .ok { color:#6FD3A0; } .verdict .no { color:#FFB4A8; }
.tag { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:.2rem .8rem; margin:.2rem .3rem .2rem 0; font-family:'Archivo',sans-serif; font-size:.85rem; color:var(--ink); }

[data-testid="stForm"] { background:var(--mist); border:1px solid var(--line); border-radius:4px; }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:4px !important; }
[data-testid="stExpander"] summary { font-family:'Archivo',sans-serif !important; font-weight:700 !important; }
.js-plotly-plot { border:1px solid var(--line); border-radius:4px; }
.stButton button, .stFormSubmitButton button { font-family:'Archivo',sans-serif; font-weight:700; border-radius:3px; border:1.5px solid var(--blue); color:var(--blue); }
@media (max-width:700px) { h1 { font-size:2.2rem !important; } .hero-body { padding:1.5rem; } }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def html(s: str):
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def body(s: str, cls=""):
    html(f"<p class='body {cls}'>{s}</p>")


def stat(label, value, note="", tone=""):
    return f"<div class='stat {tone}'><div class='v'>{value}</div><div class='l'>{label}</div><div class='n'>{note}</div></div>"


def callout(s, tone=""):
    html(f"<div class='callout {tone}'>{s}</div>")


def photo(url, caption, h=260):
    html(f"<div class='photo' style=\"background-image:url('{url}');height:{h}px\"></div><p class='cap'>{caption}</p>")


def style(fig, h=330, **kw):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=14, r=14, t=50, b=14),
                      font=dict(family="Archivo, sans-serif", color=INK, size=13), paper_bgcolor="#fff", plot_bgcolor="#fff",
                      title=dict(font=dict(family="Archivo, sans-serif", size=16)), **kw)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# -----------------------------------------------------------------------------
# SIMULATION
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def run_sim(alpha=1.0, epsilon=0.1, lambda_=1.0, T=T_STEPS, seed=42, only_ucb=False):
    env = ContextualNewsEnv(seed=seed)
    contexts = [env.sample_context() for _ in range(T)]
    ucb = LinUCB(env.k, env.d, alpha=alpha, lambda_=lambda_)
    pols = {}
    if not only_ucb:
        pols["random"] = RandomPolicy(env.k, seed)
        pols["eps"] = ContextualEpsGreedy(env.k, env.d, epsilon, seed)
    pols["ucb"] = ucb
    res = {n: {"rewards": [], "regret": [], "arms": []} for n in pols}
    best = []
    for x in contexts:
        true_scores = env.theta @ x
        p_star = sigmoid(np.max(true_scores))
        best.append(int(np.argmax(true_scores)))
        for name, pol in pols.items():
            arm = pol.select(x)
            r, p_sel = env.click(arm, x)
            res[name]["rewards"].append(r)
            res[name]["regret"].append(p_star - p_sel)
            res[name]["arms"].append(int(arm))
            pol.update(arm, x, r)
    for n in res:
        for k in res[n]:
            res[n][k] = np.array(res[n][k])
    return res, ucb, env, np.array(best)


def ucb_parts(policy, x):
    est, bonus = np.empty(policy.k), np.empty(policy.k)
    for a in range(policy.k):
        theta_hat = np.linalg.solve(policy.A[a], policy.b[a])
        v = np.linalg.solve(policy.A[a], x)
        est[a] = theta_hat @ x
        bonus[a] = policy.alpha * np.sqrt(max(float(x @ v), 0.0))
    return est, bonus


@st.cache_data(show_spinner=False)
def ab_vs_bandit(T=10000, n_test=3000, R=300, seed=1):
    """Toy two-article world (4% vs 6% CTR): fixed 50/50 A/B test then commit vs Thompson sampling."""
    rng = np.random.default_rng(seed)
    p = np.array([0.04, 0.06]); gap = p.max() - p.min()
    half = n_test // 2
    cA, cB = rng.binomial(half, p[0], R), rng.binomial(half, p[1], R)
    tie = rng.random(R) < 0.5
    wrong = ((cA > cB) | ((cA == cB) & tie)).mean()
    t = np.arange(1, T + 1)
    ab = np.where(t <= n_test, 0.5 * gap * t, 0.5 * gap * n_test + (t - n_test) * gap * wrong)
    a, b = np.ones((R, 2)), np.ones((R, 2))
    idx, reg = np.arange(R), np.zeros(T)
    for i in range(T):
        arm = rng.beta(a, b).argmax(1)
        r = (rng.random(R) < p[arm]).astype(float)
        a[idx, arm] += r; b[idx, arm] += 1 - r
        reg[i] = (p.max() - p[arm]).mean()
    return t, ab, np.cumsum(reg)


# experiment parameters live in session state so the lab form can change them
st.session_state.setdefault("alpha", 1.0)
st.session_state.setdefault("eps", 0.1)
with st.spinner("Running 5,000 simulated sessions..."):
    results, ucb, env, best_arms = run_sim(st.session_state.alpha, st.session_state.eps)

T = len(results["ucb"]["rewards"])
tot_reg = {k: float(results[k]["regret"].sum()) for k in results}
tot_rew = {k: float(results[k]["rewards"].sum()) for k in results}
reg_ratio = tot_reg["random"] / max(tot_reg["ucb"], 1e-9)
ctr_lift = (tot_rew["ucb"] / max(tot_rew["random"], 1e-9) - 1) * 100
best_rate = float(np.mean(results["ucb"]["arms"][-1000:] == best_arms[-1000:]))

# =============================================================================
# HERO
# =============================================================================
html(f"""
<div class='hero'>
  <div class='hero-img' style="background-image:url('{IMG_HERO}')"></div>
  <div class='hero-tint'></div>
  <div class='hero-body'>
    <h1>The algorithm that knows what you want to read before you do.</h1>
    <p>A contextual-bandit simulation of a personalised news homepage: LinUCB against ε-greedy and random, over {T:,} simulated sessions and no personal data.</p>
  </div>
</div>
<p class='credit'>Gamla Stan, Stockholm. Photo: Unsplash</p>""")

c1, c2, c3 = st.columns(3)
c1.markdown(stat("Less regret than random", f"{reg_ratio:.0f}×", f"{tot_reg['ucb']:,.0f} lost clicks vs {tot_reg['random']:,.0f}"), unsafe_allow_html=True)
c2.markdown(stat("Click-through lift", f"+{ctr_lift:.0f}%", f"LinUCB CTR {tot_rew['ucb']/T:.1%} vs random {tot_rew['random']/T:.1%}", "gold"), unsafe_allow_html=True)
c3.markdown(stat("Best article picked", f"{best_rate:.0%}", "Share of the last 1,000 sessions where LinUCB matched the oracle"), unsafe_allow_html=True)

# =============================================================================
# THE STORY
# =============================================================================
st.write("")
body("It is 07:30 on a grey Tuesday in November. You are on the frost-covered platform at Slussen, waiting for the train, "
     "and mist is rolling off Riddarfjärden. You pull out your phone to catch up on the news.", "drop")
body("For a week you have followed a story about electricity prices in Södermalm. You are also apartment hunting, and you happen to love "
     "reading about generative models. You open the homepage and see a football derby, then celebrity gossip, then a column on the EU parliament. "
     "You scroll past three articles you do not care about before you find one that matters.")
body("Call it the <b>Stockholm Morning Problem</b>: every reader gets the same front page, whatever their habits. "
     "Streaming services curate every tile for the individual, so a static homepage starts to feel clunky by comparison. "
     "At the scale of a large news site, even a small relative lift in click-through rate is worth real money.")
photo(IMG_NEWS, "A static front page treats every reader the same. Photo: Unsplash")
html("<div class='pull'>Personalisation without tracking you across the internet is a decision problem, not a deep-learning problem.</div>")
body("The question is how to personalise for millions of readers using only what they do in the current session, with no names, "
     "no emails and no browsing history. The answer is a problem formalised decades before the web existed: the <b>multi-armed bandit</b>.")

# --- A/B ---
st.markdown("## Why A/B testing is the wrong tool here")
body("Suppose you want to know whether a Tech piece beats a Politics piece. The classic answer is a 50/50 split for a couple of weeks, "
     "then pick the winner. Two things go wrong in a newsroom.")
body("<b>The news moves faster than the test.</b> By the time the result is significant, the story is stale. "
     "<b>Exploration has a price.</b> Every user who saw the weaker article during the test was a click you chose to lose. That lost revenue is called <em>regret</em>.")
body("A bandit learns while it serves traffic. It shifts exposure towards the better article as evidence builds and keeps a small amount of exploration in case things change. "
     "Here is a toy two-article world to show the difference (4% vs 6% click rate, my own setup, not the news simulation below).")
t, ab, ts = ab_vs_bandit()
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=ab, name="A/B test, then commit", line=dict(color=GREY, width=3)))
fig.add_trace(go.Scatter(x=t, y=ts, name="Bandit (Thompson sampling)", line=dict(color=BLUE, width=3)))
fig.add_vline(x=3000, line_dash="dot", line_color=INK, annotation_text="A/B test ends", annotation_position="top left")
fig.update_xaxes(title="Users served"); fig.update_yaxes(title="Cumulative expected lost clicks")
style(fig, 340, title="The price of learning", legend=dict(orientation="h", y=1.14, x=0))
show(fig)
callout("The A/B curve climbs steeply while the test runs, then flattens if the test picks the right article. "
        "The bandit pays less to learn because it stops showing the loser as soon as the evidence points one way.")

# --- DATA ---
st.markdown("## The data: privacy-first context")
body("Under GDPR, personalisation has to work from behavioural signals in the current session. On each visit the system observes a six-number context vector "
     "and nothing else: no name, no email, no history.")
st.dataframe(pd.DataFrame({
    "Signal": ["Political interest", "Sports interest", "Tech interest", "Mobile usage", "Morning reader", "Age (standardised)"],
    "What it captures": ["Time on current-affairs pages", "Interest in sports highlights", "Affinity for gadgets and AI",
                         "Preference for short, mobile-friendly reads", "Reads before 10 AM", "0 = 40 years, +1 = 55 years"],
    "Example": ["1.6 (high)", "0.3 (low)", "0.2 (low)", "0.5 (medium)", "1.7 (high)", "0.8 (older)"],
}), use_container_width=True, hide_index=True)

# --- METHOD ---
st.markdown("## The method: LinUCB")
photo(IMG_BAR, "A good bartender reads the room, and sometimes offers a small taste of something new. Photo: Unsplash", 240)
body("Think of a bartender in Gamla Stan. When you walk in he reads the context: suit or hoodie, early or late. "
     "He serves what he is fairly sure you will like, and now and then offers something new to learn your taste. "
     "That is the trade-off between <b>exploitation</b> (the best-known choice) and <b>exploration</b> (learning where he is unsure).")
html("""
<div class='loop'>
  <div class='step'><b>Observe</b><span>A reader arrives with a context vector x.</span></div><div class='arrow'>→</div>
  <div class='step'><b>Score</b><span>For each article: predicted click rate plus an uncertainty bonus.</span></div><div class='arrow'>→</div>
  <div class='step'><b>Show and learn</b><span>Pick the top score, watch for a click, update that article's model.</span></div>
</div>""")

# --- IMPACT ---
st.markdown(f"## The impact: {reg_ratio:.0f}× less regret than random")
body(f"I simulated {T:,} sessions and compared LinUCB with two baselines: random articles, and ε-greedy (explores at random {st.session_state.eps:.0%} of the time). "
     "<b>Regret</b> is the clicks lost against an all-knowing oracle that always shows the best article for each reader.")

steps = np.arange(1, T + 1)
COL = {"random": GREY, "eps": GOLD, "ucb": BLUE}
NAME = {"random": "Random", "eps": "ε-Greedy", "ucb": "LinUCB"}
a_col, b_col = st.columns(2)
f1, f2 = go.Figure(), go.Figure()
for k in ["random", "eps", "ucb"]:
    w = 4 if k == "ucb" else 2.5
    f1.add_trace(go.Scatter(x=steps, y=np.cumsum(results[k]["rewards"]) / steps, name=NAME[k], line=dict(color=COL[k], width=w)))
    f2.add_trace(go.Scatter(x=steps, y=np.cumsum(results[k]["regret"]), name=NAME[k], line=dict(color=COL[k], width=w)))
f1.update_xaxes(title="Sessions"); f1.update_yaxes(title="Average CTR", tickformat=".0%")
f2.update_xaxes(title="Sessions"); f2.update_yaxes(title="Lost clicks")
style(f1, 320, title="Average click-through rate", legend=dict(orientation="h", y=1.18, x=0))
style(f2, 320, title="Cumulative regret", legend=dict(orientation="h", y=1.18, x=0))
with a_col: show(f1)
with b_col: show(f2)

win = 200
f3 = go.Figure()
for k in ["random", "eps", "ucb"]:
    hit = (results[k]["arms"] == best_arms).astype(float)
    roll = np.convolve(hit, np.ones(win) / win, mode="valid")
    f3.add_trace(go.Scatter(x=np.arange(win, T + 1), y=roll, name=NAME[k], line=dict(color=COL[k], width=4 if k == "ucb" else 2.5)))
f3.update_xaxes(title="Sessions"); f3.update_yaxes(title="Chose the best article", tickformat=".0%", range=[0, 1.02])
style(f3, 320, title="How often each policy picks the oracle's article (200-session rolling average)", legend=dict(orientation="h", y=1.18, x=0))
show(f3)
body("LinUCB's uncertainty bonus is large for articles it knows little about, so early traffic goes to learning. As each article's model sharpens the bonus shrinks, "
     "and the policy settles on what works. Even sessions that did not end in a click taught it something.")

# arm shares over time
bins = np.array_split(np.arange(T), 10)
f4 = go.Figure()
pal = [BLUE, GOLD, "#5FA8D3", GREY]
for a_i, nm in enumerate(ARTICLES):
    share = [float(np.mean(results["ucb"]["arms"][b] == a_i)) for b in bins]
    f4.add_trace(go.Bar(x=[f"{b[0]+1}-{b[-1]+1}" for b in bins], y=share, name=nm, marker_color=pal[a_i]))
f4.update_layout(barmode="stack"); f4.update_yaxes(tickformat=".0%", title="Share of sessions"); f4.update_xaxes(title="Sessions")
style(f4, 320, title="What LinUCB chose to show over time", legend=dict(orientation="h", y=1.18, x=0))
show(f4)

# --- LAB ---
st.markdown("### Change the experiment")
with st.form("lab"):
    l1, l2 = st.columns(2)
    new_alpha = l1.slider("LinUCB exploration strength α", 0.0, 3.0, float(st.session_state.alpha), 0.1)
    new_eps = l2.slider("ε-Greedy exploration rate ε", 0.0, 0.5, float(st.session_state.eps), 0.01)
    if st.form_submit_button("Re-run simulation"):
        st.session_state.alpha, st.session_state.eps = new_alpha, new_eps
        st.rerun()

with st.expander("Robustness checks: α sweep and multiple seeds"):
    body("A single run can flatter an algorithm. These checks rerun the simulation across exploration strengths and random seeds.")
    r1, r2 = st.columns(2)
    if r1.button("Run α sweep"):
        grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
        with st.spinner("Sweeping α..."):
            st.session_state.sweep = (grid, [float(run_sim(alpha=g, epsilon=st.session_state.eps, only_ucb=True)[0]["ucb"]["regret"].sum()) for g in grid])
    if r2.button("Run 8 seeds"):
        with st.spinner("Running 8 seeds..."):
            st.session_state.seeds = [run_sim(alpha=st.session_state.alpha, epsilon=st.session_state.eps, seed=s)[0] for s in range(8)]
    if "sweep" in st.session_state:
        grid, regs = st.session_state.sweep
        fs = go.Figure(go.Scatter(x=grid, y=regs, mode="lines+markers", line=dict(color=BLUE, width=3), marker=dict(size=9)))
        fs.update_xaxes(title="α"); fs.update_yaxes(title="Cumulative regret")
        style(fs, 300, title="Regret vs exploration strength", showlegend=False)
        show(fs)
        body(f"Too little exploration locks in early mistakes and too much wastes clicks. In this run regret was lowest at α = {grid[int(np.argmin(regs))]}.")
    if "seeds" in st.session_state:
        runs = st.session_state.seeds
        mat = {k: np.array([r[k]["regret"].sum() for r in runs]) for k in ["random", "eps", "ucb"]}
        fm = go.Figure(go.Bar(x=[NAME[k] for k in mat], y=[mat[k].mean() for k in mat],
                              error_y=dict(type="data", array=[mat[k].std() for k in mat]), marker_color=[COL[k] for k in mat]))
        fm.update_yaxes(title="Cumulative regret (mean ± std)")
        style(fm, 300, title="Regret across 8 seeds", showlegend=False)
        show(fm)
        wins = int(np.sum((mat["ucb"] < mat["eps"]) & (mat["ucb"] < mat["random"])))
        body(f"LinUCB had the lowest regret in {wins} of {len(runs)} seeds.")

# --- LIVE DEMO ---
st.markdown("## Try it: the live recommendation engine")
body("Pick a persona or set the sliders yourself. The chart shows what the trained model believes about each article, how much it is still exploring, and the true click probability the simulator uses.")

KEYS = [f"ctx_{f}" for f in FEATURES]
defaults = PERSONAS["Commuter who follows politics"]
for k, v in zip(KEYS, defaults):
    st.session_state.setdefault(k, v)


def apply_persona():
    vec = PERSONAS[st.session_state.persona]
    if vec:
        for k, v in zip(KEYS, vec):
            st.session_state[k] = v


st.selectbox("Start from a persona", list(PERSONAS), index=1, key="persona", on_change=apply_persona)
colU, colA = st.columns([1, 1.25])
with colU:
    vals = []
    for k, lab, (lo_, hi_) in zip(KEYS, LABELS, RANGES):
        vals.append(st.slider(lab, lo_, hi_, step=0.1, key=k))
x_vec = np.array(vals)
est, bonus = ucb_parts(ucb, x_vec)
true_p = sigmoid(env.theta @ x_vec)
scores = est + bonus
pick, oracle = int(np.argmax(scores)), int(np.argmax(true_p))
with colA:
    ok = pick == oracle
    html(f"""<div class='verdict'><div class='k'>LinUCB shows</div><div class='v'>{ARTICLES[pick]}</div>
    <div class='k'>Oracle's best article: {ARTICLES[oracle]} &nbsp; <span class='{"ok" if ok else "no"}'>{"match" if ok else "different"}</span></div></div>""")
    fd = go.Figure()
    fd.add_bar(x=ARTICLES, y=est, name="Estimated click rate", marker_color=[BLUE if i == pick else "#9CC3DD" for i in range(4)])
    fd.add_bar(x=ARTICLES, y=bonus, name="Exploration bonus", marker_color=GOLD)
    fd.add_trace(go.Scatter(x=ARTICLES, y=true_p, mode="markers", name="True click probability",
                            marker=dict(symbol="diamond", size=13, color=INK, line=dict(color="#fff", width=1.5))))
    fd.update_layout(barmode="stack"); fd.update_yaxes(title="Score")
    style(fd, 340, title="Why this article?", legend=dict(orientation="h", y=-0.2, x=0))
    show(fd)
callout("The linear model estimates click rate directly, while the simulator generates clicks through a logistic function, so the estimate and the true probability will not match exactly. "
        "The persona values are my own illustrations. The model decides what to show.", "gold")

# --- APPLICATIONS ---
st.markdown("## Beyond news")
body("Anywhere a business must choose with incomplete information and then learns from the outcome, a bandit is a candidate.")
a1, a2, a3 = st.columns(3)
a1.markdown("<div class='card'><b>E-commerce</b><p>Choose between a discount, free shipping or no offer for each visitor, and learn from whether they buy.</p></div>", unsafe_allow_html=True)
a2.markdown("<div class='card'><b>Adaptive trials</b><p>Related ideas shift patients towards treatments that appear to work, so fewer receive a poor one.</p></div>", unsafe_allow_html=True)
a3.markdown("<div class='card'><b>Advertising</b><p>Pick which ad to show within milliseconds, using device and time-of-day context.</p></div>", unsafe_allow_html=True)
html("<div class='pull'>A contextual bandit sits between the rigour of A/B testing and the speed of online machine learning.</div>")

# --- LIMITS ---
st.markdown("## Limits and next steps")
callout("<b>This is a simulation.</b> Clicks come from a logistic model I control, which makes it a clean test of the algorithm but not a claim about real readers. "
        "A real system would face delayed feedback, drifting tastes, and clicks that do not equal satisfaction.", "gold")
st.markdown("""
- **Evaluate offline first.** Use logged data with inverse-propensity or doubly robust estimators before any live traffic.
- **Handle drift.** Add discounting or sliding windows, because yesterday's news is not today's.
- **Compare policies.** Try Thompson sampling and neural contextual bandits against LinUCB.
- **Protect diversity.** Add constraints so personalisation does not shrink readers' view of the news.
""")

# --- MATH ---
st.markdown("## Under the hood")
with st.expander("The mathematics of LinUCB"):
    body("For each article a, LinUCB keeps a ridge-regression model and scores it on the current context xₜ:")
    st.latex(r"\hat\theta_a = A_a^{-1} b_a, \qquad \text{score}_a = \hat\theta_a^{\top} x_t + \alpha \sqrt{x_t^{\top} A_a^{-1} x_t}")
    st.latex(r"A_a \leftarrow A_a + x_t x_t^{\top}, \qquad b_a \leftarrow b_a + r_t\, x_t, \qquad A_a^{(0)} = \lambda I")
    st.latex(r"\text{Regret}_T = \sum_{t=1}^{T}\Big(p^{*}_t - p_t(\text{shown})\Big)")
    body("The first term of the score is exploitation, the second is exploration. α scales how much uncertainty is rewarded, and λ is the ridge penalty.")
with st.expander("Core idea in code"):
    st.code('''def select(self, x):
    scores = []
    for a in range(self.k):
        theta = np.linalg.solve(self.A[a], self.b[a])
        bonus = self.alpha * np.sqrt(x @ np.linalg.solve(self.A[a], x))
        scores.append(theta @ x + bonus)
    return int(np.argmax(scores))

def update(self, arm, x, r):
    self.A[arm] += np.outer(x, x)
    self.b[arm] += r * x''', language="python")

st.markdown("---")
html("".join(f"<span class='tag'>{t}</span>" for t in ["Contextual bandits", "LinUCB", "Regret analysis", "Ridge regression", "Python", "NumPy", "Plotly", "Streamlit"]))
html(f"<p class='ui' style='font-size:.95rem;margin-top:1rem'>Written by <b>{AUTHOR}</b>, {TAGLINE}. "
     f"<a href='{GITHUB}' target='_blank'>GitHub</a> &nbsp;|&nbsp; <a href='{LINKEDIN}' target='_blank'>LinkedIn</a></p>")
