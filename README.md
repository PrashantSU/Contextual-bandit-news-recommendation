# The Stockholm Personalization Problem

**🔗 Live Interactive Demo:** [https://contextual-bandit-news-recommendation-7blaqnuabh25udhvjkh4xe.streamlit.app/]

A Contextual Bandit (LinUCB) recommendation engine designed to solve the "NewsNet" morning problem: showing users the right news article based on their real-time behavior, without invasive tracking.

## The Problem
Traditional A/B testing is too slow for fast-moving news cycles. By the time a test reaches statistical significance, the news has changed. Furthermore, static A/B tests incur massive "regret" (lost clicks) by serving sub-optimal articles to half the audience.

## The Solution
We implemented **LinUCB (Linear Upper Confidence Bound)**, which balances *exploitation* (showing what we know works) and *exploration* (testing new articles to learn user preferences).

- **Context Vector (d=6):** Political interest, Sports interest, Tech interest, Mobile usage, Morning reader score, Standardized age.
- **Arms (k=4):** Politics, Sports, Tech, Lifestyle.

## Results (5,000 Simulated Sessions)
| Algorithm | Avg CTR | Cumulative Regret (Lost Clicks) |
| :--- | :--- | :--- |
| Random Baseline | 78.9% | 866 |
| ε-Greedy (ε=0.1) | 93.7% | 187 |
| **LinUCB (α=1.0)** | **95.8%** | **36** |

*LinUCB achieved 24x less regret than random guessing by exploring intelligently based on feature uncertainty.*

## Architecture
- **Algorithm:** Per-arm ridge regression with UCB exploration bonus.
- **Frontend:** Streamlit (custom CSS for a magazine-style editorial layout).
- **Visualization:** Plotly (interactive charts).
- **Deployment:** Streamlit Community Cloud.

## How to Run Locally
```bash
git clone https://github.com/PrashantSU/Contextual-bandit-news-recommendation.git
cd Contextual-bandit-news-recommendation
pip install -r requirements.txt
streamlit run streamlit_app.py
