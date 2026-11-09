# AI TREDING — NSE + US stock research starter

A free-first Streamlit research dashboard. Features include:
- NSE and US symbol lists
- Historical price retrieval via `yfinance` (may be delayed, limited, or unavailable)
- EMA 20/50/200, RSI, MACD
- Rule-based research score (BUY/HOLD/SELL)
- Scanner, watchlist, session-only price alerts
- Session-only paper trading
- Simple EMA crossover backtest
- CSV upload and analysis
- Rule-based research note generator

## Important limitations
This is an educational starter, not a broker or trading system. It does not execute real orders.
The free data source is unofficial and may be delayed, rate-limited, incomplete, or unavailable.
When the source fails, the app displays synthetic demonstration data and labels it clearly.
Do not use synthetic data for trading decisions.
The alert list is session-only and does not send background notifications.
Paper trades are session-only. Backtesting excludes brokerage, taxes, slippage, liquidity, corporate actions and survivorship bias.
The research assistant is rule-based; it is not a connected generative AI service.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repository named `ai-treding`.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Click **Create app**.
5. Select your repository, branch `main`, and file `app.py`.
6. Click **Deploy**.

## Android
Open the deployed URL in Chrome on Android. Use the browser menu > **Add to Home screen** (or **Install app**, if offered).
This Streamlit starter is mobile-friendly, but a true PWA with offline caching and native-style install metadata is a later upgrade.

## CSV format
Minimum columns: `Date`, `Close`.
Recommended: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
