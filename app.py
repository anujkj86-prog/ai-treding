import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="AI TREDING", page_icon="📈", layout="wide")

# ----------------------------- CONFIG ---------------------------------
NSE_DEFAULT = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
               "SBIN.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS", "TATAMOTORS.NS"]
US_DEFAULT = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "KO"]

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 12px; border-radius: 10px;}
.small-note {font-size: .82rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "AAPL", "TCS.NS"]
if "paper_cash" not in st.session_state:
    st.session_state.paper_cash = 100000.0
if "paper_positions" not in st.session_state:
    st.session_state.paper_positions = {}
if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ----------------------------- DATA -----------------------------------
def normalize_yf(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    out = df[needed].copy()
    out = out.dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    return out

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol, period="6mo", interval="1d"):
    try:
        import yfinance as yf
        raw = yf.download(symbol, period=period, interval=interval,
                          auto_adjust=True, progress=False, threads=False)
        return normalize_yf(raw)
    except Exception:
        return pd.DataFrame()

def sample_history(symbol, rows=180):
    # Demonstration-only synthetic data; never represents a real market quote.
    seed = sum(ord(c) for c in symbol) % 10000
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    start = 150 + (seed % 250)
    returns = rng.normal(0.0004, 0.018, rows)
    close = start * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, .004, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, .012, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, .012, rows))
    volume = rng.integers(100000, 2000000, rows)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=dates)

def get_data(symbol, uploaded=None):
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            cols = {c.lower().strip(): c for c in df.columns}
            date_col = next((cols[k] for k in ["date", "datetime", "timestamp"] if k in cols), None)
            close_col = next((cols[k] for k in ["close", "adj close", "adj_close"] if k in cols), None)
            if date_col and close_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.set_index(date_col).sort_index()
                for c in ["Open", "High", "Low", "Close", "Volume"]:
                    match = next((v for k, v in cols.items() if k == c.lower()), None)
                    if match:
                        df[c] = pd.to_numeric(df[match], errors="coerce")
                df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
                for c in ["Open", "High", "Low"]:
                    if c not in df: df[c] = df["Close"]
                if "Volume" not in df: df["Volume"] = 0
                return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]), "Uploaded CSV"
        except Exception:
            pass
    live = fetch_history(symbol)
    if not live.empty:
        return live, "Yahoo Finance via yfinance (may be delayed; verify independently)"
    return sample_history(symbol), "Synthetic demo data — NOT real market data"

# ----------------------------- INDICATORS -----------------------------
def add_indicators(df):
    d = df.copy()
    close = d["Close"].astype(float)
    d["EMA20"] = close.ewm(span=20, adjust=False).mean()
    d["EMA50"] = close.ewm(span=50, adjust=False).mean()
    d["EMA200"] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]
    d["VolAvg20"] = d["Volume"].rolling(20).mean()
    return d

def research_signal(row):
    score = 0
    if pd.notna(row.get("RSI")):
        if row["RSI"] > 55: score += 1
        elif row["RSI"] < 45: score -= 1
    if row["Close"] > row["EMA20"] > row["EMA50"]: score += 1
    elif row["Close"] < row["EMA20"] < row["EMA50"]: score -= 1
    if row["MACD"] > row["MACD_Signal"]: score += 1
    elif row["MACD"] < row["MACD_Signal"]: score -= 1
    if score >= 2: return "BUY", score
    if score <= -2: return "SELL", score
    return "HOLD", score

def fmt_price(value, market):
    return f"{value:,.2f}" if pd.notna(value) else "—"

# ----------------------------- SIDEBAR --------------------------------
st.sidebar.title("📈 AI TREDING")
st.sidebar.caption("NSE + US market research")
page = st.sidebar.radio("Navigation", [
    "Dashboard", "Stock Scanner", "Technical Analysis", "Watchlist & Alerts",
    "Paper Trading", "Backtesting", "CSV Lab", "News & AI Research", "How to use"
])
market = st.sidebar.selectbox("Market universe", ["NSE", "US", "Both"])
default_symbols = NSE_DEFAULT if market == "NSE" else US_DEFAULT if market == "US" else NSE_DEFAULT[:5] + US_DEFAULT[:5]
st.sidebar.caption("Data can be delayed, limited or unavailable. Demo fallback is synthetic.")

# ----------------------------- PAGES ----------------------------------
st.title("📈 AI TREDING")
st.caption("Professional-style research workspace • NSE + US equities")
st.info("Research and education only. Signals are rule-based examples, not financial advice or guaranteed predictions.")

if page == "Dashboard":
    st.subheader("Market research dashboard")
    st.write("Choose a symbol to inspect its latest available history and technical setup.")
    symbol = st.selectbox("Stock symbol", default_symbols, index=0)
    df, source = get_data(symbol)
    d = add_indicators(df)
    latest = d.iloc[-1]
    signal, score = research_signal(latest)
    change = (d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100 if len(d) > 1 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", fmt_price(latest["Close"], market), f"{change:+.2f}%")
    c2.metric("RSI (14)", f"{latest['RSI']:.1f}" if pd.notna(latest["RSI"]) else "—")
    c3.metric("Research signal", signal, f"Score {score:+d}")
    c4.metric("Volume", f"{latest['Volume']:,.0f}")
    st.caption(f"Data source: {source} • Last bar: {d.index[-1]}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"))
    for col, color in [("EMA20", "#f59e0b"), ("EMA50", "#3b82f6"), ("EMA200", "#a855f7")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(width=1.4, color=color)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Candles and indicators are based on the selected data source. Synthetic fallback is for interface testing only.")

elif page == "Stock Scanner":
    st.subheader("Stock scanner")
    st.write("Scan a small watch universe first. Larger scans may hit free-source limits.")
    symbols_text = st.text_area("Symbols (comma separated)", ", ".join(default_symbols), height=90)
    min_rsi = st.slider("Minimum RSI", 0, 100, 50)
    max_rsi = st.slider("Maximum RSI", 0, 100, 75)
    min_volume_ratio = st.slider("Minimum volume / 20-day average", 0.0, 5.0, 1.0, 0.1)
    if st.button("Run scanner", type="primary"):
        rows = []
        for sym in [s.strip().upper() for s in symbols_text.split(",") if s.strip()][:30]:
            hist, source = get_data(sym)
            if hist.empty: continue
            x = add_indicators(hist).iloc[-1]
            sig, score = research_signal(x)
            vol_ratio = x["Volume"] / x["VolAvg20"] if pd.notna(x["VolAvg20"]) and x["VolAvg20"] else np.nan
            if pd.notna(x["RSI"]) and min_rsi <= x["RSI"] <= max_rsi and (pd.isna(vol_ratio) or vol_ratio >= min_volume_ratio):
                rows.append({"Symbol": sym, "Close": round(x["Close"],2), "RSI": round(x["RSI"],1),
                             "MACD": round(x["MACD"],3), "EMA20": round(x["EMA20"],2),
                             "EMA50": round(x["EMA50"],2), "Vol ratio": round(vol_ratio,2) if pd.notna(vol_ratio) else None,
                             "Signal": sig, "Score": score, "Data": source})
        if rows:
            result = pd.DataFrame(rows).sort_values(["Score","RSI"], ascending=[False,False])
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("Download scanner results", result.to_csv(index=False), "ai_treding_scan.csv", "text/csv")
        else:
            st.warning("No matches or no usable data. Try widening the filters or upload CSV data in CSV Lab.")

elif page == "Technical Analysis":
    st.subheader("Technical analysis")
    symbol = st.selectbox("Select symbol", default_symbols, key="ta_symbol")
    hist, source = get_data(symbol)
    d = add_indicators(hist)
    st.caption(f"Data: {source}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"))
    for col in ["EMA20","EMA50","EMA200"]:
        fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col))
    fig.update_layout(height=480, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        rsi_fig = go.Figure(go.Scatter(x=d.index,y=d.RSI,name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dash"); rsi_fig.add_hline(y=30, line_dash="dash")
        rsi_fig.update_layout(title="RSI (14)", height=260, yaxis_range=[0,100])
        st.plotly_chart(rsi_fig, use_container_width=True)
    with c2:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=d.index,y=d.MACD,name="MACD"))
        macd_fig.add_trace(go.Scatter(x=d.index,y=d.MACD_Signal,name="Signal"))
        macd_fig.update_layout(title="MACD (12, 26, 9)", height=260)
        st.plotly_chart(macd_fig, use_container_width=True)
    st.dataframe(d.tail(15).round(3), use_container_width=True)

elif page == "Watchlist & Alerts":
    st.subheader("Watchlist and price alerts")
    add_symbol = st.text_input("Add symbol (examples: RELIANCE.NS, AAPL)", "")
    if st.button("Add to watchlist") and add_symbol.strip():
        if add_symbol.strip().upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_symbol.strip().upper())
    st.write("Your watchlist:", ", ".join(st.session_state.watchlist) if st.session_state.watchlist else "Empty")
    if st.session_state.watchlist:
        records = []
        for sym in st.session_state.watchlist:
            hist, src = get_data(sym)
            if not hist.empty:
                d = add_indicators(hist); row = d.iloc[-1]; sig, score = research_signal(row)
                records.append({"Symbol":sym,"Close":round(row.Close,2),"RSI":round(row.RSI,1),
                                "Signal":sig,"Score":score,"Data":src})
        if records: st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    st.markdown("**Create a price threshold alert**")
    ac1, ac2 = st.columns(2)
    alert_symbol = ac1.text_input("Alert symbol", "RELIANCE.NS")
    alert_price = ac2.number_input("Alert price", min_value=0.01, value=1000.0)
    direction = st.selectbox("Alert when price is", ["Above or equal to", "Below or equal to"])
    if st.button("Save alert"):
        st.session_state.alerts.append({"Symbol":alert_symbol.upper(),"Price":alert_price,"Condition":direction})
    if st.session_state.alerts: st.dataframe(pd.DataFrame(st.session_state.alerts), use_container_width=True, hide_index=True)
    st.caption("Alerts are stored only in this browser session in this starter version. They do not run in the background or send notifications.")

elif page == "Paper Trading":
    st.subheader("Paper trading simulator")
    st.caption("Virtual trading only. Positions reset when the app session is reset; no broker connection.")
    st.metric("Virtual cash", f"{st.session_state.paper_cash:,.2f}")
    with st.form("paper_order"):
        sym = st.text_input("Symbol", "RELIANCE.NS").upper().strip()
        side = st.selectbox("Order", ["BUY","SELL"])
        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        price = st.number_input("Execution price (enter verified price)", min_value=0.01, value=100.0)
        submitted = st.form_submit_button("Place virtual order")
    if submitted:
        positions = st.session_state.paper_positions
        current_qty = positions.get(sym, 0)
        if side == "BUY":
            cost = qty * price
            if cost <= st.session_state.paper_cash:
                st.session_state.paper_cash -= cost
                positions[sym] = current_qty + qty
                st.session_state.paper_trades.append({"Time":datetime.now().isoformat(timespec="seconds"),"Symbol":sym,"Side":side,"Qty":qty,"Price":price})
                st.success("Virtual buy recorded.")
            else: st.error("Insufficient virtual cash.")
        else:
            if qty <= current_qty:
                st.session_state.paper_cash += qty * price
                positions[sym] = current_qty - qty
                st.session_state.paper_trades.append({"Time":datetime.now().isoformat(timespec="seconds"),"Symbol":sym,"Side":side,"Qty":qty,"Price":price})
                st.success("Virtual sell recorded.")
            else: st.error("You cannot sell more shares than your virtual position.")
    st.markdown("**Open positions**")
    st.dataframe(pd.DataFrame([{"Symbol":k,"Quantity":v} for k,v in st.session_state.paper_positions.items() if v]), use_container_width=True)
    st.markdown("**Trade history**")
    st.dataframe(pd.DataFrame(st.session_state.paper_trades), use_container_width=True)

elif page == "Backtesting":
    st.subheader("Simple EMA crossover backtest")
    symbol = st.selectbox("Symbol", default_symbols, key="bt_symbol")
    fast = st.slider("Fast EMA", 5, 50, 20)
    slow = st.slider("Slow EMA", 20, 200, 50)
    hist, source = get_data(symbol)
    if fast >= slow: st.warning("Fast EMA must be smaller than slow EMA.")
    else:
        d = hist.copy()
        d["Fast"] = d.Close.ewm(span=fast, adjust=False).mean()
        d["Slow"] = d.Close.ewm(span=slow, adjust=False).mean()
        d["Position"] = (d.Fast > d.Slow).astype(int)
        d["Market_Return"] = d.Close.pct_change().fillna(0)
        d["Strategy_Return"] = d.Position.shift(1).fillna(0) * d.Market_Return
        d["Equity"] = (1 + d.Strategy_Return).cumprod()
        d["BuyHold"] = (1 + d.Market_Return).cumprod()
        total = (d.Equity.iloc[-1] - 1) * 100
        bh = (d.BuyHold.iloc[-1] - 1) * 100
        st.caption(f"Data: {source}. Educational backtest; excludes brokerage, taxes, slippage and liquidity effects.")
        m1,m2,m3 = st.columns(3)
        m1.metric("Strategy return", f"{total:.2f}%")
        m2.metric("Buy & hold", f"{bh:.2f}%")
        m3.metric("Number of position changes", int(d.Position.diff().abs().fillna(0).sum()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index,y=d.Equity*100,name="EMA strategy"))
        fig.add_trace(go.Scatter(x=d.index,y=d.BuyHold*100,name="Buy & hold"))
        fig.update_layout(title="Growth of ₹/$100 starting value", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(d.tail(20).round(4), use_container_width=True)

elif page == "CSV Lab":
    st.subheader("Upload your own historical stock data")
    st.write("CSV should contain Date and Close columns. Open, High, Low and Volume are recommended.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        hist, source = get_data("CSV_UPLOAD", file)
        if hist.empty:
            st.error("Could not read this CSV. Check that it has Date and Close columns.")
        else:
            d = add_indicators(hist)
            st.success(f"Loaded {len(d)} rows.")
            st.dataframe(d.tail(25).round(3), use_container_width=True)
            st.download_button("Download analysed CSV", d.to_csv().encode("utf-8"), "ai_treding_analysed.csv", "text/csv")
            st.line_chart(d[["Close","EMA20","EMA50","EMA200"]])
    else:
        st.info("Upload a CSV file to calculate indicators and inspect the data.")

elif page == "News & AI Research":
    st.subheader("News and AI research assistant")
    st.warning("This first version does not use a paid AI model or claim to understand breaking news. The assistant below is a transparent rule-based explainer.")
    q = st.text_input("Enter a ticker to research", "AAPL").upper().strip()
    if st.button("Generate research note"):
        hist, source = get_data(q)
        if hist.empty: st.error("No data available.")
        else:
            d = add_indicators(hist); row = d.iloc[-1]; sig, score = research_signal(row)
            st.markdown(f"### Research note: {q}")
            st.write(f"**Data source:** {source}")
            st.write(f"**Rule-based signal:** {sig} (score {score:+d}/3)")
            st.write(f"**Trend:** Close is {'above' if row.Close > row.EMA50 else 'below'} EMA50.")
            st.write(f"**RSI:** {row.RSI:.1f}. RSI above 70 is commonly considered overbought; below 30 oversold. These are not standalone trade rules.")
            st.write(f"**MACD:** {'above' if row.MACD > row.MACD_Signal else 'below'} its signal line.")
            st.write("**What to verify:** company results, valuation, sector trend, major announcements, liquidity, event risk, and your maximum acceptable loss.")
    st.markdown("**News integration:** Add a verified news API later. Do not rely on scraped headlines or unverified social posts for trading decisions.")

elif page == "How to use":
    st.subheader("Project guide")
    st.markdown("""
    1. Start with Dashboard and select a symbol.
    2. Use Stock Scanner to filter a small list of symbols.
    3. Open Technical Analysis to inspect price, EMA, RSI and MACD.
    4. Add symbols to Watchlist.
    5. Use Paper Trading to practise with virtual cash.
    6. Use Backtesting to test a basic EMA crossover rule.
    7. Upload your own CSV in CSV Lab.

    **Market symbol format**
    - NSE: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
    - US: `AAPL`, `MSFT`, `NVDA`

    **Known limitations in this starter**
    - Free data may be delayed, rate-limited or unavailable.
    - Synthetic fallback is clearly labelled and must never be used as a real quote.
    - Alerts do not run in the background.
    - Paper trades are simulated and session-based.
    - Backtesting excludes costs, taxes, slippage and survivorship bias.
    - The research assistant is rule-based, not a generative AI model.
    """)

st.divider()
st.caption("AI TREDING • Starter project • Verify all data independently • Not investment advice")
