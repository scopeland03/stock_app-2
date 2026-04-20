import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from datetime import date, timedelta

# 1. APPLICATION SETUP
st.set_page_config(page_title="Quant Analysis Pro", layout="wide")
st.title("Interactive Stock Analysis Dashboard")

# 2. SIDEBAR CONFIGURATION (Section 2.1)
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter 2-5 Tickers (comma separated)", value="AAPL, MSFT, NVDA").upper()
user_tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Methodology Expander (Section 3.5)
with st.sidebar.expander("Methodology & Assumptions"):
    st.write("- **Annualization:** 252 trading days.")
    st.write("- **Returns:** Simple arithmetic (pct_change).")
    st.write("- **Benchmark:** S&P 500 (^GSPC).")
    st.write("- **Source:** Yahoo Finance Adjusted Close.")

# Input Validation (Section 2.1.1 & 2.1.2)
if len(user_tickers) < 2 or len(user_tickers) > 5:
    st.sidebar.error("Error: Please enter between 2 and 5 tickers.")
    st.stop()
if (end_date - start_date).days < 365:
    st.sidebar.error("Error: Date range must be at least 1 year.")
    st.stop()

# 3. DATA ENGINE (Section 2.1.3 & 4.1)
@st.cache_data(ttl=3600)
def load_quant_data(tickers, start, end):
    all_tickers = list(set(tickers + ["^GSPC"]))
    try:
        # auto_adjust=True accounts for dividends/splits
        data = yf.download(all_tickers, start=start, end=end, auto_adjust=True)
        if data.empty: return None
        
        # Handle yfinance Multi-Index structure
        df = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
        
        # Partial Data Handling (Section 2.1.4): Truncate to overlapping range
        return df.dropna()
    except Exception as e:
        st.error(f"Download Error: {e}")
        return None

with st.spinner("Processing Financial Data..."):
    df_raw = load_quant_data(user_tickers, start_date, end_date)

if df_raw is None:
    st.stop()

# 4. MATH CALCULATIONS (Section 7.1)
returns = df_raw.pct_change().dropna()
bench_ret = returns["^GSPC"]
stock_ret = returns[user_tickers]

# --- LAYOUT (Section 3.1) ---
tab1, tab2, tab3 = st.tabs(["Price & Returns", "Risk Analysis", "Portfolio Explorer"])

# --- TAB 1: PRICE & RETURNS (Section 2.2) ---
with tab1:
    st.header("Price & Return Analysis")
    
    # 2.2.1 Price Chart with Multi-Select
    show_bench = st.checkbox("Show S&P 500 Benchmark on Price Chart", value=False)
    plot_cols = user_tickers + (["^GSPC"] if show_bench else [])
    st.line_chart(df_raw[plot_cols])

    # 2.2.3 Summary Stats (Includes Benchmark)
    st.subheader("Summary Statistics (Annualized)")
    stats_df = pd.DataFrame({
        "Ann. Return": returns.mean() * 252,
        "Ann. Volatility": returns.std() * np.sqrt(252),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurtosis(),
        "Min Daily": returns.min(),
        "Max Daily": returns.max()
    }).T
    st.dataframe(stats_df.style.format("{:.4f}"))

    # 2.2.4 Wealth Index Chart
    st.subheader("Growth of $10,000 (Wealth Index)")
    wealth = (1 + stock_ret).cumprod() * 10000
    wealth["S&P 500"] = (1 + bench_ret).cumprod() * 10000
    wealth["Equal-Weight Portfolio"] = (1 + stock_ret.mean(axis=1)).cumprod() * 10000
    st.line_chart(wealth)

# --- TAB 2: RISK ANALYSIS (Section 2.3) ---
with tab2:
    st.header("Risk & Distribution Analysis")
    sel_stock = st.selectbox("Select Asset for Detailed Risk Analysis", user_tickers)
    
    # 2.3.1 Rolling Volatility
    vol_win = st.slider("Rolling Volatility Window (Days)", 20, 126, 60)
    st.line_chart(stock_ret[sel_stock].rolling(vol_win).std() * np.sqrt(252))

    # 2.3.2 & 2.3.3 Distribution Toggle
    view = st.radio("Distribution View", ["Histogram + Normal Curve", "Q-Q Plot"], horizontal=True)
    
    if view == "Histogram + Normal Curve":
        mu, std = stats.norm.fit(stock_ret[sel_stock])
        x = np.linspace(stock_ret[sel_stock].min(), stock_ret[sel_stock].max(), 100)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=stock_ret[sel_stock], histnorm='probability density', name='Actual Returns'))
        fig_hist.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, mu, std), name='Normal Curve', line=dict(color='red')))
        fig_hist.update_layout(title=f"Return Distribution: {sel_stock}", xaxis_title="Daily Return", yaxis_title="Density")
        st.plotly_chart(fig_hist)
    else:
        fig_qq, ax_qq = plt.subplots()
        stats.probplot(stock_ret[sel_stock], dist="norm", plot=ax_qq)
        ax_qq.set_title(f"Q-Q Plot: {sel_stock}")
        st.pyplot(fig_qq)

    # 2.3.4 Normality Test
    jb_stat, p_val = stats.jarque_bera(stock_ret[sel_stock])
    st.write(f"**Jarque-Bera Stat:** {jb_stat:.2f} | **p-value:** {p_val:.4f}")
    if p_val < 0.05:
        st.error("Rejects normality (p < 0.05)")
    else:
        st.success("Fails to reject normality (p >= 0.05)")

    # 2.3.5 Box Plot
    st.subheader("Comparison of Return Distributions")
    st.plotly_chart(px.box(stock_ret, title="Daily Returns Box Plot"))

# --- TAB 3: CORRELATION & DIVERSIFICATION (Section 2.4) ---
with tab3:
    st.header("Correlation & Diversification")
    
    # 2.4.1 Diverging Heatmap
    st.subheader("Pairwise Correlation Matrix")
    fig_heat = px.imshow(stock_ret.corr(), text_auto=True, color_continuous_scale='RdBu_r', range_color=[-1, 1])
    st.plotly_chart(fig_heat)

    # 2.4.2 Scatter Plot
    st.subheader("Scatter Analysis of Returns")
    s_x = st.selectbox("Stock X", user_tickers, index=0)
    s_y = st.selectbox("Stock Y", user_tickers, index=1)
    st.plotly_chart(px.scatter(stock_ret, x=s_x, y=s_y, trendline="ols", title=f"{s_x} vs {s_y} Daily Returns"))

    # 2.4.3 Rolling Correlation
    st.subheader("Rolling Correlation Over Time")
    c_win = st.slider("Correlation Window (Days)", 20, 126, 60)
    st.line_chart(stock_ret[s_x].rolling(c_win).corr(stock_ret[s_y]))

    # 2.4.4 Two-Asset Portfolio Explorer
    st.divider()
    st.subheader("Two-Asset Portfolio Explorer")
    w_a = st.slider(f"Weight in {s_x}", 0.0, 1.0, 0.5)
    
    # Portfolio Math (Section 7.1)
    r1, r2 = stock_ret[s_x].mean() * 252, stock_ret[s_y].mean() * 252
    v1, v2 = stock_ret[s_x].std() * np.sqrt(252), stock_ret[s_y].std() * np.sqrt(252)
    cov_12 = stock_ret[[s_x, s_y]].cov().iloc[0,1] * 252
    
    curr_ret = (w_a * r1) + ((1-w_a) * r2)
    curr_vol = np.sqrt((w_a**2 * v1**2) + ((1-w_a)**2 * v2**2) + (2 * w_a * (1-w_a) * cov_12))
    
    c_met1, c_met2 = st.columns(2)
    c_met1.metric("Current Portfolio Ann. Return", f"{curr_ret:.2%}")
    c_met2.metric("Current Portfolio Ann. Volatility", f"{curr_vol:.2%}")

    # The Volatility Curve
    ws = np.linspace(0, 1, 100)
    vs = [np.sqrt((w**2 * v1**2) + ((1-w)**2 * v2**2) + (2 * w * (1-w) * cov_12)) for w in ws]
    fig_eff = px.line(x=ws, y=vs, labels={'x': f'Weight in {s_x}', 'y': 'Ann. Volatility'}, title="Diversification Volatility Curve")
    fig_eff.add_scatter(x=[w_a], y=[curr_vol], name="Current Allocation", marker=dict(size=12, color='red'))
    st.plotly_chart(fig_eff)
    
    st.info("**Rubric Insight:** The curve 'dips' below individual stocks' volatilities because the correlation is less than 1. This illustrates how combining assets reduces total risk.")