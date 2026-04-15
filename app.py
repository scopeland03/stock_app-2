import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from datetime import date, timedelta
import math

st.set_page_config(page_title="Stock Analyzer Pro", layout="wide")
st.title("Interactive Multi-Stock Analysis Dashboard")

st.sidebar.header("Configuration")

# Multi-ticker input
ticker_input = st.sidebar.text_input("Enter 2-5 Tickers (comma separated)", value="NVDA, AMD, TSM").upper().strip()
user_tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

# Date range with 1-year enforcement
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=date.today())

if (end_date - start_date).days < 365:
    st.sidebar.error("Rubric Error: Date range must be at least 1 year.")
    st.stop()

if len(user_tickers) < 2 or len(user_tickers) > 5:
    st.sidebar.warning("Please enter between 2 and 5 tickers to proceed.")
    st.stop()

# Methodology Section (Required by Section 3.5)
with st.sidebar.expander("About & Methodology"):
    st.write("""
    - **Data Source:** Yahoo Finance
    - **Annualization:** 252 trading days
    - **Returns:** Simple arithmetic returns
    - **Risk-Free Rate:** 4.5% (default)
    """)

@st.cache_data(ttl=3600)
def load_multi_data(tickers, start, end):
    # Ensure benchmark is included
    all_tickers = list(set(tickers + ["^GSPC"]))
    try:
        # 1. Download data
        # auto_adjust=True handles splits/dividends
        # actions=False speeds up the download
        data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, actions=False)
        
        if data.empty:
            return None
        
        # 2. Fix the Multi-Index Column Issue
        # If multiple tickers are downloaded, yfinance returns (Price, Ticker) levels
        if isinstance(data.columns, pd.MultiIndex):
            # Select the 'Close' level and flatten
            if 'Close' in data.columns.levels[0]:
                clean_df = data['Close']
            else:
                # Fallback for different yf versions
                clean_df = data.iloc[:, data.columns.get_level_values(0) == 'Close']
                clean_df.columns = clean_df.columns.get_level_values(1)
        else:
            # If only one attribute is returned
            clean_df = data
            
        # 3. Clean and Validate (Section 2.1.4)
        clean_df = clean_df.dropna()
        
        # 4. Final safety check: ensure the tickers we want actually exist in the result
        available_tickers = clean_df.columns.tolist()
        for t in tickers:
            if t not in available_tickers:
                st.sidebar.warning(f"Ticker {t} not found or has insufficient data.")
                
        return clean_df
    except Exception as e:
        st.error(f"Critical Download Error: {e}")
        return None

with st.spinner("Fetching data from Yahoo Finance..."):
    df_raw = load_multi_data(user_tickers, start_date, end_date)

if df_raw is None or df_raw.empty:
    st.error("Failed to retrieve data. Please check ticker symbols.")
    st.stop()

# Calculations
daily_returns = df_raw.pct_change().dropna()
user_returns = daily_returns[user_tickers]
benchmark_returns = daily_returns["^GSPC"]


tab1, tab2, tab3 = st.tabs(["Price & Returns", "Risk Analysis", "Portfolio Explorer"])

# --- TAB 1: PRICE & RETURNS ---
with tab1:
    st.header("Price and Return Analysis")
    
    # Wealth Index (Section 2.2.4)
    st.subheader("Growth of $10,000 Investment")
    eq_weight_ret = user_returns.mean(axis=1)
    wealth_index = (1 + user_returns).cumprod() * 10000
    wealth_index["S&P 500"] = (1 + benchmark_returns).cumprod() * 10000
    wealth_index["Equal-Weight Portfolio"] = (1 + eq_weight_ret).cumprod() * 10000
    st.line_chart(wealth_index)

    # Summary Stats Table (Section 2.2.3)
    stats_df = pd.DataFrame({
        "Ann. Return": user_returns.mean() * 252,
        "Ann. Volatility": user_returns.std() * np.sqrt(252),
        "Skewness": user_returns.skew(),
        "Kurtosis": user_returns.kurtosis(),
        "Min Daily": user_returns.min(),
        "Max Daily": user_returns.max()
    }).T
    st.write("Summary Statistics", stats_df)

# --- TAB 2: RISK ANALYSIS ---
with tab2:
    st.header("Risk & Distribution")
    sel_stock = st.selectbox("Select stock for analysis", user_tickers)
    
    # --- ADDED: Rolling Volatility (Section 2.3.1) ---
    st.subheader(f"Rolling Annualized Volatility: {sel_stock}")
    vol_window = st.slider("Select Rolling Window (Days)", min_value=10, max_value=126, value=30)
    
    # Calculate rolling vol: std * sqrt(252) for annualization
    rolling_vol = user_returns[sel_stock].rolling(window=vol_window).std() * np.sqrt(252)
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, name="Rolling Vol"))
    fig_vol.update_layout(
        xaxis_title="Date", 
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%"
    )
    st.plotly_chart(fig_vol)
   
    # Fit normal distribution to get mu and std
    mu, std = stats.norm.fit(user_returns[sel_stock])
    # Create normal curve line
    x_range = np.linspace(user_returns[sel_stock].min(), user_returns[sel_stock].max(), 100)
    p = stats.norm.pdf(x_range, mu, std)
    
    fig_hist.add_trace(go.Scatter(
        x=x_range, y=p, 
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))
    
    fig_hist.update_layout(
        title=f"Returns Distribution for {sel_stock}",
        xaxis_title="Daily Return",
        yaxis_title="Density"
    )
    st.plotly_chart(fig_hist)

    # Q-Q Plot (Section 2.3.3)
    fig_qq, ax_qq = plt.subplots()
    stats.probplot(user_returns[sel_stock], dist="norm", plot=ax_qq)
    ax_qq.set_title(f"Q-Q Plot for {sel_stock}")
    st.pyplot(fig_qq)
    
    # Jarque-Bera Test (Section 2.3.4)
    jb_stat, p_val = stats.jarque_bera(user_returns[sel_stock])
    st.write(f"Jarque-Bera p-value: {p_val:.4f}")
    if p_val < 0.05:
        st.error("Rejects normality (p < 0.05)")
    else:
        st.success("Fails to reject normality (p >= 0.05)")

# --- TAB 3: PORTFOLIO EXPLORER ---
with tab3:
    st.header("Diversification & Correlation")
    
    # Heatmap (Section 2.4.1)
    st.plotly_chart(px.imshow(user_returns.corr(), text_auto=True, title="Correlation Matrix"))

    # Two-Asset Explorer (Section 2.4.4)
    s1, s2 = st.selectbox("Stock A", user_tickers), st.selectbox("Stock B", user_tickers, index=1)
    w_a = st.slider(f"Weight on {s1}", 0.0, 1.0, 0.5)
    
    # Diversification Curve Math
    vol_a = user_returns[s1].std() * np.sqrt(252)
    vol_b = user_returns[s2].std() * np.sqrt(252)
    correlation = user_returns[s1].corr(user_returns[s2])

    st.write("This curve demonstrates that combining two stocks can result in a portfolio with lower risk than either individual stock due to diversification.")
    
    w_range = np.linspace(0, 1, 100)
    vols = [np.sqrt((w**2 * vol_a**2) + ((1-w)**2 * vol_b**2) + (2*w*(1-w)*vol_a*vol_b*correlation)) for w in w_range]
    curr_v = np.sqrt((w_a**2 * vol_a**2) + ((1-w_a)**2 * vol_b**2) + (2*w_a*(1-w_a)*vol_a*vol_b*correlation))
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=w_range, y=vols, name="Volatility Curve"))
    fig_curve.add_trace(go.Scatter(x=[w_a], y=[curr_v], mode='markers', marker=dict(size=12, color='red'), name="Current Portfolio"))
    fig_curve.update_layout(title="Two-Asset Risk Curve", xaxis_title=f"Weight on {s1}", yaxis_title="Ann. Volatility")
    st.plotly_chart(fig_curve)
    
    st.info("Note: The dip in the curve shows that combining assets can lower total risk.")
