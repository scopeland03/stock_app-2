import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from datetime import date, timedelta

# 1. SETUP & CONFIG
st.set_page_config(page_title="Quant Analytics Pro", layout="wide")
st.title("Financial Risk & Portfolio Analysis")

# 2. SIDEBAR PARAMETERS
st.sidebar.header("Parameters")

start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())

if (end_date - start_date).days < 365:
    st.sidebar.error("Error: Date range must be at least 1 year.")
    st.stop()

ticker_input = st.sidebar.text_input("Tickers (2-5, comma separated)", value="AAPL, MSFT, NVDA").upper()
tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

if len(tickers) < 2 or len(tickers) > 5:
    st.sidebar.warning("Please enter between 2 and 5 tickers.")
    st.stop()

with st.sidebar.expander("About & Methodology"):
    st.write("""
    **This Streamlit application provides an interactive quantitative risk dashboard that analyzes stock performance, return distributions, and portfolio diversification through cumulative wealth indexing, normality testing, and two-asset volatility modeling.
    
             **Returns:** Simple arithmetic daily returns.
      **Annualization:** 252 trading days.
    **Data Source:** Yahoo Finance.
    """)

# 3. DATA ENGINE
@st.cache_data
def get_cleaned_data(tkrs, start, end):
    all_tkrs = list(set(tkrs + ["^GSPC"]))
    try:
        raw_df = yf.download(all_tkrs, start=start, end=end, auto_adjust=True)['Close']
    except Exception as e:
        return None, f"Download failed: {str(e)}"
    if raw_df.empty:
        return None, "No data found."
    df = raw_df.dropna()
    return df, None

with st.spinner("Fetching market data..."):
    df_raw, error_msg = get_cleaned_data(tickers, start_date, end_date)

if error_msg:
    st.error(error_msg)
    st.stop()

returns = df_raw.pct_change().dropna()
stock_list = [t for t in tickers if t in returns.columns]
stock_rets = returns[stock_list]

# 4. TABS
t1, t2, t3 = st.tabs(["Performance Summary", "Risk & Normality", "Correlation & Portfolio"])

# --- TAB 1: PERFORMANCE ---
with t1:
    st.subheader("Adjusted Closing Prices")
    selected_display = st.multiselect("Select stocks to view:", stock_list, default=stock_list)
    if selected_display:
        st.plotly_chart(px.line(df_raw[selected_display], title="Adjusted Close Over Time"), use_container_width=True)

    st.subheader("Annualized Summary Statistics")
    stats_df = pd.DataFrame({
        "Annualized Return": returns.mean() * 252,
        "Annualized Vol": returns.std() * np.sqrt(252),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurtosis(),
        "Min Daily Return": returns.min(),
        "Max Daily Return": returns.max()
    }).T
    st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Cumulative Wealth Index ($10,000 Investment)")
    ew_portfolio_rets = stock_rets.mean(axis=1)
    wealth_rets = returns.copy()
    wealth_rets["Equal-Weight Portfolio"] = ew_portfolio_rets
    wealth_index = (1 + wealth_rets).cumprod() * 10000
    st.plotly_chart(px.line(wealth_index, title="Growth of $10,000 Investment"), use_container_width=True)

# --- TAB 2: RISK & NORMALITY ---
with t2:
    st.header("Risk and Distribution Analysis")
    
    # 1. Rolling Volatility
    st.subheader("1. Rolling Volatility")
    vol_window = st.select_slider("Select Rolling Window Length (Days):", options=[30, 60, 90], value=60, key="vol_slider")
    rolling_vol = stock_rets.rolling(window=vol_window).std() * np.sqrt(252)
    fig_vol = px.line(rolling_vol, title=f"{vol_window}-Day Rolling Annualized Volatility", template="plotly_white")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()

    # 2. & 3. Distribution & Normality Plots
    col_ctrl, col_disp = st.columns([1, 3])
    with col_ctrl:
        target_stock = st.selectbox("Select a stock for detailed analysis:", stock_list)
        plot_type = st.radio("Select Plot Type:", ["Distribution Plot", "Q-Q Plot"])

    with col_disp:
        if plot_type == "Distribution Plot":
            mu, std = stats.norm.fit(stock_rets[target_stock])
            fig_dist = px.histogram(stock_rets, x=target_stock, nbins=50, histnorm='probability density', title=f"Distribution: {target_stock}")
            x_range = np.linspace(stock_rets[target_stock].min(), stock_rets[target_stock].max(), 100)
            fig_dist.add_scatter(x=x_range, y=stats.norm.pdf(x_range, mu, std), name="Normal Fit", line=dict(color='red'))
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            fig_qq, ax = plt.subplots()
            stats.probplot(stock_rets[target_stock], dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot: {target_stock}")
            st.pyplot(fig_qq)

    # 4. Normality Test (Jarque-Bera)
    jb_stat, jb_p = stats.jarque_bera(stock_rets[target_stock])
    st.write(f"**Jarque-Bera Statistic:** {jb_stat:.4f} | **p-value:** {jb_p:.4f}")
    if jb_p < 0.05:
        st.error("Result: Rejects normality (p < 0.05)")
    else:
        st.success("Result: Fails to reject normality (p >= 0.05)")

    st.divider()

    # 5. Box Plot
    st.subheader("Distribution Spread (Box Plot)")
    fig_box = px.box(stock_rets, title="Daily Return Spread Comparison", template="plotly_white")
    st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 3: CORRELATION & PORTFOLIO ---
with t3:
    st.header("Correlation & Portfolio Explorer")
    
    st.subheader("1. Correlation Heatmap")
    st.plotly_chart(px.imshow(stock_rets.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', range_color=[-1, 1]), use_container_width=True)

    st.divider()
    
    st.subheader("2. & 3. Asset Co-movement")
    c1, c2 = st.columns(2)
    with c1:
        stock_a = st.selectbox("Select Asset A", stock_list, index=0, key="t3_s1")
    with c2:
        stock_b = st.selectbox("Select Asset B", stock_list, index=1, key="t3_s2")

    st.plotly_chart(px.scatter(stock_rets, x=stock_a, y=stock_b, trendline="ols", title=f"Scatter: {stock_a} vs {stock_b}"), use_container_width=True)
    
    rc_win = st.select_slider("Rolling Correlation Window (Days)", options=[30, 60, 90], value=60, key="rc_slider")
    st.plotly_chart(px.line(stock_rets[stock_a].rolling(rc_win).corr(stock_rets[stock_b]), title="Rolling Correlation Over Time"), use_container_width=True)

    st.divider()

    st.subheader("4. Two-Asset Portfolio Explorer")
    
    # Portfolio Calculations
    mu_a, mu_b = stock_rets[stock_a].mean() * 252, stock_rets[stock_b].mean() * 252
    std_a, std_b = stock_rets[stock_a].std() * np.sqrt(252), stock_rets[stock_b].std() * np.sqrt(252)
    rho = stock_rets[stock_a].corr(stock_rets[stock_b])

    # Weights Range for Curve
    w_range = np.linspace(0, 1, 101)
    vols_range = [np.sqrt((w**2 * std_a**2) + ((1-w)**2 * std_b**2) + (2 * w * (1-w) * std_a * std_b * rho)) for w in w_range]

    # Use a placeholder for the chart so the slider can be placed below it
    chart_placeholder = st.empty()
    metric_col1, metric_col2 = st.columns(2)

    weight_a_pct = st.slider(f"Adjust Weight on {stock_a} (%)", 0.0, 100.0, 50.0)
    weight_a = weight_a_pct / 100.0
    weight_b = 1.0 - weight_a

    # Dynamic metrics based on slider
    p_ret = (weight_a * mu_a) + (weight_b * mu_b)
    p_vol = np.sqrt((weight_a**2 * std_a**2) + (weight_b**2 * std_b**2) + (2 * weight_a * weight_b * std_a * std_b * rho))

    metric_col1.metric("Annualized Portfolio Return", f"{p_ret:.2%}")
    metric_col2.metric("Annualized Portfolio Volatility", f"{p_vol:.2%}")

    # Plot the curve with the red dot marking the current allocation
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=w_range * 100, y=vols_range, name="Volatility Curve", line=dict(color='blue')))
    fig_curve.add_trace(go.Scatter(x=[weight_a_pct], y=[p_vol], mode='markers', marker=dict(size=14, color='red', symbol='x'), name="Current Selection"))
    fig_curve.update_layout(title="Diversification Curve", xaxis_title=f"Weight in {stock_a} (%)", yaxis_title="Annualized Volatility", template="plotly_white")
    
    chart_placeholder.plotly_chart(fig_curve, use_container_width=True)

    st.info(f"""
    **Understanding Diversification:** Combining {stock_a} and {stock_b} creates a portfolio with a specific risk-return profile. 
    If the correlation ({rho:.2f}) is less than 1, the curve dips, showing that the portfolio volatility can be lower 
    than the weighted average of the individual stocks.
    """)