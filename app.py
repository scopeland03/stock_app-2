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

# 2. SIDEBAR PARAMETERS & METHODOLOGY
st.sidebar.header("Parameters")

# Date Validation: Enforce 1-year minimum range
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())

if (end_date - start_date).days < 365:
    st.sidebar.error("Error: Date range must be at least 1 year.")
    st.stop()

ticker_input = st.sidebar.text_input("Tickers (2-5, comma separated)", value="AAPL, MSFT, NVDA").upper()
tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

# Ticker Count Validation
if len(tickers) < 2 or len(tickers) > 5:
    st.sidebar.warning("Please enter between 2 and 5 tickers.")
    st.stop()

with st.sidebar.expander("About & Methodology"):
    st.write("""
    **Data Source:** Yahoo Finance (Adjusted Closing Prices).
    **Returns:** Simple arithmetic daily returns.
    **Annualization:** Calculations assume 252 trading days.
    **Benchmark:** S&P 500 (^GSPC) used for comparison.
    """)

# 3. ROBUST DATA ENGINE
@st.cache_data
def get_cleaned_data(tkrs, start, end):
    all_tkrs = list(set(tkrs + ["^GSPC"]))
    try:
        raw_df = yf.download(all_tkrs, start=start, end=end, auto_adjust=True)['Close']
    except Exception as e:
        return None, f"Download failed: {str(e)}"

    if raw_df.empty:
        return None, "No data found for the selected tickers/dates."

    # Identify failed tickers (those not in columns or entirely NaN)
    missing_tkrs = [t for t in all_tkrs if t not in raw_df.columns or raw_df[t].isna().all()]
    if missing_tkrs:
        return None, f"Failed to download data for: {', '.join(missing_tkrs)}"

    # 5% Missing Data Rule
    valid_cols = []
    for col in raw_df.columns:
        pct_missing = raw_df[col].isna().mean()
        if pct_missing > 0.05:
            st.warning(f"Dropping {col}: More than 5% missing data ({pct_missing:.2%}).")
        else:
            valid_cols.append(col)
    
    # Truncate to overlapping range (dropna handles the alignment)
    df = raw_df[valid_cols].dropna()
    
    if df.empty:
        return None, "Insufficient overlapping data across selected assets."
        
    return df, None

with st.spinner("Fetching and processing market data..."):
    df_raw, error_msg = get_cleaned_data(tickers, start_date, end_date)

if error_msg:
    st.error(error_msg)
    st.stop()

# Calculations
returns = df_raw.pct_change().dropna()
stock_list = [t for t in tickers if t in returns.columns]
stock_rets = returns[stock_list]
bench_rets = returns["^GSPC"]

# 4. TABS
t1, t2, t3 = st.tabs(["Performance Summary", "Risk & Normality", "Correlation & Portfolio"])

# TAB 1: PERFORMANCE & INTERACTIVE PRICE CHART
with t1:
    st.subheader("Adjusted Closing Prices")
    selected_display = st.multiselect("Select stocks to view:", stock_list, default=stock_list)
    if selected_display:
        fig_price = px.line(df_raw[selected_display], 
                            labels={"value": "Price (USD)", "Date": "Timeline"},
                            title="Adjusted Close Over Time")
        st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Summary Statistics (Annualized)")
    # Expanded Stats: Return, Vol, Skew, Kurtosis, Min, Max
    stats_df = pd.DataFrame({
        "Annualized Return": returns.mean() * 252,
        "Annualized Vol": returns.std() * np.sqrt(252),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurtosis(),
        "Min Daily Return": returns.min(),
        "Max Daily Return": returns.max()
    }).T
    st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
    # CUMULATIVE WEALTH INDEX (Professor Requirement)
    st.subheader("Cumulative Wealth Index ($10,000 Investment)")
    
    # Calculate Equal-Weight Portfolio returns
    ew_portfolio_rets = stock_rets.mean(axis=1)
    
    # Combine all assets for the wealth index calculation
    wealth_rets = returns.copy()
    wealth_rets["Equal-Weight Portfolio"] = ew_portfolio_rets
    
    # Wealth Index Formula: 10,000 * cumulative product of (1 + returns)
    wealth_index = (1 + wealth_rets).cumprod() * 10000
    
    fig_wealth = px.line(wealth_index, 
                         labels={"value": "Portfolio Value ($)", "Date": "Timeline"},
                         title="Growth of $10,000 Investment")
    st.plotly_chart(fig_wealth, use_container_width=True)

# TAB 2: RISK & NORMALITY
with t2:
# Assuming 'stock_rets' is a DataFrame of daily simple arithmetic returns 
# for the user-selected stocks.

    st.header ("Risk and Distribution Analysis")

# Layout for controls
col_ctrl, col_display = st.columns([1, 3])

with col_ctrl:
    # Requirement: User selects a specific stock for detailed analysis
    target_stock = st.selectbox("Select a stock for distribution analysis:", stock_rets.columns)
    
    # Requirement: Toggle/Tab to switch between Histogram and Q-Q Plot
    plot_type = st.radio("Select Plot Type:", ["Distribution Plot", "Q-Q Plot"])
    
   
   
# 1. ROLLING VOLATILITY CHART WITH WINDOW SLIDER
st.subheader("1. Rolling Volatility")

# Requirement: User must be able to adjust window length using a slider or select box
vol_window = st.select_slider(
    "Select Rolling Window Length (Days):",
    options=[30, 60, 90],
    value=60,
    help="Adjust the window to see short-term vs long-term volatility trends."
)

# Calculation: Annualized Rolling Volatility = Daily Std Dev * sqrt(252)
rolling_vol = stock_rets.rolling(window=vol_window).std() * np.sqrt(252)

# Requirement: Chart must have a title and labeled axes
fig_vol = px.line(
    rolling_vol,
    title=f"{vol_window}-Day Rolling Annualized Volatility",
    labels={"value": "Annualized Volatility", "Date": "Timeline"},
    template="plotly_white"
)

# Requirement: All series on a single chart with a legend
fig_vol.update_layout(legend_title_text='Tickers')

st.plotly_chart(fig_vol, use_container_width=True)

# 2. DISTRIBUTION PLOT
if plot_type == "Distribution Plot":
    # Requirement: Histogram of daily returns
    # Requirement: Overlay with fitted normal distribution curve using scipy.stats
    mu, std = stats.norm.fit(stock_rets[target_stock])
    
    fig_dist = px.histogram(
        stock_rets, 
        x=target_stock, 
        nbins=50, 
        histnorm='probability density',
        title=f"Daily Return Distribution: {target_stock}",
        labels={target_stock: "Daily Return"}
    )
    
    # Generate the normal curve line
    x_range = np.linspace(stock_rets[target_stock].min(), stock_rets[target_stock].max(), 100)
    y_pdf = stats.norm.pdf(x_range, mu, std)
    
    fig_dist.add_scatter(x=x_range, y=y_pdf, name="Normal Fit", line=dict(color='red', width=3))
    st.plotly_chart(fig_dist, use_container_width=True)

# 3. Q-Q PLOT
    target = st.selectbox("Select stock for Risk Analysis", stock_list)
    # Requirement: Toggle/Tab for Histogram vs Q-Q
    risk_view = st.radio("Display Mode", ["Histogram", "Q-Q Plot"])
    
    if risk_view == "Histogram":
        # Requirement: Histogram with fitted normal curve
        mu, std = stats.norm.fit(stock_rets[target])
        fig_hist = px.histogram(stock_rets, x=target, nbins=50, histnorm='probability density')
        x_axis = np.linspace(stock_rets[target].min(), stock_rets[target].max(), 100)
        fig_hist.add_scatter(x=x_axis, y=stats.norm.pdf(x_axis, mu, std), name="Normal Fit", line=dict(color='red'))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        # Requirement: Q-Q Plot using scipy.stats.probplot
        fig_qq, ax = plt.subplots()
        stats.probplot(stock_rets[target], dist="norm", plot=ax)
        st.pyplot(fig_qq)

# 4. NORMALITY TEST (Jarque-Bera)
# Requirement: Display JB stat and p-value near the plot
jb_stat, jb_p = stats.jarque_bera(stock_rets[target_stock])

st.write(f"**Jarque-Bera Statistic:** {jb_stat:.4f}")
st.write(f"**p-value:** {jb_p:.4f}")

# Requirement: Brief message indicating if test rejects normality at 5% level
if jb_p < 0.05:
    st.error("Result: Rejects normality (p < 0.05)")
else:
    st.success("Result: Fails to reject normality (p >= 0.05)")

st.divider()


# 5. BOX PLOT
# Requirement: Display a box plot comparing daily return distributions of all selected stocks side by side on a single chart.

st.subheader("Distribution Spread (Box Plot)")

# Requirement: Every chart must have a title and labeled axes.
fig_box = px.box(
    stock_rets, 
    title="Daily Return Spread Comparison",
    labels={
        "value": "Daily Simple Arithmetic Return", 
        "variable": "Stock Ticker"
    },
    template="plotly_white"
)

# Requirement: All series on a single chart with a legend.
fig_box.update_layout(
    showlegend=True,
    legend_title_text='Tickers'
)

st.plotly_chart(fig_box, use_container_width=True)

# TAB 3: CORRELATION & PORTFOLIO EXPLORER
with t3:
    st.header("Correlation & Portfolio Explorer")

# 1. CORRELATION HEATMAP
# Requirement: Pairwise correlation matrix, annotated cells, diverging scale
st.subheader("1. Correlation Heatmap")
corr_matrix = stock_rets.corr()
fig_heat = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale='RdBu_r',
    range_color=[-1, 1],
    title="Annotated Pairwise Correlation Matrix",
    labels=dict(color="Correlation")
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# 2. SCATTER PLOT & 3. ROLLING CORRELATION
# Requirement: User selects two stocks for movement visualization
st.subheader("2. & 3. Asset Co-movement")
col_s1, col_s2 = st.columns(2)
with col_s1:
    stock_a = st.selectbox("Select Asset A", stock_rets.columns, index=0, key="t3_s1")
with col_s2:
    stock_b = st.selectbox("Select Asset B", stock_rets.columns, index=1, key="t3_s2")

# Scatter Plot
fig_scatter = px.scatter(
    stock_rets, x=stock_a, y=stock_b, trendline="ols",
    title=f"Daily Return Scatter: {stock_a} vs {stock_b}",
    labels={stock_a: f"{stock_a} Return", stock_b: f"{stock_b} Return"}
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Rolling Correlation
# Requirement: Adjustable rolling window length
roll_corr_window = st.select_slider("Rolling Correlation Window (Days)", options=[30, 60, 90], value=60)
rolling_corr = stock_rets[stock_a].rolling(window=roll_corr_window).corr(stock_rets[stock_b])
fig_roll_corr = px.line(
    rolling_corr,
    title=f"{roll_corr_window}-Day Rolling Correlation: {stock_a} vs {stock_b}",
    labels={"value": "Correlation", "Date": "Timeline"}
)
st.plotly_chart(fig_roll_corr, use_container_width=True)

st.divider()

# 4. TWO-ASSET PORTFOLIO EXPLORER
st.subheader("4. Two-Asset Portfolio Explorer")

# Pre-calculations for the Volatility Curve
# Annualization uses 252 trading days
mu_a, mu_b = stock_rets[stock_a].mean() * 252, stock_rets[stock_b].mean() * 252
std_a, std_b = stock_rets[stock_a].std() * np.sqrt(252), stock_rets[stock_b].std() * np.sqrt(252)
rho = stock_rets[stock_a].corr(stock_rets[stock_b])

# Generate the Curve Data
w_range = np.linspace(0, 1, 101)
vols_range = [
    np.sqrt((w**2 * std_a**2) + ((1-w)**2 * std_b**2) + (2 * w * (1-w) * std_a * std_b * rho)) 
    for w in w_range
]

# Portfolio Weight Slider (Placed at the bottom as requested)
# Requirement: Slider sets weight on Stock A from 0% to 100%
weight_a_pct = st.slider(f"Adjust Weight on {stock_a} (%)", 0.0, 100.0, 50.0)
weight_a = weight_a_pct / 100.0
weight_b = 1.0 - weight_a

# Dynamic Metrics
# Requirement: Display annualized return and volatility for current weight
curr_ret = (weight_a * mu_a) + (weight_b * mu_b)
curr_vol = np.sqrt((weight_a**2 * std_a**2) + (weight_b**2 * std_b**2) + (2 * weight_a * weight_b * std_a * std_b * rho))

m1, m2 = st.columns(2)
m1.metric("Annualized Portfolio Return", f"{curr_ret:.2%}")
m2.metric("Annualized Portfolio Volatility", f"{curr_vol:.2%}")

# Volatility Curve Chart
# Requirement: Plot vol (y) against weight (x). Mark current position
fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=w_range * 100, y=vols_range, name="Volatility Curve", line=dict(color='blue')))
fig_curve.add_trace(go.Scatter(
    x=[weight_a_pct], y=[curr_vol], 
    mode='markers', marker=dict(size=14, color='red', symbol='x'), 
    name="Current Allocation"
))

fig_curve.update_layout(
    title=f"Volatility Curve: {stock_a} & {stock_b}",
    xaxis_title=f"Weight in {stock_a} (%)",
    yaxis_title="Annualized Volatility",
    template="plotly_white"
)
st.plotly_chart(fig_curve, use_container_width=True)

# Diversification Description
# Requirement: Brief description explaining the "dip" and correlation
st.info(f"""
**The Diversification Effect:** Combining these assets produces a portfolio with lower volatility than the weighted 
average of the two. This effect is driven by the correlation (**{rho:.2f}**) between {stock_a} and {stock_b}. 
When correlation is less than 1, the curve dips, allowing for risk reduction.
""")
