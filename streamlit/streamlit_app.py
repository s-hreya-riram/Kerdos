"""
Comprehensive ML Portfolio Strategy Dashboard with Benchmark Comparison
Includes all quantstats-style metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import snowflake.connector
from scipy import stats
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Page config
st.set_page_config(
    page_title="ML Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {color: #1f77b4;}
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    .winner {
        background-color: #d4edda;
        padding: 5px;
        border-radius: 3px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SNOWFLAKE CONNECTION
# ============================================================================

@st.cache_resource
def get_snowflake_connection():
    """Connect to Snowflake, using Streamlit secrets if available."""
    
    user = os.getenv("SNOWFLAKE_USER") or st.secrets.get("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD") or st.secrets.get("SNOWFLAKE_PASSWORD")
    account = os.getenv("SNOWFLAKE_ACCOUNT") or st.secrets.get("SNOWFLAKE_ACCOUNT")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE") or st.secrets.get("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE") or st.secrets.get("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA") or st.secrets.get("SNOWFLAKE_SCHEMA", "PUBLIC")
    role = os.getenv("SNOWFLAKE_ROLE") or st.secrets.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN")

    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema,
        role=role
    )
    return conn


@st.cache_data(ttl=1)
def load_strategy_performance():
    """Load performance data for all strategies"""
    conn = get_snowflake_connection()
    
    query = """
    SELECT 
        STRATEGY_NAME,
        TIMESTAMP,
        PORTFOLIO_VALUE,
        CASH,
        IS_OUT_OF_SAMPLE
    FROM "KERDOS_FUND"."KERDOS_SCHEMA"."STRATEGY_PERFORMANCE"
    ORDER BY STRATEGY_NAME, TIMESTAMP
    """

    
    df = pd.read_sql(query, conn)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    return df


@st.cache_data(ttl=1)
def load_strategy_positions(strategy_name):
    """Load positions for a specific strategy"""
    conn = get_snowflake_connection()
    
    query = f"""
    SELECT 
        TIMESTAMP,
        SYMBOL,
        QUANTITY,
        MARKET_VALUE,
        AVG_PRICE
    FROM KERDOS_FUND.KERDOS_SCHEMA.STRATEGY_POSITIONS
    WHERE STRATEGY_NAME = '{strategy_name}'
    ORDER BY TIMESTAMP, SYMBOL
    """
    
    df = pd.read_sql(query, conn)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    return df


# ============================================================================
# METRICS CALCULATIONS
# ============================================================================

def calculate_returns(portfolio_values):
    """Calculate daily returns from portfolio values"""
    return portfolio_values.pct_change().dropna()


def calculate_comprehensive_metrics(returns, portfolio_values, risk_free_rate=0.0358):
    """
    Calculate all metrics shown in quantstats tearsheet
    """
    # Convert to numpy for faster calculations
    returns_array = returns.values
    n_days = len(returns_array)
    
    # Basic metrics
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    
    # Annualized metrics
    trading_days_per_year = 252
    years = n_days / trading_days_per_year
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Volatility
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(trading_days_per_year)
    
    # Sharpe Ratio
    excess_returns = returns - (risk_free_rate / trading_days_per_year)
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days_per_year)
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_days_per_year)
    sortino = ((returns.mean() * trading_days_per_year - risk_free_rate) / downside_std) if downside_std > 0 else 0
    
    # Drawdown metrics
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Longest drawdown period
    in_drawdown = drawdown < 0
    drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()[in_drawdown]
    if len(drawdown_periods) > 0:
        longest_dd_days = drawdown_periods.value_counts().max()
    else:
        longest_dd_days = 0
    
    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win/Loss metrics
    win_days = (returns > 0).sum()
    loss_days = (returns < 0).sum()
    win_rate = win_days / n_days if n_days > 0 else 0
    
    # Best/Worst days
    best_day = returns.max()
    worst_day = returns.min()
    
    # Monthly metrics
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    best_month = monthly_returns.max()
    worst_month = monthly_returns.min()
    win_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    win_month_pct = win_months / total_months if total_months > 0 else 0
    
    avg_up_month = monthly_returns[monthly_returns > 0].mean()
    avg_down_month = monthly_returns[monthly_returns < 0].mean()
    
    # Yearly metrics
    yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    best_year = yearly_returns.max()
    worst_year = yearly_returns.min()
    win_years = (yearly_returns > 0).sum()
    total_years = len(yearly_returns)
    win_year_pct = win_years / total_years if total_years > 0 else 0
    
    # Quarterly metrics
    quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
    win_quarters = (quarterly_returns > 0).sum()
    total_quarters = len(quarterly_returns)
    win_quarter_pct = win_quarters / total_quarters if total_quarters > 0 else 0
    
    # Skewness and Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Value at Risk (95% confidence)
    var_95 = returns.quantile(0.05)
    
    # Expected Shortfall (CVaR)
    cvar = returns[returns <= var_95].mean()
    
    # Average drawdown
    avg_drawdown = drawdown[drawdown < 0].mean()
    
    # Average drawdown days
    if len(drawdown_periods) > 0:
        avg_dd_days = drawdown_periods.value_counts().mean()
    else:
        avg_dd_days = 0
    
    # Recovery Factor
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Ulcer Index
    ulcer_index = np.sqrt((drawdown ** 2).mean())
    
    # Probabilistic Sharpe Ratio
    n = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()
    sr_std = np.sqrt((1 + (0.5 * sharpe**2) - (skew * sharpe) + (((kurt - 3) / 4) * sharpe**2)) / n)
    prob_sharpe = stats.norm.cdf((sharpe - 0) / sr_std) if sr_std > 0 else 0
    
    # Expected returns
    expected_daily = returns.mean()
    expected_monthly = (1 + expected_daily) ** 21 - 1  # Approx 21 trading days per month
    expected_yearly = (1 + expected_daily) ** trading_days_per_year - 1
    
    # Time in market (non-zero positions)
    # This would need position data - simplified here
    time_in_market = 0.95  # Placeholder
    
    metrics = {
        # Returns
        'Total Return': total_return,
        'CAGR': cagr,
        
        # Risk-adjusted
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Prob. Sharpe Ratio': prob_sharpe,
        
        # Risk
        'Volatility (ann.)': annual_vol,
        'Max Drawdown': max_drawdown,
        'Longest DD Days': longest_dd_days,
        'Avg. Drawdown': avg_drawdown,
        'Avg. Drawdown Days': avg_dd_days,
        'Ulcer Index': ulcer_index,
        
        # Distribution
        'Skew': skewness,
        'Kurtosis': kurtosis,
        'Daily Value-at-Risk': var_95,
        'Expected Shortfall (cVaR)': cvar,
        
        # Win rates
        'Win Days%': win_rate,
        'Win Month%': win_month_pct,
        'Win Quarter%': win_quarter_pct,
        'Win Year%': win_year_pct,
        
        # Best/Worst
        'Best Day': best_day,
        'Worst Day': worst_day,
        'Best Month': best_month,
        'Worst Month': worst_month,
        'Best Year': best_year,
        'Worst Year': worst_year,
        
        # Expected
        'Expected Daily%': expected_daily,
        'Expected Monthly%': expected_monthly,
        'Expected Yearly%': expected_yearly,
        
        # Other
        'Recovery Factor': recovery_factor,
        'Avg. Up Month': avg_up_month,
        'Avg. Down Month': avg_down_month,
        'Time in Market': time_in_market,
    }
    
    return metrics


def calculate_beta_alpha(strategy_returns, benchmark_returns):
    """Calculate Beta and Alpha relative to benchmark"""
    # Align returns
    combined = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(combined) < 2:
        return 0, 0, 0
    
    # Beta (covariance / variance)
    cov_matrix = np.cov(combined['strategy'], combined['benchmark'])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
    
    # Alpha (excess return)
    strategy_mean = combined['strategy'].mean() * 252
    benchmark_mean = combined['benchmark'].mean() * 252
    alpha = strategy_mean - (beta * benchmark_mean)
    
    # Correlation
    corr = combined['strategy'].corr(combined['benchmark'])
    
    return beta, alpha, corr


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    cwd = os.path.dirname(__file__)
    logo_path = os.path.join(cwd, 'assets', 'logo.png')
    logo = Image.open(logo_path)
    height, width = int(logo.height * 0.4), int(logo.width * 0.4)
    try:
        resized_image = logo.resize((width, height), Image.Resampling.LANCZOS)
    except AttributeError:
        resized_image = logo.resize((width, height), Image.LANCZOS)

    _, col2, _ = st.columns([1, 3, 1])
    with col2:
        st.image(resized_image)
    st.markdown("### Comprehensive Performance Analysis with Benchmark Comparison")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Load data
    with st.spinner("Loading data from Snowflake..."):
        try:
            perf_df = load_strategy_performance()
            
            if perf_df.empty:
                st.error("No data found in Snowflake. Please run the backtest first.")
                st.info("Run: `python run_dual_backtest.py`")
                return
            
            strategies = perf_df['STRATEGY_NAME'].unique()
            
            st.sidebar.success(f"✅ Loaded {len(perf_df)} records")
            st.sidebar.info(f"Strategies: {', '.join(strategies)}")
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Strategy selection
    if 'ML_XGBOOST' in strategies and 'SPY_BENCHMARK' in strategies:
        compare_mode = st.sidebar.checkbox("Compare with Benchmark", value=True)
    else:
        compare_mode = False
        st.sidebar.warning("Only one strategy found - comparison disabled")
    
    # Date range filter
    min_date = perf_df['TIMESTAMP'].min().date()
    max_date = perf_df['TIMESTAMP'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data by date
    if len(date_range) == 2:
        start_date, end_date = date_range
        perf_df = perf_df[
            (perf_df['TIMESTAMP'].dt.date >= start_date) &
            (perf_df['TIMESTAMP'].dt.date <= end_date)
        ]
    
    # Risk-free rate
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.58,
        step=0.01
    ) / 100
    
    # ========================================================================
    # PREPARE DATA
    # ========================================================================
    
    # Separate strategies
    ml_df = perf_df[perf_df['STRATEGY_NAME'] == 'ML_XGBOOST'].copy()
    spy_df = perf_df[perf_df['STRATEGY_NAME'] == 'SPY_BENCHMARK'].copy()
    
    if ml_df.empty:
        st.error("No ML strategy data found")
        return
    
    # Calculate returns
    ml_df = ml_df.sort_values('TIMESTAMP').set_index('TIMESTAMP')
    ml_returns = ml_df['PORTFOLIO_VALUE'].pct_change().dropna()
    
    if not spy_df.empty and compare_mode:
        spy_df = spy_df.sort_values('TIMESTAMP').set_index('TIMESTAMP')
        spy_returns = spy_df['PORTFOLIO_VALUE'].pct_change().dropna()
    else:
        spy_returns = None
    
    # Calculate metrics
    ml_metrics = calculate_comprehensive_metrics(
        ml_returns,
        ml_df['PORTFOLIO_VALUE'],
        risk_free_rate
    )
    
    if spy_returns is not None:
        spy_metrics = calculate_comprehensive_metrics(
            spy_returns,
            spy_df['PORTFOLIO_VALUE'],
            risk_free_rate
        )
        
        # Calculate Beta/Alpha
        beta, alpha, corr = calculate_beta_alpha(ml_returns, spy_returns)
        ml_metrics['Beta'] = beta
        ml_metrics['Alpha'] = alpha
        ml_metrics['Correlation'] = corr
    else:
        spy_metrics = {}
    
    # ========================================================================
    # KEY METRICS OVERVIEW
    # ========================================================================
    
    st.header("📊 Key Performance Metrics")
    
    if compare_mode and spy_metrics:
        # Comparison table
        metrics_to_show = [
            ('Total Return', '{:.2%}'),
            ('CAGR', '{:.2%}'),
            ('Sharpe Ratio', '{:.2f}'),
            ('Sortino Ratio', '{:.2f}'),
            ('Max Drawdown', '{:.2%}'),
            ('Volatility (ann.)', '{:.2%}'),
            ('Calmar Ratio', '{:.2f}'),
            ('Win Days%', '{:.1%}'),
        ]
        
        comparison_data = []
        for metric_name, fmt in metrics_to_show:
            spy_val = spy_metrics.get(metric_name, 0)
            ml_val = ml_metrics.get(metric_name, 0)
            
            # Determine winner (higher is better except for drawdown and volatility)
            if metric_name in ['Max Drawdown', 'Volatility (ann.)']:
                winner = 'SPY' if spy_val > ml_val else 'Strategy'
            else:
                winner = 'Strategy' if ml_val > spy_val else 'SPY'
            
            comparison_data.append({
                'Metric': metric_name,
                'SPY': fmt.format(spy_val),
                'Strategy': fmt.format(ml_val),
                'Winner': winner
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        def highlight_winner(row):
            if row['Winner'] == 'Strategy':
                return ['', '', 'background-color: #d4edda', '']
            elif row['Winner'] == 'SPY':
                return ['', 'background-color: #d4edda', '', '']
            return [''] * 4
        
        styled_df = comp_df.style.apply(highlight_winner, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    else:
        # Single strategy metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{ml_metrics['Total Return']:.2%}")
            st.metric("CAGR", f"{ml_metrics['CAGR']:.2%}")
        
        with col2:
            st.metric("Sharpe Ratio", f"{ml_metrics['Sharpe Ratio']:.2f}")
            st.metric("Sortino Ratio", f"{ml_metrics['Sortino Ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{ml_metrics['Max Drawdown']:.2%}")
            st.metric("Volatility", f"{ml_metrics['Volatility (ann.)']:.2%}")
        
        with col4:
            st.metric("Calmar Ratio", f"{ml_metrics['Calmar Ratio']:.2f}")
            st.metric("Win Rate", f"{ml_metrics['Win Days%']:.1%}")
    
    # ========================================================================
    # PORTFOLIO VALUE CHART
    # ========================================================================
    
    st.header("💰 Portfolio Value Over Time")
    
    fig = go.Figure()
    
    # ML Strategy
    fig.add_trace(go.Scatter(
        x=ml_df.index,
        y=ml_df['PORTFOLIO_VALUE'],
        mode='lines',
        name='ML Strategy',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # SPY Benchmark
    if compare_mode and not spy_df.empty:
        fig.add_trace(go.Scatter(
            x=spy_df.index,
            y=spy_df['PORTFOLIO_VALUE'],
            mode='lines',
            name='SPY Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Portfolio Growth Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # DETAILED METRICS TABLES
    # ========================================================================
    
    st.header("📋 Comprehensive Metrics")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Risk-Adjusted Returns",
        "Risk Metrics",
        "Win/Loss Statistics",
        "Distribution Metrics"
    ])
    
    with tab1:
        metrics_risk_adj = [
            'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
            'Prob. Sharpe Ratio', 'Recovery Factor'
        ]
        
        if compare_mode and spy_metrics:
            data = []
            for m in metrics_risk_adj:
                data.append({
                    'Metric': m,
                    'SPY': f"{spy_metrics.get(m, 0):.2%}" if 'Prob' in m or 'CAGR' in m else f"{spy_metrics.get(m, 0):.2f}",
                    'Strategy': f"{ml_metrics.get(m, 0):.2%}" if 'Prob' in m or 'CAGR' in m else f"{ml_metrics.get(m, 0):.2f}"
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        else:
            data = [{'Metric': m, 'Value': f"{ml_metrics.get(m, 0):.2%}" if 'Prob' in m or 'CAGR' in m else f"{ml_metrics.get(m, 0):.2f}"} for m in metrics_risk_adj]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    with tab2:
        metrics_risk = [
            'Volatility (ann.)', 'Max Drawdown', 'Longest DD Days',
            'Avg. Drawdown', 'Avg. Drawdown Days', 'Ulcer Index'
        ]
        
        if compare_mode and spy_metrics:
            data = []
            for m in metrics_risk:
                spy_val = spy_metrics.get(m, 0)
                ml_val = ml_metrics.get(m, 0)
                
                if 'Days' in m:
                    data.append({
                        'Metric': m,
                        'SPY': f"{spy_val:.0f}",
                        'Strategy': f"{ml_val:.0f}"
                    })
                elif 'Index' in m:
                    data.append({
                        'Metric': m,
                        'SPY': f"{spy_val:.4f}",
                        'Strategy': f"{ml_val:.4f}"
                    })
                else:
                    data.append({
                        'Metric': m,
                        'SPY': f"{spy_val:.2%}",
                        'Strategy': f"{ml_val:.2%}"
                    })
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        else:
            data = []
            for m in metrics_risk:
                val = ml_metrics.get(m, 0)
                if 'Days' in m:
                    data.append({'Metric': m, 'Value': f"{val:.0f}"})
                elif 'Index' in m:
                    data.append({'Metric': m, 'Value': f"{val:.4f}"})
                else:
                    data.append({'Metric': m, 'Value': f"{val:.2%}"})
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    with tab3:
        metrics_win = [
            'Win Days%', 'Win Month%', 'Win Quarter%', 'Win Year%',
            'Best Day', 'Worst Day', 'Best Month', 'Worst Month',
            'Best Year', 'Worst Year', 'Avg. Up Month', 'Avg. Down Month'
        ]
        
        if compare_mode and spy_metrics:
            data = []
            for m in metrics_win:
                data.append({
                    'Metric': m,
                    'SPY': f"{spy_metrics.get(m, 0):.2%}",
                    'Strategy': f"{ml_metrics.get(m, 0):.2%}"
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        else:
            data = [{'Metric': m, 'Value': f"{ml_metrics.get(m, 0):.2%}"} for m in metrics_win]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    with tab4:
        metrics_dist = [
            'Skew', 'Kurtosis', 'Daily Value-at-Risk',
            'Expected Shortfall (cVaR)', 'Expected Daily%',
            'Expected Monthly%', 'Expected Yearly%'
        ]
        
        if compare_mode and spy_metrics:
            data = []
            for m in metrics_dist:
                spy_val = spy_metrics.get(m, 0)
                ml_val = ml_metrics.get(m, 0)
                
                if 'Skew' in m or 'Kurtosis' in m:
                    data.append({
                        'Metric': m,
                        'SPY': f"{spy_val:.2f}",
                        'Strategy': f"{ml_val:.2f}"
                    })
                else:
                    data.append({
                        'Metric': m,
                        'SPY': f"{spy_val:.2%}",
                        'Strategy': f"{ml_val:.2%}"
                    })
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        else:
            data = []
            for m in metrics_dist:
                val = ml_metrics.get(m, 0)
                if 'Skew' in m or 'Kurtosis' in m:
                    data.append({'Metric': m, 'Value': f"{val:.2f}"})
                else:
                    data.append({'Metric': m, 'Value': f"{val:.2%}"})
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    # ========================================================================
    # DRAWDOWN CHART
    # ========================================================================
    
    st.header("📉 Drawdown Analysis")
    
    # Calculate drawdowns
    ml_cumulative = (1 + ml_returns).cumprod()
    ml_running_max = ml_cumulative.expanding().max()
    ml_drawdown = (ml_cumulative - ml_running_max) / ml_running_max * 100
    
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=ml_drawdown.index,
        y=ml_drawdown,
        fill='tozeroy',
        line=dict(color='#d62728', width=2),
        fillcolor='rgba(214, 39, 40, 0.2)',
        name='ML Strategy'
    ))
    
    if compare_mode and spy_returns is not None:
        spy_cumulative = (1 + spy_returns).cumprod()
        spy_running_max = spy_cumulative.expanding().max()
        spy_drawdown = (spy_cumulative - spy_running_max) / spy_running_max * 100
        
        fig_dd.add_trace(go.Scatter(
            x=spy_drawdown.index,
            y=spy_drawdown,
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            name='SPY Benchmark'
        ))
    
    fig_dd.update_layout(
        title="Drawdown from Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # ========================================================================
    # RETURNS DISTRIBUTION
    # ========================================================================
    
    st.header("📊 Returns Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=ml_returns * 100,
            nbinsx=50,
            name='ML Strategy',
            marker_color='#1f77b4',
            opacity=0.7
        ))
        
        if compare_mode and spy_returns is not None:
            fig_hist.add_trace(go.Histogram(
                x=spy_returns * 100,
                nbinsx=50,
                name='SPY Benchmark',
                marker_color='#ff7f0e',
                opacity=0.5
            ))
        
        fig_hist.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            height=400,
            barmode='overlay',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Q-Q Plot
        from scipy.stats import probplot
        
        qq = probplot(ml_returns, dist='norm')
        
        fig_qq = go.Figure()
        
        fig_qq.add_trace(go.Scatter(
            x=qq[0][0],
            y=qq[0][1],
            mode='markers',
            name='Actual',
            marker=dict(color='#1f77b4')
        ))
        
        fig_qq.add_trace(go.Scatter(
            x=qq[0][0],
            y=qq[1][0] * qq[0][0] + qq[1][1],
            mode='lines',
            name='Normal',
            line=dict(color='#d62728', dash='dash')
        ))
        
        fig_qq.update_layout(
            title="Q-Q Plot (Normality Test)",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_qq, use_container_width=True)
    
    # ========================================================================
    # MONTHLY HEATMAP
    # ========================================================================
    
    st.header("📅 Monthly Returns Heatmap")
    
    # Calculate monthly returns
    ml_monthly = (ml_returns + 1).resample('M').prod() - 1
    ml_monthly_pct = ml_monthly * 100
    
    # Pivot for heatmap
    ml_monthly_pct.index = pd.to_datetime(ml_monthly_pct.index)
    pivot_data = ml_monthly_pct.to_frame('return')
    pivot_data['year'] = pivot_data.index.year
    pivot_data['month'] = pivot_data.index.month
    
    pivot_table = pivot_data.pivot(index='year', columns='month', values='return')
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table.columns = [month_names[m-1] for m in pivot_table.columns]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn',
        zmid=0,
        text=pivot_table.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Return %")
    ))
    
    fig_heatmap.update_layout(
        title="Monthly Returns (%)",
        xaxis_title="Month",
        yaxis_title="Year",
        height=300,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>ML Portfolio Strategy Dashboard | Data from Snowflake</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)


if __name__ == "__main__":
    main()