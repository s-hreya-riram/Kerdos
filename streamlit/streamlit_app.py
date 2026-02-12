"""
Kerdos Fund Dashboard - Real Data from Snowflake
Beautiful branded interface with actual backtest performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import snowflake.connector
from scipy import stats
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Kerdos Fund Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling matching logo colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --kerdos-dark: #0f2438;
        --kerdos-white: #fffeff;
        --kerdos-blue: #29abd2;
        --kerdos-light-gray: #f8f9fa;
        --kerdos-success: #28a745;
        --kerdos-warning: #fd7e14;
        --kerdos-danger: #dc3545;
    }
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--kerdos-dark), var(--kerdos-blue));
        color: var(--kerdos-white);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--kerdos-blue), var(--kerdos-dark));
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(15, 36, 56, 0.3);
    }
    
    .dataframe thead th {
        background: var(--kerdos-dark) !important;
        color: var(--kerdos-white) !important;
        border: 1px solid var(--kerdos-blue) !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
</style>
""", unsafe_allow_html=True)

# Define Kerdos color palette
KERDOS_COLORS = {
    'primary': '#0f2438',
    'secondary': '#29abd2',
    'white': '#fffeff',
    'light_gray': '#f8f9fa',
    'success': '#28a745',
    'warning': '#fd7e14',
    'danger': '#dc3545'
}

CHART_COLOR_SEQUENCE = [
    KERDOS_COLORS['primary'], 
    KERDOS_COLORS['secondary'], 
    KERDOS_COLORS['success'], 
    KERDOS_COLORS['warning'],
    '#6f42c1',  # Purple
    '#e83e8c',  # Pink
    '#20c997',  # Teal
]

# ============================================================================
# SNOWFLAKE CONNECTION
# ============================================================================

@st.cache_resource
def get_snowflake_connection():
    """Connect to Snowflake"""
    user = os.getenv("SNOWFLAKE_USER") or st.secrets.get("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD") or st.secrets.get("SNOWFLAKE_PASSWORD")
    account = os.getenv("SNOWFLAKE_ACCOUNT") or st.secrets.get("SNOWFLAKE_ACCOUNT")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE") or st.secrets.get("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE") or st.secrets.get("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA") or st.secrets.get("SNOWFLAKE_SCHEMA", "PRODUCTION")
    role = os.getenv("SNOWFLAKE_ROLE") or st.secrets.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN")

    return snowflake.connector.connect(
        user=user, password=password, account=account,
        warehouse=warehouse, database=database, schema=schema, role=role,
        client_session_keep_alive=True
    )


@st.cache_data(ttl="14400")
def load_strategy_performance():
    """Load performance data for all strategies"""
    conn = get_snowflake_connection()
    
    query = """
    SELECT 
        STRATEGY_NAME, TIMESTAMP, PORTFOLIO_VALUE, CASH, IS_OUT_OF_SAMPLE
    FROM STRATEGY_PERFORMANCE
    ORDER BY STRATEGY_NAME, TIMESTAMP
    """
    
    df = pd.read_sql(query, conn)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    return df


@st.cache_data(ttl="14400")
def load_latest_positions(strategy_name):
    """Load latest positions for a strategy"""
    conn = get_snowflake_connection()
    
    query = f"""
    WITH latest_timestamp AS (
        SELECT MAX(TIMESTAMP) as max_ts
        FROM STRATEGY_POSITIONS
        WHERE STRATEGY_NAME = '{strategy_name}'
    )
    SELECT 
        p.SYMBOL,
        p.MARKET_VALUE
    FROM STRATEGY_POSITIONS p
    JOIN latest_timestamp lt ON p.TIMESTAMP = lt.max_ts
    WHERE p.STRATEGY_NAME = '{strategy_name}'
    AND p.MARKET_VALUE > 0
    ORDER BY p.MARKET_VALUE DESC
    """
    
    df = pd.read_sql(query, conn)
    return df


# ============================================================================
# METRICS CALCULATIONS
# ============================================================================

def calculate_comprehensive_metrics(returns, portfolio_values, risk_free_rate=0.0358):
    """Calculate all performance metrics"""
    n_days = len(returns)
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    
    trading_days_per_year = 252
    years = n_days / trading_days_per_year
    cagr = (1 + total_return) ** (1 / years) - 1
    
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(trading_days_per_year)
    
    excess_returns = returns - (risk_free_rate / trading_days_per_year)
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days_per_year)
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_days_per_year)
    sortino = ((returns.mean() * trading_days_per_year - risk_free_rate) / downside_std) if downside_std > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    in_drawdown = drawdown < 0
    drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()[in_drawdown]
    longest_dd_days = drawdown_periods.value_counts().max() if len(drawdown_periods) > 0 else 0
    
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    win_days = (returns > 0).sum()
    win_rate = win_days / n_days if n_days > 0 else 0
    
    best_day = returns.max()
    worst_day = returns.min()
    
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    best_month = monthly_returns.max()
    worst_month = monthly_returns.min()
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Volatility (ann.)': annual_vol,
        'Calmar Ratio': calmar,
        'Recovery Factor': recovery_factor,
        'Win Days%': win_rate,
        'Best Day': best_day,
        'Worst Day': worst_day,
        'Best Month': best_month,
        'Worst Month': worst_month,
        'Longest DD Days': longest_dd_days,
    }


# ============================================================================
# PAGE: FUND OVERVIEW
# ============================================================================

def render_fund_overview(perf_df):
    """Render Fund Overview page with real allocation data"""
    
    # Logo
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

    #st.markdown("## 📊 Kerdos Fund Overview")
    
    # Fund thesis
    with st.container():
        st.markdown("### 🔍 Investment Thesis")
        st.markdown("""
        The Kerdos Fund leverages advanced machine learning techniques to optimize asset allocation, 
        aiming to minimize risk while maximizing returns. By dynamically adjusting portfolio weights 
        based on predictive analytics, the fund seeks to outperform traditional benchmarks with a 
        focus on risk-adjusted performance.
        """)
        
        st.markdown("### 💡 Strategy Highlights")
        st.markdown("""
        - Utilizes **XGBoost** for predictive modeling of asset returns and risks
        - Targets a portfolio volatility of **15%** through risk optimization  
        - Implements a **dynamic rebalancing strategy** based on market conditions
        - Incorporates both traditional assets (equities, commodities) and alternative assets (crypto)
        """)
    
    # Asset universe + Current allocation
    st.subheader("🌐 Asset Universe & Current Allocation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Core Holdings:")
        
        assets = [
            ("₿", "BTC-USD", "Bitcoin (Alternative Asset)"),
            ("🇺🇸", "SPY", "S&P 500 ETF (Equity Exposure)"),
            ("🥇", "GLD", "Gold ETF (Safe Haven Asset)"),
            ("🥈", "SLV", "Silver ETF (Safe Haven Asset)"),
            ("🖥️", "SMH", "Semiconductor ETF (AI Boom Exposure)"),
            ("⚡", "ZAP", "Electrification ETF (AI Infrastructure)")
        ]
        
        for emoji, symbol, description in assets:
            st.markdown(f"**{emoji} {symbol}** - {description}")
    
    with col2:
        # Get latest positions for ML strategy
        try:
            positions_df = load_latest_positions("ML_XGBOOST")
            
            # Get latest portfolio data
            ml_df = perf_df[perf_df['STRATEGY_NAME'] == 'ML_XGBOOST'].sort_values('TIMESTAMP')
            latest_cash = ml_df.iloc[-1]['CASH']
            latest_portfolio_value = ml_df.iloc[-1]['PORTFOLIO_VALUE']

            st.metric(
                "Portfolio Value",
                f"${latest_portfolio_value:,.2f}",
                f"{(latest_portfolio_value/10000 - 1)*100:.1f}%",
                help="Total portfolio value including all positions and cash"
            )
            
            # Build allocation data
            allocation_data = {}
            
            # Add positions
            for _, row in positions_df.iterrows():
                allocation_data[row['SYMBOL']] = row['MARKET_VALUE']
            
            # Add cash if > 0
            if latest_cash > 0:
                allocation_data['CASH'] = latest_cash
            
            if allocation_data:
                # Create pie chart
                fig_pie = px.pie(
                    values=list(allocation_data.values()),
                    names=list(allocation_data.keys()),
                    title="Current Portfolio Allocation",
                    color_discrete_sequence=CHART_COLOR_SEQUENCE
                )
                fig_pie.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=12,
                    textfont_color='white'
                )
                fig_pie.update_layout(
                    font=dict(family="Inter, sans-serif", size=12),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_color=KERDOS_COLORS['primary'],
                    title_font_size=16
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            else:
                st.warning("No position data available")
                
        except Exception as e:
            st.error(f"Could not load position data: {e}")
            # Fallback to equal weight visualization
            st.info("Showing example allocation")
    
    # Investment windows
    st.subheader("📅 Investment Opportunities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📅 Next Investment Window**
        
        **February 28, 2026**
        
        💰 Recommended Minimum Investment: $1000
        """)
    
    with col2:
        st.info("""
        **📅 Final Investment Window**
        
        **March 17, 2026**
        
        💰 Recommended Minimum Investment: $1000
        """)


# ============================================================================
# PAGE: PERFORMANCE & BENCHMARK COMPARISON
# ============================================================================

def render_performance_comparison(perf_df, risk_free_rate):
    """Render comprehensive performance comparison page"""
    
    st.title("📈 Performance Analysis & Benchmark Comparison")
    
    # Prepare data
    ml_df = perf_df[perf_df['STRATEGY_NAME'] == 'ML_XGBOOST'].copy()
    spy_df = perf_df[perf_df['STRATEGY_NAME'] == 'SPY_BENCHMARK'].copy()
    
    if ml_df.empty:
        st.error("No ML strategy data found")
        return
    
    ml_df = ml_df.sort_values('TIMESTAMP').set_index('TIMESTAMP')
    ml_returns = ml_df['PORTFOLIO_VALUE'].pct_change().dropna()
    
    spy_returns = None
    if not spy_df.empty:
        spy_df = spy_df.sort_values('TIMESTAMP').set_index('TIMESTAMP')
        spy_returns = spy_df['PORTFOLIO_VALUE'].pct_change().dropna()
    
    # Calculate metrics
    ml_metrics = calculate_comprehensive_metrics(ml_returns, ml_df['PORTFOLIO_VALUE'], risk_free_rate)
    spy_metrics = calculate_comprehensive_metrics(spy_returns, spy_df['PORTFOLIO_VALUE'], risk_free_rate) if spy_returns is not None else {}
    
    # ========================================================================
    # KEY METRICS
    # ========================================================================
    
    st.subheader("🔑 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_val = f"+{(ml_metrics['Total Return'] - spy_metrics.get('Total Return', 0))*100:.1f}% vs SPY" if spy_metrics else None
        st.metric(
            "Total Return",
            f"{ml_metrics['Total Return']*100:.1f}%",
            delta=delta_val
        )
    
    with col2:
        delta_val = f"+{ml_metrics['Sharpe Ratio'] - spy_metrics.get('Sharpe Ratio', 0):.2f}" if spy_metrics else None
        st.metric(
            "Sharpe Ratio",
            f"{ml_metrics['Sharpe Ratio']:.2f}",
            delta=delta_val
        )
    
    with col3:
        delta_val = f"+{(ml_metrics['CAGR'] - spy_metrics.get('CAGR', 0))*100:.1f}%" if spy_metrics else None
        st.metric(
            "CAGR",
            f"{ml_metrics['CAGR']*100:.1f}%",
            delta=delta_val
        )
    
    with col4:
        delta_val = f"{(ml_metrics['Max Drawdown'] - spy_metrics.get('Max Drawdown', 0))*100:.1f}%" if spy_metrics else None
        st.metric(
            "Max Drawdown",
            f"{ml_metrics['Max Drawdown']*100:.1f}%",
            delta=delta_val,
            delta_color="inverse"
        )
    
    # ========================================================================
    # PERFORMANCE CHART
    # ========================================================================
    
    st.subheader("📈 Cumulative Performance Comparison")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ml_df.index,
        y=ml_df['PORTFOLIO_VALUE'],
        mode='lines',
        name='Kerdos Fund',
        line=dict(color=KERDOS_COLORS['primary'], width=3)
    ))
    
    if not spy_df.empty:
        fig.add_trace(go.Scatter(
            x=spy_df.index,
            y=spy_df['PORTFOLIO_VALUE'],
            mode='lines',
            name='SPY Benchmark',
            line=dict(color=KERDOS_COLORS['secondary'], width=2, dash='dash')
        ))
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=500,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title="Cumulative Portfolio Value Over Time",
        title_font_color=KERDOS_COLORS['primary'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # DETAILED COMPARISON TABLE
    # ========================================================================
    
    st.subheader("📊 Detailed Performance Metrics")
    
    if spy_metrics:
        comparison_df = pd.DataFrame({
            'Metric': [
                'Total Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio',
                'Max Drawdown', 'Volatility (ann.)', 'Calmar Ratio', 'Win Rate',
                'Best Day', 'Worst Day', 'Best Month', 'Worst Month', 'Recovery Factor'
            ],
            'Kerdos Fund': [
                f"{ml_metrics['Total Return']*100:.1f}%",
                f"{ml_metrics['CAGR']*100:.1f}%",
                f"{ml_metrics['Sharpe Ratio']:.2f}",
                f"{ml_metrics['Sortino Ratio']:.2f}",
                f"{ml_metrics['Max Drawdown']*100:.1f}%",
                f"{ml_metrics['Volatility (ann.)']*100:.1f}%",
                f"{ml_metrics['Calmar Ratio']:.2f}",
                f"{ml_metrics['Win Days%']*100:.1f}%",
                f"{ml_metrics['Best Day']*100:.1f}%",
                f"{ml_metrics['Worst Day']*100:.1f}%",
                f"{ml_metrics['Best Month']*100:.1f}%",
                f"{ml_metrics['Worst Month']*100:.1f}%",
                f"{ml_metrics['Recovery Factor']:.2f}",
            ],
            'SPY Benchmark': [
                f"{spy_metrics['Total Return']*100:.1f}%",
                f"{spy_metrics['CAGR']*100:.1f}%",
                f"{spy_metrics['Sharpe Ratio']:.2f}",
                f"{spy_metrics['Sortino Ratio']:.2f}",
                f"{spy_metrics['Max Drawdown']*100:.1f}%",
                f"{spy_metrics['Volatility (ann.)']*100:.1f}%",
                f"{spy_metrics['Calmar Ratio']:.2f}",
                f"{spy_metrics['Win Days%']*100:.1f}%",
                f"{spy_metrics['Best Day']*100:.1f}%",
                f"{spy_metrics['Worst Day']*100:.1f}%",
                f"{spy_metrics['Best Month']*100:.1f}%",
                f"{spy_metrics['Worst Month']*100:.1f}%",
                f"{spy_metrics['Recovery Factor']:.2f}",
            ],
            'Outperformance': [
                f"+{(ml_metrics['Total Return'] - spy_metrics['Total Return'])*100:.1f}%",
                f"+{(ml_metrics['CAGR'] - spy_metrics['CAGR'])*100:.1f}%",
                f"+{ml_metrics['Sharpe Ratio'] - spy_metrics['Sharpe Ratio']:.2f}",
                f"+{ml_metrics['Sortino Ratio'] - spy_metrics['Sortino Ratio']:.2f}",
                f"{(ml_metrics['Max Drawdown'] - spy_metrics['Max Drawdown'])*100:.1f}%",
                f"+{(ml_metrics['Volatility (ann.)'] - spy_metrics['Volatility (ann.)'])*100:.1f}%",
                f"+{ml_metrics['Calmar Ratio'] - spy_metrics['Calmar Ratio']:.2f}",
                f"+{(ml_metrics['Win Days%'] - spy_metrics['Win Days%'])*100:.1f}%",
                f"{(ml_metrics['Best Day'] - spy_metrics['Best Day'])*100:.1f}%",
                f"{(ml_metrics['Worst Day'] - spy_metrics['Worst Day'])*100:.1f}%",
                f"{(ml_metrics['Best Month'] - spy_metrics['Best Month'])*100:.1f}%",
                f"{(ml_metrics['Worst Month'] - spy_metrics['Worst Month'])*100:.1f}%",
                f"+{ml_metrics['Recovery Factor'] - spy_metrics['Recovery Factor']:.2f}",
            ]
        })
    else:
        comparison_df = pd.DataFrame({
            'Metric': list(ml_metrics.keys()),
            'Kerdos Fund': [f"{v*100:.1f}%" if isinstance(v, float) and v < 10 else f"{v:.2f}" for v in ml_metrics.values()]
        })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # RISK-RETURN ANALYSIS
    # ========================================================================
    
    st.subheader("🎯 Risk-Return Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk-return scatter
        fig_scatter = go.Figure()
        
        # Collect all data points
        scatter_data = []
        
        if spy_metrics:
            scatter_data.append({
                'x': spy_metrics['Volatility (ann.)']*100,
                'y': spy_metrics['CAGR']*100,
                'name': 'SPY',
                'color': KERDOS_COLORS['secondary']
            })
            
            fig_scatter.add_trace(go.Scatter(
                x=[spy_metrics['Volatility (ann.)']*100],
                y=[spy_metrics['CAGR']*100],
                mode='markers+text',
                name='SPY Benchmark',
                marker=dict(
                    size=25, 
                    color=KERDOS_COLORS['secondary'],
                    line=dict(color='white', width=2)
                ),
                text=['SPY'],
                textposition="top center",
                textfont=dict(size=14, color=KERDOS_COLORS['secondary'], family='Inter')
            ))
        
        scatter_data.append({
            'x': ml_metrics['Volatility (ann.)']*100,
            'y': ml_metrics['CAGR']*100,
            'name': 'Kerdos',
            'color': KERDOS_COLORS['primary']
        })
        
        fig_scatter.add_trace(go.Scatter(
            x=[ml_metrics['Volatility (ann.)']*100],
            y=[ml_metrics['CAGR']*100],
            mode='markers+text',
            name='Kerdos Fund',
            marker=dict(
                size=25, 
                color=KERDOS_COLORS['primary'],
                line=dict(color='white', width=2)
            ),
            text=['Kerdos'],
            textposition="top center",
            textfont=dict(size=14, color=KERDOS_COLORS['primary'], family='Inter')
        ))
        
        # Calculate axis ranges with padding
        all_x = [d['x'] for d in scatter_data]
        all_y = [d['y'] for d in scatter_data]
        
        x_padding = (max(all_x) - min(all_x)) * 0.3 if len(all_x) > 1 else 5
        y_padding = (max(all_y) - min(all_y)) * 0.3 if len(all_y) > 1 else 10
        
        fig_scatter.update_layout(
            title='Risk vs Return Profile',
            xaxis=dict(
                title='Annualized Volatility (%)',
                range=[min(all_x) - x_padding, max(all_x) + x_padding],
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title='CAGR (%)',
                range=[min(all_y) - y_padding, max(all_y) + y_padding],
                gridcolor='lightgray',
                showgrid=True
            ),
            height=400,
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='rgba(248, 249, 250, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color=KERDOS_COLORS['primary'],
            showlegend=False
        )
        
        # Add diagonal line for Sharpe ratio reference if we have both points
        if len(scatter_data) > 1:
            # Add a reference line showing equal Sharpe ratio
            x_line = [0, max(all_x) + x_padding]
            sharpe_ref = ml_metrics['Sharpe Ratio']
            y_line = [0, sharpe_ref * (max(all_x) + x_padding)]
            
            fig_scatter.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=f'Sharpe {sharpe_ref:.2f}',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Key advantages
        st.markdown("### 🏆 Key Advantages")
        
        if spy_metrics:
            st.success(f"""
            **📈 Return Outperformance**
            - +{(ml_metrics['Total Return'] - spy_metrics['Total Return'])*100:.1f}% Total Return
            - +{(ml_metrics['CAGR'] - spy_metrics['CAGR'])*100:.1f}% CAGR
            """)
            
            st.info(f"""
            **⚡ Risk-Adjusted Performance**
            - {ml_metrics['Sharpe Ratio']:.2f} Sharpe ({ml_metrics['Sharpe Ratio'] - spy_metrics['Sharpe Ratio']:.2f} higher)
            - {ml_metrics['Sortino Ratio']:.2f} Sortino ({ml_metrics['Sortino Ratio'] - spy_metrics['Sortino Ratio']:.2f} higher)
            """)
            
            st.warning(f"""
            **🛡️ Risk Management**
            - {abs(ml_metrics['Max Drawdown'])*100:.1f}% Max Drawdown
            - {ml_metrics['Recovery Factor']:.2f}x Recovery Factor
            """)
        else:
            st.info("Load SPY benchmark data to see comparison")
    
    # ========================================================================
    # DRAWDOWN CHART
    # ========================================================================
    
    st.subheader("📉 Drawdown Analysis")
    
    ml_cumulative = (1 + ml_returns).cumprod()
    ml_running_max = ml_cumulative.expanding().max()
    ml_drawdown = (ml_cumulative - ml_running_max) / ml_running_max * 100
    
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=ml_drawdown.index,
        y=ml_drawdown,
        fill='tozeroy',
        line=dict(color=KERDOS_COLORS['danger'], width=2),
        fillcolor=f'rgba({int(KERDOS_COLORS["danger"][1:3], 16)}, {int(KERDOS_COLORS["danger"][3:5], 16)}, {int(KERDOS_COLORS["danger"][5:7], 16)}, 0.2)',
        name='Kerdos Fund'
    ))
    
    if spy_returns is not None:
        spy_cumulative = (1 + spy_returns).cumprod()
        spy_running_max = spy_cumulative.expanding().max()
        spy_drawdown = (spy_cumulative - spy_running_max) / spy_running_max * 100
        
        fig_dd.add_trace(go.Scatter(
            x=spy_drawdown.index,
            y=spy_drawdown,
            line=dict(color=KERDOS_COLORS['secondary'], width=2, dash='dash'),
            name='SPY Benchmark'
        ))
    
    fig_dd.update_layout(
        title="Drawdown from Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=400,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color=KERDOS_COLORS['primary']
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Fund Overview", "Performance & Benchmark"],
        help="Navigate through different sections"
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📞 Contact Information
    **Fund Managers** \n 
    Kerdos Team \n
    **Next Investment Window** \n
    Feb 28, 2026 \n
    **Final Window** \n 
    Mar 17, 2026
    """)
    
    # Risk-free rate setting
    st.sidebar.markdown("---")
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.58,
        step=0.01
    ) / 100
    
    # Load data
    with st.spinner("Loading data from Snowflake..."):
        try:
            perf_df = load_strategy_performance()
            
            if perf_df.empty:
                st.error("No data found in Snowflake. Please run the backtest first.")
                st.info("Run: `python run_dual_backtest.py`")
                return
            
            st.sidebar.success(f"✅ Loaded {len(perf_df)} records")
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Route to pages
    if page == "Fund Overview":
        render_fund_overview(perf_df)
    else:
        render_performance_comparison(perf_df, risk_free_rate)


if __name__ == "__main__":
    main()