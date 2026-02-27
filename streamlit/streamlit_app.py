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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --kerdos-dark: #0f2438;
        --kerdos-white: #fffeff;
        --kerdos-blue: #29abd2;
        --kerdos-light-gray: #f1e7cb;  /* Updated to your requested background */
        --kerdos-success: #28a745;
        --kerdos-warning: #fd7e14;
        --kerdos-danger: #dc3545;
        --kerdos-accent: #5e17eb;     /* Updated to your requested purple */
        --kerdos-pink: #e83e8c;
        --kerdos-teal: #20c997;
        --kerdos-text: #716037;       /* New: your requested text color */
    }
    
    .main {
        font-family: 'Inter', sans-serif;
        background: #f1e7cb !important;  /* Updated background */
        color: #716037 !important;       /* Updated text color */
    }
    
    /* Update general text color */
    .stApp {
        background: #f1e7cb !important;
        color: #716037 !important;
    }
    
    /* Update text elements */
    p, span, div, label {
        color: #716037 !important;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--kerdos-dark), var(--kerdos-blue));
        color: var(--kerdos-white);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.025em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 4px rgba(15, 36, 56, 0.15);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--kerdos-blue), var(--kerdos-dark));
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(15, 36, 56, 0.25);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(15, 36, 56, 0.2);
    }
    
    /* Enhanced dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: none !important;
        background: white !important;
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, var(--kerdos-dark), var(--kerdos-blue)) !important;
        color: var(--kerdos-white) !important;
        border: none !important;
        padding: 12px 16px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid #e9ecef !important;
        transition: background-color 0.2s ease;
        background: white !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: rgba(241, 231, 203, 0.3) !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(41, 171, 210, 0.1) !important;
    }
    
    .dataframe tbody td {
        padding: 12px 16px !important;
        font-size: 13px !important;
        border: none !important;
        color: #716037 !important;
    }
    
    /* Metric cards enhancement */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(94, 23, 235, 0.2);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        border-color: var(--kerdos-accent);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--kerdos-dark) 0%, #1a3a52 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: var(--kerdos-white);
    }
    
    /* Info/warning/success boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1), rgba(40, 167, 69, 0.05));
        border-left: 4px solid var(--kerdos-success);
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid rgba(94, 23, 235, 0.3);
        transition: border-color 0.3s ease;
        background: white !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--kerdos-accent);
        box-shadow: 0 0 0 3px rgba(94, 23, 235, 0.1);
    }
    
    /* Custom progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--kerdos-accent), var(--kerdos-success));
    }
    
    /* Title styling with your purple */
    h1 {
        color: var(--kerdos-accent) !important;  /* Purple for main titles */
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    
    h2 {
        color: var(--kerdos-accent) !important;  /* Purple for section headers */
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    
    h3 {
        color: var(--kerdos-dark) !important;    /* Dark blue for subsections */
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--kerdos-light-gray);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--kerdos-accent), var(--kerdos-dark));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--kerdos-dark), var(--kerdos-accent));
    }
    
    /* Update markdown text */
    .markdown-text-container {
        color: #716037 !important;
    }
    
    /* Update all text in main content area */
    .main .block-container {
        color: #716037 !important;
    }
    
    /* Ensure metric values use your text color */
    [data-testid="metric-container"] [data-testid="metric-container"] {
        color: #716037 !important;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Kerdos color palette with your requested colors
KERDOS_COLORS = {
    'primary': '#0f2438',        # Deep navy - primary brand color
    'secondary': '#29abd2',      # Bright blue - secondary brand color
    'white': '#fffeff',          # Pure white with subtle warmth
    'light_gray': '#f1e7cb',     # Your requested background color
    'success': '#28a745',        # Success green
    'warning': '#fd7e14',        # Warning orange
    'danger': '#dc3545',         # Danger red
    'accent': '#5e17eb',         # Your requested purple accent
    'pink': '#e83e8c',           # Pink accent
    'teal': '#20c997',           # Teal accent
    'dark_gray': '#6c757d',      # Text gray
    'border': '#dee2e6',         # Border gray
    'text': '#716037',           # Your requested text color
}

# Enhanced chart color sequence with your new colors
CHART_COLOR_SEQUENCE = [
    KERDOS_COLORS['primary'],     # Dark navy - most important
    KERDOS_COLORS['accent'],      # Purple - your accent color
    KERDOS_COLORS['secondary'],   # Bright blue - secondary
    KERDOS_COLORS['success'],     # Green - positive
    KERDOS_COLORS['warning'],     # Orange - attention
    KERDOS_COLORS['teal'],        # Teal - cool accent
    KERDOS_COLORS['pink'],        # Pink - warm accent
    KERDOS_COLORS['danger'],      # Red - negative/risk
    '#8e44ad',                    # Deep purple
    '#2c3e50',                    # Dark blue-gray
]

competition_start = datetime(2026, 2, 28)
competition_end = datetime(2026, 4, 17)
current_date = datetime.now()

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
def load_strategy_performance(mode="historical"):
    """Load performance data - historical or competition mode"""
    conn = get_snowflake_connection()
    if mode == "competition":
        # Competition tables - Feb 28 onwards with $10k start
        query = """
        SELECT 
            STRATEGY_NAME, TIMESTAMP, PORTFOLIO_VALUE, CASH
        FROM COMPETITION_PERFORMANCE
        ORDER BY STRATEGY_NAME, TIMESTAMP
        """
    else:
        # Historical tables - full backtest
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
def load_latest_positions(strategy_name, mode="historical"):
    """Load latest positions - historical or competition mode"""
    conn = get_snowflake_connection()
    table = "COMPETITION_POSITIONS" if mode == "competition" else "STRATEGY_POSITIONS"

    query = f"""
    WITH latest_timestamp AS (
        SELECT MAX(TIMESTAMP) as max_ts
        FROM {table}
        WHERE STRATEGY_NAME = '{strategy_name}'
    )
    SELECT 
        p.SYMBOL,
        p.MARKET_VALUE
    FROM {table} p
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
# COMPETITION STATUS
# ============================================================================
def get_competition_status():
        """Get current competition status and messaging"""
        
        if current_date < competition_start:
            days_until = (competition_start - current_date).days
            return {
                'status': 'preview',
                'message': f"Competition starts in {days_until} days",
                'color': 'info'
            }
        elif current_date <= competition_end:
            days_running = (current_date - competition_start).days
            total_days = (competition_end - competition_start).days
            return {
                'status': 'active',
                'message': f"Competition active - Day {days_running} of {total_days}",
                'color': 'success'
            }
        else:
            return {
                'status': 'completed',
                'message': "Competition completed",
                'color': 'warning'
            }


# ============================================================================
# PAGE: FUND OVERVIEW
# ============================================================================


def render_fund_overview(perf_df, mode="historical"):
    """Render Fund Overview page with real allocation data"""
    if mode == "competition":
        if current_date < competition_start:
            st.info(f"""
            🏆 **Competition Preview Mode** — Competition starts Feb 28, 2026  
            Portfolio starts with $10,000 cash · Days until start: **{(competition_start - current_date).days}**
            """)
        else:
            st.info("""
            🏆 **Competition Mode** — Viewing live performance from Feb 28, 2026 onwards  
            Starting capital: $10,000 · End date: Apr 17, 2026 · Updates daily
            """)
    
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
        The **Kerdos Fund** uses a three-layer machine learning architecture to allocate capital 
        dynamically across seven complementary assets. Rather than chasing returns, we predict 
        *risk* — using XGBoost to forecast next-day volatility per asset, then allocating 
        inversely proportional to that risk. A direction classifier provides a soft bullish tilt,
        and a regime filter scales down gross exposure when market-wide volatility spikes.""")
        
        st.markdown("### 💡 Strategy Highlights (Jan 1, 2024 - Feb 26, 2026)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", "118.3%", "+74.8% vs SPY")
            st.metric("Sharpe Ratio", "1.91", "+0.99 vs SPY")
        with col2:
            st.metric("CAGR", "44.0%", "+25.7pp vs SPY")
            st.metric("Max Drawdown", "-11.7%", "Better by 7.4pp")
        with col3:
            st.metric("Calmar Ratio", "3.75", "+2.75pp vs SPY")
            st.metric("Corr. to SPY", "0.06", "Near-zero dependency")

        st.markdown("---")
        st.markdown("""
        **How it works:**
        - **Volatility model** (XGBoost Regressor, Bayesian-tuned): predicts next-day realised volatility 
        using a 90-day rolling window of price, momentum, RSI, Bollinger Band, and cross-asset features
        - **Direction classifier** (XGBoost Classifier, Bayesian-tuned): predicts P(return > 0) and applies a soft 
        tilt — bullish assets receive proportionally higher weights
        - **Regime filter**: monitors SPY realised vol; scales to 100% (CALM), 50–100% (CAUTION), 
        or 30% (FEAR) gross exposure to protect capital during market stress
        """)
    
    # Asset universe + Current allocation
    st.subheader("🌐 Asset Universe & Current Allocation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Core Holdings:")

        assets = [
            ("btc.png",  "BTC-USD", "Bitcoin",           "Crypto / Alternative — 24/7 liquidity, high-beta"),
            ("spy.png", "SPY",     "S&P 500 ETF",        "US Large Cap Equity — core benchmark exposure"),
            ("gld.png",  "GLD",     "Gold ETF",           "Safe Haven — crisis hedge, negative equity corr."),
            ("slv.png",  "SLV",     "Silver ETF",         "Safe Haven — amplified metals, higher vol than gold"),
            ("smh.png",  "SMH",     "Semiconductor ETF",  "AI / Tech — semiconductor cycle + AI boom"),
            ("zap.png",  "ZAP",     "Electrification ETF","AI Infrastructure — energy transition theme"),
            ("dfen.png",  "DFEN",    "Defense ETF",        "Geopolitical Hedge — low equity correlation"),
        ]
        
        # Create HTML with images
        assets_html = ""
        for img_file, symbol, description, details in assets:
            img_path = os.path.join(os.path.dirname(__file__), 'assets', img_file)
            
            if os.path.exists(img_path):
                # Convert image to base64 for embedding
                import base64
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                img_html = f'<img src="data:image/png;base64,{img_data}" style="width: 36px; height: 36px; vertical-align: top; margin-right: 8px;">'
            else:
                img_html = f'<span style="font-weight: bold; margin-right: 8px;">{symbol[0]}</span>'
            
            assets_html += f"""
            <div style="margin-bottom: 12px;">
                {img_html}
                <strong>{symbol}</strong> - {description}
                <br>
                <span style="color: #716037; font-size: 0.85em; margin-left: 32px;">{details}</span>
            </div>
            """
        
        st.markdown(assets_html, unsafe_allow_html=True)
    
    with col2:
        # Get latest positions for ML strategy
        try:
            # Competition preview mode - show $10k cash
            if mode == "competition" and current_date < competition_start:
                st.metric(
                    "Portfolio Value",
                    "$10,000.00",
                    "0.0%",
                    help="Starting portfolio value for competition (100% cash until Feb 28, 2026)"
                )
                
                # Show 100% cash allocation
                allocation_data = {'CASH': 10000}
                
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
                    textfont_size=16,
                    textfont_color='white'
                )
                fig_pie.update_layout(
                    font=dict(family="Inter, sans-serif", size=12, color=KERDOS_COLORS['text']),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_color=KERDOS_COLORS['text'],
                    title_font_size=16
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            else:
                # Normal mode - load actual positions
                positions_df = load_latest_positions("ML_XGBOOST", mode=mode)
                
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
                        font=dict(family="Inter, sans-serif", size=12, color=KERDOS_COLORS['text']),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font_color=KERDOS_COLORS['text'],
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
        
        **March 20, 2026**
        
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
        font=dict(family="Inter, sans-serif", color=KERDOS_COLORS['text']),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title="Cumulative Portfolio Value Over Time",
        title_font_color=KERDOS_COLORS['text'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(color=KERDOS_COLORS['text']))
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
                showgrid=True,
                title_font=dict(color=KERDOS_COLORS['text']),
                tickfont=dict(color=KERDOS_COLORS['text'])
            ),
            yaxis=dict(
                title='CAGR (%)',
                range=[min(all_y) - y_padding, max(all_y) + y_padding],
                gridcolor='lightgray',
                showgrid=True,
                title_font=dict(color=KERDOS_COLORS['text']),
                tickfont=dict(color=KERDOS_COLORS['text'])
            ),
            height=400,
            font=dict(family="Inter, sans-serif", color=KERDOS_COLORS['text']),
            plot_bgcolor='rgba(248, 249, 250, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color=KERDOS_COLORS['text'],
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
        font=dict(family="Inter, sans-serif", color=KERDOS_COLORS['text']),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color=KERDOS_COLORS['text'],
        xaxis=dict(title_font=dict(color=KERDOS_COLORS['text']), tickfont=dict(color=KERDOS_COLORS['text'])),
        yaxis=dict(title_font=dict(color=KERDOS_COLORS['text']), tickfont=dict(color=KERDOS_COLORS['text'])),
        legend=dict(font=dict(color=KERDOS_COLORS['text']))
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)

# ============================================================================
# PAGE: PAPER TRADING PERFORMANCE
# ============================================================================
def render_trading_performance(perf_df, risk_free_rate, starting_capital, start_date, end_date):
    """Render paper trading performance page with custom date range and capital"""
    
    st.title("📝 Paper Trading Performance")
    
    # Filter data for ML strategy and date range
    ml_df = perf_df[perf_df['STRATEGY_NAME'] == 'ML_XGBOOST'].copy()
    ml_df = ml_df.sort_values('TIMESTAMP')
    ml_df = ml_df[(ml_df['TIMESTAMP'] >= pd.to_datetime(start_date)) & (ml_df['TIMESTAMP'] <= pd.to_datetime(end_date))]
    
    if ml_df.empty:
        st.error("No data found for the selected date range.")
        return
    
    # Adjust portfolio values to starting capital
    initial_value = ml_df.iloc[0]['PORTFOLIO_VALUE']
    ml_df['ADJUSTED_PORTFOLIO_VALUE'] = ml_df['PORTFOLIO_VALUE'] / initial_value * starting_capital
    
    # Calculate returns
    ml_returns = ml_df['ADJUSTED_PORTFOLIO_VALUE'].pct_change().dropna()
    
    # Calculate metrics
    ml_metrics = calculate_comprehensive_metrics(ml_returns, ml_df['ADJUSTED_PORTFOLIO_VALUE'], risk_free_rate)
    
    # Display key metrics
    st.subheader("🔑 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{ml_metrics['Total Return']*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "CAGR",
            f"{ml_metrics['CAGR']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{ml_metrics['Sharpe Ratio']:.2f}"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{ml_metrics['Max Drawdown']*100:.1f}%",
            delta_color="inverse"
        )
    
    # Performance chart
    st.subheader("📈 Cumulative Performance")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ml_df['TIMESTAMP'],
        y=ml_df['ADJUSTED_PORTFOLIO_VALUE'],
        mode='lines',
        name='Kerdos Fund (Paper Trading)',
        line=dict(color=KERDOS_COLORS['primary'], width=3)
    ))

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
    "Choose a page:",
    ["Fund Overview", "Performance & Benchmark"], #, "Paper Trading Performance"],
    help="Navigate through different sections"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data View")
    view_mode = st.sidebar.radio(
        "Select view:",
        ["Historical (2024-2026)", "Competition (Feb 28 onwards)"],
        help="Switch between full historical backtest and competition window"
    )
    mode = "competition" if "Competition" in view_mode else "historical"

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📞 Contact Information
    **Fund Managers** \n 
    Kerdos Team \n
    **Next Investment Window** \n
    Feb 28, 2026 \n
    **Final Window** \n 
    Mar 20, 2026
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

    competition_start = datetime(2026, 2, 28)
    current_date = datetime.now()
    
    # Add competition status to sidebar
    if "Competition" in view_mode:
        status = get_competition_status()
        if status['color'] == 'info':
            st.sidebar.info(f"🕐 {status['message']}")
        elif status['color'] == 'success':
            st.sidebar.success(f"🏁 {status['message']}")
        else:
            st.sidebar.warning(f"🏆 {status['message']}")

    if mode == "competition" and current_date < competition_start:
        # Show preview message for competition mode before start date
        st.info(f"""
        🏆 **Competition Preview Mode**
        
        The competition starts on **February 28, 2026**. 
        Currently showing historical data for preview.
        
        Days until competition start: **{(competition_start - current_date).days}**
        """)
        # Fall back to historical mode for data loading
        mode = "historical"
    
    # Load data
    with st.spinner("Loading data from Snowflake..."):
        try:
            perf_df = load_strategy_performance(mode=mode)
            
            if perf_df.empty:
                if mode == "competition":
                    st.error("No competition data found. Run: `python run_competition_backtest.py`")
                else:
                    st.error("No historical data found. Run: `python run_dual_backtest.py`")
                return
            
            st.sidebar.success(f"✅ Loaded {len(perf_df)} records ({mode})")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Route to pages
    if page == "Fund Overview":
        # Pass the original view mode for display purposes
        original_mode = "competition" if "Competition" in view_mode else "historical"
        render_fund_overview(perf_df, mode=original_mode)
    #elif page == "Paper Trading Performance":
    #    # expand render_fund_overview to support trading with custom start and end dates
    #    # with a starting capital of 10000 USD
    #    starting_capital = st.input("Starting Capital (USD):", value=10000)
    #    start_date = st.date_input("Start Date:", value=datetime(2024, 1, 1))
    #    end_date = st.date_input("End Date:", value=datetime.now() - timedelta(days=1))
    #    render_trading_performance(perf_df, risk_free_rate, starting_capital, start_date, end_date)
    else:
        render_performance_comparison(perf_df, risk_free_rate)

if __name__ == "__main__":
    main()