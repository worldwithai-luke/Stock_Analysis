import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from utils.stock_data import StockDataFetcher
from utils.predictions import StockPredictor
import io

# Configure page
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'stock_info' not in st.session_state:
    st.session_state.stock_info = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def main():
    st.title("📈 Stock Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Stock Selection")
        
        # Stock symbol input
        stock_symbol = st.text_input(
            "Enter Stock Symbol:",
            placeholder="e.g., AAPL, GOOGL, MSFT",
            help="Enter a valid stock ticker symbol"
        ).upper()
        
        # Time period selection
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        
        selected_period = st.selectbox(
            "Select Time Period:",
            options=list(period_options.keys()),
            index=3  # Default to 1 Year
        )
        
        period = period_options[selected_period]
        
        # Fetch data button
        fetch_button = st.button("Fetch Stock Data", type="primary")
        
        # Prediction settings
        st.header("Prediction Settings")
        
        # Prediction period selection
        prediction_period_options = {
            "1 Week": 7,
            "2 Weeks": 14,
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "3 Years": 1095,
            "5 Years": 1825
        }
        
        selected_prediction_period = st.selectbox(
            "Prediction Period:",
            options=list(prediction_period_options.keys()),
            index=2,  # Default to 1 Month
            help="Select how far into the future to predict"
        )
        
        prediction_days = prediction_period_options[selected_prediction_period]
        
        ma_short = st.slider("Short MA Period:", 5, 50, 10)
        ma_long = st.slider("Long MA Period:", 20, 200, 50)
    
    # Main content area
    if stock_symbol and fetch_button:
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            try:
                # Initialize data fetcher
                data_fetcher = StockDataFetcher()
                
                # Fetch stock data
                stock_data = data_fetcher.get_stock_data(stock_symbol, period)
                stock_info = data_fetcher.get_stock_info(stock_symbol)
                
                if stock_data is not None and not stock_data.empty:
                    st.session_state.stock_data = stock_data
                    st.session_state.stock_info = stock_info
                    st.session_state.stock_symbol = stock_symbol
                    st.session_state.selected_prediction_period = selected_prediction_period
                    
                    # Generate predictions
                    predictor = StockPredictor()
                    predictions = predictor.predict_prices(
                        stock_data, 
                        days=prediction_days,
                        ma_short=ma_short,
                        ma_long=ma_long
                    )
                    st.session_state.predictions = predictions
                    
                    st.success(f"Data loaded successfully for {stock_symbol}!")
                else:
                    st.error(f"No data found for symbol: {stock_symbol}")
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.info("Please check if the stock symbol is correct and try again.")
    
    # Display data if available
    if st.session_state.stock_data is not None:
        display_stock_analysis()

def display_stock_analysis():
    stock_data = st.session_state.stock_data
    stock_info = st.session_state.stock_info
    predictions = st.session_state.predictions
    stock_symbol = st.session_state.stock_symbol
    selected_prediction_period = st.session_state.get('selected_prediction_period', 'Unknown Period')
    
    # Stock info header
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = stock_data['Close'].iloc[-1]
    prev_price = stock_data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change_pct:.2f}%"
        )
    
    with col2:
        if stock_info and 'marketCap' in stock_info:
            market_cap = stock_info['marketCap']
            if market_cap > 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            else:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            st.metric("Market Cap", market_cap_str)
        else:
            st.metric("Market Cap", "N/A")
    
    with col3:
        if stock_info and 'trailingPE' in stock_info:
            pe_ratio = stock_info['trailingPE']
            st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        else:
            st.metric("P/E Ratio", "N/A")
    
    with col4:
        avg_volume = stock_data['Volume'].tail(30).mean()
        if avg_volume > 1e6:
            volume_str = f"{avg_volume/1e6:.2f}M"
        elif avg_volume > 1e3:
            volume_str = f"{avg_volume/1e3:.2f}K"
        else:
            volume_str = f"{avg_volume:.0f}"
        st.metric("Avg Volume (30d)", volume_str)
    
    st.markdown("---")
    
    # Charts section
    st.header("📊 Interactive Charts")
    
    # Price chart with predictions
    fig_price = create_price_chart(stock_data, predictions, stock_symbol)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Volume chart
    fig_volume = create_volume_chart(stock_data, stock_symbol)
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Technical indicators chart
    fig_technical = create_technical_chart(stock_data, stock_symbol)
    st.plotly_chart(fig_technical, use_container_width=True)
    
    st.markdown("---")
    
    # Data table and download section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("📋 Historical Data")
        
        # Display recent data
        display_data = stock_data.tail(20).copy()
        display_data = display_data.round(2)
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_data,
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.header("💾 Export Data")
        
        # CSV download
        csv_buffer = io.StringIO()
        stock_data.to_csv(csv_buffer)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"{stock_symbol}_stock_data.csv",
            mime="text/csv"
        )
        
        # Predictions download
        if predictions is not None:
            pred_csv = io.StringIO()
            predictions.to_csv(pred_csv)
            pred_data = pred_csv.getvalue()
            
            st.download_button(
                label="🔮 Download Predictions",
                data=pred_data,
                file_name=f"{stock_symbol}_predictions.csv",
                mime="text/csv"
            )
    
    # Predictions section
    if predictions is not None and not predictions.empty:
        st.markdown("---")
        st.header("🔮 Price Predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Predicted Prices ({selected_prediction_period})")
            pred_display = predictions.copy()
            
            # For longer periods, show only recent and future key dates
            if len(pred_display) > 20:
                # Show first 10 and last 10 predictions
                display_df = pd.concat([
                    pred_display.head(10),
                    pred_display.tail(10)
                ])
                display_df = display_df.copy()
                display_df.index = [date.strftime('%Y-%m-%d') for date in display_df.index]
                st.dataframe(display_df.round(2), use_container_width=True)
                st.info(f"Showing first 10 and last 10 predictions out of {len(pred_display)} total predictions")
            else:
                pred_display = pred_display.copy()
                pred_display.index = [date.strftime('%Y-%m-%d') for date in pred_display.index]
                st.dataframe(pred_display.round(2), use_container_width=True)
        
        with col2:
            st.subheader("Prediction Summary")
            current_price = stock_data['Close'].iloc[-1]
            avg_prediction = predictions['Predicted_Price'].mean()
            prediction_change = ((avg_prediction - current_price) / current_price) * 100
            
            st.metric(
                "Avg Predicted Price",
                f"${avg_prediction:.2f}",
                f"{prediction_change:.2f}%"
            )
            
            st.info(
                "⚠️ **Disclaimer**: These predictions are based on simple technical indicators "
                "and should not be used as the sole basis for investment decisions. "
                "Always consult with financial professionals."
            )

def create_price_chart(stock_data, predictions, symbol):
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"{symbol} Stock Price with Predictions"]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        )
    )
    
    # Add moving averages
    if len(stock_data) >= 50:
        ma_20 = stock_data['Close'].rolling(window=20).mean()
        ma_50 = stock_data['Close'].rolling(window=50).mean()
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=ma_20,
                name="MA 20",
                line=dict(color='orange', width=1)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=ma_50,
                name="MA 50",
                line=dict(color='purple', width=1)
            )
        )
    
    # Add predictions
    if predictions is not None and not predictions.empty:
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['Predicted_Price'],
                mode='lines+markers',
                name="Predictions",
                line=dict(color='cyan', width=2, dash='dash'),
                marker=dict(size=6)
            )
        )
    
    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig

def create_volume_chart(stock_data, symbol):
    fig = go.Figure()
    
    colors = ['red' if close < open else 'green' 
              for close, open in zip(stock_data['Close'], stock_data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        )
    )
    
    fig.update_layout(
        title=f"{symbol} Trading Volume",
        height=300,
        xaxis_title="Date",
        yaxis_title="Volume",
        showlegend=False
    )
    
    return fig

def create_technical_chart(stock_data, symbol):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['RSI (Relative Strength Index)', 'MACD'],
        vertical_spacing=0.1
    )
    
    # Calculate RSI
    rsi = calculate_rsi(stock_data['Close'])
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=rsi,
            name="RSI",
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add RSI reference lines
    fig.add_shape(type="line", x0=stock_data.index[0], y0=70, x1=stock_data.index[-1], y1=70,
                  line=dict(dash="dash", color="red"), row=1, col=1)
    fig.add_shape(type="line", x0=stock_data.index[0], y0=30, x1=stock_data.index[-1], y1=30,
                  line=dict(dash="dash", color="green"), row=1, col=1)
    
    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(stock_data['Close'])
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=macd_line,
            name="MACD",
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=signal_line,
            name="Signal",
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True
    )
    
    return fig

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

if __name__ == "__main__":
    main()
