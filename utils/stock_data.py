import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

class StockDataFetcher:
    """Class to handle stock data fetching from Yahoo Finance"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbol, period="1y"):
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock ticker symbol
            period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            pandas.DataFrame: Historical stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
                return None
            
            # Clean the data
            data = data.dropna()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                st.error("Incomplete data received from Yahoo Finance")
                return None
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @st.cache_data(ttl=300)
    def get_stock_info(_self, symbol):
        """
        Fetch stock information and key metrics
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            key_info = {
                'symbol': symbol,
                'longName': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap', None),
                'trailingPE': info.get('trailingPE', None),
                'forwardPE': info.get('forwardPE', None),
                'priceToBook': info.get('priceToBook', None),
                'dividendYield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', None),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', None),
                'averageVolume': info.get('averageVolume', None),
                'regularMarketVolume': info.get('regularMarketVolume', None)
            }
            
            return key_info
            
        except Exception as e:
            st.warning(f"Could not fetch detailed info for {symbol}: {str(e)}")
            return {}
    
    def get_financial_statements(self, symbol):
        """
        Fetch financial statements (income statement, balance sheet, cash flow)
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            dict: Financial statements
        """
        try:
            ticker = yf.Ticker(symbol)
            
            financials = {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow
            }
            
            return financials
            
        except Exception as e:
            st.warning(f"Could not fetch financial statements for {symbol}: {str(e)}")
            return {}
    
    def get_analyst_recommendations(self, symbol):
        """
        Fetch analyst recommendations and price targets
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            pandas.DataFrame: Analyst recommendations
        """
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations is not None and not recommendations.empty:
                # Get the most recent recommendations
                recent_recommendations = recommendations.tail(10)
                return recent_recommendations
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Could not fetch analyst recommendations for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def validate_symbol(self, symbol):
        """
        Validate if a stock symbol exists
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid data
            if 'symbol' in info or 'shortName' in info:
                return True
            else:
                return False
                
        except Exception:
            return False
    
    def get_real_time_price(self, symbol):
        """
        Get real-time price data
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            dict: Real-time price information
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get the most recent data
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                latest = data.iloc[-1]
                
                return {
                    'price': latest['Close'],
                    'volume': latest['Volume'],
                    'timestamp': data.index[-1]
                }
            else:
                return None
                
        except Exception as e:
            st.warning(f"Could not fetch real-time data for {symbol}: {str(e)}")
            return None
