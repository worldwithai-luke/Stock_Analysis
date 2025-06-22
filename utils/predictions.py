import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """Class to handle stock price predictions using technical indicators"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def predict_prices(self, stock_data, days=10, ma_short=10, ma_long=50):
        """
        Predict future stock prices using moving averages and linear regression
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            days (int): Number of days to predict
            ma_short (int): Short moving average period
            ma_long (int): Long moving average period
        
        Returns:
            pd.DataFrame: Predicted prices with dates
        """
        try:
            # Calculate technical indicators
            features_df = self._calculate_features(stock_data, ma_short, ma_long)
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            if len(features_df) < 30:  # Need sufficient data
                return None
            
            # Prepare features and target
            feature_columns = [col for col in features_df.columns if col != 'Close']
            X = features_df[feature_columns].values
            y = features_df['Close'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model on recent data (last 100 points for better relevance)
            recent_data_points = min(100, len(X_scaled))
            X_train = X_scaled[-recent_data_points:]
            y_train = y[-recent_data_points:]
            
            self.model.fit(X_train, y_train)
            
            # Generate predictions
            predictions = self._generate_future_predictions(
                features_df, days, ma_short, ma_long
            )
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
    
    def _calculate_features(self, stock_data, ma_short, ma_long):
        """Calculate technical indicators as features"""
        df = stock_data.copy()
        
        # Moving averages
        df[f'MA_{ma_short}'] = df['Close'].rolling(window=ma_short).mean()
        df[f'MA_{ma_long}'] = df['Close'].rolling(window=ma_long).mean()
        
        # Price relative to moving averages
        df['Price_to_MA_Short'] = df['Close'] / df[f'MA_{ma_short}']
        df['Price_to_MA_Long'] = df['Close'] / df[f'MA_{ma_long}']
        
        # Volatility (rolling standard deviation)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price change and momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(df['Close'])
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        macd_line, signal_line = self._calculate_macd(df['Close'])
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Histogram'] = macd_line - signal_line
        
        # Select features for prediction
        feature_columns = [
            f'MA_{ma_short}', f'MA_{ma_long}',
            'Price_to_MA_Short', 'Price_to_MA_Long',
            'Volatility', 'Price_Change', 'Price_Change_5d', 'Momentum',
            'Volume_Ratio', 'RSI', 'BB_Position',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'Close'
        ]
        
        return df[feature_columns]
    
    def _generate_future_predictions(self, features_df, days, ma_short, ma_long):
        """Generate future price predictions"""
        last_date = features_df.index[-1]
        predictions = []
        prediction_dates = []
        
        # Get the last known features
        last_features = features_df.iloc[-1].copy()
        
        for i in range(days):
            # Predict next day
            feature_values = last_features.drop('Close').values.reshape(1, -1)
            feature_values_scaled = self.scaler.transform(feature_values)
            
            predicted_price = self.model.predict(feature_values_scaled)[0]
            
            # Add some randomness to make predictions more realistic
            # Based on historical volatility
            volatility = features_df['Volatility'].iloc[-20:].mean()
            if not np.isnan(volatility):
                noise = np.random.normal(0, volatility * 0.1)
                predicted_price += noise
            
            predictions.append(predicted_price)
            
            # Calculate next date (skip weekends)
            next_date = last_date + timedelta(days=i+1)
            while next_date.weekday() >= 5:  # Skip weekends
                next_date += timedelta(days=1)
            prediction_dates.append(next_date)
            
            # Update features for next prediction (simplified approach)
            last_features['Close'] = predicted_price
            
            # Update moving averages (approximate)
            if i == 0:
                recent_prices = list(features_df['Close'].tail(ma_short-1)) + [predicted_price]
            else:
                recent_prices = recent_prices[-(ma_short-1):] + [predicted_price]
            
            if len(recent_prices) >= ma_short:
                last_features[f'MA_{ma_short}'] = np.mean(recent_prices[-ma_short:])
                last_features['Price_to_MA_Short'] = predicted_price / last_features[f'MA_{ma_short}']
        
        # Create prediction DataFrame
        predictions_df = pd.DataFrame({
            'Predicted_Price': predictions,
            'Confidence': ['Medium'] * days  # Simplified confidence metric
        }, index=prediction_dates)
        
        return predictions_df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    
    def get_prediction_accuracy(self, stock_data, days_back=30):
        """
        Evaluate prediction accuracy on historical data
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            days_back (int): Number of days to look back for testing
        
        Returns:
            dict: Accuracy metrics
        """
        try:
            if len(stock_data) < days_back + 50:
                return {"error": "Insufficient data for accuracy testing"}
            
            # Split data
            train_data = stock_data[:-days_back]
            test_data = stock_data[-days_back:]
            
            # Make predictions on training data
            predictions = self.predict_prices(train_data, days=days_back)
            
            if predictions is None:
                return {"error": "Could not generate predictions"}
            
            # Compare with actual prices
            actual_prices = test_data['Close'].values[:len(predictions)]
            predicted_prices = predictions['Predicted_Price'].values
            
            # Calculate metrics
            mae = np.mean(np.abs(actual_prices - predicted_prices))
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            rmse = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))
            
            return {
                "mae": mae,
                "mape": mape,
                "rmse": rmse,
                "accuracy_score": max(0, 100 - mape)
            }
            
        except Exception as e:
            return {"error": f"Error calculating accuracy: {str(e)}"}
