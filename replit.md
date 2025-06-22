# Stock Analysis Dashboard

## Overview

This is a Streamlit-based stock analysis dashboard that provides comprehensive stock market analysis, including historical data visualization, technical indicators, and price predictions. The application uses Yahoo Finance data and machine learning models to deliver insights for stock market analysis.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Wide layout with expandable sidebar for user inputs
- **Visualization**: Plotly for interactive charts and graphs
- **State Management**: Streamlit session state for caching data across user interactions

### Backend Architecture
- **Data Processing**: Pandas for data manipulation and analysis
- **Machine Learning**: Scikit-learn with Linear Regression for price predictions
- **Data Source**: Yahoo Finance API via yfinance library
- **Caching**: Streamlit's built-in caching mechanism with 5-minute TTL

### Modular Structure
- **Main Application**: `app.py` - Primary Streamlit interface
- **Data Layer**: `utils/stock_data.py` - Stock data fetching and caching
- **Prediction Engine**: `utils/predictions.py` - ML-based price prediction algorithms

## Key Components

### 1. Stock Data Fetcher (`utils/stock_data.py`)
- **Purpose**: Handles all stock data retrieval from Yahoo Finance
- **Features**: 
  - Cached data fetching (5-minute TTL)
  - Error handling for invalid symbols
  - Data validation and cleaning
- **Rationale**: Separation of concerns for data access, improving maintainability and enabling easy API switching

### 2. Stock Predictor (`utils/predictions.py`)
- **Purpose**: Implements machine learning models for stock price prediction
- **Algorithm**: Linear Regression with technical indicators as features
- **Features**:
  - Technical indicator calculation (moving averages, RSI, etc.)
  - Feature scaling with StandardScaler
  - Support for both short-term and long-term predictions
- **Rationale**: Modular prediction system allows for easy algorithm upgrades and testing

### 3. Main Dashboard (`app.py`)
- **Purpose**: Primary user interface and application orchestration
- **Features**:
  - Interactive stock symbol input
  - Time period selection
  - Real-time data visualization
  - Session state management for performance

## Data Flow

1. **User Input**: User enters stock symbol and selects time period in sidebar
2. **Data Retrieval**: StockDataFetcher queries Yahoo Finance API with caching
3. **Data Processing**: Raw data is cleaned and validated
4. **Visualization**: Plotly generates interactive charts from processed data
5. **Prediction**: StockPredictor analyzes historical data and generates forecasts
6. **Display**: Results are rendered in the Streamlit interface

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **yfinance**: Yahoo Finance API client for stock data
- **plotly**: Interactive visualization library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms

### Rationale for Technology Choices
- **Streamlit**: Chosen for rapid development of data science applications with minimal frontend code
- **Yahoo Finance**: Free, reliable stock data source with comprehensive coverage
- **Plotly**: Superior interactive charting capabilities compared to matplotlib for web applications
- **Linear Regression**: Simple, interpretable model suitable for trend analysis and educational purposes

## Deployment Strategy

### Platform Configuration
- **Environment**: Python 3.11 with Nix package management
- **Deployment Target**: Autoscale deployment for automatic scaling
- **Port Configuration**: Internal port 5000, external port 80
- **Startup Command**: `streamlit run app.py --server.port 5000`

### Workflow Management
- **Parallel Execution**: Dependency installation and app startup run in parallel
- **Dependency Management**: UV package manager for fast, reliable dependency resolution
- **Hot Reload**: Streamlit's built-in development server for real-time updates

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- June 22, 2025. Initial setup